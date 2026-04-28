from pathlib import Path
from datetime import timedelta

import laspy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from LABELISATION_AUTO_LIDAR_HD_IGN.extract_lidar_features_labelisation import compute_sky_mask_deg_from_relative



IGNORED_CLASSES_FOR_SKY = np.array([0, 1, 64, 66, 67], dtype=np.int64)
TRAIN_ORDER_CORRIDOR_WIDTH_M = 1.0
TRAIN_ORDER_CORRIDOR_LENGTH_M = 5.0
AZIMUTH_OCCUPANCY_BINS = 36
AZIMUTH_OCCUPANCY_MIN_Z_REL = 1.5
AZIMUTH_OCCUPANCY_MAX_DIST = 30.0


def _as_path(path_like):
    if isinstance(path_like, Path):
        return path_like
    return Path(str(path_like))


def _prep_df(df):
    d = df.copy()
    d["time_utc"] = pd.to_datetime(d["time_utc"], errors="coerce")
    d = d.dropna(subset=["time_utc", "x_gt", "y_gt", "z_gt_ign69"]).sort_values("time_utc", kind="stable")
    return d.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_space_vehicule_info(csv_path_str):
    csv_path = _as_path(csv_path_str)
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    time_col = "time_utc" if "time_utc" in df.columns else ("time" if "time" in df.columns else None)
    if time_col is None:
        return pd.DataFrame()

    d = df.copy()
    d["time_utc"] = pd.to_datetime(d[time_col], errors="coerce", utc=True).dt.tz_localize(None)
    d["el_sv_deg"] = pd.to_numeric(d.get("el_sv_deg"), errors="coerce")
    d["az_sv_deg"] = pd.to_numeric(d.get("az_sv_deg"), errors="coerce")

    sat_col = None
    for candidate in ("gnss_sv_id", "sv_id", "sat_id", "prn"):
        if candidate in d.columns:
            sat_col = candidate
            break
    if sat_col is None:
        d["sat_name"] = "SAT"
    else:
        d["sat_name"] = d[sat_col].astype(str)

    d = d.dropna(subset=["time_utc", "el_sv_deg", "az_sv_deg"]).sort_values("time_utc", kind="stable")
    return d.reset_index(drop=True)


def _satellites_for_window(space_df, win_start, win_end):
    if space_df is None or len(space_df) == 0:
        return pd.DataFrame(columns=["el_sv_deg", "az_sv_deg", "sat_name"])

    m = (space_df["time_utc"] >= win_start) & (space_df["time_utc"] < win_end)
    out = space_df.loc[m, ["el_sv_deg", "az_sv_deg", "sat_name"]].copy()
    if len(out) == 0:
        return out

    out["el_sv_deg"] = out["el_sv_deg"].clip(lower=0.0, upper=90.0)
    out["az_sv_deg"] = np.mod(out["az_sv_deg"], 360.0)
    out = out.sort_values("el_sv_deg", ascending=False).drop_duplicates(subset=["sat_name"], keep="first")
    return out.reset_index(drop=True)


def _add_skyplot_overlay_top(fig, center_xy, satellites_df, overlay_radius):
    if satellites_df is None or len(satellites_df) == 0:
        return

    cx, cy = float(center_xy[0]), float(center_xy[1])
    r_max = float(max(3.0, overlay_radius))

    theta = np.linspace(0.0, 2.0 * np.pi, 180)
    # Cercles d'elevation du sky plot projete sur la vue XY locale
    for el in (0.0, 30.0, 60.0):
        rr = r_max * (1.0 - el / 90.0)
        fig.add_trace(
            go.Scatter(
                x=cx + rr * np.cos(theta),
                y=cy + rr * np.sin(theta),
                mode="lines",
                line={"width": 1, "dash": "dot", "color": "#4CC9F0"},
                name="Sky plot",
                showlegend=(el == 0.0),
                hoverinfo="skip",
            )
        )

    az = np.deg2rad(satellites_df["az_sv_deg"].to_numpy(dtype=float))
    el = satellites_df["el_sv_deg"].to_numpy(dtype=float)
    rr = r_max * (1.0 - el / 90.0)
    sx = cx + rr * np.sin(az)
    sy = cy + rr * np.cos(az)

    # Rayons 2D depuis le centre du sky plot vers chaque satellite.
    rays_x = []
    rays_y = []
    for x_i, y_i in zip(sx, sy):
        rays_x.extend([cx, float(x_i), None])
        rays_y.extend([cy, float(y_i), None])

    fig.add_trace(
        go.Scatter(
            x=rays_x,
            y=rays_y,
            mode="lines",
            line={"width": 1.5, "color": "#4CC9F0"},
            name="Rayons satellites",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sx,
            y=sy,
            mode="markers+text",
            marker={"size": 7, "color": "#00E5FF", "line": {"width": 1, "color": "#0B1E2D"}},
            text=satellites_df["sat_name"].astype(str).tolist(),
            textposition="top center",
            textfont={"size": 10},
            name="Satellites",
            hovertemplate="%{text}<br>Az=%{customdata[0]:.1f} deg<br>El=%{customdata[1]:.1f} deg<extra></extra>",
            customdata=np.column_stack((satellites_df["az_sv_deg"].to_numpy(), satellites_df["el_sv_deg"].to_numpy())),
        )
    )


def _add_satellite_rays_perspective(fig, satellites_df, ray_length):
    if satellites_df is None or len(satellites_df) == 0:
        return

    az = np.deg2rad(satellites_df["az_sv_deg"].to_numpy(dtype=float))
    el = np.deg2rad(satellites_df["el_sv_deg"].to_numpy(dtype=float))
    L = float(max(3.0, ray_length))

    dx = L * np.cos(el) * np.sin(az)
    dy = L * np.cos(el) * np.cos(az)
    dz = L * np.sin(el)

    lx, ly, lz = [], [], []
    for x_i, y_i, z_i in zip(dx, dy, dz):
        lx.extend([0.0, float(x_i), None])
        ly.extend([0.0, float(y_i), None])
        lz.extend([0.0, float(z_i), None])

    fig.add_trace(
        go.Scatter3d(
            x=lx,
            y=ly,
            z=lz,
            mode="lines",
            line={"width": 4, "color": "#00E5FF"},
            name="Rayons satellites",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=dx,
            y=dy,
            z=dz,
            mode="markers+text",
            marker={"size": 3, "color": "#00E5FF"},
            text=satellites_df["sat_name"].astype(str).tolist(),
            textposition="top center",
            name="Satellites",
            hovertemplate="%{text}<br>Az=%{customdata[0]:.1f} deg<br>El=%{customdata[1]:.1f} deg<extra></extra>",
            customdata=np.column_stack((satellites_df["az_sv_deg"].to_numpy(), satellites_df["el_sv_deg"].to_numpy())),
        )
    )


def _safe_direction(window_df, full_df):
    start = window_df.iloc[0][["x_gt", "y_gt"]].to_numpy(dtype=float)
    end = window_df.iloc[-1][["x_gt", "y_gt"]].to_numpy(dtype=float)
    vec = end - start
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        return vec / norm

    center_idx = int(window_df.index[len(window_df) // 2])
    i0 = max(0, center_idx - 5)
    i1 = min(len(full_df) - 1, center_idx + 5)
    vec2 = (
        full_df.iloc[i1][["x_gt", "y_gt"]].to_numpy(dtype=float)
        - full_df.iloc[i0][["x_gt", "y_gt"]].to_numpy(dtype=float)
    )
    norm2 = np.linalg.norm(vec2)
    if norm2 > 1e-6:
        return vec2 / norm2

    return np.array([1.0, 0.0], dtype=float)


@st.cache_data(show_spinner=False)
def _list_lidar_tiles(lidar_dir_str):
    lidar_dir = Path(lidar_dir_str)
    return sorted([p for p in lidar_dir.iterdir() if p.suffix.lower() in {".las", ".laz"}])


@st.cache_data(show_spinner=False)
def _tile_header_bbox(tile_path_str):
    with laspy.open(tile_path_str) as fh:
        h = fh.header
        return (float(h.min[0]), float(h.min[1]), float(h.max[0]), float(h.max[1]))


def _intersects(bbox, x_min, y_min, x_max, y_max):
    bmin_x, bmin_y, bmax_x, bmax_y = bbox
    return not (bmin_x > x_max or bmax_x < x_min or bmin_y > y_max or bmax_y < y_min)


@st.cache_data(show_spinner=False, max_entries=48)
def _read_tile_points(tile_path_str, decimation):
    with laspy.open(tile_path_str) as fh:
        las = fh.read()
    step = max(int(decimation), 1)
    x = las.x[::step]
    y = las.y[::step]
    z = las.z[::step]
    c = las.classification[::step]
    return np.column_stack((x, y, z)), np.asarray(c)


def _points_near_window(lidar_dir, window_df, radius, decimation):
    center = window_df[["x_gt", "y_gt"]].mean().to_numpy(dtype=float)
    x_min = float(window_df["x_gt"].min() - radius)
    x_max = float(window_df["x_gt"].max() + radius)
    y_min = float(window_df["y_gt"].min() - radius)
    y_max = float(window_df["y_gt"].max() + radius)

    tiles = _list_lidar_tiles(str(_as_path(lidar_dir)))
    selected = []
    for t in tiles:
        bbox = _tile_header_bbox(str(t))
        if _intersects(bbox, x_min, y_min, x_max, y_max):
            selected.append(t)

    if not selected:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int64)

    pts_parts = []
    cls_parts = []
    r2 = float(radius * radius)
    for t in selected:
        p, c = _read_tile_points(str(t), decimation)
        # Culling rapide dans la bbox de la fenetre temporelle
        m_bbox = (p[:, 0] >= x_min) & (p[:, 0] <= x_max) & (p[:, 1] >= y_min) & (p[:, 1] <= y_max)
        if not np.any(m_bbox):
            continue
        p = p[m_bbox]
        c = c[m_bbox]

        # Culling radial autour du centre de la fenetre pour limiter le volume
        d2_center = (p[:, 0] - center[0]) ** 2 + (p[:, 1] - center[1]) ** 2
        m_r = d2_center <= (r2 * 2.25)
        if not np.any(m_r):
            continue
        pts_parts.append(p[m_r])
        cls_parts.append(c[m_r])

    if not pts_parts:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int64)

    return np.vstack(pts_parts), np.concatenate(cls_parts)


def _capsule_filter(points_xy, a, b, radius):
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 < 1e-9:
        return np.sum((points_xy - a) ** 2, axis=1) <= radius * radius

    ap = points_xy - a
    t = np.clip((ap @ ab) / ab2, 0.0, 1.0)
    proj = a + np.outer(t, ab)
    d2 = np.sum((points_xy - proj) ** 2, axis=1)
    return d2 <= radius * radius


def _local_projection(points_xyz, origin_xyz, u_xy):
    dx = points_xyz[:, 0] - origin_xyz[0]
    dy = points_xyz[:, 1] - origin_xyz[1]
    lateral = -dx * u_xy[1] + dy * u_xy[0]
    longitudinal = dx * u_xy[0] + dy * u_xy[1]
    z_relative = points_xyz[:, 2] - origin_xyz[2]
    dist_horizontale = np.hypot(longitudinal, lateral)
    return longitudinal, lateral, z_relative, dist_horizontale


def _sky_mask_deg(z_relative, dist_horizontale, classes):
    return compute_sky_mask_deg_from_relative(
        z_relative=z_relative,
        dist_horizontale=dist_horizontale,
        points_classes=classes,
        min_dist_horizontale=0.8,
        ignored_classes=IGNORED_CLASSES_FOR_SKY,
    )


def _azimuth_occupancy_ratio(points_xyz, origin_xyz, z_relative):
    if len(points_xyz) == 0:
        return 0.0

    dx = points_xyz[:, 0] - float(origin_xyz[0])
    dy = points_xyz[:, 1] - float(origin_xyz[1])
    dist_h = np.hypot(dx, dy)

    obstacle_mask = (
        (z_relative > float(AZIMUTH_OCCUPANCY_MIN_Z_REL))
        & (dist_h < float(AZIMUTH_OCCUPANCY_MAX_DIST))
    )
    if not np.any(obstacle_mask):
        return 0.0

    az = np.arctan2(dy[obstacle_mask], dx[obstacle_mask])
    az = np.mod(az, 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, int(AZIMUTH_OCCUPANCY_BINS) + 1)
    counts, _ = np.histogram(az, bins=edges)
    return float(np.sum(counts > 0)) / float(AZIMUTH_OCCUPANCY_BINS)


def _downsample(points_xyz, classes, max_points, seed=42):
    n = len(points_xyz)
    if n <= max_points:
        return points_xyz, classes
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return points_xyz[idx], classes[idx]


def _class_color_array(classes):
    out = np.full((len(classes),), "#9AA0A6", dtype=object)
    out[np.isin(classes, [3, 4, 5])] = "#2ECC71"
    out[classes == 6] = "#E74C3C"
    out[classes == 17] = "#F39C12"
    out[classes == 2] = "#8E8E8E"
    return out


def _parse_timeline_input_to_idx(raw_value, timeline):
    if raw_value is None:
        return None
    s = str(raw_value).strip()
    if not s:
        return None

    if s.lstrip("+-").isdigit():
        return int(np.clip(int(s), 0, len(timeline) - 1))

    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)

    vals = timeline.values.astype("datetime64[ns]")
    target = np.datetime64(ts.to_datetime64())
    idx = int(np.argmin(np.abs(vals - target)))
    return int(np.clip(idx, 0, len(timeline) - 1))


def _sync_lidar_time_text_from_idx(idx_key, text_key, timeline):
    idx = int(np.clip(st.session_state.get(idx_key, 0), 0, len(timeline) - 1))
    st.session_state[idx_key] = idx
    st.session_state[text_key] = str(timeline[idx].strftime("%Y-%m-%d %H:%M:%S"))


def _sync_lidar_idx_from_text(idx_key, text_key, timeline, time_key=None):
    parsed_idx = _parse_timeline_input_to_idx(st.session_state.get(text_key, ""), timeline)
    if parsed_idx is None:
        return
    st.session_state[idx_key] = int(parsed_idx)
    st.session_state[text_key] = str(timeline[int(parsed_idx)].strftime("%Y-%m-%d %H:%M:%S"))
    if time_key is not None:
        st.session_state[time_key] = timeline[int(parsed_idx)].to_pydatetime()


def _sync_lidar_from_time_slider(idx_key, text_key, time_key, timeline):
    selected = st.session_state.get(time_key)
    if selected is None:
        return
    selected_ts = pd.Timestamp(selected)
    idx = int(np.searchsorted(timeline.values, np.datetime64(selected_ts), side="left"))
    idx = int(np.clip(idx, 0, len(timeline) - 1))
    st.session_state[idx_key] = idx
    st.session_state[text_key] = str(timeline[idx].strftime("%Y-%m-%d %H:%M:%S"))
    st.session_state[time_key] = timeline[idx].to_pydatetime()


def _train_rectangle(center_xy, u_xy, length_m, width_m):
    """Retourne les sommets (fermes) d'un rectangle oriente selon la direction u_xy."""
    u = np.asarray(u_xy, dtype=float)
    un = np.linalg.norm(u)
    if un < 1e-9:
        u = np.array([1.0, 0.0], dtype=float)
    else:
        u = u / un

    # Vecteur lateral (normale a la direction d'avancement)
    n = np.array([-u[1], u[0]], dtype=float)
    half_l = float(length_m) * 0.5
    half_w = float(width_m) * 0.5
    c = np.asarray(center_xy, dtype=float)

    p1 = c + u * half_l + n * half_w
    p2 = c + u * half_l - n * half_w
    p3 = c - u * half_l - n * half_w
    p4 = c - u * half_l + n * half_w

    x = np.array([p1[0], p2[0], p3[0], p4[0], p1[0]], dtype=float)
    y = np.array([p1[1], p2[1], p3[1], p4[1], p1[1]], dtype=float)
    front_center = c + u * half_l
    return x, y, front_center, u


def _oriented_rectangle_xy(center_xy, u_xy, length_m, width_m):
    """Retourne le contour ferme d'un rectangle oriente."""
    u = np.asarray(u_xy, dtype=float)
    un = np.linalg.norm(u)
    if un < 1e-9:
        u = np.array([1.0, 0.0], dtype=float)
    else:
        u = u / un

    n = np.array([-u[1], u[0]], dtype=float)
    hl = float(length_m) * 0.5
    hw = float(width_m) * 0.5
    c = np.asarray(center_xy, dtype=float)

    p1 = c + u * hl + n * hw
    p2 = c + u * hl - n * hw
    p3 = c - u * hl - n * hw
    p4 = c - u * hl + n * hw

    x = np.array([p1[0], p2[0], p3[0], p4[0], p1[0]], dtype=float)
    y = np.array([p1[1], p2[1], p3[1], p4[1], p1[1]], dtype=float)
    return x, y

def _build_top_view(
    points_xyz,
    classes,
    traj_win,
    center_xy,
    u_xy,
    train_on_top=True,
    spatial_mode="Rayon",
    radius=20.0,
    corridor_width=10.0,
    corridor_length=30.0,
    satellites_df=None,
    skyplot_scale_m=8.0,
):
    fig = go.Figure()
    colors = _class_color_array(classes)
    
    #affiche un texte qui indique si le train est on top ou en dessous
    train_status = "en dessous" if not train_on_top else "sur la surface"
    st.caption(f"Le train est {train_status}")


    # Representation du train sur la seconde courante (rectangle noir oriente).
    # Le gabarit est volontairement visible pour l'usage pedagogique.
    train_len = max(10.0, radius * 0.28)
    train_width = max(2.8, radius * 0.08)
    tx, ty, front_center, u_dir = _train_rectangle(center_xy, u_xy, train_len, train_width)

    train_body_trace = go.Scatter(
        x=tx,
        y=ty,
        mode="lines",
        fill="toself",
        fillcolor="rgba(0,0,0,0.95)",
        line={"width": 2, "color": "#F5F5F5"},
        name="Train (1 s)",
    )

    # Direction de marche: trait + pointe unique
    nose = front_center + u_dir * (train_len * 0.18)
    train_front_trace = go.Scatter(
        x=[front_center[0], nose[0]],
        y=[front_center[1], nose[1]],
        mode="lines",
        line={"width": 5, "color": "#002B35"},
        name="Sens de marche",
    )

    n_dir = np.array([-u_dir[1], u_dir[0]], dtype=float)
    arrow_len = max(0.6, train_len * 0.08)
    arrow_half_w = max(0.18, train_width * 0.20)
    base = nose - u_dir * arrow_len
    left = base + n_dir * arrow_half_w
    right = base - n_dir * arrow_half_w
    train_head_trace = go.Scatter(
        x=[nose[0], left[0], right[0], nose[0]],
        y=[nose[1], left[1], right[1], nose[1]],
        mode="lines",
        fill="toself",
        fillcolor="#002B35",
        line={"width": 0.1, "color": "#002B35"},
        name="Sens de marche",
        showlegend=False,
    )

    # IMPORTANT: on utilise Scatter (SVG) et non Scattergl pour respecter l'ordre
    # de superposition avec les traces du train (sinon WebGL peut passer devant).
    lidar_trace = go.Scatter(
        x=points_xyz[:, 0],
        y=points_xyz[:, 1],
        mode="markers",
        marker={"size": 3, "color": colors, "opacity": 0.65},
        name="Points LiDAR",
    )

    # Ordre des traces = ordre visuel (dernier trace au-dessus).
    if train_on_top:
        fig.add_trace(lidar_trace)
        fig.add_trace(train_body_trace)
        fig.add_trace(train_front_trace)
        fig.add_trace(train_head_trace)
    else:
        fig.add_trace(train_body_trace)
        fig.add_trace(train_front_trace)
        fig.add_trace(train_head_trace)
        fig.add_trace(lidar_trace)

    if spatial_mode == "Couloir":
        zx, zy = _oriented_rectangle_xy(center_xy, u_xy, corridor_length, corridor_width)
        fig.add_trace(
            go.Scatter(
                x=zx,
                y=zy,
                mode="lines",
                line={"width": 2, "dash": "dot", "color": "#F39C12"},
                name=f"Couloir {corridor_width:.1f} x {corridor_length:.1f} m",
            )
        )
    else:
        # Cercle de reference du rayon de recherche
        theta = np.linspace(0.0, 2.0 * np.pi, 180)
        fig.add_trace(
            go.Scatter(
                x=center_xy[0] + radius * np.cos(theta),
                y=center_xy[1] + radius * np.sin(theta),
                mode="lines",
                line={"width": 2, "dash": "dot", "color": "#F39C12"},
                name=f"Rayon {radius:.1f} m",
            )
        )

    _add_skyplot_overlay_top(fig, center_xy, satellites_df=satellites_df, overlay_radius=float(skyplot_scale_m))

    fig.update_layout(
        title="Vue de dessus - filtre spatial autour de 1 seconde",
        xaxis_title="X Lambert",
        yaxis_title="Y Lambert",
        template="plotly_dark",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        height=430,
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _build_slice_view(
    longitudinal,
    z_relative,
    dist_horizontale,
    sky_mask_deg,
    zone_limit,
    classes,
    display_mode="Densite lisible",
    z_axis_max=None,
    x_axis_title="Distance longitudinale (m)",
):
    fig = make_subplots(rows=1, cols=1)

    x_all = np.asarray(longitudinal)
    z_all = np.asarray(z_relative)
    x_dens = x_all
    z_dens = z_all

    # Rendu robuste: la densite est calculee sur la fenetre percentile [2, 98]
    # pour eviter l'ecrasement visuel par quelques outliers.
    if len(x_all) >= 10:
        x_q2, x_q98 = np.percentile(x_all, [2, 98])
        z_q2, z_q98 = np.percentile(z_all, [2, 98])
        m_robust = (x_all >= x_q2) & (x_all <= x_q98) & (z_all >= z_q2) & (z_all <= z_q98)
        if np.any(m_robust):
            x_dens = x_all[m_robust]
            z_dens = z_all[m_robust]

    if display_mode in {"Densite lisible", "Mixte"}:
        fig.add_trace(
            go.Histogram2dContour(
                x=x_dens,
                y=z_dens,
                ncontours=14,
                colorscale="YlOrRd",
                contours_coloring="fill",
                contours_showlines=False,
                showscale=True,
                colorbar={"title": "densite"},
                opacity=0.9,
                name="Densite",
                hovertemplate="x=%{x:.2f} m<br>z=%{y:.2f} m<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if display_mode in {"Points classes", "Mixte"}:
        colors = _class_color_array(classes)
        fig.add_trace(
            go.Scattergl(
                x=longitudinal,
                y=z_relative,
                mode="markers",
                marker={"size": 3, "color": colors, "opacity": 0.30},
                name="Projection points",
            ),
            row=1,
            col=1,
        )

    # Representation du masque ciel par droite d'elevation depuis l'antenne (origine)
    x_max = max(zone_limit, float(np.percentile(np.abs(longitudinal), 98)))
    x_line = np.linspace(-max(1.0, x_max), max(1.0, x_max), 240)
    y_line = np.tan(np.deg2rad(sky_mask_deg)) * np.abs(x_line)
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            line={"width": 2.5, "dash": "dash", "color": "#F39C12"},
            name=f"Sky mask {sky_mask_deg:.1f} deg",
        ),
        row=1,
        col=1,
    )

    fig.add_vline(x=zone_limit, line_width=1.5, line_dash="dot", line_color="#F1C40F")
    fig.add_vline(x=-zone_limit, line_width=1.5, line_dash="dot", line_color="#F1C40F")

    fig.update_layout(
        title="Coupe 2D (projection dans l'axe d'avancement, fenetre 1 seconde)",
        xaxis_title=x_axis_title,
        yaxis_title="Hauteur relative a l'antenne (m)",
        template="plotly_dark",
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        height=430,
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
    )
    if z_axis_max is not None:
        z_data_min = float(np.nanmin(z_relative)) if len(z_relative) else -1.0
        y_min = min(z_data_min, 0.0) if not np.isfinite(z_data_min) else z_data_min
        if z_axis_max <= y_min + 0.1:
            z_axis_max = y_min + 0.1
        fig.update_yaxes(range=[y_min, float(z_axis_max)])
    return fig


def _build_perspective_view(
    longitudinal,
    lateral,
    z_relative,
    classes,
    sky_mask_deg,
    z_axis_max=None,
    x_axis_title="Longitudinal (m)",
    y_axis_title="Lateral (m)",
    gnss_offset_z=0.0,
    point_size=2.0,
    satellites_df=None,
    satellite_ray_length=20.0,
):
    """Construit une vue en perspective orientee dans l'axe de deplacement."""
    colors = _class_color_array(classes)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=longitudinal,
            y=lateral,
            z=z_relative,
            mode="markers",
            marker={"size": float(point_size), "color": colors, "opacity": 0.9},
            name="Projection points",
            hovertemplate="long=%{x:.2f} m<br>lat=%{y:.2f} m<br>z=%{z:.2f} m<extra></extra>",
        )
    )

    _add_satellite_rays_perspective(fig, satellites_df=satellites_df, ray_length=float(satellite_ray_length))

    # Point d'origine pour se reperer dans la scene.
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers",
            marker={"size": 6, "color": "#000000"},
            name="Antenne",
        )
    )

    # Sky mask en perspective: demi-droite d'elevation depuis l'antenne.
    x_extent = max(1.0, float(np.percentile(np.abs(longitudinal), 98)) if len(longitudinal) else 1.0)
    x_line = np.linspace(-x_extent, x_extent, 180)
    z_line = np.tan(np.deg2rad(float(sky_mask_deg))) * np.abs(x_line)
    fig.add_trace(
        go.Scatter3d(
            x=x_line,
            y=np.zeros_like(x_line),
            z=z_line,
            mode="lines",
            line={"width": 6, "dash": "dash", "color": "#F39C12"},
            name=f"Sky mask {float(sky_mask_deg):.1f} deg",
            hovertemplate="Sky mask: %{z:.2f} m a %{x:.2f} m<extra></extra>",
        )
    )

    z_min = float(min(-1.0, np.percentile(z_relative, 1))) if len(z_relative) else -1.0
    z_max = float(z_axis_max) if z_axis_max is not None else float(max(4.0, np.percentile(z_relative, 99)))

    fig.update_layout(
        title="Projection perspective (axe d'avancement)",
        template="plotly_dark",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        height=430,
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        scene={
            "xaxis_title": x_axis_title,
            "yaxis_title": y_axis_title,
            "zaxis_title": "Hauteur relative (m)",
            "zaxis": {"range": [z_min, z_max]},
            "aspectmode": "data",
            "camera": {
                "eye": {"x": -1.8, "y": 0.0, "z": 0.45},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
            },
        },
    )

    return fig


def render_lidar_slice_2d(trajet_id, lidar_dir, matched_df, gnss_offset_z=0.0, space_vehicule_info_csv=None):
    st.title("Coupe 2D LiDAR")
    st.caption(
        "Visualisation synchronisee: vue de dessus + coupe verticale dans l'axe train, sur une fenetre glissante de 1 seconde."
    )

    required = {"time_utc", "x_gt", "y_gt", "z_gt_ign69"}
    missing = sorted(required - set(matched_df.columns))
    if missing:
        st.error(f"Colonnes manquantes pour ce module: {missing}")
        return

    df = _prep_df(matched_df)
    if df.empty:
        st.warning("Aucune donnee exploitable apres nettoyage des timestamps/positions.")
        return

    space_df = pd.DataFrame()
    if space_vehicule_info_csv:
        try:
            space_df = _load_space_vehicule_info(str(space_vehicule_info_csv))
        except Exception as e:
            st.info(f"Impossible de lire space_vehicule_info: {e}")

    c0, c1, c2, c3, c4 = st.columns([1.2, 1.3, 1.0, 1.0, 1.2])
    with c0:
        spatial_mode = st.selectbox("Mode spatial", ["Rayon", "Couloir"], index=0)
    with c1:
        if spatial_mode == "Rayon":
            radius = st.slider("Rayon de masquage (m)", min_value=5, max_value=200, value=40, step=1)
            corridor_width = None
            corridor_length = None
            zone_limit = float(radius)
            fetch_radius = float(radius)
        else:
            corridor_width = st.slider("Largeur couloir (m)", min_value=1.0, max_value=50.0, value=6.0, step=1.0)
            corridor_length = st.slider("Demi-longueur couloir (m)", min_value=5.0, max_value=120.0, value=30.0, step=1.0)
            radius = None
            # Convention alignee sur le pipeline labelisation:
            # corridor_length est une demi-longueur (half-length).
            zone_limit = float(corridor_length)
            fetch_radius = float(np.hypot(corridor_length, corridor_width * 0.5) + 2.0)
    with c2:
        decimation = st.slider("Decimation LiDAR", min_value=1, max_value=12, value=4, step=1)
    with c3:
        max_points = st.slider("Max points affiches", min_value=2000, max_value=90000, value=25000, step=1000)
    with c4:
        show_only_above = st.checkbox("Coupe: points au-dessus antenne", value=False)

    skyplot_toggle_key = f"slice2d_skyplot_enabled_{trajet_id}"
    if skyplot_toggle_key not in st.session_state:
        st.session_state[skyplot_toggle_key] = False

    toggle_col, status_col = st.columns([1.8, 3.2])
    with toggle_col:
        toggle_label = (
            "Masquer sky plot + rayons satellites"
            if st.session_state[skyplot_toggle_key]
            else "Afficher sky plot + rayons satellites"
        )
        if st.button(toggle_label, key=f"slice2d_skyplot_toggle_btn_{trajet_id}"):
            st.session_state[skyplot_toggle_key] = not st.session_state[skyplot_toggle_key]
            st.rerun()
    with status_col:
        if st.session_state[skyplot_toggle_key]:
            if len(space_df) == 0:
                st.caption("Sky plot active, mais aucune donnee satellite exploitable n'a ete trouvee.")
            else:
                st.caption("Sky plot et rayons satellites actifs (source: space_vehicule_info).")

    display_mode = st.radio(
        "Rendu coupe 2D",
        ["Densite lisible", "Mixte", "Points classes", "Perspective"],
        horizontal=True,
        index=0,
    )

    point_size_perspective = 2.0
    if display_mode == "Perspective":
        point_size_perspective = st.slider(
            "Taille points perspective",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
        )

    z_axis_max = None
    if display_mode != "Perspective":
        z_axis_max = st.slider(
            "Z max coupe 2D (m)",
            min_value=1.0,
            max_value=80.0,
            value=20.0,
            step=0.5,
        )

    t0 = df["time_utc"].dt.floor("s").min()
    t1 = df["time_utc"].dt.floor("s").max()
    if t1 <= t0:
        st.warning("Trajet trop court pour un slider de 1 seconde.")
        return

    # Defilement temporel (pas de 1 s) pilote par etat + boutons.
    timeline = pd.date_range(t0, t1, freq="s")
    if len(timeline) == 0:
        st.warning("Aucune timeline disponible pour le defilement.")
        return

    idx_key = f"slice2d_second_idx_{trajet_id}"
    text_key = f"slice2d_second_text_{trajet_id}"
    time_key = f"slice2d_second_time_{trajet_id}"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0

    max_idx = len(timeline) - 1
    st.session_state[idx_key] = int(np.clip(st.session_state[idx_key], 0, max_idx))

    moved_by_buttons = False
    n0, n1, n2, n3, n4 = st.columns([1.0, 1.0, 1.4, 1.0, 1.0])
    with n0:
        if st.button("<< 10s", key=f"slice2d_prev10_{trajet_id}"):
            st.session_state[idx_key] = max(0, st.session_state[idx_key] - 10)
            moved_by_buttons = True
    with n1:
        if st.button("< 1s", key=f"slice2d_prev1_{trajet_id}"):
            st.session_state[idx_key] = max(0, st.session_state[idx_key] - 1)
            moved_by_buttons = True
    with n2:
        st.caption("Defilement coupe 2D")
    with n3:
        if st.button("1s >", key=f"slice2d_next1_{trajet_id}"):
            st.session_state[idx_key] = min(max_idx, st.session_state[idx_key] + 1)
            moved_by_buttons = True
    with n4:
        if st.button("10s >>", key=f"slice2d_next10_{trajet_id}"):
            st.session_state[idx_key] = min(max_idx, st.session_state[idx_key] + 10)

    if (text_key not in st.session_state) or moved_by_buttons:
        _sync_lidar_time_text_from_idx(idx_key, text_key, timeline)
    if (time_key not in st.session_state) or moved_by_buttons:
        st.session_state[time_key] = timeline[int(np.clip(st.session_state[idx_key], 0, max_idx))].to_pydatetime()

    tsel0, tsel1 = st.columns([3.0, 2.0])
    with tsel0:
        st.slider(
            "Heure (pas 1 seconde)",
            min_value=t0.to_pydatetime(),
            max_value=t1.to_pydatetime(),
            value=st.session_state[time_key],
            step=timedelta(seconds=1),
            format="HH:mm:ss",
            key=time_key,
            on_change=_sync_lidar_from_time_slider,
            args=(idx_key, text_key, time_key, timeline),
        )
    with tsel1:
        st.text_input(
            "Aller a (index ou date/heure)",
            key=text_key,
            help="Exemples: 120 | 2025-04-24 05:38:10 | 05:38:10",
            on_change=_sync_lidar_idx_from_text,
            args=(idx_key, text_key, timeline, time_key),
        )

    idx_value = int(np.clip(st.session_state[idx_key], 0, max_idx))
    win_start = pd.Timestamp(timeline[idx_value])
    win_end = win_start + pd.Timedelta(seconds=1)
    win_center = win_start + pd.Timedelta(milliseconds=500)

    sat_window = pd.DataFrame()
    if st.session_state[skyplot_toggle_key] and len(space_df) > 0:
        sat_window = _satellites_for_window(space_df, win_start, win_end)

    mask_win = (df["time_utc"] >= win_start) & (df["time_utc"] < win_end)
    win_df = df.loc[mask_win]
    if len(win_df) == 0:
        # Fallback robuste: prend le point trajectoire le plus proche de l'instant choisi.
        nearest_idx = (df["time_utc"] - win_start).abs().idxmin()
        win_df = df.loc[[nearest_idx]]
        st.info("Aucun point exact dans cette seconde: utilisation du point trajectoire le plus proche.")

    st.markdown(
        f"**Trajet:** {trajet_id}  |  **Fenetre:** {win_start} -> {win_end}  |  **Points trajectoire:** {len(win_df)}"
    )

    copy_col, hint_col = st.columns([1.6, 3.4])
    with copy_col:
        preset_window_s = st.selectbox(
            "Fenetre Viz3D (s)",
            options=[1, 10, 20, 30, 45, 60, 90, 120],
            index=2,
            key=f"viz3d_window_{trajet_id}",
        )
        if st.button(f"Envoyer vers Viz3D ({preset_window_s} s)", key=f"send_to_viz3d_{trajet_id}"):
            half_window = pd.Timedelta(seconds=float(preset_window_s) / 2.0)
            preset_start = win_center - half_window
            preset_end = win_center + half_window
            st.session_state.viz3d_time_preset = {
                "trajet_id": trajet_id,
                "start": preset_start.isoformat(),
                "end": preset_end.isoformat(),
                "mode": "time",
            }
            st.success("Fenetre temporelle envoyee vers Viz3D.")
            st.rerun()
        if st.button(f"Zoom graphes analyses ({preset_window_s} s)", key=f"zoom_analysis_from_slice_{trajet_id}"):
            half_window = pd.Timedelta(seconds=float(preset_window_s) / 2.0)
            zoom_start = win_center - half_window
            zoom_end = win_center + half_window
            st.session_state[f"analysis_zoom_window_{trajet_id}"] = {
                "start": zoom_start.isoformat(),
                "end": zoom_end.isoformat(),
            }
            st.success("Zoom des graphes d'analyse mis a jour.")
    with hint_col:
        st.caption("Choisis la duree, puis cree une fenetre temporelle centree sur l'instant selectionne et pre-remplie dans Viz3D.")
        st.caption("Le meme selecteur pilote aussi le zoom des graphes d'analyse.")

    with st.spinner("Chargement des points LiDAR autour de la fenetre temporelle..."):
        points_xyz, classes = _points_near_window(lidar_dir, win_df, radius=fetch_radius, decimation=decimation)

    if len(points_xyz) == 0:
        st.warning("Aucun point LiDAR charge sur cette fenetre et ce rayon.")
        return

    origin = win_df.iloc[0][["x_gt", "y_gt", "z_gt_ign69"]].to_numpy(dtype=float)
    u_xy = _safe_direction(win_df, df)
    a = win_df.iloc[0][["x_gt", "y_gt"]].to_numpy(dtype=float)
    b = win_df.iloc[-1][["x_gt", "y_gt"]].to_numpy(dtype=float)

    if spatial_mode == "Rayon":
        keep = _capsule_filter(points_xyz[:, :2], a, b, radius=float(radius))
        points_xyz = points_xyz[keep]
        classes = classes[keep]
    if len(points_xyz) == 0:
        st.warning("Aucun point dans le couloir/rayon de la seconde selectionnee.")
        return

    points_xyz, classes = _downsample(points_xyz, classes, max_points=max_points)

    longitudinal, lateral, z_relative, dist_horizontale = _local_projection(points_xyz, origin, u_xy)
    # Alignement avec le pipeline d'extraction: decalage du bras de levier ajoute a z_relative.
    z_relative = z_relative + float(gnss_offset_z)
    if spatial_mode == "Couloir":
        half_w = float(corridor_width) * 0.5
        half_l = float(corridor_length)
        keep = (np.abs(lateral) <= half_w) & (np.abs(longitudinal) <= half_l)
        longitudinal = longitudinal[keep]
        lateral = lateral[keep]
        z_relative = z_relative[keep]
        dist_horizontale = dist_horizontale[keep]
        points_xyz = points_xyz[keep]
        classes = classes[keep]
        if len(points_xyz) == 0:
            st.warning("Aucun point dans le couloir selectionne.")
            return

    if show_only_above:
        m = z_relative > 0
        longitudinal = longitudinal[m]
        lateral = lateral[m]
        z_relative = z_relative[m]
        dist_horizontale = dist_horizontale[m]
        points_xyz = points_xyz[m]
        classes = classes[m]
        if len(longitudinal) == 0:
            st.warning("Aucun point au-dessus de l'antenne pour cette fenetre.")
            return

    
    
    plot_longitudinal = lateral
    plot_lateral = -longitudinal
    x_axis_title = "Distance laterale (m)"
    

    z_default_max = float(np.ceil(max(3.0, np.percentile(z_relative, 99)))) if len(z_relative) else 5.0

    sky = _sky_mask_deg(z_relative, dist_horizontale, classes)
    veg_density = float(np.mean(np.isin(classes, [3, 4, 5]))) if len(classes) else 0.0
    azimuth_occupancy_ratio = _azimuth_occupancy_ratio(points_xyz, origin, z_relative)
    effective_veg_density = veg_density * azimuth_occupancy_ratio
    row = df[df["time_utc"] == win_start]

    M = st.columns(6)
    i = 0
    M[i].metric("Sky mask", f"{sky:.2f} deg")
    i += 1
    M[i].metric("Densite vegetation effective", f"{effective_veg_density:.3f}")
    i += 1
    M[i].metric("CMC (multitrajet)", f"{row['gnss_feat_CMC_l1'].iloc[0]:.2f}")
    i += 1
    M[i].metric("pdop", f"{row['pdop'].iloc[0]:.2f}")
    i += 1
    M[i].metric("cn0_mean", f"{row['gnss_feat_CN0 mean'].iloc[0]:.2f} dB-Hz")
    i += 1
    M[i].metric("Label", f"{row['label'].iloc[0] if 'label' in row else 'N/A'}")
    i += 1

    st.caption(
        f"veg_density brute={veg_density:.3f} | azimuth_occupancy_ratio={azimuth_occupancy_ratio:.3f} | effective={effective_veg_density:.3f}"
    )

    # Ordre train/nuage base sur la mediane des points LiDAR dans un couloir local
    # 1 m de large x 5 m de long, oriente selon l'axe d'avancement.
    half_w = TRAIN_ORDER_CORRIDOR_WIDTH_M * 0.5
    half_l = TRAIN_ORDER_CORRIDOR_LENGTH_M * 0.5
    m_order = (np.abs(lateral) <= half_w) & (np.abs(longitudinal) <= half_l)
    if np.any(m_order):
        local_median_z = float(np.median(points_xyz[m_order, 2]))
    else:
        # Fallback de securite si aucun point dans ce micro-couloir.
        local_median_z = float(np.median(points_xyz[:, 2]))
    train_on_top = bool(origin[2] >= local_median_z)

    top_fig = _build_top_view(
        points_xyz,
        classes,
        win_df,
        win_df[["x_gt", "y_gt"]].mean().to_numpy(),
        u_xy,
        train_on_top=train_on_top,
        spatial_mode=spatial_mode,
        radius=float(radius) if radius is not None else float(fetch_radius),
        corridor_width=float(corridor_width) if corridor_width is not None else 0.0,
        corridor_length=(2.0 * float(corridor_length)) if corridor_length is not None else 0.0,
        satellites_df=sat_window,
        skyplot_scale_m=(float(radius) if radius is not None else float(fetch_radius)),
    )
    x_zone_limit = zone_limit
    if spatial_mode == "Couloir":
        x_zone_limit = float(corridor_width) * 0.5

    if z_axis_max is None:
        z_axis_max = float(np.clip(z_default_max, 1.0, 80.0))

    f1, f2 = st.columns(2)

    if display_mode == "Perspective":
        satellite_ray_length = (
            max(10.0, float(np.percentile(np.abs(plot_longitudinal), 95)))
            if len(plot_longitudinal)
            else 20.0
        )
        slice_fig = _build_perspective_view(
            longitudinal=plot_longitudinal,
            lateral=plot_lateral,
            z_relative=z_relative,
            classes=classes,
            sky_mask_deg=sky,
            z_axis_max=None,
            x_axis_title=("Lateral (m)"),
            y_axis_title=("-Longitudinal (m)"),
            gnss_offset_z=gnss_offset_z,
            point_size=point_size_perspective,
            satellites_df=sat_window,
            satellite_ray_length=float(satellite_ray_length),

        )
    else:
        slice_fig = _build_slice_view(
            plot_longitudinal,
            z_relative,
            dist_horizontale,
            sky,
            zone_limit=x_zone_limit,
            classes=classes,
            display_mode=display_mode,
            z_axis_max=float(z_axis_max),
            x_axis_title=x_axis_title,
        )

    with f1:
        st.plotly_chart(top_fig, width="stretch")
    with f2:
        st.plotly_chart(slice_fig, width="stretch")
