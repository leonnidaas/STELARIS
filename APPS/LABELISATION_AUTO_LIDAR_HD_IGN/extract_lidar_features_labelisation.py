import functools
import os
from concurrent.futures import ProcessPoolExecutor

import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from utils import DECIMATION_FACTOR, N_WORKERS, RAYON_RECHERCHE, get_traj_paths


SKY_MASK_IGNORED_CLASSES = np.array([0, 1, 64, 66, 67], dtype=np.int64)
AZIMUTH_BINS = 36
CANOPEE_MIN_Z_REL = 2.0
MIN_POINTS_FOR_STABLE_FEATURES = 20
BRIDGE_CLASS_CODE = 17
BRIDGE_CORRIDOR_WIDTH_M = 3.0
BRIDGE_TIME_HORIZON_S = 2.0
BRIDGE_MIN_POINTS_THRESHOLD = 10
MIN_ELEVATION_ANGLE_DEG = 0.0


def compute_sky_mask_deg_from_relative(z_relative, dist_horizontale, points_classes, min_dist_horizontale=0.8, ignored_classes=SKY_MASK_IGNORED_CLASSES):
    """Calcule l'angle de sky mask (deg) a partir de coordonnees relatives a l'antenne.
    args:
        z_relative: array des hauteurs relatives des points par rapport a l'antenne (z_pts - z_ant + gnss_offset_z).
        dist_horizontale: array des distances horizontales des points par rapport a la position GT (sqrt((x_pts - x_gt)^2 + (y_pts - y_gt)^2)).
        points_classes: array des classes des points LiDAR.
        min_dist_horizontale: distance horizontale minimale pour considerer un point dans le calcul du sky mask.
        ignored_classes: classes de points a ignorer dans le calcul du sky mask (ex: sol, eau, etc.).
    retourne:
        sky_mask_deg: angle en degres representant la portion de ciel visible (0 = completement masque, 90 = ciel totalement visible)."""
    z_relative = np.asarray(z_relative, dtype=float)
    dist_horizontale = np.asarray(dist_horizontale, dtype=float)
    points_classes = np.asarray(points_classes)
    ignored_classes = np.asarray(ignored_classes, dtype=np.int64)

    valid = (z_relative > 0.0) & (dist_horizontale > float(min_dist_horizontale)) & (~np.isin(points_classes.astype(np.int64), ignored_classes))
    if not np.any(valid):
        return 0.0

    angles = np.degrees(np.arctan2(z_relative[valid], np.maximum(dist_horizontale[valid], 1e-6)))
    return float(np.percentile(angles, 98))


def _add_local_heading_vectors(df):
    """Ajoute des vecteurs directeurs unitaires (dir_x, dir_y) selon la trajectoire."""
    d = df.copy()
    x = d["x_gt"].to_numpy(dtype=float)
    y = d["y_gt"].to_numpy(dtype=float)

    prev_x = np.roll(x, 1)
    prev_y = np.roll(y, 1)
    next_x = np.roll(x, -1)
    next_y = np.roll(y, -1)

    prev_x[0], prev_y[0] = x[0], y[0]
    next_x[-1], next_y[-1] = x[-1], y[-1]

    vx = next_x - prev_x
    vy = next_y - prev_y
    norm = np.hypot(vx, vy)
    norm = np.where(norm < 1e-6, 1.0, norm)

    d["dir_x"] = vx / norm
    d["dir_y"] = vy / norm

    d["speed_mps"] = d["velocity"].abs()
    return d


def _corridor_filter(local_x, local_y, center_x, center_y, dir_x, dir_y, half_width, half_length):
    """Filtre booléen des points inclus dans un rectangle orienté (couloir)."""
    dx = local_x - center_x
    dy = local_y - center_y

    longitudinal = dx * dir_x + dy * dir_y
    lateral = -dx * dir_y + dy * dir_x
    return (np.abs(longitudinal) <= half_length) & (np.abs(lateral) <= half_width)


def _empty_feature(signal_denied=0):
    return {
        "sky_mask_deg": 0,
        "obs_type": 0,
        "is_bridge": 0,
        "is_under_structure": 0,
        "veg_density": 0,
        "n_points_zone": 0,
        "enough_points_flag": 0,
        "density_near_0_5m": 0.0,
        "density_mid_5_15m": 0.0,
        "density_far_15_30m": 0.0,
        "zrel_p50": 0.0,
        "zrel_p90": 0.0,
        "zrel_p95": 0.0,
        "zrel_p99": 0.0,
        "zrel_iqr": 0.0,
        "zrel_std": 0.0,
        "occupation_ciel_azimuth_ratio": 0.0,
        "building_density": 0.0,
        "vegetation_density_low": 0.0,
        "vegetation_density_mid": 0.0,
        "vegetation_density_high": 0.0,
        "bridge_density": 0.0,
        "bridge_above_density": 0.0,
        "bridge_above_count": 0,
        "bridge_corridor_count": 0,
        "canopee_ratio": 0.0,
        "obstacle_overhead_ratio": 0.0,
        "signal_denied": int(signal_denied),
    }


def _compute_spatial_params(search_radius, spatial_mode, corridor_width, corridor_length):
    if corridor_length is None:
        corridor_length = search_radius

    half_w = max(float(corridor_width) / 2.0, 0.01)
    half_l = max(float(corridor_length), 0.01)
    preselect_radius = search_radius if spatial_mode == "circle" else float(np.hypot(half_l, half_w))
    return half_w, half_l, preselect_radius


def _compute_obs_type(is_build, is_veg, is_bridge):
    """
    Convention utilisée pour la labelisation:
    0=aucune, 1=batiment, 2=vegetation, 3=mixte, 4=bridge.
    """
    if is_build and is_veg:
        return 3
    if is_build:
        return 1
    if is_veg:
        return 2
    if is_bridge:
        return 4
    return 0


def _safe_ratio(mask, denom):
    if denom <= 0:
        return 0.0
    return float(np.sum(mask)) / float(denom)


def _safe_percentile(values, q):
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def _row_speed_mps(row):
    for col in ("speed_mps"):
        if col in row.index and pd.notna(row[col]):
            return max(float(row[col]), 0.0)
    return 0.0


def _compute_bridge_from_corridor(
    row,
    x_pts,
    y_pts,
    z_pts,
    points_classes,
    bridge_point_threshold,
    bridge_corridor_width,
    bridge_time_horizon_s,
    gnss_offset_z=0.0,
):
    """ retourne un tuple (is_bridge, bridge_count) en fonction du nombre de points classes comme pont dans un couloir devant la trajectoire.
    is_bridge est un entier binaire (0 ou 1) indiquant la presence d'un pont, tandis que bridge_count est le nombre de points classes comme pont dans le couloir. """
    if points_classes.size == 0:
        return 0, 0

    speed_mps = _row_speed_mps(row)
    corridor_length = max(speed_mps * float(bridge_time_horizon_s), 0.5)
    half_length = max(corridor_length / 2.0, 0.25)
    half_width = max(float(bridge_corridor_width) / 2.0, 0.1)

    in_bridge_corridor = _corridor_filter(
        x_pts,
        y_pts,
        float(row["x_gt"]),
        float(row["y_gt"]),
        float(row["dir_x"]),
        float(row["dir_y"]),
        half_width=half_width,
        half_length=half_length,
    )

    z_ant = float(row["z_gt_ign69"])
    z_relative = z_pts - z_ant + float(gnss_offset_z)
    above_antenna = z_relative > 0.0

    bridge_count = int(np.sum((points_classes == BRIDGE_CLASS_CODE) & in_bridge_corridor & above_antenna))
    is_bridge = int(bridge_count >= int(bridge_point_threshold))
    return is_bridge, bridge_count


def _compute_azimuth_occupancy_ratio(dx, dy, valid_mask, bins=AZIMUTH_BINS):
    if dx.size == 0 or bins <= 0:
        return 0.0

    m = np.asarray(valid_mask, dtype=bool)
    if not np.any(m):
        return 0.0

    az = np.arctan2(dy[m], dx[m])
    az = np.mod(az, 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, int(bins) + 1)
    counts, _ = np.histogram(az, bins=edges)
    occupied_bins = np.sum(counts > 0)
    return float(occupied_bins) / float(bins)


def _apply_elevation_angle_mask(x_pts, y_pts, z_pts, points_classes, row, gnss_offset_z, min_elevation_angle_deg):
    """Conserve uniquement les points au-dessus d'un angle minimal par rapport a l'horizontale de l'antenne."""
    if points_classes.size == 0:
        return x_pts, y_pts, z_pts, points_classes

    z_ant = float(row["z_gt_ign69"])
    dx = x_pts - float(row["x_gt"])
    dy = y_pts - float(row["y_gt"])
    dist_horizontale = np.sqrt(dx ** 2 + dy ** 2)
    z_relative = z_pts - z_ant + float(gnss_offset_z)

    elevation_deg = np.degrees(np.arctan2(z_relative, np.maximum(dist_horizontale, 1e-6)))
    keep = elevation_deg >= float(min_elevation_angle_deg)

    return x_pts[keep], y_pts[keep], z_pts[keep], points_classes[keep]


def _read_tile_points_bbox_filtered(
    fh,
    df_tile,
    preselect_radius,
    decimation_factor,
    lidar_chunk_size=1_000_000,
):
    """Lit une dalle en mode streaming et garde uniquement les points proches de la bbox trajectoire."""
    step = max(int(decimation_factor), 1)

    min_x = float(df_tile["x_gt"].min()) - float(preselect_radius)
    max_x = float(df_tile["x_gt"].max()) + float(preselect_radius)
    min_y = float(df_tile["y_gt"].min()) - float(preselect_radius)
    max_y = float(df_tile["y_gt"].max()) + float(preselect_radius)

    xs, ys, zs, cs = [], [], [], []

    # Fallback: anciennes versions/API sans chunk iterator.
    if not hasattr(fh, "chunk_iterator"):
        las = fh.read()
        x = las.x[::step]
        y = las.y[::step]
        z = las.z[::step]
        c = las.classification[::step]
        in_bbox = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
        return x[in_bbox], y[in_bbox], z[in_bbox], c[in_bbox]

    for points in fh.chunk_iterator(max(int(lidar_chunk_size), 1)):
        x = points.x
        y = points.y
        z = points.z
        c = points.classification

        if step > 1:
            x = x[::step]
            y = y[::step]
            z = z[::step]
            c = c[::step]

        in_bbox = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
        if np.any(in_bbox):
            xs.append(x[in_bbox])
            ys.append(y[in_bbox])
            zs.append(z[in_bbox])
            cs.append(c[in_bbox])

    if not xs:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=np.int64)

    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(zs),
        np.concatenate(cs),
    )


def _compute_point_features(
    row,
    x_pts,
    y_pts,
    z_pts,
    points_classes,
    gnss_offset_z=0.0,
    is_bridge_flag=0,
    bridge_corridor_count=0,
):
    # Repere antenne: meme convention que la visualisation (decalage ajoute a z_relative).
    z_ant = float(row["z_gt_ign69"])
    gt_x = float(row["x_gt"])
    gt_y = float(row["y_gt"])

    dx = x_pts - gt_x
    dy = y_pts - gt_y
    dist_horizontale = np.sqrt(dx ** 2 + dy ** 2) # Distance horizontale des points par rapport a la position GT.
    dist_horizontale = np.maximum(dist_horizontale, 0.1)
    z_relative = z_pts - z_ant + float(gnss_offset_z)
    n_points = int(points_classes.size)

    above = (z_relative > 0.0) & (dist_horizontale > 0.8) & ~np.isin(points_classes.astype(np.int64), SKY_MASK_IGNORED_CLASSES)
    not_ignored = ~np.isin(points_classes.astype(np.int64), SKY_MASK_IGNORED_CLASSES)

    sky_mask = compute_sky_mask_deg_from_relative(z_relative, dist_horizontale, points_classes)

    is_build = 6 in points_classes[above] if np.any(above) else False
    is_veg = np.any(np.isin(points_classes[above], [3, 4, 5])) if np.any(above) else False
    is_bridge = bool(is_bridge_flag)
    under = int(np.any((dist_horizontale < 2.0) & (z_pts > z_ant) & (~np.isin(points_classes, [0, 1]))))

    signal_denied = int(
        np.isnan(row["longitude_gnss"]) or np.isnan(row["latitude_gnss"]) or np.isnan(row["altitude_gnss"])
    )

    radial_near = dist_horizontale <= 5.0
    radial_mid = (dist_horizontale > 5.0) & (dist_horizontale <= 15.0)
    radial_far = (dist_horizontale > 15.0) & (dist_horizontale <= 30.0)

    z_p25 = _safe_percentile(z_relative, 25)
    z_p75 = _safe_percentile(z_relative, 75)
    z_std = float(np.std(z_relative)) if z_relative.size else 0.0

    occupation_ciel = _compute_azimuth_occupancy_ratio(dx, dy, above, bins=AZIMUTH_BINS)

    is_veg_low = points_classes == 3
    is_veg_mid = points_classes == 4
    is_veg_high = points_classes == 5
    is_veg_any = np.isin(points_classes, [3, 4, 5])
    bridge_above_mask = (points_classes == BRIDGE_CLASS_CODE) & (z_relative > 0.0) & (dist_horizontale > 0.8)

    canopee_mask = is_veg_any & (z_relative > float(CANOPEE_MIN_Z_REL))
    overhead_mask = not_ignored & (z_relative > 0.0) & (dist_horizontale < 5.0)

    return {
        "sky_mask_deg": round(sky_mask, 2),
        "obs_type": _compute_obs_type(is_build, is_veg, is_bridge),
        "is_bridge": int(is_bridge),
        "is_under_structure": under,
        "veg_density": round(np.mean(np.isin(points_classes, [3, 4, 5])), 3),
        "n_points_zone": n_points,
        "enough_points_flag": int(n_points >= MIN_POINTS_FOR_STABLE_FEATURES),
        "density_near_0_5m": round(_safe_ratio(radial_near, n_points), 3),
        "density_mid_5_15m": round(_safe_ratio(radial_mid, n_points), 3),
        "density_far_15_30m": round(_safe_ratio(radial_far, n_points), 3),
        "zrel_p50": round(_safe_percentile(z_relative, 50), 3),
        "zrel_p90": round(_safe_percentile(z_relative, 90), 3),
        "zrel_p95": round(_safe_percentile(z_relative, 95), 3),
        "zrel_p99": round(_safe_percentile(z_relative, 99), 3),
        "zrel_iqr": round(z_p75 - z_p25, 3),
        "zrel_std": round(z_std, 3),
        "occupation_ciel_azimuth_ratio": round(occupation_ciel, 3),
        "building_density": round(_safe_ratio(points_classes == 6, n_points), 3),
        "vegetation_density_low": round(_safe_ratio(is_veg_low, n_points), 3),
        "vegetation_density_mid": round(_safe_ratio(is_veg_mid, n_points), 3),
        "vegetation_density_high": round(_safe_ratio(is_veg_high, n_points), 3),
        "bridge_density": round(_safe_ratio(points_classes == BRIDGE_CLASS_CODE, n_points), 3),
        "bridge_above_density": round(_safe_ratio(bridge_above_mask, n_points), 3),
        "bridge_above_count": int(np.sum(bridge_above_mask)),
        "bridge_corridor_count": int(bridge_corridor_count),
        "canopee_ratio": round(_safe_ratio(canopee_mask, n_points), 3),
        "obstacle_overhead_ratio": round(_safe_ratio(overhead_mask, n_points), 3),
        "signal_denied": signal_denied,
    }


def process_single_tile_labelisation(
    tile,
    trajectory_df,
    search_radius=20.0,
    decimation_factor=1,
    spatial_mode="circle",
    corridor_width=10.0,
    corridor_length=None,
    gnss_offset_z=0.0,
    lidar_chunk_size=1_000_000,
    bridge_point_threshold=BRIDGE_MIN_POINTS_THRESHOLD,
    bridge_corridor_width=BRIDGE_CORRIDOR_WIDTH_M,
    bridge_time_horizon_s=BRIDGE_TIME_HORIZON_S,
    min_elevation_angle_deg=MIN_ELEVATION_ANGLE_DEG,
):
    """Traite une dalle LiDAR et extrait les features utiles a la labellisation."""
    try:
        with laspy.open(tile) as fh:
            h = fh.header
            mask = (
                  (trajectory_df["x_gt"] >= h.min[0])
                & (trajectory_df["x_gt"] <= h.max[0])
                & (trajectory_df["y_gt"] >= h.min[1])
                & (trajectory_df["y_gt"] <= h.max[1])
            )

            df_tile = trajectory_df[mask].copy()
            if len(df_tile) == 0:
                return None
            half_w, half_l, preselect_radius = _compute_spatial_params(
                search_radius, spatial_mode, corridor_width, corridor_length
            )

            # Couverture suffisante pour le couloir pont (longueur = vitesse * horizon).
            max_speed = float(df_tile.get("speed_mps", pd.Series([0.0])).max()) if len(df_tile) else 0.0
            max_bridge_length = max(max_speed * float(bridge_time_horizon_s), 0.5)
            bridge_preselect_radius = float(np.hypot(max_bridge_length / 2.0, float(bridge_corridor_width) / 2.0))
            preselect_radius = max(preselect_radius, bridge_preselect_radius)

            l_x, l_y, l_z, l_c = _read_tile_points_bbox_filtered(
                fh,
                df_tile,
                preselect_radius=preselect_radius,
                decimation_factor=decimation_factor,
                lidar_chunk_size=lidar_chunk_size,
            )

            if l_x.size == 0:
                features = []
                for _, row in df_tile.iterrows():
                    signal_denied = int(
                        np.isnan(row["longitude_gnss"])
                        or np.isnan(row["latitude_gnss"])
                        or np.isnan(row["altitude_gnss"])
                    )
                    features.append(_empty_feature(signal_denied=signal_denied))
                return pd.concat([df_tile.reset_index(drop=True), pd.DataFrame(features)], axis=1)

            tree = cKDTree(np.stack((l_x, l_y), axis=1))
            all_indices = tree.query_ball_point(df_tile[["x_gt", "y_gt"]].values, preselect_radius)

            features = []
            for i, indices in enumerate(all_indices):
                row = df_tile.iloc[i]
                signal_denied = int(
                    np.isnan(row["longitude_gnss"]) or np.isnan(row["latitude_gnss"]) or np.isnan(row["altitude_gnss"])
                )

                if not indices:
                    features.append(_empty_feature(signal_denied=signal_denied))
                    continue

                x_pts = l_x[indices]
                y_pts = l_y[indices]
                z_pts = l_z[indices]
                points_classes = l_c[indices]

                x_pts, y_pts, z_pts, points_classes = _apply_elevation_angle_mask(
                    x_pts,
                    y_pts,
                    z_pts,
                    points_classes,
                    row,
                    gnss_offset_z=gnss_offset_z,
                    min_elevation_angle_deg=min_elevation_angle_deg,
                )
                if points_classes.size == 0:
                    features.append(_empty_feature(signal_denied=signal_denied))
                    continue

                is_bridge_flag, bridge_corridor_count = _compute_bridge_from_corridor(
                    row,
                    x_pts,
                    y_pts,
                    z_pts,
                    points_classes,
                    bridge_point_threshold=bridge_point_threshold,
                    bridge_corridor_width=bridge_corridor_width,
                    bridge_time_horizon_s=bridge_time_horizon_s,
                    gnss_offset_z=gnss_offset_z,
                )

                if spatial_mode == "corridor":
                    in_corridor = _corridor_filter(
                        x_pts,
                        y_pts,
                        float(row["x_gt"]),
                        float(row["y_gt"]),
                        float(row["dir_x"]),
                        float(row["dir_y"]),
                        half_width=half_w,
                        half_length=half_l,
                    )
                    if not np.any(in_corridor):
                        features.append(_empty_feature(signal_denied=signal_denied))
                        continue
                    x_pts = x_pts[in_corridor]
                    y_pts = y_pts[in_corridor]
                    z_pts = z_pts[in_corridor]
                    points_classes = points_classes[in_corridor]

                features.append(
                    _compute_point_features(
                        row,
                        x_pts,
                        y_pts,
                        z_pts,
                        points_classes,
                        gnss_offset_z=gnss_offset_z,
                        is_bridge_flag=is_bridge_flag,
                        bridge_corridor_count=bridge_corridor_count,
                    )
                )

            return pd.concat([df_tile.reset_index(drop=True), pd.DataFrame(features)], axis=1)

    except Exception as e:
        return f"Erreur {tile}: {e}"


def process_lidar_tiles_for_labelisation(
    traj_id,
    output_csv=None,
    search_radius=RAYON_RECHERCHE,
    decimation_factor=DECIMATION_FACTOR,
    n_workers=N_WORKERS,
    spatial_mode="circle",
    corridor_width=10.0,
    corridor_length=None,
    gnss_offset_z=None,
    lidar_chunk_size=1_000_000,
    bridge_point_threshold=BRIDGE_MIN_POINTS_THRESHOLD,
    bridge_corridor_width=BRIDGE_CORRIDOR_WIDTH_M,
    bridge_time_horizon_s=BRIDGE_TIME_HORIZON_S,
    min_elevation_angle_deg=MIN_ELEVATION_ANGLE_DEG,
):
    """Pipeline parallele d'extraction des features LiDAR pour la labellisation."""
    if spatial_mode not in {"circle", "corridor"}:
        raise ValueError("spatial_mode doit etre 'circle' ou 'corridor'")

    config = get_traj_paths(traj_id)
    if gnss_offset_z is None:
        gnss_offset = config.get("gnss_offset", (0.0, 0.0, 0.0))
        gnss_offset_z = float(gnss_offset[2]) if isinstance(gnss_offset, (tuple, list)) and len(gnss_offset) >= 3 else 0.0
    df_traj = pd.read_csv(config["sync_csv"])
    df_traj = _add_local_heading_vectors(df_traj)
    lidar_folder = config["lidar_tiles"]

    if output_csv is None:
        output_csv = config["lidar_features_csv"]

    tiles = [f for f in os.listdir(lidar_folder) if f.endswith((".las", ".laz"))]
    tiles = [lidar_folder / t for t in tiles]

    print(f"Lancement du calcul parallele sur {n_workers} coeurs...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        process_func = functools.partial(
            process_single_tile_labelisation,
            trajectory_df=df_traj,
            search_radius=search_radius,
            decimation_factor=decimation_factor,
            spatial_mode=spatial_mode,
            corridor_width=corridor_width,
            corridor_length=corridor_length,
            gnss_offset_z=gnss_offset_z,
            lidar_chunk_size=lidar_chunk_size,
            bridge_point_threshold=bridge_point_threshold,
            bridge_corridor_width=bridge_corridor_width,
            bridge_time_horizon_s=bridge_time_horizon_s,
            min_elevation_angle_deg=min_elevation_angle_deg,
        )
        results = list(tqdm(executor.map(process_func, tiles), total=len(tiles)))

    final_dfs = [r for r in results if isinstance(r, pd.DataFrame)]
    errors = [r for r in results if isinstance(r, str)]

    if not final_dfs:
        raise RuntimeError(
            f"Aucun resultat produit lors de l'extraction des features LiDAR. {errors[0] if errors else ''}"
        )

    df_out = pd.concat(final_dfs, ignore_index=True)
    if "time_utc" in df_out.columns:
        try:
            df_out["time_utc"] = pd.to_datetime(df_out["time_utc"])
            df_out = df_out.sort_values("time_utc").reset_index(drop=True)
        except Exception:
            pass

    required_cols = [
        "time_utc",
        "latitude_gt",
        "longitude_gt",
        "sky_mask_deg",
        "obs_type",
        "is_under_structure",
        "veg_density",
    ]
    missing = [c for c in required_cols if c not in df_out.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour la labelisation: {missing}")

    df_out.to_csv(output_csv, index=False)
    print(f"\nExtraction terminee. Fichier features: {output_csv}")
    print(f"Erreurs signalees: {len(errors)}")
    return df_out


if __name__ == "__main__":
    traj_id = "BORDEAUX_COUTRAS"
    process_lidar_tiles_for_labelisation(traj_id)