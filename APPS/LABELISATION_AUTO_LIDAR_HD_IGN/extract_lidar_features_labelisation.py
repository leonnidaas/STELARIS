import functools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import tempfile

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
BRIDGE_TIME_HORIZON_S = 2.5
BRIDGE_MIN_POINTS_THRESHOLD = 10
MIN_ELEVATION_ANGLE_DEG = 0.0
AZIMUTH_OCCUPANCY_BINS = 36
AZIMUTH_OCCUPANCY_MIN_Z_REL = 1.5
AZIMUTH_OCCUPANCY_MAX_DIST = 30.0


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

    if "velocity" in d.columns:
        d["speed_mps"] = pd.to_numeric(d["velocity"], errors="coerce").abs().fillna(0.0)
    else:
        if "time_utc" in d.columns:
            dt = pd.to_datetime(d["time_utc"], errors="coerce").diff().dt.total_seconds().to_numpy(dtype=float)
        else:
            dt = np.full_like(vx, np.nan, dtype=float)
        dist = np.hypot(vx, vy)
        speed = np.divide(dist, np.where(dt > 1e-6, dt, np.nan))
        speed = np.where(np.isfinite(speed), np.abs(speed), 0.0)
        d["speed_mps"] = speed
    return d


def _corridor_filter(local_x, local_y, center_x, center_y, dir_x, dir_y, half_width, half_length):
    """Filtre booléen des points inclus dans un rectangle orienté (couloir)."""
    dx = local_x - center_x
    dy = local_y - center_y

    longitudinal = dx * dir_x + dy * dir_y
    lateral = -dx * dir_y + dy * dir_x
    return (np.abs(longitudinal) <= half_length) & (np.abs(lateral) <= half_width)


@dataclass(frozen=True)
class SpatialConfig:
    search_radius: float
    spatial_mode: str
    corridor_width: float
    corridor_length: float | None
    bridge_point_threshold: int
    bridge_corridor_width: float
    bridge_time_horizon_s: float
    min_elevation_angle_deg: float


def _empty_feature(signal_denied=0):
    return {
        "sky_mask_deg": 0,
        "obs_type": 0,
        "is_bridge": 0,
        "is_under_structure": 0,
        "veg_density_raw": 0.0,
        "veg_density": 0,
        "azimuth_occupancy_ratio": 0.0,
        "effective_veg_density": 0.0,
        "zrel_p95": 0.0,
        "building_density": 0.0,
        "obstacle_overhead_ratio": 0.0,
        "signal_denied": int(signal_denied),
    }


def _resolve_spatial_params(df_tile, cfg: SpatialConfig):
    corridor_length = cfg.search_radius if cfg.corridor_length is None else float(cfg.corridor_length)
    half_w = max(float(cfg.corridor_width) / 2.0, 0.01)
    half_l = max(float(corridor_length), 0.01)
    base_preselect = float(cfg.search_radius) if cfg.spatial_mode == "circle" else float(np.hypot(half_l, half_w))

    if len(df_tile) and "speed_mps" in df_tile.columns:
        speed_series = pd.to_numeric(df_tile["speed_mps"], errors="coerce").fillna(0.0)
        max_speed = float(speed_series.max())
    else:
        max_speed = 0.0
    max_bridge_length = max(max_speed * float(cfg.bridge_time_horizon_s), 0.5)
    bridge_preselect = float(np.hypot(max_bridge_length / 2.0, float(cfg.bridge_corridor_width) / 2.0))
    return half_w, half_l, max(base_preselect, bridge_preselect)


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


def _row_speed_mps(row):
    for col in ("speed_mps", "velocity", "speed"):
        if col in row.index and pd.notna(row[col]):
            return abs(float(row[col]))
    return 0.0


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


def _compute_azimuth_occupancy_ratio(dx, dy, z_rel, dist_h, bins=AZIMUTH_OCCUPANCY_BINS):
    if dx.size == 0 or bins <= 0:
        return 0.0

    obstacle_mask = (z_rel > float(AZIMUTH_OCCUPANCY_MIN_Z_REL)) & (dist_h < float(AZIMUTH_OCCUPANCY_MAX_DIST))
    if not np.any(obstacle_mask):
        return 0.0

    az = np.arctan2(dy[obstacle_mask], dx[obstacle_mask])
    az = np.mod(az, 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, int(bins) + 1)
    counts, _ = np.histogram(az, bins=edges)
    occupied_bins = np.sum(counts > 0)
    return float(occupied_bins) / float(bins)


def _compute_compact_features(row, x_pts, y_pts, z_pts, points_classes, cfg: SpatialConfig, gnss_offset_z=0.0):
    if points_classes.size == 0:
        return _empty_feature(
            signal_denied=int(
                np.isnan(row["longitude_gnss"]) or np.isnan(row["latitude_gnss"]) or np.isnan(row["altitude_gnss"])
            )
        )

    gt_x = float(row["x_gt"])
    gt_y = float(row["y_gt"])
    z_ant = float(row["z_gt_ign69"])

    dx = x_pts - gt_x
    dy = y_pts - gt_y
    dist_h = np.hypot(dx, dy)
    z_rel = z_pts - z_ant + float(gnss_offset_z)

    elev_deg = np.degrees(np.arctan2(z_rel, np.maximum(dist_h, 1e-6)))
    keep = elev_deg >= float(cfg.min_elevation_angle_deg)
    if not np.any(keep):
        return _empty_feature(
            signal_denied=int(
                np.isnan(row["longitude_gnss"]) or np.isnan(row["latitude_gnss"]) or np.isnan(row["altitude_gnss"])
            )
        )

    x_pts = x_pts[keep]
    y_pts = y_pts[keep]
    z_pts = z_pts[keep]
    points_classes = points_classes[keep]
    dx = dx[keep]
    dy = dy[keep]
    dist_h = np.maximum(dist_h[keep], 0.1)
    z_rel = z_rel[keep]

    classes64 = points_classes.astype(np.int64, copy=False)
    n_points = int(classes64.size)
    signal_denied = int(
        np.isnan(row["longitude_gnss"]) or np.isnan(row["latitude_gnss"]) or np.isnan(row["altitude_gnss"])
    )

    ignored = np.isin(classes64, SKY_MASK_IGNORED_CLASSES)
    not_ignored = ~ignored
    above = (z_rel > 0.0) & (dist_h > 0.8) & not_ignored

    sky_mask = compute_sky_mask_deg_from_relative(z_rel, dist_h, classes64)
    veg_mask = np.isin(classes64, [3, 4, 5])
    build_mask = classes64 == 6
    bridge_mask = classes64 == BRIDGE_CLASS_CODE

    speed_mps = _row_speed_mps(row)
    bridge_length = max(speed_mps * float(cfg.bridge_time_horizon_s), 0.5)
    in_bridge_corridor = _corridor_filter(
        x_pts,
        y_pts,
        gt_x,
        gt_y,
        float(row["dir_x"]),
        float(row["dir_y"]),
        half_width=max(float(cfg.bridge_corridor_width) / 2.0, 0.1),
        half_length=max(bridge_length / 2.0, 0.25),
    )
    bridge_count = int(np.sum(bridge_mask & in_bridge_corridor & (z_rel > 0.0)))
    is_bridge = int(bridge_count >= int(cfg.bridge_point_threshold))

    above_classes = classes64[above]
    is_build = bool(np.any(above_classes == 6)) if above_classes.size else False
    is_veg = bool(np.any(np.isin(above_classes, [3, 4, 5]))) if above_classes.size else False
    obs_type = int(np.select([is_build and is_veg, is_build, is_veg, bool(is_bridge)], [3, 1, 2, 4], default=0))

    veg_density_raw = _safe_ratio(veg_mask, n_points)
    azimuth_occupancy_ratio = _compute_azimuth_occupancy_ratio(dx, dy, z_rel, dist_h, bins=AZIMUTH_OCCUPANCY_BINS)
    effective_veg_density = veg_density_raw * azimuth_occupancy_ratio

    overhead_ratio = _safe_ratio(not_ignored & (z_rel > 0.0) & (dist_h < 5.0), n_points)
    under_structure = int(np.any((dist_h < 2.0) & (z_pts > z_ant) & (~np.isin(classes64, [0, 1]))))

    return {
        "sky_mask_deg": round(float(sky_mask), 2),
        "obs_type": obs_type,
        "is_bridge": int(is_bridge),
        "is_under_structure": under_structure,
        "veg_density_raw": round(veg_density_raw, 3),
        "veg_density": round(effective_veg_density, 3),
        "azimuth_occupancy_ratio": round(azimuth_occupancy_ratio, 3),
        "effective_veg_density": round(effective_veg_density, 3),
        "building_density": round(_safe_ratio(build_mask, n_points), 3),
        "zrel_p95": round(float(np.percentile(z_rel, 95)) if z_rel.size else 0.0, 3),
        "obstacle_overhead_ratio": round(overhead_ratio, 3),
        "signal_denied": signal_denied,
    }


def process_single_tile_labelisation(
    tile,
    trajectory_df,
    spatial_cfg,
    decimation_factor=1,
    gnss_offset_z=0.0,
    lidar_chunk_size=250_000,
    temp_output_csv=None,
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
            half_w, half_l, preselect_radius = _resolve_spatial_params(df_tile, spatial_cfg)

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
                result_df = pd.concat([df_tile.reset_index(drop=True), pd.DataFrame(features)], axis=1)
                if temp_output_csv is not None:
                    result_df.to_csv(temp_output_csv, index=False)
                    return str(temp_output_csv)
                return result_df

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

                if spatial_cfg.spatial_mode == "corridor":
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
                    _compute_compact_features(
                        row,
                        x_pts,
                        y_pts,
                        z_pts,
                        points_classes,
                        cfg=spatial_cfg,
                        gnss_offset_z=gnss_offset_z,
                    )
                )

            result_df = pd.concat([df_tile.reset_index(drop=True), pd.DataFrame(features)], axis=1)
            if temp_output_csv is not None:
                result_df.to_csv(temp_output_csv, index=False)
                return str(temp_output_csv)
            return result_df

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
    lidar_chunk_size=250_000,
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

    spatial_cfg = SpatialConfig(
        search_radius=float(search_radius),
        spatial_mode=str(spatial_mode),
        corridor_width=float(corridor_width),
        corridor_length=float(corridor_length) if corridor_length is not None else None,
        bridge_point_threshold=int(bridge_point_threshold),
        bridge_corridor_width=float(bridge_corridor_width),
        bridge_time_horizon_s=float(bridge_time_horizon_s),
        min_elevation_angle_deg=float(min_elevation_angle_deg),
    )

    df_traj = pd.read_csv(config["sync_csv"])
    df_traj = _add_local_heading_vectors(df_traj)
    lidar_folder = config["lidar_tiles"]

    if output_csv is None:
        output_csv = config["lidar_features_csv"]

    tiles = [f for f in os.listdir(lidar_folder) if f.endswith((".las", ".laz"))]
    tiles = [lidar_folder / t for t in tiles]

    n_workers = max(1, min(int(n_workers), len(tiles), (os.cpu_count() or 1)))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"lidar_features_{traj_id}_", dir=str(output_csv.parent)))

    print(f"Lancement du calcul parallele sur {n_workers} coeurs...")

    csv_parts = []
    errors = []
    process_func = functools.partial(
        process_single_tile_labelisation,
        trajectory_df=df_traj,
        spatial_cfg=spatial_cfg,
        decimation_factor=decimation_factor,
        gnss_offset_z=gnss_offset_z,
        lidar_chunk_size=lidar_chunk_size,
    )

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_func, tile, temp_output_csv=temp_dir / f"part_{i:05d}.csv"): tile
                for i, tile in enumerate(tiles)
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                res = future.result()
                if isinstance(res, str) and res.startswith("Erreur"):
                    errors.append(res)
                elif isinstance(res, str):
                    csv_parts.append(res)

        if not csv_parts:
            raise RuntimeError(
                f"Aucun resultat produit lors de l'extraction des features LiDAR. {errors[0] if errors else ''}"
            )

        wrote_any = False
        for part_path in csv_parts:
            df_part = pd.read_csv(part_path)
            if df_part.empty:
                continue
            df_part.to_csv(output_csv, index=False, mode="a" if wrote_any else "w", header=not wrote_any)
            wrote_any = True

        if not wrote_any:
            raise RuntimeError(
                f"Aucun resultat produit lors de l'extraction des features LiDAR. {errors[0] if errors else ''}"
            )

        df_out = pd.read_csv(output_csv)
    finally:
        for p in temp_dir.glob("*.csv"):
            try:
                p.unlink()
            except Exception:
                pass
        try:
            temp_dir.rmdir()
        except Exception:
            pass

    if df_out.empty:
        raise RuntimeError(
            f"Aucun resultat produit lors de l'extraction des features LiDAR. {errors[0] if errors else ''}"
        )

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