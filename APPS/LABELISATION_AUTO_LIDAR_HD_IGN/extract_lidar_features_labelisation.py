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


def compute_sky_mask_deg_from_relative(z_rel, dist_h, c_pts, min_dist_h=0.8, ignored_classes=SKY_MASK_IGNORED_CLASSES):
    """Calcule l'angle de sky mask (deg) a partir de coordonnees relatives a l'antenne."""
    z_rel = np.asarray(z_rel, dtype=float)
    dist_h = np.asarray(dist_h, dtype=float)
    c_pts = np.asarray(c_pts)
    ignored_classes = np.asarray(ignored_classes, dtype=np.int64)

    valid = (z_rel > 0.0) & (dist_h > float(min_dist_h)) & (~np.isin(c_pts.astype(np.int64), ignored_classes))
    if not np.any(valid):
        return 0.0

    angles = np.degrees(np.arctan2(z_rel[valid], np.maximum(dist_h[valid], 1e-6)))
    return float(np.percentile(angles, 100))


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
        "is_under_structure": 0,
        "veg_density": 0,
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


def _compute_point_features(row, x_pts, y_pts, z_pts, c_pts, gnss_offset_z=0.0):
    # Repere antenne: meme convention que la visualisation (decalage ajoute a z_rel).
    z_ant = float(row["z_gt_ign69"])
    gt_x = float(row["x_gt"])
    gt_y = float(row["y_gt"])

    dist_h = np.sqrt((x_pts - gt_x) ** 2 + (y_pts - gt_y) ** 2)
    dist_h = np.maximum(dist_h, 0.1)
    z_rel = z_pts - z_ant + float(gnss_offset_z)

    above = (z_rel > 0.0) & (dist_h > 0.8) & ~np.isin(c_pts.astype(np.int64), SKY_MASK_IGNORED_CLASSES)

    sky_mask = compute_sky_mask_deg_from_relative(z_rel, dist_h, c_pts)

    is_build = 6 in c_pts[above] if np.any(above) else False
    is_veg = np.any(np.isin(c_pts[above], [3, 4, 5])) if np.any(above) else False
    is_bridge = 17 in c_pts[above] if np.any(above) else False
    under = int(np.any((dist_h < 2.0) & (z_pts > z_ant + 2.0) & (~np.isin(c_pts, [0, 1]))))

    signal_denied = int(
        np.isnan(row["longitude_gnss"]) or np.isnan(row["latitude_gnss"]) or np.isnan(row["altitude_gnss"])
    )

    return {
        "sky_mask_deg": round(sky_mask, 2),
        "obs_type": _compute_obs_type(is_build, is_veg, is_bridge),
        "is_under_structure": under,
        "veg_density": round(np.mean(np.isin(c_pts, [3, 4, 5])), 3),
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

            las = fh.read()
            step = max(int(decimation_factor), 1)
            l_x, l_y, l_z, l_c = las.x[::step], las.y[::step], las.z[::step], las.classification[::step]

            tree = cKDTree(np.stack((l_x, l_y), axis=1))
            half_w, half_l, preselect_radius = _compute_spatial_params(
                search_radius, spatial_mode, corridor_width, corridor_length
            )
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
                c_pts = l_c[indices]

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
                    c_pts = c_pts[in_corridor]

                features.append(
                    _compute_point_features(
                        row,
                        x_pts,
                        y_pts,
                        z_pts,
                        c_pts,
                        gnss_offset_z=gnss_offset_z,
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
        output_csv = config["features_csv"]

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