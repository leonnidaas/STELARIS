from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd
from LABELISATION_AUTO_LIDAR_HD_IGN.labelisation import _load_labelisation_params
from LABELISATION_AUTO_LIDAR_HD_IGN.run_params import load_latest_labelisation_run_params
from utils import list_traj_ids
from modules.selection_scenario import scenario_start_label, load_traj_config, load_sorted_csv


def list_scenarios() -> list[str]:
    return list_traj_ids()


def build_trajets_map(scenarios: list[str]) -> dict[str, list[str]]:
    trajets_map: dict[str, list[str]] = {}
    for scenario_id in scenarios:
        trajet_key = scenario_id.split("__", 1)[0] if "__" in scenario_id else scenario_id
        trajets_map.setdefault(trajet_key, []).append(scenario_id)

    for key in trajets_map:
        trajets_map[key] = sorted(trajets_map[key])

    return trajets_map


def load_latest_labelisation_params(traj_id: str) -> dict:
    config = load_traj_config(traj_id)
    _, payload = load_latest_labelisation_run_params(config, traj_id ,source="IGN")
    if not payload:
        return {}

    params = payload.get("labellisation", {}).get("params_labelisation", {})
    return params if isinstance(params, dict) else {}


def build_scenario_sources_zip(traj_id: str) -> tuple[bytes | None, str, list[str]]:
    """Build an in-memory ZIP containing GT + RINEX OBS + RINEX NAV for one scenario."""
    cfg = load_traj_config(traj_id)

    files: list[tuple[str, Path | None]] = [
        ("GT", cfg.get("raw_gt")),
        ("RINEX_OBS", cfg.get("obs_file")),
        ("RINEX_NAV", cfg.get("nav_file")),
    ]

    missing: list[str] = []
    present: list[tuple[str, Path]] = []
    for label, file_path in files:
        p = Path(file_path) if file_path is not None else None
        if p is None or not p.exists() or not p.is_file():
            missing.append(label)
            continue
        present.append((label, p))

    if not present:
        return None, f"stelaris_sources_{traj_id}.zip", missing

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for label, path in present:
            zf.write(path, arcname=f"{label}/{path.name}")

    buffer.seek(0)
    return buffer.getvalue(), f"stelaris_sources_{traj_id}.zip", missing


def _normalize_label(val):
    if val is None:
        return None
    txt = str(val).strip()
    if not txt or txt.lower() == "nan":
        return None
    return txt.lower().replace("_", "-")


def _row_value(row, keys, default=0.0):
    for key in keys:
        if key in row.index:
            value = row.get(key)
            if not pd.isna(value):
                return value
    return default


def explain_label_reason(row, params_cfg: dict | None = None) -> tuple[str, list[str]]:
    cfg = _load_labelisation_params(params_cfg)
    label = _normalize_label(row.get("label")) or "other"

    values = {
        "bridge_above_count": int(_row_value(row, ("bridge_above_count",), 0) or 0),
        "bridge_above_density": float(_row_value(row, ("bridge_above_density",), 0.0) or 0.0),
        "obstacle_overhead_ratio": float(_row_value(row, ("obstacle_overhead_ratio",), 0.0) or 0.0),
        "zrel_p95": float(_row_value(row, ("zrel_p95",), 0.0) or 0.0),
        "zrel_p99": float(_row_value(row, ("zrel_p99",), 0.0) or 0.0),
        "building_density": float(_row_value(row, ("building_density",), 0.0) or 0.0),
        "veg_density": float(_row_value(row, ("veg_density",), 0.0) or 0.0),
        "density_near_0_5m": float(_row_value(row, ("density_near_0_5m",), 0.0) or 0.0),
        "density_far_15_30m": float(_row_value(row, ("density_far_15_30m",), 0.0) or 0.0),
        "occupation_ciel_azimuth_ratio": float(_row_value(row, ("occupation_ciel_azimuth_ratio",), 0.0) or 0.0),
        "sky_mask_deg": float(_row_value(row, ("sky_mask_smoothed", "sky_mask_deg"), 0.0) or 0.0),
        "zrel_std": float(_row_value(row, ("zrel_std",), 0.0) or 0.0),
        "zrel_iqr": float(_row_value(row, ("zrel_iqr",), 0.0) or 0.0),
        "n_points_zone": int(_row_value(row, ("n_points_zone",), 0) or 0),
        "speed_gt_mps_smooth": float(_row_value(row, ("speed_gt_mps_smooth", "speed_gt_mps", "speed"), 0.0) or 0.0),
        "is_under_structure": int(_row_value(row, ("is_under_structure",), 0) or 0),
        "obs_type": int(_row_value(row, ("obs_type",), 0) or 0),
        "bridge_recent_1s": int(_row_value(row, ("bridge_recent_1s",), 0) or 0),
        "bridge_core": int(_row_value(row, ("bridge_core",), 0) or 0),
        "enough_points_flag": int(_row_value(row, ("enough_points_flag",), 0) or 0),
        "vegetation_density_high": float(_row_value(row, ("vegetation_density_high",), 0.0) or 0.0),
        "vegetation_density_mid": float(_row_value(row, ("vegetation_density_mid",), 0.0) or 0.0),
        "vegetation_density_low": float(_row_value(row, ("vegetation_density_low",), 0.0) or 0.0),
        "canopee_ratio": float(_row_value(row, ("canopee_ratio",), 0.0) or 0.0),
    }

    if label == "signal_denied":
        return (
            "Label prioritaire: signal_denied (le signal GNSS est marque comme refuse).",
            [f"signal_denied={int(_row_value(row, ('signal_denied',), 0) or 0)}"],
        )

    if label == "gare":
        checks = [
            f"is_under_structure={values['is_under_structure']} (= 1 attendu)",
            f"speed_gt_mps_smooth={values['speed_gt_mps_smooth']:.2f} (<= {cfg['seuil_vitesse_gare_mps']})",
            f"n_points_zone={values['n_points_zone']} (>= {cfg['seuil_min_points_zone']} ou enough_points_flag=1)",
            f"obs_type={values['obs_type']} (1 ou 3 favorise la branche gare)",
            f"building_density={values['building_density']:.3f} (> {cfg['seuil_building_density']})",
            f"obstacle_overhead_ratio={values['obstacle_overhead_ratio']:.3f} (> {cfg['seuil_overhead_gare']})",
            f"density_near_0_5m={values['density_near_0_5m']:.3f} (> {cfg['seuil_gare_density_near']})",
            f"zrel_iqr={values['zrel_iqr']:.2f} (> {cfg['seuil_gare_zrel_iqr']})",
        ]
        return (
            "Label gare retenu car le point est sous structure, avec vitesse faible et au moins une signature locale de zone gare.",
            checks,
        )

    if label == "bridge":
        checks = [
            f"bridge_recent_1s={values['bridge_recent_1s']} (= 1 attendu si la memoire de passage sous ouvrage est active)",
            f"bridge_above_count={values['bridge_above_count']} (>= {cfg['seuil_bridge_above_count_min']})",
            f"bridge_above_density={values['bridge_above_density']:.3f} (>= {cfg['seuil_bridge_above_density_min']})",
            f"obstacle_overhead_ratio={values['obstacle_overhead_ratio']:.3f} (>= {cfg['seuil_overhead_bridge']})",
            f"zrel_p95={values['zrel_p95']:.2f} / zrel_p99={values['zrel_p99']:.2f} (seuils {cfg['seuil_bridge_zrel_p95_min']}/{cfg['seuil_bridge_zrel_p99_min']})",
        ]
        return (
            "Bridge retenu car le point fait partie d'un passage sous ouvrage avec une signature overhead/hauteur compatible.",
            checks,
        )

    if label == "tree":
        checks = [
            f"veg_density={values['veg_density']:.3f} (seuil {cfg['seuil_vegetation']})",
            f"obs_type={values['obs_type']} (= 2 favorise la branche tree)",
            f"vegetation_density_high={values['vegetation_density_high']:.3f} (>= {cfg['seuil_veg_high']})",
            f"vegetation_density_mid={values['vegetation_density_mid']:.3f} (>= {cfg['seuil_tree_veg_mid']})",
            f"zrel_p90={float(_row_value(row, ('zrel_p90',), 0.0) or 0.0):.2f} (>= {cfg['seuil_tree_zrel_p90']})",
            f"canopee_ratio={values['canopee_ratio']:.3f} (>= {cfg['seuil_canopee']})",
        ]
        return (
            "Tree retenu car la vegetation domine la zone locale.",
            checks,
        )

    if label == "build":
        checks = [
            f"obs_type={values['obs_type']} (= 1 favorise la branche build)",
            f"building_density={values['building_density']:.3f} (>= {cfg['seuil_building_density']})",
            f"density_mid_5_15m={float(_row_value(row, ('density_mid_5_15m',), 0.0) or 0.0):.3f} (>= {cfg['seuil_build_density_mid']})",
            f"zrel_p95={values['zrel_p95']:.2f} (>= {cfg['seuil_build_zrel_p95']})",
            f"vegetation_density_low={values['vegetation_density_low']:.3f} (< 0.25 dans la branche finale)",
        ]
        return (
            "Build retenu car la signature de bati est dominante autour du train.",
            checks,
        )

    if label == "open-sky":
        checks = [
            f"sky_mask_deg / smoothed={values['sky_mask_deg']:.2f} (< {cfg['seuil_ciel_ouvert']})",
            f"occupation_ciel_azimuth_ratio={values['occupation_ciel_azimuth_ratio']:.3f} (< {cfg['seuil_occupation_ciel']})",
            f"obstacle_overhead_ratio={values['obstacle_overhead_ratio']:.3f} (< {cfg['seuil_overhead_bridge']})",
            f"building_density={values['building_density']:.3f} (< {cfg['seuil_building_density']})",
            f"veg_density={values['veg_density']:.3f} (< {cfg['seuil_vegetation']})",
            f"density_near_0_5m={values['density_near_0_5m']:.3f} (< {cfg['seuil_open_sky_density_near_max']})",
            f"density_far_15_30m={values['density_far_15_30m']:.3f} (< {cfg['seuil_open_sky_density_far_max']})",
            f"zrel_p95={values['zrel_p95']:.2f} (< {cfg['seuil_open_sky_zrel_p95_max']})",
            f"zrel_std={values['zrel_std']:.2f} (< {cfg['seuil_open_sky_zrel_std_max']})",
        ]
        return (
            "Open-sky retenu car la zone autour de l'antenne reste peu obstruee.",
            checks,
        )

    if label == "mixed":
        checks = [
            f"obs_type={values['obs_type']} (= 3 ou 0 dans certaines branches mixed)",
            f"veg_density={values['veg_density']:.3f} (> {cfg['seuil_melange']} si obs_type=0)",
            f"building_density={values['building_density']:.3f} (> {cfg['seuil_mixed_building']})",
            f"zrel_iqr={values['zrel_iqr']:.2f} (> {cfg['seuil_mixed_zrel_iqr']})",
            f"zrel_std={values['zrel_std']:.2f} (> {cfg['seuil_mixed_zrel_std']})",
            f"vegetation_density_mid={values['vegetation_density_mid']:.3f}",
        ]
        return (
            "Mixed retenu en fallback quand les signatures bati / vegetation coexistent sans dominance nette.",
            checks,
        )

    checks = [
        f"building_density={values['building_density']:.3f}",
        f"veg_density={values['veg_density']:.3f}",
        f"sky_mask_deg / smoothed={values['sky_mask_deg']:.2f}",
        f"obstacle_overhead_ratio={values['obstacle_overhead_ratio']:.3f}",
        f"bridge_above_count={values['bridge_above_count']}",
    ]
    return (
        "Label determine par les regles de priorite avec fallback sur les classes restantes.",
        checks,
    )


