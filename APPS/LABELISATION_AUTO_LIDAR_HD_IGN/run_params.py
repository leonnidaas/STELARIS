"""
Gestion de la sauvegarde et du chargement des paramètres de labellisation.
"""

import json
from datetime import datetime
from pathlib import Path
from utils import INTERIM_IGN_SUBDIR

def _json_safe(value):
    """Convertit recursivement les types numpy/Path en types JSON natifs."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "as_posix"):
        try:
            return value.as_posix()
        except Exception:
            pass
    return value


def _labelisation_params_paths(config, traj_id, source: str = "IGN", timestamp=None):
    """Construit les chemins de fichiers pour sauvegarder les paramètres de labellisation.
    parameters:
    - config: dict de configuration contenant au moins "interim_dir"
    - traj_id: identifiant du trajet
    - source: source des données 
    - timestamp: horodatage
    """
    interim_dir = Path(config["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    source_subdir = str(INTERIM_IGN_SUBDIR).strip() or "ign"
    json_dir = interim_dir / source_subdir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_file = json_dir / f"params_pipeline_labelisation_{source}_{traj_id}_{ts}.json"
    latest_file = json_dir / f"params_pipeline_labelisation_{source}_{traj_id}_latest.json"
    print(f"Chemins de sauvegarde des paramètres de labellisation:\n  Run file: {run_file}\n  Latest file: {latest_file}")
    return ts, run_file, latest_file


def build_labelisation_run_payload(traj_id, pipeline_opts, params_labelisation, decimation_factor, timestamp):
    return {
        "traj_id": traj_id,
        "timestamp": timestamp,
        "extract_features_lidar_labelisation": {
            "extract_features": bool(pipeline_opts["extract_features"]),
            "n_workers": int(pipeline_opts["nb_workers"]),
            "decimation_factor": int(decimation_factor),
            "search_radius": float(pipeline_opts["search_radius"]),
            "spatial_mode": str(pipeline_opts["spatial_mode"]),
            "corridor_width": float(pipeline_opts["corridor_width"]),
            "corridor_length": float(pipeline_opts["corridor_length"]),
            "min_elevation_angle_deg": float(pipeline_opts.get("min_elevation_angle_deg", 0.0)),
        },
        "labellisation": {
            "params_labelisation": _json_safe(params_labelisation),
        },
        "pipeline_options": _json_safe(pipeline_opts),
    }


def store_labelisation_run_params(
    config,
    traj_id,
    source="IGN",
    pipeline_opts=None,
    params_labelisation=None,
    decimation_factor=1,
):
    if pipeline_opts is None:
        pipeline_opts = {}
    if params_labelisation is None:
        params_labelisation = {}

    ts, run_file, latest_file = _labelisation_params_paths(config, traj_id, source)
    payload = build_labelisation_run_payload(
        traj_id=traj_id,
        pipeline_opts=pipeline_opts,
        params_labelisation=params_labelisation,
        decimation_factor=decimation_factor,
        timestamp=ts,
    )

    for path in (run_file, latest_file):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return str(run_file), str(latest_file), payload


def load_latest_labelisation_run_params(config, traj_id, source="IGN"):
    _, _, latest_file = _labelisation_params_paths(config, traj_id, source=source, timestamp="dummy")
    if not latest_file.exists():
        return None, None

    with open(latest_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return str(latest_file), payload
