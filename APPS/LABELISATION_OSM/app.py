"""Pipeline OSM: filtrage PBF, extraction features, labelisation et fusion finale."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from FUSION.sync import process_final_label_fusion_from_files
from LABELISATION_OSM.download_latest_pbf import download_osm_pbf
from LABELISATION_OSM.extract_osm_features import extract_osm_features_for_traj
from LABELISATION_OSM.osm_labelisation import process_labelling_osm_for_traj
from LABELISATION_OSM.process_osm_pbf import filter_data_with_csv_trajectory_shape
from utils import get_traj_paths


def _load_params_from_json(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier de params introuvable: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Le fichier de params OSM doit contenir un objet JSON.")
    return data


def run_osm_pipeline(
    traj_id: str,
    osm_params: dict | None = None,
    run_step_1_download_pbf: bool = True,
    run_step_2_filter_pbf: bool = True,
    run_step_3_extract_features: bool = True,
    run_step_4_labelisation: bool = True,
    run_step_5_final_fusion: bool = True,
    buffer_m: float = 500.0,
    radius_m: float = 30.0,
    verbose: bool = True,
) -> bool:
    cfg = get_traj_paths(traj_id)
    params = dict(osm_params or {})

    if verbose:
        print(f"--- Pipeline OSM pour : {traj_id} ---")

    pbf_path = None

    print("=" * 50)
    print("ETAPE 1 : Telechargement PBF OSM")
    print("=" * 50)
    if run_step_1_download_pbf:
        try:
            pbf_path = Path(download_osm_pbf(region="france"))
            print(f"PBF telecharge: {pbf_path}")
        except Exception as e:
            print(f"Erreur telechargement PBF: {e}")
            return False
    else:
        print("Etape 1 ignoree (desactivee dans les options).")

    print("\n" + "=" * 50)
    print("ETAPE 2 : Filtrage spatial PBF avec trajectoire")
    print("=" * 50)
    if run_step_2_filter_pbf:
        try:
            source_pbf = pbf_path if pbf_path is not None else None
            if source_pbf is None:
                # Fallback: la fonction d'extraction choisira le dernier pbf disponible.
                source_pbf = Path(cfg["osm_pbf"]) if Path(cfg["osm_pbf"]).exists() else None
            if source_pbf is None:
                source_pbf = Path(download_osm_pbf(region="france"))

            filter_data_with_csv_trajectory_shape(
                input_pbf=source_pbf,
                output_pbf=cfg["osm_pbf"],
                input_csv=cfg["sync_csv"],
                buffer_m=float(buffer_m),
            )
            print(f"PBF filtre genere: {cfg['osm_pbf']}")
        except Exception as e:
            print(f"Erreur filtrage PBF: {e}")
            return False
    else:
        print("Etape 2 ignoree (desactivee dans les options).")

    print("\n" + "=" * 50)
    print("ETAPE 3 : Extraction des features OSM")
    print("=" * 50)
    if run_step_3_extract_features:
        try:
            pbf_for_extract = Path(cfg["osm_pbf"]) if Path(cfg["osm_pbf"]).exists() else None
            extract_osm_features_for_traj(
                traj_id=traj_id,
                pbf_path=pbf_for_extract,
                input_csv=cfg["sync_csv"],
                output_csv=cfg["osm_features_csv"],
                radius_m=float(radius_m),
                verbose=verbose,
            )
            print(f"Features OSM generees: {cfg['osm_features_csv']}")
        except Exception as e:
            print(f"Erreur extraction features OSM: {e}")
            return False
    else:
        print("Etape 3 ignoree (desactivee dans les options).")

    print("\n" + "=" * 50)
    print("ETAPE 4 : Labelisation OSM")
    print("=" * 50)
    if run_step_4_labelisation:
        try:
            process_labelling_osm_for_traj(
                traj_id=traj_id,
                params=params,
                input_csv=cfg["osm_features_csv"],
                output_csv_final=cfg["osm_labels_csv"],
                output_csv_interim=cfg["interim_osm_dir"] / f"features_osm_plus_labels_{traj_id}.csv",
                verbose=verbose,
            )
            print(f"Labels OSM generes: {cfg['osm_labels_csv']}")
        except Exception as e:
            print(f"Erreur labelisation OSM: {e}")
            return False
    else:
        print("Etape 4 ignoree (desactivee dans les options).")

    print("\n" + "=" * 50)
    print("ETAPE 5 : Fusion finale OSM (features + labels)")
    print("=" * 50)
    if run_step_5_final_fusion:
        try:
            process_final_label_fusion_from_files(
                path_fused_features=cfg["osm_features_csv"],
                path_labels=cfg["osm_labels_csv"],
                output_csv=cfg["final_fusion_osm_csv"],
                verbose=verbose,
            )
            print(f"Fusion finale OSM generee: {cfg['final_fusion_osm_csv']}")
        except Exception as e:
            print(f"Erreur fusion finale OSM: {e}")
            return False
    else:
        print("Etape 5 ignoree (desactivee dans les options).")

    print("\nPipeline OSM termine avec succes.")
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline de labelisation OSM")
    parser.add_argument("--traj", required=True, help="ID du trajet a traiter")
    parser.add_argument("--options-json", type=str, default=None, help="JSON d'options pipeline OSM")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    options: dict = {}
    if args.options_json:
        try:
            options = json.loads(args.options_json)
            if not isinstance(options, dict):
                raise ValueError("--options-json doit etre un objet JSON")
        except json.JSONDecodeError as e:
            print(f"JSON invalide pour --options-json: {e}")
            return 1
        except Exception as e:
            print(f"Erreur options OSM: {e}")
            return 1

    params_json = options.get("label_params_json")
    params = _load_params_from_json(params_json) if params_json else options.get("params_labelisation", {})

    ok = run_osm_pipeline(
        traj_id=args.traj,
        osm_params=params,
        run_step_1_download_pbf=bool(options.get("run_step_1_download_pbf", True)),
        run_step_2_filter_pbf=bool(options.get("run_step_2_filter_pbf", True)),
        run_step_3_extract_features=bool(options.get("run_step_3_extract_features", True)),
        run_step_4_labelisation=bool(options.get("run_step_4_labelisation", True)),
        run_step_5_final_fusion=bool(options.get("run_step_5_final_fusion", True)),
        buffer_m=float(options.get("buffer_m", 500.0)),
        radius_m=float(options.get("radius_m", 30.0)),
        verbose=bool(options.get("verbose", True)),
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
