import argparse
import sys

import pandas as pd

from FUSION.sync import process_gnss_gt_fusion
from utils import get_traj_paths, standardize_dataframe


def run_gt_gnss_fusion(traj_id: str, verbose: bool = True) -> bool:
    """Run GT+GNSS fusion independently for one trajectory."""
    cfg = get_traj_paths(traj_id)

    if verbose:
        print(f"--- Fusion GT+GNSS pour : {traj_id} ---")

    try:
        df_gt = pd.read_csv(cfg["raw_gt"])
        df_gnss = pd.read_csv(cfg["raw_gnss"])
    except FileNotFoundError as e:
        print(f"Fichier manquant: {e}")
        print("Assurez-vous d'avoir lance le traitement RINEX avant la fusion GT+GNSS.")
        return False
    except Exception as e:
        print(f"Erreur lecture des fichiers d'entree: {e}")
        return False

    try:
        process_gnss_gt_fusion(
            df_gt=standardize_dataframe(df_gt),
            df_gnss=df_gnss,
            output_csv=cfg["sync_csv"],
            gnss_offset=cfg.get("gnss_offset"),
            verbose=verbose,
        )
    except Exception as e:
        print(f"Erreur lors de la fusion GT+GNSS: {e}")
        return False

    print(f"Fusion GT+GNSS terminee: {cfg['sync_csv']}")
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fusion GT+GNSS")
    parser.add_argument("--traj", required=True, help="ID du trajet a traiter")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    ok = run_gt_gnss_fusion(traj_id=args.traj, verbose=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
