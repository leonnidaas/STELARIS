import argparse
import sys

from utils import get_traj_paths
from EXTRACTION_DES_FEATURES_GNSS.extraction_features_gnss import process_gnss_feature_extraction


def pipeline_extraction_features_gnss(traj_id: str, verbose: bool = True) -> bool:
	"""Pipeline de creation des features GNSS a partir des sorties RINEX."""
	if verbose:
		print(f"--- Extraction des features GNSS pour : {traj_id} ---")

	config = get_traj_paths(traj_id)

	print("=" * 50)
	print("ETAPE 1 : Lecture des donnees GNSS et WLS")
	print("=" * 50)

	try:
		process_gnss_feature_extraction(
			path_svstates=config["space_vehicule_info"],
			path_wlssolution=config["raw_gnss"],
			output_csv=config["gnss_features_csv"],
			path_gt=config["raw_gt"],
			verbose=verbose,
		)
	except FileNotFoundError as err:
		print(f"Fichier manquant: {err}")
		print("Assurez-vous d'avoir execute le pipeline TRAITEMENT_RINEX en amont.")
		return False
	except Exception as err:
		print(f"Erreur lors de l'extraction des features GNSS: {err}")
		return False

	print("\n" + "=" * 50)
	print("ETAPE 2 : Export du fichier final")
	print("=" * 50)
	print(f"Features GNSS exportees vers: {config['gnss_features_csv']}")

	return True


def get_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Extraction des features GNSS")
	parser.add_argument("--traj", required=True, help="ID du trajet a traiter")
	return parser


if __name__ == "__main__":
	args = get_parser().parse_args()
	success = pipeline_extraction_features_gnss(traj_id=args.traj, verbose=True)

	if success:
		print("\nPipeline termine avec succes.")
		sys.exit(0)
	else:
		print("\nLe pipeline a rencontre des erreurs.")
		sys.exit(1)
