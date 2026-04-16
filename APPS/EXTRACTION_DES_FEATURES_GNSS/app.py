import argparse
import json
import sys

from utils import get_traj_paths
from EXTRACTION_DES_FEATURES_GNSS.extraction_features_gnss import process_gnss_feature_extraction


def pipeline_extraction_features_gnss(
	traj_id: str,
	cn0_smooth_window: int = 15,
	cn0_quartile: int = 1,
	verbose: bool = True,
) -> bool:
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
			cn0_smooth_window=cn0_smooth_window,
			cn0_quartile=cn0_quartile,
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
	parser.add_argument("--options-json", type=str, default=None, help="JSON d'options pipeline GNSS")
	return parser


if __name__ == "__main__":
	args = get_parser().parse_args()
	options = {}
	if args.options_json:
		try:
			options = json.loads(args.options_json)
			if not isinstance(options, dict):
				raise ValueError("--options-json doit representer un objet JSON")
		except json.JSONDecodeError as err:
			print(f"JSON invalide pour --options-json: {err}")
			sys.exit(1)
		except Exception as err:
			print(f"Erreur options GNSS: {err}")
			sys.exit(1)

	success = pipeline_extraction_features_gnss(
		traj_id=args.traj,
		cn0_smooth_window=int(options.get("cn0_smooth_window", 15)),
		cn0_quartile=int(options.get("cn0_quartile", 1)),
		verbose=bool(options.get("verbose", True)),
	)

	if success:
		print("\nPipeline termine avec succes.")
		sys.exit(0)
	else:
		print("\nLe pipeline a rencontre des erreurs.")
		sys.exit(1)
