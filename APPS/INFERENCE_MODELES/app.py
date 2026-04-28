import argparse
import json
import sys

from INFERENCE_MODELES.inference import run_inference_for_trajet


def get_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Inference modeles sur trajet complet")
	parser.add_argument("--traj", required=True, help="ID du trajet a inferer")
	parser.add_argument("--dataset-name", required=True, help="Nom du dataset associe aux modeles")
	parser.add_argument(
		"--models",
		nargs="+",
		default=["GRU", "XGBOOST"],
		help="Modeles a utiliser (GRU, XGBOOST, CNN_1D). Exemple: --models GRU CNN_1D",
	)
	return parser


def main() -> int:
	args = get_parser().parse_args()
	try:
		out = run_inference_for_trajet(
			traj_id=args.traj,
			dataset_name=args.dataset_name,
			model_kinds=args.models,
		)
	except Exception as e:
		print(f"Erreur inference: {e}")
		return 1

	print("Inference terminee avec succes.")
	print(json.dumps(out, indent=2, ensure_ascii=True))
	return 0


if __name__ == "__main__":
	sys.exit(main())
