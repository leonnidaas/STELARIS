"""
STELARIS Orchestrator - app.py
Permet d'exécuter le pipeline complet : Prepro -> Train -> Eval
"""

import argparse
import subprocess
import sys
from datetime import datetime

def run_script(script_name, args_list):
    """Exécute un script python avec ses arguments."""
    cmd = [sys.executable, script_name] + args_list
    print(f"\nLancement de {script_name}...")
    print(f"Commande : {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Affichage du flux en temps réel
    for line in process.stdout:
        print(line, end="")
    
    process.wait()
    if process.returncode != 0:
        print(f"\nErreur dans {script_name} (Code: {process.returncode})")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="STELARIS Pipeline Manager")
    parser.add_argument(
        "--step",
        choices=["all", "prepro", "optimize_gru", "train_xgb", "train_gru", "eval"],
        default="all",
    )
    parser.add_argument("--dataset-name", type=str, default=None, help="Nom du dataset (si déjà existant)")
    parser.add_argument("--trajets", nargs="+", default=["BORDEAUX_COUTRAS", "MARTINE_01"])
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--optuna-epochs", type=int, default=60)
    parser.add_argument("--optuna-batch-size", type=int, default=16)
    parser.add_argument("--optuna-patience", type=int, default=8)
    parser.add_argument("--optuna-timeout", type=int, default=1800)
    parser.add_argument("--optuna-pruner", choices=["hyperband", "median"], default="hyperband")
    parser.add_argument("--optuna-n-jobs", type=int, default=1)
    parser.add_argument("--optuna-train-subsample", type=float, default=0.7)
    parser.add_argument("--optuna-val-subsample", type=float, default=1.0)
    parser.add_argument("--optuna-min-delta", type=float, default=1e-4)
    parser.add_argument("--optuna-mixed-precision", action="store_true", default=True)
    parser.add_argument("--no-optuna-mixed-precision", action="store_false", dest="optuna_mixed_precision")
    
    args = parser.parse_args()
    
    # 1. Génération du nom de dataset si non fourni
    ds_name = args.dataset_name if args.dataset_name else datetime.now().strftime("%Y-%m-%d_%H-%M")

    # --- ÉTAPE 1 : PREPROCESSING ---
    if args.step in ["all", "prepro"]:
        prepro_args = [
            "--dataset-name", ds_name,
            "--window-size", str(args.window_size),
            "--trajets"
        ] + args.trajets
        run_script("preprocessing.py", prepro_args)

    # --- ÉTAPE 2 : OPTIMISATION GRU (OPTUNA) ---
    if args.step in ["all", "optimize_gru"]:
        optuna_args = [
            "--dataset-name", ds_name,
            "--n-trials", str(args.optuna_trials),
            "--epochs", str(args.optuna_epochs),
            "--batch-size", str(args.optuna_batch_size),
            "--patience", str(args.optuna_patience),
            "--timeout", str(args.optuna_timeout),
            "--pruner", str(args.optuna_pruner),
            "--n-jobs", str(args.optuna_n_jobs),
            "--train-subsample-ratio", str(args.optuna_train_subsample),
            "--val-subsample-ratio", str(args.optuna_val_subsample),
            "--min-delta", str(args.optuna_min_delta),
            "--mixed-precision" if args.optuna_mixed_precision else "--no-mixed-precision",
        ]
        run_script("optimize_gru.py", optuna_args)

    # --- ÉTAPE 3 : ENTRAÎNEMENT XGBOOST ---
    if args.step in ["all", "train_xgb"]:
        xgb_args = ["--dataset-name", ds_name]
        run_script("train_xgboost.py", xgb_args)

    # --- ÉTAPE 4 : ENTRAÎNEMENT GRU ---
    if args.step in ["all", "train_gru"]:
        gru_args = ["--dataset-name", ds_name, "--epochs", "50"]
        run_script("train_gru.py", gru_args)

    # --- ÉTAPE 5 : ÉVALUATION ---
    if args.step in ["all", "eval"]:
        # Note : evaluate.py doit être mis à jour pour accepter --dataset-name
        eval_args = ["--dataset-name", ds_name]
        run_script("evaluate.py", eval_args)

    print(f"\nPipeline STELARIS terminé pour le dataset : {ds_name}")

if __name__ == "__main__":
    main()