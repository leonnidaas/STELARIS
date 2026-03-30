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
    parser.add_argument("--step", choices=["all", "prepro", "train_xgb", "train_gru", "eval"], default="all")
    parser.add_argument("--dataset-name", type=str, default=None, help="Nom du dataset (si déjà existant)")
    parser.add_argument("--trajets", nargs="+", default=["BORDEAUX_COUTRAS", "MARTINE_01"])
    parser.add_argument("--window-size", type=int, default=15)
    
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

    # --- ÉTAPE 2 : ENTRAÎNEMENT XGBOOST ---
    if args.step in ["all", "train_xgb"]:
        xgb_args = ["--dataset-name", ds_name]
        run_script("train_xgboost.py", xgb_args)

    # --- ÉTAPE 3 : ENTRAÎNEMENT GRU ---
    if args.step in ["all", "train_gru"]:
        gru_args = ["--dataset-name", ds_name, "--epochs", "50"]
        run_script("train_gru.py", gru_args)

    # --- ÉTAPE 4 : ÉVALUATION ---
    if args.step in ["all", "eval"]:
        # Note : evaluate.py doit être mis à jour pour accepter --dataset-name
        eval_args = ["--dataset-name", ds_name]
        run_script("evaluate.py", eval_args)

    print(f"\nPipeline STELARIS terminé pour le dataset : {ds_name}")

if __name__ == "__main__":
    main()