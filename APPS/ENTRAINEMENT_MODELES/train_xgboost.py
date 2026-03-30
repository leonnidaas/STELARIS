"""XGBoost training aligned with dataset/model registry utilities."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, plot_importance

from utils import TRAINING_DIR, get_dataset_path, get_model_path


N_TRIALS = 25
N_CV_SPLITS = 5
OPTUNA_RANDOM = 42


def _to_jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


def _resolve_dataset_name(dataset_name: str | None) -> str:
    if dataset_name:
        return dataset_name

    candidates = sorted(
        [p for p in TRAINING_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        npz = p / f"{p.name}_preprocessed_data.npz"
        meta = p / f"{p.name}_metadata.json"
        if npz.exists() and meta.exists():
            return p.name

    raise FileNotFoundError(
        "Aucun dataset d'entrainement detecte. Fournis --dataset-name apres preprocessing."
    )


def load_dataset_bundle(dataset_name: str) -> dict:
    cfg = get_dataset_path(dataset_name)

    required = [
        cfg["preprocessed_data"],
        cfg["metadata"],
        cfg["label_encoder_path"],
        cfg["scaler_param"],
    ]
    for p in required:
        if not Path(p).exists():
            raise FileNotFoundError(f"Artefact manquant: {p}")

    with open(cfg["metadata"], "r", encoding="utf-8") as f:
        dataset_metadata = json.load(f)

    data = np.load(cfg["preprocessed_data"], allow_pickle=True)
    label_encoder = joblib.load(cfg["label_encoder_path"])
    scaler = joblib.load(cfg["scaler_param"])

    return {
        "cfg": cfg,
        "dataset_metadata": dataset_metadata,
        "data": data,
        "label_encoder": label_encoder,
        "scaler": scaler,
    }


def build_objective(x_train, y_train, id_train, n_splits, random_state):
    """Return an Optuna objective that performs inner StratifiedGroupKFold CV."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multi:softprob",
            "eval_metric": "logloss",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []

        for tr_idx, val_idx in gkf.split(x_train, y_train, groups=id_train):
            model = XGBClassifier(**params)
            sw = compute_sample_weight(class_weight="balanced", y=y_train[tr_idx])
            model.fit(x_train[tr_idx], y_train[tr_idx], sample_weight=sw)
            y_pred = model.predict(x_train[val_idx])
            scores.append(balanced_accuracy_score(y_train[val_idx], y_pred))

        return float(np.mean(scores))

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost from preprocessed dataset pack.")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--cv-splits", type=int, default=N_CV_SPLITS)
    parser.add_argument("--optuna-seed", type=int, default=OPTUNA_RANDOM)
    parser.add_argument("--show-plots", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = _resolve_dataset_name(args.dataset_name)

    print("[1/6] Loading dataset artefacts ...")
    bundle = load_dataset_bundle(dataset_name)
    data = bundle["data"]
    dataset_meta = bundle["dataset_metadata"]
    le = bundle["label_encoder"]

    x_train = data["X_train"]
    x_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    id_train = data["id_train"]
    id_test = data["id_test"]

    overlap = set(id_train.astype(str)).intersection(set(id_test.astype(str)))
    if overlap:
        raise ValueError(f"Split invalide: {len(overlap)} segment_id presents in both train and test.")

    feature_names = dataset_meta.get("features", [])
    print(f"  Dataset: {dataset_name}")
    print(f"  Train: {x_train.shape} | Test: {x_test.shape}")
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Features count: {len(feature_names)}")

    print(f"\n[2/6] Optuna search ({args.n_trials} trials) ...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        build_objective(x_train, y_train, id_train, args.cv_splits, args.optuna_seed),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print(f"  Best balanced accuracy (CV): {study.best_trial.value:.4f}")
    print(f"  Best parameters: {study.best_trial.params}")

    print("\n[3/6] Training final model ...")
    best_params = {
        **study.best_params,
        "objective": "multi:softprob",
        "eval_metric": "logloss",
    }

    final_model = XGBClassifier(**best_params)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    final_model.fit(x_train, y_train, sample_weight=sample_weights)

    print("\n[4/6] Evaluating on test split ...")
    y_pred = final_model.predict(x_test)
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    print(f"  Balanced accuracy (test): {bal_acc:.4f}")
    print(f"  F1 weighted (test):      {f1_weighted:.4f}")
    print(f"  Prediction distribution: {Counter(y_pred)}")

    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=le.classes_)
    cm = confusion_matrix(y_test, y_pred)

    print("\n[5/6] Saving model artefacts and registry ...")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_paths = get_model_path("XGBOOST", timestamp)
    model_dir = Path(model_paths["metadata"]).parent

    model_file = model_dir / f"{model_paths['id']}.json"
    final_model.save_model(model_file)

    features_path = Path(model_paths["features"])
    pd_rows = ["feature_name\n"] + [f"{f}\n" for f in feature_names]
    with open(features_path, "w", encoding="utf-8") as f:
        f.writelines(pd_rows)

    scaler_copy_path = Path(model_paths["scaler"])
    joblib.dump(bundle["scaler"], scaler_copy_path)

    cm_img_path = model_dir / "confusion_matrix_test.png"
    cm_json_path = model_dir / "confusion_matrix_test.json"
    report_json_path = model_dir / "classification_report_test.json"
    report_txt_path = model_dir / "classification_report_test.txt"

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, xticks_rotation="vertical")
    plt.title(f"XGBoost Test Confusion Matrix - bal_acc={bal_acc:.4f}")
    plt.tight_layout()
    plt.savefig(cm_img_path, dpi=150)
    if args.show_plots:
        plt.show()
    plt.close(fig)

    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump({"labels": [str(c) for c in le.classes_], "matrix": cm.tolist()}, f, indent=4)

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(report_dict), f, indent=4)
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    if feature_names:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        final_model.get_booster().feature_names = feature_names
        plot_importance(
            final_model,
            ax=axs[0],
            max_num_features=min(25, len(feature_names)),
            importance_type="gain",
            values_format="{v:.2f}",
            title="Gain",
        )
        plot_importance(
            final_model,
            ax=axs[1],
            max_num_features=min(25, len(feature_names)),
            importance_type="weight",
            values_format="{v:.2f}",
            title="Weight",
        )
        fig.tight_layout()
        fig.savefig(model_dir / "feature_importance.png", dpi=150)
        if args.show_plots:
            plt.show()
        plt.close(fig)

    model_metadata = {
        "model_id": model_paths["id"],
        "model_type": "xgboost",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": dataset_name,
        "dataset_metadata": str(bundle["cfg"]["metadata"]),
        "features": feature_names,
        "window_size": dataset_meta.get("preprocessing", {}).get("window_size"),
        "hyperparameters": _to_jsonable(best_params),
        "optuna": {
            "n_trials": int(args.n_trials),
            "cv_splits": int(args.cv_splits),
            "best_cv_balanced_accuracy": float(study.best_trial.value),
        },
        "metrics_test": {
            "balanced_accuracy": bal_acc,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
        },
        "split_check": {
            "n_train_groups": int(len(np.unique(id_train))),
            "n_test_groups": int(len(np.unique(id_test))),
            "intersection_train_test_groups": int(len(overlap)),
        },
        "artefacts": {
            "model": str(model_file),
            "metadata": str(model_paths["metadata"]),
            "results": str(model_paths["results"]),
            "features_csv": str(features_path),
            "scaler_pkl": str(scaler_copy_path),
            "classification_report_json": str(report_json_path),
            "classification_report_txt": str(report_txt_path),
            "confusion_matrix_json": str(cm_json_path),
            "confusion_matrix_png": str(cm_img_path),
        },
    }

    with open(model_paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=4)

    with open(model_paths["results"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics_test": model_metadata["metrics_test"],
                "prediction_distribution": _to_jsonable(dict(Counter(y_pred))),
            },
            f,
            indent=4,
        )

    print(f"\n[6/6] Done. Model registry folder: {model_dir}")


if __name__ == "__main__":
    main()
