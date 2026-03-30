"""Evaluate and compare the latest GRU and XGBoost models for a dataset."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ADJUST_HUE_FUSED"] = "0"

import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

from utils import MODELS_DIR, TRAINING_DIR, get_dataset_path


OVERFITTING_THRESHOLD = 0.10


def _to_jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
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
        "Aucun dataset detecte dans DATA/03_TRAINING. Fournis --dataset-name apres preprocessing."
    )


def _safe_np_key(data: np.lib.npyio.NpzFile, candidates: list[str]) -> np.ndarray:
    for key in candidates:
        if key in data.files:
            return data[key]
    raise KeyError(f"Cle introuvable dans npz. Candidats: {candidates}. Disponibles: {data.files}")


def _maybe_rescale_flat(x_flat: np.ndarray, scaler, force_rescale: bool) -> np.ndarray:
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in is not None and x_flat.shape[1] != int(n_in):
        # X_flat (sequence flattened) can have seq_len * n_features columns,
        # while scaler was fit on base tabular features only.
        return x_flat

    if force_rescale:
        return scaler.transform(x_flat)

    # Heuristic: if already roughly centered/scaled, keep as is.
    finite = np.isfinite(x_flat)
    if not np.any(finite):
        return x_flat

    mean_abs = float(np.nanmean(np.abs(np.nanmean(x_flat, axis=0))))
    std_mean = float(np.nanmean(np.nanstd(x_flat, axis=0)))
    already_scaled = (mean_abs < 0.25) and (0.5 < std_mean < 1.5)
    return x_flat if already_scaled else scaler.transform(x_flat)


def _select_xgb_inputs(
    xgb_model: XGBClassifier,
    x_train: np.ndarray,
    x_test: np.ndarray,
    x_train_flat: np.ndarray,
    x_test_flat: np.ndarray,
    scaler,
    force_rescale_flat: bool,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    """Select matching XGBoost inputs from npz based on expected feature count."""
    expected = getattr(xgb_model, "n_features_in_", None)
    if expected is None:
        expected = xgb_model.get_booster().num_features()
    expected = int(expected)

    if x_train.shape[1] == expected:
        return x_train, x_test, "X_train/X_test", expected

    if x_train_flat.shape[1] == expected:
        x_train_eval = _maybe_rescale_flat(x_train_flat, scaler, force_rescale=force_rescale_flat)
        x_test_eval = _maybe_rescale_flat(x_test_flat, scaler, force_rescale=force_rescale_flat)
        return x_train_eval, x_test_eval, "X_train_flat/X_test_flat", expected

    raise ValueError(
        "Feature shape mismatch for XGBoost model. "
        f"Model expects {expected} features, available: "
        f"X_train={x_train.shape[1]}, X_train_flat={x_train_flat.shape[1]}."
    )


def _find_latest_model_for_dataset(model_kind: str, dataset_name: str) -> tuple[Path, dict]:
    """Find the latest model metadata under MODELS/<kind> for a dataset."""
    root = MODELS_DIR / model_kind
    if not root.exists():
        raise FileNotFoundError(f"Dossier modele introuvable: {root}")

    metadata_files = sorted(root.rglob("*_metadonnees.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    for meta_path in metadata_files:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        if meta.get("dataset_id") == dataset_name:
            return meta_path, meta

    raise FileNotFoundError(
        f"Aucun modele {model_kind} trouve pour dataset_id='{dataset_name}' sous {root}."
    )


def _load_gru_model_from_metadata(meta_path: Path, meta: dict):
    model_path = Path(meta.get("artefacts", {}).get("model", ""))
    if not model_path.exists():
        model_path = meta_path.parent / (meta.get("model_id", "") + ".keras")
    if not model_path.exists():
        raise FileNotFoundError(f"Fichier modele GRU introuvable: {model_path}")
    return keras.models.load_model(model_path), model_path


def _load_xgb_model_from_metadata(meta_path: Path, meta: dict):
    model_path = Path(meta.get("artefacts", {}).get("model", ""))
    if not model_path.exists():
        model_path = meta_path.parent / (meta.get("model_id", "") + ".json")
    if not model_path.exists():
        raise FileNotFoundError(f"Fichier modele XGBoost introuvable: {model_path}")
    model = XGBClassifier()
    model.load_model(model_path)
    return model, model_path


def compute_metrics(y_true, y_pred, y_proba, num_classes, split_label):
    y_one_hot = np.zeros((len(y_true), num_classes))
    y_one_hot[np.arange(len(y_true)), y_true] = 1

    try:
        auc = roc_auc_score(y_one_hot, y_proba, average="weighted", multi_class="ovr")
    except Exception:
        auc = None

    return {
        "split": split_label,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "cross_entropy": float(log_loss(y_one_hot, y_proba)),
        "auc": None if auc is None else float(auc),
        "y_one_hot": y_one_hot,
    }


def print_metrics(metrics: dict, model_name: str) -> None:
    width = 80
    print(f"\n{'=' * width}")
    print(f" {model_name}  -  {metrics['split']} set")
    print(f"{'=' * width}")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy  : {metrics['balanced_accuracy']:.4f}")
    print(f"  F1 (weighted)      : {metrics['f1_weighted']:.4f}")
    print(f"  F1 (macro)         : {metrics['f1_macro']:.4f}")
    print(f"  Cross-Entropy      : {metrics['cross_entropy']:.4f}")
    if metrics["auc"] is not None:
        print(f"  AUC (weighted)     : {metrics['auc']:.4f}")


def print_overfitting_analysis(train_m, test_m, model_name, threshold):
    print(f"\n--- Overfitting analysis - {model_name} ---")
    delta_acc = train_m["accuracy"] - test_m["accuracy"]
    delta_f1 = train_m["f1_weighted"] - test_m["f1_weighted"]
    print(f"  Delta Accuracy      (train - test) : {delta_acc:+.4f}")
    print(f"  Delta F1 (weighted) (train - test) : {delta_f1:+.4f}")
    flag = delta_acc > threshold or delta_f1 > threshold
    print(f"  -> {'OVERFITTING DETECTED' if flag else 'No significant overfitting'} (threshold={threshold})")


def _distribution_from_encoded(y_encoded: np.ndarray, class_names: np.ndarray) -> list[tuple[int, str, int, float]]:
    total = int(len(y_encoded))
    if total == 0:
        return []

    counts = np.bincount(y_encoded.astype(int), minlength=len(class_names))
    rows = []
    for idx, c in enumerate(counts):
        ratio = float(c / total) if total > 0 else 0.0
        rows.append((idx, str(class_names[idx]), int(c), ratio))
    return rows


def print_preprocessed_label_stats(
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: np.ndarray,
) -> None:
    print("\n" + "=" * 80)
    print(" LABEL STATS - PREPROCESSED DATASET (encoded labels)")
    print("=" * 80)

    y_all = np.concatenate([y_train, y_test], axis=0)

    for split_name, y_split in (("GLOBAL", y_all), ("TRAIN", y_train), ("TEST", y_test)):
        rows = _distribution_from_encoded(y_split, class_names)
        present = [name for _, name, c, _ in rows if c > 0]
        missing = [name for _, name, c, _ in rows if c == 0]

        print(f"\n[{split_name}] n={len(y_split):,} | classes_present={len(present)}/{len(class_names)}")
        for idx, name, count, ratio in rows:
            print(f"  - {idx:>2} | {name:<24} count={count:>7}  ratio={ratio:>7.2%}")

        if missing:
            print(f"  Missing in {split_name}: {missing}")

    missing_in_test = sorted(
        set(np.unique(y_train).astype(int)) - set(np.unique(y_test).astype(int))
    )
    if missing_in_test:
        missing_names = [str(class_names[i]) for i in missing_in_test]
        print(
            "\nInfo: classes present in TRAIN but absent in TEST "
            f"(normal with grouped/stratified split): {missing_names}"
        )


def plot_confusion_matrices(y_test, y_pred_gru, y_pred_xgb, class_names, acc_gru, acc_xgb):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels = np.arange(len(class_names))

    for ax, y_pred, name, acc, cmap in zip(
        axes,
        [y_pred_gru, y_pred_xgb],
        ["GRU", "XGBoost"],
        [acc_gru, acc_xgb],
        ["Blues", "Greens"],
    ):
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap=cmap, xticks_rotation="vertical")
        ax.set_title(f"Confusion Matrix - {name}\nAccuracy: {acc:.4f}", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_roc_curves(y_test_one_hot, gru_proba, xgb_proba, class_names, auc_gru, auc_xgb):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, proba, name, auc_val in zip(
        axes,
        [gru_proba, xgb_proba],
        ["GRU", "XGBoost"],
        [auc_gru, auc_xgb],
    ):
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_test_one_hot[:, i], proba[:, i])
            auc_cls = roc_auc_score(y_test_one_hot[:, i], proba[:, i])
            ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_cls:.3f})", linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)", linewidth=1.5)
        title = f"ROC - {name}"
        if auc_val is not None:
            title += f"\nAUC (weighted): {auc_val:.4f}"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def print_summary_table(gru_test_m: dict, xgb_test_m: dict) -> None:
    rows = [
        ("Accuracy", f"{gru_test_m['accuracy']:.4f}", f"{xgb_test_m['accuracy']:.4f}"),
        ("Balanced Acc", f"{gru_test_m['balanced_accuracy']:.4f}", f"{xgb_test_m['balanced_accuracy']:.4f}"),
        ("F1 (weighted)", f"{gru_test_m['f1_weighted']:.4f}", f"{xgb_test_m['f1_weighted']:.4f}"),
        ("F1 (macro)", f"{gru_test_m['f1_macro']:.4f}", f"{xgb_test_m['f1_macro']:.4f}"),
        ("Cross-Entropy", f"{gru_test_m['cross_entropy']:.4f}", f"{xgb_test_m['cross_entropy']:.4f}"),
        ("AUC (weighted)", f"{gru_test_m['auc']:.4f}" if gru_test_m["auc"] else "N/A", f"{xgb_test_m['auc']:.4f}" if xgb_test_m["auc"] else "N/A"),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "GRU", "XGBoost"])
    print("\n" + "=" * 60)
    print("COMPARISON TABLE (Test Set)")
    print("=" * 60)
    print(df.to_string(index=False))

    best_acc = "GRU" if gru_test_m["accuracy"] > xgb_test_m["accuracy"] else "XGBoost"
    best_f1 = "GRU" if gru_test_m["f1_weighted"] > xgb_test_m["f1_weighted"] else "XGBoost"
    print(f"\n  Best model (Accuracy) : {best_acc}")
    print(f"  Best model (F1)       : {best_f1}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latest GRU/XGBoost models for a dataset.")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--show-plots", action="store_true", default=False)
    parser.add_argument(
        "--force-rescale-flat",
        action="store_true",
        default=False,
        help="Force scaler.transform on flat arrays before XGBoost inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = _resolve_dataset_name(args.dataset_name)

    print("[1/5] Loading dataset artefacts ...")
    dataset_cfg = get_dataset_path(dataset_name)

    required = [
        dataset_cfg["preprocessed_data"],
        dataset_cfg["metadata"],
        dataset_cfg["label_encoder_path"],
        dataset_cfg["scaler_param"],
    ]
    for p in required:
        if not Path(p).exists():
            raise FileNotFoundError(f"Artefact dataset manquant: {p}")

    with open(dataset_cfg["metadata"], "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)

    data = np.load(dataset_cfg["preprocessed_data"], allow_pickle=True)
    label_encoder = joblib.load(dataset_cfg["label_encoder_path"])
    scaler = joblib.load(dataset_cfg["scaler_param"])

    x_train = _safe_np_key(data, ["X_train"])
    x_test = _safe_np_key(data, ["X_test"])
    x_train_flat = _safe_np_key(data, ["X_train_flat", "X_train"])
    x_test_flat = _safe_np_key(data, ["X_test_flat", "X_test"])
    x_tensor_train = _safe_np_key(data, ["X_tensor_train"])
    x_tensor_test = _safe_np_key(data, ["X_tensor_test"])
    y_train = _safe_np_key(data, ["y_train"])
    y_test = _safe_np_key(data, ["y_test"])

    print_preprocessed_label_stats(y_train=y_train, y_test=y_test, class_names=label_encoder.classes_)

    _ = x_train  # kept for explicit alignment checks
    _ = x_test

    print("[2/5] Resolving latest model versions for dataset ...")
    gru_meta_path, gru_meta = _find_latest_model_for_dataset("GRU", dataset_name)
    xgb_meta_path, xgb_meta = _find_latest_model_for_dataset("XGBOOST", dataset_name)

    gru_model, gru_model_path = _load_gru_model_from_metadata(gru_meta_path, gru_meta)
    xgb_model, xgb_model_path = _load_xgb_model_from_metadata(xgb_meta_path, xgb_meta)

    xgb_train_eval, xgb_test_eval, xgb_input_key, xgb_expected_features = _select_xgb_inputs(
        xgb_model=xgb_model,
        x_train=x_train,
        x_test=x_test,
        x_train_flat=x_train_flat,
        x_test_flat=x_test_flat,
        scaler=scaler,
        force_rescale_flat=args.force_rescale_flat,
    )

    print(f"  GRU model     : {gru_model_path}")
    print(f"  XGBoost model : {xgb_model_path}")
    print(f"  XGBoost input : {xgb_input_key} ({xgb_expected_features} features)")

    print("[3/5] Running inference ...")
    x_tensor_train_t = torch.from_numpy(x_tensor_train)
    x_tensor_test_t = torch.from_numpy(x_tensor_test)

    gru_train_proba = gru_model.predict(x_tensor_train_t, verbose=0)
    gru_test_proba = gru_model.predict(x_tensor_test_t, verbose=0)
    gru_train_pred = gru_train_proba.argmax(axis=1)
    gru_test_pred = gru_test_proba.argmax(axis=1)

    xgb_train_proba = xgb_model.predict_proba(xgb_train_eval)
    xgb_test_proba = xgb_model.predict_proba(xgb_test_eval)
    xgb_train_pred = xgb_model.predict(xgb_train_eval)
    xgb_test_pred = xgb_model.predict(xgb_test_eval)

    print("[4/5] Computing metrics and reports ...")
    num_classes = len(label_encoder.classes_)

    gru_train_m = compute_metrics(y_train, gru_train_pred, gru_train_proba, num_classes, "Train")
    gru_test_m = compute_metrics(y_test, gru_test_pred, gru_test_proba, num_classes, "Test")
    xgb_train_m = compute_metrics(y_train, xgb_train_pred, xgb_train_proba, num_classes, "Train")
    xgb_test_m = compute_metrics(y_test, xgb_test_pred, xgb_test_proba, num_classes, "Test")

    print_metrics(gru_train_m, "GRU")
    print_metrics(gru_test_m, "GRU")
    print_overfitting_analysis(gru_train_m, gru_test_m, "GRU", OVERFITTING_THRESHOLD)

    print("\n" + "-" * 80)

    print_metrics(xgb_train_m, "XGBoost")
    print_metrics(xgb_test_m, "XGBoost")
    print_overfitting_analysis(xgb_train_m, xgb_test_m, "XGBoost", OVERFITTING_THRESHOLD)

    print("\n--- GRU - Classification Report (Test) ---")
    gru_report_text = classification_report(y_test, gru_test_pred, target_names=label_encoder.classes_)
    print(gru_report_text)

    print("\n--- XGBoost - Classification Report (Test) ---")
    xgb_report_text = classification_report(y_test, xgb_test_pred, target_names=label_encoder.classes_)
    print(xgb_report_text)

    print_summary_table(gru_test_m, xgb_test_m)

    print("[5/5] Saving comparison report and plots ...")
    eval_dir = Path(dataset_cfg["output_dir"]) / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = eval_dir / f"comparison_report_{dataset_name}_{ts}.json"
    latest_path = eval_dir / "comparison_report.json"

    fig_cm = plot_confusion_matrices(
        y_test,
        gru_test_pred,
        xgb_test_pred,
        label_encoder.classes_,
        gru_test_m["accuracy"],
        xgb_test_m["accuracy"],
    )
    cm_plot_path = eval_dir / f"comparison_confusion_{ts}.png"
    fig_cm.savefig(cm_plot_path, dpi=150)
    if args.show_plots:
        plt.show()
    plt.close(fig_cm)

    fig_roc = plot_roc_curves(
        gru_test_m["y_one_hot"],
        gru_test_proba,
        xgb_test_proba,
        label_encoder.classes_,
        gru_test_m["auc"],
        xgb_test_m["auc"],
    )
    roc_plot_path = eval_dir / f"comparison_roc_{ts}.png"
    fig_roc.savefig(roc_plot_path, dpi=150)
    if args.show_plots:
        plt.show()
    plt.close(fig_roc)

    comparison_report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": dataset_name,
        "dataset_metadata": str(dataset_cfg["metadata"]),
        "features": dataset_meta.get("features", []),
        "window_size": dataset_meta.get("preprocessing", {}).get("window_size"),
        "models": {
            "gru": {
                "metadata_path": str(gru_meta_path),
                "model_path": str(gru_model_path),
                "train": _to_jsonable({k: v for k, v in gru_train_m.items() if k != "y_one_hot"}),
                "test": _to_jsonable({k: v for k, v in gru_test_m.items() if k != "y_one_hot"}),
                "classification_report_test": classification_report(
                    y_test,
                    gru_test_pred,
                    target_names=label_encoder.classes_,
                    output_dict=True,
                ),
            },
            "xgboost": {
                "metadata_path": str(xgb_meta_path),
                "model_path": str(xgb_model_path),
                "train": _to_jsonable({k: v for k, v in xgb_train_m.items() if k != "y_one_hot"}),
                "test": _to_jsonable({k: v for k, v in xgb_test_m.items() if k != "y_one_hot"}),
                "classification_report_test": classification_report(
                    y_test,
                    xgb_test_pred,
                    target_names=label_encoder.classes_,
                    output_dict=True,
                ),
            },
        },
        "artifacts": {
            "comparison_report": str(report_path),
            "comparison_report_latest": str(latest_path),
            "confusion_plot": str(cm_plot_path),
            "roc_plot": str(roc_plot_path),
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(comparison_report), f, indent=4)

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(comparison_report), f, indent=4)

    print(f"Saved: {report_path}")
    print(f"Saved: {latest_path}")


if __name__ == "__main__":
    main()
