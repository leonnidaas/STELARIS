"""Bidirectional GRU training aligned with dataset/model registry utilities."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ADJUST_HUE_FUSED"] = "0"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch

import keras
from keras import layers
from keras.models import Sequential
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm.keras import TqdmCallback

from utils import PARAMS_ENTRAINEMENT, TRAINING_DIR, get_dataset_path, get_model_path


tf.config.optimizer.set_jit(False)


HIDDEN_UNITS = 32
DENSE_UNITS = 16
ACTIVATION = "tanh"
RECURRENT_ACTIVATION = "sigmoid"
DROPOUT = 0.0
KERNEL_REGULARIZER = None

BATCH_SIZE = 16
EPOCHS = 127
INITIAL_LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 256


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


def build_model(
    sequence_length: int,
    input_dim: int,
    output_dim: int,
    dense_units: int,
    hidden_units: int,
    activation: str,
    recurrent_activation: str,
    dropout: float,
    kernel_regularizer,
) -> keras.Model:
    """Build a bidirectional two-layer GRU classifier."""
    gru_kwargs = dict(
        activation=activation,
        recurrent_activation=recurrent_activation,
        kernel_regularizer=kernel_regularizer,
        dropout=dropout,
    )

    model = Sequential(
        [
            layers.Input(shape=(sequence_length, input_dim)),
            layers.Bidirectional(layers.GRU(hidden_units, return_sequences=True, **gru_kwargs)),
            layers.Bidirectional(layers.GRU(hidden_units, return_sequences=False, **gru_kwargs)),
            layers.Dense(dense_units, activation="relu"),
            layers.Dense(output_dim, activation="softmax"),
        ]
    )
    return model


def check_gpu() -> None:
    print(f"TensorFlow version : {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU detected      : {gpus}")
    else:
        print("No GPU detected - running on CPU.")


def to_torch_onehot(y: np.ndarray, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.from_numpy(y), num_classes=num_classes).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRU from preprocessed dataset pack.")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--show-plots", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_gpu()

    dataset_name = _resolve_dataset_name(args.dataset_name)
    dataset_cfg = get_dataset_path(dataset_name)

    required = [
        dataset_cfg["preprocessed_data"],
        dataset_cfg["metadata"],
        dataset_cfg["label_encoder_path"],
        dataset_cfg["scaler_param"],
    ]
    for p in required:
        if not Path(p).exists():
            raise FileNotFoundError(f"Artefact manquant: {p}")

    with open(dataset_cfg["metadata"], "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)

    data = np.load(dataset_cfg["preprocessed_data"], allow_pickle=True)
    label_encoder = joblib.load(dataset_cfg["label_encoder_path"])
    scaler = joblib.load(dataset_cfg["scaler_param"])

    print("\n[1/6] Loading preprocessed tensors ...")
    x_tensor_train = data["X_tensor_train"]
    x_tensor_test = data["X_tensor_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    train_train_idx = data["train_train_idx"]
    train_val_idx = data["train_val_idx"]
    list_classes_weights = data["list_classes_weights"]
    id_train = data["id_train"]
    id_test = data["id_test"]

    overlap = set(id_train.astype(str)).intersection(set(id_test.astype(str)))
    if overlap:
        raise ValueError(f"Split invalide: {len(overlap)} segment_id presents in both train and test.")

    sequence_length = x_tensor_train.shape[1]
    input_dim = x_tensor_train.shape[2]
    output_dim = len(label_encoder.classes_)

    metadata_window = dataset_meta.get("preprocessing", {}).get("window_size")
    if metadata_window is not None and int(metadata_window) != int(sequence_length):
        print(
            f"  Warning: metadata window_size={metadata_window} "
            f"differs from tensor sequence_length={sequence_length}."
        )

    print(f"  Dataset: {dataset_name}")
    print(f"  X_tensor_train: {x_tensor_train.shape}")
    print(f"  X_tensor_test : {x_tensor_test.shape}")
    print(f"  Classes       : {list(label_encoder.classes_)}")

    x_tensor_train_t = torch.from_numpy(x_tensor_train)
    x_tensor_test_t = torch.from_numpy(x_tensor_test)
    y_one_hot_train = to_torch_onehot(y_train, output_dim)

    print("\n[2/6] Building model ...")
    model = build_model(
        sequence_length=sequence_length,
        input_dim=input_dim,
        output_dim=output_dim,
        dense_units=DENSE_UNITS,
        hidden_units=HIDDEN_UNITS,
        activation=ACTIVATION,
        recurrent_activation=RECURRENT_ACTIVATION,
        dropout=DROPOUT,
        kernel_regularizer=KERNEL_REGULARIZER,
    )
    model.summary()

    print("\n[3/6] Compile and train ...")
    n_train_steps = int(np.ceil(len(train_train_idx) / args.batch_size))

    focal_loss = keras.losses.CategoricalFocalCrossentropy(
        alpha=list_classes_weights,
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
    )

    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        first_decay_steps=max(1, n_train_steps),
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.01,
        name="CosDecay",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=focal_loss,
        metrics=["accuracy"],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        x_tensor_train_t[train_train_idx],
        y_one_hot_train[train_train_idx],
        validation_data=(
            x_tensor_train_t[train_val_idx],
            y_one_hot_train[train_val_idx],
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=[early_stop, TqdmCallback(verbose=0)],
    )

    print("\n[4/6] Evaluating on test split ...")
    y_pred_proba = model.predict(x_tensor_test_t, verbose=0)
    y_pred = y_pred_proba.argmax(axis=1)

    acc = float(np.mean(y_pred == y_test))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    idx_best = int(np.argmin(history.history["val_loss"]))

    print(f"  Accuracy (unbalanced): {acc:.4f}")
    print(f"  Balanced accuracy    : {bal_acc:.4f}")
    print(f"  F1 weighted          : {f1_weighted:.4f}")

    report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    cm = confusion_matrix(y_test, y_pred)

    print("\n[5/6] Saving model artefacts and registry ...")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_paths = get_model_path("GRU", timestamp)
    model_dir = Path(model_paths["metadata"]).parent

    model_file = Path(model_paths["model_file"])
    model.save(model_file)

    features_path = Path(model_paths["features"])
    feature_names = dataset_meta.get("features", [])
    feature_rows = ["feature_name\n"] + [f"{f}\n" for f in feature_names]
    with open(features_path, "w", encoding="utf-8") as f:
        f.writelines(feature_rows)

    scaler_copy_path = Path(model_paths["scaler"])
    joblib.dump(scaler, scaler_copy_path)

    cm_img_path = model_dir / "confusion_matrix_test.png"
    cm_json_path = model_dir / "confusion_matrix_test.json"
    report_json_path = model_dir / "classification_report_test.json"
    report_txt_path = model_dir / "classification_report_test.txt"

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(ax=ax, xticks_rotation="vertical")
    plt.title(f"GRU Test Confusion Matrix - bal_acc={bal_acc:.4f}")
    plt.tight_layout()
    plt.savefig(cm_img_path, dpi=150)
    if args.show_plots:
        plt.show()
    plt.close(fig)

    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump({"labels": [str(c) for c in label_encoder.classes_], "matrix": cm.tolist()}, f, indent=4)

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(report_dict), f, indent=4)
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    if args.show_plots:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history.history["accuracy"], label="Train")
        axes[0].plot(history.history["val_accuracy"], label="Val")
        axes[0].axhline(bal_acc, ls="--", color="red", label=f"Test bal. acc={bal_acc:.3f}")
        axes[0].axvline(idx_best, ls="--", color="red")
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(history.history["loss"], label="Train")
        axes[1].plot(history.history["val_loss"], label="Val")
        axes[1].axvline(idx_best, ls="--", color="red")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(model_dir / "training_curves.png", dpi=150)
        plt.show()
        plt.close(fig)

    model_metadata = {
        "model_id": model_paths["id"],
        "model_type": "gru",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": dataset_name,
        "dataset_metadata": str(dataset_cfg["metadata"]),
        "features": feature_names,
        "window_size": dataset_meta.get("preprocessing", {}).get("window_size"),
        "hyperparameters": {
            "hidden_units": HIDDEN_UNITS,
            "dense_units": DENSE_UNITS,
            "activation": ACTIVATION,
            "recurrent_activation": RECURRENT_ACTIVATION,
            "dropout": DROPOUT,
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "initial_learning_rate": INITIAL_LEARNING_RATE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "loss": "CategoricalFocalCrossentropy",
            "optimizer": "Adam(CosineDecayRestarts)",
        },
        "metrics_test": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "best_epoch_val_loss": idx_best,
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
        json.dump(_to_jsonable(model_metadata), f, indent=4)

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
