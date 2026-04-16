"""Optuna hyperparameter optimization for GRU on precomputed spatial split."""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ADJUST_HUE_FUSED"] = "0"

import joblib
import keras
import numpy as np
import optuna
import tensorflow as tf
from keras import layers
from keras.models import Sequential

try:
    from optuna.integration import TFKerasPruningCallback
except Exception:
    try:
        from optuna_integration.tfkeras import TFKerasPruningCallback
    except Exception as e:
        raise ImportError(
            "TFKerasPruningCallback introuvable. Installez optuna-integration."
        ) from e

from utils import TRAINING_DIR, get_dataset_path


DEFAULT_N_TRIALS = 20
DEFAULT_EPOCHS = 60
DEFAULT_BATCH_SIZE = 16
DEFAULT_EARLY_STOPPING_PATIENCE = 8
DEFAULT_OPTUNA_SEED = 42
DEFAULT_PRUNING_WARMUP_EPOCHS = 3
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_TRAIN_SUBSAMPLE_RATIO = 0.7
DEFAULT_VAL_SUBSAMPLE_RATIO = 1.0
DEFAULT_N_JOBS = 1


tf.config.optimizer.set_jit(False)


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


def check_gpu() -> None:
    print(f"TensorFlow version : {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU detected      : {gpus}")
    else:
        print("No GPU detected - running on CPU.")


def build_model(
    sequence_length: int,
    input_dim: int,
    output_dim: int,
    hidden_units: int,
    dense_units: int,
    dropout: float,
    recurrent_dropout: float,
) -> keras.Model:
    gru_kwargs = dict(
        activation="tanh",
        recurrent_activation="sigmoid",
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
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


def build_objective(
    x_tensor_train: np.ndarray,
    y_one_hot_train: np.ndarray,
    train_train_idx: np.ndarray,
    train_val_idx: np.ndarray,
    class_weights: np.ndarray,
    sequence_length: int,
    input_dim: int,
    output_dim: int,
    epochs: int,
    batch_size: int,
    patience: int,
    min_delta: float,
):
    """Build an Optuna objective that trains on train_train_idx and validates on train_val_idx."""

    x_tr = x_tensor_train[train_train_idx]
    y_tr = y_one_hot_train[train_train_idx]
    x_val = x_tensor_train[train_val_idx]
    y_val = y_one_hot_train[train_val_idx]

    def objective(trial: optuna.Trial) -> float:
        keras.backend.clear_session()
        gc.collect()

        hidden_units = trial.suggest_int("hidden_units", 8, 64, step=8)
        dense_units = trial.suggest_int("dense_units", 8, 32)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        recurrent_dropout = trial.suggest_float("recurrent_dropout", 0.1, 0.5)
        initial_learning_rate = trial.suggest_float(
            "initial_learning_rate", 1e-4, 5e-3, log=True
        )

        model = build_model(
            sequence_length=sequence_length,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            dense_units=dense_units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )

        print(
            f"\n[trial {trial.number}] start | hidden_units={hidden_units} "
            f"dense_units={dense_units} dropout={dropout:.3f} "
            f"recurrent_dropout={recurrent_dropout:.3f} lr={initial_learning_rate:.6f}"
        )

        focal_loss = keras.losses.CategoricalFocalCrossentropy(
            alpha=class_weights,
            gamma=2.0,
            from_logits=False,
            label_smoothing=0.0,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
            loss=focal_loss,
            metrics=["accuracy"],
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
        )
        pruning = TFKerasPruningCallback(trial, monitor="val_loss")

        train_ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=0,
            callbacks=[early_stop, pruning],
        )

        val_losses = history.history.get("val_loss", [])
        if not val_losses:
            print(f"[trial {trial.number}] no val_loss captured -> return inf")
            return float("inf")

        best_val_loss = float(np.min(val_losses))
        trial.set_user_attr("best_epoch", int(np.argmin(val_losses)))
        print(
            f"[trial {trial.number}] done | best_val_loss={best_val_loss:.6f} "
            f"best_epoch={trial.user_attrs.get('best_epoch', -1)}"
        )
        return best_val_loss

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize GRU hyperparameters with Optuna.")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout in seconds, 0 for no limit.",
    )
    parser.add_argument("--optuna-seed", type=int, default=DEFAULT_OPTUNA_SEED)
    parser.add_argument("--pruning-warmup-epochs", type=int, default=DEFAULT_PRUNING_WARMUP_EPOCHS)
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Early stopping min_delta on val_loss.")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS, help="Optuna parallel jobs (threads).")
    parser.add_argument(
        "--pruner",
        choices=["median", "hyperband"],
        default="hyperband",
        help="Pruner strategy for early trial termination.",
    )
    parser.add_argument(
        "--train-subsample-ratio",
        type=float,
        default=DEFAULT_TRAIN_SUBSAMPLE_RATIO,
        help="Fraction of train_train_idx used per trial (0.1-1.0).",
    )
    parser.add_argument(
        "--val-subsample-ratio",
        type=float,
        default=DEFAULT_VAL_SUBSAMPLE_RATIO,
        help="Fraction of train_val_idx used per trial (0.1-1.0).",
    )
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mixed precision when GPU is available.",
    )
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (optional).")
    return parser.parse_args()


def _subsample_indices(indices: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    ratio = float(np.clip(ratio, 0.1, 1.0))
    indices = np.asarray(indices, dtype=np.int64)
    if ratio >= 0.999:
        return indices

    keep_n = max(1, int(round(len(indices) * ratio)))
    rng = np.random.default_rng(seed)
    sampled = rng.choice(indices, size=keep_n, replace=False)
    sampled.sort()
    return sampled.astype(np.int64)


def _trial_log_callback(study: optuna.Study, trial: optuna.Trial) -> None:
    state = str(trial.state).split(".")[-1]
    if trial.value is None:
        value_txt = "None"
    else:
        value_txt = f"{float(trial.value):.6f}"

    best_value_txt = "N/A"
    if study.best_trial is not None and study.best_trial.value is not None:
        best_value_txt = f"{float(study.best_trial.value):.6f}"

    print(
        f"[optuna] trial={trial.number} state={state} value={value_txt} "
        f"best_study_value={best_value_txt}"
    )


def main() -> None:
    args = parse_args()
    check_gpu()

    has_gpu = bool(tf.config.list_physical_devices("GPU"))
    if args.mixed_precision and has_gpu:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled: mixed_float16")
    else:
        keras.mixed_precision.set_global_policy("float32")

    if int(args.n_jobs) > 1 and has_gpu:
        print("Warning: n_jobs > 1 with a single GPU can slow down training due to contention. Using n_jobs=1.")
        args.n_jobs = 1

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

    x_tensor_train = data["X_tensor_train"]
    y_train = data["y_train"]
    train_train_idx = data["train_train_idx"]
    train_val_idx = data["train_val_idx"]
    list_classes_weights = np.asarray(data["list_classes_weights"], dtype=np.float32)

    if len(train_train_idx) == 0 or len(train_val_idx) == 0:
        raise ValueError("Split invalide: train_train_idx ou train_val_idx est vide.")

    sequence_length = int(x_tensor_train.shape[1])
    input_dim = int(x_tensor_train.shape[2])
    output_dim = int(len(label_encoder.classes_))

    y_one_hot_train = keras.utils.to_categorical(y_train, num_classes=output_dim).astype(np.float32)

    sampled_train_idx = _subsample_indices(train_train_idx, args.train_subsample_ratio, args.optuna_seed + 17)
    sampled_val_idx = _subsample_indices(train_val_idx, args.val_subsample_ratio, args.optuna_seed + 29)

    print("\n[1/3] Dataset loaded for optimization")
    print(f"  Dataset: {dataset_name}")
    print(f"  X_tensor_train: {x_tensor_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  train_train_idx: {len(train_train_idx)}")
    print(f"  train_val_idx: {len(train_val_idx)}")
    print(
        f"  sampled train/val: {len(sampled_train_idx)}/{len(sampled_val_idx)} "
        f"(ratios={float(args.train_subsample_ratio):.2f}/{float(args.val_subsample_ratio):.2f})"
    )
    print(f"  Classes: {list(label_encoder.classes_)}")

    sampler = optuna.samplers.TPESampler(seed=args.optuna_seed)
    if args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=max(1, int(args.pruning_warmup_epochs)),
            max_resource=max(1, int(args.epochs)),
            reduction_factor=3,
        )
    else:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(5, min(15, args.n_trials // 4 if args.n_trials > 0 else 5)),
            n_warmup_steps=max(1, int(args.pruning_warmup_epochs)),
        )

    study_name = args.study_name or f"gru_optuna_{dataset_name}"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    objective = build_objective(
        x_tensor_train=x_tensor_train,
        y_one_hot_train=y_one_hot_train,
        train_train_idx=np.asarray(sampled_train_idx, dtype=np.int64),
        train_val_idx=np.asarray(sampled_val_idx, dtype=np.int64),
        class_weights=list_classes_weights,
        sequence_length=sequence_length,
        input_dim=input_dim,
        output_dim=output_dim,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
    )

    print("\n[2/3] Running Optuna search")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=(None if int(args.timeout) <= 0 else int(args.timeout)),
        n_jobs=max(1, int(args.n_jobs)),
        show_progress_bar=True,
        callbacks=[_trial_log_callback],
    )

    best = study.best_trial
    print("\nBest trial found")
    print(f"  Value (min val_loss): {best.value:.6f}")
    print(f"  Params: {best.params}")

    output_dir = Path(dataset_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    best_json_path = output_dir / "gru_optuna_best_params.json"
    full_json_path = output_dir / f"gru_optuna_study_{ts}.json"

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": dataset_name,
        "dataset_metadata": str(dataset_cfg["metadata"]),
        "window_size": dataset_meta.get("preprocessing", {}).get("window_size"),
        "search_space": {
            "hidden_units": [8, 64, 8],
            "dense_units": [8, 32],
            "dropout": [0.1, 0.5],
            "recurrent_dropout": [0.1, 0.5],
            "initial_learning_rate": [1e-4, 5e-3, "log"],
        },
        "objective": "minimize val_loss on spatial validation split (train_val_idx)",
        "study": {
            "name": study.study_name,
            "direction": str(study.direction),
            "n_trials_requested": int(args.n_trials),
            "n_trials_total": int(len(study.trials)),
            "n_trials_completed": int(len(completed_trials)),
            "n_trials_pruned": int(len(pruned_trials)),
            "best_value": float(best.value),
            "best_trial_number": int(best.number),
            "best_epoch": int(best.user_attrs.get("best_epoch", -1)),
            "best_params": _to_jsonable(best.params),
        },
        "fixed_training_context": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "early_stopping_patience": int(args.patience),
            "early_stopping_min_delta": float(args.min_delta),
            "pruner": str(args.pruner),
            "pruning_warmup_epochs": int(args.pruning_warmup_epochs),
            "n_jobs": int(args.n_jobs),
            "mixed_precision": bool(args.mixed_precision),
            "train_subsample_ratio": float(args.train_subsample_ratio),
            "val_subsample_ratio": float(args.val_subsample_ratio),
            "loss": "CategoricalFocalCrossentropy",
            "optimizer": "Adam",
            "split_source": str(dataset_cfg["preprocessed_data"]),
            "train_indices_key": "train_train_idx",
            "val_indices_key": "train_val_idx",
        },
    }

    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=4)

    trials_summary = []
    for t in study.trials:
        trials_summary.append(
            {
                "number": int(t.number),
                "state": str(t.state),
                "value": None if t.value is None else float(t.value),
                "best_epoch": t.user_attrs.get("best_epoch"),
                "params": _to_jsonable(t.params),
            }
        )

    full_payload = dict(payload)
    full_payload["trials"] = trials_summary

    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(full_payload), f, indent=4)

    print("\n[3/3] Optimization results saved")
    print(f"  Best params file: {best_json_path}")
    print(f"  Full study file : {full_json_path}")


if __name__ == "__main__":
    main()
