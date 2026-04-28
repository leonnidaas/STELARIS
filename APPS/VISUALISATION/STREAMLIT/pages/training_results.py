from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ui.theme import apply_theme, render_hero
from utils import MODELS_DIR, TRAINING_DIR


st.set_page_config(page_title="Resultats entrainement", layout="wide", page_icon="📉")

OVERFITTING_THRESHOLD = 0.10
PINNED_DATASETS_FILE = TRAINING_DIR / ".training_results_pinned_datasets.json"

MODEL_SPECS = {
    "GRU": {
        "display": "GRU",
        "report_key": "gru",
        "report_aliases": [],
        "family": "keras",
    },
    "XGBOOST": {
        "display": "XGBoost",
        "report_key": "xgboost",
        "report_aliases": ["xgb"],
        "family": "xgb",
    },
    "CNN_1D": {
        "display": "1D CNN",
        "report_key": "cnn_1d",
        "report_aliases": [],
        "family": "keras",
    },
}


@st.cache_data(show_spinner=False)
def _read_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _list_dataset_ids() -> list[str]:
    if not TRAINING_DIR.exists():
        return []

    dataset_ids: list[str] = []
    for d in sorted([p for p in TRAINING_DIR.iterdir() if p.is_dir()], key=lambda x: x.name):
        meta_path = d / f"{d.name}_metadata.json"
        if meta_path.exists():
            dataset_ids.append(d.name)
    return dataset_ids


def _load_pinned_dataset_ids() -> list[str]:
    payload = _read_json(str(PINNED_DATASETS_FILE))
    if not isinstance(payload, dict):
        return []

    pinned = payload.get("pinned", [])
    if not isinstance(pinned, list):
        return []

    out: list[str] = []
    for value in pinned:
        if isinstance(value, str) and value and value not in out:
            out.append(value)
    return out


def _save_pinned_dataset_ids(dataset_ids: list[str]) -> None:
    payload = _read_json(str(PINNED_DATASETS_FILE))
    aliases = payload.get("aliases", {}) if isinstance(payload, dict) else {}
    if not isinstance(aliases, dict):
        aliases = {}

    PINNED_DATASETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PINNED_DATASETS_FILE, "w", encoding="utf-8") as f:
        json.dump({"pinned": dataset_ids, "aliases": aliases}, f, indent=2)


def _load_dataset_aliases() -> dict[str, str]:
    payload = _read_json(str(PINNED_DATASETS_FILE))
    if not isinstance(payload, dict):
        return {}

    aliases = payload.get("aliases", {})
    if not isinstance(aliases, dict):
        return {}

    out: dict[str, str] = {}
    for key, value in aliases.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        key = key.strip()
        value = value.strip()
        if key and value:
            out[key] = value
    return out


def _save_dataset_aliases(aliases: dict[str, str]) -> None:
    payload = _read_json(str(PINNED_DATASETS_FILE))
    pinned = payload.get("pinned", []) if isinstance(payload, dict) else []
    if not isinstance(pinned, list):
        pinned = []

    pinned_clean = [ds for ds in pinned if isinstance(ds, str) and ds]

    PINNED_DATASETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PINNED_DATASETS_FILE, "w", encoding="utf-8") as f:
        json.dump({"pinned": pinned_clean, "aliases": aliases}, f, indent=2)


def _dataset_display_name(dataset_id: str, aliases: dict[str, str]) -> str:
    alias = aliases.get(dataset_id, "").strip()
    if alias:
        return f"{alias} ({dataset_id})"
    return dataset_id


def _ordered_datasets(dataset_ids: list[str], pinned_ids: list[str]) -> list[str]:
    pinned_present = [ds for ds in pinned_ids if ds in dataset_ids]
    others = [ds for ds in dataset_ids if ds not in pinned_present]
    return pinned_present + others


@st.cache_data(show_spinner=False)
def _latest_model_metadata(model_kind: str, dataset_id: str) -> tuple[Path | None, dict]:
    root = MODELS_DIR / model_kind
    if not root.exists():
        return None, {}

    candidates = sorted(root.rglob("*_metadonnees.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for meta_path in candidates:
        meta = _read_json(str(meta_path))
        if meta.get("dataset_id") == dataset_id:
            return meta_path, meta

    return None, {}


@st.cache_data(show_spinner=False)
def _metadata_from_path(path_str: str | None) -> tuple[Path | None, dict]:
    if not path_str:
        return None, {}
    p = Path(path_str)
    if not p.exists():
        return None, {}
    payload = _read_json(str(p))
    if not payload:
        return None, {}
    return p, payload


def _resolve_model_metadata(
    model_kind: str,
    dataset_id: str,
    report: dict,
) -> tuple[Path | None, dict]:
    meta_path, meta = _latest_model_metadata(model_kind, dataset_id)
    if meta:
        return meta_path, meta

    report_key = model_kind.lower()
    report_path = report.get("models", {}).get(report_key, {}).get("metadata_path")
    if not report_path and report_key == "xgboost":
        report_path = report.get("models", {}).get("xgb", {}).get("metadata_path")

    return _metadata_from_path(report_path)


def _report_model_info(report: dict, model_kind: str) -> dict:
    if not isinstance(report, dict):
        return {}

    models = report.get("models", {})
    if not isinstance(models, dict):
        return {}

    spec = MODEL_SPECS.get(model_kind, {})
    keys = [spec.get("report_key", model_kind.lower()), *spec.get("report_aliases", [])]
    for key in keys:
        info = models.get(key, {})
        if isinstance(info, dict) and info:
            return info

    return {}


@st.cache_data(show_spinner=False)
def _latest_comparison_report(dataset_id: str) -> tuple[Path | None, dict]:
    eval_dir = TRAINING_DIR / dataset_id / "evaluations"
    if not eval_dir.exists():
        return None, {}

    latest = eval_dir / "comparison_report.json"
    if latest.exists():
        return latest, _read_json(str(latest))

    candidates = sorted(eval_dir.glob("comparison_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None, {}

    return candidates[0], _read_json(str(candidates[0]))


@st.cache_data(show_spinner=False)
def _load_optuna_best_params(dataset_id: str) -> tuple[Path | None, dict]:
    path = TRAINING_DIR / dataset_id / "gru_optuna_best_params.json"
    if not path.exists():
        return None, {}
    return path, _read_json(str(path))


@st.cache_data(show_spinner=False)
def _load_preprocessed_dataset_summary(dataset_id: str) -> dict:
    dataset_dir = TRAINING_DIR / dataset_id
    meta_path = dataset_dir / f"{dataset_id}_metadata.json"
    npz_path = dataset_dir / f"{dataset_id}_preprocessed_data.npz"
    classes_path = dataset_dir / f"{dataset_id}_classes_param.npy"

    summary = {
        "paths": {
            "metadata": str(meta_path),
            "npz": str(npz_path),
            "classes": str(classes_path),
        },
        "metadata": _read_json(str(meta_path)) if meta_path.exists() else {},
        "shapes": {},
        "counts": {},
        "class_balance": {},
    }

    class_names = None
    if classes_path.exists():
        try:
            class_names = [str(v) for v in np.load(classes_path, allow_pickle=True).tolist()]
        except Exception:
            class_names = None

    if not npz_path.exists():
        return summary

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            for key in data.files:
                arr = data[key]
                if hasattr(arr, "shape"):
                    summary["shapes"][key] = tuple(int(v) for v in arr.shape)

            x_tensor_train = data["X_tensor_train"] if "X_tensor_train" in data.files else None
            x_tensor_test = data["X_tensor_test"] if "X_tensor_test" in data.files else None
            y_train = data["y_train"] if "y_train" in data.files else None
            y_test = data["y_test"] if "y_test" in data.files else None
            train_train_idx = data["train_train_idx"] if "train_train_idx" in data.files else None
            train_val_idx = data["train_val_idx"] if "train_val_idx" in data.files else None

            if y_train is not None:
                summary["counts"]["n_train_total"] = int(len(y_train))
            if y_test is not None:
                summary["counts"]["n_test"] = int(len(y_test))
            if train_train_idx is not None:
                summary["counts"]["n_train_inner"] = int(len(train_train_idx))
            if train_val_idx is not None:
                summary["counts"]["n_val"] = int(len(train_val_idx))
            if x_tensor_train is not None:
                summary["counts"]["window_size"] = int(x_tensor_train.shape[1])
                summary["counts"]["n_features"] = int(x_tensor_train.shape[2])
            if x_tensor_test is not None:
                summary["counts"]["n_test_sequences"] = int(x_tensor_test.shape[0])

            def _dist(encoded: np.ndarray) -> dict[str, int]:
                values, counts = np.unique(encoded.astype(int), return_counts=True)
                out = {}
                for idx, c in zip(values, counts):
                    if class_names and 0 <= int(idx) < len(class_names):
                        label = class_names[int(idx)]
                    else:
                        label = str(int(idx))
                    out[label] = int(c)
                return out

            if y_train is not None and y_test is not None:
                summary["class_balance"]["global"] = _dist(np.concatenate([y_train, y_test], axis=0))
                summary["class_balance"]["test"] = _dist(y_test)
            if y_train is not None and train_train_idx is not None:
                summary["class_balance"]["train"] = _dist(y_train[train_train_idx])
            if y_train is not None and train_val_idx is not None:
                summary["class_balance"]["val"] = _dist(y_train[train_val_idx])

    except Exception:
        return summary

    return summary


def _class_balance_table(class_balance: dict) -> pd.DataFrame:
    if not class_balance:
        return pd.DataFrame()

    labels = sorted({label for split_map in class_balance.values() for label in split_map.keys()})
    rows = []
    for label in labels:
        row = {"Classe": label}
        for split_name, split_map in class_balance.items():
            row[f"{split_name}_count"] = int(split_map.get(label, 0))
        rows.append(row)

    df = pd.DataFrame(rows)

    for split_name in class_balance.keys():
        c_col = f"{split_name}_count"
        p_col = f"{split_name}_pct"
        total = max(1, int(df[c_col].sum()))
        df[p_col] = (df[c_col] / total) * 100.0

    return df


@st.cache_data(show_spinner=False)
def _load_confusion_from_metadata(meta: dict) -> tuple[list[str], np.ndarray] | None:
    cm_path = meta.get("artefacts", {}).get("confusion_matrix_json")
    if not cm_path:
        return None

    content = _read_json(cm_path)
    labels = content.get("labels")
    matrix = content.get("matrix")

    if not isinstance(labels, list) or not isinstance(matrix, list):
        return None

    try:
        arr = np.array(matrix, dtype=float)
    except Exception:
        return None

    if arr.ndim != 2:
        return None

    return [str(v) for v in labels], arr


@st.cache_data(show_spinner=False)
def _load_keras_history(meta: dict) -> dict:
    artefacts = meta.get("artefacts", {})
    candidates = [
        artefacts.get("training_history_json"),
        str((Path(meta.get("artefacts", {}).get("metadata", "")).parent / "training_history.json")),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        payload = _read_json(candidate)
        history = payload.get("history")
        if isinstance(history, dict):
            return payload

    return {}


@st.cache_data(show_spinner=False)
def _load_xgb_history(meta: dict, meta_path: str | None = None) -> dict:
    artefacts = meta.get("artefacts", {})
    candidates = [
        artefacts.get("training_history_json"),
        str((Path(meta_path).parent / "xgboost_training_history.json")) if meta_path else None,
        str((Path(meta.get("artefacts", {}).get("metadata", "")).parent / "xgboost_training_history.json")),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        payload = _read_json(candidate)
        if payload:
            return payload

    return {}


def _resolve_xgb_importance_paths(meta: dict) -> tuple[Path | None, Path | None]:
    artefacts = meta.get("artefacts", {}) if isinstance(meta, dict) else {}

    png_path = artefacts.get("feature_importance_png")
    json_path = artefacts.get("feature_importance_gain_json")

    meta_path = artefacts.get("metadata")
    if meta_path:
        model_dir = Path(meta_path).parent
    else:
        model_dir = None

    if not png_path and model_dir is not None:
        fallback_png = model_dir / "feature_importance.png"
        if fallback_png.exists():
            png_path = str(fallback_png)

    if not json_path and model_dir is not None:
        fallback_json = model_dir / "feature_importance_gain.json"
        if fallback_json.exists():
            json_path = str(fallback_json)

    png = Path(png_path) if png_path and Path(png_path).exists() else None
    jsn = Path(json_path) if json_path and Path(json_path).exists() else None
    return png, jsn


@st.cache_data(show_spinner=False)
def _load_feature_importance_table(path: str) -> pd.DataFrame:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return pd.DataFrame()
    rows = [
        {"feature": str(k), "importance_gain": float(v)}
        for k, v in payload.items()
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    total = float(df["importance_gain"].sum())
    if total > 0.0:
        df["importance_pct"] = (df["importance_gain"] / total) * 100.0
    return df


def _comparison_metrics_table(report: dict) -> pd.DataFrame:
    rows = []

    for model_kind, spec in MODEL_SPECS.items():
        info = _report_model_info(report, model_kind)
        test_m = info.get("test", {})
        if not isinstance(test_m, dict) or not test_m:
            continue
        rows.append(
            {
                "Modele": spec["display"],
                "Accuracy": test_m.get("accuracy", np.nan),
                "Balanced Accuracy": test_m.get("balanced_accuracy", np.nan),
                "F1 Weighted": test_m.get("f1_weighted", np.nan),
                "F1 Macro": test_m.get("f1_macro", np.nan),
                "Cross Entropy": test_m.get("cross_entropy", np.nan),
                "AUC Weighted": test_m.get("auc", np.nan),
            }
        )

    return pd.DataFrame(rows)


def _model_metrics_from_metadata(model_name: str, meta: dict) -> dict | None:
    if not isinstance(meta, dict) or not meta:
        return None
    test_m = meta.get("metrics_test", {})
    if not isinstance(test_m, dict) or not test_m:
        return None

    return {
        "Modele": model_name,
        "Accuracy": test_m.get("accuracy", np.nan),
        "Balanced Accuracy": test_m.get("balanced_accuracy", np.nan),
        "F1 Weighted": test_m.get("f1_weighted", np.nan),
        "F1 Macro": test_m.get("f1_macro", np.nan),
        "Cross Entropy": test_m.get("cross_entropy", np.nan),
        "AUC Weighted": test_m.get("auc", np.nan),
    }


def _classification_report_from_metadata(meta: dict) -> dict:
    if not isinstance(meta, dict) or not meta:
        return {}
    artefacts = meta.get("artefacts", {}) if isinstance(meta.get("artefacts", {}), dict) else {}
    report_path = artefacts.get("classification_report_json")
    if not report_path:
        return {}
    payload = _read_json(str(report_path))
    return payload if isinstance(payload, dict) else {}


def _report_table_or_nan(report_dict: dict) -> pd.DataFrame:
    df = _report_table(report_dict)
    if not df.empty:
        return df

    return pd.DataFrame(
        [
            {
                "Classe": np.nan,
                "Precision": np.nan,
                "Recall": np.nan,
                "F1": np.nan,
                "Support": np.nan,
            }
        ]
    )


def _overfitting_rows(report: dict) -> list[dict]:
    rows: list[dict] = []

    for model_kind, spec in MODEL_SPECS.items():
        info = _report_model_info(report, model_kind)
        train_m = info.get("train", {})
        test_m = info.get("test", {})
        if not train_m or not test_m:
            continue

        delta_acc = float(train_m.get("accuracy", np.nan)) - float(test_m.get("accuracy", np.nan))
        delta_f1 = float(train_m.get("f1_weighted", np.nan)) - float(test_m.get("f1_weighted", np.nan))
        overfit = (delta_acc > OVERFITTING_THRESHOLD) or (delta_f1 > OVERFITTING_THRESHOLD)

        rows.append(
            {
                "Modele": spec["display"],
                "Delta Accuracy": delta_acc,
                "Delta F1 Weighted": delta_f1,
                "Surentrainement": "Oui" if overfit else "Non",
            }
        )

    return rows


def _plot_history(history_payload: dict) -> None:
    history = history_payload.get("history", {})
    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])

    if not train_loss and not val_loss:
        st.info("Historique GRU introuvable pour afficher les courbes de loss.")
        return

    best_epoch = history_payload.get("best_epoch_val_loss")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    if train_loss:
        axes[0].plot(train_loss, label="Train loss", linewidth=2)
    if val_loss:
        axes[0].plot(val_loss, label="Val loss", linewidth=2)
    if isinstance(best_epoch, int):
        axes[0].axvline(best_epoch, color="#c1121f", linestyle="--", linewidth=1.5, label="Best val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    if train_acc:
        axes[1].plot(train_acc, label="Train acc", linewidth=2)
    if val_acc:
        axes[1].plot(val_acc, label="Val acc", linewidth=2)
    if isinstance(best_epoch, int):
        axes[1].axvline(best_epoch, color="#c1121f", linestyle="--", linewidth=1.5, label="Best val loss")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    st.pyplot(fig, width="stretch")

    if train_loss and val_loss:
        final_gap = float(val_loss[-1]) - float(train_loss[-1])
        st.metric("Gap final loss (val - train)", f"{final_gap:.4f}")
        if final_gap > OVERFITTING_THRESHOLD:
            st.warning(
                f"Le gap de loss final ({final_gap:.4f}) depasse le seuil {OVERFITTING_THRESHOLD:.2f}."
            )


def _plot_xgb_history(history_payload: dict) -> None:
    val0 = history_payload.get("validation_0", {})
    val1 = history_payload.get("validation_1", {})

    # Anciennes versions: "logloss"; multiclasse: "mlogloss".
    train_logloss = val0.get("logloss", []) or val0.get("mlogloss", [])
    test_logloss = val1.get("logloss", []) or val1.get("mlogloss", [])

    if not train_logloss or not test_logloss:
        st.info("Historique XGBoost introuvable pour afficher les courbes d'entrainement.")
        return

    n = min(len(train_logloss), len(test_logloss))
    train_logloss = [float(v) for v in train_logloss[:n]]
    test_logloss = [float(v) for v in test_logloss[:n]]
    gap = [te - tr for tr, te in zip(train_logloss, test_logloss)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(train_logloss, label="Train logloss", linewidth=2)
    axes[0].plot(test_logloss, label="Test logloss", linewidth=2)
    axes[0].set_title("XGBoost logloss")
    axes[0].set_xlabel("Boosting iteration")
    axes[0].set_ylabel("Logloss")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(gap, label="Gap (test - train)", linewidth=2, color="#c1121f")
    axes[1].axhline(0.0, linestyle="--", color="#6b7280", linewidth=1)
    axes[1].set_title("Surapprentissage (gap)")
    axes[1].set_xlabel("Boosting iteration")
    axes[1].set_ylabel("Logloss gap")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    st.pyplot(fig, width="stretch")

    if gap:
        final_gap = float(gap[-1])
        st.metric("Gap final XGBoost (test - train)", f"{final_gap:.4f}")
        if final_gap > OVERFITTING_THRESHOLD:
            st.warning(
                f"Le gap final XGBoost ({final_gap:.4f}) depasse le seuil {OVERFITTING_THRESHOLD:.2f}."
            )


def _plot_confusion_matrix(title: str, labels: list[str], matrix: np.ndarray, normalize: bool) -> None:
    cm = matrix.copy()
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Verite")

    # Annotate each cell with its value to make the matrix directly readable.
    n_rows, n_cols = cm.shape
    threshold = float(np.nanmax(cm)) * 0.55 if cm.size else 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            value = float(cm[i, j])
            txt = f"{value:.2f}" if normalize else f"{int(round(value))}"
            text_color = "white" if value > threshold else "#0f172a"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=text_color)

    max_labels = 20
    if len(labels) <= max_labels:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


@st.cache_data(show_spinner=False)
def _report_table(report_dict: dict) -> pd.DataFrame:
    rows = []
    for label, metrics in report_dict.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "Classe": label,
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1": metrics.get("f1-score"),
                "Support": metrics.get("support"),
            }
        )

    return pd.DataFrame(rows)


def render_page() -> None:
    apply_theme()
    render_hero(
        "Resultats d'entrainement",
        "Courbes de loss train/val, detection de surentrainement, metriques et matrices de confusion.",
    )

    datasets = _list_dataset_ids()
    if not datasets:
        st.warning("Aucun dataset detecte dans DATA/03_TRAINING.")
        st.stop()

    pinned_ids = _load_pinned_dataset_ids()
    aliases = _load_dataset_aliases()
    datasets_ordered = _ordered_datasets(datasets, pinned_ids)

    default_dataset = datasets[-1]
    preferred_dataset = st.session_state.get("training_results_dataset_id")
    if not preferred_dataset or preferred_dataset not in datasets_ordered:
        if pinned_ids and pinned_ids[0] in datasets_ordered:
            preferred_dataset = pinned_ids[0]
        else:
            preferred_dataset = default_dataset

    selected_index = datasets_ordered.index(preferred_dataset)
    dataset_id = st.sidebar.selectbox(
        "Dataset",
        datasets_ordered,
        index=selected_index,
        format_func=lambda ds: _dataset_display_name(ds, aliases),
    )
    st.session_state["training_results_dataset_id"] = dataset_id

    with st.sidebar.expander("Acces rapide datasets", expanded=True):
        c1, c2 = st.columns(2)
        if c1.button("Remonter", width="stretch"):
            new_pinned = [dataset_id] + [ds for ds in pinned_ids if ds != dataset_id]
            _save_pinned_dataset_ids(new_pinned)
            st.rerun()
        if c2.button("Retirer", width="stretch"):
            new_pinned = [ds for ds in pinned_ids if ds != dataset_id]
            _save_pinned_dataset_ids(new_pinned)
            st.rerun()

        current_alias = aliases.get(dataset_id, "")
        alias_key = f"rename_dataset_alias_{dataset_id}"
        st.text_input(
            "Nom d'affichage",
            value=current_alias,
            key=alias_key,
            placeholder="Ex: AGONAC no coverage run",
        )

        c3, c4 = st.columns(2)
        if c3.button("Renommer", width="stretch"):
            new_alias = str(st.session_state.get(alias_key, "")).strip()
            new_aliases = dict(aliases)
            if new_alias:
                new_aliases[dataset_id] = new_alias
            else:
                new_aliases.pop(dataset_id, None)
            _save_dataset_aliases(new_aliases)
            st.rerun()

        if c4.button("Supprimer nom", width="stretch"):
            new_aliases = dict(aliases)
            new_aliases.pop(dataset_id, None)
            _save_dataset_aliases(new_aliases)
            st.rerun()

        if pinned_ids:
            st.caption("Datasets remontes (ordre prioritaire):")
            for ds in pinned_ids:
                if ds in datasets:
                    st.write(f"- {_dataset_display_name(ds, aliases)}")
        else:
            st.caption("Aucun dataset remonte pour le moment.")

    if dataset_id != default_dataset:
        st.caption(f"Dataset selectionne: {_dataset_display_name(dataset_id, aliases)}")

    report_path, report = _latest_comparison_report(dataset_id)
    optuna_path, optuna_payload = _load_optuna_best_params(dataset_id)
    gru_meta_path, gru_meta = _resolve_model_metadata("GRU", dataset_id, report)
    xgb_meta_path, xgb_meta = _resolve_model_metadata("XGBOOST", dataset_id, report)
    cnn_meta_path, cnn_meta = _resolve_model_metadata("CNN_1D", dataset_id, report)

    model_meta_map = {
        "GRU": (gru_meta_path, gru_meta),
        "XGBOOST": (xgb_meta_path, xgb_meta),
        "CNN_1D": (cnn_meta_path, cnn_meta),
    }
    available_model_kinds = [k for k, (_, m) in model_meta_map.items() if isinstance(m, dict) and bool(m)]

    if "training_results_models" not in st.session_state:
        st.session_state["training_results_models"] = available_model_kinds[:]
    else:
        kept = [m for m in st.session_state["training_results_models"] if m in MODEL_SPECS]
        if available_model_kinds and not kept:
            kept = available_model_kinds[:]
        st.session_state["training_results_models"] = kept

    selected_model_kinds = st.sidebar.multiselect(
        "Modeles a analyser",
        options=list(MODEL_SPECS.keys()),
        default=st.session_state["training_results_models"],
        format_func=lambda m: MODEL_SPECS[m]["display"],
        help="Selectionnez 1, 2 ou 3 modeles pour une analyse independante.",
    )
    st.session_state["training_results_models"] = selected_model_kinds

    if not selected_model_kinds:
        st.warning("Selectionnez au moins un modele dans la barre laterale.")
        st.stop()

    header_cols = st.columns(5)
    header_cols[0].metric("Dataset", _dataset_display_name(dataset_id, aliases))
    header_cols[1].metric("Rapport comparaison", "Oui" if report else "Non")
    header_cols[2].metric("Modele GRU", "Oui" if gru_meta else "Non")
    header_cols[3].metric("Modele XGBoost", "Oui" if xgb_meta else "Non")
    header_cols[4].metric("Modele 1D CNN", "Oui" if cnn_meta else "Non")

    with st.expander("Sources chargees", expanded=False):
        st.write({
            "comparison_report": str(report_path) if report_path else None,
            "optuna_best_params": str(optuna_path) if optuna_path else None,
            "gru_metadata": str(gru_meta_path) if gru_meta_path else None,
            "xgb_metadata": str(xgb_meta_path) if xgb_meta_path else None,
            "cnn_1d_metadata": str(cnn_meta_path) if cnn_meta_path else None,
        })

    st.subheader("Dataset preprocesse")
    preproc = _load_preprocessed_dataset_summary(dataset_id)
    preproc_meta = preproc.get("metadata", {})
    preproc_counts = preproc.get("counts", {})
    preproc_settings = preproc_meta.get("preprocessing", {})
    
    info_cols = st.columns(5)
    info_cols[0].metric("Train (total)", str(preproc_counts.get("n_train_total", "N/A")))
    info_cols[1].metric("Train (inner)", str(preproc_counts.get("n_train_inner", "N/A")))
    val_ratio = preproc_settings.get("val_split_ratio")
    test_ratio = preproc_settings.get("test_split_ratio")
    val_label = f"Val ({float(val_ratio) * 100:.0f}%)" if val_ratio is not None else "Val"
    test_label = f"Test ({float(test_ratio) * 100:.0f}%)" if test_ratio is not None else "Test"
    info_cols[2].metric(val_label, str(preproc_counts.get("n_val", "N/A")))
    info_cols[3].metric(test_label, str(preproc_counts.get("n_test", "N/A")))
    info_cols[4].metric("Features", str(preproc_counts.get("n_features", "N/A")))

    if preproc_meta:
        source_data = preproc_meta.get("source_data", {})
        prep_cfg = preproc_meta.get("preprocessing", {})
        info_cols[0].metric("Scénarios" , str(len(source_data.get("trajets", []))))
        info_cols[1].metric("Rows brutes", str(source_data.get("total_rows", "N/A")))
        info_cols[2].metric("Rows apres filtres", str(source_data.get("total_rows_after_filters", "N/A")))
        info_cols[3].metric("Rows apres stride", str(source_data.get("total_rows_after_stride", "N/A")))
        info_cols[4].metric("Window size", str(prep_cfg.get("window_size", "N/A"))) 
        
    class_balance_df = _class_balance_table(preproc.get("class_balance", {}))
    if class_balance_df.empty:
        st.info("Impossible de charger la distribution des classes depuis le dataset preprocesse.")
    else:
        st.markdown("**Equilibre des classes (count + pourcentage)**")
        st.dataframe(class_balance_df, width="stretch", hide_index=True)

        pct_cols = [c for c in class_balance_df.columns if c.endswith("_pct")]
        if pct_cols:
            chart_df = class_balance_df[["Classe", *pct_cols]].set_index("Classe")
            st.bar_chart(chart_df, width="stretch")

    st.subheader("Optimisation GRU (Optuna)")
    if optuna_payload:
        study_info = optuna_payload.get("study", {})
        best_params = study_info.get("best_params", {})
        best_value = study_info.get("best_value")

        top_cols = st.columns(3)
        top_cols[0].metric("Best val_loss", f"{float(best_value):.6f}" if best_value is not None else "N/A")
        top_cols[1].metric("Trials completes", str(study_info.get("n_trials_completed", "N/A")))
        top_cols[2].metric("Trials prunes", str(study_info.get("n_trials_pruned", "N/A")))

        if best_params:
            st.dataframe(
                pd.DataFrame(
                    [{"Hyperparametre": k, "Valeur": v} for k, v in best_params.items()]
                ),
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("Aucun resultat Optuna trouve pour ce dataset (gru_optuna_best_params.json absent).")

    keras_models_selected = [m for m in selected_model_kinds if MODEL_SPECS[m]["family"] == "keras"]
    if keras_models_selected:
        st.subheader("Courbes d'entrainement (modeles Keras)")
        for model_kind in keras_models_selected:
            _, model_meta = model_meta_map[model_kind]
            st.markdown(f"**{MODEL_SPECS[model_kind]['display']}**")
            history_payload = _load_keras_history(model_meta)
            if history_payload:
                _plot_history(history_payload)
            else:
                st.info(
                    f"Historique introuvable pour {MODEL_SPECS[model_kind]['display']}."
                )

    if "XGBOOST" in selected_model_kinds:
        st.subheader("Courbes d'entrainement et surapprentissage (XGBoost)")
        xgb_history_payload = _load_xgb_history(
            xgb_meta,
            str(xgb_meta_path) if xgb_meta_path else None,
        )
        if xgb_history_payload:
            _plot_xgb_history(xgb_history_payload)
        else:
            st.info(
                "Historique XGBoost introuvable. Lancez un nouvel entrainement XGBoost pour generer xgboost_training_history.json."
            )

        st.subheader("Feature importances (XGBoost)")
        xgb_imp_png, xgb_imp_json = _resolve_xgb_importance_paths(xgb_meta)

        if xgb_imp_png is not None:
            st.image(str(xgb_imp_png), caption="Feature importance XGBoost", width="stretch")

        if xgb_imp_json is not None:
            imp_df = _load_feature_importance_table(str(xgb_imp_json))
            if not imp_df.empty:
                st.dataframe(imp_df, width="stretch", hide_index=True)
        elif isinstance(xgb_meta, dict) and xgb_meta.get("features"):
            st.caption("Importance numerique indisponible pour cet ancien run, mais la liste des features est presente.")
            st.dataframe(
                pd.DataFrame({"feature": [str(v) for v in xgb_meta.get("features", [])]}),
                width="stretch",
                hide_index=True,
            )

        if xgb_imp_png is None and xgb_imp_json is None:
            st.info("Feature importances XGBoost introuvables pour ce dataset.")

    st.divider()
    st.subheader("Metriques de performance")

    metrics_df = _comparison_metrics_table(report)
    st.info(f'{metrics_df}')
    if metrics_df.empty:
        fallback_rows = []
        for model_kind, (_, model_meta) in model_meta_map.items():
            row = _model_metrics_from_metadata(MODEL_SPECS[model_kind]["display"], model_meta)
            if row is not None:
                fallback_rows.append(row)
        metrics_df = pd.DataFrame(fallback_rows)

    if not metrics_df.empty:
        allowed = {MODEL_SPECS[m]["display"] for m in selected_model_kinds}

        metrics_df = metrics_df[metrics_df["Modele"].isin(allowed)].reset_index(drop=True)

    if metrics_df.empty:
        st.info("Aucune metrique disponible pour ce dataset (entraine au moins un modele GRU ou XGBoost).")
    else:
        st.dataframe(metrics_df, width="stretch", hide_index=True)

    overfit_rows = _overfitting_rows(report)
    if overfit_rows:
        allowed = {MODEL_SPECS[m]["display"] for m in selected_model_kinds}
        overfit_rows = [r for r in overfit_rows if r.get("Modele") in allowed]
    if overfit_rows:
        st.markdown("**Analyse surentrainement (train vs test)**")
        st.dataframe(pd.DataFrame(overfit_rows), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Matrices de confusion")
    normalize_cm = st.checkbox("Normaliser les matrices par ligne", value=False)

    selected_pairs = [(m, model_meta_map[m][1]) for m in selected_model_kinds]
    cm_cols = st.columns(max(1, len(selected_pairs)))
    for idx, (model_kind, model_meta) in enumerate(selected_pairs):
        with cm_cols[idx]:
            cm = _load_confusion_from_metadata(model_meta)
            model_label = MODEL_SPECS[model_kind]["display"]
            if cm is None:
                st.info(f"Matrice {model_label} introuvable.")
            else:
                labels, matrix = cm
                _plot_confusion_matrix(f"{model_label} - Test", labels, matrix, normalize=normalize_cm)

    st.divider()
    st.subheader("Classification reports (test)")
    rep_cols = st.columns(max(1, len(selected_pairs)))
    for idx, (model_kind, model_meta) in enumerate(selected_pairs):
        with rep_cols[idx]:
            model_label = MODEL_SPECS[model_kind]["display"]
            report_info = _report_model_info(report, model_kind)
            model_report = report_info.get("classification_report_test", {})
            if not model_report:
                model_report = _classification_report_from_metadata(model_meta)

            st.markdown(f"**{model_label}**")
            st.dataframe(_report_table_or_nan(model_report), width="stretch", hide_index=True)


render_page()
