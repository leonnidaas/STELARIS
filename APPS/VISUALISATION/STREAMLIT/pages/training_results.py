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
def _load_gru_history(meta: dict) -> dict:
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
    models = report.get("models", {})
    rows = []

    for model_key in ("gru", "xgboost"):
        info = models.get(model_key, {})
        test_m = info.get("test", {})
        if not test_m:
            continue

        rows.append(
            {
                "Modele": model_key.upper(),
                "Accuracy": test_m.get("accuracy"),
                "Balanced Accuracy": test_m.get("balanced_accuracy"),
                "F1 Weighted": test_m.get("f1_weighted"),
                "F1 Macro": test_m.get("f1_macro"),
                "Cross Entropy": test_m.get("cross_entropy"),
                "AUC Weighted": test_m.get("auc"),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _overfitting_rows(report: dict) -> list[dict]:
    rows: list[dict] = []
    models = report.get("models", {})

    for model_key in ("gru", "xgboost"):
        info = models.get(model_key, {})
        train_m = info.get("train", {})
        test_m = info.get("test", {})
        if not train_m or not test_m:
            continue

        delta_acc = float(train_m.get("accuracy", np.nan)) - float(test_m.get("accuracy", np.nan))
        delta_f1 = float(train_m.get("f1_weighted", np.nan)) - float(test_m.get("f1_weighted", np.nan))
        overfit = (delta_acc > OVERFITTING_THRESHOLD) or (delta_f1 > OVERFITTING_THRESHOLD)

        rows.append(
            {
                "Modele": model_key.upper(),
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

    default_dataset = datasets[-1]
    dataset_id = st.sidebar.selectbox("Dataset", datasets, index=len(datasets) - 1)

    if dataset_id != default_dataset:
        st.caption(f"Dataset selectionne: {dataset_id}")

    report_path, report = _latest_comparison_report(dataset_id)
    optuna_path, optuna_payload = _load_optuna_best_params(dataset_id)
    gru_meta_path, gru_meta = _resolve_model_metadata("GRU", dataset_id, report)
    xgb_meta_path, xgb_meta = _resolve_model_metadata("XGBOOST", dataset_id, report)

    header_cols = st.columns(4)
    header_cols[0].metric("Dataset", dataset_id)
    header_cols[1].metric("Rapport comparaison", "Oui" if report else "Non")
    header_cols[2].metric("Modele GRU", "Oui" if gru_meta else "Non")
    header_cols[3].metric("Modele XGBoost", "Oui" if xgb_meta else "Non")

    with st.expander("Sources chargees", expanded=False):
        st.write({
            "comparison_report": str(report_path) if report_path else None,
            "optuna_best_params": str(optuna_path) if optuna_path else None,
            "gru_metadata": str(gru_meta_path) if gru_meta_path else None,
            "xgb_metadata": str(xgb_meta_path) if xgb_meta_path else None,
        })

    st.subheader("Dataset preprocesse")
    preproc = _load_preprocessed_dataset_summary(dataset_id)
    preproc_meta = preproc.get("metadata", {})
    preproc_counts = preproc.get("counts", {})

    info_cols = st.columns(5)
    info_cols[0].metric("Train (total)", str(preproc_counts.get("n_train_total", "N/A")))
    info_cols[1].metric("Train (inner)", str(preproc_counts.get("n_train_inner", "N/A")))
    info_cols[2].metric("Val", str(preproc_counts.get("n_val", "N/A")))
    info_cols[3].metric("Test", str(preproc_counts.get("n_test", "N/A")))
    info_cols[4].metric("Features", str(preproc_counts.get("n_features", "N/A")))

    if preproc_meta:
        source_data = preproc_meta.get("source_data", {})
        prep_cfg = preproc_meta.get("preprocessing", {})
        st.caption(
            "Trajets: "
            f"{len(source_data.get('trajets', []))} | "
            f"Rows brutes: {source_data.get('total_rows', 'N/A')} | "
            f"Rows apres filtres: {source_data.get('total_rows_after_filters', 'N/A')} | "
            f"Rows apres stride: {source_data.get('total_rows_after_stride', 'N/A')} | "
            f"window_size: {prep_cfg.get('window_size', 'N/A')}"
        )

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

    st.subheader("Courbes d'entrainement (GRU)")
    gru_history_payload = _load_gru_history(gru_meta)
    if gru_history_payload:
        _plot_history(gru_history_payload)
    else:
        st.info(
            "Historique introuvable. Lancez un nouvel entrainement GRU (la sauvegarde training_history.json est maintenant automatique)."
        )

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
        st.image(str(xgb_imp_png), caption="Feature importance XGBoost", use_container_width=True)

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
    if not metrics_df.empty:
        st.dataframe(metrics_df, width="stretch", hide_index=True)
    else:
        st.warning("Aucune metrique de comparaison disponible pour ce dataset.")

    overfit_rows = _overfitting_rows(report)
    if overfit_rows:
        st.markdown("**Analyse surentrainement (train vs test)**")
        st.dataframe(pd.DataFrame(overfit_rows), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Matrices de confusion")
    normalize_cm = st.checkbox("Normaliser les matrices par ligne", value=False)

    cm_cols = st.columns(2)
    gru_cm = _load_confusion_from_metadata(gru_meta)
    xgb_cm = _load_confusion_from_metadata(xgb_meta)

    with cm_cols[0]:
        if gru_cm is None:
            st.info("Matrice GRU introuvable.")
        else:
            labels, matrix = gru_cm
            _plot_confusion_matrix("GRU - Test", labels, matrix, normalize=normalize_cm)

    with cm_cols[1]:
        if xgb_cm is None:
            st.info("Matrice XGBoost introuvable.")
        else:
            labels, matrix = xgb_cm
            _plot_confusion_matrix("XGBoost - Test", labels, matrix, normalize=normalize_cm)

    st.divider()
    st.subheader("Classification reports (test)")
    rep_cols = st.columns(2)

    gru_report = report.get("models", {}).get("gru", {}).get("classification_report_test", {})
    xgb_report = report.get("models", {}).get("xgboost", {}).get("classification_report_test", {})

    with rep_cols[0]:
        st.markdown("**GRU**")
        df_gru = _report_table(gru_report)
        if df_gru.empty:
            st.info("Rapport GRU indisponible.")
        else:
            st.dataframe(df_gru, width="stretch", hide_index=True)

    with rep_cols[1]:
        st.markdown("**XGBoost**")
        df_xgb = _report_table(xgb_report)
        if df_xgb.empty:
            st.info("Rapport XGBoost indisponible.")
        else:
            st.dataframe(df_xgb, width="stretch", hide_index=True)


render_page()
