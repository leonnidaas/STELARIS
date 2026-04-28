from matplotlib import colors
import matplotlib.cm as color_map
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from VISUALISATION.STREAMLIT.modules.cartographie import LABEL_COLORS, render_categorical_segments_map
from VISUALISATION.STREAMLIT.services.training_service import (
    build_trajet_groups,
    expand_unique_trajets,
    list_trajets,
)
from VISUALISATION.STREAMLIT.ui.theme import apply_theme, render_hero
from utils import get_traj_paths


st.set_page_config(page_title="Evaluation Inference", layout="wide", page_icon="🧪")


RISK_RULES = {
    ("open-sky", "tree"): {
        "risk_level": "fausse_alarme",
        "risk_label": "Fausse alarme",
        "risk_text": "Open-sky -> Forest",
        "dangerous": "non",
        "color": "#FFD84D",  # Jaune
        "description": "Le modele est trop prudent. Pas de risque securite.",
    },
    ("tree", "open-sky"): {
        "risk_level": "optimisme_dangereux",
        "risk_label": "Optimisme dangereux",
        "risk_text": "Forest -> Open-sky",
        "dangerous": "oui",
        "color": "#FF8C00",  # Orange
        "description": "Le modele ignore les multitrajets des arbres. Risque de derive.",
    },
    ("open-sky", "build"): {
        "risk_level": "fausse_alarme",
        "risk_label": "Fausse alarme",
        "risk_text": "Open-sky -> Build",
        "dangerous": "non",
        "color": "#2F80ED",  # Bleu
        "description": "Erreur de contexte, risque faible.",
    },
    ("build", "open-sky"): {
        "risk_level": "critique",
        "risk_label": "CRITIQUE",
        "risk_text": "Build -> Open-sky",
        "dangerous": "oui",
        "color": "#E02020",  # Rouge
        "description": "Reflexions batiments ignorees. Risque de saut de position massif.",
    },
    ("bridge", "open-sky"): {
        "risk_level": "fatal",
        "risk_label": "FATAL",
        "risk_text": "Bridge -> Open-sky",
        "dangerous": "oui",
        "color": "#2E1065",  # Violet/noir
        "description": "Le pire scenario: signal degrade mais valide a tort.",
    },
}


LABEL_COLOR_MAP = {
    "open-sky": LABEL_COLORS["open-sky"],
    "tree": LABEL_COLORS["tree"],
    "build": LABEL_COLORS["build"],
    "bridge": LABEL_COLORS["bridge"],
    "mixed": LABEL_COLORS["mixed"],
    "gare": LABEL_COLORS["gare"],
    "signal-denied": LABEL_COLORS["signal-denied"],
    "signal_denied": LABEL_COLORS["signal-denied"],
    "other": LABEL_COLORS["other"],
}


def _normalize_label(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    return s.lower().replace("_", "-")


def _canonical_env_label(label: str | None) -> str | None:
    if label is None:
        return None
    norm = _normalize_label(label)
    if norm is None:
        return None
    aliases = {
        "forest": "tree",
        "signal denied": "signal-denied",
    }
    return aliases.get(norm, norm)


def _risk_info(true_label: str | None, pred_label: str | None) -> dict[str, str]:
    t = _canonical_env_label(true_label)
    p = _canonical_env_label(pred_label)
    if t is None or p is None:
        return {
            "risk_level": "inconnu",
            "risk_label": "Inconnu",
            "risk_text": "Transition inconnue",
            "dangerous": "inconnu",
            "color": "#9CA3AF",
            "description": "Labels insuffisants pour qualifier le risque.",
        }

    if t == p:
        return {
            "risk_level": "ok",
            "risk_label": "OK",
            "risk_text": f"{t} -> {p}",
            "dangerous": "non",
            "color": "#2CA02C",
            "description": "Prediction coherente avec le label de reference.",
        }

    rule = RISK_RULES.get((t, p))
    if rule:
        return rule

    return {
        "risk_level": "desaccord",
        "risk_label": "Desaccord",
        "risk_text": f"{t} -> {p}",
        "dangerous": "non",
        "color": "#8B5CF6",
        "description": "Desaccord non classe dans les cas critiques definis.",
    }


def _first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def _load_reference_df(traj_id: str, source_ref: str) -> pd.DataFrame:
    cfg = get_traj_paths(traj_id)
    ref_path = cfg["final_fusion_csv"] if source_ref == "LiDAR" else cfg["final_fusion_osm_csv"]
    if not ref_path.exists():
        raise FileNotFoundError(
            "Fichier reference introuvable: "
            f"{ref_path}. Pour l'evaluation (matrice de confusion + carte), "
            "il faut la GroundTruth et l'execution complete du pipeline de labelisation "
            "(jusqu'au fichier de fusion final)."
        )
    return pd.read_csv(ref_path)


def _list_inference_runs(traj_id: str) -> list[Path]:
    cfg = get_traj_paths(traj_id)
    inference_dir = Path(cfg["inference_dir"])
    if not inference_dir.exists():
        return []
    pattern = f"{traj_id}_*_inference.csv"
    return sorted(inference_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_inference_df(traj_id: str, inference_csv: Path | None = None) -> pd.DataFrame:
    cfg = get_traj_paths(traj_id)
    inf_path = Path(inference_csv) if inference_csv is not None else Path(cfg["inference_latest_csv"])
    if not inf_path.exists():
        raise FileNotFoundError(f"Inference latest introuvable: {inf_path}")
    return pd.read_csv(inf_path)


def _plot_confusion_matrix(title: str, labels: list[str], matrix: np.ndarray, normalize: bool) -> None:
    cm = matrix.astype(float, copy=True)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Verite")

    threshold = float(np.nanmax(cm)) * 0.55 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = float(cm[i, j])
            text = f"{value:.2f}" if normalize else f"{int(round(value))}"
            text_color = "white" if value > threshold else "#0f172a"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=text_color)

    if len(labels) <= 20:
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


def _merge_ref_pred(df_ref: pd.DataFrame, df_inf: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    ref_label_col = _first_existing(df_ref.columns, ["label", "env_label", "label_env"])
    if ref_label_col is None:
        raise ValueError("Aucune colonne label de reference detectee (label/env_label/label_env).")

    pred_col = _first_existing(df_inf.columns, ["pred_final", "pred_gru", "pred_xgb"])
    if pred_col is None:
        raise ValueError("Aucune colonne prediction detectee (pred_final/pred_gru/pred_xgb).")

    merge_key = _first_existing(df_ref.columns, ["gnss_feat_gps_millis", "gps_millis", "time_utc"])
    if merge_key is None or merge_key not in df_inf.columns:
        alt_key = _first_existing(df_inf.columns, ["gnss_feat_gps_millis", "gps_millis", "time_utc"])
        if alt_key is None or alt_key not in df_ref.columns:
            raise ValueError("Impossible de trouver une cle de jointure commune (gps_millis/time_utc).")
        merge_key = alt_key

    cols_ref = [merge_key, ref_label_col]
    for c in ("latitude_gt", "longitude_gt", "lat_gt", "long_gt", "time_utc"):
        if c in df_ref.columns and c not in cols_ref:
            cols_ref.append(c)

    cols_inf = [merge_key, pred_col]
    for c in (
        "pred_gru",
        "pred_xgb",
        "pred_final",
        "model_final",
        "conf_gru",
        "conf_xgb",
        "conf_final",
        "conf_max_models",
        "time_utc",
    ):
        if c in df_inf.columns and c not in cols_inf:
            cols_inf.append(c)

    merged = pd.merge(df_ref[cols_ref], df_inf[cols_inf], on=merge_key, how="inner", suffixes=("_ref", "_pred"))
    merged["label_ref_norm"] = merged[ref_label_col].map(_normalize_label)
    merged["label_pred_norm"] = merged[pred_col].map(_normalize_label)
    if "pred_gru" in merged.columns:
        merged["label_pred_gru_norm"] = merged["pred_gru"].map(_normalize_label)
    if "pred_xgb" in merged.columns:
        merged["label_pred_xgb_norm"] = merged["pred_xgb"].map(_normalize_label)
    if "pred_final" in merged.columns:
        merged["label_pred_final_norm"] = merged["pred_final"].map(_normalize_label)

    if "conf_gru" in merged.columns:
        merged["conf_gru"] = pd.to_numeric(merged["conf_gru"], errors="coerce")
    if "conf_xgb" in merged.columns:
        merged["conf_xgb"] = pd.to_numeric(merged["conf_xgb"], errors="coerce")
    if "conf_final" in merged.columns:
        merged["conf_final"] = pd.to_numeric(merged["conf_final"], errors="coerce")
    if "conf_max_models" in merged.columns:
        merged["conf_max_models"] = pd.to_numeric(merged["conf_max_models"], errors="coerce")

    if "conf_final" not in merged.columns and "conf_max_models" in merged.columns:
        merged["conf_final"] = merged["conf_max_models"]

    if (
        "label_pred_final_norm" in merged.columns
        and "conf_gru" in merged.columns
        and "conf_xgb" in merged.columns
        and "conf_final" not in merged.columns
    ):
        use_gru = merged["label_pred_final_norm"] == merged.get("label_pred_gru_norm")
        use_xgb = merged["label_pred_final_norm"] == merged.get("label_pred_xgb_norm")
        merged["conf_final"] = np.nan
        merged.loc[use_gru, "conf_final"] = merged.loc[use_gru, "conf_gru"]
        merged.loc[use_xgb, "conf_final"] = merged.loc[use_xgb, "conf_xgb"]
        fallback = merged["conf_final"].isna()
        if fallback.any():
            conf_pair = merged.loc[fallback, ["conf_gru", "conf_xgb"]]
            # max ligne a ligne en ignorant les NaN; si les 2 sont NaN -> reste NaN (sans warning)
            merged.loc[fallback, "conf_final"] = conf_pair.max(axis=1, skipna=True)

    merged = merged.dropna(subset=["label_ref_norm", "label_pred_norm"]).reset_index(drop=True)
    return merged, ref_label_col, pred_col


def render_page() -> None:
    apply_theme()
    render_hero(
        "Evaluation modele vs labelisation",
        "Compare visuellement et quantitativement les predictions d'inference avec la reference LiDAR/OSM.",
    )

    trajets_disponibles = list_trajets()
    if not trajets_disponibles:
        st.error("Aucun trajet disponible.")
        st.stop()

    grouped_trajets = build_trajet_groups(trajets_disponibles)
    trajets_uniques = sorted(grouped_trajets.keys())

    c1, c2 = st.columns([2, 1])
    with c1:
        mode = st.radio("Selection trajet", ["Trajets uniques", "Scenario manuel"], horizontal=True)
        if mode == "Trajets uniques":
            unique_id = st.selectbox("Trajet unique", trajets_uniques, index=0)
            scenario_mode = st.radio("Scenario", ["Premier scenario", "Choisir un scenario"], horizontal=True)
            scenarios = grouped_trajets.get(unique_id, [])
            if scenario_mode == "Premier scenario":
                trajet_id = scenarios[0] if scenarios else ""
            else:
                trajet_id = st.selectbox("Scenario", scenarios, index=0 if scenarios else None)
        else:
            trajet_id = st.selectbox("Scenario", trajets_disponibles, index=0)

    with c2:
        source_ref = st.selectbox("Reference labelisation", ["LiDAR", "OSM"], index=0)

    selected_inference_path = None
    inference_runs = _list_inference_runs(trajet_id) if trajet_id else []
    if trajet_id:
        cfg = get_traj_paths(trajet_id)
        latest_path = Path(cfg["inference_latest_csv"])
        options: list[tuple[str, Path]] = []
        if latest_path.exists():
            options.append((f"latest ({latest_path.name})", latest_path))
        options.extend([(p.name, p) for p in inference_runs if p != latest_path])

        if options:
            labels = [label for label, _ in options]
            chosen_label = st.selectbox("Fichier inference", labels, index=0)
            selected_inference_path = dict(options)[chosen_label]
        else:
            st.warning("Aucun fichier d'inference disponible pour ce trajet.")

    if not trajet_id:
        st.warning("Aucun trajet selectionne.")
        st.stop()

    try:
        df_ref = _load_reference_df(trajet_id, source_ref)
        df_inf = _load_inference_df(trajet_id, selected_inference_path)
        merged, ref_label_col, pred_col = _merge_ref_pred(df_ref, df_inf)
    except Exception as e:
        st.error(f"Impossible de charger/aligner les donnees: {e}")
        st.stop()

    if merged.empty:
        st.warning("Aucune ligne comparable entre reference et inference.")
        st.stop()

    y_true = merged["label_ref_norm"].to_numpy()
    y_pred = merged["label_pred_norm"].to_numpy()

    labels = sorted(set(y_true).union(set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted")
    f1m = f1_score(y_true, y_pred, average="macro")
    disagree = (y_true != y_pred)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Points compares", f"{len(merged):,}")
    m2.metric("Accuracy", f"{acc:.3f}")
    m3.metric("Balanced acc", f"{bal_acc:.3f}")
    m4.metric("F1 weighted", f"{f1w:.3f}")
    m5.metric("Desaccord", f"{100.0 * float(np.mean(disagree)):.1f}%")

    selected_inf_name = selected_inference_path.name if selected_inference_path is not None else "latest"
    st.caption(
        f"Reference={source_ref} ({ref_label_col}) | Prediction={pred_col} | Trajet={trajet_id} | Inference={selected_inf_name}"
    )

    normalize_cm = st.checkbox("Normaliser les matrices par ligne", value=False)
    _plot_confusion_matrix(
        "Matrice de confusion normalisee" if normalize_cm else "Matrice de confusion",
        labels,
        cm,
        normalize=normalize_cm,
    )

    comp = pd.DataFrame({"true": y_true, "pred": y_pred})
    true_dist = comp["true"].value_counts(normalize=True).rename("true_ratio")
    pred_dist = comp["pred"].value_counts(normalize=True).rename("pred_ratio")
    dist_df = pd.concat([true_dist, pred_dist], axis=1).fillna(0.0).reset_index().rename(columns={"index": "label"})
    dist_m = dist_df.melt(id_vars=["label"], value_vars=["true_ratio", "pred_ratio"], var_name="serie", value_name="ratio")
    fig_dist = px.bar(dist_m, x="label", y="ratio", color="serie", barmode="group", title="Distribution labels: reference vs prediction")
    st.plotly_chart(fig_dist, width="stretch")

    merged["is_disagreement"] = disagree.astype(int)
    t_col = _first_existing(merged.columns, ["time_utc_ref", "time_utc_pred", "time_utc"])
    if t_col is not None:
        ts = pd.to_datetime(merged[t_col], errors="coerce")
        ts_df = pd.DataFrame({"time": ts, "disagreement": merged["is_disagreement"]}).dropna(subset=["time"])
        if not ts_df.empty:
            fig_ts = px.scatter(ts_df, x="time", y="disagreement", title="Desaccords dans le temps (1=desaccord)", opacity=0.35)
            st.plotly_chart(fig_ts, width="stretch")

    lat_col = _first_existing(merged.columns, ["latitude_gt", "lat_gt"])
    lon_col = _first_existing(merged.columns, ["longitude_gt", "long_gt"])
    if lat_col and lon_col:
        map_df = merged[[lat_col, lon_col, "is_disagreement", "label_ref_norm", "label_pred_norm"]].copy()
        optional_cols = [
            "label_pred_gru_norm",
            "label_pred_xgb_norm",
            "label_pred_final_norm",
            "conf_gru",
            "conf_xgb",
            "conf_final",
        ]
        for col in optional_cols:
            if col in merged.columns:
                map_df[col] = merged[col]

        map_df = map_df.dropna(subset=[lat_col, lon_col])
        if not map_df.empty:
            map_df = map_df.rename(columns={lat_col: "latitude_gt", lon_col: "longitude_gt"})

            map_mode = st.selectbox(
                "Affichage carte",
                [
                    "Label vrai",
                    "Prediction modele 1 (GRU)",
                    "Prediction modele 2 (XGBoost)",
                    "Prediction fusion",
                    "Risque (dangerosite)",
                    "Desaccord simple",
                    "Confiance"
                ],
                index=4,
            )

            if map_mode == "Label vrai":
                map_df["map_category"] = map_df["label_ref_norm"].map(_canonical_env_label)
                map_color_map = LABEL_COLOR_MAP
                legend_title = "Labels reference"
                popup_fields = ["label_ref_norm", "label_pred_gru_norm", "label_pred_xgb_norm", "label_pred_final_norm", "conf_gru", "conf_xgb", "conf_final"]
            elif map_mode == "Prediction modele 1 (GRU)":
                if "label_pred_gru_norm" not in map_df.columns:
                    st.info("Colonne pred_gru absente dans ce CSV d'inference.")
                    map_df = pd.DataFrame()
                else:
                    map_df["map_category"] = map_df["label_pred_gru_norm"].map(_canonical_env_label)
                    map_color_map = LABEL_COLOR_MAP
                    legend_title = "Prediction GRU"
                    popup_fields = ["label_ref_norm", "label_pred_gru_norm", "conf_gru"]
            elif map_mode == "Prediction modele 2 (XGBoost)":
                if "label_pred_xgb_norm" not in map_df.columns:
                    st.info("Colonne pred_xgb absente dans ce CSV d'inference.")
                    map_df = pd.DataFrame()
                else:
                    map_df["map_category"] = map_df["label_pred_xgb_norm"].map(_canonical_env_label)
                    map_color_map = LABEL_COLOR_MAP
                    legend_title = "Prediction XGBoost"
                    popup_fields = ["label_ref_norm", "label_pred_xgb_norm", "conf_xgb"]
            elif map_mode == "Prediction fusion":
                if "label_pred_final_norm" not in map_df.columns:
                    st.info("Colonne pred_final absente dans ce CSV d'inference.")
                    map_df = pd.DataFrame()
                else:
                    map_df["map_category"] = map_df["label_pred_final_norm"].map(_canonical_env_label)
                    map_color_map = LABEL_COLOR_MAP
                    legend_title = "Prediction fusion"
                    popup_fields = ["label_ref_norm", "label_pred_final_norm", "conf_final", "conf_gru", "conf_xgb"]
            
            elif map_mode == "Desaccord simple":
                map_df["map_category"] = map_df["is_disagreement"].map({0: "Accord", 1: "Desaccord"})
                map_color_map = {"Desaccord": "red", "Accord": "green"}
                legend_title = "Desaccord"
                popup_fields = ["label_ref_norm", "label_pred_gru_norm", "label_pred_xgb_norm", "label_pred_final_norm", "conf_gru", "conf_xgb", "conf_final"]
            elif map_mode == "Confiance": # dégradé entre min et max de conf_final
                if "conf_final" not in map_df.columns:
                    st.info("Colonne conf_final absente dans ce CSV d'inference.")
                    map_df = pd.DataFrame()
                else:
                    map_df["map_category"] = (map_df["conf_final"].round(1)).astype(str)  # Arrondi pour limiter le nombre de catégories distinctes
                    
                    colormap = color_map.get_cmap('plasma') # 'jet' ou 'viridis' pour une bonne lisibilité
                    keys = map_df["map_category"].unique()
                    map_color_map = {k: mcolors.to_hex(colormap(float(k))) for k in keys}  # Utilise la couleur hex directement depuis la colonne
                    legend_title = "Confiance prediction finale"
                    popup_fields = ["label_ref_norm", "label_pred_final_norm", "conf_final", "conf_gru", "conf_xgb"]
            else:
                risk_info = map_df.apply(
                    lambda row: _risk_info(row.get("label_ref_norm"), row.get("label_pred_final_norm", row.get("label_pred_norm"))),
                    axis=1,
                )
                risk_df = pd.DataFrame(list(risk_info))
                map_df = pd.concat([map_df.reset_index(drop=True), risk_df], axis=1)
                map_df["map_category"] = map_df["risk_label"]
                map_color_map = {
                    label: color
                    for label, color in map_df[["risk_label", "color"]].drop_duplicates().itertuples(index=False)
                }
                legend_title = "Risque de prediction"
                popup_fields = [
                    "label_ref_norm",
                    "label_pred_gru_norm",
                    "label_pred_xgb_norm",
                    "label_pred_final_norm",
                    "conf_gru",
                    "conf_xgb",
                    "conf_final",
                    "risk_text",
                    "risk_label",
                    "dangerous",
                    "description",
                ]

                dangerous_ratio = float((map_df["dangerous"] == "oui").mean()) if not map_df.empty else 0.0
                st.metric("Predictions dangereuses", f"{100.0 * dangerous_ratio:.1f}%")

            if not map_df.empty:
                map_df = map_df.dropna(subset=["map_category"])
            if not map_df.empty:
                map_mode_key = map_mode.lower().replace(" ", "_").replace("(", "").replace(")", "")
                render_categorical_segments_map(
                    map_df.reset_index(drop=True),
                    lat_col="latitude_gt",
                    lon_col="longitude_gt",
                    category_col="map_category",
                    category_color_map=map_color_map,
                    key_suffix=f"eval_{trajet_id}_{map_mode_key}",
                    title="Vue spatiale des labels / predictions / risque",
                    detail_points_label="Afficher des points detailles",
                    detail_points_default=True,
                    legend_title=legend_title,
                    gt_label="Trajet Ground Truth",
                    ordered_categories=None,
                    popup_fields=popup_fields,
                )

    with st.expander("Voir les points en desaccord", expanded=False):
        show_cols = [c for c in ["label_ref_norm", "label_pred_norm", t_col, lat_col, lon_col] if c is not None]
        st.dataframe(merged.loc[merged["is_disagreement"] == 1, show_cols].head(2000), width="stretch", hide_index=True)


render_page()