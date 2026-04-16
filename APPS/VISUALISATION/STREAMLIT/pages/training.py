from datetime import datetime
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from VISUALISATION.STREAMLIT.services.training_service import (
    APPS_ROOT,
    TRAINING_ROOT,
    TRAINING_SCRIPTS,
    artefact_changes,
    build_artefacts_map,
    build_trajet_groups,
    expand_unique_trajets,
    extract_classification_reports,
    kill_running_training_processes,
    list_trajets,
    parse_eval_metrics,
    run_script,
    snapshot_artefacts,
    trajet_unique_id,
)
from VISUALISATION.STREAMLIT.ui.theme import apply_theme, render_hero
from utils import PARAMS_ENTRAINEMENT, TRAINING_DIR, get_traj_paths


st.set_page_config(page_title="Training IA", layout="wide", page_icon="🧠")


def _ensure_state() -> None:
    if "training_running" not in st.session_state:
        st.session_state.training_running = False
    if "training_results" not in st.session_state:
        st.session_state.training_results = []
    if "training_artefact_changes" not in st.session_state:
        st.session_state.training_artefact_changes = []


@st.cache_data(show_spinner=False)
def _list_previous_datasets() -> list[str]:
    if not TRAINING_DIR.exists():
        return []

    candidates: list[tuple[float, str]] = []
    for d in TRAINING_DIR.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / f"{d.name}_metadata.json"
        if not meta_path.exists():
            continue
        candidates.append((meta_path.stat().st_mtime, d.name))

    candidates.sort(reverse=True)
    return [name for _, name in candidates]


@st.cache_data(show_spinner=False)
def _load_dataset_trajets(dataset_id: str) -> list[str]:
    meta_path = TRAINING_DIR / dataset_id / f"{dataset_id}_metadata.json"
    if not meta_path.exists():
        return []

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return []

    trajets = metadata.get("source_data", {}).get("trajets", [])
    if not isinstance(trajets, list):
        return []
    return [str(t) for t in trajets]


@st.cache_data(show_spinner=False)
def _load_dataset_features(dataset_id: str) -> list[str]:
    meta_path = TRAINING_DIR / dataset_id / f"{dataset_id}_metadata.json"
    if not meta_path.exists():
        return []

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return []

    features = metadata.get("features", [])
    if not isinstance(features, list):
        return []
    return [str(feat) for feat in features]


@st.cache_data(show_spinner=False)
def _list_available_gnss_features(trajets_key: tuple[str, ...]) -> list[str]:
    available_sets: list[set[str]] = []
    for trajet_id in trajets_key:
        try:
            csv_path = get_traj_paths(trajet_id)["final_fusion_csv"]
            cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        except Exception:
            continue

        gnss_cols = {c for c in cols if str(c).startswith("gnss_feat_")}
        if gnss_cols:
            available_sets.append(gnss_cols)

    if not available_sets:
        return []

    common = set.intersection(*available_sets)
    return sorted(common)


def render_page() -> None:
    apply_theme()
    _ensure_state()

    render_hero(
        "Training des modeles IA",
        "Pilotage du preprocessing, de l'optimisation GRU (Optuna), de l'entrainement et de l'evaluation.",
    )

    is_running = st.session_state.training_running
    if is_running:
        st.warning("Entrainement en cours, les parametres sont verrouilles.")

    stop_col_a, stop_col_b = st.columns([1, 3])
    with stop_col_a:
        stop_clicked = st.button(
            "Arreter execution en cours",
            type="secondary",
            width="stretch",
            help="Envoie un signal d'arret aux scripts ENTRAINEMENT_MODELES en cours.",
        )
    with stop_col_b:
        st.caption("Bouton d'urgence pour stopper le pipeline lance depuis cette page.")

    if stop_clicked:
        kill_info = kill_running_training_processes()
        st.session_state.training_running = False
        if kill_info["found"] == 0:
            st.info("Aucun process d'entrainement detecte.")
        else:
            st.warning(
                "Arret demande: "
                f"{kill_info['found']} process detecte(s), "
                f"TERM={kill_info['terminated']}, KILL={kill_info['killed']}."
            )
        st.rerun()

    left, right = st.columns([2, 1])
    with left:
        dataset_name = st.text_input(
            "Nom du dataset",
            value=datetime.now().strftime("%Y-%m-%d_%H-%M"),
            disabled=is_running,
            help="Le meme dataset_name sera utilise pour preprocessing, train et evaluation.",
        )

        default_steps = [
            "Preprocessing",
            "Entrainement GRU",
            "Entrainement XGBoost",
            "Evaluation",
        ]
        default_steps = [s for s in default_steps if s in TRAINING_SCRIPTS]

        selected_steps = st.multiselect(
            "Etapes a executer (ordre fixe)",
            options=list(TRAINING_SCRIPTS.keys()),
            default=default_steps,
            disabled=is_running,
        )

    trajets_disponibles = list_trajets()
    grouped_trajets = build_trajet_groups(trajets_disponibles)
    trajets_uniques = sorted(grouped_trajets.keys())

    default_trajets = PARAMS_ENTRAINEMENT.get("trajets", ["BORDEAUX_COUTRAS", "MARTINE_01"])
    default_unique = sorted({trajet_unique_id(t) for t in default_trajets if trajet_unique_id(t) in grouped_trajets})
    if not default_unique:
        default_unique = trajets_uniques[:2] if len(trajets_uniques) >= 2 else trajets_uniques

    reuse_previous = st.checkbox(
        "Recuperer les memes scenarios d'un entrainement precedent",
        value=False,
        disabled=is_running,
        help="Remplace la selection actuelle par les trajets enregistres dans le metadata d'un dataset precedent.",
    )

    selected_trajets: list[str] = []
    if reuse_previous:
        previous_datasets = [d for d in _list_previous_datasets() if d != dataset_name]
        if not previous_datasets:
            st.info("Aucun dataset precedent avec metadata source_data.trajets n'a ete trouve.")
        else:
            selected_previous_dataset = st.selectbox(
                "Dataset precedent a reutiliser",
                options=previous_datasets,
                disabled=is_running,
            )
            previous_trajets = _load_dataset_trajets(selected_previous_dataset)

            if not previous_trajets:
                st.warning("Aucun trajet trouve dans le metadata de ce dataset precedent.")
                selected_trajets = []
            else:
                selected_trajets = [t for t in previous_trajets if t in trajets_disponibles]
                missing = [t for t in previous_trajets if t not in trajets_disponibles]

                st.caption(
                    f"{len(selected_trajets)} scenario(s) recuperes depuis {selected_previous_dataset}."
                )
                if missing:
                    st.warning(
                        f"{len(missing)} scenario(s) absents de l'environnement courant ont ete ignores."
                    )
    else:
        selection_mode = st.radio(
            "Mode de selection des donnees preprocessing",
            options=["Trajets uniques", "Scenarios manuels"],
            horizontal=True,
            disabled=is_running,
        )

        if selection_mode == "Trajets uniques":
            unique_pick_mode = st.radio(
                "Mode de selection des scenarios par trajet unique",
                options=["Tous les scenarios", "Premier scenario uniquement"],
                horizontal=True,
                disabled=is_running,
            )

            selected_uniques = st.multiselect(
                "Trajets uniques",
                options=trajets_uniques,
                default=default_unique,
                disabled=is_running,
                format_func=lambda t: f"{t} ({len(grouped_trajets.get(t, []))} scenarios)",
                help="Chaque trajet unique ajoute automatiquement tous ses scenarios.",
            )
            if unique_pick_mode == "Tous les scenarios":
                selected_trajets = expand_unique_trajets(selected_uniques, grouped_trajets)
            else:
                selected_trajets = [
                    grouped_trajets[t][0]
                    for t in selected_uniques
                    if grouped_trajets.get(t)
                ]
            st.caption(f"{len(selected_trajets)} scenario(s) seront utilises pour le preprocessing.")
        else:
            default_scenarios = [t for t in default_trajets if t in trajets_disponibles]
            if not default_scenarios:
                default_scenarios = trajets_disponibles[:2] if len(trajets_disponibles) >= 2 else trajets_disponibles

            selected_trajets = st.multiselect(
                "Scenarios pour creer le dataset (preprocessing)",
                options=trajets_disponibles,
                default=default_scenarios,
                disabled=is_running,
                help="Utilise uniquement ces scenarios lors de l'etape Preprocessing.",
            )

    st.subheader("Features GNSS (Preprocessing)")
    gnss_candidates = _list_available_gnss_features(tuple(sorted(selected_trajets))) if selected_trajets else []

    if not gnss_candidates:
        st.info("Aucune feature GNSS commune detectee pour les scenarios selectionnes.")
        selected_features: list[str] = []
    else:
        default_cfg_features = [
            str(f) for f in PARAMS_ENTRAINEMENT.get("features", [])
            if str(f).startswith("gnss_feat_")
        ]
        default_features = [f for f in default_cfg_features if f in gnss_candidates]
        if not default_features:
            default_features = gnss_candidates[: min(6, len(gnss_candidates))]

        reuse_feature_set = st.checkbox(
            "Reutiliser les memes features qu'un dataset precedent",
            value=False,
            disabled=is_running,
            help="Charge la liste des features sauvegardees dans le metadata d'un dataset precedent.",
        )

        if reuse_feature_set:
            feature_datasets = [d for d in _list_previous_datasets() if d != dataset_name]
            if not feature_datasets:
                st.info("Aucun dataset precedent trouve pour recharger des features.")
                selected_features = []
            else:
                previous_features_dataset = st.selectbox(
                    "Dataset precedent (features)",
                    options=feature_datasets,
                    disabled=is_running,
                )
                previous_features = [
                    f for f in _load_dataset_features(previous_features_dataset)
                    if f.startswith("gnss_feat_")
                ]
                reused_features = [f for f in previous_features if f in gnss_candidates]
                missing_features = [f for f in previous_features if f not in gnss_candidates]

                if reused_features:
                    selected_features = reused_features
                    st.caption(
                        f"{len(reused_features)} feature(s) GNSS rechargee(s) depuis {previous_features_dataset}."
                    )
                else:
                    selected_features = []
                    st.warning("Aucune des features GNSS du dataset precedent n'est disponible ici.")

                if missing_features:
                    st.warning(
                        f"{len(missing_features)} feature(s) absente(s) dans les scenarios actuels ont ete ignorees."
                    )

            if selected_features:
                with st.expander("Voir les features reutilisees", expanded=False):
                    st.code("\n".join(selected_features), language="text")

            st.caption(
                f"{len(selected_features)} feature(s) reutilisee(s) sur {len(gnss_candidates)} disponible(s)."
            )

        else:
            selected_features = st.multiselect(
                "Features GNSS a utiliser",
                options=gnss_candidates,
                default=default_features,
                disabled=is_running,
                help="Ces colonnes seront passees a l'etape Preprocessing via --features.",
            )

            st.caption(f"{len(selected_features)} feature(s) selectionnee(s) sur {len(gnss_candidates)} disponible(s).")

    with right:
        continue_on_error = st.checkbox(
            "Continuer si erreur",
            value=False,
            disabled=is_running,
            help="Si active, les etapes suivantes sont lancees meme en cas d'echec.",
        )

        with st.expander("Parametres Optuna GRU", expanded=False):
            optuna_n_trials = st.number_input(
                "Nombre de trials",
                min_value=5,
                max_value=500,
                value=20,
                step=5,
                disabled=is_running,
            )
            optuna_epochs = st.number_input(
                "Epochs max par trial",
                min_value=10,
                max_value=500,
                value=60,
                step=10,
                disabled=is_running,
            )
            optuna_batch_size = st.number_input(
                "Batch size",
                min_value=4,
                max_value=256,
                value=16,
                step=4,
                disabled=is_running,
            )
            optuna_patience = st.number_input(
                "Patience early stopping",
                min_value=3,
                max_value=100,
                value=8,
                step=1,
                disabled=is_running,
            )
            optuna_timeout = st.number_input(
                "Timeout (sec, 0 = illimite)",
                min_value=0,
                max_value=86400,
                value=1800,
                step=60,
                disabled=is_running,
            )
            optuna_pruner = st.selectbox(
                "Pruner",
                options=["hyperband", "median"],
                index=0,
                disabled=is_running,
            )
            optuna_n_jobs = st.number_input(
                "Parallel jobs (n_jobs)",
                min_value=1,
                max_value=8,
                value=1,
                step=1,
                disabled=is_running,
                help="Sur GPU unique, garder 1. >1 est surtout utile pour CPU/multi-workers.",
            )
            optuna_train_subsample = st.slider(
                "Sous-echantillon train (ratio)",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.05,
                disabled=is_running,
            )
            optuna_val_subsample = st.slider(
                "Sous-echantillon val (ratio)",
                min_value=0.1,
                max_value=1.0,
                value=1.0,
                step=0.05,
                disabled=is_running,
            )
            optuna_min_delta = st.number_input(
                "Early stopping min_delta",
                min_value=0.0,
                max_value=0.01,
                value=0.0001,
                step=0.0001,
                format="%.4f",
                disabled=is_running,
            )
            optuna_mixed_precision = st.checkbox(
                "Activer mixed precision (GPU)",
                value=True,
                disabled=is_running,
            )
            st.caption(
                "Conseil: pour un premier run, garder 20 trials / 60 epochs / timeout 1800s."
            )

    window_size = st.number_input(
        "Window size (preprocessing)",
        min_value=3,
        max_value=121,
        value=int(PARAMS_ENTRAINEMENT.get("window_size", 5)),
        step=2,
        disabled=is_running,
    )

    max_live_lines = st.slider(
        "Lignes de logs visibles",
        min_value=50,
        max_value=1200,
        value=300,
        step=50,
        disabled=is_running,
    )

    st.subheader("Etat des artefacts")
    artefacts_map = build_artefacts_map(dataset_name)
    artefact_rows = [
        {
            "Artefact": name,
            "Present": "Oui" if path.exists() else "Non",
            "Chemin": str(path),
        }
        for name, path in artefacts_map.items()
    ]
    st.dataframe(pd.DataFrame(artefact_rows), width="stretch", hide_index=True)

    run_clicked = st.button(
        "Lancer l'entrainement",
        type="primary",
        disabled=is_running,
        width="stretch",
    )

    if run_clicked:
        if not selected_steps:
            st.error("Selectionne au moins une etape.")
        elif "Preprocessing" in selected_steps and not selected_trajets:
            st.error("Selectionne au moins un trajet pour le preprocessing.")
        elif "Preprocessing" in selected_steps and not selected_features:
            st.error("Selectionne au moins une feature GNSS pour le preprocessing.")
        elif not TRAINING_ROOT.exists():
            st.error(f"Dossier introuvable: {TRAINING_ROOT}")
        else:
            st.session_state.training_running = True
            st.session_state.training_results = []
            st.session_state.training_artefact_changes = []

            artefacts_before = snapshot_artefacts(artefacts_map)

            step_args_map = {
                "Preprocessing": [
                    "--dataset-name",
                    dataset_name,
                    "--window-size",
                    str(int(window_size)),
                    "--features",
                    *selected_features,
                    "--trajets",
                    *selected_trajets,
                ],
                "Optimisation GRU (Optuna)": [
                    "--dataset-name",
                    dataset_name,
                    "--n-trials",
                    str(int(optuna_n_trials)),
                    "--epochs",
                    str(int(optuna_epochs)),
                    "--batch-size",
                    str(int(optuna_batch_size)),
                    "--patience",
                    str(int(optuna_patience)),
                    "--timeout",
                    str(int(optuna_timeout)),
                    "--pruner",
                    str(optuna_pruner),
                    "--n-jobs",
                    str(int(optuna_n_jobs)),
                    "--train-subsample-ratio",
                    str(float(optuna_train_subsample)),
                    "--val-subsample-ratio",
                    str(float(optuna_val_subsample)),
                    "--min-delta",
                    str(float(optuna_min_delta)),
                    "--mixed-precision" if optuna_mixed_precision else "--no-mixed-precision",
                ],
                "Entrainement GRU": ["--dataset-name", dataset_name],
                "Entrainement XGBoost": ["--dataset-name", dataset_name],
                "Evaluation": ["--dataset-name", dataset_name],
            }

            progress = st.progress(0, text="Demarrage du pipeline...")
            log_placeholder = st.empty()
            total = len(selected_steps)

            results = []
            global_ok = True
            for idx, step in enumerate(selected_steps, start=1):
                progress.progress(
                    int((idx - 1) / total * 100),
                    text=f"Execution [{idx}/{total}] {step}",
                )

                script_name = TRAINING_SCRIPTS[step]
                script_args = step_args_map.get(step, [])
                ok, logs = run_script(
                    script_name,
                    script_args=script_args,
                    log_placeholder=log_placeholder,
                    max_live_lines=max_live_lines,
                )

                results.append((step, ok, logs))
                if not ok:
                    global_ok = False
                    if not continue_on_error:
                        break

            st.session_state.training_results = results
            st.session_state.training_artefact_changes = artefact_changes(artefacts_before, artefacts_map)
            st.session_state.training_running = False

            progress.progress(100, text="Execution terminee")
            if global_ok:
                st.success("Pipeline training termine avec succes.")
            else:
                st.error("Pipeline termine avec au moins une erreur.")

    st.divider()
    st.subheader("Resultats")

    results = st.session_state.training_results
    if not results:
        st.info("Aucune execution lancee dans cette session.")
        return

    resume_rows = [
        {"Etape": step, "Statut": "OK" if ok else "ECHEC"}
        for step, ok, _ in results
    ]
    st.dataframe(pd.DataFrame(resume_rows), width="stretch", hide_index=True)

    changes = st.session_state.training_artefact_changes
    if changes:
        st.markdown("**Artefacts crees ou modifies**")
        st.dataframe(pd.DataFrame(changes), width="stretch", hide_index=True)

    eval_logs = ""
    for step, ok, logs in results:
        if step == "Evaluation" and ok:
            eval_logs = logs
            break

    if eval_logs:
        st.markdown("**Metriques d'evaluation**")
        metrics_df = parse_eval_metrics(eval_logs)
        if not metrics_df.empty:
            pivot_df = metrics_df.pivot_table(
                index=["Modele", "Split"],
                columns="Metrique",
                values="Valeur",
                aggfunc="first",
            ).reset_index()
            st.dataframe(pivot_df, width="stretch", hide_index=True)
        else:
            st.warning("Impossible d'extraire automatiquement les metriques depuis les logs.")

        reports = extract_classification_reports(eval_logs)
        for model_name, report in reports.items():
            with st.expander(f"Classification report - {model_name}", expanded=False):
                st.code(report)

    st.markdown("**Logs par etape**")
    for step, ok, logs in results:
        icon = "✅" if ok else "❌"
        with st.expander(f"{icon} {step}", expanded=not ok):
            st.code(logs if logs else "(aucun log)")

    st.caption(f"Execution racine: {APPS_ROOT}")

if __name__ == "__main__":
    render_page()
