from typing import Any

import streamlit as st
from LABELISATION_AUTO_LIDAR_HD_IGN.run_params import load_latest_labelisation_run_params
from VISUALISATION.STREAMLIT.services.pipeline_service import (
    APPS_ROOT,
    PHASES,
    build_trajet_scenarios_map,
    detect_csv_changes,
    expected_csv_outputs,
    kill_running_pipeline_processes,
    list_trajets,
    resolve_python_for_env,
    run_module,
    run_pipeline_complet,
    scenario_start_label,
    snapshot_mtimes,
)
from VISUALISATION.STREAMLIT.ui.theme import apply_theme, render_hero
from utils import CHUNK_SIZE, N_WORKERS, PARAMS_LABELISATION, get_traj_paths, merge_labelisation_params

st.set_page_config(page_title="Pilotage Pipelines", layout="wide", page_icon="⚙️")

LABEL_PARAM_SPECS: dict[str, dict[str, Any]] = {
    "distance_scan": {"kind": "int", "min": 1, "max": 200, "step": 1},
    "seuil_min_points_zone": {"kind": "int", "min": 1, "max": 10000, "step": 1},
    "seuil_bridge_above_count_min": {"kind": "int", "min": 0, "max": 1000, "step": 1},
    "bridge_min_score": {"kind": "int", "min": 1, "max": 10, "step": 1},
    "seuil_open_sky_soft_score": {"kind": "int", "min": 1, "max": 10, "step": 1},
}


USEFUL_LABEL_PARAMS_ORDER = [
    "seuil_vegetation",
    "seuil_ciel_ouvert",
    "distance_scan",
    "seuil_building_density",
    "seuil_build_zrel_p95",
    "seuil_overhead_bridge",
]


OSM_LABEL_PARAM_SPECS: dict[str, dict[str, Any]] = {
    "smooth_window": {"kind": "int", "min": 1, "max": 101, "step": 2},
}


def _ensure_state() -> None:
    if "pipeline_running" not in st.session_state:
        st.session_state.pipeline_running = False
    if "pending_run" not in st.session_state:
        st.session_state.pending_run = None
    if "last_resume" not in st.session_state:
        st.session_state.last_resume = []
    if "last_csv_changes" not in st.session_state:
        st.session_state.last_csv_changes = []
    if "last_run_phase_logs" not in st.session_state:
        st.session_state.last_run_phase_logs = []
    if "show_last_run_logs" not in st.session_state:
        st.session_state.show_last_run_logs = False
    if "last_run_logs_enabled" not in st.session_state:
        st.session_state.last_run_logs_enabled = False
    if "logs_enabled_for_run" not in st.session_state:
        st.session_state.logs_enabled_for_run = False


def render_page() -> None:
    apply_theme()
    _ensure_state()

    render_hero(
        "Pilotage des Pipelines",
        "Lancer un pipeline complet ou une selection de phases, pour un scenario ou un lot de scenarios.",
    )

    is_running = st.session_state.pipeline_running
    if is_running:
        st.warning("Pipeline en cours: les parametres sont verrouilles jusqu'a la fin de l'execution.")

    stop_col_a, stop_col_b = st.columns([1, 3])
    with stop_col_a:
        stop_clicked = st.button(
            "Arreter execution en cours",
            type="secondary",
            width="stretch",
            help="Envoie un signal d'arret aux scripts de pipeline lances depuis cette page.",
        )
    with stop_col_b:
        st.caption("Bouton d'urgence pour stopper le pipeline lance depuis cette page.")

    if stop_clicked:
        kill_info = kill_running_pipeline_processes()
        st.session_state.pipeline_running = False
        st.session_state.pending_run = None
        if kill_info["found"] == 0:
            st.info("Aucun process de pipeline detecte.")
        else:
            st.warning(
                "Arret demande: "
                f"{kill_info['found']} process detecte(s), "
                f"TERM={kill_info['terminated']}, KILL={kill_info['killed']}."
            )
        st.rerun()

    scenarios = list_trajets()
    if not scenarios:
        st.error("Aucun trajet trouve dans le dossier GroundTruth.")
        return

    trajets_map = build_trajet_scenarios_map(scenarios)
    trajets = sorted(trajets_map.keys())

    col_a, col_b = st.columns(2)
    with col_a:
        mode = st.radio(
            "Mode d'execution",
            ["Pipeline complet", "Selection de phases"],
            horizontal=True,
            disabled=is_running,
        )
    with col_b:
        cible = st.radio(
            "Cible",
            ["Un scenario", "Un scenario par trajet", "Tous les scenarios d'un trajet", "Selection manuelle"],
            horizontal=True,
            disabled=is_running,
        )

    phases_choisies: list[str] = []
    if mode == "Selection de phases":
        phases_choisies = st.multiselect(
            "Phases a executer",
            list(PHASES.keys()),
            default=["Traitement RINEX"],
            disabled=is_running,
        )

    if cible == "Un scenario":
        selected_trajet = st.selectbox("Trajet", trajets, disabled=is_running)
        scenarios_du_trajet = trajets_map.get(selected_trajet, [])
        selected_scenario = st.selectbox(
            "Scenario (debut)",
            scenarios_du_trajet,
            format_func=scenario_start_label,
            disabled=is_running,
        )
        selected_trajets = [selected_scenario]
    elif cible == "Un scenario par trajet":
        selected_trajets = []
        st.markdown("**Selectionnez un scenario pour chaque trajet**")
        selection = st.multiselect(
            "Premier scenario de chaque trajet",
            trajets,
            format_func=lambda tid: f"{tid} | {scenario_start_label(trajets_map.get(tid, [None])[0]) if trajets_map.get(tid) else 'Aucun scenario'}",
            disabled=is_running,
            )
        selected_trajets = [trajets_map[tid][0] for tid in selection if trajets_map.get(tid)]  # Ne garde que les trajets avec scenario
        
        if selected_trajets:
            st.info(f"{len(selected_trajets)} scenario(s) seront lances (un par trajet).")
        else:
            st.warning("Aucun trajet disponible avec des scenarios.")
    elif cible == "Tous les scenarios d'un trajet":
        selected_trajet = st.selectbox("Trajet", trajets, disabled=is_running)
        selected_trajets = trajets_map.get(selected_trajet, [])
        st.info(f"{len(selected_trajets)} scenario(s) seront lances pour le trajet {selected_trajet}.")
    else:
        selected_trajets = st.multiselect(
            "Scenarios",
            scenarios,
            default=scenarios,
            format_func=lambda sid: f"{sid.split('__', 1)[0]} | {scenario_start_label(sid)}",
            disabled=is_running,
        )

    stop_on_error = st.checkbox(
        "Arreter a la premiere erreur",
        value=True,
        disabled=is_running,
    )

    toggle_label = (
        "Desactiver logs pour le prochain run"
        if st.session_state.logs_enabled_for_run
        else "Activer logs pour le prochain run"
    )
    if st.button(toggle_label, type="secondary", disabled=is_running, width="stretch"):
        st.session_state.logs_enabled_for_run = not st.session_state.logs_enabled_for_run
        st.rerun()

    st.caption(
        "Logs prochain run: "
        + ("ACTIVES" if st.session_state.logs_enabled_for_run else "DESACTIVES")
    )
    max_live_lines = st.slider(
        "Nombre de lignes de logs live visibles",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
        disabled=is_running or (not st.session_state.logs_enabled_for_run),
    )

    show_lidar_params = mode == "Pipeline complet" or ("Pipeline Labelisation LiDAR" in phases_choisies)
    show_osm_params = mode == "Pipeline complet" or ("Pipeline Labelisation OSM" in phases_choisies)
    show_gnss_params = mode == "Pipeline complet" or ("Extraction features GNSS" in phases_choisies)

    label_workers = int(N_WORKERS)
    label_chunk_size = int(CHUNK_SIZE)
    label_verify_integrity = False
    label_extract_features = False
    label_spatial_mode = "circle"
    radius = 10.0
    corridor_width = 6.0
    corridor_length = 30.0
    min_elevation_angle_deg = 10.0
    step1_select_tiles = True
    step2_download_tiles = True
    step3_fusion_gt_gnss = False
    step4_extract_lidar = True
    step5_fusion_features = True
    step6_label = True
    step7_final_fusion = True
    labelisation_params_ui = merge_labelisation_params(PARAMS_LABELISATION)

    osm_step1_download_pbf = True
    osm_step2_filter_pbf = True
    osm_step3_extract_features = True
    osm_step4_labelisation = True
    osm_step5_final_fusion = True
    osm_buffer_m = 500.0
    osm_radius_m = 30.0
    osm_label_params_ui: dict[str, Any] = {"smooth_window": 5}

    gnss_cn0_smooth_window = 15
    gnss_cn0_quartile = 1

    if show_gnss_params:
        with st.expander("Parametres extraction features GNSS", expanded=False):
            gnss_cn0_smooth_window = st.slider(
                "Taille fenetre glissante CN0",
                min_value=1,
                max_value=101,
                value=15,
                step=2,
                disabled=is_running,
                help="Utilisee pour CN0_mean_smoothed, CN0_std_smoothed et les nouvelles features quartiles.",
            )
            gnss_cn0_quartile = st.selectbox(
                "Quartile satellites CN0 a selectionner",
                options=[1, 2, 3, 4],
                index=0,
                disabled=is_running,
                format_func=lambda q: f"Q{q}",
                help="Q1: 25% CN0 les plus faibles, Q4: 25% CN0 les plus eleves.",
            )

    if show_lidar_params:
        with st.expander("Parametres du pipeline Labelisation LiDAR", expanded=False):
            st.markdown("Sous-etapes a executer")
            step1_select_tiles = st.checkbox("Etape 1 - Selection des tuiles", value=True, disabled=is_running)
            step2_download_tiles = st.checkbox("Etape 2 - Telechargement des tuiles", value=True, disabled=is_running)
            step3_fusion_gt_gnss = st.checkbox(
                "Etape 3 - Fusion GT + GNSS (legacy)",
                value=False,
                disabled=is_running,
                help="Lance plutot la phase dediee 'Fusion GT + GNSS' pour une execution independante.",
            )
            step4_extract_lidar = st.checkbox("Etape 4 - Extraction features LiDAR", value=True, disabled=is_running)
            step5_fusion_features = st.checkbox("Etape 5 - Fusion features", value=True, disabled=is_running)
            step6_label = st.checkbox("Etape 6 - Labelisation", value=True, disabled=is_running)
            step7_final_fusion = st.checkbox("Etape 7 - Fusion finale", value=True, disabled=is_running)
            st.divider()

            label_workers = st.number_input(
                "Workers",
                min_value=1,
                max_value=64,
                value=int(N_WORKERS),
                step=1,
                disabled=is_running,
            )
            label_chunk_size = st.number_input(
                "Chunk size telechargement",
                min_value=1024,
                max_value=100_000_000,
                value=int(CHUNK_SIZE),
                step=1024,
                disabled=is_running,
            )
            label_verify_integrity = st.checkbox("Verifier l'integrite des tuiles", value=False, disabled=is_running)
            label_extract_features = st.checkbox(
                "Forcer la re-extraction des features LiDAR",
                value=True,
                disabled=is_running,
            )
            label_spatial_mode = st.selectbox(
                "Mode spatial pour la selection des tuiles",
                ["circle", "corridor"],
                index=0,
                help="Le mode 'corridor' est plus rapide et plus representatif de la zone d'interet.",
                disabled=is_running,
            )
            if label_spatial_mode == "circle":
                radius = st.slider(
                    "Rayon de recherche (m)",
                    min_value=5.0,
                    max_value=200.0,
                    value=80.0,
                    step=5.0,
                    disabled=is_running,
                )
            else:
                corridor_width = st.slider(
                    "Largeur couloir (m)",
                    min_value=5.0,
                    max_value=200.0,
                    value=6.0,
                    step=1.0,
                    disabled=is_running,
                )
                corridor_length = st.slider(
                    "Demi-longueur couloir (m)",
                    min_value=5.0,
                    max_value=300.0,
                    value=30.0,
                    step=1.0,
                    disabled=is_running,
                )

            min_elevation_angle_deg = st.slider(
                "Angle minimal au-dessus de l'horizontale antenne (deg)",
                min_value=0.0,
                max_value=45.0,
                value=10.0,
                step=1.0,
                disabled=is_running,
                help="Les points sous cet angle sont ignores pour l'extraction des features et la labellisation.",
            )

            st.divider()
            st.markdown("Parametres de labelisation (override pour ce run)")
            st.caption("Affichage volontairement reduit aux parametres utiles avec les features LiDAR compactes.")
            merged_params = merge_labelisation_params(PARAMS_LABELISATION)
            for param_key in USEFUL_LABEL_PARAMS_ORDER:
                if param_key not in merged_params:
                    continue
                default_val = merged_params[param_key]
                spec = LABEL_PARAM_SPECS.get(param_key, {"kind": "float", "min": 0.0, "max": 1000.0, "step": 0.01})
                widget_key = f"label_param_{param_key}"
                if spec["kind"] == "int":
                    val = st.number_input(
                        param_key,
                        min_value=int(spec["min"]),
                        max_value=int(spec["max"]),
                        value=int(default_val),
                        step=int(spec["step"]),
                        disabled=is_running,
                        key=widget_key,
                    )
                    labelisation_params_ui[param_key] = int(val)
                else:
                    val = st.number_input(
                        param_key,
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        value=float(default_val),
                        step=float(spec["step"]),
                        disabled=is_running,
                        key=widget_key,
                    )
                    labelisation_params_ui[param_key] = float(val)

    if show_osm_params:
        with st.expander("Parametres du pipeline Labelisation OSM", expanded=False):
            st.markdown("Sous-etapes a executer")
            osm_step1_download_pbf = st.checkbox("Etape 1 - Telechargement PBF OSM", value=True, disabled=is_running)
            osm_step2_filter_pbf = st.checkbox("Etape 2 - Filtrage spatial PBF", value=True, disabled=is_running)
            osm_step3_extract_features = st.checkbox("Etape 3 - Extraction features OSM", value=True, disabled=is_running)
            osm_step4_labelisation = st.checkbox("Etape 4 - Labelisation OSM", value=True, disabled=is_running)
            osm_step5_final_fusion = st.checkbox("Etape 5 - Fusion finale OSM", value=True, disabled=is_running)
            st.divider()

            osm_buffer_m = st.slider(
                "Buffer trajectoire pour filtre PBF (m)",
                min_value=50.0,
                max_value=2000.0,
                value=500.0,
                step=50.0,
                disabled=is_running,
            )
            osm_radius_m = st.slider(
                "Rayon des features OSM (m)",
                min_value=5.0,
                max_value=200.0,
                value=30.0,
                step=5.0,
                disabled=is_running,
            )

            st.divider()
            st.markdown("Parametres de labelisation OSM")
            for param_key, default_val in osm_label_params_ui.copy().items():
                spec = OSM_LABEL_PARAM_SPECS.get(param_key, {"kind": "int", "min": 1, "max": 101, "step": 2})
                widget_key = f"osm_label_param_{param_key}"
                val = st.number_input(
                    param_key,
                    min_value=int(spec["min"]),
                    max_value=int(spec["max"]),
                    value=int(default_val),
                    step=int(spec["step"]),
                    disabled=is_running,
                    key=widget_key,
                )
                osm_label_params_ui[param_key] = int(val)

    lidar_phase_params: dict[str, Any] = {
        "nb_workers": int(label_workers),
        "chunk_size": int(label_chunk_size),
        "verifier_integrite": bool(label_verify_integrity),
        "extract_features": bool(label_extract_features),
        "spatial_mode": str(label_spatial_mode),
        "search_radius": float(radius),
        "corridor_width": float(corridor_width),
        "corridor_length": float(corridor_length),
        "min_elevation_angle_deg": float(min_elevation_angle_deg),
        "run_step_1_select_tiles": bool(step1_select_tiles),
        "run_step_2_download_tiles": bool(step2_download_tiles),
        "run_step_3_fusion_gt_gnss": bool(step3_fusion_gt_gnss),
        "run_step_4_extract_lidar": bool(step4_extract_lidar),
        "run_step_5_fusion_features": bool(step5_fusion_features),
        "run_step_6_labelisation": bool(step6_label),
        "run_step_7_final_fusion": bool(step7_final_fusion),
        "params_labelisation": labelisation_params_ui,
        "verbose": True,
    }

    osm_phase_params: dict[str, Any] = {
        "run_step_1_download_pbf": bool(osm_step1_download_pbf),
        "run_step_2_filter_pbf": bool(osm_step2_filter_pbf),
        "run_step_3_extract_features": bool(osm_step3_extract_features),
        "run_step_4_labelisation": bool(osm_step4_labelisation),
        "run_step_5_final_fusion": bool(osm_step5_final_fusion),
        "buffer_m": float(osm_buffer_m),
        "radius_m": float(osm_radius_m),
        "params_labelisation": dict(osm_label_params_ui),
        "verbose": True,
    }

    phase_params: dict[str, Any] = {
        "Extraction features GNSS": {
            "cn0_smooth_window": int(gnss_cn0_smooth_window),
            "cn0_quartile": int(gnss_cn0_quartile),
            "verbose": True,
        },
        "Pipeline Labelisation LiDAR": lidar_phase_params,
        "Pipeline Labelisation OSM": osm_phase_params,
    }

    if show_lidar_params and selected_trajets:
        preview_traj = selected_trajets[0]
        try:
            preview_cfg = get_traj_paths(preview_traj)
            latest_path, latest_payload = load_latest_labelisation_run_params(preview_cfg, preview_traj, source="IGN")
            with st.expander("Dernier JSON de parametres (interim)", expanded=False):
                if latest_payload is None:
                    st.info("Aucun fichier de parametres precedent trouve dans l'interim pour ce trajet.")
                else:
                    st.caption(f"Source: {latest_path}")
                    st.json(latest_payload)
        except Exception as e:
            st.info(f"Impossible de charger le dernier JSON de parametres: {e}")

    lancer = st.button("Lancer", type="primary", width="stretch", disabled=is_running)
    if lancer and not is_running:
        st.session_state.last_run_phase_logs = []
        st.session_state.show_last_run_logs = False
        st.session_state.last_run_logs_enabled = bool(st.session_state.logs_enabled_for_run)
        st.session_state.pending_run = {
            "mode": mode,
            "phases_choisies": phases_choisies,
            "selected_trajets": selected_trajets,
            "stop_on_error": stop_on_error,
            "logs_enabled_for_run": bool(st.session_state.logs_enabled_for_run),
            "max_live_lines": int(max_live_lines),
            "phase_params": phase_params,
        }
        st.session_state.pipeline_running = True
        st.rerun()

    if not is_running:
        if st.session_state.last_resume:
            st.divider()
            st.subheader("Resume")
            st.dataframe(st.session_state.last_resume, width="stretch")

            st.subheader("CSV crees / modifies")
            if st.session_state.last_csv_changes:
                st.dataframe(st.session_state.last_csv_changes, width="stretch")
            else:
                st.info("Aucun CSV cree ou modifie detecte pour cette execution.")

        if st.session_state.last_run_phase_logs and st.session_state.last_run_logs_enabled:
            label = "Masquer les logs du dernier run" if st.session_state.show_last_run_logs else "Voir les logs du dernier run"
            if st.button(label, type="secondary", width="stretch"):
                st.session_state.show_last_run_logs = not st.session_state.show_last_run_logs
                st.rerun()

            if st.session_state.show_last_run_logs:
                st.markdown("Logs du dernier run (par trajet puis phase)")
                for traj_id, phase_name, phase_ok, logs in st.session_state.last_run_phase_logs:
                    status = "OK" if phase_ok else "ERROR"
                    with st.expander(f"{traj_id} | {phase_name} ({status})"):
                        st.code(logs or "(aucun log)")

        return

    run_cfg = st.session_state.pending_run
    if not run_cfg:
        st.session_state.pipeline_running = False
        return

    mode = run_cfg["mode"]
    phases_choisies = run_cfg.get("phases_choisies", [])
    selected_trajets = run_cfg["selected_trajets"]
    stop_on_error = run_cfg["stop_on_error"]
    logs_enabled_for_run = bool(run_cfg.get("logs_enabled_for_run", False))
    max_live_lines = int(run_cfg.get("max_live_lines", 300))
    phase_params = run_cfg["phase_params"]

    if not selected_trajets:
        st.warning("Selectionne au moins un trajet.")
        st.session_state.pipeline_running = False
        st.session_state.pending_run = None
        return

    if mode == "Selection de phases" and not phases_choisies:
        st.warning("Selectionne au moins une phase.")
        st.session_state.pipeline_running = False
        st.session_state.pending_run = None
        return

    st.info(f"Execution depuis: {APPS_ROOT}")
    st.caption(
        "Interpreteurs detectes - RINEX: "
        f"{resolve_python_for_env('rinex')} | Label/GNSS: {resolve_python_for_env('label')}"
    )

    resume = []
    all_csv_changes = []
    run_phase_logs: list[tuple[str, str, bool, str]] = []
    progress = st.progress(0, text=f"trajet 1/{len(selected_trajets)}") 

    try:
        for idx, traj_id in enumerate(selected_trajets, start=1):
            with st.container(border=True):
                st.subheader(f"Trajet: {traj_id}")

                if mode == "Pipeline complet":
                    expected_paths = []
                    for phase in PHASES:
                        expected_paths.extend(expected_csv_outputs(traj_id, phase))
                    before = snapshot_mtimes(expected_paths)

                    phase_log_placeholders = None
                    for phase_name in PHASES:
                        st.markdown(f"⏳ {phase_name}")
                    if logs_enabled_for_run:
                        # Single live area: reset at each phase like the training page.
                        live_placeholder = st.empty()
                        phase_log_placeholders = {phase_name: live_placeholder for phase_name in PHASES}

                    ok, details = run_pipeline_complet(
                        traj_id,
                        stop_on_error=stop_on_error,
                        phase_log_placeholders=phase_log_placeholders,
                        phase_params=phase_params,
                        max_live_lines=max_live_lines,
                    )

                    if logs_enabled_for_run:
                        for phase_name, phase_ok, phase_logs in details:
                            run_phase_logs.append((traj_id, phase_name, phase_ok, phase_logs))

                    for phase_name, phase_ok, _logs in details:
                        icon = "✅" if phase_ok else "❌"
                        st.markdown(f"{icon} {phase_name}")
                    csv_changes = detect_csv_changes(before)
                    for item in csv_changes:
                        item["trajet"] = traj_id
                        item["phase"] = "ALL"
                    all_csv_changes.extend(csv_changes)

                    resume.append(
                        {
                            "trajet": traj_id,
                            "mode": mode,
                            "phase": "ALL",
                            "status": "OK" if ok else "ERROR",
                            "csv_changes": len(csv_changes),
                        }
                    )
                else:
                    ordered_phases = [phase_name for phase_name in PHASES if phase_name in phases_choisies]
                    live_placeholder = st.empty() if logs_enabled_for_run else None
                    for phase_name in ordered_phases:
                        expected_paths = expected_csv_outputs(traj_id, phase_name)
                        before = snapshot_mtimes(expected_paths)

                        st.markdown(f"⏳ {phase_name}")
                        ok, _logs = run_module(
                            phase_name,
                            traj_id,
                            log_placeholder=live_placeholder,
                            phase_params=phase_params,
                            max_live_lines=max_live_lines,
                        )
                        if logs_enabled_for_run:
                            run_phase_logs.append((traj_id, phase_name, ok, _logs))
                        icon = "✅" if ok else "❌"
                        st.markdown(f"{icon} {phase_name}")

                        csv_changes = detect_csv_changes(before)
                        for item in csv_changes:
                            item["trajet"] = traj_id
                            item["phase"] = phase_name
                        all_csv_changes.extend(csv_changes)

                        resume.append(
                            {
                                "trajet": traj_id,
                                "mode": mode,
                                "phase": phase_name,
                                "status": "OK" if ok else "ERROR",
                                "csv_changes": len(csv_changes),
                            }
                        )

                        if not ok and stop_on_error:
                            break

            next_idx = min(idx + 1, len(selected_trajets))
            progress.progress(idx / len(selected_trajets), text=f"trajet {next_idx}/{len(selected_trajets)}")

    finally:
        st.session_state.pipeline_running = False
        st.session_state.pending_run = None

    st.session_state.last_resume = resume
    st.session_state.last_csv_changes = all_csv_changes
    st.session_state.last_run_phase_logs = run_phase_logs
    st.session_state.last_run_logs_enabled = logs_enabled_for_run
    st.rerun()


render_page()
