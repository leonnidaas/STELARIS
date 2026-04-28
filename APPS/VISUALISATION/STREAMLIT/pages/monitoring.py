import os

import streamlit as st

from modules.analyses import render_analyses
from modules.cartographie import render_carte_monitoring
from modules.correlation_monitoring import render_correlation_module
from modules.lidar_slice_2d import render_lidar_slice_2d
from modules.monitoring_breakdown import render_monitoring_breakdown
from modules.viz_3d import render_viz_3d
from services.monitoring_service import (
    build_scenario_sources_zip,
    build_trajets_map,
    list_scenarios,
    load_latest_labelisation_params,
    load_sorted_csv,
    load_traj_config,
    scenario_start_label,
)
from ui.theme import apply_theme, render_hero

st.set_page_config(page_title="Monitoring STELARIS", layout="wide", page_icon="📈")


@st.cache_data(show_spinner=False)
def _load_traj_config_cached(traj_id_local: str) -> dict:
    return load_traj_config(traj_id_local)


@st.cache_data(show_spinner=False)
def _load_data_cached(csv_path: str):
    return load_sorted_csv(csv_path)


@st.cache_data(show_spinner=False)
def _load_latest_labelisation_params_cached(traj_id_local: str) -> dict:
    return load_latest_labelisation_params(traj_id_local)


@st.cache_data(show_spinner=False)
def _build_scenario_sources_zip_cached(traj_id_local: str) -> tuple[bytes | None, str, list[str]]:
    return build_scenario_sources_zip(traj_id_local)


def render_page() -> None:
    apply_theme()
    st.sidebar.title("STELARIS")

    scenarios_disponibles = list_scenarios()
    if not scenarios_disponibles:
        st.error("Aucun trajet disponible dans INTERIM.")
        st.stop()

    trajets_map = build_trajets_map(scenarios_disponibles)
    trajets_disponibles = sorted(trajets_map.keys())
    trajet_id = st.sidebar.selectbox("Selectionner un trajet", trajets_disponibles)

    scenarios_trajet = sorted(trajets_map.get(trajet_id, []))
    scenario_id = st.sidebar.selectbox(
        "Selectionner un scenario (debut)",
        scenarios_trajet,
        format_func=scenario_start_label,
    )

    st.sidebar.divider()
    # bouton de telechargement enrouge
    st.sidebar.subheader("Telechargement sources")
    zip_bytes, zip_name, missing_sources = _build_scenario_sources_zip_cached(scenario_id)
    if zip_bytes is None:
        st.sidebar.warning("Aucun fichier source telechargeable pour ce scenario.")
    else:
        st.sidebar.download_button(
            "Telecharger GT + RINEX (OBS/NAV)",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            use_container_width=True,
            icon="📥",
        )

    if missing_sources:
        st.sidebar.caption(f"Sources manquantes: {', '.join(missing_sources)}")

    render_hero(
        f"Monitoring - Trajet: {trajet_id}",
        f"Scenario actif: {scenario_id}",
    )

    config = _load_traj_config_cached(scenario_id)
    csv_file = config["final_fusion_csv"]
    lidar_dir = config["lidar_tiles"]
    gnss_offset = config["gnss_offset"]
    space_vehicule_info_csv = config.get("space_vehicule_info")
    gnss_offset_z = float(gnss_offset[2])

    if not os.path.exists(csv_file):
        st.warning(f"Fichier final introuvable: {csv_file}")
        st.info("Lance d'abord le pipeline de labelisation pour ce trajet.")
        st.stop()

    matched_df = _load_data_cached(str(csv_file)).reset_index(drop=True)
    labelisation_params = _load_latest_labelisation_params_cached(scenario_id)

    render_monitoring_breakdown(
        matched_df=matched_df,
        labelisation_params=labelisation_params,
        key_suffix=scenario_id,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Modules a afficher")
    show_viz3d = st.sidebar.checkbox("Visualisation 3D", value=False)
    show_lidar_slice = st.sidebar.checkbox("Coupe 2D LiDAR (1s)", value=False)
    show_analyses = st.sidebar.checkbox("Analyses", value=False)
    show_carte = st.sidebar.checkbox("Carte 2D", value=False)
    show_corr = st.sidebar.checkbox("Correlation / Autocorrelation", value=False)

    if not any([show_viz3d, show_lidar_slice, show_analyses, show_carte, show_corr]):
        st.info("Selectionne au moins un module dans la barre laterale.")

    if show_viz3d:
        render_viz_3d(scenario_id, lidar_dir, matched_df, gnss_offset=gnss_offset)
        st.divider()

    if show_lidar_slice:
        render_lidar_slice_2d(
            scenario_id,
            lidar_dir,
            matched_df,
            gnss_offset_z=gnss_offset_z,
            space_vehicule_info_csv=space_vehicule_info_csv,
        )
        st.divider()

    if show_analyses:
        render_analyses(matched_df, trajet_id=scenario_id)
        st.divider()

    if show_carte:
        render_carte_monitoring(matched_df, scenario_id)

    if show_corr:
        render_correlation_module(matched_df, trajet_id=scenario_id)
        st.divider()


render_page()
