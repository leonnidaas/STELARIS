from pathlib import Path
from timeit import main

import folium
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html
from streamlit_folium import st_folium
from services.home_service import (
    collect_all_routes,
    count_csv_recursive,
    count_scenarios,
    count_unique_routes,
    estimate_total_km_from_files,
    load_grid_data,
    calculate_total_duration_from_files,
)
from ui.theme import apply_theme, render_hero
from utils import GT_DIR, INTERIM_DIR, PROCESSED_DIR

st.set_page_config(page_title="STELARIS", layout="wide", page_icon="🚀")

GRID_PATH = Path("/media/leon_peltzer/DATA/leon/STELARIS/DATA/MAILLAGE_LIDAR_HD.json")


@st.cache_data(show_spinner=False)
def _collect_all_routes_cached(gt_dir: Path, sample_stride: int = 100) -> list[dict[str, object]]:
    return collect_all_routes(gt_dir, sample_stride=sample_stride)


@st.cache_data(show_spinner=False)
def _load_grid_data_cached(path: Path) -> dict | None:
    return load_grid_data(path)


def _add_lidar_grid_layer(m: folium.Map, grid_data: dict | None) -> None:
    if not grid_data:
        st.warning("Grille LiDAR indisponible: fichier manquant ou invalide.")
        return

    folium.GeoJson(
        grid_data,
        name="Grille LiDAR HD",
        style_function=lambda _: {
            "fillColor": "#ff7800",
            "fillOpacity": 0.18,
            "color": "#0058d3",
            "weight": 1,
            "opacity": 0.35,
        },
        tooltip=folium.GeoJsonTooltip(fields=["id", "name"], aliases=["ID", "Name"]),
    ).add_to(m)


def _render_kpis() -> None:
    # 2 cols 2 lignes pour les KPIs

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Scenarios detectes", count_scenarios(GT_DIR))
    with col2:
        st.metric("Tajets uniques", count_unique_routes(INTERIM_DIR))
    with col3:
        res = 5  # Correspond à une segmentation d'environ 5km par segment
        st.metric(f"Km de lignes uniques +/- {res}km", estimate_total_km_from_files(PROCESSED_DIR, res=res))
    with col4:
        st.metric("Heures de trajet cumulees", calculate_total_duration_from_files(PROCESSED_DIR))

def _render_navigation() -> None:
    st.subheader("Navigation")
    st.caption("Accede rapidement aux modules principaux.")

    cols = st.columns(3)
    with cols[0]:
        if st.button("📈 Ouvrir Monitoring", use_container_width=True):
            st.switch_page("pages/monitoring.py")
    with cols[1]:
        if st.button("⚙️ Ouvrir Pilotage Pipelines", use_container_width=True):
            st.switch_page("pages/pipeline_labelisation.py")
    with cols[2]:
        if st.button("🤖 Ouvrir Entrainement", use_container_width=True):
            st.switch_page("pages/training.py")


def _render_quick_start() -> None:
    st.subheader("Demarrage rapide")
    st.markdown(
        """
        <div class="section-card">
            <ol class="quick-start">
                <li>Lance Pilotage Pipelines pour executer une phase ou le pipeline complet.</li>
                <li>Une fois les CSV generes, ouvre Monitoring pour analyser les trajets.</li>
                <li>Utilise les modules 3D, Analyses et Carte pour valider la qualite des donnees.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_all_routes_map() -> None:
    st.subheader("Carte - Toutes les traces")
    sample_stride = 500
    st.caption(f"Affichage de toutes les lignes avec un echantillonnage de 1/{sample_stride}.")

    records = _collect_all_routes_cached(GT_DIR, sample_stride=sample_stride)
    if not records:
        st.warning("Aucune trace exploitable trouvee dans GROUNDTRUTH.")
        return

    cmap = plt.get_cmap("viridis")

    all_coords = np.vstack([rec["coords"] for rec in records])
    center_lat = float(np.nanmean(all_coords[:, 0]))
    center_lon = float(np.nanmean(all_coords[:, 1]))

    unique_lines = sorted({str(rec["route_key"]) for rec in records})
    line_index = {k: i for i, k in enumerate(unique_lines)}

    controls_col1, controls_col2 = st.columns([2, 2])
    with controls_col1:
        tile_choice = st.selectbox(
            "Fond de carte",
            ["OpenStreetMap", "CartoDB positron", "Satellite"],
            index=1,
            key="home_all_routes_tiles",
        )
    with controls_col2:
        show_ign = st.checkbox("Afficher les tuiles LiDAR HD IGN", key="home_show_ign_lidar")

    tile_config = {
        "OpenStreetMap": {
            "tiles": "OpenStreetMap",
            "attr": "&copy; OpenStreetMap contributors",
        },
        "CartoDB positron": {
            "tiles": "CartoDB positron",
            "attr": "&copy; OpenStreetMap contributors &copy; CARTO",
        },
        "Satellite": {
            "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "attr": "Tiles &copy; Esri",
        },
    }

    map_config = tile_config[tile_choice]
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles=map_config["tiles"],
        attr=map_config["attr"],
        control_scale=True,
        prefer_canvas=True,
    )

    if show_ign:
        _add_lidar_grid_layer(m, _load_grid_data_cached(GRID_PATH))

    for rec in records:
        route_key = str(rec["route_key"])
        idx = line_index[route_key]
        ratio = idx / max(1, len(unique_lines) - 1)
        color_hex = mcolors.to_hex(cmap(0.15 + 0.80 * ratio))

        coords = rec["coords"]
        points = np.column_stack([coords[:, 0], coords[:, 1]]).tolist()

        folium.PolyLine(
            points,
            color=color_hex,
            weight=2.2,
            opacity=0.68,
            tooltip=f"Ligne: {route_key} | Scenario: {rec['scenario']}",
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    try:
        st_folium(m, use_container_width=True, height=620, returned_objects=[])
    except Exception as e:
        st.warning(f"Rendu streamlit-folium indisponible, fallback HTML actif ({e}).")
        st_html(m._repr_html_(), height=620, scrolling=False)

    line_counts: dict[str, int] = {}
    for rec in records:
        key = str(rec["route_key"])
        line_counts[key] = line_counts.get(key, 0) + 1

    st.caption(
        f"Scenarios utilises: {len(records)} | lignes physiques: {len(unique_lines)} | "
        f"echantillonnage: 1/{sample_stride}"
    )

    top_lines = sorted(line_counts.items(), key=lambda x: x[1], reverse=True)
    st.write("Nombre de scenarios affiches par ligne")
    st.bar_chart(pd.DataFrame(top_lines[:30], columns=["ligne", "scenarios"]).set_index("ligne") , sort="-scenarios")


def render_page() -> None:
    apply_theme()
    render_hero(
        "STELARIS Control Center",
        "Supervision des scenarios, pilotage des pipelines et controle qualite des trajectoires GNSS.",
    )
    _render_kpis()

    st.divider()
    _render_navigation()

    st.divider()
    _render_quick_start()

    st.divider()
    _render_all_routes_map()

    st.markdown("---")
    st.caption("STELARIS v2.3 | Donnees issues de la Geoplateforme IGN | 2026")

if __name__ == "__main__":
    render_page()
