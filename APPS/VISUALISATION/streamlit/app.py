from pathlib import Path

import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from streamlit.components.v1 import html as st_html

from utils import GT_DIR, INTERIM_DIR, PROCESSED_DIR, iter_gt_scenario_dirs, iter_scenario_dirs


st.set_page_config(page_title="STELARIS Control Center", layout="wide", page_icon="🛰️")


def _count_scenarios(path: Path) -> int:
    return len(iter_scenario_dirs(path))


def _count_csv_recursive(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.rglob("*.csv")))


LAT_ALIASES = ["latitude", "lat", "lat_gt", "latitude_gt", "y_coord", "latitude[deg]"]
LON_ALIASES = ["longitude", "lon", "long", "lon_gt", "longitude_gt", "x_coord", "longitude[deg]"]


def _pick_coord_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    cols = {str(c).strip().lower(): c for c in df.columns}

    lat_col = next((cols[a] for a in LAT_ALIASES if a in cols), None)
    lon_col = next((cols[a] for a in LON_ALIASES if a in cols), None)
    if lat_col is None or lon_col is None:
        return None
    return str(lat_col), str(lon_col)


def _find_coord_columns_in_txt(df: pd.DataFrame) -> tuple[int, int] | None:
    best_pair = None
    best_score = -1

    # On teste les paires de colonnes numeriques et on garde la plus plausible.
    for i in range(df.shape[1]):
        col_i = pd.to_numeric(df.iloc[:, i], errors="coerce")
        if col_i.notna().sum() < 10:
            continue
        for j in range(df.shape[1]):
            if j == i:
                continue
            col_j = pd.to_numeric(df.iloc[:, j], errors="coerce")
            if col_j.notna().sum() < 10:
                continue

            valid = col_i.notna() & col_j.notna()
            if valid.sum() < 10:
                continue

            # Score en orientation normale (lat, lon).
            score_normal = (
                (col_i.between(-90, 90))
                & (col_j.between(-180, 180))
                & valid
            ).sum()
            # Score en orientation inverse (lat <- col_j, lon <- col_i).
            score_swap = (
                (col_j.between(-90, 90))
                & (col_i.between(-180, 180))
                & valid
            ).sum()

            if score_normal >= score_swap:
                score = int(score_normal)
                pair = (i, j)
            else:
                score = int(score_swap)
                pair = (j, i)

            if score > best_score:
                best_score = score
                best_pair = pair

    if best_pair is None or best_score < 10:
        return None
    return best_pair


def _load_coords_from_gt_file(gt_file: Path, max_points: int = 3000) -> np.ndarray | None:
    try:
        if gt_file.suffix.lower() == ".csv":
            df = pd.read_csv(gt_file)
            picked = _pick_coord_columns(df)
            if picked is None:
                return None
            lat_col, lon_col = picked
            coords = df[[lat_col, lon_col]].copy()
        else:
            # Fallback texte Delph: detection automatique des colonnes lat/lon.
            df = pd.read_csv(gt_file, sep=r"\s+", comment="#", header=None)
            if df.shape[1] < 5:
                return None
            detected = _find_coord_columns_in_txt(df)
            if detected is None:
                return None
            lat_idx, lon_idx = detected
            coords = df.iloc[:, [lat_idx, lon_idx]].copy()
            coords.columns = ["latitude", "longitude"]

        coords = coords.apply(pd.to_numeric, errors="coerce").dropna()
        if len(coords) < 2:
            return None

        # Garde uniquement les valeurs plausibles.
        coords = coords[
            (coords.iloc[:, 0] >= -90)
            & (coords.iloc[:, 0] <= 90)
            & (coords.iloc[:, 1] >= -180)
            & (coords.iloc[:, 1] <= 180)
        ]
        if len(coords) < 2:
            return None

        # Downsample pour garder une page fluide.
        step = max(1, int(np.ceil(len(coords) / max_points)))
        arr = coords.iloc[::step].to_numpy(dtype=float)
        return arr
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _collect_all_routes(gt_dir: Path, max_points_per_trace: int = 3000):
    records = []

    for traj_dir in iter_gt_scenario_dirs(gt_dir):
        # Nom logique du trajet: avant "__" quand present.
        route_key = traj_dir.name.split("__", 1)[0]

        gt_files = list(traj_dir.glob("*.csv"))
        if not gt_files:
            continue
        
        coords = _load_coords_from_gt_file(gt_files[0], max_points=max_points_per_trace)
        if coords is None:
            continue

        records.append({"route_key": route_key, "coords": coords, "scenario": traj_dir.name})

    if not records:
        return [], {}

    route_counts: dict[str, int] = {}
    for rec in records:
        route_counts[rec["route_key"]] = route_counts.get(rec["route_key"], 0) + 1

    # Une seule trace representative par trajet pour alleger fortement le rendu.
    representative: dict[str, dict] = {}
    for rec in records:
        key = rec["route_key"]
        if key not in representative:
            representative[key] = rec

    route_records = list(representative.values())
    return route_records, route_counts


def _render_static_all_routes_map() -> None:
    st.subheader("Carte - Tous Les Traces")
    st.caption("Epaisseur du trait proportionnelle au nombre d'occurrences du meme trajet.")

    records, route_counts = _collect_all_routes(GT_DIR, max_points_per_trace=400)
    if not records:
        st.warning("Aucun trace exploitable trouve dans GROUNDTRUTH.")
        return

    max_count = max(route_counts.values()) if route_counts else 1
    cmap = plt.get_cmap("viridis")

    all_coords = np.vstack([rec["coords"] for rec in records])
    center_lat = float(np.nanmean(all_coords[:, 0]))
    center_lon = float(np.nanmean(all_coords[:, 1]))

    tile_choice = st.selectbox(
        "Fond de carte",
        ["OpenStreetMap", "CartoDB positron", "Satellite"],
        index=1,
        key="home_all_routes_tiles",
    )

    tile_config = {
        "CartoDB positron": {
            "tiles": "CartoDB positron",
            "attr": "&copy; OpenStreetMap contributors &copy; CARTO",
        },
        
    }

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles=tile_config[tile_choice]["tiles"],
        attr=tile_config[tile_choice]["attr"],
        control_scale=True,
        prefer_canvas=True,
    )
    ign_lidar_wms = "https://data.geopf.ign.fr/wms-r?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
    show_ign = st.checkbox("Afficher les tuiles LiDAR HD IGN (lentes)", key="home_show_ign_lidar")
    if show_ign:
        folium.WmsTileLayer(
            url=ign_lidar_wms,
            layers="LIDARHD_MAILLAGE_DISPO",
            name="COUVERTURE_LIDAR_HD_IGN",
            fmt="image/png",
            transparent=True,
            overlay=True,
            control=True,
        ).add_to(m)
    for rec in records:
        route_key = rec["route_key"]
        count = route_counts.get(route_key, 1)
        weight = 0.6 + 4.4 * (count / max_count)
        color = cmap(min(0.95, 0.25 + 0.75 * (count / max_count)))
        color_hex = mcolors.to_hex(color)

        lat = rec["coords"][:, 0]
        lon = rec["coords"][:, 1]
        points = np.column_stack([lat, lon]).tolist()
        folium.PolyLine(
            points,
            color=color_hex,
            weight=weight,
            opacity=0.55,
            tooltip=f"{route_key} | occurrences: {count}",
        ).add_to(m)

    try:
        st_folium(m, use_container_width=True, height=620, returned_objects=[])
    except Exception as e:
        st.warning(f"Rendu streamlit-folium indisponible, fallback HTML active ({e}).")
        st_html(m._repr_html_(), height=620, scrolling=False)

    top_routes = sorted(route_counts.items(), key=lambda x: x[1], reverse=True)
    st.caption(f"Trajets uniques affiches: {len(records)} | scenarios total: {sum(route_counts.values())}")
    st.write("Répétitions :")
    # on fait un histogrammes des trejats et de leur répétition pour voir les trajets les plus fréquents
    st.bar_chart(pd.DataFrame(top_routes[:30], columns=["trajet", "occurrences"]).set_index("trajet"))  

st.title("STELARIS Control Center")
st.caption("Accueil de l'application: supervision, lancement des pipelines et visualisation des trajets.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Trajets detectes", _count_scenarios(GT_DIR))
with col2:
    st.metric("Dossiers interim", _count_scenarios(INTERIM_DIR))
with col3:
    st.metric("CSV produits", _count_csv_recursive(PROCESSED_DIR))

st.divider()

st.subheader("Navigation")
st.write("Choisis une page dans la barre laterale ou utilise les raccourcis ci-dessous.")

i = 3
try:
    cols = st.columns(i)
    with cols[0]:
        st.page_link("pages/monitoring.py", label="Ouvrir Monitoring", icon="📈")
    with cols[1]:
        st.page_link("pages/pipeline_labelisation.py", label="Ouvrir Pilotage Pipelines", icon="⚙️")
    with cols[2]:
        st.page_link("pages/training.py", label="Ouvrir l'entrainement", icon="🤖")
except Exception:
    st.info("Utilise la barre laterale pour aller sur Monitoring ou Pilotage Pipelines.")

st.divider()

st.subheader("Demarrage rapide")
st.markdown(
    """
1. Lance la page Pilotage Pipelines pour executer une phase ou le pipeline complet.
2. Quand les CSV finaux sont generes, ouvre la page Monitoring pour analyser les trajets.
3. Utilise les modules 3D, Analyses et Carte pour le controle qualite.
"""
)

st.divider()
_render_static_all_routes_map()