import os
import re

import pandas as pd
import streamlit as st

from modules.analyses import render_analyses
from modules.cartographie import render_carte
from modules.lidar_slice_2d import render_lidar_slice_2d
from modules.viz_3d import render_viz_3d
from utils import get_traj_paths, list_traj_ids


st.set_page_config(page_title="Monitoring STELARIS", layout="wide", page_icon="📈")

st.sidebar.title("STELARIS")

scenarios_disponibles = list_traj_ids()
if not scenarios_disponibles:
	st.error("Aucun trajet disponible dans INTERIM.")
	st.stop()

trajets_map: dict[str, list[str]] = {}
for scenario_id in scenarios_disponibles:
	trajet_key = scenario_id.split("__", 1)[0] if "__" in scenario_id else scenario_id
	trajets_map.setdefault(trajet_key, []).append(scenario_id)

trajets_disponibles = sorted(trajets_map.keys())
trajet_id = st.sidebar.selectbox("Selectionner un trajet", trajets_disponibles)


def _scenario_start_label(sid: str) -> str:
	"""Retourne un libelle court avec la date/heure de debut du scenario."""
	m = re.search(r"__SCENARIO_(\d{8})_(\d{6}(?:\.\d+)?)", sid)
	if not m:
		return sid

	date_raw = m.group(1)
	time_raw = m.group(2)
	date_fmt = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
	time_fmt = f"{time_raw[:2]}:{time_raw[2:4]}:{time_raw[4:]}"
	return f"{date_fmt} {time_fmt}"


scenarios_trajet = sorted(trajets_map.get(trajet_id, []))
scenario_id = st.sidebar.selectbox(
	"Selectionner un scenario (debut)",
	scenarios_trajet,
	format_func=_scenario_start_label,
)

st.title(f"Monitoring - Trajet: {trajet_id}")
st.caption(f"Scenario: {scenario_id}")


@st.cache_data
def load_traj_config(traj_id_local):
	return get_traj_paths(traj_id_local)


@st.cache_data
def get_data(csv_path):
	df = pd.read_csv(csv_path)
	if "time_utc" in df.columns:
		df = df.sort_values("time_utc", kind="stable")
	return df


config = load_traj_config(scenario_id)
csv_file = config["final_fusion_csv"]
lidar_dir = config["lidar_tiles"]
gnss_offset = config["gnss_offset"]
gnss_offset_z = float(gnss_offset[2]) 

if not os.path.exists(csv_file):
	st.warning(f"Fichier final introuvable: {csv_file}")
	st.info("Lance d'abord le pipeline de labelisation pour ce trajet.")
	st.stop()

matched_df = get_data(csv_file)

st.sidebar.divider()
st.sidebar.subheader("Modules a afficher")
show_viz3d = st.sidebar.checkbox("Visualisation 3D", value=True)
show_lidar_slice = st.sidebar.checkbox("Coupe 2D LiDAR (1s)", value=True)
show_analyses = st.sidebar.checkbox("Analyses", value=False)
show_carte = st.sidebar.checkbox("Carte 2D", value=False)

if not any([show_viz3d, show_lidar_slice, show_analyses, show_carte]):
	st.info("Selectionne au moins un module dans la barre laterale.")

if show_viz3d:
	render_viz_3d(scenario_id, lidar_dir, matched_df, gnss_offset=gnss_offset)
	st.divider()

if show_lidar_slice:
	render_lidar_slice_2d(scenario_id, lidar_dir, matched_df, gnss_offset_z=gnss_offset_z)
	st.divider()

if show_analyses:
	render_analyses(matched_df, trajet_id=scenario_id)
	st.divider()

if show_carte:
	render_carte(matched_df, scenario_id)
