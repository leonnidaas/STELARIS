import os

import pandas as pd
import streamlit as st

from modules.analyses import render_analyses
from modules.cartographie import render_carte
from modules.lidar_slice_2d import render_lidar_slice_2d
from modules.viz_3d import render_viz_3d
from utils import INTERIM_DIR, get_traj_paths


st.set_page_config(page_title="Monitoring STELARIS", layout="wide", page_icon="📈")

st.sidebar.title("STELARIS")

trajets_disponibles = sorted(os.listdir(INTERIM_DIR))
if not trajets_disponibles:
	st.error("Aucun trajet disponible dans INTERIM.")
	st.stop()

trajet_id = st.sidebar.selectbox("Selectionner un trajet", trajets_disponibles)
st.title(f"Monitoring - Trajet: {trajet_id}")


@st.cache_data
def load_traj_config(traj_id_local):
	return get_traj_paths(traj_id_local)


@st.cache_data
def get_data(csv_path):
	df = pd.read_csv(csv_path)
	if "time_utc" in df.columns:
		df = df.sort_values("time_utc", kind="stable")
	return df


config = load_traj_config(trajet_id)
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
	render_viz_3d(trajet_id, lidar_dir, matched_df, gnss_offset=gnss_offset)
	st.divider()

if show_lidar_slice:
	render_lidar_slice_2d(trajet_id, lidar_dir, matched_df, gnss_offset_z=gnss_offset_z)
	st.divider()

if show_analyses:
	render_analyses(matched_df, trajet_id=trajet_id)
	st.divider()

if show_carte:
	render_carte(matched_df, trajet_id)
