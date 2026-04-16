import sys
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="STELARIS", layout="wide", page_icon="🚀")


MODULE_PREFIXES_TO_UNLOAD = (
	"services",
	"modules",
	"ui",
	"utils",
	"ENTRAINEMENT_MODELES",
	"EXTRACTION_DES_FEATURES_GNSS",
	"FUSION",
	"LABELISATION_AUTO_LIDAR_HD_IGN",
	"TELECHARGEMENT_SCENARIO",
	"TRAITEMENT_RINEX",
)


def _should_unload_module(module_name: str) -> bool:
	for prefix in MODULE_PREFIXES_TO_UNLOAD:
		if module_name == prefix or module_name.startswith(f"{prefix}."):
			return True
	return False


def _reload_external_state() -> int:
	st.cache_data.clear()
	st.cache_resource.clear()

	unloaded = 0
	for module_name in list(sys.modules):
		if _should_unload_module(module_name):
			del sys.modules[module_name]
			unloaded += 1
	return unloaded


def _render_global_reload_button() -> None:
	with st.sidebar:
		st.divider()
		st.subheader("Maintenance")
		if st.button("Recharger CSV + code externe", use_container_width=True):
			unloaded = _reload_external_state()
			st.session_state["_last_manual_reload"] = {
				"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				"modules": unloaded,
			}
			st.rerun()

		info = st.session_state.get("_last_manual_reload")
		if info:
			st.caption(
				f"Dernier rechargement: {info['ts']} | modules decharges: {info['modules']}"
			)




home = st.Page("pages/home.py", title="Accueil", icon="🏠", default=True)
pipeline = st.Page("pages/pipeline_labelisation.py", title="Pipeline de labelisation", icon="⚙️")
monitoring = st.Page("pages/monitoring.py", title="Tableau de bord", icon="📊")
training = st.Page("pages/training.py", title="Entrainement des modeles", icon="🧠")
training_results = st.Page("pages/training_results.py", title="Resultats entrainement", icon="📉")

_render_global_reload_button()
nav = st.navigation([home, pipeline, monitoring, training, training_results])
nav.run()
