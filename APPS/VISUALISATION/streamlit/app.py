from pathlib import Path

import streamlit as st

from utils import GT_DIR, INTERIM_DIR, PROCESSED_DIR


st.set_page_config(page_title="STELARIS Control Center", layout="wide", page_icon="🛰️")


def _count_dirs(path: Path) -> int:
    if not path.exists():
        return 0
    return len([p for p in path.iterdir() if p.is_dir()])


def _count_csv_recursive(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.rglob("*.csv")))


st.title("STELARIS Control Center")
st.caption("Accueil de l'application: supervision, lancement des pipelines et visualisation des trajets.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Trajets detectes", _count_dirs(GT_DIR))
with col2:
    st.metric("Dossiers interim", _count_dirs(INTERIM_DIR))
with col3:
    st.metric("CSV produits", _count_csv_recursive(PROCESSED_DIR))

st.divider()

st.subheader("Navigation")
st.write("Choisis une page dans la barre laterale ou utilise les raccourcis ci-dessous.")

try:
    c1, c2 = st.columns(2)
    with c1:
        st.page_link("pages/monitoring.py", label="Ouvrir Monitoring", icon="📈")
    with c2:
        st.page_link("pages/pipeline_labelisation.py", label="Ouvrir Pilotage Pipelines", icon="⚙️")
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