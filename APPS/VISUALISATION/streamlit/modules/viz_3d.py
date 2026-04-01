import streamlit as st
import os
import pandas as pd
from VISUALISATION.VISUALISATION_3D.visualiseur import LidarVisualizer
import datetime
import threading
import queue
import time
import inspect
from datetime import timedelta


def _run_viz_worker(lidar_dir, matched_df, methode_fn, method_args, result_queue):
    try:
        viz = LidarVisualizer(lidar_dir, matched_df)
        method_args = dict(method_args)

        def _on_flythrough_tick(payload):
            result_queue.put(("tick", payload))

        if methode_fn == "show_corridor_flythrough":
            target_fn = getattr(viz, methode_fn)
            sig = inspect.signature(target_fn)
            if "progress_callback" in sig.parameters:
                method_args["progress_callback"] = _on_flythrough_tick

        getattr(viz, methode_fn)(**method_args)
        result_queue.put(("success", "Visualisation terminée."))
    except Exception as exc:
        result_queue.put(("error", f"Erreur pendant la visualisation: {exc}"))


def _get_time_bounds(matched_df):
    """Calcule une seule fois la plage horaire pour limiter le cout des reruns."""
    t = pd.to_datetime(matched_df['time_utc'], errors='coerce')
    t = t.dropna()
    if t.empty:
        raise ValueError("Aucun timestamp valide dans la colonne time_utc.")
    return t.min().to_pydatetime().time(), t.max().to_pydatetime().time()

def render_viz_3d(trajet_id, LIDAR_DIR, matched_df, gnss_offset):
    # Etats persistants de lancement asynchrone
    if 'viz_running' not in st.session_state:
        st.session_state.viz_running = False
    if 'viz_thread' not in st.session_state:
        st.session_state.viz_thread = None
    if 'viz_result_queue' not in st.session_state:
        st.session_state.viz_result_queue = queue.Queue()
    if 'viz_feedback' not in st.session_state:
        st.session_state.viz_feedback = None
    if 'viz_live_time' not in st.session_state:
        st.session_state.viz_live_time = None
    if 'viz_live_progress' not in st.session_state:
        st.session_state.viz_live_progress = 0.0
    if 'viz_time_bounds' not in st.session_state:
        st.session_state.viz_time_bounds = {}

    time_range_state_key = f"viz3d_time_range_{trajet_id}"

    # Preset temporel provenant de la coupe 2D (session_state partage entre modules)
    time_preset = st.session_state.get('viz3d_time_preset')
    preset_start_time = None
    preset_end_time = None
    if isinstance(time_preset, dict) and time_preset.get('trajet_id') == trajet_id and time_preset.get('mode') == 'time':
        try:
            preset_start_time = pd.to_datetime(time_preset.get('start')).time()
            preset_end_time = pd.to_datetime(time_preset.get('end')).time()
        except Exception:
            preset_start_time = None
            preset_end_time = None

    # Récupérer les retours du thread de fond
    while not st.session_state.viz_result_queue.empty():
        status, message = st.session_state.viz_result_queue.get()
        if status == "tick" and isinstance(message, dict):
            st.session_state.viz_live_time = message.get("time_utc")
            st.session_state.viz_live_progress = float(message.get("progress", 0.0))
            continue
        st.session_state.viz_running = False
        st.session_state.viz_feedback = (status, message)

    # Garde-fou si le thread est fini sans message (arrêt inattendu)
    if st.session_state.viz_thread is not None and not st.session_state.viz_thread.is_alive():
        if st.session_state.viz_running:
            st.session_state.viz_running = False
            st.session_state.viz_feedback = ("warning", "Visualisation arrêtée.")
        st.session_state.viz_thread = None

    # --- CSS CORRECTIF (Plus agressif) ---
    st.title(f"Visualisation 3D :")
    col1, col2 = st.columns([4, 3])

    with col1:
        
        # Paramètres spécifiques à la 3D dans le corps ou la sidebar
        width = st.slider("Largeur du corridor (m)", 10, 200, 50)
        factor = st.slider("Sous-échantillonnage", 1, 20, value=1)
        point_size = st.slider("Taille des points", 1.0, 5.0, 2.0, 0.5)
        hide_wires = st.checkbox("Masquer les fils / non classé (classes 0-1)", value=False)
        
        methode = st.radio("Méthode de visualisation", ["Fly-through", "Corridor statique"])
        if methode == "Fly-through":
            st.info("Le mode Fly-through vous permet de naviguer à travers le nuage de points comme si " \
            "vous étiez à bord du train. Utilisez les commandes clavier pour avancer, reculer, tourner, et ajuster la vue.")
            methode_fn = "show_corridor_flythrough"
        else:
            st.info("Le mode Corridor statique affiche un tronçon fixe du nuage de points autour de la trajectoire.\n"
            "\n⚠️ Attention le corridor statique peut être plus lourd à afficher pour de longues sections. veuillez ne pas selectioner plus de 30 mins de trajet sans sous-échantillonner (factor > 1) pour éviter les problèmes de performance.")
            methode_fn = "show_corridor"

        # --- PARAMÈTRES AVANCÉS FLY-THROUGH (affichés uniquement si sélectionné) ---   
        viewer_options = {}
        if methode_fn == "show_corridor_flythrough":
            with st.expander("Paramètres avancés Fly-through", expanded=False):
                camera_radius = st.slider("Rayon de chargement caméra (m)", 20.0, 300.0, 150.0, 5.0)
                refresh_distance = st.slider("Distance de rafraîchissement (m)", 5.0, 120.0, 50.0, 1.0)
                camera_height = st.slider("Hauteur caméra offset (m)", 0.0, 8.0, 2.2, 0.1)
                lookahead_distance = st.slider("Distance de visée (m)", 1.0, 30.0, 5.0, 0.5)
                lookahead_points = st.slider("Lookahead points", 1, 50, 12)
                step_size = st.slider("Pas du fly-through", 0.01, 1.0, 0.04, 0.01)
                camera_pitch_deg = st.slider("Pitch caméra (°)", -45.0, 20.0, -20.0, 1.0)
                camera_fov_deg = st.slider("Champ de vision FOV (°)", 40, 90, 90)
                flythrough_frame_delay = st.slider("Frame delay (s)", 0.001, 0.2, 0.05, 0.001)
                camera_near_clip = st.slider("Near clip (m)", 0.01, 1.0, 0.05, 0.01)
                pause_focus_distance = st.slider("Distance focus pause (m)", 0.005, 0.5, 0.02, 0.005)
                camera_anchor_mode = st.selectbox("Ancrage caméra", ["offset", "antenna"])
                camera_view_mode = st.selectbox("Mode de visée", ["forward", "sky"])

                viewer_options = {
                    "camera_radius": camera_radius,
                    "refresh_distance": refresh_distance,
                    "camera_height": camera_height,
                    "lookahead_distance": lookahead_distance,
                    "lookahead_points": lookahead_points,
                    "step_size": step_size,
                    "camera_pitch_deg": camera_pitch_deg,
                    "camera_fov_deg": camera_fov_deg,
                    "flythrough_frame_delay": flythrough_frame_delay,
                    "camera_near_clip": camera_near_clip,
                    "pause_focus_distance": pause_focus_distance,
                    "camera_anchor_mode": camera_anchor_mode,
                    "camera_view_mode": camera_view_mode,
                    "point_size": point_size,
                }
            
        mode_selection = st.radio("Mode de filtrage", ["Temporel", "Pourcentage"])
        valid_inputs = True


        # --- FILTRAGE DES DONNÉES SELON LE MODE SÉLECTIONNÉ ---
        if mode_selection == "Temporel":
            try :
                bounds_key = f"{trajet_id}::{len(matched_df)}"
                if bounds_key not in st.session_state.viz_time_bounds:
                    st.session_state.viz_time_bounds[bounds_key] = _get_time_bounds(matched_df)
                min_time, max_time = st.session_state.viz_time_bounds[bounds_key]
                st.info(f"Plage disponible : {min_time} - {max_time}")
                pas_de_temps = st.selectbox("Pas de temps (s)", options=[1, 5, 10, 15, 20, 30, 60], index=2)
                default_start, default_end = min_time, max_time

                saved_range = st.session_state.get(time_range_state_key)
                if (
                    isinstance(saved_range, (tuple, list))
                    and len(saved_range) == 2
                    and min_time <= saved_range[0] <= max_time
                    and min_time <= saved_range[1] <= max_time
                    and saved_range[1] > saved_range[0]
                ):
                    default_start, default_end = saved_range[0], saved_range[1]

                if preset_start_time is not None and preset_end_time is not None:
                    default_start = max(min_time, preset_start_time)
                    default_end = min(max_time, preset_end_time)
                    if default_end <= default_start:
                        default_start, default_end = min_time, max_time
                    st.session_state['viz3d_time_preset'] = None

                range_val = st.slider(
                    "Plage du trajet (HH:MM:SS)",
                      min_time, max_time   , 
                      value=(default_start, default_end)  ,
                      step=timedelta(seconds=pas_de_temps),
                      format="HH:mm:ss")
                st.session_state[time_range_state_key] = range_val
                start, end = range_val
                time_start = start
                time_end = end
                start , end = str(start), str(end)
                if time_end <= time_start :
                    valid_inputs = False
                    st.error("L'heure de début doit être strictement inférieure à l'heure de fin.")
                elif time_start < min_time or time_end > max_time:
                    valid_inputs = False
                    st.error("Les heures doivent être dans la plage disponible.")
                mode = "time"
            
            except ValueError as e:
                valid_inputs = False
                st.error(f"Format d'heure invalide. Utilisez HH:MM:SS. Détail de l'erreur : {e}")
        else:
            range_val = st.slider("Plage du trajet (%)", 0.0, 100.0, [0.0, 100.0] , step=0.5)
            start, end = range_val[0]/100, range_val[1]/100
            if end == start:
                valid_inputs = False
                st.error("L'intervalle de pourcentage est vide.")
            mode = "percent"


        # --- BOUTON DE LANCEMENT ---
        button_col = st.container()
        if st.session_state.viz_running:
            st.info("Visualisation 3D en cours... vous pouvez continuer à utiliser la page.")
            live_time = st.session_state.viz_live_time
            if live_time:
                st.metric("Heure fly-through (UTC)", f"{live_time}")
            else:
                st.caption("Heure fly-through (UTC): initialisation...")

        if button_col.button(
            "LANCER LA VISUALISATION", 
            icon="🚀",
            key="launch_btn",
            disabled=st.session_state.viz_running,
            help="Le bouton sera désactivé pendant l'exécution de la visualisation"
        ):
            if not valid_inputs:
                st.warning("Corrigez les paramètres avant de lancer la visualisation.")
            else:
                method_args = {
                    "start": start,
                    "end": end,
                    "mode": mode,
                    "width": width,
                    "factor": factor,
                    "hide_wires": hide_wires,
                    "gnss_offset": gnss_offset,
                    "point_size": point_size,
                }
                if methode_fn == "show_corridor_flythrough":
                    method_args.update(viewer_options)
                    # Moins de ticks vers Streamlit pour limiter la contention/GIL.
                    method_args.setdefault("flythrough_time_refresh_s", 1.0)

                st.session_state.viz_feedback = None
                st.session_state.viz_live_time = None
                st.session_state.viz_live_progress = 0.0
                st.session_state.viz_running = True
                st.session_state.viz_thread = threading.Thread(
                    target=_run_viz_worker,
                    args=(LIDAR_DIR, matched_df.copy(), methode_fn, method_args, st.session_state.viz_result_queue),
                    daemon=True,
                )
                st.session_state.viz_thread.start()
            st.rerun()
        
        if st.session_state.viz_feedback:
            status, message = st.session_state.viz_feedback
            if status == "success":
                st.success(message)
            elif status == "error":
                st.error(message)
            else:
                st.warning(message)

        if st.session_state.viz_running:
            # Rafraîchissement léger pour afficher le temps fly-through en quasi temps réel.
            
            time.sleep(1.0)
            st.rerun()

    with col2:
        st.subheader("📋 Légende & Commandes")
        def render_legend():
            # Affichage de la légende propre
            legend_data = [
                ("Sol ", "#804019", "Code 2"),
                ("Végétation", "#00FA00", "Codes 3-5"),
                ("Bâtiments", "#FA0000", "Code 6"),
                ("Tablier de Ponts", "#FA8000", "Code 17"),
                ("Non-classé", "#808080", "Code 1"),
                ("Eau", "#2B87FFFF", "Code 9"),
                ("Poteaux / Caténaires", "#FA00FA", "Codes 64"),
                ("Trajectoire", "#000000", "Ligne Noire")
            ]
            
            for label, color, code in legend_data:
                st.markdown(f"""
                    <div class="legend-box" style="background-color: {color}; padding: 5px; margin-bottom: 5px; border-radius: 3px; font-weight: bold;">
                        <span style="background-color: #FFFFFF; margin-left: 6px; border-radius: 12px; padding: 3px 10px ; ">{label}</span>
                        <span style="float:right; opacity:0.9; margin-right: 5px; color: #000000;">{code}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.divider()
            st.subheader("⌨️ Commandes Clavier")
            st.write("""
            [NAVIGATION]
            - **Z (W) / S** : Avancer / Reculer
            - **Q (A) / D** : Gauche / Droite
            - **R / F**     : Monter / Descendre
            - **M**         : Reset la vue
            - **L**         : Arrêt du fly-through

            [MODES]
            - **C**         : Switch Couleur (Classe / Altitude)
            - **T**         : Lancer un fly-through le long de la trajectoire
            - **5 / 6**     : Near clip proche / loin
            - **, / .**     : Fly-through plus rapide / plus lent
            - **V**         : Toggle visée (avant / ciel)
            - **B**         : Toggle ancrage (offset / antenne)
            - **Echap**     : Quitter
            """)
        render_legend()