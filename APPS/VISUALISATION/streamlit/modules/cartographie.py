import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import branca.colormap as bcm

def render_carte(df, traj_id):
    st.title(f"🗺️ Cartographie Interactive :")
    st.markdown("Visualisez la précision du GNSS et l'impact de l'environnement sur la carte.")
    # --- FILTRES DE COULEUR ---
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Configuration")
        # Sélection de la métrique pour la couleur
        metrics = {
            "Erreur Latérale (m)": "err_laterale",
            "Erreur Longitudinale (m)": "err_longitudinale",
            "Masque de Ciel (°)": "sky_mask_deg",
            "Densité Végétation": "veg_density",
        }
        selection = st.selectbox("Colorer le trajet par :", list(metrics.keys()))
        target_col = metrics[selection]

        TILE_CONFIG = {"Satellite": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
        }, 
        "OpenStreetMap": {
            "tiles": "OpenStreetMap",
            "attr": "&copy; OpenStreetMap contributors"
        }, 
        "CartoDB positron": {
            "tiles": "CartoDB positron",
            "attr": "&copy; CartoDB"
        }}
        # Options de la carte
        map_style = st.selectbox(
            "Fond de carte",
            ["Aucun", "OpenStreetMap", "CartoDB positron", "Satellite"],
            index=3,
            help="Le fond de carte dépend d'un serveur distant et peut être lent."
        )
        show_gnss = st.checkbox("Afficher la trace GNSS brute", value=False)
        max_points = st.slider("Points max à afficher", 500, 10000, 3000, 500)
        show_detail_points = st.checkbox("Afficher des points détaillés (plus lent)", value=True)
        color_scale_mode = st.selectbox(
            "Échelle de couleur",
            ["Robuste (P2-P98)", "Min-Max"],
            index=0,
            help="Le mode robuste ignore les valeurs extrêmes pour mieux répartir les couleurs.",
        )

    # --- PRÉPARATION DES DONNÉES ---
    # Nettoyage de base pour éviter les coordonnées invalides dans Folium
    df_plot = df.copy()
    df_plot = df_plot.replace([np.inf, -np.inf], np.nan)

    df_gt = df_plot.dropna(subset=['latitude_gt', 'longitude_gt', target_col]).copy()
    if df_gt.empty:
        st.error("Aucune coordonnée Ground Truth valide à afficher.")
        return
    # Création de l'échelle de couleur (Colormap)
    values = df_gt[target_col].astype(float).dropna()
    data_min = float(values.min())
    data_max = float(values.max())

    if color_scale_mode == "Robuste (P2-P98)":
        min_val = float(np.percentile(values, 2))
        max_val = float(np.percentile(values, 98))
    else:
        min_val = data_min
        max_val = data_max

    if max_val <= min_val:
        max_val = min_val + 1e-9

    norm = colors.Normalize(vmin=min_val, vmax=max_val, clip=True)
    colormap = cm.get_cmap('jet') # 'jet' ou 'viridis' pour une bonne lisibilité

    # --- CRÉATION DE LA CARTE ---
    with col2:
        location=[df_gt['latitude_gt'].mean(), df_gt['longitude_gt'].mean()]
        # Centrer sur le trajet
        tile_choice = None if map_style == "Aucun" else map_style
        m = folium.Map(
            location=location,
            zoom_start=15,
            tiles=TILE_CONFIG[tile_choice]["tiles"] if tile_choice else None,
            prefer_canvas=True,
            control_scale=True,
            attr=TILE_CONFIG[tile_choice]["attr"] if tile_choice else None
        )

        # 1. Ajout de la ligne de base (Ground Truth) en gris discret
        points_gt = df_gt[['latitude_gt', 'longitude_gt']].values.tolist()
        if len(points_gt) >= 2:
            folium.PolyLine(points_gt, color="#333333", weight=2, opacity=0.5).add_to(m)

        # 2. Légende métrique (barre de couleur) utilisée aussi pour la ColorLine
        metric_colormap = bcm.LinearColormap(
            colors=['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000'],
            vmin=min_val,
            vmax=max_val,
            caption=selection,
        )
        metric_colormap.add_to(m)

        # 3. Tracé coloré optimisé en une seule couche (beaucoup plus fluide au pan/zoom)
        step = max(1, int(np.ceil(len(df_gt) / max_points)))
        df_render = df_gt.iloc[::step].reset_index(drop=True)
        if len(df_render) >= 2:
            traj_positions = df_render[['latitude_gt', 'longitude_gt']].values.tolist()
            traj_values = df_render[target_col].astype(float).tolist()
            folium.ColorLine(
                positions=traj_positions,
                colors=traj_values,
                colormap=metric_colormap,
                weight=4,
                opacity=0.9,
            ).add_to(m)

        # Points optionnels pour inspection locale (désactivés par défaut car coûteux)
        if show_detail_points:
            detail_step = max(1, int(np.ceil(len(df_gt) / 600)))
            for i in range(0, len(df_gt), detail_step):
                row = df_gt.iloc[i]
                rgba = colormap(norm(row[target_col]))
                hex_color = colors.to_hex(rgba)
                folium.CircleMarker(
                    location=[row['latitude_gt'], row['longitude_gt']],
                    radius=3,
                    color=hex_color,
                    fill=True,
                    fill_color=hex_color,
                    fill_opacity=0.75,
                    popup=f"""
                        <b>Heure :</b> {row['time_utc']}<br>
                        <b>{selection} :</b> {row[target_col]:.2f}<br>
                        <b>Structure :</b> {'Oui' if row['is_under_structure'] == 1 else 'Non'}
                        <b>Coordonnées :</b> ({row['latitude_gt']:.5f}, {row['longitude_gt']:.5f})
                    """,
                ).add_to(m)


        # 4. Trace GNSS optionnelle
        if show_gnss:
            gnss_cols = ['latitude_gnss', 'longitude_gnss']
            if all(col in df_plot.columns for col in gnss_cols):
                df_gnss = df_plot.dropna(subset=gnss_cols)
                points_gnss = df_gnss[gnss_cols].values.tolist()
                if len(points_gnss) >= 2:
                    folium.PolyLine(points_gnss, color="red", weight=2, dash_array='5', label="GNSS Brut").add_to(m)
                else:
                    st.warning("Trace GNSS non affichée: coordonnées valides insuffisantes.")
            else:
                st.warning("Trace GNSS non affichée: colonnes GNSS absentes.")

        # 5. Légende des objets cartographiques
        map_legend_html = f"""
        <div style="
            position: absolute;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
            background-color: rgba(255, 255, 255, 0.95);
            border: 1px solid #bdbdbd;
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 13px;
            line-height: 1.45;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        ">
            <div style="font-weight: 600; margin-bottom: 6px;">Légende</div>
            <div><span style="display:inline-block; width:16px; height:2px; background:#333333; margin-right:6px; vertical-align:middle;"></span>Trajet Ground Truth</div>
            <div><span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#1f77b4; margin-right:6px;"></span>Points colorés ({selection})</div>
            {"<div><span style='display:inline-block; width:16px; height:2px; border-top:2px dashed red; margin-right:6px; vertical-align:middle;'></span>Trace GNSS brute</div>" if show_gnss else ""}
            <div style="margin-top:6px; color:#555;">Min: {min_val:.2f} | Max: {max_val:.2f}</div>
            <div style="color:#777;">Plage réelle: {data_min:.2f} à {data_max:.2f}</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(map_legend_html))

        # Affichage
        st_folium(m, width=None, height=600, returned_objects=[])

    # --- BARRE DE LÉGENDE ---
    st.divider()
    st.caption(
        f"Échelle pour {selection} ({color_scale_mode}) : "
        f"Min ({min_val:.2f}) → Max ({max_val:.2f}) | "
        f"Plage réelle: {data_min:.2f} → {data_max:.2f}"
    )