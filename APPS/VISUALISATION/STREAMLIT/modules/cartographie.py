import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import branca.colormap as bcm


LABEL_COLORS = {
    "open-sky": "#636EFA",
    "tree": "#00CC96",
    "build": "#EF553B",
    "bridge": "#FF9900",
    "mixed": "#AB63FA",
    "gare": "#F319E8",
    "signal-denied": "#000000",
    "other": "#7F8C8D",
}


def _normalize_label(val):
    if val is None:
        return None
    txt = str(val).strip()
    if not txt or txt.lower() == "nan":
        return None
    return txt.lower().replace("_", "-")

def render_carte(df, traj_id):
    st.title(f"🗺️ Cartographie Interactive :")
    st.markdown("Visualisez la précision du GNSS et l'impact de l'environnement sur la carte.")
    # --- FILTRES DE COULEUR ---
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.subheader("Configuration")
        # Sélection de la métrique pour la couleur
        metrics = {
            "Erreur Latérale (m)": "err_laterale",
            "Erreur Longitudinale (m)": "err_longitudinale",
            "Masque de Ciel (°)": "sky_mask_deg",
            "Densité Végétation": "veg_density",
            "Densité Bâti": "building_density",
            "Densité Pont (globale)": "bridge_density",
            "Densité Pont au-dessus": "bridge_above_density",
            "Nb points pont au-dessus": "bridge_above_count",
            "Obstacle overhead": "obstacle_overhead_ratio",
            "Densité proche 0-5m": "density_near_0_5m",
            "Densité moyenne 5-15m": "density_mid_5_15m",
            "Densité lointaine 15-30m": "density_far_15_30m",
            "zrel p95": "zrel_p95",
            "zrel p99": "zrel_p99",
            "zrel std": "zrel_std",
            "Occupation azimut": "occupation_ciel_azimuth_ratio",
            "Labels": "label",
        }
        selection = st.selectbox("Colorer le trajet par :", list(metrics.keys()), index=4)
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
        force_all_points = st.checkbox("Afficher tous les points valides", value=True)
        max_points = st.slider(
            "Points max à afficher (si option ci-dessus désactivée)",
            min_value=1,
            max_value=50000,
            value=10000,
            step=100,
            disabled=force_all_points,
        )
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

    total_points = int(len(df_plot))
    valid_points = int(len(df_gt))
    is_label_metric = target_col == "label"

    label_counts = None
    label_color_map = None
    min_val = None
    max_val = None
    data_min = None
    data_max = None
    norm = None
    colormap = None

    if is_label_metric:
        df_gt["_label_norm"] = df_gt[target_col].map(_normalize_label)
        df_gt = df_gt.dropna(subset=["_label_norm"]).copy()
        if df_gt.empty:
            st.error("Aucun label valide a afficher pour la cartographie.")
            return

        label_counts = df_gt["_label_norm"].value_counts()
        label_color_map = {
            "open-sky": LABEL_COLORS["open-sky"],
            "tree": LABEL_COLORS["tree"],
            "build": LABEL_COLORS["build"],
            "bridge": LABEL_COLORS["bridge"],
            "mixed": LABEL_COLORS["mixed"],
            "gare": LABEL_COLORS["gare"],
            "signal-denied": LABEL_COLORS["signal-denied"],
            "signal_denied": LABEL_COLORS["signal-denied"],
            "other": LABEL_COLORS["other"],
        }
    else:
        # Création de l'échelle de couleur (Colormap) pour métriques numériques
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

    render_step = 1 if force_all_points else max(1, int(np.ceil(len(df_gt) / max_points)))
    rendered_points = int(np.ceil(len(df_gt) / render_step))
    st.caption(
        f"Points dataset: {total_points} | points valides ({selection}): {valid_points} | points rendus: {rendered_points}"
    )

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

        metric_colormap = None
        if not is_label_metric:
            # 2. Légende métrique (barre de couleur) utilisée aussi pour la ColorLine
            metric_colormap = bcm.LinearColormap(
                colors=['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000'],
                vmin=min_val,
                vmax=max_val,
                caption=selection,
            )
            metric_colormap.add_to(m)

        # 3. Tracé coloré optimisé en une seule couche (beaucoup plus fluide au pan/zoom)
        df_render = df_gt.iloc[::render_step].reset_index(drop=True)
        if len(df_render) >= 2:
            traj_positions = df_render[['latitude_gt', 'longitude_gt']].values.tolist()
            if is_label_metric:
                # ColorLine est numerique: pour les labels on trace des segments colores.
                for i in range(1, len(df_render)):
                    lbl = df_render.iloc[i]["_label_norm"]
                    seg_color = label_color_map.get(lbl, "#82adc9")
                    folium.PolyLine(
                        [traj_positions[i - 1], traj_positions[i]],
                        color=seg_color,
                        weight=4,
                        opacity=0.9,
                    ).add_to(m)
            else:
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
            detail_step = render_step
            for i in range(0, len(df_gt), detail_step):
                row = df_gt.iloc[i]
                if is_label_metric:
                    row_label = row.get("_label_norm")
                    hex_color = label_color_map.get(row_label, "#82adc9")
                    metric_value = str(row.get(target_col, ""))
                else:
                    rgba = colormap(norm(row[target_col]))
                    hex_color = colors.to_hex(rgba)
                    metric_value = f"{float(row[target_col]):.2f}"

                time_val = row["time_utc"] if "time_utc" in row.index else "N/A"
                is_under_structure = row["is_under_structure"] if "is_under_structure" in row.index else None
                structure_txt = "N/A" if is_under_structure is None else ("Oui" if is_under_structure == 1 else "Non")
                folium.CircleMarker(
                    location=[row['latitude_gt'], row['longitude_gt']],
                    radius=3,
                    color=hex_color,
                    fill=True,
                    fill_color=hex_color,
                    fill_opacity=0.75,
                    popup=f"""
                        <b>Heure :</b> {time_val}<br>
                        <b>{selection} :</b> {metric_value}<br>
                        <b>Structure :</b> {structure_txt}
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
        if is_label_metric:
            ordered_labels = [
                "tree", "mixed", "signal-denied", "other", "open-sky", "gare", "build", "bridge"
            ]
            observed_labels = list(label_counts.index)
            ordered_observed = [lbl for lbl in ordered_labels if lbl in observed_labels]
            ordered_observed.extend([lbl for lbl in observed_labels if lbl not in ordered_observed])
            label_rows = "".join(
                [
                    (
                        "<div><span style='display:inline-block; width:10px; height:10px; border-radius:50%; "
                        f"background:{label_color_map.get(lbl, '#82adc9')}; margin-right:6px;'></span>"
                        f"{lbl} ({int(label_counts[lbl])})</div>"
                    )
                    for lbl in ordered_observed
                ]
            )

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
                <div style="margin-top:6px; color:#555; font-weight:600;">Classes</div>
                {label_rows}
            </div>
            """
        else:
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

        # Affichage de la carte (sans breakdown local)
        st_folium(
            m,
            width=None,
            height=600,
            returned_objects=[],
            key=f"monitoring_map_{traj_id}",
        )

    # --- BARRE DE LÉGENDE ---
    st.divider()
    if is_label_metric:
        n_classes = int(label_counts.shape[0])
        n_points = int(label_counts.sum())
        st.caption(
            f"Coloration catégorielle pour {selection} : {n_classes} classes detectees, {n_points} points affichables."
        )
    else:
        st.caption(
            f"Échelle pour {selection} ({color_scale_mode}) : "
            f"Min ({min_val:.2f}) → Max ({max_val:.2f}) | "
            f"Plage réelle: {data_min:.2f} → {data_max:.2f}"
        )