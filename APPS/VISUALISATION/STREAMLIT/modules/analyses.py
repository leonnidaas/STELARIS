import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


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
    if pd.isna(val):
        return None
    txt = str(val).strip()
    if not txt:
        return None
    return txt.lower().replace("_", "-")


def _display_label_name(label):
    return label.replace("-", " ")


def _render_label_summary(df):
    label_col_candidates = ["label", "env_label", "label_env"]
    label_col = next((c for c in label_col_candidates if c in df.columns), None)
    if label_col is None:
        st.info("Aucune colonne de labels detectee pour afficher le resume de labellisation.")
        return

    label_series = df[label_col].map(_normalize_label).dropna()
    if label_series.empty:
        st.info("La colonne de labels est vide.")
        return

    counts = label_series.value_counts(dropna=False)
    total = int(counts.sum())

    preferred_order = [
        "tree",
        "mixed",
        "signal-denied",
        "other",
        "open-sky",
        "gare",
        "build",
        "bridge",
    ]

    ordered_labels = [lbl for lbl in preferred_order if lbl in counts.index]
    ordered_labels.extend(sorted([lbl for lbl in counts.index if lbl not in ordered_labels]))

    st.subheader("Resultat de la labellisation")

    stats_rows = []
    for lbl in ordered_labels:
        c = int(counts[lbl])
        pct = (100.0 * c / total) if total else 0.0
        stats_rows.append(
            {
                "label": lbl,
                "label_display": _display_label_name(lbl),
                "count": c,
                "pct": pct,
                "color": LABEL_COLORS.get(lbl, "#82adc9"),
            }
        )
    stats_df = pd.DataFrame(stats_rows)
    stats_df = stats_df.sort_values("pct", ascending=False).reset_index(drop=True)

    top_row = stats_df.iloc[0]
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Points labels", f"{total:,}")
    c2.metric("Classe dominante", top_row["label_display"].title())
    c3.metric("Part dominante", f"{top_row['pct']:.1f}%")

    fig_stats = go.Figure()
    fig_stats.add_trace(
        go.Bar(
            x=stats_df["pct"],
            y=stats_df["label_display"],
            orientation="h",
            marker={"color": stats_df["color"].tolist()},
            customdata=stats_df[["count"]],
            text=[f"{p:.1f}%" for p in stats_df["pct"]],
            textposition="outside",
            hovertemplate="Classe: %{y}<br>Points: %{customdata[0]:,}<br>Part: %{x:.1f}%<extra></extra>",
        )
    )
    fig_stats.update_layout(
        height=max(280, 46 * len(stats_df) + 80),
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="Part (%)",
        yaxis_title=None,
        template="plotly_white",
        showlegend=False,
    )
    fig_stats.update_xaxes(range=[0, max(100.0, float(stats_df["pct"].max() * 1.15))])
    fig_stats.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_stats, width="stretch")

    with st.expander("Voir le resume texte"):
        max_label_len = max([len(lbl) for lbl in ordered_labels] + [5])
        max_count_len = max([len(str(int(counts[lbl]))) for lbl in ordered_labels] + [1])
        lines = ["=== Resultat de la labellisation ==="]
        for lbl in ordered_labels:
            c = int(counts[lbl])
            pct = (100.0 * c / total) if total else 0.0
            lines.append(
                f"  {lbl.ljust(max_label_len)} : {str(c).rjust(max_count_len)} points ({pct:5.1f}%)"
            )
        st.code("\n".join(lines), language="text")


def _build_features_config(work_df: pd.DataFrame) -> dict[str, tuple[str, str, bool]]:
    known_features = {
        "Erreur Latérale": ("err_laterale", "Erreur (m)", False),
        "Erreur Longitudinale": ("err_longitudinale", "Erreur (m)", False),
        "Masque Ciel": ("sky_mask_deg", "Masque (deg)", True),
        "Densité Végétation": ("veg_density", "Densité", True),
        "Densité Bâti": ("building_density", "Densité", True),
        "Densité Pont (globale)": ("bridge_density", "Densité", True),
        "Densité Pont au-dessus": ("bridge_above_density", "Densité", True),
        "Pont au-dessus (count)": ("bridge_above_count", "Nb points", False),
        "Overhead ratio": ("obstacle_overhead_ratio", "Ratio", True),
        "Densité 0-5m": ("density_near_0_5m", "Densité", True),
        "Densité 5-15m": ("density_mid_5_15m", "Densité", True),
        "Densité 15-30m": ("density_far_15_30m", "Densité", True),
        "zrel p90": ("zrel_p90", "z rel (m)", False),
        "zrel p95": ("zrel_p95", "z rel (m)", False),
        "zrel p99": ("zrel_p99", "z rel (m)", False),
        "zrel IQR": ("zrel_iqr", "Dispersion (m)", False),
        "zrel std": ("zrel_std", "Dispersion (m)", False),
        "Occupation azimut": ("occupation_ciel_azimuth_ratio", "Ratio", True),
        "CMC": ("gnss_feat_CMC_e1", "cm", True),
        "Label Environnement": ("label", "Label", False),
        "Vitesse (m/s)": ("velocity", "Vitesse (m/s)", False),
        "CN0 (dB-Hz)": ("gnss_feat_CN0 mean", "CN0 (dB-Hz)", False),
    }

    features_config: dict[str, tuple[str, str, bool]] = {}
    used_cols: set[str] = set()

    for display_name, cfg in known_features.items():
        col_name, _, _ = cfg
        if col_name in work_df.columns:
            features_config[display_name] = cfg
            used_cols.add(col_name)

    numeric_columns = [
        col
        for col in work_df.select_dtypes(include="number").columns
        if col != "time_utc" and col not in used_cols
    ]
    for col in numeric_columns:
        features_config[col] = (col, col, False)

    return features_config

def render_analyses(df, trajet_id=None):
    st.title("Analyse des erreurs et Environnement")
    work_df = df.copy()
    if "time_utc" in work_df.columns:
        work_df["time_utc"] = pd.to_datetime(work_df["time_utc"], errors="coerce")
        work_df = work_df.dropna(subset=["time_utc"]).sort_values("time_utc", kind="stable")
    _render_label_summary(work_df)
    if "time_utc" not in work_df.columns or work_df.empty:
        st.info("Impossible d'afficher les graphes temporels: colonne time_utc absente ou vide.")
        return

    # 1. Configuration des caractéristiques disponibles
    # Format: (Nom affiché, Nom colonne, Unité/Titre Y, Remplissage sous courbe)
    features_config = _build_features_config(work_df)

    # 2. Couleurs spécifiques pour le ruban de labels (Heatmap)
    color_map = {
        "open-sky": LABEL_COLORS["open-sky"],
        "tree": LABEL_COLORS["tree"],
        "build": LABEL_COLORS["build"],
        "bridge": LABEL_COLORS["bridge"],
        "mixed": LABEL_COLORS["mixed"],
        "gare": LABEL_COLORS["gare"],
        "signal_denied": "#000000",
        "signal-denied": LABEL_COLORS["signal-denied"],
        "other": LABEL_COLORS["other"],
    }

    # 3. Sélection et Ordre (L'ordre de sélection = l'ordre d'affichage)
    st.subheader("Configuration de l'affichage")

    # L'ordre dans lequel l'utilisateur sélectionne les éléments définit l'ordre des subplots
    selected_names = st.multiselect(
    "Cliquez sur les éléments dans l'ordre d'affichage souhaité (ex: 1er clic = Haut)",
    options=list(features_config.keys()),
    default=[
        name
        for name in ["Erreur Latérale", "Masque Ciel", "Label Environnement"]
        if name in features_config
    ] or list(features_config.keys())[:3]
    )

    if not selected_names:
        st.info("Sélectionnez au moins un paramètre.")
        return

    # 4. Création des Subplots
    fig = make_subplots(
        rows=len(selected_names),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=selected_names,
    )

    for row_idx, name in enumerate(selected_names, start=1):
        col_name, y_title, fill_to_zero = features_config[name]
        
        if col_name not in work_df.columns:
            st.warning(f"Colonne '{col_name}' absente du dataset.")
            continue

        if col_name == "label":
            # --- MODE HEATMAP (RUBAN DE COULEUR) ---
            # On groupe par label pour créer les segments de couleur
            unique_labels = work_df[col_name].dropna().unique()
            for l in unique_labels:
                mask = work_df[col_name] == l
                fig.add_trace(
                    go.Bar(
                        x=work_df.loc[mask, "time_utc"],
                        y=[1] * mask.sum(),
                        name=str(l),
                        marker_color=color_map.get(str(l).lower(), "#82adc9"),
                        showlegend=True,
                        customdata=[str(l)] * mask.sum(),
                        hovertemplate="Label: %{customdata}<extra></extra>",
                        # hoverinfo="x unified", # On peut aussi afficher le label au hover, mais ça peut être très verbeux
                        width=1000, # Largeur d'un bloc en ms (ajuster selon votre freq)
                        offset=-500, # Décalage pour centrer les blocs sur les timestamps
                    ),
                    row=row_idx, col=1
                )
            fig.update_yaxes(showticklabels=False, row=row_idx, col=1)
        
        else:
            # --- MODE COURBE CLASSIQUE ---
            
            fig.add_trace(
                go.Scatter(
                    x=work_df["time_utc"],
                    y=work_df[col_name],
                    name=name,
                    mode="lines",
                    fill="tozeroy" if fill_to_zero else None,
                    line=dict(width=1.5),
                    hovertemplate="%{y:.2f}"
                    
                ),
                row=row_idx,
                col=1,
            )
        
        fig.update_yaxes(title_text=y_title, row=row_idx, col=1)

    # 5. Mise en page globale
    fig.update_layout(
        hoverdistance=100, # Distance en pixels pour déclencher le hover
        spikedistance=-1, # Affiche les spikelines même si le curseur n'est pas exactement sur un point
        height=max(300, 220 * len(selected_names)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=80, b=40),
        barmode='overlay', # Essentiel pour que les barres fassent un ruban
        hovermode="x unified", # Affiche les infos de tous les traces alignées verticalement
    )

    fig.update_xaxes(title_text="Temps (UTC)", row=len(selected_names), col=1)

    zoom_key = f"analysis_zoom_window_{trajet_id if trajet_id else 'global'}"
    zoom_payload = st.session_state.get(zoom_key)
    if zoom_payload:
        try:
            z0 = pd.to_datetime(zoom_payload.get("start"))
            z1 = pd.to_datetime(zoom_payload.get("end"))
            if pd.notna(z0) and pd.notna(z1) and z1 > z0:
                fig.update_xaxes(range=[z0, z1])
        except Exception:
            pass

    st.plotly_chart(fig, width="stretch")

# Exemple d'appel (si vous testez en local) :
# render_analyses(votre_dataframe)