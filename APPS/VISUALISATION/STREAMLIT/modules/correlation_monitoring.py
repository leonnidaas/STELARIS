import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


GNSS_PREFIX = "gnss_feat_"

DEFAULT_FEATURES = [
    "gnss_feat_NSV",
    "gnss_feat_EL mean",
    "gnss_feat_EL std",
    "gnss_feat_pdop",
    "gnss_feat_CN0 mean",
    "gnss_feat_CN0 std",
    "gnss_feat_CMC_l1",
    "gnss_feat_CMC_e1",
]


def _select_default_features(numeric_columns: list[str]) -> list[str]:
    defaults = [col for col in DEFAULT_FEATURES if col in numeric_columns]
    if defaults:
        return defaults
    return numeric_columns[: min(8, len(numeric_columns))]


def _corr_matrix_figure(corr_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.to_numpy(dtype=float),
            x=corr_df.columns.tolist(),
            y=corr_df.index.tolist(),
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": "corr"},
            hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=max(380, 42 * len(corr_df.index) + 140),
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        template="plotly_white",
    )
    return fig


def _rolling_corr_figure(
    time_values: pd.Series,
    rolling_corr: pd.Series,
    x_col: str,
    y_col: str,
    window: int,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=rolling_corr,
            mode="lines",
            name="Rolling corr",
            line={"width": 1.8, "color": "#1F77B4"},
            hovertemplate="%{x}<br>corr=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Correlation glissante ({x_col} vs {y_col}) - fenetre={window}",
        xaxis_title="Temps (UTC)",
        yaxis_title="Correlation",
        yaxis={"range": [-1.05, 1.05]},
        height=320,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        template="plotly_white",
    )
    return fig


def _autocorr_matrix_figure(autocorr_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=autocorr_df.to_numpy(dtype=float),
            x=autocorr_df.columns.tolist(),
            y=autocorr_df.index.tolist(),
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": "autocorr"},
            hovertemplate="lag=%{y}<br>var=%{x}<br>acf=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Variables",
        yaxis_title="Lag (nb points)",
        height=max(360, 14 * len(autocorr_df.index) + 150),
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        template="plotly_white",
    )
    return fig


def _single_acf_figure(acf_series: pd.Series, variable: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=acf_series.index.tolist(),
            y=acf_series.to_numpy(dtype=float),
            mode="lines+markers",
            name=variable,
            line={"width": 1.7, "color": "#FF7F0E"},
            marker={"size": 5},
            hovertemplate="lag=%{x}<br>acf=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Autocorrelation detail - {variable}",
        xaxis_title="Lag (nb points)",
        yaxis_title="ACF",
        yaxis={"range": [-1.05, 1.05]},
        height=300,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        template="plotly_white",
    )
    return fig


def render_correlation_module(df: pd.DataFrame, trajet_id: str | None = None) -> None:
    st.title("Matrice de correlation et autocorrelation")
    if trajet_id:
        st.caption(f"Analyse temporelle sur le trajet/scenario actif: {trajet_id}")

    if df is None or len(df) == 0:
        st.info("Dataset vide: impossible de calculer les correlations.")
        return

    work_df = df.copy()
    if "time_utc" in work_df.columns:
        work_df["time_utc"] = pd.to_datetime(work_df["time_utc"], errors="coerce")
        work_df = work_df.dropna(subset=["time_utc"]).sort_values("time_utc", kind="stable")
    else:
        st.info("Colonne time_utc absente: analyse temporelle indisponible.")
        return

    numeric_df = work_df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    gnss_columns = [
        c
        for c in numeric_df.columns
        if c.startswith(GNSS_PREFIX) and numeric_df[c].notna().sum() > 5
    ]
    if len(gnss_columns) < 2:
        st.info("Pas assez de features GNSS numeriques exploitables (minimum 2).")
        return

    st.subheader("Configuration")
    col_cfg_1, col_cfg_2, col_cfg_3 = st.columns(3)

    with col_cfg_1:
        selected_features = st.multiselect(
            "Variables pour la matrice",
            options=gnss_columns,
            default=_select_default_features(gnss_columns),
            help="Selectionne les features GNSS a inclure dans la matrice de correlation globale.",
        )

    with col_cfg_2:
        corr_method = st.selectbox(
            "Methode",
            options=["pearson", "spearman"],
            index=0,
        )
        min_periods = st.slider(
            "Min points par paire",
            min_value=5,
            max_value=200,
            value=20,
            step=5,
        )

    with col_cfg_3:
        resample_mode = st.selectbox(
            "Reechantillonnage temporel",
            options=["Aucun", "1s", "2s", "5s", "10s"],
            index=0,
            help="Permet de lisser les frequences tres elevees avant le calcul des correlations.",
        )

    if len(selected_features) < 2:
        st.info("Selectionne au moins 2 variables pour la matrice de correlation.")
        return

    corr_base = work_df[["time_utc", *selected_features]].copy()
    if resample_mode != "Aucun":
        corr_base = (
            corr_base.set_index("time_utc")
            .resample(resample_mode)
            .mean(numeric_only=True)
            .dropna(how="all")
            .reset_index()
        )

    corr_input = corr_base[selected_features]
    corr_df = corr_input.corr(method=corr_method, min_periods=min_periods)

    st.plotly_chart(
        _corr_matrix_figure(corr_df, f"Matrice de correlation ({corr_method})"),
        width="stretch",
    )

    st.subheader("Correlation glissante dans le temps")
    roll_c1, roll_c2, roll_c3 = st.columns(3)
    with roll_c1:
        x_col = st.selectbox("Variable X", options=selected_features, index=0)
    with roll_c2:
        y_default_index = 1 if len(selected_features) > 1 else 0
        y_col = st.selectbox("Variable Y", options=selected_features, index=y_default_index)
    with roll_c3:
        roll_window = st.slider(
            "Fenetre glissante (points)",
            min_value=10,
            max_value=min(1000, max(30, len(corr_base) // 2)),
            value=min(120, max(20, len(corr_base) // 8)),
            step=5,
        )

    rolling_input = corr_base[["time_utc", x_col, y_col]].dropna()
    if len(rolling_input) <= roll_window:
        st.info("Pas assez de points pour la correlation glissante avec cette fenetre.")
    else:
        rolling_corr = rolling_input[x_col].rolling(window=roll_window, min_periods=max(10, roll_window // 3)).corr(
            rolling_input[y_col]
        )
        st.plotly_chart(
            _rolling_corr_figure(rolling_input["time_utc"], rolling_corr, x_col, y_col, roll_window),
            width="stretch",
        )

    st.subheader("Autocorrelation (ACF) par variable")
    acf_c1, acf_c2 = st.columns(2)
    with acf_c1:
        acf_features = st.multiselect(
            "Variables ACF",
            options=selected_features,
            default=selected_features[: min(6, len(selected_features))],
        )
    with acf_c2:
        max_lag = st.slider(
            "Lag max (points)",
            min_value=5,
            max_value=min(300, max(20, len(corr_base) // 4)),
            value=min(60, max(10, len(corr_base) // 15)),
            step=1,
        )

    if not acf_features:
        st.info("Selectionne au moins une variable pour l'autocorrelation.")
        return

    acf_rows = []
    lag_values = list(range(1, max_lag + 1))
    for lag in lag_values:
        row = {}
        for feature in acf_features:
            series = corr_base[feature].dropna()
            row[feature] = float(series.autocorr(lag=lag)) if len(series) > lag else np.nan
        acf_rows.append(row)

    autocorr_df = pd.DataFrame(acf_rows, index=lag_values)
    autocorr_df.index.name = "lag"

    st.plotly_chart(
        _autocorr_matrix_figure(autocorr_df, "Matrice d'autocorrelation (ACF)"),
        width="stretch",
    )

    acf_focus = st.selectbox("Variable detail ACF", options=acf_features, index=0)
    st.plotly_chart(_single_acf_figure(autocorr_df[acf_focus], acf_focus), width="stretch")
