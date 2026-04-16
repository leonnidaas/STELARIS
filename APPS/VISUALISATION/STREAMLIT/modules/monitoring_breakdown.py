import pandas as pd
import streamlit as st

from services.monitoring_service import explain_label_reason


def _pick_row_by_time_slider(df: pd.DataFrame, key_suffix: str):
    if "time_utc" not in df.columns:
        return None, "La colonne time_utc est absente."

    times = pd.to_datetime(df["time_utc"], errors="coerce", utc=True).dt.tz_convert(None)
    valid_idx = times[times.notna()].index

    if len(valid_idx) == 0:
        raw_vals = df["time_utc"].dropna().astype(str).unique().tolist()
        if not raw_vals:
            return None, "Aucune valeur time_utc exploitable."
        selected_raw = st.select_slider(
            "time_utc",
            options=raw_vals,
            key=f"time_utc_select_{key_suffix}",
        )
        picked = df[df["time_utc"].astype(str) == selected_raw].iloc[0]
        return picked, None

    min_time = times.loc[valid_idx].min().to_pydatetime()
    max_time = times.loc[valid_idx].max().to_pydatetime()

    selected_time = st.slider(
        "time_utc",
        min_value=min_time,
        max_value=max_time,
        value=min_time,
        format="YYYY-MM-DD HH:mm:ss",
        key=f"time_utc_slider_{key_suffix}",
    )

    selected_ts = pd.Timestamp(selected_time)
    deltas = (times.loc[valid_idx] - selected_ts).abs()
    picked_idx = int(deltas.idxmin())
    picked = df.loc[picked_idx]
    nearest_time = times.loc[picked_idx]
    delta_s = abs((nearest_time - selected_ts).total_seconds())

    if delta_s > 0:
        st.caption(
            f"time_utc demande: {selected_ts} | point le plus proche: {nearest_time} | ecart: {delta_s:.3f}s"
        )

    return picked, None


def render_monitoring_breakdown(
    matched_df: pd.DataFrame,
    labelisation_params: dict | None,
    key_suffix: str = "default",
) -> None:
    st.subheader("Analyse d'un point")
    if matched_df.empty:
        st.info("Aucun point disponible dans ce CSV.")
        return

    with st.expander("Decomposer le choix du label", expanded=False):
        picked, err = _pick_row_by_time_slider(matched_df, key_suffix=key_suffix)
        if picked is None:
            st.warning(err or "Impossible de selectionner un point.")
            return

        picked_label = picked.get("label", "N/A")
        st.caption(
            f"Point selectionne: time_utc={picked.get('time_utc', 'N/A')} | label={picked_label}"
        )

        summary = {
            "label": picked_label,
            "latitude_gt": picked.get("latitude_gt", "N/A"),
            "longitude_gt": picked.get("longitude_gt", "N/A"),
            "obs_type": picked.get("obs_type", "N/A"),
            "speed_gt_mps_smooth": picked.get("speed_gt_mps_smooth", picked.get("speed_gt_mps", "N/A")),
            "sky_mask_deg": picked.get("sky_mask_smoothed", picked.get("sky_mask_deg", "N/A")),
            "building_density": picked.get("building_density", "N/A"),
            "veg_density": picked.get("veg_density", "N/A"),
            "zrel_p95": picked.get("zrel_p95", "N/A"),
            "zrel_std": picked.get("zrel_std", "N/A"),
            "bridge_above_count": picked.get("bridge_above_count", "N/A"),
        }
        st.json(summary)

        reason, checks = explain_label_reason(picked, params_cfg=labelisation_params or {})
        st.markdown(f"**Pourquoi ce label ?** {reason}")
        for line in checks:
            st.write(f"- {line}")

        if labelisation_params:
            with st.expander("Seuils du dernier run de labelisation", expanded=False):
                st.json(labelisation_params)
        else:
            st.caption("Aucun parametre de run trouve; l'explication utilise les seuils par defaut du pipeline.")