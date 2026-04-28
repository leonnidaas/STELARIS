import json

import pandas as pd
import streamlit as st

from VISUALISATION.STREAMLIT.services.training_service import (
	APPS_ROOT,
	build_trajet_groups,
	expand_unique_trajets,
	list_training_datasets,
	list_trajets_for_inference,
	run_script,
)
from VISUALISATION.STREAMLIT.ui.theme import apply_theme, render_hero
from utils import TRAINING_DIR, get_traj_paths


st.set_page_config(page_title="Inference IA", layout="wide", page_icon="✏️")


def _ensure_state() -> None:
	if "inference_running" not in st.session_state:
		st.session_state.inference_running = False
	if "inference_last_logs" not in st.session_state:
		st.session_state.inference_last_logs = ""
	if "inference_last_result" not in st.session_state:
		st.session_state.inference_last_result = {}


@st.cache_data(show_spinner=False)
def _datasets() -> list[str]:
	return list_training_datasets(TRAINING_DIR)


def _latest_outputs_for_trajet(traj_id: str) -> tuple[str | None, str | None]:
	try:
		paths = get_traj_paths(traj_id)
	except Exception:
		return None, None

	csv_path = paths.get("inference_latest_csv")
	json_path = paths.get("inference_latest_json")
	return (
		str(csv_path) if csv_path is not None and csv_path.exists() else None,
		str(json_path) if json_path is not None and json_path.exists() else None,
	)


def render_page() -> None:
	apply_theme()
	_ensure_state()

	render_hero(
		"Inference sur trajet complet",
		"Charge les poids des modeles associes a un dataset et produit un CSV d'inference par trajet.",
	)

	is_running = st.session_state.inference_running

	datasets = _datasets()
	if not datasets:
		st.error("Aucun dataset d'entrainement detecte dans DATA/03_TRAINING.")
		st.stop()

	trajets_disponibles = list_trajets_for_inference()
	if not trajets_disponibles:
		st.error("Aucun trajet detecte avec OBS+NAV disponibles dans DATA/00_RAW/GNSS_RINEX.")
		st.stop()

	grouped_trajets = build_trajet_groups(trajets_disponibles)
	trajets_uniques = sorted(grouped_trajets.keys())

	left, right = st.columns([2, 1])
	with left:
		dataset_name = st.selectbox(
			"Dataset associe aux modeles",
			options=datasets,
			index=0,
			disabled=is_running,
		)

		selected_models = st.multiselect(
			"Modeles a utiliser pour l'inference",
			options=["GRU", "XGBOOST", "CNN_1D"],
			default=["GRU", "XGBOOST"],
			disabled=is_running,
			help="Choisis 1, 2 ou 3 modeles. pred_final prend le label du modele le plus confiant a chaque ligne.",
		)

		selection_mode = st.radio(
			"Mode de selection trajet",
			options=["Trajets uniques", "Scenario manuel"],
			horizontal=True,
			disabled=is_running,
		)

		if selection_mode == "Trajets uniques":
			unique_pick_mode = st.radio(
				"Scenario a inferer",
				options=["Premier scenario uniquement", "Tous les scenarios"],
				horizontal=True,
				disabled=is_running,
			)

			selected_uniques = st.multiselect(
				"Trajets uniques",
				options=trajets_uniques,
				default=trajets_uniques[:1],
				disabled=is_running,
				format_func=lambda t: f"{t} ({len(grouped_trajets.get(t, []))} scenarios)",
			)

			if unique_pick_mode == "Tous les scenarios":
				selected_trajets = expand_unique_trajets(selected_uniques, grouped_trajets)
			else:
				selected_trajets = [
					grouped_trajets[t][0]
					for t in selected_uniques
					if grouped_trajets.get(t)
				]
		else:
			selected_trajets = st.multiselect(
				"Scenarios a inferer",
				options=trajets_disponibles,
				default=trajets_disponibles[:1],
				disabled=is_running,
			)

		st.caption(f"{len(selected_trajets)} trajet(s) a inferer.")

	with right:
		max_live_lines = st.slider(
			"Lignes de logs visibles",
			min_value=50,
			max_value=1200,
			value=300,
			step=50,
			disabled=is_running,
		)

		continue_on_error = st.checkbox(
			"Continuer si erreur sur un trajet",
			value=True,
			disabled=is_running,
		)

	launch = st.button(
		"Lancer inference",
		type="primary",
		width="stretch",
		disabled=is_running,
	)

	if launch:
		if not selected_trajets:
			st.error("Selectionne au moins un trajet.")
		elif not selected_models:
			st.error("Selectionne au moins un modele.")
		else:
			st.session_state.inference_running = True
			st.session_state.inference_last_logs = ""
			st.session_state.inference_last_result = {}

			progress = st.progress(0, text="Demarrage inference...")
			log_placeholder = st.empty()

			per_traj_results: list[dict] = []
			global_ok = True
			total = len(selected_trajets)

			for idx, traj_id in enumerate(selected_trajets, start=1):
				progress.progress(
					int((idx - 1) / total * 100),
					text=f"Inference [{idx}/{total}] {traj_id}",
				)

				ok, logs = run_script(
					"INFERENCE_MODELES.app",
					[
						"--traj",
						str(traj_id),
						"--dataset-name",
						str(dataset_name),
						"--models",
						*selected_models,
					],
					log_placeholder=log_placeholder,
					max_live_lines=max_live_lines,
				)

				csv_path, json_path = _latest_outputs_for_trajet(traj_id)
				per_traj_results.append(
					{
						"trajet": traj_id,
						"statut": "OK" if ok else "ECHEC",
						"latest_csv": csv_path or "-",
						"latest_json": json_path or "-",
					}
				)

				st.session_state.inference_last_logs = logs
				if not ok:
					global_ok = False
					if not continue_on_error:
						break

			progress.progress(100, text="Inference terminee")
			st.session_state.inference_running = False
			st.session_state.inference_last_result = {
				"dataset": dataset_name,
				"models": selected_models,
				"rows": per_traj_results,
				"ok": global_ok,
			}

			if global_ok:
				st.success("Inference terminee avec succes.")
			else:
				st.error("Inference terminee avec au moins une erreur.")

	st.divider()
	st.subheader("Dernier resultat")

	last_result = st.session_state.inference_last_result
	if not last_result:
		st.info("Aucune inference lancee dans cette session.")
	else:
		st.caption(f"Dataset: {last_result.get('dataset', '-')}")
		st.caption(f"Modeles: {', '.join(last_result.get('models', [])) or '-'}")
		st.dataframe(pd.DataFrame(last_result.get("rows", [])), width="stretch", hide_index=True)

		successful = [r for r in last_result.get("rows", []) if r.get("statut") == "OK"]
		if successful:
			picked = successful[-1]
			json_path = picked.get("latest_json")
			csv_path = picked.get("latest_csv")

			if json_path and json_path != "-":
				try:
					with open(json_path, "r", encoding="utf-8") as f:
						payload = json.load(f)
						st.markdown("**Metadata inference (latest)**")
						st.json(payload)

					analysis = payload.get("analysis", {}) if isinstance(payload, dict) else {}
					per_model = analysis.get("per_model", {}) if isinstance(analysis, dict) else {}
					if per_model:
						rows = []
						for model_name, info in per_model.items():
							conf = info.get("confidence", {}) if isinstance(info, dict) else {}
							rows.append(
								{
									"modele": model_name,
									"n_predictions": int(info.get("n_predictions", 0)),
									"conf_mean": float(conf.get("mean", 0.0)),
									"conf_median": float(conf.get("median", 0.0)),
									"conf_min": float(conf.get("min", 0.0)),
									"conf_max": float(conf.get("max", 0.0)),
								}
							)
						st.markdown("**Analyse independante par modele**")
						st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

					pairwise = analysis.get("pairwise_agreement", {}) if isinstance(analysis, dict) else {}
					if pairwise:
						pair_rows = [
							{"comparaison": k, "taux_accord": float(v)}
							for k, v in pairwise.items()
						]
						st.markdown("**Accord entre modeles**")
						st.dataframe(pd.DataFrame(pair_rows), width="stretch", hide_index=True)

					if isinstance(analysis, dict) and "all_models_unanimity_rate" in analysis:
						st.caption(
							"Taux d'unanimite tous modeles: "
							f"{float(analysis.get('all_models_unanimity_rate', 0.0)):.2%}"
						)
				except Exception as e:
					st.warning(f"Impossible de lire le JSON latest: {e}")

			if csv_path and csv_path != "-":
				try:
					df_preview = pd.read_csv(csv_path, nrows=200)
					st.markdown("**Apercu CSV inference (200 lignes max)**")
					st.dataframe(df_preview, width="stretch", hide_index=True)
				except Exception as e:
					st.warning(f"Impossible de lire le CSV latest: {e}")

	logs = st.session_state.inference_last_logs
	if logs:
		with st.expander("Logs", expanded=False):
			st.code(logs)

	st.caption(f"Execution racine: {APPS_ROOT}")


if __name__ == "__main__":
	render_page()
