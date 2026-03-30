import subprocess
import sys
import shutil
import json
from typing import Any
from pathlib import Path
from datetime import datetime

import streamlit as st
from LABELISATION_AUTO_LIDAR_HD_IGN.run_params import load_latest_labelisation_run_params

from utils import CHUNK_SIZE, GT_DIR, N_WORKERS, PYTHON_LABELISATION_INTERPRETER, PYTHON_RINEX_INTERPRETER, get_traj_paths


APPS_ROOT = Path(__file__).resolve().parents[3]


PHASES = {
	"Traitement RINEX": {
		"module": "TRAITEMENT_RINEX.app",
		"env": "rinex",
	},
	"Extraction features GNSS": {
		"module": "EXTRACTION_DES_FEATURES_GNSS.app",
		"env": "label",
	},
	"Pipeline Labelisation LiDAR": {
		"module": "LABELISATION_AUTO_LIDAR_HD_IGN.app",
		"env": "label",
	},
}


def list_trajets() -> list[str]:
	return sorted([p.name for p in GT_DIR.iterdir() if p.is_dir()])


def _resolve_python_for_env(env_name: str) -> str:
	if env_name == "rinex":
		configured = PYTHON_RINEX_INTERPRETER
		fallback = APPS_ROOT / "venv_rinex" / "bin" / "python"
	else:
		configured = PYTHON_LABELISATION_INTERPRETER
		fallback = APPS_ROOT / "venv_label" / "bin" / "python"

	if configured:
		p = Path(configured)
		if p.is_absolute() and p.exists():
			return str(p)
		if (APPS_ROOT / configured).exists():
			return str(APPS_ROOT / configured)
		if shutil.which(configured):
			return configured

	if fallback.exists():
		return str(fallback)

	return sys.executable


def _render_live_log(log_placeholder, lines: list[str], max_lines: int = 200) -> None:
	if not lines:
		log_placeholder.code("(demarrage...)")
		return
	log_placeholder.code("\n".join(lines[-max_lines:]))


def _build_phase_extra_args(phase_name: str, phase_params: dict[str, Any] | None) -> list[str]:
	if phase_name != "Pipeline Labelisation LiDAR" or not phase_params:
		return []

	return ["--options-json", json.dumps(phase_params)]


def _logs_indicate_failure(logs: str) -> bool:
	if not logs:
		return False
	lower_logs = logs.lower()

	# Erreurs certaines
	hard_markers = [
		"traceback (most recent call last):",
		"le pipeline a rencontre des erreurs.",
		"le pipeline a rencontré des erreurs.",
		"pipeline a rencontre des erreurs",
		"pipeline a rencontré des erreurs",
	]
	if any(marker in lower_logs for marker in hard_markers):
		return True

	# Lignes d'erreur explicites sans penaliser les formulations de type "aucune erreur"
	for line in lower_logs.splitlines():
		s = line.strip()
		if s.startswith("erreur :") or s.startswith("erreur lors de") or s.startswith("fichier manquant"):
			return True

	return False


def run_module(
	phase_name: str,
	traj_id: str,
	log_placeholder=None,
	phase_params: dict[str, Any] | None = None,
	max_live_lines: int = 200,
) -> tuple[bool, str]:
	phase_cfg = PHASES[phase_name]
	module_name = phase_cfg["module"]
	python_exec = _resolve_python_for_env(phase_cfg["env"])
	cmd = [python_exec, "-m", module_name, "--traj", traj_id] + _build_phase_extra_args(phase_name, phase_params)
	header = f"[python] {python_exec}\n[cmd] {' '.join(cmd)}"
	lines = [header]

	if log_placeholder is not None:
		_render_live_log(log_placeholder, lines, max_lines=max_live_lines)

	proc = subprocess.Popen(
		cmd,
		cwd=APPS_ROOT,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		bufsize=1,
	)

	assert proc.stdout is not None
	for line in proc.stdout:
		lines.append(line.rstrip("\n"))
		if log_placeholder is not None:
			_render_live_log(log_placeholder, lines, max_lines=max_live_lines)

	return_code = proc.wait()
	logs = "\n".join(lines).strip()
	ok = return_code == 0 and not _logs_indicate_failure(logs)
	return ok, logs


def run_pipeline_complet(
	traj_id: str,
	stop_on_error: bool = True,
	phase_log_placeholders: dict[str, Any] | None = None,
	phase_params: dict[str, Any] | None = None,
	max_live_lines: int = 200,
) -> tuple[bool, list[tuple[str, bool, str]]]:
	resultats = []
	global_ok = True
	for phase_name in PHASES:
		placeholder = None if phase_log_placeholders is None else phase_log_placeholders.get(phase_name)
		ok, logs = run_module(
			phase_name,
			traj_id,
			log_placeholder=placeholder,
			phase_params=phase_params,
			max_live_lines=max_live_lines,
		)
		resultats.append((phase_name, ok, logs))
		if not ok:
			global_ok = False
			if stop_on_error:
				break
	return global_ok, resultats


def _expected_csv_outputs(traj_id: str, phase_name: str) -> list[Path]:
	cfg = get_traj_paths(traj_id)
	mapping = {
		"Traitement RINEX": [cfg["raw_gnss"], cfg["space_vehicule_info"]],
		"Extraction features GNSS": [cfg["gnss_features_csv"]],
		"Pipeline Labelisation LiDAR": [
			cfg["sync_csv"],
			cfg["features_csv"],
			cfg["fusion_features_csv"],
			cfg["labels_csv"],
			cfg["labels_plus_features_csv"],
			cfg["final_fusion_csv"],
		],
	}
	return [Path(p) for p in mapping.get(phase_name, [])]


def _snapshot_mtimes(paths: list[Path]) -> dict[str, float | None]:
	return {str(p): (p.stat().st_mtime if p.exists() else None) for p in paths}


def _detect_csv_changes(before: dict[str, float | None]) -> list[dict[str, str]]:
	changes = []
	for p_str, before_mtime in before.items():
		p = Path(p_str)
		if not p.exists():
			continue
		after_mtime = p.stat().st_mtime
		if before_mtime is None:
			status = "created"
		elif after_mtime > before_mtime + 1e-6:
			status = "updated"
		else:
			continue
		changes.append(
			{
				"status": status,
				"file": p.name,
				"path": str(p),
				"modified_at": datetime.fromtimestamp(after_mtime).strftime("%Y-%m-%d %H:%M:%S"),
			}
		)
	return changes


def render_page() -> None:
	st.set_page_config(page_title="Pilotage Pipelines", layout="wide", page_icon="⚙️")

	if "pipeline_running" not in st.session_state:
		st.session_state.pipeline_running = False
	if "pending_run" not in st.session_state:
		st.session_state.pending_run = None
	if "last_resume" not in st.session_state:
		st.session_state.last_resume = []
	if "last_csv_changes" not in st.session_state:
		st.session_state.last_csv_changes = []
	if "last_logs" not in st.session_state:
		st.session_state.last_logs = []

	st.title("Pilotage des Pipelines")
	st.caption("Lancer un pipeline complet ou une phase unique, pour un trajet ou tous les trajets.")

	is_running = st.session_state.pipeline_running
	if is_running:
		st.warning("Pipeline en cours: les parametres sont verrouilles jusqu'a la fin de l'execution.")

	trajets = list_trajets()
	if not trajets:
		st.error("Aucun trajet trouve dans le dossier GroundTruth.")
		return

	col_a, col_b = st.columns(2)
	with col_a:
		mode = st.radio(
			"Mode d'execution",
			["Pipeline complet", "Phase unique"],
			horizontal=True,
			disabled=is_running,
		)
	with col_b:
		cible = st.radio(
			"Cible",
			["Un trajet", "Tous les trajets"],
			horizontal=True,
			disabled=is_running,
		)

	phase_choisie = None
	if mode == "Phase unique":
		phase_choisie = st.radio("Phase a executer", list(PHASES.keys()), disabled=is_running)

	if cible == "Un trajet":
		selected_trajets = [st.selectbox("Trajet", trajets, disabled=is_running)]
	else:
		selected_trajets = st.multiselect("Trajets", trajets, default=trajets, disabled=is_running)

	stop_on_error = st.checkbox(
		"Arreter a la premiere erreur (mode pipeline complet)",
		value=True,
		disabled=is_running,
	)
	max_live_lines = st.slider(
		"Nombre de lignes de logs live visibles",
		min_value=50,
		max_value=1000,
		value=300,
		step=50,
		disabled=is_running,
	)

	show_lidar_params = mode == "Pipeline complet" or (
		mode == "Phase unique" and phase_choisie == "Pipeline Labelisation LiDAR"
	)

	label_workers = int(N_WORKERS)
	label_chunk_size = int(CHUNK_SIZE)
	label_verify_integrity = False
	label_extract_features = False
	label_spatial_mode = "circle"
	radius = 10.0
	corridor_width = 6.0
	corridor_length = 30.0

	if show_lidar_params:
		with st.expander("Parametres du pipeline Labelisation LiDAR", expanded=False):
			label_workers = st.number_input(
				"Workers",
				min_value=1,
				max_value=64,
				value=int(N_WORKERS),
				step=1,
				disabled=is_running,
			)
			label_chunk_size = st.number_input(
				"Chunk size telechargement",
				min_value=1024,
				max_value=100_000_000,
				value=int(CHUNK_SIZE),
				step=1024,
				disabled=is_running,
			)
			label_verify_integrity = st.checkbox("Verifier l'integrite des tuiles", value=False, disabled=is_running)
			label_extract_features = st.checkbox(
				"Forcer la re-extraction des features LiDAR",
				value=False,
				disabled=is_running,
			)
			label_spatial_mode = st.selectbox(
				"Mode spatial pour la selection des tuiles",
				["circle", "corridor"],
				index=0,
				help="Le mode 'corridor' est plus rapide et plus representatif de la zone d'interet, mais peut manquer des points en cas de trajectoire sinueuse ou de decrochage GNSS.",
				disabled=is_running,
				on_change=None,  # Evite de reset les sliders de rayon/couloir lors du changement de mode
			)
			if label_spatial_mode == "circle":
				radius = st.slider(
					"Rayon de recherche (m)",
					min_value=1.0,
					max_value=50.0,
					value=20.0,
					step=1.0,
					disabled=is_running,
				)
			elif label_spatial_mode == "corridor":
				corridor_width = st.slider(
					"Largeur couloir (m)",
					min_value=1.0,
					max_value=50.0,
					value=6.0,
					step=1.0,
					disabled=is_running,
				)
				corridor_length = st.slider(
					"Demi-longueur couloir (m)",
					min_value=5.0,
					max_value=120.0,
					value=30.0,
					step=1.0,
					disabled=is_running,
				)

	phase_params = {
		"nb_workers": int(label_workers),
		"chunk_size": int(label_chunk_size),
		"verifier_integrite": bool(label_verify_integrity),
		"extract_features": bool(label_extract_features),
		"spatial_mode": str(label_spatial_mode),
		"search_radius": float(radius),
		"corridor_width": float(corridor_width),
		"corridor_length": float(corridor_length),
		"verbose": True,
	}

	if show_lidar_params and selected_trajets:
		preview_traj = selected_trajets[0]
		try:
			preview_cfg = get_traj_paths(preview_traj)
			latest_path, latest_payload = load_latest_labelisation_run_params(preview_cfg, preview_traj)
			with st.expander("Dernier JSON de parametres (interim)", expanded=False):
				if latest_payload is None:
					st.info("Aucun fichier de parametres precedent trouve dans l'interim pour ce trajet.")
				else:
					st.caption(f"Source: {latest_path}")
					st.json(latest_payload)
		except Exception as e:
			st.info(f"Impossible de charger le dernier JSON de parametres: {e}")

	lancer = st.button("Lancer", type="primary", width="stretch", disabled=is_running)
	if lancer and not is_running:
		st.session_state.pending_run = {
			"mode": mode,
			"phase_choisie": phase_choisie,
			"selected_trajets": selected_trajets,
			"stop_on_error": stop_on_error,
			"phase_params": phase_params,
			"max_live_lines": int(max_live_lines),
		}
		st.session_state.pipeline_running = True
		st.rerun()

	if not is_running:
		if st.session_state.last_resume:
			st.divider()
			st.subheader("Resume")
			st.dataframe(st.session_state.last_resume, width="stretch")

			st.subheader("CSV crees / modifies")
			if st.session_state.last_csv_changes:
				st.dataframe(st.session_state.last_csv_changes, width="stretch")
			else:
				st.info("Aucun CSV cree ou modifie detecte pour cette execution.")

		if st.session_state.last_logs:
			st.subheader("Logs du dernier run")
			for item in st.session_state.last_logs:
				label = f"{item['trajet']} - {item['phase']} ({item['status']})"
				with st.expander(label):
					st.code(item.get("logs", "(aucun log)"))
		return

	run_cfg = st.session_state.pending_run
	if not run_cfg:
		st.session_state.pipeline_running = False
		return

	mode = run_cfg["mode"]
	phase_choisie = run_cfg["phase_choisie"]
	selected_trajets = run_cfg["selected_trajets"]
	stop_on_error = run_cfg["stop_on_error"]
	phase_params = run_cfg["phase_params"]
	max_live_lines = run_cfg["max_live_lines"]

	if not selected_trajets:
		st.warning("Selectionne au moins un trajet.")
		st.session_state.pipeline_running = False
		st.session_state.pending_run = None
		return

	st.info(f"Execution depuis: {APPS_ROOT}")
	st.caption(
		"Interpreteurs detectes - RINEX: "
		f"{_resolve_python_for_env('rinex')} | Label/GNSS: {_resolve_python_for_env('label')}"
	)

	resume = []
	all_csv_changes = []
	all_logs = []
	progress = st.progress(0)

	try:
		for idx, traj_id in enumerate(selected_trajets, start=1):
			with st.container(border=True):
				st.subheader(f"Trajet: {traj_id}")

				if mode == "Pipeline complet":
					expected_paths = []
					for ph in PHASES:
						expected_paths.extend(_expected_csv_outputs(traj_id, ph))
					before = _snapshot_mtimes(expected_paths)

					phase_log_placeholders = {}
					for phase_name in PHASES:
						st.markdown(f"⏳ {phase_name}")
						phase_log_placeholders[phase_name] = st.empty()

					ok, details = run_pipeline_complet(
						traj_id,
						stop_on_error=stop_on_error,
						phase_log_placeholders=phase_log_placeholders,
						phase_params=phase_params,
						max_live_lines=max_live_lines,
					)
					for phase_name, phase_ok, logs in details:
						icon = "✅" if phase_ok else "❌"
						st.markdown(f"{icon} {phase_name}")
						with st.expander(f"Logs - {phase_name}"):
							st.code(logs or "(aucun log)")
						all_logs.append(
							{
								"trajet": traj_id,
								"phase": phase_name,
								"status": "OK" if phase_ok else "ERROR",
								"logs": logs,
							}
						)

					csv_changes = _detect_csv_changes(before)
					for item in csv_changes:
						item["trajet"] = traj_id
						item["phase"] = "ALL"
					all_csv_changes.extend(csv_changes)

					resume.append(
						{
							"trajet": traj_id,
							"mode": mode,
							"phase": "ALL",
							"status": "OK" if ok else "ERROR",
							"csv_changes": len(csv_changes),
						}
					)
				else:
					expected_paths = _expected_csv_outputs(traj_id, phase_choisie)
					before = _snapshot_mtimes(expected_paths)

					log_placeholder = st.empty()
					ok, logs = run_module(
						phase_choisie,
						traj_id,
						log_placeholder=log_placeholder,
						phase_params=phase_params,
						max_live_lines=max_live_lines,
					)
					icon = "✅" if ok else "❌"
					st.markdown(f"{icon} {phase_choisie}")
					with st.expander("Logs"):
						st.code(logs or "(aucun log)")

					all_logs.append(
						{
							"trajet": traj_id,
							"phase": phase_choisie,
							"status": "OK" if ok else "ERROR",
							"logs": logs,
						}
					)

					csv_changes = _detect_csv_changes(before)
					for item in csv_changes:
						item["trajet"] = traj_id
						item["phase"] = phase_choisie
					all_csv_changes.extend(csv_changes)

					resume.append(
						{
							"trajet": traj_id,
							"mode": mode,
							"phase": phase_choisie,
							"status": "OK" if ok else "ERROR",
							"csv_changes": len(csv_changes),
						}
					)

			progress.progress(idx / len(selected_trajets))
	finally:
		st.session_state.pipeline_running = False
		st.session_state.pending_run = None

	st.session_state.last_resume = resume
	st.session_state.last_csv_changes = all_csv_changes
	st.session_state.last_logs = all_logs
	st.rerun()


render_page()
