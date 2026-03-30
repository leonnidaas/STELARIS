import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import GT_DIR, PARAMS_ENTRAINEMENT, PYTHON_LABELISATION_INTERPRETER, get_dataset_path


APPS_ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = APPS_ROOT / "ENTRAINEMENT_MODELES"

TRAINING_SCRIPTS = {
	"Preprocessing": "ENTRAINEMENT_MODELES.preprocessing",
	"Entrainement GRU": "ENTRAINEMENT_MODELES.train_gru",
	"Entrainement XGBoost": "ENTRAINEMENT_MODELES.train_xgboost",
	"Evaluation": "ENTRAINEMENT_MODELES.evaluate",
}

def _list_trajets() -> list[str]:
	if not GT_DIR.exists():
		return []
	return sorted([p.name for p in GT_DIR.iterdir() if p.is_dir()])


def _build_artefacts_map(dataset_name: str) -> dict[str, Path]:
	cfg = get_dataset_path(dataset_name)
	eval_dir = cfg["output_dir"] / "evaluations"
	return {
		"Donnees pretraitees": cfg["preprocessed_data"],
		"Classes": cfg["classes_param"],
		"Scaler": cfg["scaler_param"],
		"Label Encoder": cfg["label_encoder_path"],
		"Metadata Dataset": cfg["metadata"],
		"Rapport Evaluation (latest)": eval_dir / "comparison_report.json",
	}


def _resolve_training_python() -> str:
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


def _snapshot_artefacts(artefacts_map: dict[str, Path]) -> dict[str, float | None]:
	return {
		name: (path.stat().st_mtime if path.exists() else None)
		for name, path in artefacts_map.items()
	}


def _artefact_changes(before: dict[str, float | None], artefacts_map: dict[str, Path]) -> list[dict[str, str]]:
	changes = []
	for name, path in artefacts_map.items():
		if not path.exists():
			continue
		after_mtime = path.stat().st_mtime
		before_mtime = before.get(name)

		if before_mtime is None:
			status = "created"
		elif after_mtime > before_mtime + 1e-6:
			status = "updated"
		else:
			continue

		changes.append(
			{
				"artefact": name,
				"status": status,
				"path": str(path),
				"modified_at": datetime.fromtimestamp(after_mtime).strftime("%Y-%m-%d %H:%M:%S"),
			}
		)
	return changes


def _render_live_log(log_placeholder, lines: list[str], max_lines: int = 300) -> None:
	if not lines:
		log_placeholder.code("(demarrage...)")
		return
	log_placeholder.code("\n".join(lines[-max_lines:]))


def run_script(
	script_name: str,
	script_args: list[str],
	log_placeholder,
	max_live_lines: int = 300,
) -> tuple[bool, str]:
	python_exec = _resolve_training_python()
	cmd = [python_exec, "-m", script_name] + script_args

	env = dict(**dict(), **{})
	env.update({
		"MPLBACKEND": "Agg",  # Evite les fenetres matplotlib cote serveur
	})

	lines = [f"[python] {python_exec}", f"[cwd] {APPS_ROOT}", f"[cmd] {' '.join(cmd)}"]
	_render_live_log(log_placeholder, lines, max_lines=max_live_lines)

	proc = subprocess.Popen(
		cmd,
		cwd=APPS_ROOT,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		bufsize=1,
		env={**dict(**subprocess.os.environ), **env},
	)

	assert proc.stdout is not None
	for line in proc.stdout:
		lines.append(line.rstrip("\n"))
		_render_live_log(log_placeholder, lines, max_lines=max_live_lines)

	return_code = proc.wait()
	logs = "\n".join(lines).strip()
	return return_code == 0, logs


def run_training_pipeline(
	selected_steps: list[str],
	step_args_map: dict[str, list[str]],
	log_placeholder,
	max_live_lines: int,
	continue_on_error: bool,
) -> tuple[bool, list[tuple[str, bool, str]]]:
	results: list[tuple[str, bool, str]] = []
	global_ok = True

	for step in selected_steps:
		script = TRAINING_SCRIPTS[step]
		ok, logs = run_script(
			script,
			script_args=step_args_map.get(step, []),
			log_placeholder=log_placeholder,
			max_live_lines=max_live_lines,
		)
		results.append((step, ok, logs))
		if not ok:
			global_ok = False
			if not continue_on_error:
				break

	return global_ok, results


def _parse_eval_metrics(eval_logs: str) -> pd.DataFrame:
	section_re = re.compile(r"^\s*(GRU|XGBoost)\s+—\s+(Train|Test)\s+set\s*$")
	metric_re = re.compile(
		r"^\s*(Accuracy|Balanced Accuracy|F1 \(weighted\)|F1 \(macro\)|Cross-Entropy|AUC \(weighted\))\s*:\s*([0-9]*\.?[0-9]+)\s*$"
	)

	current_model = None
	current_split = None
	rows = []

	for raw_line in eval_logs.splitlines():
		line = raw_line.strip()
		m_sec = section_re.match(line)
		if m_sec:
			current_model, current_split = m_sec.group(1), m_sec.group(2)
			continue

		m_metric = metric_re.match(line)
		if m_metric and current_model and current_split:
			rows.append(
				{
					"Modele": current_model,
					"Split": current_split,
					"Metrique": m_metric.group(1),
					"Valeur": float(m_metric.group(2)),
				}
			)

	if not rows:
		return pd.DataFrame()

	df = pd.DataFrame(rows)
	return df


def _extract_classification_reports(eval_logs: str) -> dict[str, str]:
	reports = {}
	blocks = {
		"GRU": r"--- GRU – Classification Report \(Test\) ---",
		"XGBoost": r"--- XGBoost – Classification Report \(Test\) ---",
	}

	for model_name, header in blocks.items():
		start = re.search(header, eval_logs)
		if not start:
			continue

		rest = eval_logs[start.end():].lstrip("\n")
		end = re.search(r"\n--- .*Classification Report .*---|\n={20,}|\n\[", rest)
		reports[model_name] = rest[: end.start()].strip() if end else rest.strip()

	return reports


def render_page() -> None:
	st.set_page_config(page_title="Training IA", layout="wide", page_icon="🧠")

	if "training_running" not in st.session_state:
		st.session_state.training_running = False
	if "training_results" not in st.session_state:
		st.session_state.training_results = []
	if "training_artefact_changes" not in st.session_state:
		st.session_state.training_artefact_changes = []

	st.title("Training des modeles IA")
	st.caption("Pilotage du preprocessing, de l'entrainement GRU/XGBoost et de l'evaluation.")

	is_running = st.session_state.training_running
	if is_running:
		st.warning("Entrainement en cours, les parametres sont verrouilles.")

	left, right = st.columns([2, 1])
	with left:
		dataset_name = st.text_input(
			"Nom du dataset",
			value=datetime.now().strftime("%Y-%m-%d_%H-%M"),
			disabled=is_running,
			help="Le meme dataset_name sera utilise pour preprocessing, train et evaluation.",
		)

		selected_steps = st.multiselect(
			"Etapes a executer (ordre fixe)",
			options=list(TRAINING_SCRIPTS.keys()),
			default=list(TRAINING_SCRIPTS.keys()),
			disabled=is_running,
		)

	trajets_disponibles = _list_trajets()
	default_trajets = PARAMS_ENTRAINEMENT.get("trajets", ["BORDEAUX_COUTRAS", "MARTINE_01"])
	default_trajets = [t for t in default_trajets if t in trajets_disponibles]
	if not default_trajets:
		default_trajets = trajets_disponibles[:2] if len(trajets_disponibles) >= 2 else trajets_disponibles

	selected_trajets = st.multiselect(
		"Trajets pour creer le dataset (preprocessing)",
		options=trajets_disponibles,
		default=default_trajets,
		disabled=is_running,
		help="Utilise uniquement ces trajets lors de l'etape Preprocessing.",
	)
	with right:
		continue_on_error = st.checkbox(
			"Continuer si erreur",
			value=False,
			disabled=is_running,
			help="Si active, les etapes suivantes sont lancees meme en cas d'echec.",
		)

	window_size = st.number_input(
		"Window size (preprocessing)",
		min_value=3,
		max_value=121,
		value=int(PARAMS_ENTRAINEMENT.get("window_size", 5)),
		step=2,
		disabled=is_running,
	)

	max_live_lines = st.slider(
		"Lignes de logs visibles",
		min_value=50,
		max_value=1200,
		value=300,
		step=50,
		disabled=is_running,
	)

	st.subheader("Etat des artefacts")
	artefacts_map = _build_artefacts_map(dataset_name)
	artefact_rows = []
	for name, path in artefacts_map.items():
		artefact_rows.append(
			{
				"Artefact": name,
				"Present": "Oui" if path.exists() else "Non",
				"Chemin": str(path),
			}
		)
	st.dataframe(pd.DataFrame(artefact_rows), width='stretch', hide_index=True)

	run_clicked = st.button(
		"Lancer l'entrainement",
		type="primary",
		disabled=is_running,
		width='stretch',
	)

	if run_clicked:
		if not selected_steps:
			st.error("Selectionne au moins une etape.")
		elif "Preprocessing" in selected_steps and not selected_trajets:
			st.error("Selectionne au moins un trajet pour le preprocessing.")
		elif not TRAINING_ROOT.exists():
			st.error(f"Dossier introuvable: {TRAINING_ROOT}")
		else:
			st.session_state.training_running = True
			st.session_state.training_results = []
			st.session_state.training_artefact_changes = []

			artefacts_before = _snapshot_artefacts(artefacts_map)

			step_args_map = {
				"Preprocessing": [
					"--dataset-name", dataset_name,
					"--window-size", str(int(window_size)),
					"--trajets", *selected_trajets,
				],
				"Entrainement GRU": ["--dataset-name", dataset_name],
				"Entrainement XGBoost": ["--dataset-name", dataset_name],
				"Evaluation": ["--dataset-name", dataset_name],
			}

			progress = st.progress(0, text="Demarrage du pipeline...")
			log_placeholder = st.empty()
			total = len(selected_steps)

			results = []
			global_ok = True
			for idx, step in enumerate(selected_steps, start=1):
				progress.progress(
					int((idx - 1) / total * 100),
					text=f"Execution [{idx}/{total}] {step}",
				)

				script_name = TRAINING_SCRIPTS[step]
				script_args = step_args_map.get(step, [])
				ok, logs = run_script(
					script_name,
					script_args=script_args,
					log_placeholder=log_placeholder,
					max_live_lines=max_live_lines,
				)

				results.append((step, ok, logs))
				if not ok:
					global_ok = False
					if not continue_on_error:
						break

			st.session_state.training_results = results
			st.session_state.training_artefact_changes = _artefact_changes(artefacts_before, artefacts_map)
			st.session_state.training_running = False

			progress.progress(100, text="Execution terminee")
			if global_ok:
				st.success("Pipeline training termine avec succes.")
			else:
				st.error("Pipeline termine avec au moins une erreur.")

	st.divider()
	st.subheader("Resultats")

	results = st.session_state.training_results
	if not results:
		st.info("Aucune execution lancee dans cette session.")
	else:
		resume_rows = [
			{"Etape": step, "Statut": "OK" if ok else "ECHEC"}
			for step, ok, _ in results
		]
		st.dataframe(pd.DataFrame(resume_rows), width='stretch', hide_index=True)

		changes = st.session_state.training_artefact_changes
		if changes:
			st.markdown("**Artefacts crees ou modifies**")
			st.dataframe(pd.DataFrame(changes), width='stretch', hide_index=True)

		eval_logs = ""
		for step, ok, logs in results:
			if step == "Evaluation" and ok:
				eval_logs = logs
				break

		if eval_logs:
			st.markdown("**Metriques d'evaluation**")
			metrics_df = _parse_eval_metrics(eval_logs)
			if not metrics_df.empty:
				pivot_df = metrics_df.pivot_table(
					index=["Modele", "Split"],
					columns="Metrique",
					values="Valeur",
					aggfunc="first",
				).reset_index()
				st.dataframe(pivot_df, width='stretch', hide_index=True)
			else:
				st.warning("Impossible d'extraire automatiquement les metriques depuis les logs.")

			reports = _extract_classification_reports(eval_logs)
			for model_name, report in reports.items():
				with st.expander(f"Classification report - {model_name}", expanded=False):
					st.code(report)

		st.markdown("**Logs par etape**")
		for step, ok, logs in results:
			icon = "✅" if ok else "❌"
			with st.expander(f"{icon} {step}", expanded=not ok):
				st.code(logs if logs else "(aucun log)")


render_page()
