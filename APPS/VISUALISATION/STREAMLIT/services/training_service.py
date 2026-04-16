import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from utils import PYTHON_LABELISATION_INTERPRETER, get_dataset_path, list_traj_ids, ROOT_PATH


APPS_ROOT = ROOT_PATH / "APPS"
TRAINING_ROOT = APPS_ROOT / "ENTRAINEMENT_MODELES"

TRAINING_SCRIPTS = {
    "Preprocessing": "ENTRAINEMENT_MODELES.preprocessing",
    "Optimisation GRU (Optuna)": "ENTRAINEMENT_MODELES.optimize_gru",
    "Entrainement GRU": "ENTRAINEMENT_MODELES.train_gru",
    "Entrainement XGBoost": "ENTRAINEMENT_MODELES.train_xgboost",
    "Evaluation": "ENTRAINEMENT_MODELES.evaluate",
}


def list_trajets() -> list[str]:
    return list_traj_ids()


def trajet_unique_id(trajet_id: str) -> str:
    if "__" in trajet_id:
        return trajet_id.split("__", 1)[0]
    return trajet_id


def build_trajet_groups(trajets: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for trajet in sorted(trajets):
        key = trajet_unique_id(trajet)
        groups.setdefault(key, []).append(trajet)
    return groups


def expand_unique_trajets(selected_unique: list[str], grouped_trajets: dict[str, list[str]]) -> list[str]:
    expanded: list[str] = []
    for unique_id in selected_unique:
        expanded.extend(grouped_trajets.get(unique_id, []))
    return expanded


def build_artefacts_map(dataset_name: str) -> dict[str, Path]:
    cfg = get_dataset_path(dataset_name)
    eval_dir = cfg["output_dir"] / "evaluations"
    return {
        "Donnees pretraitees": cfg["preprocessed_data"],
        "Classes": cfg["classes_param"],
        "Scaler": cfg["scaler_param"],
        "Label Encoder": cfg["label_encoder_path"],
        "Metadata Dataset": cfg["metadata"],
        "Optuna GRU (best params)": cfg["output_dir"] / "gru_optuna_best_params.json",
        "Rapport Evaluation (latest)": eval_dir / "comparison_report.json",
    }


def resolve_training_python() -> str:
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


def snapshot_artefacts(artefacts_map: dict[str, Path]) -> dict[str, float | None]:
    return {
        name: (path.stat().st_mtime if path.exists() else None)
        for name, path in artefacts_map.items()
    }


def artefact_changes(before: dict[str, float | None], artefacts_map: dict[str, Path]) -> list[dict[str, str]]:
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


def render_live_log(log_placeholder, lines: list[str], max_lines: int = 300) -> None:
    if not lines:
        log_placeholder.code("(demarrage...)")
        return
    log_placeholder.code("\n".join(lines[-max_lines:]))


def run_script(script_name: str, script_args: list[str], log_placeholder, max_live_lines: int = 300) -> tuple[bool, str]:
    python_exec = resolve_training_python()
    cmd = [python_exec, "-m", script_name] + script_args

    env = dict(subprocess.os.environ)
    env["MPLBACKEND"] = "Agg"

    lines = [f"[python] {python_exec}", f"[cwd] {APPS_ROOT}", f"[cmd] {' '.join(cmd)}"]
    render_live_log(log_placeholder, lines, max_lines=max_live_lines)

    proc = subprocess.Popen(
        cmd,
        cwd=APPS_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        render_live_log(log_placeholder, lines, max_lines=max_live_lines)

    return_code = proc.wait()
    logs = "\n".join(lines).strip()
    return return_code == 0, logs


def _find_running_training_pids() -> list[int]:
    pids: set[int] = set()

    # Fast path with pgrep: match module names directly to avoid patterns starting with '-'.
    for script_module in TRAINING_SCRIPTS.values():
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", "--", script_module],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            continue

        for raw in out.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                pids.add(int(raw))
            except ValueError:
                continue

    # Fallback path with ps in case pgrep is unavailable or misses wrapped commands.
    if not pids:
        try:
            ps_out = subprocess.check_output(
                ["ps", "-eo", "pid=,args="],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            markers = tuple(TRAINING_SCRIPTS.values())
            for raw in ps_out.splitlines():
                line = raw.strip()
                if not line:
                    continue
                if not any(m in line for m in markers):
                    continue

                parts = line.split(maxsplit=1)
                if not parts:
                    continue
                try:
                    pid = int(parts[0])
                except ValueError:
                    continue
                if pid != subprocess.os.getpid():
                    pids.add(pid)
        except Exception:
            pass

    return sorted(pids)


def kill_running_training_processes() -> dict[str, int]:
    pids = _find_running_training_pids()
    if not pids:
        return {"found": 0, "terminated": 0, "killed": 0}

    terminated = 0
    killed = 0

    def _kill_children(parent_pid: int, sig: str) -> None:
        try:
            children_out = subprocess.check_output(
                ["pgrep", "-P", str(parent_pid)],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return

        for child_raw in children_out.splitlines():
            child_raw = child_raw.strip()
            if not child_raw:
                continue
            subprocess.run(["kill", sig, child_raw], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for pid in pids:
        try:
            _kill_children(pid, "-TERM")
            subprocess.run(["kill", "-TERM", str(pid)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            terminated += 1
        except Exception:
            continue

    time.sleep(0.8)

    still_running = _find_running_training_pids()
    for pid in still_running:
        try:
            _kill_children(pid, "-KILL")
            subprocess.run(["kill", "-KILL", str(pid)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            killed += 1
        except Exception:
            continue

    return {"found": len(pids), "terminated": terminated, "killed": killed}


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


def parse_eval_metrics(eval_logs: str) -> pd.DataFrame:
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

    return pd.DataFrame(rows)


def extract_classification_reports(eval_logs: str) -> dict[str, str]:
    reports = {}
    blocks = {
        "GRU": r"--- GRU – Classification Report \(Test\) ---",
        "XGBoost": r"--- XGBoost – Classification Report \(Test\) ---",
    }

    for model_name, header in blocks.items():
        start = re.search(header, eval_logs)
        if not start:
            continue

        rest = eval_logs[start.end() :].lstrip("\n")
        end = re.search(r"\n--- .*Classification Report .*---|\n={20,}|\n\[", rest)
        reports[model_name] = rest[: end.start()].strip() if end else rest.strip()

    return reports
