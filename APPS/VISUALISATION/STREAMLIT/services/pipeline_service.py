import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from utils import PYTHON_LABELISATION_INTERPRETER, PYTHON_RINEX_INTERPRETER, get_traj_paths, list_traj_ids, ROOT_PATH
from VISUALISATION.STREAMLIT.modules.selection_scenario import scenario_start_label 

APPS_ROOT = ROOT_PATH / "APPS"
LIVE_RENDER_MIN_INTERVAL_SEC = 0.2
MAX_BUFFERED_LOG_LINES = 4000

PHASES = {
    "Traitement RINEX": {
        "module": "TRAITEMENT_RINEX.app",
        "env": "rinex",
    },
    "Extraction features GNSS": {
        "module": "EXTRACTION_DES_FEATURES_GNSS.app",
        "env": "label",
    },
    "Fusion GT + GNSS": {
        "module": "FUSION.app",
        "env": "label",
    },
    "Pipeline Labelisation LiDAR": {
        "module": "LABELISATION_AUTO_LIDAR_HD_IGN.app",
        "env": "label",
    },
    "Pipeline Labelisation OSM": {
        "module": "LABELISATION_OSM.app",
        "env": "label",
    },
}


def _find_running_pipeline_pids() -> list[int]:
    pids: set[int] = set()

    try:
        ps_out = subprocess.check_output(
            ["ps", "-eo", "pid=,args="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        markers = tuple(phase_cfg["module"] for phase_cfg in PHASES.values())
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


def kill_running_pipeline_processes() -> dict[str, int]:
    pids = _find_running_pipeline_pids()
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

    still_running = _find_running_pipeline_pids()
    for pid in still_running:
        try:
            _kill_children(pid, "-KILL")
            subprocess.run(["kill", "-KILL", str(pid)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            killed += 1
        except Exception:
            continue

    return {"found": len(pids), "terminated": terminated, "killed": killed}


def list_trajets() -> list[str]:
    return list_traj_ids()



def build_trajet_scenarios_map(scenarios: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for scenario_id in scenarios:
        trajet_key = scenario_id.split("__", 1)[0] if "__" in scenario_id else scenario_id
        out.setdefault(trajet_key, []).append(scenario_id)
    for k in out:
        out[k] = sorted(out[k])
    return out


def resolve_python_for_env(env_name: str) -> str:
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


def render_live_log(log_placeholder, lines: list[str], max_lines: int = 200) -> None:
    if not lines:
        log_placeholder.code("(demarrage...)")
        return
    log_placeholder.code("\n".join(lines[-max_lines:]))


def _resolve_phase_params(phase_name: str, phase_params: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(phase_params, dict):
        return None

    scoped = phase_params.get(phase_name)
    if isinstance(scoped, dict):
        return scoped

    return phase_params


def build_phase_extra_args(phase_name: str, phase_params: dict[str, Any] | None) -> list[str]:
    scoped_params = _resolve_phase_params(phase_name, phase_params)
    if phase_name not in {
        "Extraction features GNSS",
        "Pipeline Labelisation LiDAR",
        "Pipeline Labelisation OSM",
    } or not scoped_params:
        return []

    return ["--options-json", json.dumps(scoped_params)]


def logs_indicate_failure(logs: str) -> bool:
    if not logs:
        return False
    lower_logs = logs.lower()

    hard_markers = [
        "traceback (most recent call last):",
        "le pipeline a rencontre des erreurs.",
        "le pipeline a rencontré des erreurs.",
        "pipeline a rencontre des erreurs",
        "pipeline a rencontré des erreurs",
    ]
    if any(marker in lower_logs for marker in hard_markers):
        return True

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
    python_exec = resolve_python_for_env(phase_cfg["env"])
    cmd = [python_exec, "-m", module_name, "--traj", traj_id] + build_phase_extra_args(phase_name, phase_params)
    header = f"[python] {python_exec}\n[cmd] {' '.join(cmd)}"
    lines = [header]

    if log_placeholder is not None:
        render_live_log(log_placeholder, lines, max_lines=max_live_lines)

    dropped_line_count = 0
    last_render_at = time.monotonic()

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
        if len(lines) > MAX_BUFFERED_LOG_LINES:
            overflow = len(lines) - MAX_BUFFERED_LOG_LINES
            # Keep the command header (index 0) and the most recent output lines.
            del lines[1 : 1 + overflow]
            dropped_line_count += overflow

        if log_placeholder is not None:
            now = time.monotonic()
            if now - last_render_at >= LIVE_RENDER_MIN_INTERVAL_SEC:
                render_live_log(log_placeholder, lines, max_lines=max_live_lines)
                last_render_at = now

    return_code = proc.wait()
    final_lines = lines.copy()
    if dropped_line_count > 0:
        final_lines.insert(1, f"... {dropped_line_count} lignes de log masquees pour limiter la memoire ...")

    if log_placeholder is not None:
        render_live_log(log_placeholder, final_lines, max_lines=max_live_lines)

    logs = "\n".join(final_lines).strip()
    ok = return_code == 0 and not logs_indicate_failure(logs)
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


def expected_csv_outputs(traj_id: str, phase_name: str) -> list[Path]:
    cfg = get_traj_paths(traj_id)

    mapping = {
        "Traitement RINEX": [cfg["raw_gnss"], cfg["space_vehicule_info"]],
        "Extraction features GNSS": [cfg["gnss_features_csv"]],
        "Fusion GT + GNSS": [cfg["sync_csv"]],
        "Pipeline Labelisation LiDAR": [
            cfg["lidar_features_csv"],
            cfg["fusion_features_csv"],
            cfg["lidar_labels_csv"],
            cfg["labels_plus_features_csv"],
            cfg["final_fusion_csv"],
        ],
        "Pipeline Labelisation OSM": [
            cfg["osm_pbf"],
            cfg["osm_features_csv"],
            cfg["osm_labels_csv"],
            cfg["final_fusion_osm_csv"],
        ],
    }
    return [Path(p) for p in mapping.get(phase_name, [])]


def snapshot_mtimes(paths: list[Path]) -> dict[str, float | None]:
    return {str(p): (p.stat().st_mtime if p.exists() else None) for p in paths}


def detect_csv_changes(before: dict[str, float | None]) -> list[dict[str, str]]:
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
