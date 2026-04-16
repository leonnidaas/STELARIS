import re 
from pathlib import Path
import pandas as pd
from utils import get_traj_paths

def scenario_start_label(sid: str) -> str:
    m = re.search(r"__SCENARIO_(\d{8})_(\d{6}(?:\.\d+)?)", sid)
    if not m:
        return sid

    date_raw = m.group(1)
    time_raw = m.group(2)
    date_fmt = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
    time_fmt = f"{time_raw[:2]}:{time_raw[2:4]}:{time_raw[4:]}"
    return f"{date_fmt} {time_fmt}"


def load_traj_config(traj_id: str) -> dict:
    return get_traj_paths(traj_id)


def load_sorted_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "time_utc" in df.columns:
        df = df.sort_values("time_utc", kind="stable")
    return df
