import json
import os
from pathlib import Path
import h3
import numpy as np
import pandas as pd
from utils import iter_gt_scenario_dirs, iter_scenario_dirs, COLUMN_MAPPING, standardize_dataframe
from ENTRAINEMENT_MODELES.preprocessing import assign_geographic_segments
from os import listdir

LAT_ALIASES = COLUMN_MAPPING["latitude"]
LON_ALIASES = COLUMN_MAPPING["longitude"]


def count_scenarios(path: Path) -> int:
    return len(iter_scenario_dirs(path))

def count_unique_routes(path: Path) -> int:
    route_keys = set()
    for scenario_dir in iter_scenario_dirs(path):
        route_key = scenario_dir.name.split("__", 1)[0]
        route_keys.add(route_key)
    return len(route_keys)


def get_unique_cells_from_file(file_path, res=10):
    """Extrait les IDs de cellules H3 uniques d'un seul fichier."""
    df = pd.read_csv(file_path) # Ou pd.read_csv
    
    # Critique : Filtrage obligatoire de la GT avant extraction
    df = df[df['gt_status'] == 'VALID'] 
    
    # On vectorise l'opération h3
    cells = [h3.geo_to_h3(lat, lon, res) for lat, lon in zip(df['lat'], df['lon'])]
    return set(cells)

def estimate_total_km_from_files(file_paths, res=5):
    decimation = 10
    merged_df = pd.DataFrame()
    #On regarde tous les fichiers pour extraire les coordonnées et faire la segmentation géographique
    for file in file_paths.rglob("fusion*.csv"):
        df = pd.read_csv(file)
        df = df.iloc[::decimation]  # Décimation pour accélérer le traitement
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    if merged_df.empty:
        return 0

    merged_df = standardize_dataframe(merged_df)
    if "latitude_gt" not in merged_df.columns and "latitude" in merged_df.columns:
        merged_df.rename(columns={"latitude": "latitude_gt"}, inplace=True)
    if "longitude_gt" not in merged_df.columns and "longitude" in merged_df.columns:
        merged_df.rename(columns={"longitude": "longitude_gt"}, inplace=True)

    if "latitude_gt" not in merged_df.columns or "longitude_gt" not in merged_df.columns:
        return 0

    df = assign_geographic_segments(merged_df, grid_size_km=res)
    unique_segments = df['segment_id'].unique()
    km_uniques = len(unique_segments) * float(res)
    return  km_uniques  

def calculate_total_duration_from_files(file_paths):
    """
    retourne le total_duration en heures cumulées"""
    total_duration = 0.0
    for file in file_paths.rglob("fusion*.csv"):
        df = pd.read_csv(os.path.join(file_paths, file))
        time_start = pd.to_datetime(df['time_utc'].iloc[0], errors='coerce')
        time_end = pd.to_datetime(df['time_utc'].iloc[-1], errors='coerce')
        if pd.notna(time_start) and pd.notna(time_end):
            duration = (time_end - time_start).total_seconds() / 3600.0  # Convertir en heures
            total_duration += duration
    return int(total_duration)

def count_csv_recursive(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.rglob("*.csv")))


def pick_coord_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    cols = {str(c).strip().lower(): c for c in df.columns}

    lat_col = next((cols[a] for a in LAT_ALIASES if a in cols), None)
    lon_col = next((cols[a] for a in LON_ALIASES if a in cols), None)
    if lat_col is None or lon_col is None:
        return None
    return str(lat_col), str(lon_col)


def find_coord_columns_in_txt(df: pd.DataFrame) -> tuple[int, int] | None:
    best_pair = None
    best_score = -1

    for i in range(df.shape[1]):
        col_i = pd.to_numeric(df.iloc[:, i], errors="coerce")
        if col_i.notna().sum() < 10:
            continue

        for j in range(df.shape[1]):
            if j == i:
                continue

            col_j = pd.to_numeric(df.iloc[:, j], errors="coerce")
            if col_j.notna().sum() < 10:
                continue

            valid = col_i.notna() & col_j.notna()
            if valid.sum() < 10:
                continue

            score_normal = (
                (col_i.between(-90, 90))
                & (col_j.between(-180, 180))
                & valid
            ).sum()
            score_swap = (
                (col_j.between(-90, 90))
                & (col_i.between(-180, 180))
                & valid
            ).sum()

            if score_normal >= score_swap:
                score = int(score_normal)
                pair = (i, j)
            else:
                score = int(score_swap)
                pair = (j, i)

            if score > best_score:
                best_score = score
                best_pair = pair

    if best_pair is None or best_score < 10:
        return None
    return best_pair


def load_coords_from_gt_file(
    gt_file: Path,
    max_points: int | None = None,
    sample_stride: int | None = None,
) -> np.ndarray | None:
    try:
        if gt_file.suffix.lower() == ".csv":
            df = pd.read_csv(gt_file)
            picked = pick_coord_columns(df)
            if picked is None:
                return None
            lat_col, lon_col = picked
            coords = df[[lat_col, lon_col]].copy()
        else:
            df = pd.read_csv(gt_file, sep=r"\s+", comment="#", header=None)
            if df.shape[1] < 5:
                return None

            detected = find_coord_columns_in_txt(df)
            if detected is None:
                return None

            lat_idx, lon_idx = detected
            coords = df.iloc[:, [lat_idx, lon_idx]].copy()
            coords.columns = ["latitude", "longitude"]

        coords = coords.apply(pd.to_numeric, errors="coerce").dropna()
        if len(coords) < 2:
            return None

        coords = coords[
            (coords.iloc[:, 0] >= -90)
            & (coords.iloc[:, 0] <= 90)
            & (coords.iloc[:, 1] >= -180)
            & (coords.iloc[:, 1] <= 180)
        ]
        if len(coords) < 2:
            return None

        if sample_stride is not None and sample_stride > 1:
            arr = coords.iloc[::sample_stride].to_numpy(dtype=float)
        elif max_points is not None and max_points > 0:
            step = max(1, int(np.ceil(len(coords) / max_points)))
            arr = coords.iloc[::step].to_numpy(dtype=float)
        else:
            arr = coords.to_numpy(dtype=float)
        return arr
    except Exception:
        return None


def collect_all_routes(gt_dir: Path, sample_stride: int = 100) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for traj_dir in iter_gt_scenario_dirs(gt_dir):
        route_key = traj_dir.name.split("__", 1)[0]
        gt_files = list(traj_dir.glob("*.csv"))
        if not gt_files:
            continue

        coords = load_coords_from_gt_file(gt_files[0], sample_stride=sample_stride)
        if coords is None or len(coords) < 2:
            continue

        records.append({"route_key": route_key, "coords": coords, "scenario": traj_dir.name})

    return records


def load_grid_data(path: Path) -> dict | None:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
