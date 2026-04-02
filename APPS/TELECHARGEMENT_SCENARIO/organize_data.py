import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import reverse_geocoder as rg
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

APPS_DIR = Path(__file__).resolve().parents[1]
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from utils import GNSS_NAV, GNSS_OBS, GT_DIR


LAT_ALIASES = {
    "latitude",
    "lat",
    "lat_gt",
    "latitude_gt",
    "y_coord",
    "latitude[deg]",
}
LON_ALIASES = {
    "longitude",
    "lon",
    "long",
    "lon_gt",
    "longitude_gt",
    "x_coord",
    "longitude[deg]",
}


def classify_file(file_path: Path) -> str | None:
    low = file_path.name.lower()

    if low.endswith("raw.obs"):
        return "obs"
    if low.endswith("raw.nav"):
        return "nav"
    if low == "groundtruth.txt" or low.endswith("_groundtruth.csv"):
        return "groundtruth"
    if low == "speed.txt":
        return "speed"

    return None


def sanitize_token(value: str) -> str:
    return (
        value.upper()
        .replace(" ", "-")
        .replace("/", "_")
        .replace("\\", "_")
    )


def get_city_offline(lat: float, lon: float) -> str:
    CITY_ALIASES = {
    "BEGLES": "BORDEAUX",
    "CENON": "BORDEAUX",
    "LORMONT": "BORDEAUX",
    "COUTRAS": "COUTRAS", # On garde si c'est correct
}
    try:
        results = rg.search((lat, lon), verbose=False)
        city = sanitize_token(results[0]["name"])
        return CITY_ALIASES.get(city, city)
    except Exception:
        raise RuntimeError(f"Error reverse geocoding lat={lat}, lon={lon}: {sys.exc_info()[1]}")


def find_lat_lon_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    cols = {str(c).strip().lower(): c for c in df.columns}

    lat_col = None
    lon_col = None

    for alias in LAT_ALIASES:
        if alias in cols:
            lat_col = cols[alias]
            break

    for alias in LON_ALIASES:
        if alias in cols:
            lon_col = cols[alias]
            break

    if lat_col is None or lon_col is None:
        return None

    return str(lat_col), str(lon_col)


def load_groundtruth_txt(gt_file: Path) -> pd.DataFrame:
    """Charge un groundtruth.txt Delph sans fusion speed."""
    nav_cols = [
        "date", "time", "userData", "latitude", "longitude",
        "ellipsoidHeight", "orthometricHeight", "northingSd", "eastingSd",
        "heightSd", "northingEastingCv", "heading", "roll", "pitch",
        "headingSd", "rollSd", "pitchSd", "speedNorth", "speedEast",
        "speedVertical", "speedNorthSd", "speedEastSd", "speedVerticalSd",
    ]
    return pd.read_csv(gt_file, sep=r"\s+", comment="#", names=nav_cols)

def get_depart_fin_from_gt(gt_file: Path) -> tuple[str, str]:
    try:
        if gt_file.name.lower().endswith(".csv"):
            df = pd.read_csv(gt_file)
        else:
            df = load_groundtruth_txt(gt_file)

        lat_lon_cols = find_lat_lon_columns(df)
        if lat_lon_cols is None:
            raise ValueError("No latitude/longitude columns found in GT file")

        lat_col, lon_col = lat_lon_cols
        series = df[[lat_col, lon_col]].dropna()
        if len(series) < 2:
            raise ValueError("Not enough valid lat/lon points in GT file")

        dep_lat = float(series.iloc[0][lat_col])
        dep_lon = float(series.iloc[0][lon_col])
        fin_lat = float(series.iloc[-1][lat_col])
        fin_lon = float(series.iloc[-1][lon_col])

        depart = get_city_offline(dep_lat, dep_lon)
        fin = get_city_offline(fin_lat, fin_lon)
        return depart, fin
    
    except Exception:
        raise RuntimeError(f"Error processing GT file {gt_file} for depart/fin: {sys.exc_info()[1]}")



def scenario_id_from_dir(source_root: Path, dir_path: Path) -> str:
    rel = dir_path.relative_to(source_root)
    if rel == Path("."):
        return sanitize_token(source_root.name)
    return sanitize_token(str(rel).replace("/", "__"))


def line_id_from_scenario_name(scenario_name: str) -> str:
    """Extrait l'identifiant de ligne d'un nom scenario de type LINE__SCENARIO."""
    return scenario_name.split("__", 1)[0]


def choose_one_per_type(files: list[Path]) -> dict[str, Path]:
    by_type: dict[str, list[Path]] = {
        "obs": [],
        "nav": [],
        "groundtruth": [],
        "speed": [],
    }

    for f in files:
        kind = classify_file(f)
        if kind is None:
            continue
        by_type[kind].append(f)

    selected: dict[str, Path] = {}
    for kind, candidates in by_type.items():
        if candidates:
            selected[kind] = sorted(candidates, key=lambda p: p.name.lower())[0]

    return selected


def copy_or_move(src: Path, dst: Path, dry_run: bool, move: bool) -> None:
    action = "MOVE" if move else "COPY"

    if dry_run:
        return f"DRY-{action}"

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "SKIP"

    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)

    return action


def organize_source(
    source_root: Path,
    dry_run: bool = False,
    move: bool = False,
    max_scenarios: int | None = None,
    verbose: bool = False,
    group_by_line: bool = False,
) -> None:
    print(f"Scan recursive: {source_root}")

    processed = 0
    skipped = 0
    started_at = time.time()

    walk_entries = list(os.walk(source_root))
    total_dirs = len(walk_entries)

    pbar = tqdm(walk_entries, total=total_dirs, desc="Scan", unit="dir")
    for idx, (current_dir, _, filenames) in enumerate(pbar, start=1):
        if idx > 1:
            elapsed = max(time.time() - started_at, 1e-6)
            avg_per_dir = elapsed / idx
            remaining_secs = max((total_dirs - idx) * avg_per_dir, 0.0)
            eta_clock = datetime.fromtimestamp(time.time() + remaining_secs).strftime("%H:%M:%S")
            if hasattr(pbar, "set_postfix_str"):
                pbar.set_postfix_str(f"fin~{eta_clock}")

        dir_path = Path(current_dir)
        files = [dir_path / name for name in filenames]
        selected = choose_one_per_type(files)

        if "obs" not in selected or "nav" not in selected:
            skipped += 1
            continue
        if "groundtruth" not in selected and "speed" not in selected:
            skipped += 1
            continue

        scenario_id = scenario_id_from_dir(source_root, dir_path)

        if "groundtruth" in selected:
            depart, fin = get_depart_fin_from_gt(selected["groundtruth"])
        else:
            depart, fin = "UNKNOWN", "UNKNOWN"

        scenario_name = f"{depart}_{fin}__{scenario_id}"
        scenario_name = sanitize_token(scenario_name)
        line_id = line_id_from_scenario_name(scenario_name)

        processed += 1
        if verbose:
            print(f"\nScenario: {scenario_name}")
            print(f"Source dir: {dir_path}")

        if group_by_line:
            obs_dst = GNSS_OBS / line_id / scenario_name / selected["obs"].name
            nav_dst = GNSS_NAV / line_id / scenario_name / selected["nav"].name
        else:
            obs_dst = GNSS_OBS / scenario_name / selected["obs"].name
            nav_dst = GNSS_NAV / scenario_name / selected["nav"].name
        obs_status = copy_or_move(selected["obs"], obs_dst, dry_run=dry_run, move=move)
        nav_status = copy_or_move(selected["nav"], nav_dst, dry_run=dry_run, move=move)
        gt_status = "-"
        speed_status = "-"

        if "groundtruth" in selected:
            if group_by_line:
                gt_dst = GT_DIR / line_id / scenario_name / selected["groundtruth"].name
            else:
                gt_dst = GT_DIR / scenario_name / selected["groundtruth"].name
            gt_status = copy_or_move(selected["groundtruth"], gt_dst, dry_run=dry_run, move=move)

        if "speed" in selected:
            if group_by_line:
                speed_dst = GT_DIR / line_id / scenario_name / selected["speed"].name
            else:
                speed_dst = GT_DIR / scenario_name / selected["speed"].name
            speed_status = copy_or_move(selected["speed"], speed_dst, dry_run=dry_run, move=move)

        print(
            f"[{processed}] {scenario_name} | obs:{obs_status} nav:{nav_status} "
            f"gt:{gt_status} speed:{speed_status}"
        )

        if max_scenarios is not None and processed >= max_scenarios:
            print(f"\nReached max scenarios limit: {max_scenarios}")
            break

    if processed == 0:
        print("No scenario found with obs + nav + (groundtruth or speed).")
    else:
        print(f"\nDone. Scenarios processed: {processed} | scanned-no-match: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan recursively a source folder and organize obs/nav/groundtruth/speed "
            "into RAW folders with scenario name depart_fin_id_scenario."
        )
    )
    parser.add_argument("--source_dir", required=True, help="Source directory to scan")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions only, no file write",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copy",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Limit number of detected scenarios to process",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-scenario paths and warnings",
    )
    parser.add_argument(
        "--group-by-line",
        action="store_true",
        help="Store files under one extra folder level per line (LINE/SCENARIO)",
    )

    args = parser.parse_args()
    source_root = Path(args.source_dir).expanduser().resolve()

    if not source_root.exists() or not source_root.is_dir():
        raise SystemExit(f"Invalid source directory: {source_root}")

    organize_source(
        source_root=source_root,
        dry_run=args.dry_run,
        move=args.move,
        max_scenarios=args.max_scenarios,
        verbose=args.verbose,
        group_by_line=args.group_by_line,
    )


if __name__ == "__main__":
    main()
