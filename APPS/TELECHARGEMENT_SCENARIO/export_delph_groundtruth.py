import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from utils import GT_DIR, iter_scenario_dirs


def load_scenario_delph(scenario_dir: Path) -> pd.DataFrame:
    """Load and merge Delph groundtruth + speed exports for one scenario folder."""
    nav_file = scenario_dir / "groundtruth.txt"
    speed_file = scenario_dir / "speed.txt"

    nav_cols = [
        "date",
        "time",
        "userData",
        "latitude",
        "longitude",
        "ellipsoidHeight",
        "orthometricHeight",
        "northingSd",
        "eastingSd",
        "heightSd",
        "northingEastingCv",
        "heading",
        "roll",
        "pitch",
        "headingSd",
        "rollSd",
        "pitchSd",
        "speedNorth",
        "speedEast",
        "speedVertical",
        "speedNorthSd",
        "speedEastSd",
        "speedVerticalSd",
    ]
    df_nav = pd.read_csv(nav_file, sep=r"\s+", comment="#", names=nav_cols)

    speed_cols = [
        "date",
        "time",
        "speedForward",
        "speedForwardSd",
        "speedLeft",
        "speedUp",
        "speedLeftSd",
        "speedUpSd",
    ]
    df_speed = pd.read_csv(speed_file, sep=r"\s+", comment="#", names=speed_cols)

    def to_millis(df: pd.DataFrame) -> pd.Series:
        dt_str = df["date"] + " " + df["time"]
        return pd.to_datetime(dt_str, format="%Y/%m/%d %H:%M:%S.%f").astype("int64") // 10**6
    
    df_nav["time_utc"] = pd.to_datetime(df_nav["date"] + " " + df_nav["time"], format="%Y/%m/%d %H:%M:%S.%f", utc=True)
    df_speed["time_utc"] = pd.to_datetime(df_speed["date"] + " " + df_speed["time"], format="%Y/%m/%d %H:%M:%S.%f", utc=True)
    df_nav["gps_millis"] = to_millis(df_nav)
    df_speed["gps_millis"] = to_millis(df_speed)

    df = pd.merge(
        df_nav,
        df_speed[["time_utc", "speedForward"]],
        on="time_utc",
        how="inner",
    )

    df["velocity"] = df["speedForward"]

    coords = np.radians(df[["latitude", "longitude"]])
    distances = 6371000 * 2 * np.arcsin(
        np.sqrt(
            np.sin(np.diff(coords["latitude"]) / 2) ** 2
            + np.cos(coords["latitude"][:-1].values)
            * np.cos(coords["latitude"][1:].values)
            * np.sin(np.diff(coords["longitude"]) / 2) ** 2
        )
    )
    df["distance_trip"] = np.concatenate(([0], np.cumsum(distances)))
    df.drop(columns=["date", "time", "userData"], inplace=True)
    return df


def export_if_missing_csv(scenario_dir: Path, dry_run: bool, recompute: bool) -> str:
    """Export Delph CSV only when CSV is missing and txt inputs are available."""
    if not scenario_dir.is_dir():
        return "SKIP_NOT_DIR"

    gt_txt = scenario_dir / "groundtruth.txt"
    speed_txt = scenario_dir / "speed.txt"

    if not gt_txt.exists() or not speed_txt.exists():
        return "SKIP_INPUTS"

    output_csv = scenario_dir / f"{scenario_dir.name}_GroundTruth_Delph.csv"
    if output_csv.exists():
        if not recompute:
            return "SKIP_EXISTS"
        
    if dry_run:
        print(f"[DRY-RUN] {scenario_dir} -> {output_csv}")
        return "DRY_EXPORT"

    df_delph = load_scenario_delph(scenario_dir)
    df_delph.to_csv(output_csv, index=False)
    print(f"[EXPORT] {output_csv}")
    return "EXPORT"


def scan_groundtruth_folders(root: Path, dry_run: bool, recursive: bool, recompute: bool) -> None:
    if recursive:
        candidates = sorted([p for p in root.rglob("*") if p.is_dir()])
    else:
        # Auto-supporte le layout plat et LINE/SCENARIO.
        candidates = iter_scenario_dirs(root)

    if not candidates:
        print(f"No scenario folder found in: {root}")
        return

    counts = {
        "EXPORT": 0,
        "DRY_EXPORT": 0,
        "SKIP_EXISTS": 0,
        "SKIP_INPUTS": 0,
        "SKIP_NOT_DIR": 0,
        "WARN": 0,
    }

    for scenario_dir in tqdm(candidates, desc="Scanning scenarios"):
        try:
            status = export_if_missing_csv(scenario_dir, dry_run=dry_run, recompute=recompute)
        except Exception as exc:
            status = "WARN"
            print(f"[WARN] {scenario_dir}: {exc}")

        counts[status] = counts.get(status, 0) + 1

    print(
        "Done | "
        f"exported:{counts['EXPORT']} dry:{counts['DRY_EXPORT']} "
        f"already_csv:{counts['SKIP_EXISTS']} missing_inputs:{counts['SKIP_INPUTS']} "
        f"warn:{counts['WARN']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan groundtruth folders and export Delph CSV when groundtruth.txt and speed.txt "
            "are present and CSV is missing."
        )
    )
    parser.add_argument(
        "--groundtruth-dir",
        default=str(GT_DIR),
        help="Path to GROUNDTRUTH root directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions only, no file write",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan nested directories recursively (deeper than LINE/SCENARIO)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute and overwrite existing CSV files",
    )
    args = parser.parse_args()
    gt_root = Path(args.groundtruth_dir).expanduser().resolve()

    if not gt_root.exists() or not gt_root.is_dir():
        raise SystemExit(f"Invalid groundtruth directory: {gt_root}")

    scan_groundtruth_folders(gt_root, dry_run=args.dry_run, recursive=args.recursive, recompute=args.recompute)


if __name__ == "__main__":
    main()
