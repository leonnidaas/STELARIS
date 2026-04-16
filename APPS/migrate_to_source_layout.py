import argparse
import re
import shutil
import sys
from pathlib import Path

APPS_DIR = Path(__file__).resolve().parent
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from utils import (
    INTERIM_DIR,
    INTERIM_FUSION_SUBDIR,
    INTERIM_GNSS_SUBDIR,
    INTERIM_IGN_SUBDIR,
    INTERIM_OSM_SUBDIR,
    PROCESSED_DIR,
    PROCESSED_FUSION_SUBDIR,
    PROCESSED_GNSS_SUBDIR,
    PROCESSED_IGN_SUBDIR,
    PROCESSED_OSM_SUBDIR,
    iter_scenario_dirs,
)


JSON_SUBDIR_NAME = "json"


FILENAME_TARGET_PATTERNS: list[tuple[re.Pattern[str], str, str | None]] = [
    (re.compile(r"^params_pipeline_labelisation_.*\.json$", re.IGNORECASE), "ign", JSON_SUBDIR_NAME),
    (re.compile(r"^fusion_gt_gnss_lidar_features_.*\.csv$", re.IGNORECASE), "fusion", None),
    (re.compile(r"^fusion_finale_gnss_lidar_gt_label_.*\.csv$", re.IGNORECASE), "fusion", None),
    (re.compile(r"^gnss_position_.*\.csv$", re.IGNORECASE), "gnss", None),
    (re.compile(r"^space_vehicule_info_.*\.csv$", re.IGNORECASE), "gnss", None),
    (re.compile(r"^fusion_gt_gnss_.*\.csv$", re.IGNORECASE), "fusion", None),
    (re.compile(r"^features_gnss_.*\.csv$", re.IGNORECASE), "gnss", None),
    (re.compile(r"^features_lidar_.*\.csv$", re.IGNORECASE), "ign", None),
    (re.compile(r"^features_lidar_plus_labels_.*\.csv$", re.IGNORECASE), "ign", None),
    (re.compile(r"^final_labeled_.*\.csv$", re.IGNORECASE), "ign", None),
    (re.compile(r"^final_labeled_lidar_.*\.csv$", re.IGNORECASE), "ign", None),
    (re.compile(r"^features_osm_.*\.csv$", re.IGNORECASE), "osm", None),
    (re.compile(r"^final_labeled_osm_.*\.csv$", re.IGNORECASE), "osm", None),
]


def _target_for_filename(file_name: str) -> tuple[str, str | None] | None:
    for pattern, source_key, subdir in FILENAME_TARGET_PATTERNS:
        if pattern.match(file_name):
            return source_key, subdir
    return None


def _build_source_dir_map(root_type: str) -> dict[str, str]:
    if root_type == "interim":
        return {
            "gnss": INTERIM_GNSS_SUBDIR,
            "ign": INTERIM_IGN_SUBDIR,
            "osm": INTERIM_OSM_SUBDIR,
            "fusion": INTERIM_FUSION_SUBDIR,
        }
    return {
        "gnss": PROCESSED_GNSS_SUBDIR,
        "ign": PROCESSED_IGN_SUBDIR,
        "osm": PROCESSED_OSM_SUBDIR,
        "fusion": PROCESSED_FUSION_SUBDIR,
    }


def _iter_known_files_to_reclassify(scenario_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for file_path in scenario_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if _target_for_filename(file_path.name) is not None:
            candidates.append(file_path)
    return sorted(candidates)


def _safe_move_file(src: Path, dst: Path, dry_run: bool, verbose: bool) -> bool:
    """Move a file if destination does not exist; return True when moved."""
    if not src.exists() or not src.is_file():
        return False

    if src.resolve() == dst.resolve():
        return False

    if dst.exists():
        if verbose:
            print(f"  [SKIP_EXISTS] {dst}")
        return False

    if dry_run:
        print(f"  [DRY-RUN] MOVE_FILE {src} -> {dst}")
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    if verbose:
        print(f"  [MOVED] {src} -> {dst}")
    return True


def _cleanup_empty_source_dirs(root: Path, source_dirs: dict[str, str], dry_run: bool, verbose: bool) -> None:
    """Remove empty source dirs created by mistake in old runs."""
    for name in source_dirs.values():
        candidate = root / name
        if not candidate.exists() or not candidate.is_dir():
            continue
        json_dir = candidate / JSON_SUBDIR_NAME
        if json_dir.exists() and json_dir.is_dir():
            try:
                next(json_dir.iterdir())
            except StopIteration:
                if dry_run:
                    print(f"  [DRY-RUN] RMDIR_EMPTY {json_dir}")
                else:
                    json_dir.rmdir()
                    if verbose:
                        print(f"  [RMDIR_EMPTY] {json_dir}")

        try:
            next(candidate.iterdir())
            continue
        except StopIteration:
            if dry_run:
                print(f"  [DRY-RUN] RMDIR_EMPTY {candidate}")
            else:
                candidate.rmdir()
                if verbose:
                    print(f"  [RMDIR_EMPTY] {candidate}")


def migrate_scenario_dir(
    scenario_dir: Path,
    source_dirs: dict[str, str],
    dry_run: bool,
    verbose: bool,
) -> tuple[int, int, int]:
    """Rescan a scenario directory and reclassify known files into source folders."""
    moved = 0
    skipped = 0
    unchanged = 0

    candidates = _iter_known_files_to_reclassify(scenario_dir)
    for src in candidates:
        target = _target_for_filename(src.name)
        if target is None:
            skipped += 1
            continue

        source_key, subdir = target
        source_folder_name = source_dirs[source_key]
        dst = scenario_dir / source_folder_name
        if subdir:
            dst = dst / subdir
        dst = dst / src.name

        if src.resolve() == dst.resolve():
            unchanged += 1
            continue

        if _safe_move_file(src, dst, dry_run=dry_run, verbose=verbose):
            moved += 1
        else:
            skipped += 1

    _cleanup_empty_source_dirs(scenario_dir, source_dirs=source_dirs, dry_run=dry_run, verbose=verbose)
    return moved, skipped, unchanged


def run_migration(dry_run: bool, verbose: bool) -> None:
    print("=== Migration vers layout par source (gnss/ign/osm/fusion) + json ign/json ===")

    interim_dirs = iter_scenario_dirs(INTERIM_DIR)
    processed_dirs = iter_scenario_dirs(PROCESSED_DIR)
    interim_source_dirs = _build_source_dir_map("interim")
    processed_source_dirs = _build_source_dir_map("processed")

    interim_moved = 0
    interim_skipped = 0
    interim_unchanged = 0
    for scenario_dir in interim_dirs:
        if verbose:
            print(f"[INTERIM] {scenario_dir}")
        moved, skipped, unchanged = migrate_scenario_dir(
            scenario_dir,
            source_dirs=interim_source_dirs,
            dry_run=dry_run,
            verbose=verbose,
        )
        interim_moved += moved
        interim_skipped += skipped
        interim_unchanged += unchanged

    processed_moved = 0
    processed_skipped = 0
    processed_unchanged = 0
    for scenario_dir in processed_dirs:
        if verbose:
            print(f"[PROCESSED] {scenario_dir}")
        moved, skipped, unchanged = migrate_scenario_dir(
            scenario_dir,
            source_dirs=processed_source_dirs,
            dry_run=dry_run,
            verbose=verbose,
        )
        processed_moved += moved
        processed_skipped += skipped
        processed_unchanged += unchanged

    print("\n=== Resume migration source layout ===")
    print(
        f"INTERIM scenarios:{len(interim_dirs)} moved_files:{interim_moved} "
        f"unchanged_already_ok:{interim_unchanged} skipped_or_conflict:{interim_skipped}"
    )
    print(
        f"PROCESSED scenarios:{len(processed_dirs)} moved_files:{processed_moved} "
        f"unchanged_already_ok:{processed_unchanged} skipped_or_conflict:{processed_skipped}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Move already computed files from scenario root directories to source folders "
            "(gnss/ign/osm/fusion) for INTERIM and PROCESSED."
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Show planned actions without modifying files")
    parser.add_argument("--verbose", action="store_true", help="Show detailed operations")
    args = parser.parse_args()

    run_migration(dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
