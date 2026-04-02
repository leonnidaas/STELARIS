import argparse
import shutil
from pathlib import Path

import sys

APPS_DIR = Path(__file__).resolve().parents[1]
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from utils import GNSS_NAV, GNSS_OBS, GT_DIR, INTERIM_DIR, PROCESSED_DIR, RAW_DIR


def infer_line_id(scenario_name: str) -> str:
    if "__" in scenario_name:
        return scenario_name.split("__", 1)[0]
    return "UNKNOWN_LINE"


def merge_move_dir(src: Path, dst: Path, dry_run: bool, verbose: bool) -> None:
    """Move src into dst. If dst exists, merge recursively without overwriting files."""
    if dry_run:
        print(f"[DRY-RUN] MOVE_DIR {src} -> {dst}")
        return

    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            if item.is_dir() and target.is_dir():
                merge_move_dir(item, target, dry_run=False, verbose=verbose)
                continue
            if verbose:
                print(f"  [SKIP_EXISTS] {target}")
            continue

        shutil.move(str(item), str(target))

    try:
        src.rmdir()
    except OSError:
        if verbose:
            print(f"  [KEEP_NON_EMPTY] {src}")


def migrate_root_flat_to_line(root: Path, dry_run: bool, verbose: bool) -> tuple[int, int]:
    """Migrate ROOT/SCENARIO to ROOT/LINE/SCENARIO for directories matching *__*."""
    moved = 0
    skipped = 0

    if not root.exists():
        return moved, skipped

    for entry in sorted([p for p in root.iterdir() if p.is_dir()]):
        scenario = entry.name
        if "__" not in scenario:
            skipped += 1
            continue

        line_id = infer_line_id(scenario)
        target = root / line_id / scenario
        if entry.resolve() == target.resolve():
            skipped += 1
            continue

        merge_move_dir(entry, target, dry_run=dry_run, verbose=verbose)
        moved += 1

    return moved, skipped


def migrate_annexe_trajet(dry_run: bool, verbose: bool) -> tuple[int, int, int]:
    """Migrate ANNEXE_TRAJET/SCENARIO -> ANNEXE_TRAJET/LIGNES/LINE/SCENARIOS/SCENARIO.

    Also promotes Bras_de_levier.json to COMMON if missing.
    """
    annexe_root = RAW_DIR / "ANNEXE_TRAJET"
    moved = 0
    skipped = 0
    promoted = 0

    if not annexe_root.exists():
        return moved, skipped, promoted

    for entry in sorted([p for p in annexe_root.iterdir() if p.is_dir()]):
        scenario = entry.name
        if scenario == "LIGNES":
            skipped += 1
            continue
        if "__" not in scenario:
            skipped += 1
            continue

        line_id = infer_line_id(scenario)
        target = annexe_root / "LIGNES" / line_id / "SCENARIOS" / scenario
        common_dir = annexe_root / "LIGNES" / line_id / "COMMON"
        common_lever = common_dir / "Bras_de_levier.json"

        src_lever = entry / "Bras_de_levier.json"
        if src_lever.exists() and not common_lever.exists():
            if dry_run:
                print(f"[DRY-RUN] COPY_COMMON {src_lever} -> {common_lever}")
            else:
                common_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_lever, common_lever)
            promoted += 1

        merge_move_dir(entry, target, dry_run=dry_run, verbose=verbose)
        moved += 1

    return moved, skipped, promoted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate flat scenario directories to line/scenario layout for "
            "GROUNDTRUTH, GNSS OBS/NAV, INTERIM, PROCESSED and ANNEXE_TRAJET."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned actions without modifying files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed merge/skip logs",
    )

    args = parser.parse_args()

    print("=== Migration vers layout LINE/SCENARIO ===")

    gt_moved, gt_skipped = migrate_root_flat_to_line(GT_DIR, dry_run=args.dry_run, verbose=args.verbose)
    obs_moved, obs_skipped = migrate_root_flat_to_line(GNSS_OBS, dry_run=args.dry_run, verbose=args.verbose)
    nav_moved, nav_skipped = migrate_root_flat_to_line(GNSS_NAV, dry_run=args.dry_run, verbose=args.verbose)
    interim_moved, interim_skipped = migrate_root_flat_to_line(INTERIM_DIR, dry_run=args.dry_run, verbose=args.verbose)
    processed_moved, processed_skipped = migrate_root_flat_to_line(PROCESSED_DIR, dry_run=args.dry_run, verbose=args.verbose)
    ann_moved, ann_skipped, ann_promoted = migrate_annexe_trajet(dry_run=args.dry_run, verbose=args.verbose)

    print("\n=== Résumé ===")
    print(f"GROUNDTRUTH moved:{gt_moved} skipped:{gt_skipped}")
    print(f"GNSS_OBS moved:{obs_moved} skipped:{obs_skipped}")
    print(f"GNSS_NAV moved:{nav_moved} skipped:{nav_skipped}")
    print(f"INTERIM moved:{interim_moved} skipped:{interim_skipped}")
    print(f"PROCESSED moved:{processed_moved} skipped:{processed_skipped}")
    print(f"ANNEXE_TRAJET moved:{ann_moved} skipped:{ann_skipped} common_lever_promoted:{ann_promoted}")


if __name__ == "__main__":
    main()
