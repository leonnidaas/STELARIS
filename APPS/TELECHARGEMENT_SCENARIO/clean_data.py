import argparse
import os
import shutil
from pathlib import Path

from utils import DATA_DIR

TARGET_ROOT = DATA_DIR / "-01_UNORGANIZED"


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


def pick_one_file_per_type(files: list[Path]) -> dict[str, Path]:
    by_type: dict[str, list[Path]] = {
        "obs": [],
        "nav": [],
        "groundtruth": [],
        "speed": [],
    }

    for file_path in files:
        kind = classify_file(file_path)
        if kind is None:
            continue
        by_type[kind].append(file_path)

    selected: dict[str, Path] = {}
    for kind, candidates in by_type.items():
        if not candidates:
            continue
        selected[kind] = sorted(candidates, key=lambda p: p.name.lower())[0]

    return selected


def scenario_name_from_dir(source_root: Path, dir_path: Path) -> str:
    """retourne le dossier parant des fichiers obs/nav/groundtruth pour avoir le nom du scénario"""
    rel = dir_path.relative_to(source_root)
    if rel == Path("."):
        return source_root.name
    return str(rel).replace("/", "__")


def organize_source(source_root: Path, target_root: Path, dry_run: bool = False, move: bool = False) -> None:
    print(f"Scan recursif de: {source_root}")
    target_root.mkdir(parents=True, exist_ok=True)

    found_count = 0

    for current_dir, _, filenames in os.walk(source_root):
        dir_path = Path(current_dir)
        files = [dir_path / name for name in filenames]

        selected_by_type = pick_one_file_per_type(files)
        has_obs = "obs" in selected_by_type
        has_nav = "nav" in selected_by_type
        has_gt_or_speed = "groundtruth" in selected_by_type or "speed" in selected_by_type

        if not (has_obs and has_nav and has_gt_or_speed):
            continue

        found_count += 1
        scenario_name = scenario_name_from_dir(source_root, dir_path)
        scenario_dir = target_root / scenario_name

        print(f"\nScenario detecte: {scenario_name}")
        print(f"Source: {dir_path}")
        print(f"Destination: {scenario_dir}")

        if not dry_run:
            scenario_dir.mkdir(parents=True, exist_ok=True)

        selected_files = [
            selected_by_type["obs"],
            selected_by_type["nav"],
        ]
        if "groundtruth" in selected_by_type:
            selected_files.append(selected_by_type["groundtruth"])
        if "speed" in selected_by_type:
            selected_files.append(selected_by_type["speed"])

        for src in selected_files:
            dst = scenario_dir / src.name
            action = "deplace" if move else "copie"

            if dry_run:
                print(f"  [DRY-RUN] {action}: {src} -> {dst}")
                continue

            if dst.exists():
                print(f"  [SKIP] existe deja: {dst}")
                continue

            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(src, dst)

            print(f"  [{action.upper()}] {src.name}")

    if found_count == 0:
        print("Aucun dossier contenant obs + nav + (groundtruth ou speed) n'a ete trouve.")
    else:
        print(f"\nTermine. Scenarios traites: {found_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Explore un dossier source de maniere recursive et cree un dossier scenario "
            "dans DATA/-01_UNORGANIZED pour chaque sous-dossier contenant "
            "obs + nav + (groundtruth ou speed)."
        )
    )
    parser.add_argument(
        "--source_dir",
        help="Dossier source a explorer")
    parser.add_argument(
        "--target-dir",
        default=str(TARGET_ROOT),
        help=f"Dossier cible (defaut: {TARGET_ROOT})",
    )
    parser.add_argument(
        "--dry-run",
        # On utilise 'type' au lieu de 'action' pour accepter True/False
        type=lambda x: (str(x).lower() == 'true'), 
        default=True,
        help="Active ou désactive le mode dry-run (True/False)",
    )
    parser.add_argument(
        "--move",
        # On utilise 'type' au lieu de 'action' pour accepter True/False
        type=lambda x: (str(x).lower() == 'true'), 
        default=False,
        help="Deplace les fichiers au lieu de les copier",
    )

    args = parser.parse_args()
    source_root = Path(args.source_dir).expanduser().resolve()
    target_root = Path(args.target_dir).expanduser().resolve()

    if not source_root.exists() or not source_root.is_dir():
        raise SystemExit(f"Dossier source invalide: {source_root}")

    organize_source(
        source_root=source_root,
        target_root=target_root,
        dry_run=args.dry_run,
        move=args.move,
    )

if __name__ == "__main__":
    main()