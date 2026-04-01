#!/usr/bin/env python3
"""Range des fichiers de scenarios vers DATA/00_RAW selon config.yml."""

from pathlib import Path
import argparse
import os
import shutil
import sys

import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_root(config: dict, config_path: Path) -> Path:
    env_root = os.getenv("ROOT_PATH")
    if env_root:
        return Path(env_root).expanduser().resolve()

    root_path_cfg = config.get("root_path")
    if root_path_cfg:
        return Path(root_path_cfg).expanduser().resolve()

    # Par defaut: config.yml est dans APPS/, le projet est son parent.
    return config_path.resolve().parent.parent


def classify_file(file_path: Path) -> str | None:
    low = file_path.name.lower()

    if low in {"speed.txt", "groundtruth.txt"}:
        return "groundtruth"
    if low.endswith("_groundtruth.csv"):
        return "groundtruth"
    if low.endswith("raw.obs"):
        return "obs"
    if low.endswith("raw.nav"):
        return "nav"

    return None


def destination_for(file_path: Path, scenario_name: str, project_root: Path, config: dict) -> Path | None:
    kind = classify_file(file_path)
    if kind is None:
        return None

    data_dir = config["data_dir"]
    raw_dir = config["raw_dir"]

    if kind == "groundtruth":
        return (
            project_root
            / data_dir
            / raw_dir
            / config["gt_dir"]
            / scenario_name
            / file_path.name
        )

    gnss_base = project_root / data_dir / raw_dir / config["gnss_dir"]
    if kind == "obs":
        return gnss_base / config["gnss_obs_dir"] / scenario_name / file_path.name

    return gnss_base / config["gnss_nav_dir"] / scenario_name / file_path.name


def move_file(src: Path, dst: Path, dry_run: bool, overwrite: bool) -> str:
    if src.resolve() == dst.resolve():
        return "already_in_place"

    if dst.exists():
        if not overwrite:
            return "exists_skip"
        if not dry_run:
            dst.unlink()

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    return "moved"


def organize_one_dataset(
    dataset_dir: Path,
    project_root: Path,
    config: dict,
    dry_run: bool = False,
    delete_unmapped: bool = False,
    overwrite: bool = False,
) -> None:
    if not dataset_dir.is_dir():
        return

    scenario_name = dataset_dir.name
    print(f"\nDossier source : {dataset_dir} (scenario: {scenario_name})")

    moved = []
    skipped_exists = []
    unmapped = []
    deleted = []
    errors = []

    for item in dataset_dir.iterdir():
        if not item.is_file():
            continue

        destination = destination_for(item, scenario_name, project_root, config)
        if destination is not None:
            try:
                result = move_file(item, destination, dry_run=dry_run, overwrite=overwrite)
                if result == "moved":
                    moved.append((item.name, str(destination)))
                else:
                    skipped_exists.append((item.name, str(destination)))
            except Exception as e:
                errors.append((item.name, str(e)))
            continue

        unmapped.append(item.name)
        if delete_unmapped:
            try:
                if not dry_run:
                    item.unlink()
                deleted.append(item.name)
            except Exception as e:
                errors.append((item.name, str(e)))

    print("  Deplaces :")
    for name, dst in moved:
        print(f"    - {name} -> {dst}")

    if skipped_exists:
        print("  Deja presents (non ecrases) :")
        for name, dst in skipped_exists:
            print(f"    - {name} -> {dst}")

    if unmapped:
        print("  Non reconnus :")
        for name in unmapped:
            print(f"    - {name}")

    if deleted:
        print("  Supprimes (non reconnus) :")
        for name in deleted:
            print(f"    - {name}")

    if errors:
        print("  Erreurs :")
        for name, err in errors:
            print(f"    - {name}: {err}")


def main() -> None:
    default_config = Path(__file__).resolve().parents[1] / "config.yml"

    parser = argparse.ArgumentParser(
        description=(
            "Trie et range les fichiers utiles d'un dossier de datasets "
            "vers l'architecture DATA/00_RAW selon config.yml."
        )
    )
    parser.add_argument(
        "root_dir",
        help="Dossier contenant les sous-dossiers de scenarios a ranger",
    )
    parser.add_argument(
        "--config",
        default=str(default_config),
        help=f"Chemin vers config.yml (defaut: {default_config})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche seulement ce qui serait deplace/supprime",
    )
    parser.add_argument(
        "--delete-unmapped",
        action="store_true",
        help="Supprime aussi les fichiers non reconnus",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ecrase les fichiers destination s'ils existent deja",
    )

    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    config_path = Path(args.config)

    if not root_dir.exists():
        print(f"Erreur : chemin inexistant : {root_dir}")
        sys.exit(1)

    if not root_dir.is_dir():
        print(f"Erreur : ce n'est pas un dossier : {root_dir}")
        sys.exit(1)

    if not config_path.exists():
        print(f"Erreur : config introuvable : {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    project_root = resolve_project_root(config, config_path)

    print(f"Racine projet: {project_root}")
    print(f"Config: {config_path}")

    subdirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    if not subdirs:
        print("Aucun sous-dossier trouve.")
        sys.exit(0)

    print(f"{len(subdirs)} sous-dossier(s) trouve(s).")

    for subdir in subdirs:
        organize_one_dataset(
            dataset_dir=subdir,
            project_root=project_root,
            config=config,
            dry_run=args.dry_run,
            delete_unmapped=args.delete_unmapped,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
