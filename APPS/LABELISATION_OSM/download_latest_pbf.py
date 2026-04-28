import subprocess
from datetime import datetime
from pathlib import Path

import requests

from utils import OSM_PBF, OSM_URL_EUROPE


def _is_valid_pbf(path: Path) -> bool:
    """Valide un PBF via osmium. Retourne False si fichier corrompu/incomplet."""
    if not path.exists() or path.stat().st_size <= 0:
        return False

    cmd = ["osmium", "fileinfo", "-e", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        return True

    details = (proc.stderr or proc.stdout or "").strip()
    if details:
        print(f"--- PBF invalide ({path.name}): {details}")
    return False


def download_osm_pbf(
    region="france",
    dest_folder=OSM_PBF,
    source=OSM_URL_EUROPE,
    chunk_size=8192,
    force=False,
    validate_with_osmium=True,
):
    """Telecharge le dernier PBF OSM pour une region Geofabrik.

    Si le fichier du jour existe mais est corrompu (EOF, fichier tronque),
    il est supprime puis retente automatiquement.
    """
    file_name = f"{region}-latest.osm.pbf"
    url = f"{source}{file_name}"
    date_str = datetime.now().strftime("%Y-%m-%d")
    dest_path = Path(dest_folder) / f"{region}_{date_str}.osm.pbf"

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        if (not validate_with_osmium) or _is_valid_pbf(dest_path):
            print(f"--- Fichier deja present et valide: {dest_path}")
            return dest_path
        print(f"--- Fichier present mais invalide, suppression: {dest_path}")
        dest_path.unlink(missing_ok=True)

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)

    print(f"--- Telechargement de {url}...")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        tmp_path.replace(dest_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    if validate_with_osmium and not _is_valid_pbf(dest_path):
        dest_path.unlink(missing_ok=True)
        raise RuntimeError(f"PBF telecharge invalide: {dest_path}")

    print(f"--- Telechargement termine: {dest_path}")
    return dest_path

# Utilisation
# pbf_file = download_osm_pbf("france")