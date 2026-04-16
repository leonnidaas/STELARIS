import requests
import os
from datetime import datetime
from pathlib import Path
from utils import OSM_URL_EUROPE, OSM_PBF

def download_osm_pbf(region="france", dest_folder=OSM_PBF, source=OSM_URL_EUROPE , chunk_size=8192):
    """Télécharge le dernier fichier PBF d'OSM pour la région spécifiée depuis"
    Geofabrik"""
    file_name = f"{region}-latest.osm.pbf"
    url = f"{source}{file_name}"
    date_str = datetime.now().strftime("%Y-%m-%d")
    dest_path = Path(dest_folder) / f"{region}_{date_str}.osm.pbf"
    
    # Création du dossier si inexistant
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Vérification si le fichier existe déjà
    if dest_path.exists():
        print(f"--- Fichier déjà présent : {dest_path}. Suppression manuelle requise pour mise à jour.")
        return dest_path

    print(f"--- Téléchargement de {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size):
                f.write(chunk)
    
    print(f"--- Téléchargement terminé : {dest_path}")
    return dest_path

# Utilisation
# pbf_file = download_osm_pbf("france")