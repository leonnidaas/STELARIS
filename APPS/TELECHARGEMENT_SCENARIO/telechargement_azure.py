"""Sous résèrve d'avoir la clé SAS du file share azure !!"""
import os
import requests
import yaml
from dotenv import load_dotenv
from xml.etree import ElementTree
from tqdm import tqdm # Pour la barre d'avancement

# --- CHARGEMENT ---
load_dotenv()
SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN")
ACCOUNT_NAME = os.getenv("AZURE_ACCOUNT_NAME")
SHARE_NAME = os.getenv("AZURE_SHARE_NAME")

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

# Extraction des paramètres du YAML
BASE_DATA = config['data_dir']      # DATA
RAW_DIR = config['raw_dir']        # 00_RAW
GNSS_DIR = config['gnss_dir']      # GNSS_RINEX
NAV_DIR = config['gnss_nav_dir']   # NAV
OBS_DIR = config['gnss_obs_dir']   # OBS
GT_DIR = config['gt_dir']          # GROUNDTRUTH
CHUNK_SIZE = config.get('chunk_size', 1048576) # 1 Mo par défaut

AZURE_SCENARIOS_PATH = "B81585/scenarios" 
TARGET_SCENARIOS = [
    "0_full_coverage",
    "1_partial_coverage",
    "2_hole_in_middle",
    "3_no_coverage"
]

def list_files_in_folder(azure_folder_path):
    """Récupère la liste des fichiers d'un dossier Azure."""
    url = f"https://{ACCOUNT_NAME}.file.core.windows.net/{SHARE_NAME}/{azure_folder_path}{SAS_TOKEN}&restype=directory&comp=list"
    try:
        r = requests.get(url)
        r.raise_for_status()
        tree = ElementTree.fromstring(r.content)
        # On stocke le nom et la taille pour la barre de progression
        files = []
        # On construit la liste des fichiers avec leur chemin complet et leur taille
        for file_node in tree.findall(".//File"):
            name = file_node.find("Name").text
            size = int(file_node.find("Properties/Content-Length").text)
            files.append({'path': f"{azure_folder_path}/{name}", 'size': size})
        return files
    except Exception as e:
        print(f"Erreur scan {azure_folder_path}: {e}")
        return []

def get_local_path(azure_path):
    """Définit la destination selon l'architecture du projet."""
    filename = os.path.basename(azure_path)
    parts = azure_path.split('/')
    scenario_name = parts[-2]
    
    fn_lower = filename.lower()
    # Tri GNSS
    if fn_lower.endswith('raw.obs'):
        return os.path.join(BASE_DATA, RAW_DIR, GNSS_DIR, OBS_DIR, scenario_name, filename)
    elif fn_lower.endswith('raw.nav'):
        return os.path.join(BASE_DATA, RAW_DIR, GNSS_DIR, NAV_DIR, scenario_name, filename)
    # Tri GT et Speed dans le même dossier
    elif any(fn_lower.endswith(x) for x in ['groundtruth.csv', 'groundtruth.txt', 'speed.txt']):
        return os.path.join(BASE_DATA, RAW_DIR, GT_DIR, scenario_name, filename)
    return None

def download_with_progress(azure_path, local_path, total_size):
    """Télécharge avec barre de progression et saute si le fichier existe déjà."""
    
    # --- VÉRIFICATION DOUBLON ---
    if os.path.exists(local_path):
        # Optionnel : vérifier si la taille correspond pour être sûr que le fichier est complet
        if os.path.getsize(local_path) == total_size:
            return False # Fichier déjà présent et complet, on saute

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = f"https://{ACCOUNT_NAME}.file.core.windows.net/{SHARE_NAME}/{azure_path}{SAS_TOKEN}"
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        # Configuration de la barre tqdm pour ce fichier
        with open(local_path, 'wb') as f, tqdm(
            desc=os.path.basename(local_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            leave=False # La barre disparaît après la fin du fichier pour ne pas encombrer
        ) as bar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                size = f.write(chunk)
                bar.update(size)
    return True

if __name__ == "__main__":
    print(f"Initialisation de la synchronisation sélective...")
    
    for scenario in TARGET_SCENARIOS:
        full_azure_path = f"{AZURE_SCENARIOS_PATH}/{scenario}"
        files_metadata = list_files_in_folder(full_azure_path)
        
        print(f"\n>>> Scénario : {scenario}")
        
        for file_info in files_metadata:
            dest = get_local_path(file_info['path'])
            if dest:
                was_downloaded = download_with_progress(file_info['path'], dest, file_info['size'])
                if not was_downloaded:
                    print(f"  [Sauté] {os.path.basename(file_info['path'])} est déjà à jour.")

    print("\nSynchronisation terminée.")