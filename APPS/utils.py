import os
from pathlib import Path

from flask import json
import pyproj
from dotenv import load_dotenv
import yaml


load_dotenv()


_CONFIG_PATH = Path(__file__).with_name("config.yml")
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
	_CONFIG = yaml.safe_load(f) or {}


def _get_cfg(key, default=None):
	return _CONFIG.get(key, default)

ROOT_PATH = Path(os.getenv("ROOT_PATH", _get_cfg("root_path", Path(__file__).parent.parent)))

DATA_DIR = ROOT_PATH / _get_cfg("data_dir", "DATA")
RAW_DIR = DATA_DIR / _get_cfg("raw_dir", "00_RAW")
INTERIM_DIR = DATA_DIR / _get_cfg("interim_dir", "01_INTERIM")
PROCESSED_DIR = DATA_DIR / _get_cfg("processed_dir", "02_PROCESSED")
TRAINING_DIR = DATA_DIR / _get_cfg("training_dir", "03_TRAINING")

GNSS_DIR = RAW_DIR / _get_cfg("gnss_dir", "GNSS_RINEX")
GNSS_OBS = GNSS_DIR / _get_cfg("gnss_obs_dir", "OBS")
GNSS_NAV = GNSS_DIR / _get_cfg("gnss_nav_dir", "NAV")
GT_DIR = RAW_DIR / _get_cfg("gt_dir", "GROUNDTRUTH")

LIDAR_TA = RAW_DIR / Path(_get_cfg("lidar_ta_relpath", "IGN_LiDAR/TA_diff_pkk_lidarhd_classe.shp"))
OSM_PBF = RAW_DIR / Path(_get_cfg("osm_pbf_relpath", "OSM/corse.osm.pbf"))

GRILLE_ALTITUDE_DIR = RAW_DIR / Path(
	_get_cfg("grille_altitude_relpath", "IGN/GRILLE_CONVERTION_ALTITUDE_WGS84_IGN69")
)
LIDAR_DIR = RAW_DIR / Path(_get_cfg("lidar_dir_relpath", "IGN/TUILES_LIDAR_HD"))

MODELS_DIR = ROOT_PATH / _get_cfg("models_dir", "MODELS")
long_col = _get_cfg("long_col", "longitude")
lat_col = _get_cfg("lat_col", "latitude")

WFS_URL = _get_cfg("wfs_url", "https://data.geopf.fr/wfs/ows")
WFS_VERSION = str(_get_cfg("wfs_version", "2.0.0"))
LIDAR_LAYER = _get_cfg("lidar_layer", "IGNF_NUAGES-DE-POINTS-LIDAR-HD:dalle")

RAYON_RECHERCHE = float(_get_cfg("rayon_recherche", 20.0))
N_WORKERS = int(_get_cfg("n_workers", 10))
DECIMATION_FACTOR = int(_get_cfg("decimation_factor", 1))

GRILLE_CONVERTION_ALTITUDE_WGS84_IGN69 = GRILLE_ALTITUDE_DIR / _get_cfg(
	"grille_altitude_file", "fr_ign_RAF20.tif"
)
os.environ["PROJ_LIB"] = GRILLE_CONVERTION_ALTITUDE_WGS84_IGN69.parent.as_posix()
os.environ["PROJ_DATA"] = GRILLE_CONVERTION_ALTITUDE_WGS84_IGN69.parent.as_posix()
pyproj.datadir.set_data_dir(GRILLE_CONVERTION_ALTITUDE_WGS84_IGN69.parent.as_posix())

PYTHON_RINEX_INTERPRETER = os.getenv(
	"PYTHON_RINEX_INTERPRETER", _get_cfg("python_rinex_interpreter", "python")
)
PYTHON_LABELISATION_INTERPRETER = os.getenv(
	"PYTHON_LABELISATION_INTERPRETER", _get_cfg("python_labelisation_interpreter", "python")
)

CHUNK_SIZE = int(_get_cfg("chunk_size", 1048576))

PARAMS_LABELISATION = _get_cfg("params_labelisation")
if not isinstance(PARAMS_LABELISATION, dict):
    raise ValueError(
        "params_labelisation doit etre defini comme un dictionnaire dans config.yml"
    )

COLUMN_MAPPING = _get_cfg("column_mapping")
if isinstance(COLUMN_MAPPING, list):
    # Supporte un format YAML "liste de paires" et le convertit en dictionnaire.
    try:
        COLUMN_MAPPING = dict(COLUMN_MAPPING)
    except Exception as e:
        raise ValueError(f"column_mapping invalide dans config.yml: {e}") from e

if not isinstance(COLUMN_MAPPING, dict):
    raise ValueError(
        "column_mapping doit etre un dictionnaire dans config.yml, exemple: column_mapping: {latitude: [...]}"
    )

PARAMS_ENTRAINEMENT = _get_cfg("params_entrainement")
if not isinstance(PARAMS_ENTRAINEMENT, dict):
    raise ValueError(
        "params_entrainement doit etre defini comme un dictionnaire dans config.yml"
    )

def standardize_dataframe(df):
    """Renomme les colonnes du DataFrame vers les standards internes."""
    rename_dict = {}
    for standard_name, aliases in COLUMN_MAPPING.items():
        for col in df.columns:
            if col.lower() in [a.lower() for a in aliases]:
                rename_dict[col] = standard_name
                break

    return df.rename(columns=rename_dict)


def parse_lever_arm(file_path):
    """Extrait le tuple (x, y, z) depuis le fichier de bras de levier."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return (config["GNSS_BOGIE_X"], config["GNSS_BOGIE_Y"], config["GNSS_BOGIE_Z"])
    except FileNotFoundError:
        print(f"Fichier de bras de levier non trouvé : {file_path}")
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as out_f:
            json.dump({
                "GNSS_BOGIE_X": 11.45375,
                "GNSS_BOGIE_Y": 0.598,
                "GNSS_BOGIE_Z": -4.137,
            }, out_f, indent=4)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return (config["GNSS_BOGIE_X"], config["GNSS_BOGIE_Y"], config["GNSS_BOGIE_Z"])
        except Exception as e:
            print(f"Erreur lors de la lecture des metadonnees de bras de levier : {e}")
            return (11.45375, 0.598, -4.137)


def _extract_line_id(traj_id: str) -> str:
    """Extrait un identifiant de ligne depuis un traj_id de type LINE__SCENARIO."""
    if "__" in traj_id:
        return traj_id.split("__", 1)[0]
    return traj_id


def iter_scenario_dirs(root_dir: Path) -> list[Path]:
    """Retourne les dossiers de scenarios d'un root, en supportant layout plat et LINE/SCENARIO."""
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        return []

    scenario_dirs: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue

        # Layout plat: GROUNDTRUTH/SCENARIO
        if "__" in child.name:
            scenario_dirs.append(child)
            continue

        # Layout par ligne: GROUNDTRUTH/LINE/SCENARIO
        for sub in child.iterdir():
            if sub.is_dir() and "__" in sub.name:
                scenario_dirs.append(sub)

    return sorted(scenario_dirs)


def iter_gt_scenario_dirs(gt_root: Path | None = None) -> list[Path]:
    """Retourne les dossiers de scenarios GT, en supportant layout plat et LINE/SCENARIO."""
    root = GT_DIR if gt_root is None else Path(gt_root)
    return iter_scenario_dirs(root)


def list_traj_ids() -> list[str]:
    """Liste les IDs de trajets disponibles dans GROUNDTRUTH."""
    return [p.name for p in iter_gt_scenario_dirs(GT_DIR)]


def _find_traj_dir(base_dir: Path, traj_id: str) -> Path | None:
    """Trouve le dossier d'un trajet en supportant les layouts plat et par ligne."""
    direct = base_dir / traj_id
    if direct.exists() and direct.is_dir():
        return direct

    matches = sorted([p for p in base_dir.glob(f"*/{traj_id}") if p.is_dir()])
    if matches:
        return matches[0]

    return None


def _resolve_annexe_dirs(traj_id: str) -> tuple[str, Path, Path, Path]:
    """Résout les dossiers d'annexe (legacy + nouvelle structure par ligne)."""
    line_id = _extract_line_id(traj_id)

    legacy_traj_dir = RAW_DIR / "ANNEXE_TRAJET" / traj_id
    line_common_dir = RAW_DIR / "ANNEXE_TRAJET" / "LIGNES" / line_id / "COMMON"
    line_traj_dir = RAW_DIR / "ANNEXE_TRAJET" / "LIGNES" / line_id / "SCENARIOS" / traj_id

    # Priorité au scénario dédié, puis legacy, puis common ligne.
    if line_traj_dir.exists():
        traj_annexe_dir = line_traj_dir
    elif legacy_traj_dir.exists():
        traj_annexe_dir = legacy_traj_dir
    else:
        traj_annexe_dir = line_traj_dir

    traj_annexe_dir.mkdir(parents=True, exist_ok=True)
    line_common_dir.mkdir(parents=True, exist_ok=True)

    return line_id, traj_annexe_dir, line_common_dir, legacy_traj_dir


def _resolve_work_dirs(base_dir: Path, traj_id: str, line_id: str) -> tuple[Path, Path, Path]:
    """Résout les dossiers de travail (nouveau layout + fallback legacy)."""
    legacy_dir = base_dir / traj_id
    line_dir = base_dir / line_id / traj_id

    # Lecture descendante: si un dossier legacy existe déjà et le nouveau n'existe pas,
    # on continue à l'utiliser. Sinon, on bascule vers le nouveau layout.
    if legacy_dir.exists() and not line_dir.exists():
        active_dir = legacy_dir
    else:
        active_dir = line_dir

    active_dir.mkdir(parents=True, exist_ok=True)
    return active_dir, line_dir, legacy_dir


def _resolve_shared_lidar_dirs(traj_interim_dir: Path, line_id: str) -> tuple[Path, Path, Path]:
    """Résout les chemins partagés des tuiles/listes LiDAR au niveau trajet (ligne)."""
    legacy_tiles_dir = traj_interim_dir / "tiles_laz"
    shared_root_dir = INTERIM_DIR / line_id / "_SHARED"
    shared_tiles_dir = shared_root_dir / "tiles_laz"

    shared_root_dir.mkdir(parents=True, exist_ok=True)
    shared_tiles_dir.mkdir(parents=True, exist_ok=True)
    return shared_tiles_dir, shared_root_dir, legacy_tiles_dir


def get_traj_paths(traj_id: str):
    """Genere dynamiquement tous les chemins utiles pour un trajet dans un dictionnaire.
    { "id": traj_id,
      "raw_gt": <chemin du fichier GT brut>,
      "interim_dir": <dossier interim pour ce trajet>,
      "final_dir": <dossier final pour ce trajet>,
      "raw_gnss": <chemin du fichier de positionnement GNSS brut>,
      "space_vehicule_info": <chemin du fichier d'info espace vehicule>,
      "sync_csv": <chemin du fichier de fusion GT-GNSS>,
      "features_csv": <chemin du fichier d'extraction des features LiDAR>,
            "fusion_features_csv": <chemin du fichier de fusion GT + features GNSS + features LiDAR>,
            "final_fusion_csv": <chemin du fichier final fusionne GT + features GNSS + features LiDAR + label>,
    "labels_plus_features_csv": <chemin du fichier de fusion des features et labels>,
    "lidar_tiles": <chemin du dossier contenant les tuiles LiDAR (partage ligne)>,
      "obs_file": <chemin du fichier RINEX OBS>,
      "nav_file": <chemin du fichier RINEX NAV>,
      "labels_csv": <chemin du fichier final traité>,
      "gnss_offset": <tuple du bras de levier (x,y,z)>
      "gnss_features_csv": <chemin du fichier de features extraites du GNSS> 
    }

    """
    traj_gt_dir = _find_traj_dir(GT_DIR, traj_id)
    if traj_gt_dir is None:
        raise AssertionError(f"Trajet {traj_id} non trouvé dans {GT_DIR} (layout plat ou par ligne)")

    traj_rinex_obs_dir = _find_traj_dir(GNSS_OBS, traj_id)
    traj_rinex_nav_dir = _find_traj_dir(GNSS_NAV, traj_id)
    if traj_rinex_obs_dir is None or traj_rinex_nav_dir is None:
        raise RuntimeError(
            f"Fichiers RINEX OBS/NAV introuvables pour {traj_id} dans {GNSS_OBS} / {GNSS_NAV} "
            "(layout plat ou par ligne)"
        )

    line_id, traj_annexe_dir, line_common_annexe_dir, legacy_annexe_dir = _resolve_annexe_dirs(traj_id)
    traj_interim_dir, interim_line_dir, interim_legacy_dir = _resolve_work_dirs(INTERIM_DIR, traj_id, line_id)
    traj_processed_file, processed_line_dir, processed_legacy_dir = _resolve_work_dirs(PROCESSED_DIR, traj_id, line_id)
    lidar_tiles_dir, lidar_shared_root, lidar_tiles_legacy_dir = _resolve_shared_lidar_dirs(traj_interim_dir, line_id)
    lidar_urls_file = lidar_shared_root / f"urls_{line_id}.txt"
    lidar_size_cache_file = lidar_shared_root / "remote_sizes_cache.json"

    # Priorité: scenario annexe > legacy annexe > annexe commune de ligne.
    lever_candidates = [
        traj_annexe_dir / "Bras_de_levier.json",
        legacy_annexe_dir / "Bras_de_levier.json",
        line_common_annexe_dir / "Bras_de_levier.json",
    ]
    existing_lever_path = next((p for p in lever_candidates if p.exists()), None)
    lever_arm_path = existing_lever_path if existing_lever_path is not None else line_common_annexe_dir / "Bras_de_levier.json"
    lever_arm = parse_lever_arm(lever_arm_path)

    try:
        print(f"Recherche du fichier GT pour {traj_id} dans {traj_gt_dir}")
        gt_file = list(traj_gt_dir.glob("*.csv"))[0]
    except Exception as e:
        return f"Erreur : Aucun fichier GT trouve pour {traj_id} dans {GT_DIR} : {e}"

    try:
        print(f"Recherche du fichier RINEX OBS pour {traj_id} dans {traj_rinex_obs_dir}")
        obs_file = list(traj_rinex_obs_dir.glob("*.obs"))[0]
        print(f"Recherche du fichier RINEX NAV pour {traj_id} dans {traj_rinex_nav_dir}")
        nav_file = list(traj_rinex_nav_dir.glob("*.nav"))[0]
    except Exception as e:
        raise RuntimeError(
            f"Fichiers RINEX OBS ou NAV introuvables pour {traj_id} dans {traj_rinex_obs_dir} ou {traj_rinex_nav_dir} : {e}"
        ) from e

    return {
        "id": traj_id,
        "line_id": line_id,
        "raw_gt": gt_file,
        "interim_dir": traj_interim_dir,
        "final_dir": traj_processed_file,
        "raw_gnss": traj_interim_dir / f"gnss_position_{traj_id}.csv",
        "space_vehicule_info": traj_interim_dir / f"space_vehicule_info_{traj_id}.csv",
        "sync_csv": traj_interim_dir / f"fusion_gt_gnss_{traj_id}.csv",
        "features_csv": traj_interim_dir / f"features_lidar_{traj_id}.csv",
        "fusion_features_csv": traj_interim_dir / f"fusion_gt_gnss_lidar_features_{traj_id}.csv",
        "final_fusion_csv": traj_processed_file / f"fusion_finale_gnss_lidar_gt_label_{traj_id}.csv",
        "labels_plus_features_csv": traj_interim_dir / f"features_lidar_plus_labels_{traj_id}.csv",
        "lidar_tiles": lidar_tiles_dir,
        "lidar_url_list_file": lidar_urls_file,
        "lidar_size_cache_file": lidar_size_cache_file,
        "lidar_shared_root": lidar_shared_root,
        "lidar_tiles_legacy_dir": lidar_tiles_legacy_dir,
        "obs_file": obs_file,
        "nav_file": nav_file,
        "labels_csv": traj_processed_file / f"final_labeled_{traj_id}.csv",
        "gnss_offset": lever_arm,
        "gnss_features_csv": traj_processed_file / f"features_gnss_{traj_id}.csv",
        "annexe_traj_dir": traj_annexe_dir,
        "annexe_line_common_dir": line_common_annexe_dir,
        "lever_arm_file": lever_arm_path,
        "interim_line_dir": interim_line_dir,
        "interim_legacy_dir": interim_legacy_dir,
        "processed_line_dir": processed_line_dir,
        "processed_legacy_dir": processed_legacy_dir,
    }


def get_model_path(model_name: str, timestamp: str) -> dict[str, Path]:
    """Retourne le chemin des différentes infos du modele a partir de son ID (nom + date de creation),
      en utilisant les conventions de nommage. Par la suite on pourait faire un script qui scan les json pour retouver les modèles avec des param spécifiques.
      sortie  :
      {id : model_id,
       model_file: <chemin du fichier de poids et de parametres> .keras pour le GRU, .json pour le XGBoost,
       features: <chemin du fichier de features utilisées pour entrainer le modèle>
       scaler : <chemin du fichier du scaler utilisé pour entrainer le modèle>
       metadonnees: <chemin du fichier de metadonnees du modèle (date de creation, score, etc, id du dataset utilisé.)>
       results: <chemin du fichier de résultats d'évaluation du modèle sur le dataset de test>
       
       }"""
    model_id = f"{model_name}_{timestamp}"
    model_name_dir = MODELS_DIR / model_name
    model_name_dir.mkdir(parents=True, exist_ok=True)
    model_dir = model_name_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    model_file = model_dir / f"{model_id}.keras"
    features_file = model_dir / f"{model_id}_features.csv"
    scaler_file = model_dir / f"{model_id}_scaler.pkl"
    metadonnees_file = model_dir / f"{model_id}_metadonnees.json"
    results_file = model_dir / f"{model_id}_results.json"

    return {
        "id": model_id,
        "model_file": model_file,
        "features": features_file,
        "scaler": scaler_file,
        "metadata": metadonnees_file,
        "results": results_file,
    }

def get_dataset_path(dataset_name: str) -> dict[str, Path]:
    """Retourne le chemin du dataset a partir de son nom, en utilisant les conventions de nommage.
    {id: dataset_name,
     train: <chemin du fichier de train X + y>,
     test: <chemin du fichier de test X + y>,
     metadata: <chemin du fichier de metadonnees du dataset (date de creation, modele utilisé pour le créer, etc.)>}"""
    dataset_dir=TRAINING_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    classes_param_path = dataset_dir / f"{dataset_name}_classes_param.npy"
    scaler_param_path = dataset_dir / f"{dataset_name}_scaler.pkl"
    preprocessed_data_path = dataset_dir / f"{dataset_name}_preprocessed_data.npz"
    metadata_file = dataset_dir / f"{dataset_name}_metadata.json"
    label_encoder = dataset_dir / f"{dataset_name}_label_encoder.pkl"
    return {
        "id": dataset_name,
        "output_dir": dataset_dir,
        "classes_param": classes_param_path,
        "scaler_param": scaler_param_path,
        "preprocessed_data": preprocessed_data_path,
        "label_encoder_path": label_encoder,
        "metadata": metadata_file,
    }