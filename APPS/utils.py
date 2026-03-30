import os
from pathlib import Path

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
    except Exception as e:
        print(f"Erreur lors de la lecture des metadonnees de bras de levier : {e}")
        return (0.0, 0.0, 0.0)


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
      "lidar_tiles": <chemin du dossier contenant les tuiles LiDAR>,
      "obs_file": <chemin du fichier RINEX OBS>,
      "nav_file": <chemin du fichier RINEX NAV>,
      "labels_csv": <chemin du fichier final traité>,
      "gnss_offset": <tuple du bras de levier (x,y,z)>
      "gnss_features_csv": <chemin du fichier de features extraites du GNSS> 
    }

    """
    assert traj_id in [dir.name for dir in GT_DIR.iterdir() if dir.is_dir()], f"Trajet {traj_id} non trouvé dans la liste"
    traj_interim_dir = INTERIM_DIR / traj_id
    traj_interim_dir.mkdir(parents=True, exist_ok=True)
    traj_gt_dir = GT_DIR / traj_id
    traj_processed_file = PROCESSED_DIR / traj_id
    traj_processed_file.mkdir(parents=True, exist_ok=True)
    traj_annexe_dir = RAW_DIR / "ANNEXE_TRAJET" / traj_id
    traj_annexe_dir.mkdir(parents=True, exist_ok=True)
    traj_rinex_obs_dir = GNSS_OBS / traj_id
    traj_rinex_nav_dir = GNSS_NAV / traj_id

    lever_arm = parse_lever_arm(traj_annexe_dir / "Bras_de_levier.json")

    try:
        print(f"Recherche du fichier GT pour {traj_id} dans {GT_DIR}")
        gt_file = list(traj_gt_dir.glob("*.csv"))[0]
    except Exception as e:
        return f"Erreur : Aucun fichier GT trouve pour {traj_id} dans {GT_DIR} : {e}"

    try:
        print(f"Recherche du fichier RINEX OBS pour {traj_id} dans {GNSS_OBS}")
        obs_file = list(traj_rinex_obs_dir.glob("*.obs"))[0]
        print(f"Recherche du fichier RINEX NAV pour {traj_id} dans {GNSS_NAV}")
        nav_file = list(traj_rinex_nav_dir.glob("*.nav"))[0]
    except Exception as e:
        raise RuntimeError(
            f"Fichiers RINEX OBS ou NAV introuvables pour {traj_id} dans {GNSS_OBS} ou {GNSS_NAV} : {e}"
        ) from e

    return {
        "id": traj_id,
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
        "lidar_tiles": traj_interim_dir / "tiles_laz",
        "obs_file": obs_file,
        "nav_file": nav_file,
        "labels_csv": traj_processed_file / f"final_labeled_{traj_id}.csv",
        "gnss_offset": lever_arm,
        "gnss_features_csv": traj_processed_file / f"features_gnss_{traj_id}.csv"
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