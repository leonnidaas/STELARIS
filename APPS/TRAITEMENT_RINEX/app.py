import subprocess
import sys
from utils import get_traj_paths, PYTHON_RINEX_INTERPRETER
from TRAITEMENT_RINEX.proto_data import process_rinex_files
import argparse

def main(traj_id : str):

    config = get_traj_paths(traj_id)
    nav_file = config.get("nav_file")
    obs_file = config.get("obs_file")
    pvt_output = config.get("raw_gnss")
    space_vehicule_info_output = config.get("space_vehicule_info")

    process_rinex_files( obs_file,nav_file, space_vehicule_info_output, pvt_output, verbose=True)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traitement des fichiers RINEX")
    parser.add_argument("--traj", required=True, help="ID du trajet à traiter")
    args = parser.parse_args()
    main(args.traj)