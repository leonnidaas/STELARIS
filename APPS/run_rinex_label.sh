#!/bin/bash

trajets=("BORDEAUX_COUTRAS" "COUTRAS_BORDEAUX" "BORDEAUX_YGOS-SAINT-SATURNIN" "MARTINE_01" "MARTINE_02")
    

for traj in "${trajets[@]}"; do
    # echo "Traitement de la trajectoire : $traj"
    # source venv_rinex/bin/activate
    # python3 -m TRAITEMENT_RINEX.app --traj "$traj"
    # deactivate

    source venv_label/bin/activate
    python3 -m EXTRACTION_DES_FEATURES_GNSS.app --traj "$traj"
    python3 -m LABELISATION_AUTO_LIDAR_HD_IGN.app --traj "$traj"
    deactivate
done



#--- ETAPE 1 : TRAITEMENT RINEX ---
# source venv_rinex/bin/activate
# python3 -m TRAITEMENT_RINEX.app --traj "BORDEAUX_COUTRAS"
# deactivate


# # --- ETAPE 2 : LABÉLISATION & ML ---
# source venv_label/bin/activate
# python3 -m LABELISATION_AUTO_LIDAR_HD_IGN.app --traj "BORDEAUX_COUTRAS"
# deactivate