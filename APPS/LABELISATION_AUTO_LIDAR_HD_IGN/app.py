""" 
Ce script execute l'ensemble du pipeline de sélection et de téléchargement des tuiles LiDAR puis
    de labelisation d'un trajet donné. Ce pipeline ne peut etre executé que une fois celui qui genere 
    les Wls a été lancé pour le trajet en question.
"""
import os
import sys
import traceback
import argparse
import json
import pandas as pd
from utils import CHUNK_SIZE, get_traj_paths, standardize_dataframe, WFS_URL, LIDAR_LAYER, WFS_VERSION, RAYON_RECHERCHE, N_WORKERS, DECIMATION_FACTOR, PARAMS_LABELISATION, LIDAR_DIR, merge_labelisation_params
from LABELISATION_AUTO_LIDAR_HD_IGN.telechargement_tuilles_lidar import download_tiles
from LABELISATION_AUTO_LIDAR_HD_IGN.choix_des_tuiles_lidar import download_tile_url_list
from LABELISATION_AUTO_LIDAR_HD_IGN.extract_lidar_features_labelisation import process_lidar_tiles_for_labelisation
from LABELISATION_AUTO_LIDAR_HD_IGN.run_params import store_labelisation_run_params
from FUSION.sync import (
    process_feature_fusion_from_files,
    process_final_label_fusion_from_files,
    process_gnss_gt_fusion,
)
from LABELISATION_AUTO_LIDAR_HD_IGN.labelisation import process_labelling

def _resolve_pipeline_options(options=None, **overrides):
    """Construit les options finales du pipeline à partir d'un dict et d'overrides éventuels."""
    resolved = {
        "nb_workers": 10,
        "verifier_integrite": False,
        "verbose": True,
        "extract_features": True,
        "chunk_size": CHUNK_SIZE,
        "spatial_mode": "circle",
        "search_radius": RAYON_RECHERCHE,
        "corridor_width": 6.0,
        "corridor_length": 30.0,
        "min_elevation_angle_deg": 0.0,
        "run_step_1_select_tiles": True,
        "run_step_2_download_tiles": True,
        "run_step_3_fusion_gt_gnss": False,
        "run_step_4_extract_lidar": True,
        "run_step_5_fusion_features": True,
        "run_step_6_labelisation": True,
        "run_step_7_final_fusion": True,
        "params_labelisation": dict(PARAMS_LABELISATION),
    }
    if isinstance(options, dict):
        resolved.update(options)

    # Les paramètres explicites priment si fournis.
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value

    return resolved


def pipeline_labelisation(
    traj_id,
    options=None,
    nb_workers=None,
    verifier_integrite=None,
    verbose=None,
    extract_features=None,
    chunk_size=None,
    spatial_mode=None,
    search_radius=None,
    corridor_width=None,
    corridor_length=None,
    min_elevation_angle_deg=None,
):
    """
    Pipeline complet de labelisation.
    
    Args:
        traj_id: ID du trajet
        options: Dictionnaire d'options du pipeline (prioritaire sur les defaults)
        nb_workers: Override du nombre de téléchargements parallèles
        verifier_integrite: Override de la vérification d'intégrité des fichiers LAZ
        verbose: Override de l'affichage des messages de progression
        extract_features: Override du recalcul des features LiDAR
        chunk_size: Override de la taille des chunks de téléchargement
        spatial_mode: Override du mode spatial pour la sélection des tuiles
    """
    opts = _resolve_pipeline_options(
        options=options,
        nb_workers=nb_workers,
        verifier_integrite=verifier_integrite,
        verbose=verbose,
        extract_features=extract_features,
        chunk_size=chunk_size,
        spatial_mode=spatial_mode,
        search_radius=search_radius,
        corridor_width=corridor_width,
        corridor_length=corridor_length,
        min_elevation_angle_deg=min_elevation_angle_deg,
    )

    nb_workers = int(opts["nb_workers"])
    verifier_integrite = bool(opts["verifier_integrite"])
    verbose = bool(opts["verbose"])
    extract_features = bool(opts["extract_features"])
    chunk_size = int(opts["chunk_size"])
    spatial_mode = str(opts["spatial_mode"])
    search_radius = float(opts["search_radius"])
    corridor_width = float(opts["corridor_width"])
    corridor_length = float(opts["corridor_length"])
    min_elevation_angle_deg = float(opts.get("min_elevation_angle_deg", 0.0))
    run_step_1_select_tiles = bool(opts.get("run_step_1_select_tiles", True))
    run_step_2_download_tiles = bool(opts.get("run_step_2_download_tiles", True))
    run_step_3_fusion_gt_gnss = bool(opts.get("run_step_3_fusion_gt_gnss", False))
    run_step_4_extract_lidar = bool(opts.get("run_step_4_extract_lidar", True))
    run_step_5_fusion_features = bool(opts.get("run_step_5_fusion_features", True))
    run_step_6_labelisation = bool(opts.get("run_step_6_labelisation", True))
    run_step_7_final_fusion = bool(opts.get("run_step_7_final_fusion", True))
    params_labelisation = merge_labelisation_params(opts.get("params_labelisation"))
    opts["params_labelisation"] = params_labelisation

    if verbose:
        print(f"--- Pipeline de labelisation pour : {traj_id} ---")

    config = get_traj_paths(traj_id)
    list_url_file = config.get("lidar_url_list_file", config["lidar_tiles"] / f"urls_{traj_id}.txt")
    size_cache_file = config.get("lidar_size_cache_file", config["lidar_tiles"] / "remote_sizes_cache.json")
    features_file = config.get("fusion_features_csv")
    labels_file = config.get("lidar_labels_csv")

    try:
        run_params_file, latest_params_file, _ = store_labelisation_run_params(
            config=config,
            traj_id=traj_id,
            source="IGN",
            pipeline_opts=opts,
            params_labelisation=params_labelisation,
            decimation_factor=DECIMATION_FACTOR,
        )
        print(f"Parametres run sauvegardes: {run_params_file}")
        print(f"Parametres run (latest): {latest_params_file}")
    except Exception as e:
        print(f"Attention: impossible de sauvegarder les parametres de run ({e})")
    
    print("=" * 50)
    print("ÉTAPE 1 : Sélection des tuiles LiDAR à telécharger")
    print("=" * 50)
    if run_step_1_select_tiles:
        try:
            has_existing_urls = False
            existing_count = 0
            if os.path.exists(list_url_file):
                with open(list_url_file, "r", encoding="utf-8") as f:
                    existing_count = sum(1 for line in f if line.strip())
                has_existing_urls = existing_count > 0

            # Si on a déjà une liste non vide, on saute la requête WFS.
            if has_existing_urls:
                print(f"Liste de tuiles déjà présente ({existing_count} URLs), étape 1 ignorée : {list_url_file}")
            else:
                download_tile_url_list(
                    traj_id=traj_id,
                    gt_file=config["raw_gt"],
                    output_file=list_url_file,
                    wfs_url=WFS_URL,
                    target_layer=LIDAR_LAYER,
                    version=WFS_VERSION,
                    verbose=True
                )
        except Exception as e:
            print(f"Erreur lors de la sélection des tuiles : {e}")
            return False
    else:
        print("Étape 1 ignorée (désactivée dans les options).")

    print("=" * 50)
    print("ÉTAPE 2 : Téléchargement des tuiles LiDAR")
    print("=" * 50)
    if run_step_2_download_tiles:
        try:
            stats = download_tiles(
                traj_tiles_dir=config["lidar_tiles"],
                file_list=list_url_file,
                output_folder=LIDAR_DIR,
                max_workers=nb_workers,
                chunk_size=chunk_size,
                verify_integrity=verifier_integrite,
                size_cache_file=size_cache_file,
                refresh_sizes=False,
                prefetch_sizes=True,
                verbose=True
            )

            if stats['fail'] > 0:
                print(f"\nAttention : {stats['fail']} fichiers n'ont pas pu être téléchargés.")
                return False

        except FileNotFoundError as e:
            print(f"Erreur : {e}")
            return False
        except ValueError as e:
            print(f"Erreur : {e}")
            return False
        except Exception as e:
            print(f"Erreur inattendue : {e}")
            traceback.print_exc()
            return False
    else:
        print("Étape 2 ignorée (désactivée dans les options).")
    
    print("\n" + "=" * 50)
    print("ÉTAPE 3 : fusion de la GT et du GNSS pur")
    print("=" * 50)
    if run_step_3_fusion_gt_gnss:
        # TODO : faire un messgae auto qui demande à bien vérifier les colones et à les ajouter dans le dictionaire de mapping si besoin, et à vérifier que les fichiers d'entrée ont bien les colonnes nécessaires (ex: time_utc, latitude, longitude, altitude) avant de lancer la fusion.
        try:
            # Charger les données GT et GNSS
            df_gt = pd.read_csv(config["raw_gt"])
            df_gnss = pd.read_csv(config["raw_gnss"])
            gnss_offset = config["gnss_offset"]
            if df_gnss is None:
                print("Fichier GNSS non trouvé, étape fusion ignorée.")
            else:
                process_gnss_gt_fusion(
                    df_gt=standardize_dataframe(df_gt),
                    df_gnss=df_gnss,
                    output_csv=config["sync_csv"],
                    gnss_offset=gnss_offset,
                    verbose=True
                )
                print("Fusion GT-GNSS complétée avec succès !")
        except FileNotFoundError as e:
            print(f"Fichier manquant : {e}")
            print("   Assurez-vous d'avoir généré le fichier PVT avec le script d'exploitation des RINEXs.")
            return False
        except ValueError as e:
            print(f"Erreur lors de la fusion : {e}")
            return False
        except Exception as e:
            print(f"Erreur inattendue lors de la fusion : {e}")
            traceback.print_exc()
            return False
    else:
        print("Étape 3 ignorée (désactivée dans les options).")

    print("\n" + "=" * 50)
    print("ÉTAPE 4 : Extraction des features LiDAR")
    print("=" * 50)

    if run_step_4_extract_lidar:
        gnss_offset_z = config.get("gnss_offset", (0.0, 0.0, 0.0))[2] if config.get("gnss_offset") else None
        if extract_features or (not os.path.exists(config["lidar_features_csv"])):
            # On recalcule les features si le fichier n'existe pas ou si l'extraction est demandée
            print("Extraction des features, fichier non présent ou extraction demandée : ", config["lidar_features_csv"])
            try:
                process_lidar_tiles_for_labelisation(
                    traj_id,
                    output_csv=config["lidar_features_csv"],
                    search_radius=search_radius,
                    decimation_factor=DECIMATION_FACTOR,
                    n_workers=nb_workers,
                    spatial_mode=spatial_mode,
                    corridor_width=corridor_width,
                    corridor_length=corridor_length,
                    gnss_offset_z=gnss_offset_z,
                    min_elevation_angle_deg=min_elevation_angle_deg,
                )
                print("Extraction des features terminée !")
            except Exception as e:
                print(f"Erreur lors de l'extraction des features : {e}")
                return False
        else:
            print("Fichier de features déjà présent et extraction non demandée, étape 4 ignorée : ", config["lidar_features_csv"])
    else:
        print("Étape 4 ignorée (désactivée dans les options).")

    print("\n" + "=" * 50)
    print("ÉTAPE 5 : Fusion GT + features GNSS + features LiDAR")
    print("=" * 50)

    if run_step_5_fusion_features:
        try:
            gnss_features_file = config["gnss_features_csv"]
            lidar_features_file = config["lidar_features_csv"]
            fusion_features_file = config["fusion_features_csv"]

            if not os.path.exists(gnss_features_file):
                print(f"Fichier features GNSS absent : {gnss_features_file}")
                print("Lance d'abord l'extraction GNSS (module EXTRACTION_DES_FEATURES_GNSS).")
                return False

            if not os.path.exists(lidar_features_file):
                print(f"Fichier features LiDAR absent : {lidar_features_file}")
                print("L'etape 4 doit produire ce fichier avant la fusion des features.")
                return False

            process_feature_fusion_from_files(
                path_lidar_features=lidar_features_file,
                path_gnss_features=gnss_features_file,
                path_gt=config["raw_gt"],
                output_csv=fusion_features_file,
                verbose=True,
            )
            print(f"Fusion des features terminee : {fusion_features_file}")
        except Exception as e:
            print(f"Erreur lors de la fusion des features : {e}")
            traceback.print_exc()
            return False
    else:
        print("Étape 5 ignorée (désactivée dans les options).")

    print("\n" + "=" * 50)
    print("ÉTAPE 6 : Labellisation de l'environnement avec le lidar HD")
    print("=" * 50)

    if run_step_6_labelisation:
        try:
            # Le fichier d'entrée est la sortie de l'étape 5 (fusion multi-sources)
            features_file = config.get("fusion_features_csv")
            if features_file is None:
                print("⚠ Chemin du fichier enrichi non configuré, utilisation du nom par défaut")
                # Génération d'un nom par défaut basé sur le trajet
                features_file = config["final_dir"] / f"env_lidar_{traj_id}.csv"

            labels_file = config.get("lidar_labels_csv")
            labels_features_file = config.get("labels_plus_features_csv")
            if labels_file is None:
                print("⚠ Chemin du fichier de labels final non configuré, utilisation du nom par défaut")
            if labels_features_file is None:
                print("⚠ Chemin du fichier de fusion labels+features non configuré, utilisation du nom par défaut")

            process_labelling(
                input_csv=str(features_file),
                params=params_labelisation,
                output_csv_final=str(labels_file),
                output_csv_interim=str(labels_features_file) if labels_features_file else None,
                verbose=True
            )
            print(f"Labellisation terminée ! Résultats : {labels_file}")
        except FileNotFoundError as e:
            print(f"Fichier d'entrée manquant : {e}")
            print("   Assurez-vous que l'étape 4 (extraction des features) a été exécutée avec succès.")
            return False
        except ValueError as e:
            print(f"Erreur lors de la labellisation : {e}")
            return False
        except Exception as e:
            print(f"Erreur inattendue lors de la labellisation : {e}")
            return False
    else:
        print("Étape 6 ignorée (désactivée dans les options).")
    


    print("\n" + "=" * 50)
    print("ÉTAPE 7 : Fusion finale LiDAR + GNSS + GT + label")
    print("=" * 50 + "\n")
    if run_step_7_final_fusion:
        try:
            final_fusion_file = config.get("final_fusion_csv")
            if labels_file and features_file and final_fusion_file:
                process_final_label_fusion_from_files(
                    path_fused_features=features_file,
                    path_labels=labels_file,
                    output_csv=final_fusion_file,
                    verbose=True,
                )
                print(f"Fusion finale terminee : {final_fusion_file}")
            else:
                print("Chemins de sortie non configures, etape 7 ignoree.")
        except Exception as e:
            print(f"Erreur lors de la fusion finale : {e}")
            traceback.print_exc()
            return False
    else:
        print("Étape 7 ignorée (désactivée dans les options).")
    
    print("\n" + "=" * 50)
    print("Pipeline de labelisation terminé avec succès !")
    print("=" * 50 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traitement des fichiers RINEX")
    parser.add_argument("--traj", required=True, help="ID du trajet à traiter")
    parser.add_argument("--workers", type=int, default=None, help="Nombre de workers pour le telechargement/extraction")
    parser.add_argument("--chunk-size", type=int, default=None, help="Taille des chunks de telechargement")
    parser.add_argument("--verify-integrity", action="store_true", help="Verifier l'integrite des fichiers LAZ")
    parser.add_argument("--extract-features", action="store_true", help="Forcer la re-extraction des features LiDAR")
    parser.add_argument("--skip-extract-features", action="store_true", help="Ne pas re-extraire les features LiDAR si deja presentes")
    parser.add_argument(
        "--options-json",
        type=str,
        default=None,
        help="Dictionnaire JSON des options pipeline (ex: '{\"nb_workers\": 8, \"chunk_size\": 1048576}')",
    )
    args = parser.parse_args()
    try:
        # Configuration : ID du trajet à traiter
        TRAJ_ID = args.traj  # Exemple : "BORDEAUX_COUTRAS"

        options_from_json = {}
        if args.options_json:
            try:
                options_from_json = json.loads(args.options_json)
                if not isinstance(options_from_json, dict):
                    raise ValueError("--options-json doit representer un objet JSON")
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON invalide pour --options-json: {e}") from e

        if args.extract_features and args.skip_extract_features:
            raise ValueError("Options incompatibles: --extract-features et --skip-extract-features")

        extract_features_override = None
        if args.extract_features:
            extract_features_override = True
        elif args.skip_extract_features:
            extract_features_override = False
        
        # Lancer le pipeline
        print(f"\n{'='*50}")
        print(f"Pipeline de labelisation - Trajet : {TRAJ_ID}")
        print(f"{'='*50}\n")
        
        pipeline_options = dict(options_from_json)
        pipeline_options.setdefault("verbose", True)

        if args.workers is not None:
            pipeline_options["nb_workers"] = int(args.workers)
        elif "nb_workers" not in pipeline_options:
            pipeline_options["nb_workers"] = int(N_WORKERS)

        if args.chunk_size is not None:
            pipeline_options["chunk_size"] = int(args.chunk_size)
        elif "chunk_size" not in pipeline_options:
            pipeline_options["chunk_size"] = int(CHUNK_SIZE)

        if args.verify_integrity:
            pipeline_options["verifier_integrite"] = True
        elif "verifier_integrite" not in pipeline_options:
            pipeline_options["verifier_integrite"] = False

        if extract_features_override is not None:
            pipeline_options["extract_features"] = extract_features_override
        elif "extract_features" not in pipeline_options:
            pipeline_options["extract_features"] = False

        success = pipeline_labelisation(
            traj_id=TRAJ_ID,
            options=pipeline_options,
        )
        
        if success:
            print(f"\n{'='*25}")
            print("Pipeline terminé avec succès !")
            print(f"{'='*25}")
            sys.exit(0)
        else:
            print(f"\n{'='*25}")
            print("Le pipeline a rencontré des erreurs.")
            print(f"{'='*25}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nInterruption par l'utilisateur.")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*25}")
        print(f"Erreur : {e}")
        print(f"{'='*25}")
        sys.exit(1)