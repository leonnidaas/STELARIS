"""
Module de labellisation automatique de l'environnement GNSS.

Ce module prend les données enrichies par extract_lidar_features et applique
des règles de labellisation pour catégoriser l'environnement (open-sky, tunnel,
bridge, tree, build, mixed, gare, other).
"""

import argparse
import pandas as pd
import numpy as np
import sys




def auto_label_environment(df_input,  params, output_csv_final=None, output_csv_interim=None, verbose=True):
    """
    Labellisation automatique de l'environnement GNSS basée sur les features LiDAR.
    
    Le DataFrame d'entrée doit contenir au moins les colonnes suivantes :
        - time_utc : Timestamp de chaque point
        - latitude_gt, longitude_gt : Coordonnées GPS
        - sky_mask_deg : Masque ciel en degrés
        - obs_type : Type d'obstruction (0=aucune, 1=bâtiment, 2=végétation, 3=mixte, 4=bridge)
        - is_under_structure : 1 si sous une structure (pont/tunnel), 0 sinon
        - veg_density : Densité de végétation dans un rayon de 20m (0 à 1)
        - signal_denied : 1 si signal GPS denied (tunnel), 0 sinon

    params est un dictionnaire de paramètres qui dois contenir:
        - seuil_vegetation : Seuil de densité de végétation pour labelliser 'tree'
        - seuil_melange : Seuil de densité de végétation pour labelliser 'mixed' si obs_type=0
        - seuil_ciel_ouvert : Seuil de sky_mask_smoothed pour labelliser 'open-sky'
        - distance_scan : Distance en nombre de points pour le lissage du sky_mask
    
    Parameters
    ----------
    df_input : pd.DataFrame
        DataFrame contenant les features extraites du LiDAR
    output_csv_final : str, optional
        Chemin du fichier CSV de sortie (si None, pas de sauvegarde)
    output_csv_interim : str, optional
        Chemin du fichier CSV intermédiaire pour l'analyse (si None, pas de sauvegarde)
    params : dict
        Dictionnaire de paramètres pour la labellisation (seuils, distances, etc.)
    verbose : bool, optional
        Afficher les messages de progression (default: True)
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec les colonnes 'latitude_gt', 'longitude_gt', 'label'
        
    Labels possibles :
        - 'open-sky' : Ciel dégagé
        - 'signal_denied' : Tunnel (signal GPS denied)
        - 'bridge' : Sous un pont
        - 'tree' : Environnement végétalisé
        - 'build' : Environnement bâti
        - 'mixed' : Environnement mixte (bâti + végétation)
        - 'gare' : Gare / Zone de triage
        - 'other' : Autre (non catégorisé)
    
    Raises
    ------
    ValueError
        Si des colonnes requises sont manquantes
    """
    # Vérification des colonnes requises
    required_cols = ['time_utc', 'latitude_gt', 'longitude_gt', 'sky_mask_deg', 
                     'obs_type', 'is_under_structure', 'veg_density']
    missing_cols = [col for col in required_cols if col not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {missing_cols}")
    
    df = df_input.copy()
    df['time_utc'] = pd.to_datetime(df['time_utc'])
    df = df.sort_values('time_utc')

    if verbose:
        print(f"Labellisation de {len(df)} points...")

    # 1. Initialisation du label par défaut
    df['label'] = 'other'  # Valeur par défaut si aucune condition n'est remplie
    
    # 2. Identification des structures (Ponts / signal_denied)
    
    # Tunnel (signal denied)
    if 'signal_denied' in df.columns:
        df.loc[(df['signal_denied'] == 1), 'label'] = 'signal_denied'
    
    # Ponts (sous structure mais pas de signal denied)
    df.loc[(df['is_under_structure'] == 1) & (df['obs_type'] == 4), 'label'] = 'bridge'  # obs_type 4 = bridge

    # 3. Labellisation par obstruction (si pas déjà sous une structure)
    mask_no_struct = ~df['label'].str.contains('bridge|signal_denied')

    # Tree
    df.loc[mask_no_struct & (df['veg_density'] > params['seuil_vegetation']) & (df['obs_type'] == 2), 'label'] = 'tree'
    
    # Build
    df.loc[mask_no_struct & (df['obs_type'] == 1), 'label'] = 'build'
    
    # Mixed
    df.loc[mask_no_struct & (df['obs_type'] == 3), 'label'] = 'mixed'
    df.loc[mask_no_struct & (df['veg_density'] > params['seuil_melange']) & (df['obs_type'] == 0), 'label'] = 'mixed'
    
    # Open Sky Urban vs Rural
    # On considère Urban si du bâti est détecté dans le voisinage (obs_type 1 ou 3)
    df['sky_mask_smoothed'] = df['sky_mask_deg'].rolling(window=params['distance_scan'], center=True).mean()
    df.loc[mask_no_struct & (df['sky_mask_smoothed'] < params['seuil_ciel_ouvert']), 'label'] = 'open-sky'
    
    
    # Gare / Triage (Basé sur la densité de points non-naturels et structure)
    # Souvent caractérisé par beaucoup de caténaires et structures proches
    gare_mask = (df['is_under_structure'] == 1) & (df['obs_type'] == 1)
    df.loc[gare_mask, 'label'] = 'gare'

    # Nettoyage et statistiques
    df_result = df[['latitude_gt', 'longitude_gt', 'label']].copy()
    
    if verbose:
        print("\n=== Résultat de la labellisation ===")
        label_counts = df_result['label'].value_counts()
        for label, count in label_counts.items():
            pct = 100 * count / len(df_result)
            print(f"  {label:15s} : {count:6d} points ({pct:5.1f}%)")
    
    # Sauvegarde si demandée
    if output_csv_final:
        df_result.to_csv(output_csv_final, index=False)
        if verbose:
            print(f"\n✓ Résultat sauvegardé : {output_csv_final}")
    if output_csv_interim:
        df.to_csv(output_csv_interim, index=False)
        if verbose:
            print(f"✓ Fichier intermédiaire sauvegardé : {output_csv_interim}")
    
    return df_result


def process_labelling(input_csv, params, output_csv_final=None, output_csv_interim=None, verbose=True):
    """
    Pipeline complet de labellisation depuis un fichier CSV enrichi.
    
    Parameters
    ----------
    input_csv : str
        Chemin du fichier CSV d'entrée (sortie de extract_lidar_features)
    params : dict
        Dictionnaire de paramètres pour la labellisation (seuils, distances, etc.)
    output_csv_final : str, optional
        Chemin du fichier CSV de sortie (si None, génère un nom automatique)
    output_csv_interim : str, optional
        Chemin du fichier CSV intermédiaire pour l'analyse (si None, pas de sauvegarde)
    verbose : bool, optional
        Afficher les messages de progression
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec les labels appliqués
    
    Raises
    ------
    FileNotFoundError
        Si le fichier d'entrée n'existe pas
    ValueError
        Si le fichier d'entrée est invalide
    """
    import os
    
    # Vérifier que le fichier existe
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Le fichier {input_csv} n'existe pas.")
    
    if verbose:
        print(f"Chargement du fichier : {input_csv}")
    
    # Charger les données
    df_input = pd.read_csv(input_csv)
    
    if len(df_input) == 0:
        raise ValueError(f"Le fichier {input_csv} est vide.")
    
    # Générer un nom de sortie si non fourni
    if output_csv_final is None:
        base_name = os.path.splitext(input_csv)[0]
        output_csv_final = f"{base_name}_labeled.csv"
    
    # Appliquer la labellisation
    df_result = auto_label_environment(df_input, params=params, output_csv_final=output_csv_final, output_csv_interim=output_csv_interim, verbose=verbose)
    
    return df_result


def get_parser():
    """Parser pour la CLI."""
    parser = argparse.ArgumentParser(
        description="Labellisation automatique de l'environnement GNSS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Exemples d'utilisation :
    %(prog)s -i features_enrichies.csv -o labels.csv
    %(prog)s -i features_enrichies.csv --interim interim.csv
    %(prog)s -i features_enrichies.csv --quiet
            """
        )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Fichier CSV d\'entrée (sortie de extract_lidar_features)')
    parser.add_argument('-p', '--params', default=None,
                        help='Fichier de paramètres JSON')
    parser.add_argument('-o', '--output', default=None,
                        help='Fichier CSV de sortie (génère un nom automatique si non fourni)')
    parser.add_argument('--interim', default=None,
                        help='Fichier CSV intermédiaire pour l\'analyse (si None, pas de sauvegarde)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Mode silencieux (pas de messages)')
    
    return parser


def main():
    """Point d'entrée pour la CLI."""
    args = get_parser().parse_args()
    
    try:
        process_labelling(
            input_csv=args.input,
            output_csv_final=args.output,
            output_csv_interim=args.interim,
            params=args.params,
            verbose=not args.quiet
        )
        return 0
    except FileNotFoundError as e:
        print(f"ERREUR : {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"ERREUR : {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERREUR inattendue : {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":

    from APPS.utils import get_traj_paths

    traj_id = "BORDEAUX_COUTRAS"
    config = get_traj_paths(traj_id)
    features_file = config.get("features_csv")
    