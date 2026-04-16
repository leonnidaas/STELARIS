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


REQUIRED_COLS = [
    'time_utc',
    'latitude_gt',
    'longitude_gt',
    'sky_mask_deg',
    'obs_type',
    'is_under_structure',
    'veg_density',
]

OPTIONAL_NUMERIC_COLS = {
    'signal_denied': 0.0,
    'is_bridge': 0.0,
    'n_points_zone': 0.0,
    'enough_points_flag': 0.0,
    'density_near_0_5m': 0.0,
    'density_mid_5_15m': 0.0,
    'density_far_15_30m': 0.0,
    'zrel_p50': 0.0,
    'zrel_p90': 0.0,
    'zrel_p95': 0.0,
    'zrel_p99': 0.0,
    'zrel_iqr': 0.0,
    'zrel_std': 0.0,
    'occupation_ciel_azimuth_ratio': 0.0,
    'building_density': 0.0,
    'vegetation_density_low': 0.0,
    'vegetation_density_mid': 0.0,
    'vegetation_density_high': 0.0,
    'bridge_density': 0.0,
    'bridge_above_density': 0.0,
    'bridge_above_count': 0.0,
    'canopee_ratio': 0.0,
    'obstacle_overhead_ratio': 0.0,
}


def _ensure_numeric_column(df, col_name, default_value=0.0):
    if col_name not in df.columns:
        df[col_name] = default_value
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(default_value)


def _infer_speed_gt_mps(df):
    speed_candidates = [
        "speed_gt_mps",
        "vitesse_gt",
        "velocity_gt",
        "speed",
        "velocity",
    ]
    for c in speed_candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if all(col in df.columns for col in ["x_gt", "y_gt", "time_utc"]):
        x = pd.to_numeric(df["x_gt"], errors="coerce")
        y = pd.to_numeric(df["y_gt"], errors="coerce")
        dt = pd.to_datetime(df["time_utc"]).diff().dt.total_seconds().fillna(0.0)
        dx = x.diff().fillna(0.0)
        dy = y.diff().fillna(0.0)
        dist = np.sqrt(dx * dx + dy * dy)
        dt = dt.replace(0.0, np.nan)
        v = (dist / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return v

    return pd.Series(np.zeros(len(df), dtype=float), index=df.index)


def _load_labelisation_params(params):
    p = params or {}
    return {
        'seuil_vegetation': float(p.get('seuil_vegetation', 0.60)),
        'seuil_melange': float(p.get('seuil_melange', 0.20)),
        'seuil_ciel_ouvert': float(p.get('seuil_ciel_ouvert', 24.0)),
        'distance_scan': int(p.get('distance_scan', 5)),
        'seuil_occupation_ciel': float(p.get('seuil_occupation_ciel', 0.12)),
        'seuil_overhead_bridge': float(p.get('seuil_overhead_bridge', 0.06)),
        'seuil_overhead_gare': float(p.get('seuil_overhead_gare', 0.12)),
        'seuil_veg_high': float(p.get('seuil_veg_high', 0.50)),
        'seuil_canopee': float(p.get('seuil_canopee', 0.10)),
        'seuil_building_density': float(p.get('seuil_building_density', 0.08)),
        'seuil_mixed_building': float(p.get('seuil_mixed_building', 0.06)),
        'seuil_mixed_veg': float(p.get('seuil_mixed_veg', 0.18)),
        'seuil_min_points_zone': int(p.get('seuil_min_points_zone', 20)),
        'seuil_vitesse_gare_mps': float(p.get('seuil_vitesse_gare_mps', 6.0)),
        'seuil_bridge_density_min': float(p.get('seuil_bridge_density_min', 0.015)),
        'seuil_bridge_above_density_min': float(p.get('seuil_bridge_above_density_min', 0.001)),
        'seuil_bridge_above_count_min': int(p.get('seuil_bridge_above_count_min', 0)),
        'seuil_bridge_zrel_p95_min': float(p.get('seuil_bridge_zrel_p95_min', 2.2)),
        'seuil_bridge_zrel_p99_min': float(p.get('seuil_bridge_zrel_p99_min', 3.5)),
        'bridge_min_score': int(p.get('bridge_min_score', 1)),
        'bridge_persistence_window_s': float(p.get('bridge_persistence_window_s', 1.0)),
        'seuil_gare_density_near': float(p.get('seuil_gare_density_near', 0.22)),
        'seuil_gare_zrel_iqr': float(p.get('seuil_gare_zrel_iqr', 1.2)),
        'seuil_tree_veg_mid': float(p.get('seuil_tree_veg_mid', 0.20)),
        'seuil_tree_zrel_p90': float(p.get('seuil_tree_zrel_p90', 2.5)),
        'seuil_build_density_mid': float(p.get('seuil_build_density_mid', 0.30)),
        'seuil_build_zrel_p95': float(p.get('seuil_build_zrel_p95', 2.2)),
        'seuil_open_sky_density_near_max': float(p.get('seuil_open_sky_density_near_max', 0.16)),
        'seuil_open_sky_density_far_max': float(p.get('seuil_open_sky_density_far_max', 0.35)),
        'seuil_open_sky_zrel_p95_max': float(p.get('seuil_open_sky_zrel_p95_max', 2.2)),
        'seuil_open_sky_zrel_std_max': float(p.get('seuil_open_sky_zrel_std_max', 1.6)),
        'seuil_open_sky_soft_score': int(p.get('seuil_open_sky_soft_score', 4)),
        'seuil_mixed_zrel_iqr': float(p.get('seuil_mixed_zrel_iqr', 1.6)),
        'seuil_mixed_zrel_std': float(p.get('seuil_mixed_zrel_std', 1.4)),
    }


def _prepare_labelisation_df(df_input):
    missing_cols = [col for col in REQUIRED_COLS if col not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {missing_cols}")

    df = df_input.copy()
    df['time_utc'] = pd.to_datetime(df['time_utc'])
    df = df.sort_values('time_utc')

    for col_name, default_value in OPTIONAL_NUMERIC_COLS.items():
        _ensure_numeric_column(df, col_name, default_value=default_value)

    df['speed_gt_mps'] = df["velocity"].abs() if "velocity" in df.columns else _infer_speed_gt_mps(df)
    df['speed_gt_mps_smooth'] = (
        df['speed_gt_mps']
        .rolling(window=3, center=True, min_periods=1)
        .median()
        .fillna(df['speed_gt_mps'])
    )
    return df


def _compute_bridge_core_and_recent(df, cfg):
    bridge_semantic = (df['obs_type'] == 4).astype(int)
    bridge_density_hit = (df['bridge_density'] > cfg['seuil_bridge_density_min']).astype(int)
    bridge_above_hit = (
        (df['bridge_above_density'] > cfg['seuil_bridge_above_density_min'])
        & (df['bridge_above_count'] >= cfg['seuil_bridge_above_count_min'])
    ).astype(int)
    bridge_overhead_hit = (df['obstacle_overhead_ratio'] > cfg['seuil_overhead_bridge']).astype(int)
    bridge_height_hit = (
        (df['zrel_p95'] > cfg['seuil_bridge_zrel_p95_min'])
        | (df['zrel_p99'] > cfg['seuil_bridge_zrel_p99_min'])
    ).astype(int)

    bridge_score = bridge_semantic + bridge_density_hit + bridge_overhead_hit
    bridge_core_strong = (
        (df['is_under_structure'] == 1)
        & (bridge_above_hit == 1)
        & (
            ((bridge_semantic == 1) & ((bridge_overhead_hit == 1) | (bridge_density_hit == 1)))
            | ((bridge_semantic == 1) & (bridge_height_hit == 1))
        )
    )
    bridge_core_scored = (
        (df['is_under_structure'] == 1)
        & (bridge_above_hit == 1)
        & (bridge_height_hit == 1)
        & (bridge_score >= max(1, cfg['bridge_min_score']))
    )
    bridge_core = (bridge_core_strong | bridge_core_scored).astype(int)

    window_str = f"{max(0.1, float(cfg['bridge_persistence_window_s']))}s"
    bridge_recent = (
        pd.Series(bridge_core.to_numpy(dtype=float), index=df['time_utc'])
        .rolling(window=window_str, min_periods=1, closed='both')
        .max()
        .fillna(0)
        .astype(int)
    )
    return bridge_core, bridge_recent.to_numpy()




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
    cfg = _load_labelisation_params(params)
    df = _prepare_labelisation_df(df_input)

    if verbose:
        print(f"Labellisation de {len(df)} points...")

    # 1. Initialisation du label par défaut
    df['label'] = 'other'  # Valeur par défaut si aucune condition n'est remplie
    
    # 2. Identification des structures (Ponts / signal_denied)
    
    # Tunnel (signal denied)
    if 'signal_denied' in df.columns:
        df.loc[(df['signal_denied'] == 1), 'label'] = 'signal_denied'
    
    # Marqueur pont instantane puis persistance sur 1 seconde.
    # IMPORTANT: on ne veut "bridge" que pour un passage SOUS ouvrage.
    bridge_core, bridge_recent = _compute_bridge_core_and_recent(df, cfg)
    df['bridge_recent_1s'] = bridge_recent

    # Gare: structure + signatures locales + vitesse faible
    gare_candidate = (
        (df['label'] != 'signal_denied')
        & (df['label'] != 'bridge')
        & (df['bridge_recent_1s'] == 0)
        & (bridge_core == 0)
        & (df['is_under_structure'] == 1)
        & (df['speed_gt_mps_smooth'].abs() <= cfg['seuil_vitesse_gare_mps'])
        & (
            (df['obs_type'].isin([1, 3]))
            | (df['building_density'] > cfg['seuil_building_density'])
            | (df['obstacle_overhead_ratio'] > cfg['seuil_overhead_gare'])
            | (df['density_near_0_5m'] > cfg['seuil_gare_density_near'])
            | (df['zrel_iqr'] > cfg['seuil_gare_zrel_iqr'])
        )
        & ((df['n_points_zone'] >= cfg['seuil_min_points_zone']) | (df['enough_points_flag'] == 1))
    )
    df.loc[gare_candidate, 'label'] = 'gare'

    # Pont: on garde 1 seconde de memoire pour ne pas perdre le passage sous ouvrage
    bridge_mask = (
        (df['label'] != 'signal_denied')
        & (df['label'] != 'gare')
        & (df['bridge_recent_1s'] == 1)
    )
    df.loc[bridge_mask, 'label'] = 'bridge'

    # Regle prioritaire demandee: bridge si non denied et is_bridge == 1.
    bridge_direct_mask = (
        (df['signal_denied'] != 1)
        & (df['is_bridge'] == 1)
    )
    df.loc[bridge_direct_mask, 'label'] = 'bridge'

    # 3. Labellisation par obstruction (si pas déjà sous une structure)
    mask_no_struct = ~df['label'].str.contains('bridge|signal_denied|gare')

    # Tree
    tree_mask = mask_no_struct & (
        ((df['veg_density'] > cfg['seuil_vegetation']) & (df['obs_type'] == 2))
        | (df['vegetation_density_high'] > cfg['seuil_veg_high'])
        | ((df['vegetation_density_mid'] > cfg['seuil_tree_veg_mid']) & (df['zrel_p90'] > cfg['seuil_tree_zrel_p90']))
        | (df['canopee_ratio'] > cfg['seuil_canopee'])
    )
    df.loc[tree_mask, 'label'] = 'tree'
    
    # Build
    build_mask = mask_no_struct & (
        (df['obs_type'] == 1)
        | (df['building_density'] > cfg['seuil_building_density'])
        | (
            (df['density_mid_5_15m'] > cfg['seuil_build_density_mid'])
            & (df['zrel_p95'] > cfg['seuil_build_zrel_p95'])
            & (df['vegetation_density_low'] < 0.25)
        )
    )
    df.loc[build_mask, 'label'] = 'build'
    
    # Open Sky Urban vs Rural
    # On considère Urban si du bâti est détecté dans le voisinage (obs_type 1 ou 3)
    df['sky_mask_smoothed'] = df['sky_mask_deg'].rolling(window=cfg['distance_scan'], center=True).mean()
    df['sky_mask_smoothed'] = df['sky_mask_smoothed'].fillna(df['sky_mask_deg'])
    open_sky_strict = mask_no_struct & (
        (df['sky_mask_smoothed'] < cfg['seuil_ciel_ouvert'])
        & (df['occupation_ciel_azimuth_ratio'] < cfg['seuil_occupation_ciel'])
        & (df['obstacle_overhead_ratio'] < cfg['seuil_overhead_bridge'])
        & (df['building_density'] < cfg['seuil_building_density'])
        & (df['veg_density'] < cfg['seuil_vegetation'])
        & (df['density_near_0_5m'] < cfg['seuil_open_sky_density_near_max'])
        & (df['density_far_15_30m'] < cfg['seuil_open_sky_density_far_max'])
        & (df['zrel_p95'] < cfg['seuil_open_sky_zrel_p95_max'])
        & (df['zrel_std'] < cfg['seuil_open_sky_zrel_std_max'])
    )
    df.loc[open_sky_strict, 'label'] = 'open-sky'

    # Variante plus permissive pour recuperer les open-sky rates faibles.
    open_sky_score = (
        (df['sky_mask_smoothed'] < cfg['seuil_ciel_ouvert']).astype(int)
        + (df['occupation_ciel_azimuth_ratio'] < (cfg['seuil_occupation_ciel'] * 1.4)).astype(int)
        + (df['obstacle_overhead_ratio'] < (cfg['seuil_overhead_bridge'] * 1.2)).astype(int)
        + (df['building_density'] < (cfg['seuil_building_density'] * 1.2)).astype(int)
        + (df['veg_density'] < (cfg['seuil_vegetation'] * 1.2)).astype(int)
        + (df['density_near_0_5m'] < (cfg['seuil_open_sky_density_near_max'] * 1.25)).astype(int)
        + (df['zrel_p95'] < (cfg['seuil_open_sky_zrel_p95_max'] * 1.2)).astype(int)
    )
    open_sky_soft = mask_no_struct & (df['label'] == 'other') & (open_sky_score >= cfg['seuil_open_sky_soft_score'])
    df.loc[open_sky_soft, 'label'] = 'open-sky'

    # Mixed en dernier recours pour eviter de sur-labelliser des zones qui sont
    # plutot urbaines (build) ou ouvertes (open-sky).
    mask_other = (df['label'] == 'other')
    mixed_mask = mask_other & (
        (df['obs_type'] == 3)
        | ((df['veg_density'] > cfg['seuil_melange']) & (df['obs_type'] == 0))
        | (
            (df['building_density'] > cfg['seuil_mixed_building'])
            & (df['veg_density'] > cfg['seuil_mixed_veg'])
            & (df['zrel_iqr'] > cfg['seuil_mixed_zrel_iqr'])
            & (df['zrel_std'] > cfg['seuil_mixed_zrel_std'])
        )
    )
    df.loc[mixed_mask, 'label'] = 'mixed'

    # Fallback final pour eviter trop de 'other'.
    remaining_other = (df['label'] == 'other')
    df.loc[remaining_other & (df['building_density'] > 0.04), 'label'] = 'build'
    df.loc[remaining_other & (df['veg_density'] > 0.25), 'label'] = 'tree'
    df.loc[
        remaining_other
        & (df['label'] == 'other')
        & (df['sky_mask_smoothed'] < (cfg['seuil_ciel_ouvert'] * 1.2))
        & (df['obstacle_overhead_ratio'] < (cfg['seuil_overhead_bridge'] * 1.4)),
        'label'
    ] = 'open-sky'
    df.loc[df['label'] == 'other', 'label'] = 'mixed'
    
    
    # Reprise gare tardive si un point n'a pas ete capture par la passe amont.
    gare_mask_fallback = (
        (df['label'] == 'other')
        & gare_candidate
    )
    df.loc[gare_mask_fallback, 'label'] = 'gare'

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
    features_file = config.get("lidar_features_csv")
    