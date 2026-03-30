from pathlib import Path
from math import ceil
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from owslib.wfs import WebFeatureService
from utils import get_traj_paths, standardize_dataframe, lat_col, long_col


def _build_fast_traj_buffer(gdf_trajet_l93: gpd.GeoDataFrame, buffer_m: float = 500.0, max_points: int = 5000, simplify_tol_m: float = 2.0):
    """Construit un buffer robuste et rapide pour les trajets très longs."""
    coords = [(geom.x, geom.y) for geom in gdf_trajet_l93.geometry if geom is not None and not geom.is_empty]
    if not coords:
        return None

    # Décimation adaptative : limite le nombre de sommets traités par le buffer
    if max_points and len(coords) > max_points:
        step = ceil(len(coords) / max_points)
        coords = coords[::step]
        if coords[-1] != (gdf_trajet_l93.geometry.iloc[-1].x, gdf_trajet_l93.geometry.iloc[-1].y):
            coords.append((gdf_trajet_l93.geometry.iloc[-1].x, gdf_trajet_l93.geometry.iloc[-1].y))

    # Suppression des doublons consécutifs (accélère la géométrie finale)
    cleaned_coords = [coords[0]]
    for coord in coords[1:]:
        if coord != cleaned_coords[-1]:
            cleaned_coords.append(coord)

    if len(cleaned_coords) >= 2:
        trajet_line = LineString(cleaned_coords)
        if simplify_tol_m and simplify_tol_m > 0:
            trajet_line = trajet_line.simplify(simplify_tol_m)
        return trajet_line.buffer(buffer_m)

    return gdf_trajet_l93.geometry.iloc[[0]].buffer(buffer_m).iloc[0]

def download_tile_url_list(traj_id : str, gt_file : Path, output_file : Path, wfs_url : str, target_layer : str, version : str = '2.0.0', verbose : bool = True ):
    """
    Télécharge la liste des URLs des tuiles LiDAR pour un trajet donné.
    Exemple traj_id : "SC01" ou "Bordeaux_Coutras"
    """

    if verbose:
        print(f"--- Pipeline LiDAR pour : {traj_id} ---")

    # 2. Connexion WFS
    wfs = WebFeatureService(url=wfs_url, version=version)
    # 3. Chargement et Trajet
    df_gt = pd.read_csv(gt_file)
    df_gt = standardize_dataframe(df_gt) # On vérifie si les colonnes sont 'longitude' ou 'longitude_gt' selon les gt 
    
    gdf_trajet = gpd.GeoDataFrame(
        df_gt, 
        geometry=gpd.points_from_xy(df_gt[long_col], df_gt[lat_col]), 
        crs="EPSG:4326"
    )

    # 4. BBOX pour la requête (WGS84)
    bbox = (df_gt[long_col].min(), df_gt[lat_col].min(), 
            df_gt[long_col].max(), df_gt[lat_col].max()) 

    # 5. Requête GetFeature
    response = wfs.getfeature(
        typename=target_layer,
        bbox=bbox,
        srsname='urn:ogc:def:crs:EPSG::4326',
        outputFormat='application/json'
    )

    # 6. Traitement spatial
    gdf_dalles = gpd.read_file(response)
    
    # Conversion en Lambert-93 pour le calcul en mètres
    gdf_trajet_l93 = gdf_trajet.to_crs("EPSG:2154")
    gdf_dalles_l93 = gdf_dalles.to_crs("EPSG:2154")

    # 7. Buffer de 500m autour du trajet complet (optimisé pour longs trajets)
    trajet_buffer = _build_fast_traj_buffer(
        gdf_trajet_l93,
        buffer_m=500.0,
        max_points=5000,
        simplify_tol_m=2.0,
    )

    if trajet_buffer is None:
        raise ValueError("Aucun point valide dans le trajet pour construire le buffer.")

    # 8. Intersection
    # Préfiltrage via index spatial puis test exact d'intersection
    try:
        candidate_idx = gdf_dalles_l93.sindex.query(trajet_buffer, predicate="intersects")
        dalles_candidates = gdf_dalles_l93.iloc[candidate_idx]
    except Exception:
        dalles_candidates = gdf_dalles_l93

    dalles_selectionnees = dalles_candidates[dalles_candidates.intersects(trajet_buffer)]
    
    # 9. Sauvegarde
    urls = dalles_selectionnees['url'].unique().tolist() if 'url' in dalles_selectionnees.columns else []
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for url in urls:
            f.write(url + "\n")

    if verbose:
        print(f"Terminé : {len(urls)} dalles identifiées pour {traj_id}.")
        print(f"Liste enregistrée dans : {output_file}")

if __name__ == "__main__":
    from utils import WFS_URL, LIDAR_LAYER, WFS_VERSION
    config = get_traj_paths("BORDEAUX_COUTRAS")
    download_tile_url_list("BORDEAUX_COUTRAS" , config["raw_gt"], config["lidar_tiles"] / "urls.txt", WFS_URL, LIDAR_LAYER, WFS_VERSION)