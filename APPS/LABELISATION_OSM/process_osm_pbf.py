import json
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, mapping


def filter_railway_data(input_pbf, output_pbf):
    print("--- Filtrage des données ferroviaires en cours...")

    cmd = [
        "osmium", "tags-filter", 
        input_pbf, 
        "wr/railway=rail", "wr/building", "wr/natural=wood", 
        "-o", output_pbf, "--overwrite"
    ]
    subprocess.run(cmd, check=True)
    print(f"--- Fichier filtré créé : {output_pbf}")

def filter_region(input_pbf, output_pbf, bbox):
    print("--- Filtrage de la région d'intérêt en cours...")

    cmd = [
        "osmium", "extract", 
        input_pbf, 
        "-b", ",".join(map(str, bbox)), 
        "-o", output_pbf, "--overwrite"
    ]
    subprocess.run(cmd, check=True)
    print(f"--- Fichier extrait créé : {output_pbf}")


def filter_region_with_shape(input_pbf, output_pbf, shape_geojson_path):
    """Filtre le PBF avec une shape (GeoJSON) au lieu d'une bbox."""
    print("--- Filtrage de la région d'intérêt (shape) en cours...")

    cmd = [
        "osmium",
        "extract",
        "-p",
        str(shape_geojson_path),
        str(input_pbf),
        "-o",
        str(output_pbf),
        "--overwrite",
    ]
    subprocess.run(cmd, check=True)
    print(f"--- Fichier extrait créé : {output_pbf}")

def filter_data_and_region(input_pbf, output_pbf, bbox):
    temp_pbf = input_pbf.parent / f"temp_filtered_{input_pbf.name}"
    filter_railway_data(input_pbf, temp_pbf)
    filter_region(temp_pbf, output_pbf, bbox)
    temp_pbf.unlink()  # Supprimer le fichier temporaire
    print(f"--- Filtrage complet terminé : {output_pbf}")


def filter_data_and_shape(input_pbf, output_pbf, shape_geojson_path):
    """Filtre tags puis applique un filtre spatial sur shape polygonale."""
    temp_pbf = Path(input_pbf).parent / f"temp_filtered_{Path(input_pbf).name}"
    try:
        filter_railway_data(input_pbf, temp_pbf)
        filter_region_with_shape(temp_pbf, output_pbf, shape_geojson_path)
        print(f"--- Filtrage complet terminé : {output_pbf}")
    finally:
        if temp_pbf.exists():
            temp_pbf.unlink()


def _pick_lon_lat_columns(df):
    lon_candidates = ["longitude", "longitude_gt", "lon", "x"]
    lat_candidates = ["latitude", "latitude_gt", "lat", "y"]

    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    if lon_col is None or lat_col is None:
        raise ValueError(
            "Le CSV doit contenir des colonnes de coordonnees, ex: longitude/latitude ou longitude_gt/latitude_gt."
        )
    return lon_col, lat_col


def build_buffered_shape_from_csv(input_csv, buffer_m=500.0):
    """Construit un polygone buffer autour de la trajectoire depuis un CSV.

    Retourne un dict GeoJSON (FeatureCollection) contenant la shape en EPSG:4326.
    """
    df = pd.read_csv(input_csv)
    lon_col, lat_col = _pick_lon_lat_columns(df)

    traj = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    ).dropna(subset=[lon_col, lat_col])

    if traj.empty:
        raise ValueError("Aucun point valide dans le CSV pour construire la shape.")

    utm_crs = traj.estimate_utm_crs()
    traj_utm = traj.to_crs(utm_crs)

    # Shape qui suit la trajectoire: LineString ordonnee puis buffer metrique.
    coords = list(zip(traj_utm.geometry.x, traj_utm.geometry.y))
    if len(coords) < 2:
        raise ValueError("La trajectoire doit contenir au moins 2 points pour construire une shape.")

    line = LineString(coords)

    corridor = line.buffer(float(buffer_m))
    corridor_wgs84 = gpd.GeoSeries([corridor], crs=utm_crs).to_crs("EPSG:4326").iloc[0]

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "trajectory_buffer"},
                "geometry": mapping(corridor_wgs84),
            }
        ],
    }


def filter_data_with_csv_trajectory_shape(input_pbf, output_pbf, input_csv, buffer_m=500.0):
    """Filtre un PBF en utilisant une shape bufferisée autour de la trajectoire CSV."""
    shape = build_buffered_shape_from_csv(input_csv=input_csv, buffer_m=buffer_m)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(shape, tmp)

    try:
        filter_data_and_shape(input_pbf=input_pbf, output_pbf=output_pbf, shape_geojson_path=tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()



# Usage
# filter_railway_data("data/france-latest.osm.pbf", "data/france-rails-env.osm.pbf")
# bbox = [lon_min, lat_min, lon_max, lat_max]
# filter_region("data/france-rails-env.osm.pbf", "data/france-rails-env-bordeaux.osm.pbf", bbox=[-0.8, 44.5, 0.1, 45.1])