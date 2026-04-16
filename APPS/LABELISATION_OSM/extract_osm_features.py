"""Extraction de features OSM depuis un fichier .osm.pbf pour la labellisation.

Le script:
1) lit la trajectoire (fusion GT/GNSS par defaut),
2) lit des couches OSM pertinentes depuis un PBF,
3) calcule des indicateurs geospatiaux autour de chaque point,
4) sauvegarde le resultat dans le chemin centralise de utils.get_traj_paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from utils import FILTERED_OSM_PBF, OSM_PBF, get_traj_paths, standardize_dataframe


DEFAULT_RADIUS_M = 30.0


def _choose_latest_pbf() -> Path:
	"""Retourne le .osm.pbf le plus recent dans FILTERED_OSM_PBF puis OSM_PBF."""
	candidates: list[Path] = []

	for base in [Path(FILTERED_OSM_PBF), Path(OSM_PBF)]:
		if base.exists():
			candidates.extend(base.glob("*.osm.pbf"))

	if not candidates:
		raise FileNotFoundError(
			f"Aucun fichier .osm.pbf trouve dans {FILTERED_OSM_PBF} ni {OSM_PBF}."
		)

	return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_osm_layer(pbf_path: Path, layer: str) -> gpd.GeoDataFrame:
	"""Lit une couche OSM de maniere robuste, retourne un GeoDataFrame vide si absent."""
	try:
		gdf = gpd.read_file(pbf_path, layer=layer)
		if gdf is None or gdf.empty:
			return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
		if gdf.crs is None:
			gdf = gdf.set_crs("EPSG:4326")
		return gdf
	except Exception:
		return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def _valid_tag_mask(series: pd.Series | None, index: pd.Index) -> pd.Series:
	"""Retourne un masque boolean indiquant les lignes avec une valeur de tag OSM valide."""
	if series is None:
		return pd.Series(False, index=index)
	s = series.astype(str).str.strip()
	return s.notna() & (s != "") & (s.str.lower() != "nan")


def _build_osm_subsets(
	lines_wm: gpd.GeoDataFrame,
	polys_wm: gpd.GeoDataFrame,
	points_wm: gpd.GeoDataFrame,
) -> dict[str, gpd.GeoDataFrame]:
	"""Construit les sous-ensembles OSM utiles pour les features."""

	def _select(gdf: gpd.GeoDataFrame, mask: pd.Series | np.ndarray) -> gpd.GeoDataFrame:
		"""Retourne les lignes de gdf correspondant au masque, ou un GeoDataFrame vide si aucun match."""
		if gdf.empty:
			return gdf
		out = gdf.loc[mask].copy()
		return out if not out.empty else gdf.iloc[0:0].copy()

	road_mask = _valid_tag_mask(lines_wm.get("highway"), lines_wm.index)
	rail_mask = _valid_tag_mask(lines_wm.get("railway"), lines_wm.index)
	bridge_mask = _valid_tag_mask(lines_wm.get("bridge"), lines_wm.index)
	tunnel_mask = _valid_tag_mask(lines_wm.get("tunnel"), lines_wm.index)

	building_mask = _valid_tag_mask(polys_wm.get("building"), polys_wm.index)
	natural_col = polys_wm.get("natural")
	landuse_col = polys_wm.get("landuse")
	water_mask = pd.Series(False, index=polys_wm.index)
	green_mask = pd.Series(False, index=polys_wm.index)

	if natural_col is not None:
		n = natural_col.astype(str).str.lower()
		water_mask = water_mask | n.isin(["water", "wetland", "bay"])
		green_mask = green_mask | n.isin(["wood", "tree", "scrub", "grassland", "heath"])

	if landuse_col is not None:
		l = landuse_col.astype(str).str.lower()
		water_mask = water_mask | l.isin(["reservoir", "basin"])
		green_mask = green_mask | l.isin(["forest", "grass", "meadow", "farmland", "orchard"])

	station_mask = pd.Series(False, index=points_wm.index)
	if not points_wm.empty:
		railway_col = points_wm.get("railway")
		pt_col = points_wm.get("public_transport")
		amenity_col = points_wm.get("amenity")
		if railway_col is not None:
			station_mask = station_mask | railway_col.astype(str).str.lower().isin(["station", "halt", "tram_stop"])
		if pt_col is not None:
			station_mask = station_mask | pt_col.astype(str).str.lower().isin(["station", "stop_position", "platform"])
		if amenity_col is not None:
			station_mask = station_mask | amenity_col.astype(str).str.lower().isin(["bus_station", "ferry_terminal"])

	return {
		"roads": _select(lines_wm, road_mask),
		"rails": _select(lines_wm, rail_mask),
		"bridges": _select(lines_wm, bridge_mask),
		"tunnels": _select(lines_wm, tunnel_mask),
		"buildings": _select(polys_wm, building_mask),
		"greens": _select(polys_wm, green_mask),
		"waters": _select(polys_wm, water_mask),
		"stations": _select(points_wm, station_mask),
	}


def _counts_within_radius(points_wm: gpd.GeoDataFrame, features_wm: gpd.GeoDataFrame, radius_m: float) -> np.ndarray:
	if points_wm.empty or features_wm.empty:
		return np.zeros(len(points_wm), dtype=int)

	buffers = points_wm[["geometry"]].copy()
	buffers["geometry"] = buffers.geometry.buffer(radius_m)
	buffers = gpd.GeoDataFrame(buffers, geometry="geometry", crs=points_wm.crs)

	joined = gpd.sjoin(
		buffers,
		features_wm[["geometry"]],
		how="left",
		predicate="intersects",
	)
	counts = joined.groupby(joined.index).size()
	out = np.zeros(len(points_wm), dtype=int)
	out[counts.index.to_numpy()] = counts.to_numpy(dtype=int)

	# Les lignes sans match apparaissent quand meme avec index_right = NaN.
	if "index_right" in joined.columns:
		nan_only = joined["index_right"].isna()
		if nan_only.any():
			nan_idx = joined.index[nan_only]
			out[nan_idx.to_numpy()] = 0

	return out


def _nearest_distance(points_wm: gpd.GeoDataFrame, features_wm: gpd.GeoDataFrame) -> np.ndarray:
	if points_wm.empty:
		return np.array([], dtype=float)
	if features_wm.empty:
		return np.full(len(points_wm), np.nan, dtype=float)

	# Methode simple et robuste compatible geopandas>=0.10.
	result = np.full(len(points_wm), np.nan, dtype=float)
	for idx, geom in enumerate(points_wm.geometry):
		result[idx] = float(features_wm.distance(geom).min())
	return result


def _environment_rule(row: pd.Series) -> str:
	"""Regle simple de pre-labellisation OSM."""
	if row["osm_tunnel_count_30m"] > 0:
		return "tunnel"
	if row["osm_bridge_count_30m"] > 0:
		return "bridge"
	if row["osm_rail_count_30m"] >= 2 and row["osm_station_count_30m"] > 0:
		return "gare"
	if row["osm_building_count_30m"] >= 4:
		return "build"
	if row["osm_green_count_30m"] >= 3:
		return "tree"
	if row["osm_road_count_30m"] <= 1 and row["osm_building_count_30m"] == 0 and row["osm_green_count_30m"] == 0:
		return "open-sky"
	return "other"


def extract_osm_features_for_traj(
	traj_id: str,
	pbf_path: Path | str | None = None,
	input_csv: Path | str | None = None,
	output_csv: Path | str | None = None,
	radius_m: float = DEFAULT_RADIUS_M,
	verbose: bool = True,
) -> Path:
	"""Extrait des features OSM pour un trajet et les sauvegarde en CSV."""
	cfg = get_traj_paths(traj_id)

	pbf = Path(pbf_path) if pbf_path is not None else _choose_latest_pbf()
	if not pbf.exists():
		raise FileNotFoundError(f"PBF introuvable: {pbf}")

	in_csv = Path(input_csv) if input_csv is not None else Path(cfg["sync_csv"])
	if not in_csv.exists():
		raise FileNotFoundError(
			f"Fichier d'entree introuvable: {in_csv}. Lancez d'abord la fusion GT/GNSS."
		)

	out_csv = Path(output_csv) if output_csv is not None else Path(cfg["osm_features_csv"])
	out_csv.parent.mkdir(parents=True, exist_ok=True)

	if verbose:
		print(f"--- Chargement trajectoire: {in_csv}")
	df = pd.read_csv(in_csv)
	df = standardize_dataframe(df)

	required_any = {"latitude_gt", "latitude"}
	if not (required_any & set(df.columns)):
		raise ValueError("Colonne latitude_gt/latitude manquante dans le CSV d'entree.")
	required_any = {"longitude_gt", "longitude"}
	if not (required_any & set(df.columns)):
		raise ValueError("Colonne longitude_gt/longitude manquante dans le CSV d'entree.")

	lat_col = "latitude_gt" if "latitude_gt" in df.columns else "latitude"
	lon_col = "longitude_gt" if "longitude_gt" in df.columns else "longitude"

	traj_wgs84 = gpd.GeoDataFrame(
		df.copy(),
		geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
		crs="EPSG:4326",
	)

	utm_crs = traj_wgs84.estimate_utm_crs()
	traj_wm = traj_wgs84.to_crs(utm_crs)

	if verbose:
		print(f"--- Chargement PBF: {pbf}")
	lines = _read_osm_layer(pbf, "lines").to_crs(utm_crs)
	multipolygons = _read_osm_layer(pbf, "multipolygons").to_crs(utm_crs)
	points = _read_osm_layer(pbf, "points").to_crs(utm_crs)

	subsets = _build_osm_subsets(lines, multipolygons, points)

	if verbose:
		print("--- Calcul des features de proximite OSM")
	features = pd.DataFrame(index=traj_wm.index)
	features["osm_road_count_30m"] = _counts_within_radius(traj_wm, subsets["roads"], radius_m)
	features["osm_rail_count_30m"] = _counts_within_radius(traj_wm, subsets["rails"], radius_m)
	features["osm_bridge_count_30m"] = _counts_within_radius(traj_wm, subsets["bridges"], radius_m)
	features["osm_tunnel_count_30m"] = _counts_within_radius(traj_wm, subsets["tunnels"], radius_m)
	features["osm_building_count_30m"] = _counts_within_radius(traj_wm, subsets["buildings"], radius_m)
	features["osm_green_count_30m"] = _counts_within_radius(traj_wm, subsets["greens"], radius_m)
	features["osm_water_count_30m"] = _counts_within_radius(traj_wm, subsets["waters"], radius_m)
	features["osm_station_count_30m"] = _counts_within_radius(traj_wm, subsets["stations"], radius_m)

	features["osm_dist_road_m"] = _nearest_distance(traj_wm, subsets["roads"])
	features["osm_dist_rail_m"] = _nearest_distance(traj_wm, subsets["rails"])
	features["osm_dist_building_m"] = _nearest_distance(traj_wm, subsets["buildings"])
	features["osm_dist_station_m"] = _nearest_distance(traj_wm, subsets["stations"])

	features = features.fillna(-1.0)
	features["osm_prelabel"] = features.apply(_environment_rule, axis=1)

	keep_cols = [c for c in ["time_utc", lat_col, lon_col] if c in df.columns]
	out_df = pd.concat([df[keep_cols].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
	out_df.to_csv(out_csv, index=False)

	if verbose:
		print(f"--- Features OSM sauvegardees: {out_csv} ({len(out_df)} lignes)")
	return out_csv


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Extraction des features OSM depuis un PBF")
	parser.add_argument("--traj", required=True, help="Identifiant du trajet")
	parser.add_argument("--pbf", help="Chemin explicite vers un fichier .osm.pbf")
	parser.add_argument("--input-csv", help="CSV d'entree (defaut: sync_csv du trajet)")
	parser.add_argument("--output-csv", help="CSV de sortie (defaut: osm_features_csv du trajet)")
	parser.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M, help="Rayon de recherche en metres")
	parser.add_argument("--quiet", action="store_true", help="Desactive les logs")
	return parser


def main() -> int:
	parser = _build_parser()
	args = parser.parse_args()

	try:
		extract_osm_features_for_traj(
			traj_id=args.traj,
			pbf_path=args.pbf,
			input_csv=args.input_csv,
			output_csv=args.output_csv,
			radius_m=float(args.radius_m),
			verbose=not args.quiet,
		)
		return 0
	except Exception as e:
		print(f"Erreur extraction OSM: {e}")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
