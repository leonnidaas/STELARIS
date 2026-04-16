"""Labellisation OSM de l'environnement GNSS.

Ce module lit les features OSM extraites en amont, applique des regles
de classification, puis sauvegarde un CSV final dans les chemins de utils.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import get_traj_paths


DEFAULT_OSM_LABEL_PARAMS = {
	"tunnel_count_min": 1,
	"bridge_count_min": 1,
	"gare_rail_count_min": 2,
	"gare_station_count_min": 1,
	"building_count_min": 4,
	"green_count_min": 3,
	"open_sky_road_count_max": 1,
	"open_sky_building_count_max": 0,
	"open_sky_green_count_max": 0,
	"smooth_window": 5,
}


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()

	numeric_cols_defaults = {
		"osm_road_count_30m": 0,
		"osm_rail_count_30m": 0,
		"osm_bridge_count_30m": 0,
		"osm_tunnel_count_30m": 0,
		"osm_building_count_30m": 0,
		"osm_green_count_30m": 0,
		"osm_water_count_30m": 0,
		"osm_station_count_30m": 0,
	}

	for col, default in numeric_cols_defaults.items():
		if col not in out.columns:
			out[col] = default
		out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)

	if "time_utc" in out.columns:
		out["time_utc"] = pd.to_datetime(out["time_utc"], errors="coerce")
		out = out.sort_values("time_utc").reset_index(drop=True)

	return out


def _label_row(row: pd.Series, p: dict) -> str:
	if row["osm_tunnel_count_30m"] >= p["tunnel_count_min"]:
		return "signal_denied"
	if row["osm_bridge_count_30m"] >= p["bridge_count_min"]:
		return "bridge"
	if (
		row["osm_rail_count_30m"] >= p["gare_rail_count_min"]
		and row["osm_station_count_30m"] >= p["gare_station_count_min"]
	):
		return "gare"
	if row["osm_building_count_30m"] >= p["building_count_min"]:
		return "build"
	if row["osm_green_count_30m"] >= p["green_count_min"]:
		return "tree"
	if (
		row["osm_road_count_30m"] <= p["open_sky_road_count_max"]
		and row["osm_building_count_30m"] <= p["open_sky_building_count_max"]
		and row["osm_green_count_30m"] <= p["open_sky_green_count_max"]
	):
		return "open-sky"
	return "other"


def _smooth_labels_majority_safe(labels: pd.Series, window: int) -> pd.Series:
	w = max(int(window), 1)
	if w <= 1 or labels.empty:
		return labels.copy()

	smoothed = []
	vals = labels.astype(str).tolist()
	n = len(vals)
	half = w // 2
	for i in range(n):
		a = max(0, i - half)
		b = min(n, i + half + 1)
		chunk = pd.Series(vals[a:b])
		smoothed.append(chunk.value_counts().index[0])
	return pd.Series(smoothed, index=labels.index)


def auto_label_environment_osm(
	df_input: pd.DataFrame,
	params: dict | None = None,
	output_csv_final: str | Path | None = None,
	output_csv_interim: str | Path | None = None,
	verbose: bool = True,
) -> pd.DataFrame:
	p = dict(DEFAULT_OSM_LABEL_PARAMS)
	if isinstance(params, dict):
		p.update(params)

	df = _normalize_df(df_input)
	df["label"] = df.apply(lambda row: _label_row(row, p), axis=1)
	df["label"] = _smooth_labels_majority_safe(df["label"], p.get("smooth_window", 5))

	lat_col = "latitude_gt" if "latitude_gt" in df.columns else ("latitude" if "latitude" in df.columns else None)
	lon_col = "longitude_gt" if "longitude_gt" in df.columns else ("longitude" if "longitude" in df.columns else None)

	cols = [c for c in ["time_utc", lat_col, lon_col, "label"] if c is not None and c in df.columns]
	if "label" not in cols:
		cols.append("label")
	df_result = df[cols].copy()

	if verbose:
		counts = df_result["label"].value_counts()
		print("=== Resultat labelisation OSM ===")
		for k, v in counts.items():
			pct = 100.0 * float(v) / max(len(df_result), 1)
			print(f"  {k:15s}: {int(v):6d} points ({pct:5.1f}%)")

	if output_csv_final:
		out_final = Path(output_csv_final)
		out_final.parent.mkdir(parents=True, exist_ok=True)
		df_result.to_csv(out_final, index=False)
		if verbose:
			print(f"CSV final sauvegarde: {out_final}")

	if output_csv_interim:
		out_interim = Path(output_csv_interim)
		out_interim.parent.mkdir(parents=True, exist_ok=True)
		df.to_csv(out_interim, index=False)
		if verbose:
			print(f"CSV interim sauvegarde: {out_interim}")

	return df_result


def process_labelling_osm(
	input_csv: str | Path,
	params: dict | None = None,
	output_csv_final: str | Path | None = None,
	output_csv_interim: str | Path | None = None,
	verbose: bool = True,
) -> pd.DataFrame:
	in_path = Path(input_csv)
	if not in_path.exists():
		raise FileNotFoundError(f"Le fichier d'entree n'existe pas: {in_path}")

	if verbose:
		print(f"Chargement features OSM: {in_path}")
	df_input = pd.read_csv(in_path)
	if df_input.empty:
		raise ValueError(f"Le fichier {in_path} est vide.")

	if output_csv_final is None:
		output_csv_final = in_path.with_name(f"{in_path.stem}_labeled.csv")

	return auto_label_environment_osm(
		df_input=df_input,
		params=params,
		output_csv_final=output_csv_final,
		output_csv_interim=output_csv_interim,
		verbose=verbose,
	)


def process_labelling_osm_for_traj(
	traj_id: str,
	params: dict | None = None,
	input_csv: str | Path | None = None,
	output_csv_final: str | Path | None = None,
	output_csv_interim: str | Path | None = None,
	verbose: bool = True,
) -> pd.DataFrame:
	cfg = get_traj_paths(traj_id)
	in_csv = Path(input_csv) if input_csv is not None else Path(cfg["osm_features_csv"])
	out_final = Path(output_csv_final) if output_csv_final is not None else Path(cfg["osm_labels_csv"])
	out_interim = Path(output_csv_interim) if output_csv_interim is not None else in_csv.with_name(
		f"{in_csv.stem}_plus_labels.csv"
	)

	return process_labelling_osm(
		input_csv=in_csv,
		params=params,
		output_csv_final=out_final,
		output_csv_interim=out_interim,
		verbose=verbose,
	)


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Labellisation OSM depuis features_osm")
	parser.add_argument("--traj", help="ID trajet (utilise les chemins get_traj_paths)")
	parser.add_argument("-i", "--input", help="CSV features OSM en entree")
	parser.add_argument("-o", "--output", help="CSV final de sortie")
	parser.add_argument("--interim", help="CSV intermediaire (features + label)")
	parser.add_argument("--smooth-window", type=int, default=5, help="Fenetre de lissage majoritaire")
	parser.add_argument("-q", "--quiet", action="store_true", help="Mode silencieux")
	return parser


def main() -> int:
	args = _build_parser().parse_args()
	params = {"smooth_window": int(args.smooth_window)}

	try:
		if args.traj:
			process_labelling_osm_for_traj(
				traj_id=args.traj,
				params=params,
				input_csv=args.input,
				output_csv_final=args.output,
				output_csv_interim=args.interim,
				verbose=not args.quiet,
			)
		else:
			if not args.input:
				raise ValueError("Sans --traj, l'argument --input est requis.")
			process_labelling_osm(
				input_csv=args.input,
				params=params,
				output_csv_final=args.output,
				output_csv_interim=args.interim,
				verbose=not args.quiet,
			)
		return 0
	except Exception as e:
		print(f"Erreur labelisation OSM: {e}")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
