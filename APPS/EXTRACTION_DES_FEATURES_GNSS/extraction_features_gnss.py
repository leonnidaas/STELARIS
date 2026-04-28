import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


FORMAT = "%(asctime)s %(module)s %(levelname)s %(message)s"
DATEFORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


STATS = ["mean", "std"]
STATS_FUNCTIONS = {
	"mean": np.nanmean,
	"std": np.nanstd,
}


def _parse_datetime_series(series: pd.Series) -> pd.Series:
	"""Parse several timestamp formats used in project CSV files."""
	try:
		dt = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
	except TypeError:
		dt = pd.to_datetime(series, utc=True, errors="coerce")
	if dt.notna().all():
		return dt

	dt_alt = pd.to_datetime(series, utc=True, errors="coerce", format="%Y%m%d-%H%M%S.%f")
	return dt.fillna(dt_alt)


def _resolve_wls_columns(df_wls: pd.DataFrame) -> pd.DataFrame:
	"""Normalize WLS column names so downstream feature creation is stable."""
	rename_map = {}

	if "lon_rx_wls_deg" in df_wls.columns:
		rename_map["lon_rx_wls_deg"] = "longitude_wls"
	elif "longitude" in df_wls.columns:
		rename_map["longitude"] = "longitude_wls"

	if "lat_rx_wls_deg" in df_wls.columns:
		rename_map["lat_rx_wls_deg"] = "latitude_wls"
	elif "latitude" in df_wls.columns:
		rename_map["latitude"] = "latitude_wls"

	if "alt_rx_wls_m" in df_wls.columns:
		rename_map["alt_rx_wls_m"] = "altitude_wls"
	elif "altitude" in df_wls.columns:
		rename_map["altitude"] = "altitude_wls"

	if "time_utc" in df_wls.columns:
		rename_map["time_utc"] = "time_utc_wls"
	if "time" in df_wls.columns and "time_utc_wls" not in df_wls.columns:
		rename_map["time"] = "time_utc_wls"

	return df_wls.rename(columns=rename_map)


def detector_cycle_slip(phi_a: np.ndarray, phi_b: np.ndarray, thresh_lambda_a_b: float, n_win: int) -> np.ndarray:
	"""Detect cycle slips from the geometry-free combination on a rolling polynomial fit."""
	phi_diff = phi_a - phi_b
	n = len(phi_diff)
	indic = np.zeros(n, dtype=bool)

	x_win = np.arange(n_win)
	count_wait = 0
	for i in range(n):
		phi_diff_win = phi_diff[i - n_win:i]
		if len(phi_diff_win) == n_win and count_wait == 0:
			p = np.polynomial.Polynomial.fit(x_win, phi_diff_win, deg=2)
			error = np.abs(p(n_win) - phi_diff[i])
			if error > thresh_lambda_a_b:
				indic[i] = True
				count_wait = n_win
		elif count_wait > 0:
			count_wait -= 1

	return indic


def remove_mean(y: np.ndarray, indic: np.ndarray) -> np.ndarray:
	"""Remove segment-wise mean between cycle-slip boundaries."""
	z = np.copy(y)

	if np.sum(indic) > 0:
		limits = np.where(indic)[0]

		i_inf = limits[0]
		z[:i_inf] -= np.mean(y[:i_inf])

		if len(limits) > 1:
			for i in range(1, len(limits)):
				i_inf = limits[i - 1]
				i_sup = limits[i]
				z[i_inf:i_sup] -= np.mean(y[i_inf:i_sup])
		else:
			i_sup = i_inf

		z[i_sup:] -= np.mean(y[i_sup:])
	else:
		z -= np.mean(y)

	return z


def create_multipath_data(df_sv: pd.DataFrame) -> pd.DataFrame:
	"""Compute CMC indicators from raw pseudorange/carrier phase measurements."""
	clight = 299792458.0

	dict_freq = {
		"l1": 1575.42e6,
		"l2": 1227.60e6,
		"l5": 1176.45e6,
		"e1": 1575.42e6,
		"e5a": 1176.450e6,
		"e5b": 1207.140e6,
	}

	d = {id_sat: {} for id_sat in set(df_sv["gnss_sv_id"]) }
	freq_scale = {signal: clight / freq for signal, freq in dict_freq.items()}
	f_l1_sq = dict_freq["l1"] ** 2
	f_l2_sq = dict_freq["l2"] ** 2
	f_e1_sq = dict_freq["e1"] ** 2
	f_e5a_sq = dict_freq["e5a"] ** 2
	delta_l1_l2 = f_l1_sq - f_l2_sq
	delta_e1_e5a = f_e1_sq - f_e5a_sq

	logger.info("Step CMC 1/3: read data and store measurements")
	for row in df_sv.itertuples(index=False):
		t = row.gps_millis
		signal_type = row.signal_type
		id_sat = row.gnss_sv_id

		if signal_type not in dict_freq:
			continue

		if t not in d[id_sat]:
			d[id_sat][t] = {}

		rho = row.raw_pr_m
		phi = row.carrier_phase
		if np.isnan(rho) or np.isnan(phi):
			continue

		d[id_sat][t][f"{signal_type}-rho"] = rho
		d[id_sat][t][f"{signal_type}-phi"] = phi * freq_scale[signal_type]

	logger.info("Step CMC 2/3: compute CMC at each time")
	for id_sat, d_sat in d.items():
		for t, sat_t in d_sat.items():
			if "l1-rho" in sat_t and "l2-rho" in sat_t:
				phi_diff = sat_t["l1-phi"] - sat_t["l2-phi"]
				i_1 = f_l2_sq / delta_l1_l2 * phi_diff
				i_2 = f_l1_sq / delta_l1_l2 * phi_diff
				sat_t["CMC_1"] = sat_t["l1-rho"] - sat_t["l1-phi"] - 2 * i_1
				sat_t["CMC_2"] = sat_t["l2-rho"] - sat_t["l2-phi"] - 2 * i_2

			if "e1-rho" in sat_t and "e5a-rho" in sat_t:
				phi_diff = sat_t["e1-phi"] - sat_t["e5a-phi"]
				i_1 = f_e5a_sq / delta_e1_e5a * phi_diff
				i_5 = f_e1_sq / delta_e1_e5a * phi_diff
				sat_t["CMC_1"] = sat_t["e1-rho"] - sat_t["e1-phi"] - 2 * i_1
				sat_t["CMC_5"] = sat_t["e5a-rho"] - sat_t["e5a-phi"] - 2 * i_5

	logger.info("Step CMC 3/3: filter CMC estimates for each satellite")
	df_cmc = []

	for id_sat, d_sat in d.items():

		if id_sat.startswith("G"):
			x = np.array(
				[
					[t, d_sat[t]["l1-phi"], d_sat[t]["l2-phi"], d_sat[t]["CMC_1"], d_sat[t]["CMC_2"]]
					for t in d_sat.keys()
					if ("l1-rho" in d_sat[t]) and ("l2-rho" in d_sat[t])
				]
			)
			thresh = clight / dict_freq["l2"] - clight / dict_freq["l1"]
			gnss_id = "gps"
		elif id_sat.startswith("E"):
			x = np.array(
				[
					[t, d_sat[t]["e1-phi"], d_sat[t]["e5a-phi"], d_sat[t]["CMC_1"], d_sat[t]["CMC_5"]]
					for t in d_sat.keys()
					if ("e1-rho" in d_sat[t]) and ("e5a-rho" in d_sat[t])
				]
			)
			thresh = clight / dict_freq["e5a"] - clight / dict_freq["e1"]
			gnss_id = "galileo"
		else:
			x = np.array([])
			thresh = 0.0
			gnss_id = "unknown"

		if len(x) == 0:
			continue

		y1 = x[:, 3]
		y2 = x[:, 4]

		indic = detector_cycle_slip(x[:, 1], x[:, 2], thresh, n_win=10)
		zm1 = remove_mean(y1, indic)
		zm2 = remove_mean(y2, indic)

		t = x[:, 0]
		df_cmc_local = pd.DataFrame({"gps_millis": t, "CMC_a": zm1, "CMC_b": zm2, "gnss_id": gnss_id})
		df_cmc.append(df_cmc_local)

	if not df_cmc:
		return pd.DataFrame(columns=["gps_millis", "CMC_a", "CMC_b", "gnss_id"])

	return pd.concat(df_cmc, ignore_index=True)


def _select_quartile_values(data: np.ndarray, quartile: int) -> np.ndarray:
	"""Return CN0 values that belong to the selected quartile bucket (1..4)."""
	if len(data) == 0:
		return np.array([])

	clean = data[~np.isnan(data)]
	if len(clean) == 0:
		return np.array([])

	q_idx = int(np.clip(quartile, 1, 4)) - 1
	buckets = np.array_split(np.sort(clean), 4)
	return buckets[q_idx]


def create_feature_dataset(
	df_svstates: pd.DataFrame,
	df_wls: pd.DataFrame,
	df_cmc: pd.DataFrame,
	cn0_smooth_window: int = 15,
	cn0_quartile: int = 1,
) -> pd.DataFrame:
	"""Build GNSS feature table from per-satellite and WLS data."""
	df_wls = _resolve_wls_columns(df_wls.copy())
	cn0_smooth_window = max(int(cn0_smooth_window), 1)
	cn0_quartile = int(np.clip(cn0_quartile, 1, 4))

	df = pd.DataFrame()
	list_t = np.unique(df_svstates["gps_millis"])
	df["gps_millis"] = list_t

	sat_numbers = []
	d_cn0 = {stat: [] for stat in STATS}
	d_elevation = {stat: [] for stat in STATS}

	d_cmc_l1 = []
	d_cmc_e1 = []
	d_cmc_l2 = []
	d_cmc_e5 = []
	d_time_utc = []
	d_cn0_q_mean = []
	d_cn0_q_std = []

	# Pre-index by timestamp to avoid repeated full-dataframe filtering in the loop.
	sv_groups = {k: v for k, v in df_svstates.groupby("gps_millis", sort=False)}

	if not df_cmc.empty:
		cmc_abs = df_cmc.assign(abs_cmc_a=np.abs(df_cmc["CMC_a"]), abs_cmc_b=np.abs(df_cmc["CMC_b"]))
		cmc_gps = cmc_abs[cmc_abs["gnss_id"] == "gps"]
		cmc_gal = cmc_abs[cmc_abs["gnss_id"] == "galileo"]
		cmc_l1_map = cmc_gps.groupby("gps_millis", sort=False)["abs_cmc_a"].mean()
		cmc_l2_map = cmc_gps.groupby("gps_millis", sort=False)["abs_cmc_b"].mean()
		cmc_e1_map = cmc_gal.groupby("gps_millis", sort=False)["abs_cmc_a"].mean()
		cmc_e5_map = cmc_gal.groupby("gps_millis", sort=False)["abs_cmc_b"].mean()
	else:
		cmc_l1_map = pd.Series(dtype=float)
		cmc_l2_map = pd.Series(dtype=float)
		cmc_e1_map = pd.Series(dtype=float)
		cmc_e5_map = pd.Series(dtype=float)

	for t in tqdm(list_t, total=len(list_t), desc="Feature extraction"):
		sub_data_timestamps = sv_groups.get(t)
		if sub_data_timestamps is None:
			continue
		t_utc = sub_data_timestamps["time"].iloc[0] if "time" in sub_data_timestamps.columns else pd.NaT

		indic_l1_e1 = sub_data_timestamps["observation_code"] == "1C"
		sub_data_timestamps = sub_data_timestamps[indic_l1_e1]

		unique_sat = np.unique(sub_data_timestamps["gnss_sv_id"])
		sat_numbers.append(len(unique_sat))

		cn0_data = sub_data_timestamps["cn0_dbhz"].values
		elevation_data = sub_data_timestamps["el_sv_deg"].values
		cn0_all_nan = len(cn0_data) == 0 or np.isnan(cn0_data).all()
		elevation_all_nan = len(elevation_data) == 0 or np.isnan(elevation_data).all()

		for stat in STATS:
			if not cn0_all_nan:
				val = STATS_FUNCTIONS[stat](cn0_data)
			else:
				val = np.nan
			d_cn0[stat].append(val)

			if not elevation_all_nan:
				val = STATS_FUNCTIONS[stat](elevation_data)
			else:
				val = np.nan
			d_elevation[stat].append(val)

		cn0_quartile_values = _select_quartile_values(elevation_data, cn0_quartile)
		if len(cn0_quartile_values) > 0:
			d_cn0_q_mean.append(np.nanmean(cn0_quartile_values))
			d_cn0_q_std.append(np.nanstd(cn0_quartile_values))
		else:
			d_cn0_q_mean.append(np.nan)
			d_cn0_q_std.append(np.nan)

		d_cmc_l1.append(cmc_l1_map.get(t, np.nan))
		d_cmc_e1.append(cmc_e1_map.get(t, np.nan))
		d_cmc_l2.append(cmc_l2_map.get(t, np.nan))
		d_cmc_e5.append(cmc_e5_map.get(t, np.nan))
		d_time_utc.append(t_utc)

	df["NSV"] = sat_numbers
	for stat in STATS:
		df[f"EL {stat}"] = np.array(d_elevation[stat])
		df[f"CN0 {stat}"] = np.array(d_cn0[stat])
		df[f"CN0_{stat}_smoothed"] = (
			df[f"CN0 {stat}"].rolling(window=cn0_smooth_window, min_periods=1, center=True).mean().values
		)
	df["CN0_q_mean"] = np.array(d_cn0_q_mean)
	df["CN0_q_std"] = np.array(d_cn0_q_std)
	df["CN0_q_mean_smoothed"] = df["CN0_q_mean"].rolling(
		window=cn0_smooth_window,
		min_periods=1,
		center=True,
	).mean().values
	df["CN0_q_std_smoothed"] = df["CN0_q_std"].rolling(
		window=cn0_smooth_window,
		min_periods=1,
		center=True,
	).mean().values
	df["CN0_q_selected_quartile"] = cn0_quartile
	# Approximation 5 secondes a 1 Hz: variabilite CN0 locale utile pour detecter des scintillations.
	df["CN0_var_5s"] = df["CN0 mean"].rolling(window=5, min_periods=2).var().values
	df["CMC_l1"] = d_cmc_l1
	df["CMC_e1"] = d_cmc_e1
	df["CMC_l2"] = d_cmc_l2
	df["CMC_e5"] = d_cmc_e5
	df["time_utc"] = _parse_datetime_series(pd.Series(d_time_utc)).dt.tz_convert(None)

	# Merge with WLS to keep trajectory and PDOP information.
	merged = pd.merge(df, df_wls, on="gps_millis", how="left")
	merged = merged.sort_values("time_utc", kind="stable").reset_index(drop=True)

	keep_columns = [
		"gps_millis",
		"time_utc",
		"NSV",
		"EL mean",
		"EL std",
		"CN0 mean",
		"CN0 std",
		"CN0_mean_smoothed",
		"CN0_std_smoothed",
		"CN0_q_mean",
		"CN0_q_std",
		"CN0_q_mean_smoothed",
		"CN0_q_std_smoothed",
		"CN0_q_selected_quartile",
		"CN0_var_5s",
		"pdop",
		"CMC_l1",
		"CMC_e1",
		"CMC_l2",
		"CMC_e5",
	]
	existing_cols = [col for col in keep_columns if col in merged.columns]
	return merged[existing_cols]


def add_groundtruth_nearest(df_features: pd.DataFrame, df_gt: pd.DataFrame) -> pd.DataFrame:
	"""Add nearest GT latitude/longitude to feature rows based on UTC time."""
	df_out = df_features.copy()
	if "time_utc" not in df_out.columns:
		logger.warning("No time_utc in feature table, skipping GT merge.")
		return df_out

	gt = df_gt.copy()
	gt_time_col = None
	for candidate in ["utc_time", "time_utc", "time"]:
		if candidate in gt.columns:
			gt_time_col = candidate
			break

	if gt_time_col is None:
		logger.warning("No GT time column found, skipping GT merge.")
		return df_out

	if "longitude" not in gt.columns or "latitude" not in gt.columns:
		try:
			gt = gt[gt["longitude [deg]"] != 0].rename(
				columns={"longitude [deg]": "longitude", "latitude [deg]": "latitude"}
			)
		except Exception:
			logger.warning("No longitude/latitude columns found in GT, skipping GT merge.")
			return df_out

	gt["time_gt"] = _parse_datetime_series(gt[gt_time_col]).dt.tz_convert(None)
	gt = gt.dropna(subset=["time_gt", "longitude", "latitude"]).sort_values("time_gt")

	feats = df_out.copy()
	feats_valid = feats[feats["time_utc"].notna()].sort_values("time_utc")
	feats_nan = feats[feats["time_utc"].isna()]

	merged = pd.merge_asof(
		feats_valid,
		gt[["time_gt", "latitude", "longitude"]],
		left_on="time_utc",
		right_on="time_gt",
		direction="nearest",
	)

	if not feats_nan.empty:
		feats_nan = feats_nan.copy()
		feats_nan["latitude"] = np.nan
		feats_nan["longitude"] = np.nan
		merged = pd.concat([merged, feats_nan], ignore_index=True)

	merged = merged.sort_values("time_utc", kind="stable")

	merged = merged.rename(columns={"latitude": "lat_gt", "longitude": "long_gt"})
	return merged.drop(columns=["time_gt"], errors="ignore")


def process_gnss_feature_extraction(
	path_svstates: str | Path,
	path_wlssolution: str | Path,
	output_csv: str | Path,
	path_gt: str | Path | None = None,
	cn0_smooth_window: int = 15,
	cn0_quartile: int = 1,
	verbose: bool = True,
) -> pd.DataFrame:
	"""Run the full GNSS feature extraction pipeline and save CSV output."""
	path_svstates = Path(path_svstates)
	path_wlssolution = Path(path_wlssolution)
	output_csv = Path(output_csv)

	if verbose:
		logger.info("Read sv_states from %s", path_svstates)
	df_svstates = pd.read_csv(path_svstates)

	if verbose:
		logger.info("Read WLS solution from %s", path_wlssolution)
	df_wls = pd.read_csv(path_wlssolution)

	cmc_data = create_multipath_data(df_svstates)
	df_features = create_feature_dataset(
		df_svstates,
		df_wls,
		cmc_data,
		cn0_smooth_window=cn0_smooth_window,
		cn0_quartile=cn0_quartile,
	)

	if path_gt is not None and Path(path_gt).exists():
		if verbose:
			logger.info("Read GT from %s", path_gt)
		df_gt = pd.read_csv(path_gt)
		df_features = add_groundtruth_nearest(df_features, df_gt)
	elif path_gt is not None:
		logger.warning("GT file does not exist: %s", path_gt)

	output_csv.parent.mkdir(parents=True, exist_ok=True)
	df_features.to_csv(output_csv, index=False)

	if verbose:
		logger.info("GNSS features saved to %s (%d rows)", output_csv, len(df_features))

	return df_features
