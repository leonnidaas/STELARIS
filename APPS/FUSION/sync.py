import pandas as pd
from geopy.distance import geodesic
import numpy as np
from pyproj import Transformer

from utils import standardize_dataframe


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse mixed datetime formats used across GT/GNSS/feature files."""
    try:
        dt = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except TypeError:
        dt = pd.to_datetime(series, utc=True, errors="coerce")
    return dt.dt.tz_localize(None)


def _find_first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def fusion_df(df_gt: pd.DataFrame, df_gnss: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les donnees GT et GNSS sur la base des timestamps."""
    gt = standardize_dataframe(df_gt.copy())
    gnss = df_gnss.copy()

    gt_time_col = _find_first_existing(list(gt.columns), ["time_utc", "utc_time", "time"])
    gnss_time_col = _find_first_existing(list(gnss.columns), ["time_utc", "utc_time", "time"])

    if gt_time_col is None or gnss_time_col is None:
        raise ValueError("Colonnes temporelles manquantes pour fusion GT/GNSS.")

    gt["time_utc"] = _parse_datetime_series(gt[gt_time_col])
    gnss["time_utc"] = _parse_datetime_series(gnss[gnss_time_col])

    df_gt_1hz = gt.copy()
    df_gt_1hz["time_utc"] = df_gt_1hz["time_utc"].dt.floor("s")
    df_gt_1hz = df_gt_1hz.drop_duplicates(subset="time_utc", keep="first")

    start_recouvrement = max(df_gt_1hz["time_utc"].min(), gnss["time_utc"].min())
    end_recouvrement = min(df_gt_1hz["time_utc"].max(), gnss["time_utc"].max())

    df_gt_1hz = df_gt_1hz.sort_values("time_utc")
    gnss = gnss.sort_values("time_utc")

    matched_df = pd.merge(df_gt_1hz, gnss, on="time_utc", how="outer", suffixes=("_gt", "_gnss"))
    mask = (matched_df["time_utc"] >= start_recouvrement) & (matched_df["time_utc"] <= end_recouvrement)
    matched_df = matched_df.loc[mask].sort_values("time_utc").reset_index(drop=True)

    rename_map = {
        "latitude": "latitude_gt",
        "longitude": "longitude_gt",
        "altitude": "altitude_gt",
        "lat_rx_wls_deg": "latitude_gnss",
        "lon_rx_wls_deg": "longitude_gnss",
        "alt_rx_wls_m": "altitude_gnss",
    }
    matched_df = matched_df.rename(columns=rename_map)

    return matched_df


def get_step_distance(index: int, df: pd.DataFrame, lat_col: str, lon_col: str) -> float:
    if index == 0:
        return 0.0

    curr_point = (df.at[index, lat_col], df.at[index, lon_col])
    prev_point = (df.at[index - 1, lat_col], df.at[index - 1, lon_col])

    if pd.isna(curr_point[0]) or pd.isna(curr_point[1]) or pd.isna(prev_point[0]) or pd.isna(prev_point[1]):
        return 0.0

    return geodesic(prev_point, curr_point).meters


def conversion_ign69(lats: np.ndarray, lons: np.ndarray, alts_wgs84: np.ndarray):
    try:
        transformer = Transformer.from_crs("EPSG:4979", "EPSG:2154+5720", always_xy=True)
        x, y, z_ign69 = transformer.transform(lons, lats, alts_wgs84)

        if np.isinf(x[0]):
            return None, None, None

        return x, y, z_ign69
    except Exception:
        return None, None, None


def calculate_track_errors(df: pd.DataFrame, lever_x: float, lever_y: float) -> pd.DataFrame:
    gt_coords = df[["x_gt", "y_gt"]].values
    gnss_coords = df[["x_gnss", "y_gnss"]].values

    error_vec = gnss_coords - gt_coords
    tangent = np.diff(gt_coords, axis=0, append=gt_coords[-1:].copy())

    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    u_tangent = tangent / norms

    df["err_longitudinale"] = np.einsum("ij,ij->i", error_vec, u_tangent)
    df["err_longitudinale"] = df["err_longitudinale"] -9999999999999999999999999

    u_normal = np.stack([-u_tangent[:, 1], u_tangent[:, 0]], axis=1)
    df["err_laterale"] = np.einsum("ij,ij->i", error_vec, u_normal)
    df["err_laterale"] = df["err_laterale"] -9999999999999999999999999

    return df


def process_gnss_gt_fusion(
    df_gt: pd.DataFrame,
    df_gnss: pd.DataFrame,
    output_csv=None,
    gnss_offset: tuple[float, float, float] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    lever_x, lever_y, lever_z = (0.0, 0.0, 0.0)
    if gnss_offset is not None:
        if len(gnss_offset) != 3:
            raise ValueError("gnss_offset doit contenir 3 valeurs (x, y, z) en metres.")
        lever_x, lever_y, lever_z = (float(gnss_offset[0]), float(gnss_offset[1]), float(gnss_offset[2]))

    if verbose:
        print("Etape 1 : Fusion des donnees GT et GNSS...")

    matched_df = fusion_df(df_gt, df_gnss)

    if verbose:
        print(f"  ✓ {len(matched_df)} points fusionnes")
        print("Etape 2 : Calcul des distances...")

    matched_df["dist_step_m_gt"] = matched_df.index.to_series().apply(
        lambda i: get_step_distance(i, matched_df, "latitude_gt", "longitude_gt")
    )
    matched_df["dist_step_m_gnss"] = matched_df.index.to_series().apply(
        lambda i: get_step_distance(i, matched_df, "latitude_gnss", "longitude_gnss")
    )

    if verbose:
        print("Etape 3 : Calcul des ecarts GT-GNSS...")

    def convert_lever_arm_to_lat_lon_offset(lat, lon, lever_x_m, lever_y_m):
        earth_radius = 6378137.0
        delta_lat = (lever_y_m / earth_radius) * (180.0 / np.pi)
        delta_lon = (lever_x_m / (earth_radius * np.cos(np.radians(lat)))) * (180.0 / np.pi)
        return delta_lat, delta_lon
    
    def get_ecart_gt_gnss(row ,lever_x=lever_x, lever_y=lever_y):
        point_gt = (row["latitude_gt"], row["longitude_gt"])
        if lever_x != 0.0 or lever_y != 0.0:
            delta_lat, delta_lon = convert_lever_arm_to_lat_lon_offset(
                row["latitude_gt"], row["longitude_gt"], lever_x, lever_y
            )
            point_gt = (row["latitude_gt"] + delta_lat, row["longitude_gt"] + delta_lon)
        point_gnss = (row["latitude_gnss"], row["longitude_gnss"])
        if pd.isna(point_gt[0]) or pd.isna(point_gt[1]) or pd.isna(point_gnss[0]) or pd.isna(point_gnss[1]):
            return None
        return geodesic(point_gt, point_gnss).meters

    matched_df["ecart_gt_gnss_geodesic_m"] = matched_df.apply(get_ecart_gt_gnss, axis=1, lever_x=lever_x, lever_y=lever_y)

    if verbose:
        print("Etape 4 : Conversion WGS84 -> Lambert 93 / IGN69...")

    matched_df["altitude_gt"] = -matched_df["altitude_gt"]

    x, y, z = conversion_ign69(
        matched_df["latitude_gt"].values,
        matched_df["longitude_gt"].values,
        matched_df["altitude_gt"].values,
    )

    if x is None:
        raise ValueError("Conversion IGN69 echouee. Le fichier .tif n'est pas detecte par PROJ.")

    matched_df["x_gt"], matched_df["y_gt"], matched_df["z_gt_ign69"] = conversion_ign69(
        matched_df["latitude_gt"].values,
        matched_df["longitude_gt"].values,
        matched_df["altitude_gt"].values,
    )

    matched_df["x_gnss"], matched_df["y_gnss"], matched_df["z_gnss_ign69"] = conversion_ign69(
        matched_df["latitude_gnss"].values,
        matched_df["longitude_gnss"].values,
        matched_df["altitude_gnss"].values,
    )

    matched_df["ecart_gt_gnss_m"] = np.hypot(
        matched_df["x_gnss"] - matched_df["x_gt"],
        matched_df["y_gnss"] - matched_df["y_gt"],
    )
    matched_df["ecart_gt_gnss_3d_m"] = np.sqrt(
        (matched_df["x_gnss"] - matched_df["x_gt"]) ** 2
        + (matched_df["y_gnss"] - matched_df["y_gt"]) ** 2
        + (matched_df["z_gnss_ign69"] - matched_df["z_gt_ign69"]) ** 2
    )

    if verbose:
        print("Etape 5 : Calcul des erreurs longitudinales et laterales...")

    matched_df = calculate_track_errors(matched_df, lever_x=lever_x, lever_y=lever_y)

    if output_csv:
        matched_df.to_csv(output_csv, index=False)
        if verbose:
            print(f"  ✓ Resultat sauvegarde : {output_csv}")

    return matched_df


def process_feature_fusion(
    df_lidar_features: pd.DataFrame,
    df_gnss_features: pd.DataFrame,
    df_gt: pd.DataFrame | None = None,
    output_csv=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fusion temporelle des features LiDAR et GNSS, avec ajout GT brut optionnel."""
    lidar = df_lidar_features.copy()
    gnss = df_gnss_features.copy()

    if "time_utc" not in lidar.columns:
        raise ValueError("Le fichier LiDAR doit contenir une colonne time_utc.")
    if "time_utc" not in gnss.columns:
        raise ValueError("Le fichier features GNSS doit contenir une colonne time_utc.")

    lidar["time_utc"] = _parse_datetime_series(lidar["time_utc"])
    gnss["time_utc"] = _parse_datetime_series(gnss["time_utc"])

    lidar = lidar.sort_values("time_utc").reset_index(drop=True)
    gnss = gnss.sort_values("time_utc").reset_index(drop=True)

    gnss_payload_cols = [col for col in gnss.columns if col not in ["time_utc"]]
    gnss_prefixed = gnss[["time_utc"] + gnss_payload_cols].rename(
        columns={col: f"gnss_feat_{col}" for col in gnss_payload_cols}
    )

    fused = pd.merge_asof(
        lidar,
        gnss_prefixed,
        on="time_utc",
        direction="nearest",
    )

    if df_gt is not None:
        gt = standardize_dataframe(df_gt.copy())
        gt_time_col = _find_first_existing(list(gt.columns), ["time_utc", "utc_time", "time"])
        if gt_time_col is not None and "latitude" in gt.columns and "longitude" in gt.columns:
            gt["time_utc"] = _parse_datetime_series(gt[gt_time_col])
            gt_keep = ["time_utc", "latitude", "longitude"]
            if "altitude" in gt.columns:
                gt_keep.append("altitude")
            gt_view = gt[gt_keep].sort_values("time_utc").rename(
                columns={
                    "latitude": "gt_raw_latitude",
                    "longitude": "gt_raw_longitude",
                    "altitude": "gt_raw_altitude",
                }
            )
            fused = pd.merge_asof(
                fused.sort_values("time_utc"),
                gt_view,
                on="time_utc",
                direction="nearest",
            )

    if output_csv:
        fused.to_csv(output_csv, index=False)
        if verbose:
            print(f"  ✓ Fusion features sauvegardee : {output_csv}")

    return fused


def process_feature_fusion_from_files(
    path_lidar_features,
    path_gnss_features,
    path_gt=None,
    output_csv=None,
    verbose: bool = True,
) -> pd.DataFrame:
    if verbose:
        print("Lecture des fichiers pour fusion LiDAR + GNSS + GT...")

    df_lidar = pd.read_csv(path_lidar_features)
    df_gnss = pd.read_csv(path_gnss_features)
    df_gt = pd.read_csv(path_gt) if path_gt else None

    return process_feature_fusion(
        df_lidar_features=df_lidar,
        df_gnss_features=df_gnss,
        df_gt=df_gt,
        output_csv=output_csv,
        verbose=verbose,
    )


def process_final_label_fusion(
    df_fused_features: pd.DataFrame,
    df_labels: pd.DataFrame,
    output_csv=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fusion finale: ajoute la colonne label au jeu deja fusionne (LiDAR+GNSS+GT)."""
    fused = df_fused_features.copy()
    labels = df_labels.copy()

    if "label" not in labels.columns:
        raise ValueError("Le fichier labels doit contenir la colonne label.")

    # Priorite 1: merge par timestamp si possible.
    if "time_utc" in fused.columns and "time_utc" in labels.columns:
        fused["time_utc"] = _parse_datetime_series(fused["time_utc"])
        labels["time_utc"] = _parse_datetime_series(labels["time_utc"])

        labels = labels.sort_values("time_utc")
        fused = fused.sort_values("time_utc")

        # On conserve une seule info label par timestamp.
        labels_view = labels[["time_utc", "label"]].drop_duplicates(subset=["time_utc"], keep="last")

        out = pd.merge_asof(
            fused,
            labels_view,
            on="time_utc",
            direction="nearest",
        )
    # Priorite 2: merge sur latitude/longitude GT (fichier labels final actuel).
    elif all(col in fused.columns for col in ["latitude_gt", "longitude_gt"]) and all(
        col in labels.columns for col in ["latitude_gt", "longitude_gt"]
    ):
        fused["_lat_key"] = fused["latitude_gt"].round(8)
        fused["_lon_key"] = fused["longitude_gt"].round(8)
        labels["_lat_key"] = labels["latitude_gt"].round(8)
        labels["_lon_key"] = labels["longitude_gt"].round(8)

        labels_view = labels[["_lat_key", "_lon_key", "label"]].drop_duplicates(
            subset=["_lat_key", "_lon_key"], keep="last"
        )

        out = pd.merge(fused, labels_view, on=["_lat_key", "_lon_key"], how="left")
        out = out.drop(columns=["_lat_key", "_lon_key"], errors="ignore")
    else:
        raise ValueError(
            "Impossible de fusionner les labels: il faut soit time_utc dans les deux tables, "
            "soit latitude_gt/longitude_gt dans les deux tables."
        )

    if output_csv:
        out.to_csv(output_csv, index=False)
        if verbose:
            print(f"  ✓ Fusion finale labels+features sauvegardee : {output_csv}")

    return out


def process_final_label_fusion_from_files(
    path_fused_features,
    path_labels,
    output_csv=None,
    verbose: bool = True,
) -> pd.DataFrame:
    if verbose:
        print("Lecture des fichiers pour fusion finale features + label...")

    df_features = pd.read_csv(path_fused_features)
    df_labels = pd.read_csv(path_labels)

    return process_final_label_fusion(
        df_fused_features=df_features,
        df_labels=df_labels,
        output_csv=output_csv,
        verbose=verbose,
    )

