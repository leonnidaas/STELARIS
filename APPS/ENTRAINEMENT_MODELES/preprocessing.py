"""Preprocessing pipeline for GNSS time-series classification.

Main improvements:
- Geographic segmentation by fixed distance chunks
- Robust split with group isolation by segment_id + stratification on labels
- Sequence generation that never crosses segment boundaries
- Structured artefact export (npz + scaler/label encoder pkl + metadata.json)
"""

from __future__ import annotations


import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from utils import PARAMS_ENTRAINEMENT, TRAINING_DIR, get_dataset_path, get_traj_paths


prefix = "gnss_feat_"
DEFAULT_FEATURE_NAMES = ["NSV", "EL mean", "EL std", "pdop", "CN0 std", "CMC_l1"]
DEFAULT_FEATURE_NAMES = [f"{prefix}{name}" for name in DEFAULT_FEATURE_NAMES]
DATA_SET_NAME = "final_fusion_csv"

R_TEST = float(PARAMS_ENTRAINEMENT.get("test_size", 0.2))
R_VAL = float(PARAMS_ENTRAINEMENT.get("val_size", 0.1))
RANDOM_STATE_SPLIT = int(PARAMS_ENTRAINEMENT.get("random_state_split", 42))
RANDOM_STATE_VAL = int(PARAMS_ENTRAINEMENT.get("random_state_val", 123))
SEGMENT_LENGTH_M = float(PARAMS_ENTRAINEMENT.get("segment_length_m", 2000.0))
SPEED_THRESHOLD_MPS = float(PARAMS_ENTRAINEMENT.get("speed_threshold_mps", 0.5))
STATIONARY_KEEP_SECONDS = float(PARAMS_ENTRAINEMENT.get("stationary_keep_seconds", 30.0))
STATIONARY_KEEP_ROWS = int(PARAMS_ENTRAINEMENT.get("stationary_keep_rows", 30))
LAST_SEGMENT_MIN_LENGTH_M = float(PARAMS_ENTRAINEMENT.get("last_segment_min_length_m", 500.0))


def _downsample_stationary_rows(
    df: pd.DataFrame,
    speed_threshold_mps: float,
    keep_seconds: float,
    keep_rows: int,
) -> pd.DataFrame:
    """Reduce stationary density: keep all moving points, sparse sampling at low speed."""
    if "__trajet_id" not in df.columns:
        raise ValueError("Colonne '__trajet_id' manquante pour le downsampling d'immobilite.")

    if "gps_millis" not in df.columns:
        raise ValueError("Colonne 'gps_millis' manquante pour le downsampling d'immobilite.")

    df = df.copy()
    speed = df["velocity"]
    df["__speed_mps"] = speed

    keep_mask = np.zeros(len(df), dtype=bool)
    grouped = df.groupby("__trajet_id", sort=False).groups

    for _, idx in grouped.items():
        idx = np.array(idx)
        sub = df.iloc[idx].sort_values("gps_millis", kind="stable")
        sub_idx = sub.index.values

        s = pd.to_numeric(sub["__speed_mps"], errors="coerce").values.astype(np.float64)
        t = pd.to_numeric(sub["gps_millis"], errors="coerce").values.astype(np.float64) * 1e-3

        last_kept_time = -np.inf
        last_kept_pos = -10**9

        for pos, global_idx in enumerate(sub_idx):
            v = s[pos]
            is_stationary = np.isfinite(v) and v < speed_threshold_mps

            if not is_stationary:
                keep_mask[global_idx] = True
                continue

            dt_ok = (np.isfinite(t[pos]) and (t[pos] - last_kept_time >= keep_seconds))
            row_ok = (pos - last_kept_pos) >= max(1, keep_rows)

            if (last_kept_pos < 0) or dt_ok or row_ok:
                keep_mask[global_idx] = True
                if np.isfinite(t[pos]):
                    last_kept_time = t[pos]
                last_kept_pos = pos

    out = df.loc[keep_mask].copy()
    out.drop(columns=["__speed_mps"], inplace=True, errors="ignore")
    out = out.sort_values(["__trajet_id", "gps_millis"], kind="stable").reset_index(drop=True)
    return out


def assign_geographic_segments(
    df: pd.DataFrame,
    segment_length_m: float = 2000.0,
    last_segment_min_length_m: float = 500.0,
) -> pd.DataFrame:
    """Assign a unique segment_id per row, trajectory by trajectory.

    Segment id format: <TRAJET>_SEG_<N>
    """
    if "__trajet_id" not in df.columns:
        raise ValueError("Colonne interne '__trajet_id' manquante pour la segmentation.")

    df = df.copy()

    # Stable ordering by trajectory then time where available.
    if "gps_millis" in df.columns:
        df = df.sort_values(["__trajet_id", "gps_millis"], kind="stable").reset_index(drop=True)
    else:
        df = df.sort_values(["__trajet_id"], kind="stable").reset_index(drop=True)

    if "distance_trip" not in df.columns:
        raise ValueError(
            "Impossible de segmenter: colonne de distance cumulée introuvable. "
            "Attendu en priorité: 'distance_trip'."
        )

    segment_ids = np.empty(len(df), dtype=object)

    for traj_id, idx in df.groupby("__trajet_id", sort=False).groups.items():
        idx = np.array(idx)
        sub = df.iloc[idx]

        dist = pd.to_numeric(sub["distance_trip"], errors="coerce").values.astype(np.float64)
        if np.all(np.isnan(dist)):
            raise ValueError(
                "Impossible de segmenter: la colonne 'distance_trip' est entièrement invalide (NaN)."
            )
        valid0 = np.nanmin(dist)
        dist = np.where(np.isnan(dist), valid0, dist)
        cum_dist = dist - dist[0]

        seg_index = np.floor(cum_dist / max(segment_length_m, 1.0)).astype(int) + 1
        n_seg = int(seg_index.max())
        if n_seg >= 2:
            tail_len = float(np.nanmax(cum_dist) - (n_seg - 1) * max(segment_length_m, 1.0))
            if tail_len < float(last_segment_min_length_m):
                seg_index[seg_index == n_seg] = n_seg - 1

        width = max(2, len(str(int(np.max(seg_index)))))
        segment_ids[idx] = [f"{traj_id}_SEG_{int(s):0{width}d}" for s in seg_index]

    df["segment_id"] = segment_ids
    return df


def create_sequences_centered(
    data: np.ndarray,
    sequence_length: int,
    t: np.ndarray,
    segment_codes: np.ndarray,
    delta_t_lim: float = 1.05,
) -> np.ndarray:
    """Build centered temporal windows without crossing segment boundaries."""
    n_rows = len(data)
    assert n_rows == len(t) == len(segment_codes)
    assert sequence_length % 2 == 1, "sequence_length must be odd"

    k = (sequence_length - 1) // 2
    windows = []

    for i in range(n_rows):
        t_i = t[i]
        seg_i = segment_codes[i]
        x_i = [data[i]]

        for j in range(1, k + 1):
            idx = max(0, i - j)
            is_same_segment = segment_codes[idx] == seg_i
            is_close_in_time = abs(t[idx] - t_i) < j * delta_t_lim
            x_i.append(data[idx] if (is_same_segment and is_close_in_time) else x_i[-1])
        x_i.reverse()

        for j in range(1, k + 1):
            idx = min(i + j, n_rows - 1)
            is_same_segment = segment_codes[idx] == seg_i
            is_close_in_time = abs(t[idx] - t_i) < j * delta_t_lim
            x_i.append(data[idx] if (is_same_segment and is_close_in_time) else x_i[-1])

        windows.append(np.array(x_i))

    return np.array(windows)


def get_data(traj_id_list: list[str]) -> pd.DataFrame:
    """Load and concatenate data from final fusion CSVs with trajectory id."""
    df_features = pd.DataFrame()
    for traj_id in traj_id_list:
        traj_paths = get_traj_paths(traj_id)
        data_file = traj_paths[DATA_SET_NAME]
        if not data_file.exists():
            raise FileNotFoundError(f"Fichier de features introuvable: {data_file}")
        df_i = pd.read_csv(data_file)
        df_i["__trajet_id"] = traj_id
        df_features = pd.concat([df_features, df_i], ignore_index=True)
    return df_features


def _sanitize_feature_names(feature_names: list[str]) -> tuple[list[str], list[str]]:
    forbidden_coords = {"longitude", "latitude", "lon", "lat", "longitude_gt", "latitude_gt", "long_gt", "lat_gt"}
    kept = [f for f in feature_names if f not in forbidden_coords]
    dropped = [f for f in feature_names if f in forbidden_coords]
    return kept, dropped


def _class_distribution(y: np.ndarray, label_encoder: LabelEncoder | None = None) -> dict[str, dict[str, float]]:
    values, counts = np.unique(y, return_counts=True)
    total = float(np.sum(counts)) if len(counts) else 1.0

    out = {}
    for v, c in zip(values, counts):
        if label_encoder is not None and np.issubdtype(type(v), np.integer):
            name = str(label_encoder.inverse_transform([int(v)])[0])
        else:
            name = str(v)
        out[name] = {
            "count": int(c),
            "ratio": float(c / total),
        }
    return out


def _pick_best_stratified_group_split(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
    shuffle: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick one split from StratifiedGroupKFold with best ratio + class balance.

    Falls back to GroupShuffleSplit if SGKF is not feasible.
    """
    n_samples = len(y)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise ValueError("Split impossible: moins de 2 groupes segment_id disponibles.")

    test_size = float(np.clip(test_size, 0.05, 0.5))
    n_splits = max(2, int(round(1.0 / test_size)))
    n_splits = min(n_splits, n_groups)

    y_values, y_counts = np.unique(y, return_counts=True)
    global_prop = {v: c / y_counts.sum() for v, c in zip(y_values, y_counts)}

    try:
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=(random_state if shuffle else None),
        )

        best = None
        best_score = np.inf

        for tr_idx, te_idx in sgkf.split(np.zeros(n_samples), y, groups):
            test_ratio_real = len(te_idx) / n_samples
            te_y = y[te_idx]
            te_values, te_counts = np.unique(te_y, return_counts=True)
            te_prop = {v: c / te_counts.sum() for v, c in zip(te_values, te_counts)}

            prop_gap = 0.0
            for v in y_values:
                prop_gap += abs(global_prop.get(v, 0.0) - te_prop.get(v, 0.0))

            # Penalize strong deviation from requested test ratio.
            size_gap = abs(test_ratio_real - test_size)
            score = size_gap + prop_gap

            if score < best_score:
                best_score = score
                best = (tr_idx, te_idx)

        if best is None:
            raise ValueError("Aucun split SGKF valide trouvé.")

        return best

    except Exception:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr_idx, te_idx = next(gss.split(np.zeros(n_samples), y, groups))
        return tr_idx, te_idx


def _save_preprocessing_metadata(
    metadata_path: Path,
    dataset_name: str,
    ordered_feature_names: list[str],
    traj_id_list: list[str],
    window_size: int,
    segment_length_m: float,
    class_distribution_global: dict,
    class_distribution_train: dict,
    class_distribution_val: dict,
    class_distribution_test: dict,
    total_rows_raw: int,
    total_rows_after_filters: int,
    total_rows_after_stride: int,
    test_split_ratio: float,
    val_split_ratio: float,
    n_segments: int,
    speed_threshold_mps: float,
    stationary_keep_seconds: float,
    stationary_keep_rows: int,
    last_segment_min_length_m: float,
    artefacts: dict[str, str],
) -> None:
    metadata = {
        "dataset_name": dataset_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features": ordered_feature_names,
        "preprocessing": {
            "scaler_type": "StandardScaler",
            "window_size": int(window_size),
            "stride": int(1),
            "test_split_ratio": float(test_split_ratio),
            "val_split_ratio": float(val_split_ratio),
            "window_mode": "centered",
            "segment_length_m": float(segment_length_m),
            "last_segment_min_length_m": float(last_segment_min_length_m),
            "split_strategy": "StratifiedGroupKFold (fallback GroupShuffleSplit)",
            "group_key": "segment_id",
            "stratify_key": "label",
            "speed_threshold_mps": float(speed_threshold_mps),
            "stationary_keep_seconds": float(stationary_keep_seconds),
            "stationary_keep_rows": int(stationary_keep_rows),
        },
        "source_data": {
            "trajets": traj_id_list,
            "total_rows": int(total_rows_raw),
            "total_rows_after_filters": int(total_rows_after_filters),
            "total_rows_after_stride": int(total_rows_after_stride),
            "n_segments": int(n_segments),
        },
        "class_distribution": {
            "global": class_distribution_global,
            "train": class_distribution_train,
            "val": class_distribution_val,
            "test": class_distribution_test,
        },
        "artefacts": artefacts,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=True)


def main(
        traj_id_list: list[str],
        dataset_name: str,
        feature_names: list[str],
        window_size: int,
        stride: int,
        test_split_ratio: float,
        shuffle: bool,
        segment_length_m: float,
        speed_threshold_mps: float,
        stationary_keep_seconds: float,
        stationary_keep_rows: int,
        last_segment_min_length_m: float,
    ) -> None:
    
    # Gestion des chemins de sauvegarde
    config = get_dataset_path(dataset_name)
    output_dir = config["output_dir"]
    classes_param_path = config["classes_param"]
    scaler_param_path = config["scaler_param"]
    preprocessed_data_path = config["preprocessed_data"]
    metadata_path = config["metadata"]
    label_encoder_path = config["label_encoder_path"]
    
    print("[1/7] Loading data ...")
    df_features = get_data(traj_id_list)
    total_rows_raw = int(len(df_features))

    print("\n[2/8] Stationary downsampling ...")
    n_before = len(df_features)
    df_features = _downsample_stationary_rows(
        df_features,
        speed_threshold_mps=speed_threshold_mps,
        keep_seconds=stationary_keep_seconds,
        keep_rows=stationary_keep_rows,
    )
    print(f"  Rows before/after downsampling: {n_before:,} -> {len(df_features):,}")

    print("\n[3/8] Geographic segmentation ...")
    df_features = assign_geographic_segments(
        df_features,
        segment_length_m=segment_length_m,
        last_segment_min_length_m=last_segment_min_length_m,
    )

    safe_feature_names, dropped_coords = _sanitize_feature_names(feature_names)
    if dropped_coords:
        print(f"  Info: colonnes geographiques retirees des features train: {dropped_coords}")
    if not safe_feature_names:
        raise ValueError("Aucune feature exploitable apres suppression des coordonnees brutes.")

    missing_features = [c for c in safe_feature_names if c not in df_features.columns]
    if missing_features:
        raise ValueError(f"Features manquantes dans le dataset: {missing_features}")

    if "label" not in df_features.columns:
        raise ValueError("Colonne 'label' manquante dans le dataset.")
    if "gps_millis" not in df_features.columns:
        raise ValueError("Colonne 'gps_millis' manquante dans le dataset.")

    print("\n[4/8] Filtering, encoding and sequence build ...")
    
    x = df_features[safe_feature_names].values
    y_label = df_features["label"].values
    t = pd.to_numeric(df_features["gps_millis"], errors="coerce").values * 1e-3
    segment_id = df_features["segment_id"].astype(str).values

    mask_finite = np.isfinite(x).all(axis=1) & np.isfinite(t)
    x = x[mask_finite]
    y_label = y_label[mask_finite]
    t = t[mask_finite]
    segment_id = segment_id[mask_finite]

    effective_window_size = int(window_size)
    if effective_window_size % 2 == 0:
        effective_window_size += 1
        print(
            f"  Warning: window_size pair detecte ({window_size}). "
            f"window_size_effective={effective_window_size} sera utilise."
        )

    if stride > 1:
        keep_idx = np.arange(0, len(x), max(1, int(stride)))
        x = x[keep_idx]
        y_label = y_label[keep_idx]
        t = t[keep_idx]
        segment_id = segment_id[keep_idx]

    total_rows_after_filters = int(np.sum(mask_finite))
    total_rows_after_stride = int(len(x))

    le = LabelEncoder().fit(y_label) # On encode les labels avant de faire les splits pour garantir la cohérence des classes dans les splits et le calcul des class weights, même si certaines classes sont absentes de certains splits.
    y = le.transform(y_label)

    # Segment codes are used to prevent sequence contamination across segments.
    segment_codes, _ = pd.factorize(segment_id, sort=False)
    x_tensor = create_sequences_centered(
        x,
        sequence_length=effective_window_size,
        t=t,
        segment_codes=segment_codes,
    )

    print(f"  Samples after filtering: {len(x):,}")
    print(f"  Class distribution:      {Counter(y_label)}")
    print(f"  Segments uniques:        {len(np.unique(segment_id))}")

    list_classes_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    print(f"  Class weights: {dict(enumerate(list_classes_weights))}")

    print("\n[5/8] Robust train/val/test split (stratified + grouped) ...")
    train_idx, test_idx = _pick_best_stratified_group_split(
        y=y,
        groups=segment_id,
        test_size=test_split_ratio,
        random_state=RANDOM_STATE_SPLIT,
        shuffle=shuffle,
    )

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    id_train, id_test = segment_id[train_idx], segment_id[test_idx]
    t_train, t_test = t[train_idx], t[test_idx]
    x_tensor_train = x_tensor[train_idx]
    x_tensor_test = x_tensor[test_idx]

    val_ratio_in_train = R_VAL / max(1e-6, (1.0 - test_split_ratio))
    train_train_local_idx, train_val_local_idx = _pick_best_stratified_group_split(
        y=y_train,
        groups=id_train,
        test_size=val_ratio_in_train,
        random_state=RANDOM_STATE_VAL,
        shuffle=shuffle,
    )

    print("\n[6/8] Scaling ...")
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    for i in range(effective_window_size):
        x_tensor_train[:, i, :] = scaler.transform(x_tensor_train[:, i, :])
        x_tensor_test[:, i, :] = scaler.transform(x_tensor_test[:, i, :])

    x_train_flat = x_tensor_train.reshape(x_tensor_train.shape[0], -1)
    x_test_flat = x_tensor_test.reshape(x_tensor_test.shape[0], -1)

    print("\n[7/8] Saving artefacts ...")
    np.save(classes_param_path, le.classes_)
    joblib.dump(scaler, scaler_param_path)
    joblib.dump(le, label_encoder_path)

    np.savez_compressed(
        preprocessed_data_path,
        X_train=x_train,
        X_test=x_test,
        X_train_flat=x_train_flat,
        X_test_flat=x_test_flat,
        X_tensor_train=x_tensor_train,
        X_tensor_test=x_tensor_test,
        y_train=y_train,
        y_test=y_test,
        id_train=id_train,
        id_test=id_test,
        t_train=t_train,
        t_test=t_test,
        train_train_idx=train_train_local_idx,
        train_val_idx=train_val_local_idx,
        list_classes_weights=list_classes_weights,
    )

    print("\n[8/8] Saving metadata JSON ...")
    class_global = _class_distribution(y, le)
    class_train = _class_distribution(y_train[train_train_local_idx], le)
    class_val = _class_distribution(y_train[train_val_local_idx], le)
    class_test = _class_distribution(y_test, le)

    _save_preprocessing_metadata(
        metadata_path=metadata_path,
        dataset_name=dataset_name,
        ordered_feature_names=safe_feature_names,
        traj_id_list=traj_id_list,
        window_size=effective_window_size,
        segment_length_m=segment_length_m,
        class_distribution_global=class_global,
        class_distribution_train=class_train,
        class_distribution_val=class_val,
        class_distribution_test=class_test,
        total_rows_raw=total_rows_raw,
        total_rows_after_filters=total_rows_after_filters,
        total_rows_after_stride=total_rows_after_stride,
        test_split_ratio=test_split_ratio,
        val_split_ratio=R_VAL,
        n_segments=len(np.unique(segment_id)),
        speed_threshold_mps=speed_threshold_mps,
        stationary_keep_seconds=stationary_keep_seconds,
        stationary_keep_rows=stationary_keep_rows,
        last_segment_min_length_m=last_segment_min_length_m,
        artefacts={
            "output_dir": str(output_dir),
            "classes_param_npy": str(classes_param_path),
            "scaler_pkl": str(scaler_param_path),
            "label_encoder_pkl": str(label_encoder_path),
            "preprocessed_data_npz": str(preprocessed_data_path),
            "metadata_json": str(metadata_path),
        },
    )

    print(f"\n  training_dir          -> {TRAINING_DIR}")
    print(f"  classes_param.npy    -> {classes_param_path}")
    print(f"  scaler_param.pkl     -> {scaler_param_path}")
    print(f"  label_encoder.pkl    -> {label_encoder_path}")
    print(f"  preprocessed_data    -> {preprocessed_data_path}")
    print(f"  metadata.json        -> {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocessing du dataset d'entrainement.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d_%H-%M"),
        help="Nom du dataset cible (dossier sous DATA/03_TRAINING).",
    )
    parser.add_argument(
        "--trajets",
        nargs="+",
        default=PARAMS_ENTRAINEMENT.get("trajets", ["BORDEAUX_COUTRAS", "MARTINE_01"]),
        help="Liste des trajets sources.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=PARAMS_ENTRAINEMENT.get("features", DEFAULT_FEATURE_NAMES),
        help="Colonnes features a utiliser.",
    )
    parser.add_argument("--window-size", type=int, default=PARAMS_ENTRAINEMENT.get("window_size", 5))
    parser.add_argument("--stride", type=int, default=PARAMS_ENTRAINEMENT.get("stride", 1))
    parser.add_argument("--test-split-ratio", type=float, default=PARAMS_ENTRAINEMENT.get("test_size", 0.2))
    parser.add_argument("--shuffle", action="store_true", default=PARAMS_ENTRAINEMENT.get("shuffle", False))
    parser.add_argument(
        "--segment-length-m",
        type=float,
        default=PARAMS_ENTRAINEMENT.get("segment_length_m", SEGMENT_LENGTH_M),
        help="Longueur d'un segment geographique (en metres).",
    )
    parser.add_argument(
        "--speed-threshold-mps",
        type=float,
        default=PARAMS_ENTRAINEMENT.get("speed_threshold_mps", SPEED_THRESHOLD_MPS),
        help="Seuil de vitesse (m/s) pour detecter l'immobilite.",
    )
    parser.add_argument(
        "--stationary-keep-seconds",
        type=float,
        default=PARAMS_ENTRAINEMENT.get("stationary_keep_seconds", STATIONARY_KEEP_SECONDS),
        help="En immobilite, garde au plus un point par periode (secondes).",
    )
    parser.add_argument(
        "--stationary-keep-rows",
        type=int,
        default=PARAMS_ENTRAINEMENT.get("stationary_keep_rows", STATIONARY_KEEP_ROWS),
        help="En immobilite, garde au plus un point toutes les N lignes.",
    )
    parser.add_argument(
        "--last-segment-min-length-m",
        type=float,
        default=PARAMS_ENTRAINEMENT.get("last_segment_min_length_m", LAST_SEGMENT_MIN_LENGTH_M),
        help="Fusionne le dernier segment avec le precedent s'il est plus court que ce seuil.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        traj_id_list=args.trajets,
        dataset_name=args.dataset_name,
        feature_names=args.features,
        window_size=args.window_size,
        stride=max(1, int(args.stride)),
        test_split_ratio=float(args.test_split_ratio),
        shuffle=bool(args.shuffle),
        segment_length_m=float(args.segment_length_m),
        speed_threshold_mps=float(args.speed_threshold_mps),
        stationary_keep_seconds=float(args.stationary_keep_seconds),
        stationary_keep_rows=max(1, int(args.stationary_keep_rows)),
        last_segment_min_length_m=float(args.last_segment_min_length_m),
    )
