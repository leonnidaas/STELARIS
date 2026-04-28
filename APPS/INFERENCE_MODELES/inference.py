from __future__ import annotations

from collections import Counter
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from keras.models import load_model

from EXTRACTION_DES_FEATURES_GNSS.extraction_features_gnss import process_gnss_feature_extraction
from ENTRAINEMENT_MODELES.preprocessing import create_sequences_centered
from utils import MODELS_DIR, PYTHON_RINEX_INTERPRETER, ROOT_PATH, get_dataset_path, get_traj_paths


APPS_ROOT = ROOT_PATH / "APPS"
SUPPORTED_MODELS = ("GRU", "XGBOOST", "CNN_1D")
MODEL_COLUMN_KEYS = {
	"GRU": "gru",
	"XGBOOST": "xgb",
	"CNN_1D": "cnn_1d",
}


def _find_latest_model_for_dataset(model_kind: str, dataset_name: str) -> tuple[Path, dict]:
	root = MODELS_DIR / model_kind
	if not root.exists():
		raise FileNotFoundError(f"Dossier modele introuvable: {root}")

	metadata_files = sorted(root.rglob("*_metadonnees.json"), key=lambda p: p.stat().st_mtime, reverse=True)

	for meta_path in metadata_files:
		try:
			with open(meta_path, "r", encoding="utf-8") as f:
				meta = json.load(f)
		except Exception:
			continue

		if meta.get("dataset_id") == dataset_name:
			return meta_path, meta

	raise FileNotFoundError(
		f"Aucun modele {model_kind} trouve pour dataset_id='{dataset_name}' sous {root}."
	)


def _load_gru_model_from_metadata(meta_path: Path, meta: dict):
	model_path = Path(meta.get("artefacts", {}).get("model", ""))
	if not model_path.exists():
		model_path = meta_path.parent / (meta.get("model_id", "") + ".keras")
	if not model_path.exists():
		raise FileNotFoundError(f"Fichier modele GRU introuvable: {model_path}")
	return load_model(model_path), model_path


def _load_cnn_model_from_metadata(meta_path: Path, meta: dict):
	model_path = Path(meta.get("artefacts", {}).get("model", ""))
	if not model_path.exists():
		model_path = meta_path.parent / (meta.get("model_id", "") + ".keras")
	if not model_path.exists():
		raise FileNotFoundError(f"Fichier modele CNN_1D introuvable: {model_path}")
	return load_model(model_path), model_path


def _load_xgb_model_from_metadata(meta_path: Path, meta: dict):
	model_path = Path(meta.get("artefacts", {}).get("model", ""))
	if not model_path.exists():
		model_path = meta_path.parent / (meta.get("model_id", "") + ".json")
	if not model_path.exists():
		raise FileNotFoundError(f"Fichier modele XGBoost introuvable: {model_path}")
	model = XGBClassifier()
	model.load_model(model_path)
	return model, model_path


def _normalize_model_selection(model_kinds: list[str] | None) -> list[str]:
	if not model_kinds:
		return ["GRU", "XGBOOST"]

	normalized: list[str] = []
	for raw in model_kinds:
		mk = str(raw).strip().upper()
		if mk in {"XGB", "XGBOOST"}:
			mk = "XGBOOST"
		elif mk in {"CNN", "CNN1D", "CNN_1D", "1D_CNN"}:
			mk = "CNN_1D"
		elif mk == "GRU":
			mk = "GRU"
		else:
			raise ValueError(f"Modele non supporte: {raw}. Attendus: {SUPPORTED_MODELS}")

		if mk not in normalized:
			normalized.append(mk)

	if not normalized:
		raise ValueError("Aucun modele selectionne pour l'inference.")

	return normalized


def _sanitize_feature_names(feature_names: list[str]) -> list[str]:
	forbidden_coords = {
		"longitude",
		"latitude",
		"lon",
		"lat",
		"longitude_gt",
		"latitude_gt",
		"long_gt",
		"lat_gt",
	}
	return [f for f in feature_names if f not in forbidden_coords]


def _label_to_col_suffix(label: str) -> str:
	"""Convert arbitrary label names to safe column suffixes."""
	s = str(label).strip().lower()
	s = re.sub(r"[^a-z0-9]+", "_", s)
	s = re.sub(r"_+", "_", s).strip("_")
	return s or "unknown"


def _harmonize_gnss_feature_columns(df: pd.DataFrame, required_features: list[str]) -> pd.DataFrame:
	"""Accept both prefixed and unprefixed GNSS feature naming conventions."""
	out = df.copy()

	for feat in required_features:
		if feat in out.columns:
			continue

		if feat.startswith("gnss_feat_"):
			base = feat[len("gnss_feat_"):]
			if base in out.columns:
				out[feat] = out[base]
				continue
		else:
			prefixed = f"gnss_feat_{feat}"
			if prefixed in out.columns:
				out[feat] = out[prefixed]

	if "gnss_feat_gps_millis" not in out.columns and "gps_millis" in out.columns:
		out["gnss_feat_gps_millis"] = out["gps_millis"]

	return out


def _prepare_inference_tensors(df: pd.DataFrame, feature_names: list[str], window_size: int, scaler):
	missing = [c for c in feature_names if c not in df.columns]
	if missing:
		raise ValueError(f"Features manquantes pour inference: {missing}")
	if "gnss_feat_gps_millis" not in df.columns:
		raise ValueError("Colonne 'gnss_feat_gps_millis' manquante pour inference.")

	x = df[feature_names].to_numpy(dtype=float)
	t = pd.to_numeric(df["gnss_feat_gps_millis"], errors="coerce").to_numpy(dtype=float) * 1e-3

	mask_finite = np.isfinite(x).all(axis=1) & np.isfinite(t)
	idx_valid = np.where(mask_finite)[0]
	if len(idx_valid) == 0:
		raise ValueError("Aucune ligne exploitable apres filtrage NaN/inf pour inference.")

	x_valid = x[mask_finite]
	t_valid = t[mask_finite]
	x_valid_scaled = scaler.transform(x_valid)

	effective_window = int(window_size)
	if effective_window % 2 == 0:
		effective_window += 1

	segment_codes = np.zeros(len(x_valid_scaled), dtype=np.int32)
	x_tensor = create_sequences_centered(
		x_valid,
		sequence_length=effective_window,
		t=t_valid,
		segment_codes=segment_codes,
	)

	for i in range(effective_window):
		x_tensor[:, i, :] = scaler.transform(x_tensor[:, i, :])

	return {
		"mask_finite": mask_finite,
		"idx_valid": idx_valid,
		"x_valid_scaled": x_valid_scaled,
		"x_tensor": x_tensor,
	}


def _ensure_rinex_outputs(traj_paths: dict) -> tuple[Path, Path]:
	raw_gnss_path = Path(traj_paths["raw_gnss"])
	svstates_path = Path(traj_paths["space_vehicule_info"])

	if raw_gnss_path.exists() and svstates_path.exists():
		return raw_gnss_path, svstates_path

	obs_file = Path(traj_paths["obs_file"])
	nav_file = Path(traj_paths["nav_file"])
	if not obs_file.exists() or not nav_file.exists():
		raise FileNotFoundError(
			f"RINEX manquants pour inference. OBS={obs_file.exists()} NAV={nav_file.exists()}"
		)

	rinex_python = _resolve_rinex_python()
	cmd = [rinex_python, "-m", "TRAITEMENT_RINEX.app", "--traj", str(traj_paths["id"])]
	proc = subprocess.run(cmd, cwd=APPS_ROOT, check=False, capture_output=True, text=True)
	if proc.returncode != 0:
		logs = "\n".join([s for s in [proc.stdout, proc.stderr] if s]).strip()
		raise RuntimeError(
			"Echec TRAITEMENT_RINEX pour inference. "
			f"Cmd={' '.join(cmd)}\n{logs}"
		)

	if not raw_gnss_path.exists() or not svstates_path.exists():
		raise FileNotFoundError("Les sorties TRAITEMENT_RINEX n'ont pas ete generees.")

	return raw_gnss_path, svstates_path


def _build_gnss_features_for_inference(traj_paths: dict) -> Path:
	raw_gnss_path, svstates_path = _ensure_rinex_outputs(traj_paths)
	gnss_features_csv = Path(traj_paths["gnss_features_csv"])

	process_gnss_feature_extraction(
		path_svstates=svstates_path,
		path_wlssolution=raw_gnss_path,
		output_csv=gnss_features_csv,
		path_gt=None,
		verbose=True,
	)

	if not gnss_features_csv.exists():
		raise FileNotFoundError(f"Extraction GNSS non produite: {gnss_features_csv}")

	return gnss_features_csv


def _resolve_rinex_python() -> str:
	configured = str(PYTHON_RINEX_INTERPRETER or "").strip()
	fallback = APPS_ROOT / "venv_rinex" / "bin" / "python"

	if configured:
		p = Path(configured)
		if p.is_absolute() and p.exists():
			return str(p)
		if (APPS_ROOT / configured).exists():
			return str(APPS_ROOT / configured)
		if shutil.which(configured):
			return configured

	if fallback.exists():
		return str(fallback)

	return sys.executable


def run_inference_for_trajet(
	traj_id: str,
	dataset_name: str,
	model_kinds: list[str] | None = None,
) -> dict[str, str]:
	traj_paths = get_traj_paths(traj_id)
	data_csv = _build_gnss_features_for_inference(traj_paths)
	selected_models = _normalize_model_selection(model_kinds)

	dataset_cfg = get_dataset_path(dataset_name)
	if not dataset_cfg["metadata"].exists():
		raise FileNotFoundError(f"Metadata dataset introuvable: {dataset_cfg['metadata']}")
	if not dataset_cfg["label_encoder_path"].exists():
		raise FileNotFoundError(f"Label encoder introuvable: {dataset_cfg['label_encoder_path']}")
	if not dataset_cfg["scaler_param"].exists():
		raise FileNotFoundError(f"Scaler introuvable: {dataset_cfg['scaler_param']}")

	with open(dataset_cfg["metadata"], "r", encoding="utf-8") as f:
		dataset_meta = json.load(f)

	features = [str(f) for f in dataset_meta.get("features", [])]
	if not features:
		raise ValueError("Aucune feature declaree dans le metadata dataset.")
	feature_names = _sanitize_feature_names(features)
	window_size = int(dataset_meta.get("preprocessing", {}).get("window_size", 5))

	label_encoder = joblib.load(dataset_cfg["label_encoder_path"])
	scaler = joblib.load(dataset_cfg["scaler_param"])

	loaders = {
		"GRU": _load_gru_model_from_metadata,
		"XGBOOST": _load_xgb_model_from_metadata,
		"CNN_1D": _load_cnn_model_from_metadata,
	}

	models_ctx: dict[str, dict] = {}
	for model_kind in selected_models:
		meta_path, meta = _find_latest_model_for_dataset(model_kind, dataset_name)
		model, model_path = loaders[model_kind](meta_path, meta)
		models_ctx[model_kind] = {
			"meta_path": meta_path,
			"meta": meta,
			"model": model,
			"model_path": model_path,
		}

	df = pd.read_csv(data_csv)
	df = _harmonize_gnss_feature_columns(df, feature_names)
	prepared = _prepare_inference_tensors(df, feature_names, window_size, scaler)

	x_tensor = prepared["x_tensor"]
	x_valid_scaled = prepared["x_valid_scaled"]
	idx_valid = prepared["idx_valid"]

	n_classes = len(label_encoder.classes_)
	class_names = [str(c) for c in label_encoder.classes_]

	out = df.copy()
	out["pred_final"] = pd.Series(index=out.index, dtype="object")
	out["model_final"] = pd.Series(index=out.index, dtype="object")
	out["conf_final"] = np.nan
	out["conf_max_models"] = np.nan

	predictions: dict[str, dict] = {}
	for model_kind in selected_models:
		key = MODEL_COLUMN_KEYS[model_kind]
		model_obj = models_ctx[model_kind]["model"]

		if model_kind == "XGBOOST":
			proba = np.asarray(model_obj.predict_proba(x_valid_scaled), dtype=float)
		else:
			proba = np.asarray(model_obj.predict(x_tensor, verbose=0), dtype=float)

		y_idx = np.argmax(proba, axis=1)
		conf = np.max(proba, axis=1)
		labels = label_encoder.inverse_transform(np.clip(y_idx, 0, n_classes - 1))

		pred_col = f"pred_{key}"
		conf_col = f"conf_{key}"
		out[pred_col] = pd.Series(index=out.index, dtype="object")
		out[conf_col] = np.nan
		out.loc[idx_valid, pred_col] = labels
		out.loc[idx_valid, conf_col] = conf

		for j, class_name in enumerate(class_names):
			suffix = _label_to_col_suffix(class_name)
			col = f"proba_{key}_{suffix}"
			out[col] = np.nan
			out.loc[idx_valid, col] = proba[:, j]

		predictions[model_kind] = {
			"labels": labels,
			"conf": conf,
		}

	stacked_conf = np.stack([predictions[m]["conf"] for m in selected_models], axis=1)
	best_model_idx = np.argmax(stacked_conf, axis=1)
	best_conf = np.max(stacked_conf, axis=1)
	best_model_names = np.array(selected_models, dtype=object)[best_model_idx]

	final_labels = np.empty(len(idx_valid), dtype=object)
	for row_idx, best_model_name in enumerate(best_model_names):
		final_labels[row_idx] = predictions[str(best_model_name)]["labels"][row_idx]

	out.loc[idx_valid, "pred_final"] = final_labels
	out.loc[idx_valid, "model_final"] = best_model_names
	out.loc[idx_valid, "conf_final"] = best_conf
	out.loc[idx_valid, "conf_max_models"] = best_conf

	analysis_per_model: dict[str, dict] = {}
	for model_kind in selected_models:
		model_key = MODEL_COLUMN_KEYS[model_kind]
		labels = predictions[model_kind]["labels"]
		conf = predictions[model_kind]["conf"]
		analysis_per_model[model_key] = {
			"n_predictions": int(len(labels)),
			"prediction_distribution": {
				str(k): int(v) for k, v in Counter(labels).items()
			},
			"confidence": {
				"mean": float(np.mean(conf)),
				"median": float(np.median(conf)),
				"min": float(np.min(conf)),
				"max": float(np.max(conf)),
			},
		}

	pairwise_agreement: dict[str, float] = {}
	if len(selected_models) >= 2:
		for i in range(len(selected_models)):
			for j in range(i + 1, len(selected_models)):
				left = selected_models[i]
				right = selected_models[j]
				left_labels = np.asarray(predictions[left]["labels"], dtype=object)
				right_labels = np.asarray(predictions[right]["labels"], dtype=object)
				rate = float(np.mean(left_labels == right_labels))
				pairwise_agreement[
					f"{MODEL_COLUMN_KEYS[left]}__vs__{MODEL_COLUMN_KEYS[right]}"
				] = rate

	if selected_models:
		stacked_labels = np.stack(
			[np.asarray(predictions[m]["labels"], dtype=object) for m in selected_models],
			axis=1,
		)
		unanimous_mask = np.all(stacked_labels == stacked_labels[:, [0]], axis=1)
		unanimity_rate = float(np.mean(unanimous_mask))
	else:
		unanimity_rate = 0.0

	inference_dir = Path(traj_paths["inference_dir"])
	inference_dir.mkdir(parents=True, exist_ok=True)

	ts = datetime.now().strftime("%Y%m%d-%H%M%S")
	out_csv = inference_dir / f"{traj_id}_{ts}_inference.csv"
	out_json = inference_dir / f"{traj_id}_{ts}_inference_dataset.json"

	latest_csv = Path(traj_paths["inference_latest_csv"])
	latest_json = Path(traj_paths["inference_latest_json"])

	out.to_csv(out_csv, index=False)

	payload = {
		"traj_id": traj_id,
		"created_at": datetime.now().isoformat(timespec="seconds"),
		"dataset_name": dataset_name,
		"dataset_metadata": str(dataset_cfg["metadata"]),
		"models": {
			model_kind.lower(): {
				"metadata": str(models_ctx[model_kind]["meta_path"]),
				"weights": str(models_ctx[model_kind]["model_path"]),
				"model_id": str(models_ctx[model_kind]["meta"].get("model_id", "")),
				"dataset_id": str(models_ctx[model_kind]["meta"].get("dataset_id", "")),
			}
			for model_kind in selected_models
		},
		"inputs": {
			"pipeline": "OBS/NAV -> TRAITEMENT_RINEX -> EXTRACTION_DES_FEATURES_GNSS",
			"obs_file": str(traj_paths["obs_file"]),
			"nav_file": str(traj_paths["nav_file"]),
			"raw_gnss_csv": str(traj_paths["raw_gnss"]),
			"space_vehicule_info_csv": str(traj_paths["space_vehicule_info"]),
			"trajet_csv": str(data_csv),
			"n_rows_total": int(len(df)),
			"n_rows_inferred": int(len(idx_valid)),
			"window_size": int(window_size),
			"selected_models": [MODEL_COLUMN_KEYS[m] for m in selected_models],
			"model_class_labels": class_names,
		},
		"analysis": {
			"ensemble_strategy": "max_confidence_per_row",
			"per_model": analysis_per_model,
			"pairwise_agreement": pairwise_agreement,
			"all_models_unanimity_rate": unanimity_rate,
			"final_prediction_distribution": {
				str(k): int(v) for k, v in Counter(final_labels).items()
			},
		},
		"outputs": {
			"inference_csv": str(out_csv),
			"inference_json": str(out_json),
		},
	}

	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, ensure_ascii=True)

	shutil.copyfile(out_csv, latest_csv)
	shutil.copyfile(out_json, latest_json)

	return {
		"inference_csv": str(out_csv),
		"inference_json": str(out_json),
		"latest_inference_csv": str(latest_csv),
		"latest_inference_json": str(latest_json),
	}
