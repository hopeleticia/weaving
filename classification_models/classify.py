"""Classify new sensor records using an already-trained, saved classifier.

Unlike the individual training scripts, this file does not train a model,
select a threshold, or calculate evaluation metrics. It loads the saved model
and its preprocessing metadata, then applies the validation-selected threshold
to classify each row as normal or anomaly (alarm codes 2 and 8 combined).

Use this file to apply a delivered model to new data without retraining.
Input CSVs must already be engineering-scaled and contain the required sensor
columns; alarm codes and ground-truth labels are not required. Missing sensor
values are replaced using the saved training medians. A new output CSV retains
the input columns and adds anomaly_probability, predicted_binary, and
predicted_label; the input file is not modified.

"""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from importlib.metadata import version
from model_artifacts import FILENAMES


def load_artifact(folder, trust_pickle=False):
    meta = json.loads((folder / "preprocessing.json").read_text(encoding="utf-8"))
    name = meta["model"]
    path = folder / FILENAMES[name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["model_sha256"]:
        raise ValueError("Model checksum differs from preprocessing metadata")
    if meta.get("schema_version") != 1:
        raise ValueError("Unsupported artifact schema")
    features = meta["features"]
    medians = np.asarray(meta["medians"], dtype=float)
    if not features or len(set(features)) != len(features) or len(medians) != len(features) or not np.isfinite(medians).all():
        raise ValueError("Invalid features or imputation medians")
    if not 0 <= meta["threshold"] <= 1:
        raise ValueError("Invalid decision threshold")
    library = "scikit-learn" if name == "decision_tree" else name
    if version(library) != meta["versions"][library]:
        if name == "decision_tree":
            raise ValueError("Decision Tree requires the scikit-learn version recorded at export")
        warnings.warn("Model library version differs from export environment")
    if name == "decision_tree":
        if not trust_pickle:
            raise ValueError("Pickle can execute code. Use --trust-pickle only for artifacts from a trusted source.")
        import pickle
        from sklearn.tree import DecisionTreeClassifier
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, DecisionTreeClassifier) or not np.array_equal(model.classes_, [0, 1]):
            raise ValueError("Expected a binary DecisionTreeClassifier with classes [0, 1]")
        count = model.n_features_in_
    elif name == "lightgbm":
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(path))
        count = model.num_feature()
    elif name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(str(path))
        count = model.n_features_in_
    else:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(path))
        count = len(model.feature_names_)
    if count != len(features):
        raise ValueError("Model feature count does not match metadata")
    return model, meta


def predict_frame(frame, model, meta):
    missing = set(meta["features"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing sensor columns: {sorted(missing)}")
    x = frame[meta["features"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    x = np.where(np.isnan(x), np.asarray(meta["medians"], dtype=np.float32), x)
    scores = np.empty(0)
    if len(x):
        scores = np.asarray(model.predict(x) if meta["model"] == "lightgbm" else model.predict_proba(x)[:, 1])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid anomaly probabilities")
    result = frame.copy()
    result["anomaly_probability"] = scores
    result["predicted_binary"] = (scores >= meta["threshold"]).astype(np.int8)
    result["predicted_label"] = np.where(result.predicted_binary == 1, "anomaly", "normal")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Exported artifact folder")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-pickle", action="store_true", help="Allow loading a trusted Decision Tree pickle; it can execute code")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        raise ValueError("Use a new output filename; source and existing files are not overwritten")
    model, meta = load_artifact(args.model_dir, trust_pickle=args.trust_pickle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        first = True
        for frame in pd.read_csv(args.input, chunksize=100000):
            predict_frame(frame, model, meta).to_csv(handle, index=False, header=first)
            first = False
        if first:
            predict_frame(pd.read_csv(args.input, nrows=0), model, meta).to_csv(handle, index=False)
    print("Predictions saved:", args.output)


if __name__ == "__main__":
    main()
