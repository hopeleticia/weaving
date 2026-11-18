"""Save a trained classifier together with the information needed to reuse it.

The individual training scripts call save_artifact after fitting their model
and selecting a decision threshold. This helper is imported by those scripts;
it is not a standalone training or classification command.

An artifact is the saved model plus its accompanying metadata. This helper
creates an artifact subfolder inside the run's output directory and writes:
- The fitted model: model.cbm for CatBoost, model.ubj for XGBoost, model.txt
  for LightGBM, or model.pkl for Decision Tree.
- preprocessing.json: sensor feature order, training medians for missing-value
  replacement, the selected threshold, label mappings, and library versions used to check the saved model file's integrity.

Keep the model file and preprocessing.json together when sharing or deploying
the classifier. The classification loader needs both to reproduce the intended
input preparation and decision rule without retraining. Input sensor data must
already be signed-decoded and engineering-scaled; this helper does not scale
data or save the training dataset.

An existing artifact folder is not overwritten. Decision Tree uses pickle;
only load such files from a trusted source because loading can execute code.
"""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

FILENAMES = {"catboost": "model.cbm", "xgboost": "model.ubj", "lightgbm": "model.txt", "decision_tree": "model.pkl"}


def save_artifact(model, name, features, medians, threshold, output_dir):
    folder = Path(output_dir) / "artifact"
    folder.mkdir(exist_ok=False)
    path = folder / FILENAMES[name]
    if name == "decision_tree":
        import pickle
        with path.open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif name == "lightgbm":
        model.booster_.save_model(str(path))
    else:
        model.save_model(str(path))
    metadata = {
        "schema_version": 1, "model": name, "model_file": path.name,
        "features": list(features), "medians": [float(v) for v in medians],
        "threshold": float(threshold), "decision_rule": "probability >= threshold",
        "labels": {"0": "normal", "1": "anomaly"},
        "alarm_mapping": {"0": 0, "2": 1, "8": 1},
        "inputs": "Already signed-decoded and engineering-scaled; do not scale again",
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "versions": {n: version(n) for n in ["scikit-learn" if name == "decision_tree" else name, "numpy", "pandas", "scikit-learn"]},
    }
    (folder / "preprocessing.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print("Saved fitted model and preprocessing:", folder)
    return folder
