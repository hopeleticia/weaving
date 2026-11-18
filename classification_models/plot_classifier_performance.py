"""Plot four classifiers' saved validation/test metrics without retraining."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MODELS = {"catboost": "CatBoost", "xgboost": "XGBoost",
          "lightgbm": "LightGBM", "decision_tree": "Decision Tree"}
METRICS = {"precision": "Precision", "recall": "Recall", "f1": "F1 score"}


def load_results(runs_dir, overrides):
    rows, sources = [], {}
    reference = None
    reference_source = None
    for model in MODELS:
        folder = overrides.get(model)
        if folder is None:
            candidates = sorted((runs_dir / model).glob("*/metrics.json"))
            if not candidates:
                raise ValueError(f"No metrics.json found under {runs_dir / model}")
            # Timestamp folder names sort chronologically; never select by score.
            path = candidates[-1]
        else:
            path = folder / "metrics.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("model") != model:
            raise ValueError(f"{path}: expected model {model}")
        signature = {
            "features": report["features"],
            "files": {split: sorted(report["input_files"][split])
                      for split in ("train", "validation", "test")},
            "rows": {split: report["computation"][split + "_rows"]
                     for split in ("train", "validation", "test")},
        }
        if reference is not None and signature != reference:
            details = []
            if signature["features"] != reference["features"]:
                details.append("Feature lists or their order differ.")
            for split in ("train", "validation", "test"):
                if (signature["files"][split] != reference["files"][split]
                        or signature["rows"][split] != reference["rows"][split]):
                    details.append(
                        f"{split}: selected run has {signature['rows'][split]:,} rows / "
                        f"{len(signature['files'][split])} files; reference has "
                        f"{reference['rows'][split]:,} rows / {len(reference['files'][split])} files.")
            reversed_splits = all(
                signature["files"][a] == reference["files"][b]
                and signature["rows"][a] == reference["rows"][b]
                for a, b in (("validation", "test"), ("test", "validation")))
            if reversed_splits:
                details.append("Validation and test datasets are REVERSED relative to the reference.")
            raise ValueError(
                f"Cannot compare {model} ({path.parent.name}) with {reference_source}.\n"
                + "\n".join(details)
                + "\nSelect an existing matching run with --" + model.replace("_", "-")
                + " RUN_FOLDER, or retrain with matching splits. Do not relabel saved metrics.")
        reference = signature
        reference_source = f"{model} ({path.parent.name})"
        sources[model] = {"run": path.parent.name, "metrics_file": str(path.resolve())}
        for split in ("validation", "test"):
            result = report[split]
            row = {"model": model, "split": split, "run": path.parent.name}
            for key in (*METRICS, "false_positive_rate"):
                value = float(result[key])
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{path}: invalid {split} {key}")
                row[key + "_percent"] = value * 100
            rows.append(row)
    return pd.DataFrame(rows), sources


def plot_split(table, split, output, target):
    data = table[table.split == split].set_index("model").loc[list(MODELS)]
    title = "Validation" if split == "validation" else "Testing"
    fig, (ax, fpr_ax) = plt.subplots(1, 2, figsize=(15, 5.5),
                                    gridspec_kw={"width_ratios": [2, 1]}, layout="constrained")
    x, width = np.arange(len(MODELS)), 0.24
    for i, (key, label) in enumerate(METRICS.items()):
        bars = ax.bar(x + (i - 1) * width, data[key + "_percent"], width,
                      label=label, color=["#2878B5", "#DB8B24", "#26846A"][i])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.axhline(target, color="#A33A36", linestyle="--", linewidth=1.2,
               label=f"F1 target ({target:g}%)")
    ax.set(xticks=x, xticklabels=list(MODELS.values()), ylim=(0, 108),
           ylabel="Score (%)", title="Precision, recall and anomaly F1")
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    # A shared FPR scale across both figures supports direct visual comparison.
    maximum = max(float(table.false_positive_rate_percent.max()) * 1.35, 0.001)
    bars = fpr_ax.bar(x, data.false_positive_rate_percent, color="#956240", width=0.6)
    fpr_ax.bar_label(bars, fmt="%.5f", padding=4, fontsize=8)
    fpr_ax.set(xticks=x, xticklabels=list(MODELS.values()), ylim=(0, maximum),
               ylabel="False-positive rate (%)", title="False-positive rate (lower is better)")
    fpr_ax.tick_params(axis="x", labelrotation=20)
    for axis in (ax, fpr_ax):
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{title} classification performance", fontsize=15)
    fig.savefig(output / f"{split}_performance.png", dpi=220)
    fig.savefig(output / f"{split}_performance.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=BASE / "runs")
    parser.add_argument("--output-dir", type=Path, default=BASE / "performance_plots")
    parser.add_argument("--f1-target", type=float, default=85.0, help="Reference F1 target in percent")
    for model in MODELS:
        parser.add_argument("--" + model.replace("_", "-"), type=Path,
                            help="Explicit run directory; otherwise use latest timestamp with metrics.json")
    args = parser.parse_args()
    if not 0 <= args.f1_target <= 100:
        parser.error("--f1-target must be between 0 and 100")
    table, sources = load_results(args.runs_dir, {m: getattr(args, m) for m in MODELS})
    output = args.output_dir / datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    output.mkdir(parents=True, exist_ok=False)
    for split in ("validation", "test"):
        plot_split(table, split, output, args.f1_target)
    table.to_csv(output / "performance_comparison.csv", index=False)
    (output / "plot_sources.json").write_text(json.dumps({
        "sources": sources, "f1_target_percent": args.f1_target,
        "selection": "Explicit overrides or latest timestamp per model; not highest score",
        "note": "Matching filenames and row counts do not verify identical dataset contents."
    }, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print("Plots and source manifest saved to:", output)


if __name__ == "__main__":
    main()
