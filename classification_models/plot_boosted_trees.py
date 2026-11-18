"""Plot individual trees from saved boosted classifiers, without retraining."""
import argparse
import json
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from November.classification_models.classify import load_artifact

BASE = Path(__file__).resolve().parent
MODELS = {"catboost": "CatBoost", "xgboost": "XGBoost", "lightgbm": "LightGBM"}


def leaf(value):
    return {"text": f"Leaf score\n{float(value):.6g}"}


def extract_tree(model, name, features, index):
    if name == "lightgbm":
        trees = model.dump_model()["tree_info"]
        raw = trees[index]

        def convert(node):
            if "leaf_value" in node:
                return leaf(node["leaf_value"])
            if node["decision_type"] != "<=":
                raise ValueError("Only numeric LightGBM splits are supported")
            return {"text": f'{features[node["split_feature"]]} <= {node["threshold"]:.6g}',
                    "left": convert(node["left_child"]), "right": convert(node["right_child"]),
                    "left_label": "Yes", "right_label": "No"}
        return convert(raw["tree_structure"]), len(trees), raw
    if name == "xgboost":
        trees = model.get_booster().get_dump(dump_format="json")
        raw = json.loads(trees[index])

        def convert(node):
            if "leaf" in node:
                return leaf(node["leaf"])
            feature = node["split"]
            if feature.startswith("f") and feature[1:].isdigit():
                feature = features[int(feature[1:])]
            children = {child["nodeid"]: child for child in node["children"]}
            return {"text": f'{feature} < {node["split_condition"]:.6g}',
                    "left": convert(children[node["yes"]]), "right": convert(children[node["no"]]),
                    "left_label": "Yes", "right_label": "No"}
        return convert(raw), len(trees), raw
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "catboost.json"
        model.save_model(str(path), format="json")
        exported = json.loads(path.read_text(encoding="utf-8"))
    trees = exported["oblivious_trees"]
    raw = trees[index]
    splits = raw.get("splits") or []
    mapping = {f["feature_index"]: f["flat_feature_index"]
               for f in exported["features_info"]["float_features"]}

    def convert(depth, bits):
        if depth == len(splits):
            return leaf(raw["leaf_values"][bits])
        split = splits[depth]
        if split["split_type"] != "FloatFeature":
            raise ValueError("Only numeric CatBoost splits are supported")
        feature = features[mapping[split["float_feature_index"]]]
        # CatBoost leaf indices encode split outcomes as bits, split 0 first.
        return {"text": f'{feature} > {split["border"]:.6g}',
                "left": convert(depth + 1, bits),
                "right": convert(depth + 1, bits | (1 << depth)),
                "left_label": "No", "right_label": "Yes"}
    return convert(0, 0), len(trees), raw


def draw_tree(tree, title, path, max_depth):
    terminals = []
    nodes, edges = [], []

    def visit(node, depth):
        truncated = "left" in node and depth >= max_depth
        record = {"depth": depth, "leaf": "left" not in node,
                  "text": node["text"] + ("\n[deeper branches omitted]" if truncated else "")}
        nodes.append(record)
        if "left" not in node or truncated:
            record["x"] = len(terminals)
            terminals.append(record)
        else:
            left = visit(node["left"], depth + 1)
            right = visit(node["right"], depth + 1)
            record["x"] = (left["x"] + right["x"]) / 2
            edges.extend([(record, left, node["left_label"]), (record, right, node["right_label"])])
        return record

    visit(tree, 0)
    depth = max(n["depth"] for n in nodes)
    fig, ax = plt.subplots(figsize=(max(10, len(terminals) * 2.8), max(5, (depth + 1) * 1.7)))
    for parent, child, label in edges:
        ax.plot([parent["x"], child["x"]], [-parent["depth"], -child["depth"]], color="#777777", zorder=1)
        ax.text((parent["x"] + child["x"]) / 2, -(parent["depth"] + child["depth"]) / 2,
                label, ha="center", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    for node in nodes:
        label = "\n".join(textwrap.fill(line, width=29) for line in node["text"].splitlines())
        ax.text(node["x"], -node["depth"], label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "#E0F0E5" if node["leaf"] else "#E2EDF6",
                      "edgecolor": "#577080"}, zorder=2)
    ax.set_xlim(-0.65, max(0.65, len(terminals) - 0.35))
    ax.set_ylim(-depth - 0.6, 0.6)
    ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.text(0.5, 0.03, "One component of the ensemble, not the complete classifier. Leaf scores are not anomaly probabilities.\n"
             "Numeric routes shown after median imputation. Full unrounded splits are saved in JSON.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=BASE / "runs")
    parser.add_argument("--tree-index", type=int, default=0, help="Zero-based index; 0 selects the first tree")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum displayed depth; root is depth 0 (0-5)")
    for name in MODELS:
        parser.add_argument("--" + name, type=Path, help="Explicit run folder; otherwise latest timestamp with artifact metadata")
    args = parser.parse_args()
    if args.tree_index < 0 or not 0 <= args.max_depth <= 5:
        parser.error("Use a nonnegative tree index and max-depth between 0 and 5")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    for name, label in MODELS.items():
        run = getattr(args, name)
        if run is None:
            candidates = sorted((args.runs_dir / name).glob("*/artifact/preprocessing.json"))
            if not candidates:
                raise ValueError(f"No exported model found for {name}")
            run = candidates[-1].parent.parent
        model, meta = load_artifact(run / "artifact")
        if meta["model"] != name:
            raise ValueError(f"Expected {name} artifact in {run}")
        try:
            tree, count, raw = extract_tree(model, name, meta["features"], args.tree_index)
        except IndexError as exc:
            raise ValueError(f"Tree index {args.tree_index} is outside {name}'s ensemble") from exc
        output = run / "tree_plots" / stamp
        output.mkdir(parents=True, exist_ok=False)
        title = f"{label}: Tree {args.tree_index + 1} of {count} | Display depth <= {args.max_depth}"
        stem = output / f"tree_{args.tree_index + 1:04d}_structure"
        draw_tree(tree, title, stem, args.max_depth)
        stem.with_suffix(".json").write_text(json.dumps({
            "model": name, "source_run": run.name, "model_sha256": meta["model_sha256"],
            "tree_index": args.tree_index, "ensemble_trees": count, "features": meta["features"],
            "display_max_depth": args.max_depth, "tree": raw,
        }, indent=2), encoding="utf-8")
        print(f"{label}: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
