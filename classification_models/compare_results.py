"""Combine saved classifier evaluation results into a comparison table.

Use this script after the individual training scripts have finished. Supply
the CatBoost, XGBoost, and LightGBM run folders with --catboost, --xgboost, and
--lightgbm. Optionally include a Decision Tree run with --decision-tree. Each
folder must contain metrics.json. Set --output-dir for the comparison reports.

The script reads existing results; it does not train models, load their weights,
classify data, change thresholds, or automatically choose the best run. It
includes separate validation and test rows for each supplied model.

Precision, recall, anomaly F1, and false-positive rate are expressed as
percentages. The full CSV also records the selected probability threshold,
source run name, and saved computation metrics. Computation times retain their
original units and depend on hardware and runtime load.

Outputs are comparison.csv, a shorter comparison.md table, and a console
summary. Existing comparison files in the output folder are overwritten.
This script does not generate plots or verify that the runs used matching
features and dataset splits. Select comparable runs before interpreting model
differences; otherwise the table may reflect different evaluation conditions.
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['catboost', 'xgboost', 'lightgbm']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--decision-tree', type=Path, help='Optional adapted Decision Tree run with metrics.json')
    args = parser.parse_args()
    rows = []
    for name in ['catboost', 'xgboost', 'lightgbm'] + (['decision_tree'] if args.decision_tree else []):
        path = getattr(args, name)
        m = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
        for split in ['validation', 'test']:
            result = m[split]
            rows.append({
                'model': name, 'run': path.name, 'split': split,
                **{key + '_percent': result[key] * 100 for key in ['precision', 'recall', 'f1', 'false_positive_rate']},
                'threshold': m['selected_threshold'],
                **m['computation'],
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / 'comparison.csv', index=False)
    columns = ['model', 'split', 'f1_percent', 'precision_percent', 'recall_percent', 'false_positive_rate_percent', 'training_seconds', 'test_prediction_seconds']
    lines = ['# Classifier Comparison', '', 'Source run IDs and all timing fields are in comparison.csv. Times depend on hardware and runtime load.', '', '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(f'{row[c]:.6f}' if isinstance(row[c], float) else str(row[c]) for c in columns) + ' |')
    (args.output_dir / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(frame[columns].to_string(index=False))


if __name__ == '__main__':
    main()
