# Weaving Machine Alarm Classification

## Project Overview

This project uses PLC sensor measurements to distinguish normal weaving-machine
operation from alarm conditions. The goal is a reproducible classification system
with an anomaly-class F1 score of at least **85%** and a false positive rate (FPR) below **5%** on the specified test dataset.

This package includes CatBoost, XGBoost, LightGBM and Decision Tree training,
evaluation, model export and classification of additional data. 

LightGBM is the proposed model for classification.

## Classification Task

Each CSV row is classified independently:

| Alarm_Code | Binary target | Description |
| --- | --- | --- |
| 0 | 0: normal | No active target alarm, including pre-alarm rows with Code 0 |
| 2 | 1: anomaly | Weft-break alarm |
| 8 | 1: anomaly | Warp-break alarm |

Codes 2 and 8 form one anomaly class for binary classification with the Code 0 class. Other codes are excluded from training and
evaluation. 

The ten model inputs are:

```text
Infeed_Speed_PV
Weaving_Line_Speed
Motor_RPM
INFEED_TORQUE
Loadcell_Weight_PV
Loadcell_PV
Dancer_Pressure_PV
Dancer_Gyro_Position
Temperature_PV
Humidity_PV
```

Alarm_Code supplies labels only. Weaving_Run, timestamps and textual labels are
not predictor features.

## Implemented Models

CatBoost, XGBoost and LightGBM combine boosted decision trees; Decision Tree uses
one fitted tree. All four produce an anomaly probability, which is compared with
a validation-selected decision threshold.

| Model | Trees | Learning rate | Tree configuration | Imbalance handling |
| --- | --- | --- | --- | --- |
| CatBoost | 800 | 0.05 | Depth 6; ordered boosting | Class weights |
| XGBoost | 800 | 0.03 | Depth 6; histogram trees | scale_pos_weight |
| LightGBM | 800 | 0.03 | Up to 31 leaves; no explicit maximum depth | Class weights |
| Decision Tree | 1 | Not applicable | Entropy; maximum depth 10 | None |

The following sections distinguish parameters explicitly configured in the code
from settings inherited from the installed library. They describe model
configurations. The constructors are in each script's
`build_model` function; dependency versions are pinned in requirements.txt.

### CatBoost

CatBoost uses ordered boosting with symmetric trees: within a tree, the same
split is used across the nodes at a given level. A depth-6 symmetric tree has
up to 64 leaves. The tree ensemble produces the anomaly-class probability.

| Setting | Configuration | Source |
| --- | --- | --- |
| Boosting iterations | 800 | Explicit |
| Learning rate | 0.05 | Explicit |
| Tree depth | 6 | Explicit |
| Boosting method | Ordered | Explicit |
| Tree growth policy | SymmetricTree | Library default |
| Sampling method | MVS (Minimum Variance Sampling) | Explicit |
| Loss | Binary Logloss | Inferred by library for binary labels |
| Class weights | Normal: 1; anomaly: training normal/anomaly ratio | Explicit |
| Leaf regularization | l2_leaf_reg not overridden | Library default |
| Random seed | 42 | Explicit |
| CPU threads | All available (thread_count=-1) | Explicit |

MVS controls training-example sampling. Feature sampling and remaining
regularization settings are not separately tuned in this implementation.
`allow_writing_files=False` disables CatBoost's automatic training files; the
script still explicitly exports the fitted model as `model.cbm`.

### XGBoost

XGBoost constructs depth-limited decision trees using histogram-based split
search. Its trees need not be symmetric. Successive trees refine the ensemble's
classification score, which is converted to a binary-class probability.

| Setting | Configuration | Source |
| --- | --- | --- |
| Boosting rounds | 800 | Explicit |
| Learning rate | 0.03 | Explicit |
| Maximum tree depth | 6 | Explicit |
| Tree construction | hist | Explicit |
| Row sampling | subsample=0.8 | Explicit |
| Feature sampling per tree | colsample_bytree=0.8 | Explicit |
| Positive-class weight | scale_pos_weight=training normal/anomaly ratio | Explicit |
| Leaf update constraint | max_delta_step=3 | Explicit |
| Evaluation metric | logloss | Explicit |
| Objective | binary:logistic | Classifier default |
| L1/L2 regularization | reg_alpha and reg_lambda not overridden | Library defaults |
| Random seed | 42 | Explicit |
| CPU parallelism | n_jobs=-1 | Explicit |

Row and feature sampling introduce variation across trees. `max_delta_step`
limits the magnitude of leaf-weight updates; it is not a tree-depth parameter.
Other split constraints and regularization use the installed library defaults.
The fitted model is exported as `model.ubj`.

### LightGBM

LightGBM uses histogram-based, leaf-wise tree growth, expanding the leaf with
the greatest split gain rather than requiring a symmetric tree. Each tree is
limited to 31 leaves, while `max_depth=-1` imposes no explicit depth cap.

| Setting | Configuration | Source |
| --- | --- | --- |
| Boosting iterations | 800 | Explicit |
| Learning rate | 0.03 | Explicit |
| Maximum leaves per tree | 31 | Explicit |
| Maximum depth | -1 (no explicit cap) | Explicit |
| Boosting type | gbdt | Library default |
| Objective | Binary classification | Inferred by library for binary labels |
| Feature sampling per tree | colsample_bytree=0.8 | Explicit |
| Row sampling setting | subsample=0.8 | Explicit, but inactive |
| Row sampling frequency | subsample_freq=0 | Library default; disables row bagging |
| Class weights | Normal: 1; anomaly: training normal/anomaly ratio | Explicit |
| L1/L2 regularization | reg_alpha=0; reg_lambda=0 | Library defaults |
| Minimum samples per leaf | min_child_samples=20 | Library default |
| Random seed | 42 | Explicit |
| CPU parallelism | n_jobs=-1 | Explicit |

The configured 0.8 row-sampling fraction does not mean that 20% of rows are
discarded during training: bagging remains disabled because its frequency is 0.
Feature sampling is active. The fitted model is exported as `model.txt`.

### Decision Tree

Decision Tree uses a single entropy-based tree rather than a boosted ensemble.
Each observation follows sensor-value splits to a leaf; its anomaly probability
is the fraction of anomaly training samples in that leaf.

| Setting | Configuration | Source |
| --- | --- | --- |
| Classifier | sklearn.tree.DecisionTreeClassifier | Explicit |
| Split criterion | entropy | Explicit |
| Maximum depth | 10 | Explicit |
| Minimum samples to split a node | 10 | Explicit |
| Minimum samples per leaf | 5 | Explicit |
| Class weights | None; equal sample weights | Explicit |
| Random seed | 0 | Explicit |
| Split selection | best | Library default |
| Features considered per split | All ten (max_features=None) | Library default |
| Cost-complexity pruning | ccp_alpha=0.0 | Library default |
| Maximum leaf count | No explicit cap (max_leaf_nodes=None) | Library default |
| Learning rate / boosting rounds | Not applicable | Single-tree model |

The completed run `2026-11-17T01-36-20.047661Z` fitted a tree with **depth 4,
5 leaves and 9 nodes**. These are measured properties of that fit, not fixed
settings or a guarantee for another dataset. Its validation-selected threshold
was **1.0**; the implemented rule is probability >= threshold. A leaf probability
of 1.0 reflects the training samples in that leaf, not certainty on new data.

The script uses the same ten features, train-only median imputation and fixed
validation/test protocol as the other classifiers. It does not apply
StandardScaler or class weighting. No iterative training-loss curve is generated
for this model. Diagnostics include `tree_structure.png` (first four levels),
`tree_rules.txt`, `model_structure.json` and impurity-based feature importance.

Weights are saved as `model.pkl`, accompanied by `preprocessing.json`.
Pickle can execute code: only load trusted artifacts, with the recorded
scikit-learn version. The classification command requires `--trust-pickle`.
The [notebook review](decision_tree_review.md) documents the source experiment;
the results reported below come from the package's completed validation-based run.

### Shared Training and Evaluation

- Inputs: the ten numeric sensors listed above; target: normal versus combined Code 2/8 anomaly.
- Preprocessing: train-only median imputation; inputs are already engineering-scaled.
- Class weighting: boosted classifiers use normal weight 1 and anomaly weight equal to the training normal/anomaly ratio (6.333333 here); Decision Tree uses no class weighting.
- Training: boosted classifiers use seed 42 and 800 rounds; Decision Tree uses seed 0 and a single fit. None uses validation data to fit trees or for early stopping.
- Threshold: selected using validation anomaly F1 under the specified FPR cap, then fixed for testing.
- Evaluation: report precision, recall, F1, FPR, confusion matrices and computation times.

Boosted-model training log-loss curves use training data only. Library defaults should be
interpreted with the pinned dependency versions, not assumed constant across
software upgrades. The model artifact records the export environment versions.

## Package Contents

| File or directory | Purpose |
| --- | --- |
| train_catboost.py | Train, evaluate and export CatBoost |
| train_xgboost.py | Train, evaluate and export XGBoost |
| train_lightgbm.py | Train, evaluate and export LightGBM |
| train_decision_tree.py | Train, evaluate and export Decision Tree |
| model_artifacts.py | Export native weights and preprocessing metadata |
| predict.py | Classify a CSV using any supported exported model |
| compare_results.py | Generate comparison tables from three or four result directories |
| requirements.txt | Pinned dependencies |
| test_artifacts.py | Small model save/reload and inference checks |
| data/ | Default dataset directories |
| reference_results/ | Supplied performance comparison and source identifiers |
| runs/ | Results and model artifacts created during training |

Keep the Python files together: training and classification import
`model_artifacts.py`.

## Installation

Open a terminal in the package directory. The packaging environment uses Python
3.14.3; dependencies are pinned in requirements.txt.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS, activate with `source .venv/bin/activate` instead.
The scripts run on CPU. Training and evaluation arrays are loaded into memory,
so RAM requirements depend on dataset size. Run models sequentially to avoid
competing for CPU and memory.

## Dataset Requirements

Place the separately supplied, preprocessed CSVs in:

```text
data/
  train/
  validation/
  test/
```

The default filename pattern is `plc_log_????????.csv`; subfolders are not
searched. Training and evaluation require the ten sensors plus Alarm_Code.

Inputs must already have signed-INT16 decoding and engineering-unit scaling
applied. **Do not decode or scale them again.** Scaled values may contain decimals
and should not be truncated to integers. Missing or nonfinite sensor values are
imputed with medians learned from training data only.

The supplied comparison uses:

| Role | Dataset designation | Rows |
| --- | --- | ---: |
| Training | train(0701~0831) | 1,080,200 |
| Validation | October_copy | 13,338,574 |
| Test | July | 7,849,549 |

The training source is generated data; validation and testing use cleaned
machine logs. The July designation includes August and September files.
October_copy excludes normal rows simultaneously meeting Alarm_Code=0,
Weaving_Run=100, Infeed_Speed_SV=0, Infeed_Speed_PV=0 and Weaving_Line_Speed=0.
The trainers apply no additional stopped-machine filtering.

Keep files and observations disjoint across splits. Cleaning and dataset
selection define the population to which reported results apply.

## Training and Evaluation

With the default directory layout, run:

```powershell
python train_catboost.py
python train_xgboost.py
python train_lightgbm.py
python train_decision_tree.py
```

For alternative locations, specify paths explicitly:

```powershell
python train_lightgbm.py --train-dir "data/train" --validation-dir "data/validation" --test-dir "data/test" --output-dir "runs/lightgbm"
```

The same options work for all four trainers. Defaults are relative to the
package directory; explicit relative paths resolve from the terminal's working
directory.

1. Fit the classifier and imputation values using training data only.
2. Select the threshold maximizing validation anomaly F1 under an FPR cap of 0.02 (2%).
3. Apply that fixed threshold to testing data.
4. Save metrics, plots and the fitted model with its preprocessing settings.

Use `--fpr-cap 0.02` to specify the cap. This constrains validation threshold
selection, not the false-positive rate on every future dataset.

## Results and Model Files

Run and plot folders use a Windows-safe ISO-style UTC timestamp, for example
`2026-11-17T03-55-08.544747Z`. The date is year-month-day, the time is
hour-minute-second, the fractional digits preserve uniqueness, and `Z` means UTC.
Korea Standard Time is UTC+9. Timestamp changes do not change the recorded instant.
Model names remain the parent folders: `runs/lightgbm/<timestamp>/artifact`.
`run_name_migration.json` maps the original compact names to the readable names.
References inside this package use the readable names; external commands or
documents pointing to the old paths must be updated too.

Each execution creates `runs/<model>/<timestamp>/` containing:

- Precision, recall, anomaly F1, FPR and confusion matrices.
- Per-file and per-alarm-code reports.
- Feature importance and probability samples for all classifiers; training log-loss curves for boosted models, tree diagrams and rules for Decision Tree.
- Data-loading, training, classification and threshold-selection timings.
- An `artifact/` directory containing the fitted model and preprocessing.json.

| Model | Saved model file |
| --- | --- |
| CatBoost | model.cbm |
| XGBoost | model.ubj |
| LightGBM | model.txt |
| Decision Tree | model.pkl (trusted-source pickle only) |

The JSON stores ordered features, training medians, threshold, label mapping,
library versions and model checksum. **Distribute both artifact files together**,
alongside metrics from that same execution. Weights are generated when training
runs; source code alone is not a fitted model.

Training loss describes the training data and does not establish generalization.
Reported total computation time excludes final model export. Timing comparisons
depend on hardware and concurrent workload.

## Classifying New Observations

```powershell
python predict.py --model-dir "runs/lightgbm/RUN_ID/artifact" --input "sensor_data.csv" --output "predictions.csv"
```

Replace RUN_ID with the actual directory name. To use another classifier, point
`--model-dir` to its artifact folder; no code modification is needed. For a trusted
Decision Tree artifact, also supply `--trust-pickle`:

```powershell
python predict.py --model-dir "runs/decision_tree/RUN_ID/artifact" --input "sensor_data.csv" --output "classification_results.csv" --trust-pickle
```

Classification requires the ten preprocessed sensors but not Alarm_Code. The output
retains input columns and adds anomaly_probability, predicted_binary and
predicted_label. Probability greater than or equal to the stored threshold
produces an anomaly classification. Classification does not filter operating states.
The script refuses to overwrite the input or an existing output file.

## Performance Comparison

Reference measurements for the dataset assignment above are:

| Model | Validation F1 (%) | Test precision (%) | Test recall (%) | Test F1 (%) | Test FPR (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CatBoost | 84.5353 | 99.6274 | 72.6344 | 84.0160 | 0.005564 |
| XGBoost | 84.2696 | 99.6171 | 67.3694 | 80.3795 | 0.005304 |
| LightGBM | 84.2591 | 99.4305 | 85.0981 | 91.7077 | 0.009984 |
| Decision Tree | 83.0034 | 99.4376 | 85.1710 | 91.7531 | 0.009867 |

LightGBM and Decision Tree exceed the 85% test F1 target, with test FPR below
5%, for this evaluation population. Decision Tree has a slightly higher test F1;
LightGBM has a higher validation F1. Neither result establishes performance on
every operating state or a new dataset.

### Computation and Source Runs

| Model | Training (s) | Test classification (s) | Total execution (s) | Run ID |
| --- | ---: | ---: | ---: | --- |
| CatBoost | 112.67 | 0.6022 | 201.83 | 2026-11-16T06-16-44.740518Z |
| XGBoost | 13.30 | 3.9889 | 74.49 | 2026-11-16T06-20-24.236911Z |
| LightGBM | 7.25 | 13.5672 | 101.11 | 2026-11-16T06-21-47.353520Z |
| Decision Tree | 0.42 | 0.0659 | 63.12 | 2026-11-17T01-36-20.047661Z |

Each source report is at `runs/<model>/<run ID>/metrics.json`, using
`decision_tree` as the Decision Tree directory name. Test classification times
cover scoring all 7,849,549 test rows, excluding loading. Total time excludes
final model export. Hardware and workload affect timings. The supplied
`reference_results/` tables cover the three boosted classifiers; the command
below generates a four-model table from the specified completed runs.

To generate a comparison for exported models:

```powershell
python compare_results.py --catboost "runs/catboost/CATBOOST_RUN_ID" --xgboost "runs/xgboost/XGBOOST_RUN_ID" --lightgbm "runs/lightgbm/LIGHTGBM_RUN_ID" --decision-tree "runs/decision_tree/DECISION_TREE_RUN_ID" --output-dir "comparison"
```

Replace each identifier with its actual directory name. Compare models using the
same datasets and preprocessing. Outputs are comparison.csv and comparison.md,
including FPR percentages and computation times.

## Verification and Evaluation Scope

### Representative Boosted-Tree Plots

```powershell
python plot_boosted_trees.py
```

This loads the latest saved CatBoost, XGBoost and LightGBM artifacts without
retraining. It saves PNG/PDF figures and full selected-tree JSON under each
run's `tree_plots/<timestamp>/`. Explicit `--catboost`, `--xgboost` and
`--lightgbm` run directories can override automatic selection.

The default is the first tree, labelled with its position and actual ensemble
size, displayed through depth 3. Use `--tree-index 9` for the tenth tree, or
`--max-depth 4` to show another level. Deeper branches are explicitly marked as
omitted. Individual trees are components, not the complete classifiers; their
leaf scores are not final anomaly probabilities. Native JSON preserves exact
split values while figure labels are rounded for readability. No Graphviz
installation is required.

```powershell
python test_artifacts.py
```

These small checks verify model save/reload consistency, feature ordering,
imputation and checksum validation without processing project datasets.
