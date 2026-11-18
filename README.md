# Weaving Machine Monitoring and Alarm Analysis

This project investigates how weaving-machine PLC sensor data can support
machine monitoring, yarn-break alarm analysis, and data-driven anomaly detection.
It brings together data preparation, signal analysis, machine-learning
classification, and temporal reconstruction and forecasting experiments.

The work includes machine-collected logs and generated datasets, used to study
different operating conditions, alarm patterns, and model behavior.

## Project Goals

- Prepare consistent, traceable sensor datasets from PLC records.
- Examine machine behavior before, during, and after alarm events.
- Identify useful signals and distinguish alarm behavior from normal operation.
- Compare classification and sequence-based anomaly detection methods.
- Evaluate detection quality, false alarms, and computational requirements.
- Provide reproducible code and reusable models for technical review and
  certification preparation.

The project performance target is an anomaly F1 score of at least 85% on an
agreed evaluation dataset. Results apply to their documented data and protocols,
not automatically to every machine or operating condition.

## Data and Machine Signals

PLC logs contain timestamps, alarm codes, operating-state indicators, and sensor
measurements. Signals studied include:

| Signal group | Examples |
| --- | --- |
| Machine motion | Infeed speed, weaving line speed, motor RPM |
| Load and tension-related measurements | Infeed torque, loadcell readings |
| Dancer mechanism | Pressure and position |
| Environment | Temperature and humidity |
| Operating context | Machine run/stop state, alarm codes, timestamps |

The main alarm codes investigated are Code 2 (weft-break alarm) and Code 8
(warp-break alarm). Code 0 indicates no active alarm code; it does not necessarily
mean that the machine is running. Alarm codes and screening labels are
operational annotations, not independent confirmation of a physical fault.

Each experiment defines its own features, labels, and eligible observations.
Binary classifiers combine Codes 2 and 8, while event analysis can examine them
separately. Sequence models require contiguous observations and may evaluate
fewer records than row-based classifiers.

## Project Workflow

### Data Preparation

Preparation includes inspecting alarm-code distributions, interpreting PLC
register values, applying tag-specific engineering scaling, reviewing missing
values and timestamps, and standardizing labels and columns.

The preprocessing workflows use signed-INT16 decoding followed by scaling based
on the PLC tag list. Engineering-scaled values can contain decimals; they must
not be rounded back to integers or scaled a second time.

Cleaning rules depend on the dataset and experiment. Stopped-machine records,
zero-speed periods, and real/fake candidate labels are reviewed in context.
A stopped machine alone is not sufficient evidence that an alarm is false.

### Alarm and Waveform Analysis

Exploratory analysis examines alarm counts, event boundaries, operating states,
and sensor behavior around alarm onset. Pre-alarm waveform windows help inspect
level changes, trends, increased variation, and changes in direction or slope.

These analyses support dataset-quality checks and investigation of model
shortcuts, such as reliance on zero speed rather than an alarm-specific pattern.

### Model Development

Model development is a collaborative, ongoing part of the project. The examples
below cover approaches implemented or being explored by team members; they are
not an exhaustive list or a claim that every implementation is available in
this repository. Additional models can be evaluated under the same documented
experimental framework.

| Approach | Example models across the team | Purpose |
| --- | --- | --- |
| Sensor-record classification | CatBoost, XGBoost, LightGBM, Decision Tree | Classify the current observation as normal or anomalous |
| Sequence reconstruction | LSTM-AE, TCN-AE, CNN-AE | Reconstruct observed sensor windows and score reconstruction error |
| Sensor forecasting | GRU, TCN | Forecast subsequent sensor measurements and score forecast error |
| Other sequence-model experiments | BiLSTM and additional selected architectures | Investigate temporal patterns; define classification, reconstruction, or forecasting targets in each experiment |

Classification and forecasting are different tasks. The classifiers do not
predict future alarm onset. A forecast-error detector requires the forecasted
sensor measurement to become available before its error can be calculated.
Reconstruction and forecast-error results alone do not demonstrate advance
warning of an alarm.

### Evaluation and Interpretation

Experiments use documented training, validation, and test assignments.
Training fits models and preprocessing; validation supports model or threshold
selection; testing evaluates the fixed configuration.

Evaluation includes precision, recall, F1, false-positive rate, confusion
matrices, and computation times. Per-file and per-alarm-code reports reveal
differences that aggregate scores can hide. Sequence experiments also examine
loss curves, error distributions, and validation-derived thresholds.

Compare models using matching datasets, preprocessing, target definitions, and
evaluation units. Sampled sequence targets, individual rows, and complete alarm
events are not interchangeable. Repeatedly examined datasets provide retrospective
experimental evidence, not a new independent certification test.



## Contributing Model Experiments

Each team member should provide a self-contained model folder or notebook with:

- A brief description of the task, architecture, and implementation status.
- Configurable dataset paths rather than contributor-specific absolute paths.
- Dependencies and instructions for training, evaluation, and saved-model use.
- Dataset versions, cleaning rules, features, labels, and split assignments.
- Sequence length, stride, forecast horizon, and gap handling where applicable.
- Training settings, random seeds, and the validation-based selection procedure.
- Evaluation metrics, computation times, and the units used for reporting results.
- Saved model and preprocessing information when sharing a fitted implementation.

For direct comparisons, agree on a common evaluation protocol before comparing
results. Document necessary model-specific differences rather than assuming
all models use identical inputs or score the same observations. New contributions
can extend the model overview without replacing other team members' work.

## Using and Sharing the Work

1. Choose a workflow and read its local documentation.
2. Obtain the datasets and confirm their preprocessing state.
3. Set local or Google Drive paths in the relevant script or notebook.
4. Install that workflow's dependencies in an appropriate environment.
5. Record dataset assignments, settings, thresholds, and reports for each run.

Large datasets, model weights, and generated reports may be excluded from the
GitHub checkout. Share machine data only with appropriate
authorization.

When distributing a fitted model, include its preprocessing settings, feature
order, decision threshold, environment requirements, and matching evaluation
reports. Software checks of saving and loading do not establish detection
accuracy or certification readiness.
