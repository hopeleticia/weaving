from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Iterator

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "dataset"
RESULT_DIR = PROJECT_ROOT / "results" / "mlp_ae_full_dataset"
MODEL_DIR = PROJECT_ROOT / "models"

CSV_PATTERN = "plc_log_*.csv"

TIMESTAMP_COLUMN = "logged_at"
ALARM_COLUMN = "Alarm_Code"

EMPTY_COLUMNS = {
    "tag27",
    "tag28",
    "tag29",
    "tag30",
}

# Model input-এ timestamp ও weak label যাবে না
EXCLUDED_COLUMNS = {
    TIMESTAMP_COLUMN,
    ALARM_COLUMN,
    *EMPTY_COLUMNS,
}

# PLC unsigned 16-bit register থেকে signed value conversion
SIGNED_UINT16_COLUMNS = {
    "Dancer_Gyro_Position",
    "Motor_RPM",
}

DECODE_SIGNED_UINT16 = True

CSV_CHUNK_SIZE = 100_000
BATCH_SIZE = 2048
EPOCHS = 15

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 4

# Validation normal error-এর percentile
ANOMALY_PERCENTILE = 99.0

# Training-এ শুধু Alarm_Code == 0 rows ব্যবহার হবে
TRAIN_ONLY_NORMAL = True

# বড় dataset হওয়ায় প্রতি Nতম row ব্যবহার করা যায়।
# 1 দিলে সব row; 2 দিলে প্রতি দ্বিতীয় row।
TRAIN_ROW_STEP = 1
VALIDATION_ROW_STEP = 1
TEST_ROW_STEP = 1

NUM_WORKERS = 0
RANDOM_SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# File discovery and split
# ============================================================

def discover_files() -> list[Path]:
    files = sorted(DATA_DIR.glob(CSV_PATTERN))

    if not files:
        raise FileNotFoundError(
            f"কোনো CSV পাওয়া যায়নি: {DATA_DIR.resolve()}"
        )

    if len(files) < 3:
        raise ValueError(
            "Train, validation এবং test split-এর জন্য "
            "কমপক্ষে ৩টি CSV প্রয়োজন।"
        )

    return files


def chronological_split(
    files: list[Path],
) -> tuple[list[Path], list[Path], list[Path]]:

    if len(files) == 8:
        return files[:6], files[6:7], files[7:]

    train_end = max(1, int(len(files) * 0.75))
    validation_end = max(
        train_end + 1,
        int(len(files) * 0.875),
    )
    validation_end = min(
        validation_end,
        len(files) - 1,
    )

    train_files = files[:train_end]
    validation_files = files[
        train_end:validation_end
    ]
    test_files = files[validation_end:]

    if not validation_files or not test_files:
        raise ValueError("Chronological split তৈরি হয়নি।")

    return train_files, validation_files, test_files


# ============================================================
# PLC preprocessing
# ============================================================

def decode_signed_uint16(
    series: pd.Series,
) -> pd.Series:

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(np.float64)

    decoded = np.where(
        values > 32767,
        values - 65536,
        values,
    )

    return pd.Series(
        decoded,
        index=series.index,
        dtype=np.float64,
    )


def identify_candidate_features(
    first_file: Path,
) -> list[str]:

    header = pd.read_csv(
        first_file,
        nrows=5,
        low_memory=False,
    )

    features = [
        column
        for column in header.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if not features:
        raise ValueError("কোনো feature পাওয়া যায়নি।")

    return features


def preprocess_chunk(
    chunk: pd.DataFrame,
    candidate_features: list[str],
    row_step: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:

    if row_step > 1:
        chunk = chunk.iloc[::row_step].copy()

    required_columns = (
        candidate_features
        + [TIMESTAMP_COLUMN, ALARM_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in chunk.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    timestamps = pd.to_datetime(
        chunk[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    alarm_codes = pd.to_numeric(
        chunk[ALARM_COLUMN],
        errors="coerce",
    ).fillna(0)

    feature_df = chunk[
        candidate_features
    ].copy()

    for column in candidate_features:
        if (
            DECODE_SIGNED_UINT16
            and column in SIGNED_UINT16_COLUMNS
        ):
            feature_df[column] = decode_signed_uint16(
                feature_df[column]
            )
        else:
            feature_df[column] = pd.to_numeric(
                feature_df[column],
                errors="coerce",
            )

    feature_df = feature_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # ছোট missing region interpolate
    feature_df = feature_df.interpolate(
        method="linear",
        axis=0,
        limit_direction="both",
    )

    feature_df = feature_df.fillna(
        feature_df.median(numeric_only=True)
    )

    feature_df = feature_df.fillna(0.0)

    valid_mask = timestamps.notna()

    feature_df = feature_df.loc[
        valid_mask
    ].reset_index(drop=True)

    timestamps = timestamps.loc[
        valid_mask
    ].reset_index(drop=True)

    alarm_codes = alarm_codes.loc[
        valid_mask
    ].reset_index(drop=True)

    return feature_df, timestamps, alarm_codes


# ============================================================
# Scaler
# ============================================================

def fit_scaler(
    train_files: list[Path],
    candidate_features: list[str],
) -> tuple[StandardScaler, list[str], list[str]]:

    print()
    print("=" * 78)
    print("Analyzing training features")
    print("=" * 78)

    first_scaler = StandardScaler()
    fitted = False
    total_rows = 0

    for file_path in train_files:
        print(f"Scanning: {file_path.name}")

        for chunk in pd.read_csv(
            file_path,
            chunksize=CSV_CHUNK_SIZE,
            low_memory=False,
        ):
            features, _, alarms = preprocess_chunk(
                chunk,
                candidate_features,
                TRAIN_ROW_STEP,
            )

            if TRAIN_ONLY_NORMAL:
                normal_mask = (
                    alarms.to_numpy() == 0
                )
                features = features.loc[normal_mask]

            if features.empty:
                continue

            first_scaler.partial_fit(
                features.to_numpy(dtype=np.float64)
            )

            total_rows += len(features)
            fitted = True

    if not fitted:
        raise RuntimeError(
            "Scaler fit করার data পাওয়া যায়নি।"
        )

    variances = np.asarray(
        first_scaler.var_,
        dtype=np.float64,
    )

    retained_features = [
        feature
        for feature, variance
        in zip(candidate_features, variances)
        if np.isfinite(variance) and variance > 1e-10
    ]

    removed_features = [
        feature
        for feature, variance
        in zip(candidate_features, variances)
        if not (
            np.isfinite(variance)
            and variance > 1e-10
        )
    ]

    if not retained_features:
        raise RuntimeError(
            "সব feature constant পাওয়া গেছে।"
        )

    print(f"Processed rows      : {total_rows:,}")
    print(f"Candidate features  : {len(candidate_features)}")
    print(f"Retained features   : {len(retained_features)}")
    print(f"Removed features    : {len(removed_features)}")

    if removed_features:
        print("\nConstant/invalid features removed:")
        for feature in removed_features:
            print(f"  - {feature}")

    # শুধু retained features দিয়ে final scaler
    final_scaler = StandardScaler()
    final_fitted = False

    for file_path in train_files:
        for chunk in pd.read_csv(
            file_path,
            chunksize=CSV_CHUNK_SIZE,
            low_memory=False,
        ):
            features, _, alarms = preprocess_chunk(
                chunk,
                candidate_features,
                TRAIN_ROW_STEP,
            )

            features = features[retained_features]

            if TRAIN_ONLY_NORMAL:
                normal_mask = (
                    alarms.to_numpy() == 0
                )
                features = features.loc[normal_mask]

            if features.empty:
                continue

            final_scaler.partial_fit(
                features.to_numpy(dtype=np.float64)
            )

            final_fitted = True

    if not final_fitted:
        raise RuntimeError("Final scaler fit হয়নি।")

    return (
        final_scaler,
        retained_features,
        removed_features,
    )


# ============================================================
# Streaming tabular dataset
# ============================================================

class PLCTabularDataset(IterableDataset):

    def __init__(
        self,
        files: list[Path],
        candidate_features: list[str],
        retained_features: list[str],
        scaler: StandardScaler,
        normal_only: bool,
        row_step: int,
    ) -> None:
        super().__init__()

        self.files = files
        self.candidate_features = candidate_features
        self.retained_features = retained_features
        self.scaler = scaler
        self.normal_only = normal_only
        self.row_step = row_step

    def __iter__(
        self,
    ) -> Iterator[dict[str, torch.Tensor]]:

        for file_path in self.files:
            for chunk in pd.read_csv(
                file_path,
                chunksize=CSV_CHUNK_SIZE,
                low_memory=False,
            ):
                features, timestamps, alarms = (
                    preprocess_chunk(
                        chunk,
                        self.candidate_features,
                        self.row_step,
                    )
                )

                features = features[
                    self.retained_features
                ]

                labels = (
                    alarms.to_numpy() != 0
                ).astype(np.int8)

                if self.normal_only:
                    keep_mask = labels == 0

                    features = features.loc[
                        keep_mask
                    ].reset_index(drop=True)

                    timestamps = timestamps.loc[
                        keep_mask
                    ].reset_index(drop=True)

                    labels = labels[keep_mask]

                if features.empty:
                    continue

                scaled_features = self.scaler.transform(
                    features.to_numpy(dtype=np.float64)
                ).astype(np.float32)

                timestamp_ns = (
                    timestamps.astype("int64")
                    .to_numpy(dtype=np.int64)
                )

                for index in range(
                    len(scaled_features)
                ):
                    yield {
                        "x": torch.from_numpy(
                            scaled_features[index].copy()
                        ),
                        "timestamp_ns": torch.tensor(
                            timestamp_ns[index],
                            dtype=torch.int64,
                        ),
                        "label": torch.tensor(
                            labels[index],
                            dtype=torch.int64,
                        ),
                    }


def create_loader(
    dataset: PLCTabularDataset,
    batch_size: int,
) -> DataLoader:

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ============================================================
# MLP Autoencoder
# ============================================================

class MLPAutoencoder(nn.Module):

    def __init__(
        self,
        input_dimension: int,
    ) -> None:
        super().__init__()

        hidden_1 = max(64, input_dimension * 4)
        hidden_2 = max(32, input_dimension * 2)
        latent_dimension = max(8, input_dimension // 2)

        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),

            nn.Linear(hidden_2, latent_dimension),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dimension, hidden_2),
            nn.ReLU(),

            nn.Linear(hidden_2, hidden_1),
            nn.ReLU(),

            nn.Linear(hidden_1, input_dimension),
        )

        self.input_dimension = input_dimension
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.latent_dimension = latent_dimension

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        latent = self.encoder(x)
        reconstruction = self.decoder(latent)

        return reconstruction


# ============================================================
# Train/evaluate
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_fp16: bool,
    scaler: torch.amp.GradScaler,
) -> tuple[float, int]:

    model.train()

    total_loss = 0.0
    batch_count = 0
    sample_count = 0

    progress_bar = tqdm(
        loader,
        desc="Training",
        unit="batch",
    )

    for batch in progress_bar:
        x = batch["x"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_fp16,
        ):
            reconstruction = model(x)
            loss = criterion(reconstruction, x)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        batch_count += 1
        sample_count += x.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    mean_loss = (
        total_loss / batch_count
        if batch_count > 0
        else float("nan")
    )

    return mean_loss, sample_count


@torch.inference_mode()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_fp16: bool,
) -> tuple[float, int]:

    model.eval()

    total_loss = 0.0
    batch_count = 0
    sample_count = 0

    for batch in tqdm(
        loader,
        desc="Validation",
        unit="batch",
    ):
        x = batch["x"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_fp16,
        ):
            reconstruction = model(x)
            loss = criterion(reconstruction, x)

        total_loss += float(loss.item())
        batch_count += 1
        sample_count += x.size(0)

    mean_loss = (
        total_loss / batch_count
        if batch_count > 0
        else float("nan")
    )

    return mean_loss, sample_count


@torch.inference_mode()
def collect_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> pd.DataFrame:

    model.eval()

    all_errors: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []

    for batch in tqdm(
        loader,
        desc="Scoring",
        unit="batch",
    ):
        x = batch["x"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_fp16,
        ):
            reconstruction = model(x)

            row_errors = torch.mean(
                (reconstruction - x) ** 2,
                dim=1,
            )

        all_errors.append(
            row_errors.float().cpu().numpy()
        )

        all_labels.append(
            batch["label"].cpu().numpy()
        )

        all_timestamps.append(
            batch["timestamp_ns"].cpu().numpy()
        )

    if not all_errors:
        raise RuntimeError(
            "Score করার sample পাওয়া যায়নি।"
        )

    return pd.DataFrame(
        {
            "logged_at": pd.to_datetime(
                np.concatenate(all_timestamps)
            ),
            "reconstruction_error": np.concatenate(
                all_errors
            ),
            "weak_label": np.concatenate(
                all_labels
            ).astype(np.int8),
        }
    )


# ============================================================
# Plot functions
# ============================================================

def save_training_history(
    history_df: pd.DataFrame,
) -> None:

    plt.figure(figsize=(10, 5))

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        marker="o",
        label="Train loss",
    )

    plt.plot(
        history_df["epoch"],
        history_df["validation_loss"],
        marker="o",
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction MSE")
    plt.title("MLP Autoencoder Training History")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "training_history.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    normalized: bool,
    output_name: str,
) -> None:

    plt.figure(figsize=(7, 6))

    plt.imshow(matrix)
    plt.title(
        "Normalized Confusion Matrix"
        if normalized
        else "Confusion Matrix"
    )

    plt.colorbar()

    class_names = ["Normal", "Anomaly"]

    tick_positions = np.arange(
        len(class_names)
    )

    plt.xticks(
        tick_positions,
        class_names,
    )

    plt.yticks(
        tick_positions,
        class_names,
    )

    threshold = (
        matrix.max() / 2.0
        if matrix.size > 0
        else 0
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if normalized:
                text = f"{matrix[row, column]:.3f}"
            else:
                text = f"{int(matrix[row, column]):,}"

            plt.text(
                column,
                row,
                text,
                horizontalalignment="center",
                verticalalignment="center",
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "black"
                ),
                fontsize=12,
            )

    plt.ylabel("Actual weak label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / output_name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    roc_auc: float,
) -> None:

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(y_true, scores)
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"MLP-AE AUC = {roc_auc:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random baseline",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    average_precision: float,
) -> None:

    precision_values, recall_values, _ = (
        precision_recall_curve(
            y_true,
            scores,
        )
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        recall_values,
        precision_values,
        label=f"AP = {average_precision:.4f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_anomaly_score_plot(
    test_scores: pd.DataFrame,
    threshold: float,
) -> None:

    maximum_points = 25_000

    if len(test_scores) > maximum_points:
        step = int(
            np.ceil(
                len(test_scores)
                / maximum_points
            )
        )

        plot_df = test_scores.iloc[
            ::step
        ].copy()
    else:
        plot_df = test_scores.copy()

    plt.figure(figsize=(16, 6))

    plt.plot(
        plot_df["logged_at"],
        plot_df["reconstruction_error"],
        linewidth=0.7,
        label="Reconstruction error",
    )

    plt.axhline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.6f}",
    )

    anomaly_rows = plot_df[
        plot_df["predicted_anomaly"] == 1
    ]

    if not anomaly_rows.empty:
        plt.scatter(
            anomaly_rows["logged_at"],
            anomaly_rows["reconstruction_error"],
            marker="x",
            s=16,
            label="Detected anomaly",
        )

    plt.xlabel("Time")
    plt.ylabel("Reconstruction MSE")
    plt.title("MLP-AE Anomaly Scores on Test Dataset")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "test_anomaly_scores.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_fp16_weights(
    model: nn.Module,
    output_path: Path,
) -> None:

    fp16_state = {}

    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu()

        if tensor.is_floating_point():
            tensor = tensor.half()

        fp16_state[name] = tensor

    torch.save(fp16_state, output_path)


# ============================================================
# Main
# ============================================================

def main() -> None:

    set_seed(RANDOM_SEED)

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Files and split
    # --------------------------------------------------------

    all_files = discover_files()

    train_files, validation_files, test_files = (
        chronological_split(all_files)
    )

    print("=" * 78)
    print("MLP Autoencoder Full Dataset Experiment")
    print("=" * 78)

    print("\nTrain files:")
    for file_path in train_files:
        print(f"  - {file_path.name}")

    print("\nValidation files:")
    for file_path in validation_files:
        print(f"  - {file_path.name}")

    print("\nTest files:")
    for file_path in test_files:
        print(f"  - {file_path.name}")

    split_data = {
        "train_files": [
            path.name for path in train_files
        ],
        "validation_files": [
            path.name for path in validation_files
        ],
        "test_files": [
            path.name for path in test_files
        ],
    }

    with open(
        RESULT_DIR / "data_split.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_data,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Features and scaler
    # --------------------------------------------------------

    candidate_features = identify_candidate_features(
        all_files[0]
    )

    scaler, retained_features, removed_features = (
        fit_scaler(
            train_files,
            candidate_features,
        )
    )

    joblib.dump(
        scaler,
        RESULT_DIR / "standard_scaler.joblib",
    )

    with open(
        RESULT_DIR / "retained_features.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            retained_features,
            file,
            indent=2,
        )

    with open(
        RESULT_DIR / "removed_features.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            removed_features,
            file,
            indent=2,
        )

    print("\nRetained features:")

    for feature in retained_features:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # Dataset and loader
    # --------------------------------------------------------

    train_dataset = PLCTabularDataset(
        files=train_files,
        candidate_features=candidate_features,
        retained_features=retained_features,
        scaler=scaler,
        normal_only=True,
        row_step=TRAIN_ROW_STEP,
    )

    validation_dataset = PLCTabularDataset(
        files=validation_files,
        candidate_features=candidate_features,
        retained_features=retained_features,
        scaler=scaler,
        normal_only=True,
        row_step=VALIDATION_ROW_STEP,
    )

    train_loader = create_loader(
        train_dataset,
        BATCH_SIZE,
    )

    validation_loader = create_loader(
        validation_dataset,
        BATCH_SIZE,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_fp16 = device.type == "cuda"

    print()
    print("=" * 78)
    print("Training configuration")
    print("=" * 78)
    print(f"Device              : {device}")
    print(f"FP16 AMP enabled    : {use_fp16}")
    print(f"Input features      : {len(retained_features)}")
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Maximum epochs      : {EPOCHS}")
    print(f"Threshold percentile: {ANOMALY_PERCENTILE}")

    if not use_fp16:
        print(
            "\nWARNING: CUDA GPU পাওয়া যায়নি। "
            "Training CPU FP32-তে চলবে।"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MLPAutoencoder(
        input_dimension=len(retained_features)
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    amp_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_fp16,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Hidden layer 1      : {model.hidden_1}")
    print(f"Hidden layer 2      : {model.hidden_2}")
    print(f"Latent dimension   : {model.latent_dimension}")
    print(f"Trainable parameters: {parameter_count:,}")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history: list[dict] = []

    best_validation_loss = float("inf")
    best_model_state = None
    no_improvement_count = 0

    training_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):

        print()
        print("=" * 78)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("=" * 78)

        epoch_start = time.perf_counter()

        train_loss, train_samples = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_fp16=use_fp16,
            scaler=amp_scaler,
        )

        validation_loss, validation_samples = (
            evaluate_loss(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                use_fp16=use_fp16,
            )
        )

        epoch_time = (
            time.perf_counter() - epoch_start
        )

        print(f"Train loss         : {train_loss:.8f}")
        print(f"Validation loss    : {validation_loss:.8f}")
        print(f"Train samples      : {train_samples:,}")
        print(f"Validation samples : {validation_samples:,}")
        print(f"Epoch time         : {epoch_time:.2f} s")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_samples": train_samples,
                "validation_samples": validation_samples,
                "epoch_time_seconds": epoch_time,
            }
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            no_improvement_count = 0

            print("Best model updated.")

        else:
            no_improvement_count += 1

            print(
                "No improvement: "
                f"{no_improvement_count}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        if (
            no_improvement_count
            >= EARLY_STOPPING_PATIENCE
        ):
            print("Early stopping activated.")
            break

    total_training_time = (
        time.perf_counter() - training_start
    )

    if best_model_state is None:
        raise RuntimeError(
            "Best model তৈরি হয়নি।"
        )

    model.load_state_dict(best_model_state)

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        RESULT_DIR / "training_history.csv",
        index=False,
    )

    save_training_history(history_df)

    # --------------------------------------------------------
    # Model save
    # --------------------------------------------------------

    checkpoint_path = (
        MODEL_DIR
        / "mlp_ae_full_dataset_checkpoint.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dimension": len(retained_features),
            "retained_features": retained_features,
            "hidden_1": model.hidden_1,
            "hidden_2": model.hidden_2,
            "latent_dimension": model.latent_dimension,
            "best_validation_loss": (
                best_validation_loss
            ),
            "fp16_amp_training": use_fp16,
        },
        checkpoint_path,
    )

    fp16_path = (
        MODEL_DIR
        / "mlp_ae_full_dataset_fp16_weights.pt"
    )

    save_fp16_weights(
        model,
        fp16_path,
    )

    # --------------------------------------------------------
    # Validation threshold
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("Calculating threshold from validation normal data")
    print("=" * 78)

    threshold_dataset = PLCTabularDataset(
        files=validation_files,
        candidate_features=candidate_features,
        retained_features=retained_features,
        scaler=scaler,
        normal_only=True,
        row_step=VALIDATION_ROW_STEP,
    )

    threshold_loader = create_loader(
        threshold_dataset,
        BATCH_SIZE,
    )

    validation_scores = collect_scores(
        model=model,
        loader=threshold_loader,
        device=device,
        use_fp16=use_fp16,
    )

    threshold = float(
        np.percentile(
            validation_scores[
                "reconstruction_error"
            ],
            ANOMALY_PERCENTILE,
        )
    )

    validation_scores.to_csv(
        RESULT_DIR
        / "validation_reconstruction_scores.csv",
        index=False,
    )

    print(f"Threshold: {threshold:.8f}")

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("Evaluating complete test file")
    print("=" * 78)

    test_dataset = PLCTabularDataset(
        files=test_files,
        candidate_features=candidate_features,
        retained_features=retained_features,
        scaler=scaler,
        normal_only=False,
        row_step=TEST_ROW_STEP,
    )

    test_loader = create_loader(
        test_dataset,
        BATCH_SIZE,
    )

    test_start = time.perf_counter()

    test_scores = collect_scores(
        model=model,
        loader=test_loader,
        device=device,
        use_fp16=use_fp16,
    )

    test_time = (
        time.perf_counter() - test_start
    )

    test_scores["anomaly_threshold"] = threshold

    test_scores["predicted_anomaly"] = (
        test_scores["reconstruction_error"]
        > threshold
    ).astype(np.int8)

    test_scores.to_csv(
        RESULT_DIR / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    y_true = test_scores[
        "weak_label"
    ].to_numpy(dtype=np.int8)

    y_pred = test_scores[
        "predicted_anomaly"
    ].to_numpy(dtype=np.int8)

    anomaly_scores = test_scores[
        "reconstruction_error"
    ].to_numpy(dtype=np.float64)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else float("nan")
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else float("nan")
    )

    balanced_accuracy = (
        (recall + specificity) / 2
    )

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(
            y_true,
            anomaly_scores,
        )

        average_precision = average_precision_score(
            y_true,
            anomaly_scores,
        )
    else:
        roc_auc = float("nan")
        average_precision = float("nan")

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Anomaly"],
        digits=6,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Anomaly"],
        output_dict=True,
        zero_division=0,
    )

    with open(
        RESULT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report_text)

    pd.DataFrame(report_dict).transpose().to_csv(
        RESULT_DIR / "classification_report.csv"
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=["Actual_Normal", "Actual_Anomaly"],
        columns=[
            "Predicted_Normal",
            "Predicted_Anomaly",
        ],
    )

    confusion_df.to_csv(
        RESULT_DIR / "confusion_matrix.csv"
    )

    normalized_matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
        normalize="true",
    )

    save_confusion_matrix_plot(
        matrix,
        normalized=False,
        output_name="confusion_matrix.png",
    )

    save_confusion_matrix_plot(
        normalized_matrix,
        normalized=True,
        output_name=(
            "confusion_matrix_normalized.png"
        ),
    )

    if len(np.unique(y_true)) == 2:
        save_roc_curve(
            y_true,
            anomaly_scores,
            roc_auc,
        )

        save_pr_curve(
            y_true,
            anomaly_scores,
            average_precision,
        )

    save_anomaly_score_plot(
        test_scores,
        threshold,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    test_samples = len(test_scores)

    throughput = (
        test_samples / test_time
        if test_time > 0
        else float("inf")
    )

    detected_anomalies = int(
        y_pred.sum()
    )

    actual_weak_anomalies = int(
        y_true.sum()
    )

    checkpoint_size_mb = (
        checkpoint_path.stat().st_size
        / (1024 ** 2)
    )

    fp16_size_mb = (
        fp16_path.stat().st_size
        / (1024 ** 2)
    )

    summary = {
        "model": "Tabular MLP Autoencoder",
        "label_type": "Alarm_Code weak label",
        "precision_mode": (
            "CUDA AMP FP16"
            if use_fp16
            else "CPU FP32"
        ),
        "number_of_features": len(
            retained_features
        ),
        "train_files": len(train_files),
        "validation_files": len(
            validation_files
        ),
        "test_files": len(test_files),
        "epochs_completed": len(history_df),
        "parameters": parameter_count,
        "best_validation_loss": (
            best_validation_loss
        ),
        "threshold_percentile": (
            ANOMALY_PERCENTILE
        ),
        "anomaly_threshold": threshold,
        "training_time_seconds": (
            total_training_time
        ),
        "test_inference_time_seconds": (
            test_time
        ),
        "test_samples": test_samples,
        "throughput_samples_per_second": (
            throughput
        ),
        "actual_weak_anomalies": (
            actual_weak_anomalies
        ),
        "predicted_anomalies": (
            detected_anomalies
        ),
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1,
        "false_positive_rate": (
            false_positive_rate
        ),
        "false_negative_rate": (
            false_negative_rate
        ),
        "roc_auc": roc_auc,
        "pr_auc_average_precision": (
            average_precision
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "checkpoint_size_mb": (
            checkpoint_size_mb
        ),
        "fp16_weights_size_mb": (
            fp16_size_mb
        ),
    }

    pd.DataFrame([summary]).to_csv(
        RESULT_DIR / "summary.csv",
        index=False,
    )

    with open(
        RESULT_DIR / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Console result
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("MLP-AE experiment completed")
    print("=" * 78)

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(report_text)

    print(f"TN                    : {tn:,}")
    print(f"FP                    : {fp:,}")
    print(f"FN                    : {fn:,}")
    print(f"TP                    : {tp:,}")
    print(f"Accuracy              : {accuracy:.6f}")
    print(
        f"Balanced accuracy     : "
        f"{balanced_accuracy:.6f}"
    )
    print(f"Precision             : {precision:.6f}")
    print(f"Recall/Sensitivity    : {recall:.6f}")
    print(f"Specificity           : {specificity:.6f}")
    print(f"F1-score              : {f1:.6f}")
    print(
        f"False Positive Rate   : "
        f"{false_positive_rate:.6f}"
    )
    print(
        f"False Negative Rate   : "
        f"{false_negative_rate:.6f}"
    )
    print(f"ROC-AUC               : {roc_auc:.6f}")
    print(
        f"PR-AUC/AP             : "
        f"{average_precision:.6f}"
    )
    print(f"Threshold             : {threshold:.8f}")
    print(
        f"Training time         : "
        f"{total_training_time:.2f} s"
    )
    print(
        f"Test inference time   : "
        f"{test_time:.2f} s"
    )
    print(
        f"Throughput            : "
        f"{throughput:.2f} samples/s"
    )
    print(
        f"Checkpoint size       : "
        f"{checkpoint_size_mb:.3f} MB"
    )
    print(
        f"FP16 weights size     : "
        f"{fp16_size_mb:.3f} MB"
    )

    print()
    print(f"Results saved in: {RESULT_DIR}")
    print(f"Model saved in  : {checkpoint_path}")


if __name__ == "__main__":
    main()