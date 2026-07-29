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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "dataset"
RESULT_DIR = PROJECT_ROOT / "results" / "lstm_ae_full_dataset"
MODEL_DIR = PROJECT_ROOT / "models"

CSV_PATTERN = "plc_log_*.csv"

TIMESTAMP_COLUMN = "logged_at"
ALARM_COLUMN = "Alarm_Code"

# Empty এবং control/state columns model input-এ ব্যবহার করা হবে না।
EXCLUDED_COLUMNS = {
    TIMESTAMP_COLUMN,
    ALARM_COLUMN,
    "tag27",
    "tag28",
    "tag29",
    "tag30",
}

# PLC register যদি unsigned 16-bit হিসেবে save হয়ে থাকে,
# তাহলে কিছু signed sensor value decode করা প্রয়োজন।
#
# উদাহরণ:
# 65535 -> -1
# 65503 -> -33
#
# Dancer_Gyro_Position-এর sample values দেখে signed decoding যৌক্তিক।
# Motor_RPM-এ 65535 আসায় এটিকেও configurable রাখা হয়েছে।
SIGNED_UINT16_COLUMNS = {
    "Dancer_Gyro_Position",
    "Motor_RPM",
}

DECODE_SIGNED_UINT16 = True

# CSV streaming
CSV_CHUNK_SIZE = 100_000

# Sequence settings
SEQUENCE_LENGTH = 128

# প্রতি 32টি point পর একটি window:
# sampling প্রায় 9.9 Hz হলে overlap থাকবে, কিন্তু window সংখ্যা নিয়ন্ত্রিত থাকবে।
STRIDE = 32

# Model settings
HIDDEN_SIZE = 64
LATENT_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.0

# Training settings
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0
PATIENCE = 3

# Validation normal-window reconstruction error-এর percentile
ANOMALY_PERCENTILE = 99.0

# Alarm_Code != 0 window-কে training থেকে বাদ দেওয়া হবে।
# এতে known alarm period model-এর normal pattern হিসেবে শেখার ঝুঁকি কমবে।
TRAIN_ONLY_ALARM_FREE_WINDOWS = True

# Windows-এ DataLoader multiprocessing Windows-এ সমস্যা করতে পারে।
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
# File discovery and chronological split
# ============================================================

def discover_csv_files() -> list[Path]:
    files = sorted(DATA_DIR.glob(CSV_PATTERN))

    if not files:
        raise FileNotFoundError(
            f"কোনো CSV পাওয়া যায়নি: {DATA_DIR.resolve()}"
        )

    if len(files) < 3:
        raise ValueError(
            "Chronological train/validation/test split-এর জন্য "
            "কমপক্ষে তিনটি CSV প্রয়োজন।"
        )

    return files


def chronological_split(
    files: list[Path],
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    আটটি file থাকলে 6/1/1 split করবে।
    অন্য সংখ্যা হলে প্রায় 75%/12.5%/12.5% chronological split।
    """

    number_of_files = len(files)

    if number_of_files == 8:
        train_files = files[:6]
        validation_files = files[6:7]
        test_files = files[7:]
        return train_files, validation_files, test_files

    train_end = max(1, int(number_of_files * 0.75))
    validation_end = max(
        train_end + 1,
        int(number_of_files * 0.875),
    )

    validation_end = min(validation_end, number_of_files - 1)

    train_files = files[:train_end]
    validation_files = files[train_end:validation_end]
    test_files = files[validation_end:]

    if not validation_files or not test_files:
        raise ValueError("Dataset split সঠিকভাবে তৈরি হয়নি।")

    return train_files, validation_files, test_files


# ============================================================
# PLC preprocessing
# ============================================================

def decode_signed_uint16(
    values: pd.Series,
) -> pd.Series:
    """
    Unsigned 16-bit PLC register-কে signed integer-এ convert করে।

    0 ... 32767      -> অপরিবর্তিত
    32768 ... 65535  -> value - 65536
    """

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    ).astype(np.float64)

    signed_values = np.where(
        numeric_values > 32767,
        numeric_values - 65536,
        numeric_values,
    )

    return pd.Series(
        signed_values,
        index=values.index,
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

    candidate_features = [
        column
        for column in header.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if not candidate_features:
        raise ValueError("কোনো candidate feature পাওয়া যায়নি।")

    return candidate_features


def preprocess_chunk(
    chunk: pd.DataFrame,
    candidate_features: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns:
        feature_df
        timestamps
        alarm_codes
    """

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
            f"CSV-তে columns নেই: {missing_columns}"
        )

    timestamps = pd.to_datetime(
        chunk[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    alarm_codes = pd.to_numeric(
        chunk[ALARM_COLUMN],
        errors="coerce",
    ).fillna(0)

    feature_df = chunk[candidate_features].copy()

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

    # Short missing region interpolation
    feature_df = feature_df.interpolate(
        method="linear",
        axis=0,
        limit_direction="both",
    )

    # কোনো value তবুও missing থাকলে column median
    column_medians = feature_df.median(numeric_only=True)
    feature_df = feature_df.fillna(column_medians)

    # সম্পূর্ণ empty column থাকলে 0
    feature_df = feature_df.fillna(0.0)

    valid_timestamp_mask = timestamps.notna()

    feature_df = feature_df.loc[
        valid_timestamp_mask
    ].reset_index(drop=True)

    timestamps = timestamps.loc[
        valid_timestamp_mask
    ].reset_index(drop=True)

    alarm_codes = alarm_codes.loc[
        valid_timestamp_mask
    ].reset_index(drop=True)

    return feature_df, timestamps, alarm_codes


# ============================================================
# Streaming scaler fitting
# ============================================================

def fit_scaler(
    train_files: list[Path],
    candidate_features: list[str],
) -> tuple[StandardScaler, list[str], np.ndarray]:
    """
    শুধু training files দিয়ে StandardScaler partial_fit করে।
    Constant বা near-constant features বাদ দেয়।
    """

    print()
    print("=" * 78)
    print("Fitting scaler on training files")
    print("=" * 78)

    initial_scaler = StandardScaler()
    fitted = False
    processed_rows = 0

    for file_path in train_files:
        print(f"Processing scaler file: {file_path.name}")

        for chunk in pd.read_csv(
            file_path,
            chunksize=CSV_CHUNK_SIZE,
            low_memory=False,
        ):
            features, _, alarm_codes = preprocess_chunk(
                chunk,
                candidate_features,
            )

            if TRAIN_ONLY_ALARM_FREE_WINDOWS:
                normal_mask = (
                    alarm_codes.to_numpy() == 0
                )
                features = features.loc[normal_mask]

            if features.empty:
                continue

            initial_scaler.partial_fit(
                features.to_numpy(dtype=np.float64)
            )

            processed_rows += len(features)
            fitted = True

    if not fitted:
        raise RuntimeError(
            "Scaler fit করার মতো valid training data পাওয়া যায়নি।"
        )

    variance = np.asarray(
        initial_scaler.var_,
        dtype=np.float64,
    )

    retained_mask = (
        np.isfinite(variance)
        & (variance > 1e-10)
    )

    retained_features = [
        feature
        for feature, keep
        in zip(candidate_features, retained_mask)
        if keep
    ]

    removed_features = [
        feature
        for feature, keep
        in zip(candidate_features, retained_mask)
        if not keep
    ]

    if not retained_features:
        raise RuntimeError(
            "সব feature constant পাওয়া গেছে।"
        )

    print(f"Scaler rows           : {processed_rows:,}")
    print(f"Candidate features    : {len(candidate_features)}")
    print(f"Retained features     : {len(retained_features)}")
    print(f"Removed constants     : {len(removed_features)}")

    if removed_features:
        print("Removed features:")
        for feature in removed_features:
            print(f"  - {feature}")

    # নতুন scaler শুধু retained features দিয়ে fit করা হবে।
    final_scaler = StandardScaler()
    final_fitted = False

    for file_path in train_files:
        for chunk in pd.read_csv(
            file_path,
            chunksize=CSV_CHUNK_SIZE,
            low_memory=False,
        ):
            features, _, alarm_codes = preprocess_chunk(
                chunk,
                candidate_features,
            )

            features = features[retained_features]

            if TRAIN_ONLY_ALARM_FREE_WINDOWS:
                normal_mask = (
                    alarm_codes.to_numpy() == 0
                )
                features = features.loc[normal_mask]

            if features.empty:
                continue

            final_scaler.partial_fit(
                features.to_numpy(dtype=np.float64)
            )
            final_fitted = True

    if not final_fitted:
        raise RuntimeError("Final scaler fit ব্যর্থ হয়েছে।")

    return final_scaler, retained_features, retained_mask


# ============================================================
# Streaming sequence dataset
# ============================================================

class PLCSequenceDataset(IterableDataset):
    """
    সব CSV memory-তে load না করে chunk-by-chunk sequence তৈরি করে।

    PyTorch IterableDataset large streaming data-এর জন্য উপযুক্ত,
    যেখানে random access ব্যয়বহুল বা অপ্রয়োজনীয়। 
    """

    def __init__(
        self,
        files: list[Path],
        candidate_features: list[str],
        retained_features: list[str],
        scaler: StandardScaler,
        sequence_length: int,
        stride: int,
        alarm_free_only: bool,
    ) -> None:
        super().__init__()

        self.files = files
        self.candidate_features = candidate_features
        self.retained_features = retained_features
        self.scaler = scaler
        self.sequence_length = sequence_length
        self.stride = stride
        self.alarm_free_only = alarm_free_only

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        carry_features: np.ndarray | None = None
        carry_timestamps: np.ndarray | None = None
        carry_alarms: np.ndarray | None = None

        for file_path in self.files:
            # File boundary-তে carry reset করি।
            # এক দিনের শেষ এবং পরের দিনের শুরু মিশিয়ে window তৈরি হবে না।
            carry_features = None
            carry_timestamps = None
            carry_alarms = None

            for chunk in pd.read_csv(
                file_path,
                chunksize=CSV_CHUNK_SIZE,
                low_memory=False,
            ):
                features, timestamps, alarms = preprocess_chunk(
                    chunk,
                    self.candidate_features,
                )

                features = features[self.retained_features]

                feature_array = self.scaler.transform(
                    features.to_numpy(dtype=np.float64)
                ).astype(np.float32)

                timestamp_array = (
                    timestamps.astype("int64")
                    .to_numpy(dtype=np.int64)
                )

                alarm_array = alarms.to_numpy(
                    dtype=np.float32
                )

                if carry_features is not None:
                    feature_array = np.concatenate(
                        [carry_features, feature_array],
                        axis=0,
                    )

                    timestamp_array = np.concatenate(
                        [carry_timestamps, timestamp_array],
                        axis=0,
                    )

                    alarm_array = np.concatenate(
                        [carry_alarms, alarm_array],
                        axis=0,
                    )

                total_rows = len(feature_array)

                if total_rows < self.sequence_length:
                    carry_features = feature_array
                    carry_timestamps = timestamp_array
                    carry_alarms = alarm_array
                    continue

                final_start = (
                    total_rows - self.sequence_length
                )

                for start in range(
                    0,
                    final_start + 1,
                    self.stride,
                ):
                    end = start + self.sequence_length

                    window_alarm = alarm_array[start:end]

                    if (
                        self.alarm_free_only
                        and np.any(window_alarm != 0)
                    ):
                        continue

                    window = feature_array[start:end]

                    weak_alarm_label = float(
                        np.any(window_alarm != 0)
                    )

                    yield {
                        "x": torch.from_numpy(
                            window.copy()
                        ),
                        "timestamp_ns": torch.tensor(
                            timestamp_array[end - 1],
                            dtype=torch.int64,
                        ),
                        "alarm_label": torch.tensor(
                            weak_alarm_label,
                            dtype=torch.float32,
                        ),
                    }

                # পরবর্তী chunk-এর boundary sequence ধরে রাখা
                carry_length = self.sequence_length - 1

                carry_features = feature_array[
                    -carry_length:
                ].copy()

                carry_timestamps = timestamp_array[
                    -carry_length:
                ].copy()

                carry_alarms = alarm_array[
                    -carry_length:
                ].copy()


# ============================================================
# LSTM Autoencoder
# ============================================================

class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        number_of_features: int,
        hidden_size: int,
        latent_size: int,
        number_of_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        effective_dropout = (
            dropout if number_of_layers > 1 else 0.0
        )

        self.encoder = nn.LSTM(
            input_size=number_of_features,
            hidden_size=hidden_size,
            num_layers=number_of_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.to_latent = nn.Linear(
            hidden_size,
            latent_size,
        )

        self.decoder = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=number_of_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.output_layer = nn.Linear(
            hidden_size,
            number_of_features,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = x.size(1)

        _, (hidden_state, _) = self.encoder(x)

        last_hidden = hidden_state[-1]

        latent = self.to_latent(last_hidden)

        repeated_latent = latent.unsqueeze(1).repeat(
            1,
            sequence_length,
            1,
        )

        decoded_sequence, _ = self.decoder(
            repeated_latent
        )

        reconstruction = self.output_layer(
            decoded_sequence
        )

        return reconstruction


# ============================================================
# Training and evaluation functions
# ============================================================

def create_loader(
    dataset: PLCSequenceDataset,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_fp16: bool,
    grad_scaler: torch.amp.GradScaler,
) -> tuple[float, int]:
    model.train()

    running_loss = 0.0
    number_of_batches = 0
    number_of_windows = 0

    progress = tqdm(
        loader,
        desc="Training",
        unit="batch",
    )

    for batch in progress:
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

        grad_scaler.scale(loss).backward()

        grad_scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )

        grad_scaler.step(optimizer)
        grad_scaler.update()

        running_loss += float(loss.item())
        number_of_batches += 1
        number_of_windows += x.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    mean_loss = (
        running_loss / number_of_batches
        if number_of_batches > 0
        else float("nan")
    )

    return mean_loss, number_of_windows


@torch.inference_mode()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_fp16: bool,
) -> tuple[float, int]:
    model.eval()

    running_loss = 0.0
    number_of_batches = 0
    number_of_windows = 0

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

        running_loss += float(loss.item())
        number_of_batches += 1
        number_of_windows += x.size(0)

    mean_loss = (
        running_loss / number_of_batches
        if number_of_batches > 0
        else float("nan")
    )

    return mean_loss, number_of_windows


@torch.inference_mode()
def collect_reconstruction_errors(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> pd.DataFrame:
    model.eval()

    all_errors: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    all_alarm_labels: list[np.ndarray] = []

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

            # প্রতিটি sequence/window-এর reconstruction MSE
            window_errors = torch.mean(
                (reconstruction - x) ** 2,
                dim=(1, 2),
            )

        all_errors.append(
            window_errors.float().cpu().numpy()
        )

        all_timestamps.append(
            batch["timestamp_ns"].cpu().numpy()
        )

        all_alarm_labels.append(
            batch["alarm_label"].cpu().numpy()
        )

    if not all_errors:
        raise RuntimeError(
            "Reconstruction score তৈরি করার মতো window পাওয়া যায়নি।"
        )

    errors = np.concatenate(all_errors)
    timestamp_ns = np.concatenate(all_timestamps)
    alarm_labels = np.concatenate(
        all_alarm_labels
    ).astype(np.int8)

    return pd.DataFrame(
        {
            "logged_at": pd.to_datetime(
                timestamp_ns
            ),
            "reconstruction_error": errors,
            "alarm_weak_label": alarm_labels,
        }
    )


def save_fp16_state_dict(
    model: nn.Module,
    output_path: Path,
) -> None:
    """
    Deployment comparison-এর জন্য FP16 weights save করে।
    Training checkpoint আলাদাভাবে FP32 master weights রাখে।
    """

    fp16_state = {}

    for key, value in model.state_dict().items():
        tensor = value.detach().cpu()

        if tensor.is_floating_point():
            tensor = tensor.half()

        fp16_state[key] = tensor

    torch.save(fp16_state, output_path)


# ============================================================
# Plotting
# ============================================================

def save_training_plot(
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
    plt.title("LSTM Autoencoder Training History")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "training_history.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_test_score_plot(
    test_df: pd.DataFrame,
    threshold: float,
) -> None:
    # পুরো দিনের graph অত্যন্ত dense হতে পারে।
    # Visualization-এর জন্য সর্বোচ্চ 20,000 point রাখা হচ্ছে।
    maximum_plot_points = 20_000

    if len(test_df) > maximum_plot_points:
        plot_step = int(
            np.ceil(
                len(test_df)
                / maximum_plot_points
            )
        )
        plot_df = test_df.iloc[::plot_step].copy()
    else:
        plot_df = test_df.copy()

    plt.figure(figsize=(16, 6))

    plt.plot(
        plot_df["logged_at"],
        plot_df["reconstruction_error"],
        linewidth=0.8,
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
            s=22,
            label="Detected anomaly",
        )

    plt.xlabel("Time")
    plt.ylabel("Window reconstruction MSE")
    plt.title(
        "Full-Dataset LSTM-AE Anomaly Score: Test Day"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "test_anomaly_score.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


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
    # 1. Discover and split files
    # --------------------------------------------------------

    all_files = discover_csv_files()

    train_files, validation_files, test_files = (
        chronological_split(all_files)
    )

    print("=" * 78)
    print("Full PLC dataset files")
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

    split_information = {
        "train_files": [
            file.name for file in train_files
        ],
        "validation_files": [
            file.name for file in validation_files
        ],
        "test_files": [
            file.name for file in test_files
        ],
    }

    with open(
        RESULT_DIR / "data_split.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_information,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # 2. Features and scaler
    # --------------------------------------------------------

    candidate_features = identify_candidate_features(
        all_files[0]
    )

    scaler, retained_features, _ = fit_scaler(
        train_files,
        candidate_features,
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

    print()
    print("Retained model features:")
    for feature in retained_features:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # 3. Datasets
    # --------------------------------------------------------

    train_dataset = PLCSequenceDataset(
        files=train_files,
        candidate_features=candidate_features,
        retained_features=retained_features,
        scaler=scaler,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        alarm_free_only=TRAIN_ONLY_ALARM_FREE_WINDOWS,
    )

    # Validation loss এবং threshold শুধু alarm-free windows দিয়ে
    validation_normal_dataset = PLCSequenceDataset(
        files=validation_files,
        candidate_features=candidate_features,
        retained_features=retained_features,
        scaler=scaler,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        alarm_free_only=True,
    )

    test_dataset = PLCSequenceDataset(
        files=test_files,
        candidate_features=candidate_features,
        retained_features=retained_features,
        scaler=scaler,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        alarm_free_only=False,
    )

    train_loader = create_loader(
        train_dataset,
        BATCH_SIZE,
    )

    validation_loader = create_loader(
        validation_normal_dataset,
        BATCH_SIZE,
    )

    test_loader = create_loader(
        test_dataset,
        BATCH_SIZE,
    )

    # --------------------------------------------------------
    # 4. Device and FP16
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
    print(f"Device             : {device}")
    print(f"CUDA available     : {torch.cuda.is_available()}")
    print(f"FP16 AMP enabled   : {use_fp16}")
    print(f"Sequence length    : {SEQUENCE_LENGTH}")
    print(f"Stride             : {STRIDE}")
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"Epochs             : {EPOCHS}")
    print(f"Features           : {len(retained_features)}")

    if not use_fp16:
        print()
        print(
            "WARNING: GPU পাওয়া যায়নি। "
            "Training FP32 CPU-তে চলবে।"
        )
        print(
            "FP16 training চালাতে CUDA-enabled "
            "PyTorch এবং NVIDIA GPU প্রয়োজন।"
        )

    # --------------------------------------------------------
    # 5. Model
    # --------------------------------------------------------

    model = LSTMAutoencoder(
        number_of_features=len(retained_features),
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        number_of_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_fp16,
    )

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Trainable parameters: "
        f"{number_of_parameters:,}"
    )

    # --------------------------------------------------------
    # 6. Train
    # --------------------------------------------------------

    history: list[dict[str, float | int]] = []

    best_validation_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    total_training_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        print()
        print("=" * 78)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("=" * 78)

        epoch_start = time.perf_counter()

        train_loss, train_windows = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_fp16=use_fp16,
            grad_scaler=grad_scaler,
        )

        validation_loss, validation_windows = (
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

        print(f"Train loss          : {train_loss:.8f}")
        print(f"Validation loss     : {validation_loss:.8f}")
        print(f"Training windows    : {train_windows:,}")
        print(f"Validation windows  : {validation_windows:,}")
        print(f"Epoch time          : {epoch_time:.2f} s")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_windows": train_windows,
                "validation_windows": validation_windows,
                "epoch_time_seconds": epoch_time,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            print("Best model updated.")
        else:
            epochs_without_improvement += 1

            print(
                "No improvement count: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping activated.")
            break

    total_training_time = (
        time.perf_counter()
        - total_training_start
    )

    if best_model_state is None:
        raise RuntimeError(
            "Best model state তৈরি হয়নি।"
        )

    model.load_state_dict(best_model_state)

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        RESULT_DIR / "training_history.csv",
        index=False,
    )

    save_training_plot(history_df)

    # --------------------------------------------------------
    # 7. Save model
    # --------------------------------------------------------

    checkpoint_path = (
        MODEL_DIR
        / "lstm_ae_full_dataset_checkpoint.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_configuration": {
                "number_of_features": len(
                    retained_features
                ),
                "hidden_size": HIDDEN_SIZE,
                "latent_size": LATENT_SIZE,
                "number_of_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "sequence_length": SEQUENCE_LENGTH,
            },
            "retained_features": retained_features,
            "best_validation_loss": best_validation_loss,
            "training_time_seconds": total_training_time,
            "fp16_amp_training": use_fp16,
        },
        checkpoint_path,
    )

    fp16_model_path = (
        MODEL_DIR
        / "lstm_ae_full_dataset_fp16_weights.pt"
    )

    save_fp16_state_dict(
        model,
        fp16_model_path,
    )

    # --------------------------------------------------------
    # 8. Validation threshold
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("Calculating anomaly threshold")
    print("=" * 78)

    # নতুন loader প্রয়োজন, কারণ IterableDataset একবার iterate হয়েছে।
    threshold_validation_dataset = PLCSequenceDataset(
        files=validation_files,
        candidate_features=candidate_features,
        retained_features=retained_features,
        scaler=scaler,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        alarm_free_only=True,
    )

    threshold_validation_loader = create_loader(
        threshold_validation_dataset,
        BATCH_SIZE,
    )

    validation_scores = collect_reconstruction_errors(
        model=model,
        loader=threshold_validation_loader,
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

    print(
        f"Threshold percentile : "
        f"{ANOMALY_PERCENTILE}"
    )
    print(
        f"Anomaly threshold    : "
        f"{threshold:.8f}"
    )

    # --------------------------------------------------------
    # 9. Test scoring
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("Scoring complete test day")
    print("=" * 78)

    test_start = time.perf_counter()

    test_scores = collect_reconstruction_errors(
        model=model,
        loader=test_loader,
        device=device,
        use_fp16=use_fp16,
    )

    test_inference_time = (
        time.perf_counter()
        - test_start
    )

    test_scores["anomaly_threshold"] = threshold

    test_scores["predicted_anomaly"] = (
        test_scores["reconstruction_error"]
        > threshold
    ).astype(np.int8)

    test_scores.to_csv(
        RESULT_DIR / "test_anomaly_scores.csv",
        index=False,
    )

    save_test_score_plot(
        test_scores,
        threshold,
    )

    # --------------------------------------------------------
    # 10. Summary and weak-label evaluation
    # --------------------------------------------------------

    total_test_windows = len(test_scores)

    detected_anomalies = int(
        test_scores["predicted_anomaly"].sum()
    )

    anomaly_rate = (
        100.0
        * detected_anomalies
        / total_test_windows
        if total_test_windows > 0
        else float("nan")
    )

    test_windows_per_second = (
        total_test_windows / test_inference_time
        if test_inference_time > 0
        else float("inf")
    )

    y_true = test_scores[
        "alarm_weak_label"
    ].to_numpy(dtype=np.int8)

    y_pred = test_scores[
        "predicted_anomaly"
    ].to_numpy(dtype=np.int8)

    weak_precision = float("nan")
    weak_recall = float("nan")
    weak_f1 = float("nan")
    tn = fp = fn = tp = 0

    if len(np.unique(y_true)) >= 2:
        weak_precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        weak_recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        weak_f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

    checkpoint_size_mb = (
        checkpoint_path.stat().st_size
        / (1024 ** 2)
    )

    fp16_size_mb = (
        fp16_model_path.stat().st_size
        / (1024 ** 2)
    )

    summary = {
        "model": "LSTM Autoencoder",
        "training_precision": (
            "CUDA AMP FP16"
            if use_fp16
            else "CPU FP32"
        ),
        "number_of_files": len(all_files),
        "number_of_train_files": len(train_files),
        "number_of_validation_files": len(
            validation_files
        ),
        "number_of_test_files": len(test_files),
        "number_of_features": len(retained_features),
        "sequence_length": SEQUENCE_LENGTH,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "epochs_completed": len(history_df),
        "parameters": number_of_parameters,
        "best_validation_loss": (
            best_validation_loss
        ),
        "threshold_percentile": (
            ANOMALY_PERCENTILE
        ),
        "anomaly_threshold": threshold,
        "total_training_time_seconds": (
            total_training_time
        ),
        "test_inference_time_seconds": (
            test_inference_time
        ),
        "test_windows": total_test_windows,
        "test_windows_per_second": (
            test_windows_per_second
        ),
        "detected_anomaly_windows": (
            detected_anomalies
        ),
        "detected_anomaly_rate_percent": (
            anomaly_rate
        ),
        "weak_alarm_precision": weak_precision,
        "weak_alarm_recall": weak_recall,
        "weak_alarm_f1": weak_f1,
        "weak_alarm_tn": int(tn),
        "weak_alarm_fp": int(fp),
        "weak_alarm_fn": int(fn),
        "weak_alarm_tp": int(tp),
        "fp32_checkpoint_size_mb": (
            checkpoint_size_mb
        ),
        "fp16_weights_size_mb": fp16_size_mb,
    }

    summary_df = pd.DataFrame([summary])

    summary_df.to_csv(
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

    print()
    print("=" * 78)
    print("Experiment completed successfully")
    print("=" * 78)
    print(f"Precision mode       : {summary['training_precision']}")
    print(f"Best validation loss : {best_validation_loss:.8f}")
    print(f"Anomaly threshold    : {threshold:.8f}")
    print(f"Test windows         : {total_test_windows:,}")
    print(f"Detected anomalies   : {detected_anomalies:,}")
    print(f"Anomaly rate         : {anomaly_rate:.3f}%")
    print(f"Throughput           : {test_windows_per_second:.2f} windows/s")
    print(f"Weak alarm precision : {weak_precision}")
    print(f"Weak alarm recall    : {weak_recall}")
    print(f"Weak alarm F1        : {weak_f1}")
    print(f"FP32 checkpoint      : {checkpoint_size_mb:.3f} MB")
    print(f"FP16 weights         : {fp16_size_mb:.3f} MB")
    print()
    print(f"Results folder       : {RESULT_DIR}")
    print(f"Main checkpoint      : {checkpoint_path}")
    print(f"FP16 weights         : {fp16_model_path}")


if __name__ == "__main__":
    main()