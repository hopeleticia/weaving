from pathlib import Path
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline


# ============================================================
# Configuration
# ============================================================

DATA_FILE = Path("dataset/plc_log_20260710.csv")
RESULT_DIR = Path("results/chronos_single_signal")

MODEL_NAME = "amazon/chronos-bolt-tiny"

# প্রথমে একটি গুরুত্বপূর্ণ dynamic signal দিয়ে পরীক্ষা
SIGNAL_COLUMN = "Motor_RPM"

# Dataset-এর শুরু থেকে কত নম্বর row নেওয়া হবে
START_ROW = 0

# প্রথম পরীক্ষার জন্য 5000 row যথেষ্ট
NUMBER_OF_ROWS = 5000

# Model আগের কতটি timestep দেখবে
CONTEXT_LENGTH = 512

# পরবর্তী কতটি timestep forecast করবে
# Chronos-Bolt-এর সর্বোচ্চ prediction length 64
PREDICTION_LENGTH = 64

# Quantile forecast levels
QUANTILE_LEVELS = [0.1, 0.5, 0.9]

# Robust threshold multiplier
MAD_MULTIPLIER = 3.0


# ============================================================
# Helper Functions
# ============================================================

def select_device() -> tuple[str, torch.dtype]:
    """
    CUDA থাকলে GPU, না থাকলে CPU ব্যবহার করবে।
    """

    if torch.cuda.is_available():
        return "cuda", torch.float16

    return "cpu", torch.float32


def load_dataset() -> pd.DataFrame:
    """
    CSV file থেকে নির্দিষ্ট continuous segment load করবে।
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset পাওয়া যায়নি:\n{DATA_FILE.resolve()}"
        )

    print("=" * 70)
    print("Loading PLC dataset")
    print("=" * 70)

    # START_ROW শূন্যের বেশি হলে প্রথম দিকের data row skip করবে।
    # Header কখনো skip হবে না।
    rows_to_skip = (
        range(1, START_ROW + 1)
        if START_ROW > 0
        else None
    )

    df = pd.read_csv(
        DATA_FILE,
        skiprows=rows_to_skip,
        nrows=NUMBER_OF_ROWS,
        low_memory=False,
    )

    print(f"Loaded rows    : {len(df):,}")
    print(f"Loaded columns : {len(df.columns)}")

    required_columns = {"logged_at", SIGNAL_COLUMN}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Dataset-এ প্রয়োজনীয় column পাওয়া যায়নি: "
            f"{sorted(missing_columns)}"
        )

    # Datetime conversion
    df["logged_at"] = pd.to_datetime(
        df["logged_at"],
        errors="coerce",
    )

    # Selected signal numeric conversion
    df[SIGNAL_COLUMN] = pd.to_numeric(
        df[SIGNAL_COLUMN],
        errors="coerce",
    )

    original_rows = len(df)

    # Invalid timestamp বা signal value বাদ দেওয়া
    df = (
        df.dropna(subset=["logged_at", SIGNAL_COLUMN])
        .sort_values("logged_at")
        .reset_index(drop=True)
    )

    removed_rows = original_rows - len(df)

    print(f"Valid rows     : {len(df):,}")
    print(f"Removed rows   : {removed_rows:,}")

    required_length = CONTEXT_LENGTH + PREDICTION_LENGTH

    if len(df) < required_length:
        raise ValueError(
            f"কমপক্ষে {required_length}টি valid row প্রয়োজন, "
            f"কিন্তু পাওয়া গেছে {len(df)}টি।"
        )

    return df


def calculate_sampling_statistics(
    timestamps: pd.Series,
) -> dict[str, float]:
    """
    Sampling interval এবং approximate sampling frequency বের করবে।
    """

    time_diff_ms = (
        timestamps.diff()
        .dt.total_seconds()
        .mul(1000)
        .dropna()
    )

    if time_diff_ms.empty:
        return {
            "median_interval_ms": float("nan"),
            "mean_interval_ms": float("nan"),
            "sampling_frequency_hz": float("nan"),
        }

    median_interval_ms = float(time_diff_ms.median())
    mean_interval_ms = float(time_diff_ms.mean())

    sampling_frequency_hz = (
        1000.0 / median_interval_ms
        if median_interval_ms > 0
        else float("nan")
    )

    return {
        "median_interval_ms": median_interval_ms,
        "mean_interval_ms": mean_interval_ms,
        "sampling_frequency_hz": sampling_frequency_hz,
    }


def calculate_robust_threshold(
    errors: np.ndarray,
) -> tuple[float, float, float]:
    """
    Median Absolute Deviation ব্যবহার করে anomaly threshold বানাবে।

    threshold = median(error) + multiplier × MAD
    """

    median_error = float(np.median(errors))

    mad = float(
        np.median(
            np.abs(errors - median_error)
        )
    )

    # যদি সব error প্রায় একই হয় এবং MAD শূন্য হয়,
    # তখন standard deviation fallback ব্যবহার করা হবে।
    if mad == 0:
        standard_deviation = float(np.std(errors))

        threshold = (
            median_error
            + MAD_MULTIPLIER * standard_deviation
        )
    else:
        threshold = (
            median_error
            + MAD_MULTIPLIER * mad
        )

    return median_error, mad, float(threshold)


def save_forecast_plot(
    timestamps: pd.Series,
    actual: np.ndarray,
    prediction: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    anomaly_labels: np.ndarray,
) -> Path:
    """
    Actual, forecast এবং prediction interval plot করবে।
    """

    output_path = RESULT_DIR / "motor_rpm_forecast.png"

    plt.figure(figsize=(15, 6))

    plt.plot(
        timestamps,
        actual,
        linewidth=1.8,
        label="Actual Motor RPM",
    )

    plt.plot(
        timestamps,
        prediction,
        linewidth=1.8,
        label="Chronos-Bolt prediction",
    )

    plt.fill_between(
        timestamps,
        lower_bound,
        upper_bound,
        alpha=0.25,
        label="10%-90% prediction interval",
    )

    anomaly_indices = np.where(anomaly_labels == 1)[0]

    if anomaly_indices.size > 0:
        plt.scatter(
            timestamps.iloc[anomaly_indices],
            actual[anomaly_indices],
            marker="x",
            s=75,
            linewidths=2,
            label="Potential anomaly",
        )

    plt.title(
        "Chronos-Bolt Tiny Zero-Shot Forecast: Motor RPM"
    )
    plt.xlabel("Time")
    plt.ylabel("Motor RPM")
    plt.xticks(rotation=30)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def save_error_plot(
    timestamps: pd.Series,
    absolute_error: np.ndarray,
    threshold: float,
    anomaly_labels: np.ndarray,
) -> Path:
    """
    Forecast error বা anomaly score plot করবে।
    """

    output_path = RESULT_DIR / "motor_rpm_anomaly_score.png"

    plt.figure(figsize=(15, 5))

    plt.plot(
        timestamps,
        absolute_error,
        linewidth=1.8,
        label="Absolute forecasting error",
    )

    plt.axhline(
        y=threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.3f}",
    )

    anomaly_indices = np.where(anomaly_labels == 1)[0]

    if anomaly_indices.size > 0:
        plt.scatter(
            timestamps.iloc[anomaly_indices],
            absolute_error[anomaly_indices],
            marker="x",
            s=75,
            linewidths=2,
            label="Detected anomaly",
        )

    plt.title(
        "Chronos Forecasting Error-Based Anomaly Score"
    )
    plt.xlabel("Time")
    plt.ylabel("Absolute error")
    plt.xticks(rotation=30)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


# ============================================================
# Main Experiment
# ============================================================

def main() -> None:
    """
    Chronos-Bolt Tiny pretrained model দিয়ে একটি signal-এর
    zero-shot forecasting এবং anomaly scoring চালাবে।
    """

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Dataset load
    # --------------------------------------------------------

    df = load_dataset()

    sampling_stats = calculate_sampling_statistics(
        df["logged_at"]
    )

    print()
    print("=" * 70)
    print("Sampling information")
    print("=" * 70)
    print(
        "Median interval : "
        f"{sampling_stats['median_interval_ms']:.3f} ms"
    )
    print(
        "Mean interval   : "
        f"{sampling_stats['mean_interval_ms']:.3f} ms"
    )
    print(
        "Approx. rate    : "
        f"{sampling_stats['sampling_frequency_hz']:.3f} Hz"
    )

    # --------------------------------------------------------
    # 2. Context এবং future actual values তৈরি
    # --------------------------------------------------------

    context_values = (
        df[SIGNAL_COLUMN]
        .iloc[:CONTEXT_LENGTH]
        .to_numpy(dtype=np.float32)
    )

    future_start = CONTEXT_LENGTH
    future_end = CONTEXT_LENGTH + PREDICTION_LENGTH

    actual_values = (
        df[SIGNAL_COLUMN]
        .iloc[future_start:future_end]
        .to_numpy(dtype=np.float32)
    )

    actual_timestamps = (
        df["logged_at"]
        .iloc[future_start:future_end]
        .reset_index(drop=True)
    )

    print()
    print("=" * 70)
    print("Selected experiment segment")
    print("=" * 70)
    print(f"Signal             : {SIGNAL_COLUMN}")
    print(f"Context length     : {CONTEXT_LENGTH}")
    print(f"Prediction length  : {PREDICTION_LENGTH}")
    print(
        f"Context range      : "
        f"{df['logged_at'].iloc[0]} to "
        f"{df['logged_at'].iloc[CONTEXT_LENGTH - 1]}"
    )
    print(
        f"Prediction range   : "
        f"{actual_timestamps.iloc[0]} to "
        f"{actual_timestamps.iloc[-1]}"
    )

    # --------------------------------------------------------
    # 3. Device select
    # --------------------------------------------------------

    device, torch_dtype = select_device()

    print()
    print("=" * 70)
    print("Loading pretrained Chronos model")
    print("=" * 70)
    print(f"Model       : {MODEL_NAME}")
    print(f"Device      : {device}")
    print(f"Torch dtype : {torch_dtype}")

    # --------------------------------------------------------
    # 4. Model load
    # --------------------------------------------------------

    model_load_start = time.perf_counter()

    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_NAME,
        device_map=device,
        torch_dtype=torch_dtype,
    )

    model_load_time = (
        time.perf_counter()
        - model_load_start
    )

    print(
        f"Model load time: "
        f"{model_load_time:.4f} seconds"
    )

    # Batch dimension:
    # shape = [1, CONTEXT_LENGTH]
    context_tensor = torch.tensor(
        context_values,
        dtype=torch.float32,
    ).unsqueeze(0)

    # --------------------------------------------------------
    # 5. Zero-shot forecast
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Running zero-shot forecast")
    print("=" * 70)

    inference_start = time.perf_counter()

    with torch.inference_mode():
        quantiles, mean = pipeline.predict_quantiles(
            inputs=context_tensor,
            prediction_length=PREDICTION_LENGTH,
            quantile_levels=QUANTILE_LEVELS,
        )

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    # Tensor থেকে NumPy
    quantiles_np = (
        quantiles
        .detach()
        .cpu()
        .numpy()
    )

    mean_np = (
        mean
        .detach()
        .cpu()
        .numpy()
    )

    print(f"Quantiles shape : {quantiles_np.shape}")
    print(f"Mean shape      : {mean_np.shape}")

    # Expected shapes:
    # quantiles = [batch, prediction_length, number_of_quantiles]
    # mean      = [batch, prediction_length]

    prediction = mean_np[0].reshape(-1)

    lower_bound = (
        quantiles_np[0, :, 0]
        .reshape(-1)
    )

    median_prediction = (
        quantiles_np[0, :, 1]
        .reshape(-1)
    )

    upper_bound = (
        quantiles_np[0, :, 2]
        .reshape(-1)
    )

    # Defensive validation
    expected_length = PREDICTION_LENGTH

    arrays_to_validate = {
        "actual_values": actual_values,
        "prediction": prediction,
        "lower_bound": lower_bound,
        "median_prediction": median_prediction,
        "upper_bound": upper_bound,
    }

    for array_name, array_value in arrays_to_validate.items():
        if len(array_value) != expected_length:
            raise RuntimeError(
                f"{array_name}-এর length "
                f"{len(array_value)}, expected "
                f"{expected_length}।"
            )

    # --------------------------------------------------------
    # 6. Forecast error এবং anomaly score
    # --------------------------------------------------------

    residual = actual_values - prediction
    absolute_error = np.abs(residual)
    squared_error = np.square(residual)

    mae = float(np.mean(absolute_error))
    mse = float(np.mean(squared_error))
    rmse = float(np.sqrt(mse))

    # Percentage metrics-এর denominator zero হলে সমস্যা এড়ানো
    nonzero_mask = np.abs(actual_values) > 1e-8

    if np.any(nonzero_mask):
        mape = float(
            np.mean(
                np.abs(
                    residual[nonzero_mask]
                    / actual_values[nonzero_mask]
                )
            )
            * 100
        )
    else:
        mape = float("nan")

    median_error, mad, threshold = (
        calculate_robust_threshold(
            absolute_error
        )
    )

    anomaly_labels = (
        absolute_error > threshold
    ).astype(np.int8)

    # Prediction interval-এর বাইরে থাকাও একটি useful indicator
    outside_interval = (
        (actual_values < lower_bound)
        | (actual_values > upper_bound)
    ).astype(np.int8)

    anomaly_count = int(anomaly_labels.sum())
    outside_interval_count = int(outside_interval.sum())

    anomaly_percentage = (
        100.0
        * anomaly_count
        / PREDICTION_LENGTH
    )

    inference_ms_per_point = (
        inference_time
        * 1000.0
        / PREDICTION_LENGTH
    )

    points_per_second = (
        PREDICTION_LENGTH / inference_time
        if inference_time > 0
        else float("inf")
    )

    print()
    print("=" * 70)
    print("Forecast evaluation")
    print("=" * 70)
    print(f"Inference time       : {inference_time:.4f} seconds")
    print(f"Time per point       : {inference_ms_per_point:.4f} ms")
    print(f"Throughput           : {points_per_second:.2f} points/s")
    print(f"MAE                  : {mae:.4f}")
    print(f"MSE                  : {mse:.4f}")
    print(f"RMSE                 : {rmse:.4f}")
    print(f"MAPE                 : {mape:.4f}%")
    print(f"Median error         : {median_error:.4f}")
    print(f"MAD                  : {mad:.4f}")
    print(f"Anomaly threshold    : {threshold:.4f}")
    print(
        f"Detected anomalies   : "
        f"{anomaly_count}/{PREDICTION_LENGTH}"
    )
    print(
        f"Anomaly percentage   : "
        f"{anomaly_percentage:.2f}%"
    )
    print(
        f"Outside prediction CI: "
        f"{outside_interval_count}/{PREDICTION_LENGTH}"
    )

    # --------------------------------------------------------
    # 7. Prediction results save
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        {
            "logged_at": actual_timestamps,
            "actual": actual_values,
            "prediction_mean": prediction,
            "prediction_median": median_prediction,
            "lower_quantile_0_1": lower_bound,
            "upper_quantile_0_9": upper_bound,
            "residual": residual,
            "absolute_error": absolute_error,
            "squared_error": squared_error,
            "anomaly_threshold": threshold,
            "predicted_anomaly": anomaly_labels,
            "outside_prediction_interval": outside_interval,
        }
    )

    prediction_csv = (
        RESULT_DIR
        / "chronos_motor_rpm_predictions.csv"
    )

    result_df.to_csv(
        prediction_csv,
        index=False,
    )

    # --------------------------------------------------------
    # 8. Summary save
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "signal": SIGNAL_COLUMN,
                "data_file": str(DATA_FILE),
                "start_row": START_ROW,
                "loaded_rows": len(df),
                "device": device,
                "torch_dtype": str(torch_dtype),
                "context_length": CONTEXT_LENGTH,
                "prediction_length": PREDICTION_LENGTH,
                "median_sampling_interval_ms": (
                    sampling_stats["median_interval_ms"]
                ),
                "mean_sampling_interval_ms": (
                    sampling_stats["mean_interval_ms"]
                ),
                "sampling_frequency_hz": (
                    sampling_stats["sampling_frequency_hz"]
                ),
                "model_load_time_seconds": model_load_time,
                "inference_time_seconds": inference_time,
                "inference_ms_per_point": inference_ms_per_point,
                "throughput_points_per_second": points_per_second,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "mape_percent": mape,
                "median_absolute_error": median_error,
                "mad": mad,
                "anomaly_threshold": threshold,
                "predicted_anomaly_points": anomaly_count,
                "anomaly_percentage": anomaly_percentage,
                "outside_prediction_interval_points": (
                    outside_interval_count
                ),
            }
        ]
    )

    summary_csv = RESULT_DIR / "summary.csv"

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    # --------------------------------------------------------
    # 9. Plots save
    # --------------------------------------------------------

    forecast_plot = save_forecast_plot(
        timestamps=actual_timestamps,
        actual=actual_values,
        prediction=prediction,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        anomaly_labels=anomaly_labels,
    )

    error_plot = save_error_plot(
        timestamps=actual_timestamps,
        absolute_error=absolute_error,
        threshold=threshold,
        anomaly_labels=anomaly_labels,
    )

    # --------------------------------------------------------
    # 10. Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Experiment completed successfully")
    print("=" * 70)
    print(f"Prediction CSV : {prediction_csv}")
    print(f"Summary CSV    : {summary_csv}")
    print(f"Forecast plot  : {forecast_plot}")
    print(f"Error plot     : {error_plot}")


if __name__ == "__main__":
    main()