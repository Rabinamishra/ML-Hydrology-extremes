from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

HYDROLOGY_FILE = ROOT / "data" / "reference_hydrology_daily.csv"
PREDICTION_FILE = ROOT / "data" / "ml_predictions.csv"

METRICS_FILE = ROOT / "data" / "extreme_event_metrics.csv"
FIGURE_FILE = ROOT / "figures" / "extreme_event_evaluation.png"


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

TRAIN_END = pd.Timestamp("2022-01-01")


# ------------------------------------------------------------
# LOAD REFERENCE HYDROLOGY
# ------------------------------------------------------------

reference = pd.read_csv(
    HYDROLOGY_FILE,
    parse_dates=["date"]
)

# Separate training and test periods using the full
# reference hydrology dataset.
train_reference = reference[
    reference["date"] < TRAIN_END
].copy()

test_reference = reference[
    reference["date"] >= TRAIN_END
].copy()


# ------------------------------------------------------------
# LOAD ML PREDICTIONS
# ------------------------------------------------------------

predictions = pd.read_csv(
    PREDICTION_FILE,
    parse_dates=["date"]
)

# Merge predictions with the test-period reference data.
test = test_reference.merge(
    predictions[["date", "predicted_streamflow_mm"]],
    on="date",
    how="inner"
)


# ------------------------------------------------------------
# CHECK DATA
# ------------------------------------------------------------

if len(train_reference) == 0:
    raise ValueError("No training-period reference data found.")

if len(test) == 0:
    raise ValueError("No overlapping test-period predictions found.")


# ------------------------------------------------------------
# EXTREME THRESHOLDS
# ------------------------------------------------------------

# IMPORTANT:
# Thresholds are calculated ONLY from the training period.
# They are then applied to the independent test period.

p95_threshold = train_reference[
    "precipitation_mm"
].quantile(0.95)

p99_threshold = train_reference[
    "precipitation_mm"
].quantile(0.99)


print("\nTRAINING-DERIVED PRECIPITATION THRESHOLDS")
print("-" * 45)
print(f"95th percentile: {p95_threshold:.2f} mm/day")
print(f"99th percentile: {p99_threshold:.2f} mm/day")


# ------------------------------------------------------------
# CLASSIFY TEST OBSERVATIONS
# ------------------------------------------------------------

test["extreme_95"] = (
    test["precipitation_mm"] >= p95_threshold
)

test["extreme_99"] = (
    test["precipitation_mm"] >= p99_threshold
)


# ------------------------------------------------------------
# EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate(label, subset):

    y_true = subset["streamflow_mm"].values
    y_pred = subset["predicted_streamflow_mm"].values

    if len(subset) == 0:
        print(f"\n{label}")
        print("-" * 40)
        print("No observations in this category.")
        return {
            "condition": label,
            "n": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan
        }

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    if len(subset) >= 2 and np.var(y_true) > 0:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan

    print(f"\n{label}")
    print("-" * 40)
    print(f"Events/observations: {len(subset)}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    return {
        "condition": label,
        "n": len(subset),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ------------------------------------------------------------
# DEFINE GROUPS
# ------------------------------------------------------------

non_extreme = test[
    test["precipitation_mm"] < p95_threshold
]

top_5 = test[
    test["precipitation_mm"] >= p95_threshold
]

top_1 = test[
    test["precipitation_mm"] >= p99_threshold
]


# ------------------------------------------------------------
# RUN EVALUATION
# ------------------------------------------------------------

results = []

results.append(
    evaluate(
        "Non-extreme precipitation",
        non_extreme
    )
)

results.append(
    evaluate(
        "Top 5% precipitation",
        top_5
    )
)

results.append(
    evaluate(
        "Top 1% precipitation",
        top_1
    )
)


metrics = pd.DataFrame(results)

metrics.to_csv(
    METRICS_FILE,
    index=False
)


# ------------------------------------------------------------
# LARGEST TEST-PERIOD PRECIPITATION EVENT
# ------------------------------------------------------------

largest_event_date = test.loc[
    test["precipitation_mm"].idxmax(),
    "date"
]

event_window = test[
    (test["date"] >= largest_event_date - pd.Timedelta(days=7))
    &
    (test["date"] <= largest_event_date + pd.Timedelta(days=14))
]


# ------------------------------------------------------------
# FIGURE
# ------------------------------------------------------------

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.bar(
    event_window["date"],
    event_window["precipitation_mm"],
    alpha=0.35,
    label="Precipitation"
)

ax1.set_ylabel("Precipitation (mm/day)")
ax1.set_xlabel("Date")

ax2 = ax1.twinx()

ax2.plot(
    event_window["date"],
    event_window["streamflow_mm"],
    linewidth=2,
    label="Reference streamflow"
)

ax2.plot(
    event_window["date"],
    event_window["predicted_streamflow_mm"],
    linewidth=2,
    linestyle="--",
    label="ML predicted streamflow"
)

ax2.set_ylabel("Streamflow (mm/day)")

ax1.axvline(
    largest_event_date,
    linestyle=":",
    linewidth=1.5,
    label="Largest precipitation event"
)

fig.suptitle(
    "ML Hydrological Response During the Largest Extreme Precipitation Event"
)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left"
)

fig.tight_layout()

fig.savefig(
    FIGURE_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


print("\nExtreme-event analysis complete.")
print(f"Saved metrics: {METRICS_FILE}")
print(f"Saved figure:  {FIGURE_FILE}")