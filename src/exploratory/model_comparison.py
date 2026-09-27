from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

HYDROLOGY_FILE = ROOT / "data" / "reference_hydrology_daily.csv"
PREDICTION_FILE = ROOT / "data" / "ml_predictions.csv"

OUTPUT_FILE = ROOT / "data" / "model_comparison_extremes.csv"
FIGURE_FILE = ROOT / "figures" / "model_comparison_extremes.png"


# ============================================================
# SETTINGS
# ============================================================

TRAIN_END = pd.Timestamp("2022-01-01")


# ============================================================
# LOAD DATA
# ============================================================

reference = pd.read_csv(
    HYDROLOGY_FILE,
    parse_dates=["date"]
)

predictions = pd.read_csv(
    PREDICTION_FILE,
    parse_dates=["date"]
)


# ============================================================
# DERIVE EXTREME THRESHOLDS FROM TRAINING PERIOD
# ============================================================

training = reference[
    reference["date"] < TRAIN_END
]

p95 = training["precipitation_mm"].quantile(0.95)
p99 = training["precipitation_mm"].quantile(0.99)


# ============================================================
# TEST DATA + PREDICTIONS
# ============================================================

test = reference[
    reference["date"] >= TRAIN_END
].merge(
    predictions[
        [
            "date",
            "climate_only_prediction_mm",
            "predicted_streamflow_mm"
        ]
    ],
    on="date",
    how="inner"
)


# ============================================================
# CLASSIFY PRECIPITATION
# ============================================================

test["condition"] = "Non-extreme"

test.loc[
    test["precipitation_mm"] >= p95,
    "condition"
] = "Top 5%"

test.loc[
    test["precipitation_mm"] >= p99,
    "condition"
] = "Top 1%"


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


# ============================================================
# COMPARE MODELS
# ============================================================

results = []

models = {
    "Climate-only": "climate_only_prediction_mm",
    "Climate + memory": "predicted_streamflow_mm"
}

conditions = [
    "Non-extreme",
    "Top 5%",
    "Top 1%"
]


for condition in conditions:

    subset = test[
        test["condition"] == condition
    ]

    for model_name, prediction_column in models.items():

        y_true = subset[
            "streamflow_mm"
        ].values

        y_pred = subset[
            prediction_column
        ].values

        mae, rmse, r2 = evaluate(
            y_true,
            y_pred
        )

        results.append({
            "condition": condition,
            "model": model_name,
            "n": len(subset),
            "MAE_mm_day": mae,
            "RMSE_mm_day": rmse,
            "R2": r2
        })


results = pd.DataFrame(results)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nMODEL PERFORMANCE ACROSS PRECIPITATION EXTREMES")
print("=" * 65)

print(
    f"\nTraining-derived thresholds:"
)

print(
    f"95th percentile = {p95:.2f} mm/day"
)

print(
    f"99th percentile = {p99:.2f} mm/day"
)


for condition in conditions:

    print(f"\n{condition}")
    print("-" * 65)

    subset = results[
        results["condition"] == condition
    ]

    for _, row in subset.iterrows():

        print(
            f"{row['model']:<25}"
            f"N={int(row['n']):4d}  "
            f"MAE={row['MAE_mm_day']:.4f}  "
            f"RMSE={row['RMSE_mm_day']:.4f}  "
            f"R²={row['R2']:.4f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FIGURE
# ============================================================

pivot = results.pivot(
    index="condition",
    columns="model",
    values="R2"
)

x = np.arange(len(conditions))

width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    x - width / 2,
    pivot.loc[
        conditions,
        "Climate-only"
    ],
    width,
    label="Climate-only"
)

plt.bar(
    x + width / 2,
    pivot.loc[
        conditions,
        "Climate + memory"
    ],
    width,
    label="Climate + memory"
)

plt.xticks(
    x,
    conditions
)

plt.ylabel("R²")

plt.xlabel(
    "Precipitation condition"
)

plt.title(
    "Model Performance Across Precipitation Extremes"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nComparison complete.")

print(
    f"Saved: {OUTPUT_FILE}"
)

print(
    f"Saved: {FIGURE_FILE}"
)