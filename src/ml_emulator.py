from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "reference_hydrology_daily.csv"

PREDICTION_FILE = ROOT / "data" / "ml_predictions.csv"

COMPARISON_FILE = ROOT / "data" / "model_comparison.csv"

FIGURE_1 = ROOT / "figures" / "reference_vs_ml_streamflow.png"
FIGURE_2 = ROOT / "figures" / "ml_scatter_performance.png"
FIGURE_3 = ROOT / "figures" / "feature_importance.png"
FIGURE_4 = ROOT / "figures" / "model_comparison.png"


# ============================================================
# SETTINGS
# ============================================================

TRAIN_END = pd.Timestamp("2022-01-01")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Seasonal features
df["day_of_year"] = df["date"].dt.dayofyear

df["sin_doy"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365.25
)

df["cos_doy"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365.25
)


# ------------------------------------------------------------
# Precipitation lags
# ------------------------------------------------------------

for lag in [1, 2, 3, 7, 14]:

    df[f"precip_lag_{lag}"] = (
        df["precipitation_mm"].shift(lag)
    )


# ------------------------------------------------------------
# Antecedent precipitation
# ------------------------------------------------------------

for window in [3, 7, 14]:

    df[f"precip_sum_{window}d"] = (
        df["precipitation_mm"]
        .rolling(window)
        .sum()
    )


# ------------------------------------------------------------
# Hydrological memory
# ------------------------------------------------------------

df["flow_lag_1"] = (
    df["streamflow_mm"].shift(1)
)

df["flow_lag_2"] = (
    df["streamflow_mm"].shift(2)
)

df["flow_lag_3"] = (
    df["streamflow_mm"].shift(3)
)

df["soil_storage_lag1"] = (
    df["soil_storage_mm"].shift(1)
)

df["groundwater_storage_lag1"] = (
    df["groundwater_storage_mm"].shift(1)
)


# ============================================================
# FEATURE GROUPS
# ============================================================

climate_features = [

    "precipitation_mm",
    "temperature_C",
    "pet_mm",

    "sin_doy",
    "cos_doy",

    "precip_lag_1",
    "precip_lag_2",
    "precip_lag_3",
    "precip_lag_7",
    "precip_lag_14",

    "precip_sum_3d",
    "precip_sum_7d",
    "precip_sum_14d"
]


memory_features = climate_features + [

    "flow_lag_1",
    "flow_lag_2",
    "flow_lag_3",

    "soil_storage_lag1",
    "groundwater_storage_lag1"
]


TARGET = "streamflow_mm"


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

required_columns = list(
    set(climate_features + memory_features + [TARGET])
)

df_model = df.dropna(
    subset=required_columns
).copy()


# ============================================================
# TEMPORAL SPLIT
# ============================================================

train = df_model[
    df_model["date"] < TRAIN_END
].copy()

test = df_model[
    df_model["date"] >= TRAIN_END
].copy()


print("\nML EMULATOR COMPARISON")
print("=" * 50)

print(
    f"Training observations: {len(train):,}"
)

print(
    f"Testing observations:  {len(test):,}"
)


# ============================================================
# MODEL FUNCTION
# ============================================================

def train_model(features):

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[features],
        train[TARGET]
    )

    prediction = model.predict(
        test[features]
    )

    return model, prediction


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


# ============================================================
# TRAIN BOTH MODELS
# ============================================================

climate_model, climate_prediction = train_model(
    climate_features
)

memory_model, memory_prediction = train_model(
    memory_features
)


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

y_true = test[TARGET].values

climate_mae, climate_rmse, climate_r2 = calculate_metrics(
    y_true,
    climate_prediction
)

memory_mae, memory_rmse, memory_r2 = calculate_metrics(
    y_true,
    memory_prediction
)


print("\nOVERALL TEST PERFORMANCE")
print("-" * 50)

print("\nClimate-only model")
print(f"MAE  : {climate_mae:.4f} mm/day")
print(f"RMSE : {climate_rmse:.4f} mm/day")
print(f"R²   : {climate_r2:.4f}")

print("\nClimate + hydrological memory")
print(f"MAE  : {memory_mae:.4f} mm/day")
print(f"RMSE : {memory_rmse:.4f} mm/day")
print(f"R²   : {memory_r2:.4f}")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({

    "date": test["date"].values,

    "streamflow_mm":
        y_true,

    "climate_only_prediction_mm":
        climate_prediction,

    "predicted_streamflow_mm":
        memory_prediction
})

predictions.to_csv(
    PREDICTION_FILE,
    index=False
)


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "model": [
        "Climate-only",
        "Climate + hydrological memory"
    ],

    "MAE_mm_day": [
        climate_mae,
        memory_mae
    ],

    "RMSE_mm_day": [
        climate_rmse,
        memory_rmse
    ],

    "R2": [
        climate_r2,
        memory_r2
    ]
})

comparison.to_csv(
    COMPARISON_FILE,
    index=False
)


# ============================================================
# FIGURE 1 — TIME SERIES
# ============================================================

plot_start = test["date"].iloc[0]
plot_end = test["date"].iloc[min(365, len(test) - 1)]

plot_df = predictions[
    (predictions["date"] >= plot_start)
    &
    (predictions["date"] <= plot_end)
]

plt.figure(figsize=(12, 6))

plt.plot(
    plot_df["date"],
    plot_df["streamflow_mm"],
    linewidth=1.5,
    label="Reference streamflow"
)

plt.plot(
    plot_df["date"],
    plot_df["climate_only_prediction_mm"],
    linewidth=1,
    label="Climate-only RF"
)

plt.plot(
    plot_df["date"],
    plot_df["predicted_streamflow_mm"],
    linewidth=1,
    linestyle="--",
    label="Climate + memory RF"
)

plt.xlabel("Date")
plt.ylabel("Streamflow (mm/day)")
plt.title("Reference and ML-Emulated Streamflow")

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2 — SCATTER PERFORMANCE
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_true,
    memory_prediction,
    alpha=0.35,
    s=18
)

max_value = max(
    y_true.max(),
    memory_prediction.max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--",
    linewidth=1.5
)

plt.xlabel("Reference streamflow (mm/day)")
plt.ylabel("Predicted streamflow (mm/day)")

plt.title(
    f"Climate + Memory RF\n"
    f"R² = {memory_r2:.3f}"
)

plt.tight_layout()

plt.savefig(
    FIGURE_2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3 — FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    memory_model.feature_importances_,
    index=memory_features
).sort_values(ascending=True)

plt.figure(figsize=(9, 7))

importance.tail(15).plot(
    kind="barh"
)

plt.xlabel("Random Forest feature importance")
plt.ylabel("Feature")

plt.title(
    "Most Important Features for Hydrological Emulation"
)

plt.tight_layout()

plt.savefig(
    FIGURE_3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 4 — MODEL COMPARISON
# ============================================================

plt.figure(figsize=(7, 5))

plt.bar(
    comparison["model"],
    comparison["R2"]
)

plt.ylabel("R²")
plt.title("Temporal Test Performance")

plt.ylim(
    min(0, comparison["R2"].min() - 0.1),
    1
)

plt.xticks(
    rotation=10
)

plt.tight_layout()

plt.savefig(
    FIGURE_4,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\nSaved:")
print(f"- {PREDICTION_FILE}")
print(f"- {COMPARISON_FILE}")
print(f"- {FIGURE_1}")
print(f"- {FIGURE_2}")
print(f"- {FIGURE_3}")
print(f"- {FIGURE_4}")

print("\nML emulator comparison complete.")