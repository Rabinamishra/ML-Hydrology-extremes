"""
Saturation-Aware Machine-Learning Hydrological Emulator

Compares:
1. Climate-only model
2. Climate + hydrological memory
3. Saturation-aware model

The saturation-aware model adds physically informed predictors derived
from information available before the prediction timestep.

Temporal split:
    Training: before 2022-01-01
    Testing:  2022-01-01 onward

Extreme thresholds are derived from the training period only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SPLIT_DATE = "2022-01-01"
SOIL_CAPACITY = 150.0

P95 = 0.95
P99 = 0.99

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 18,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    path = DATA_DIR / "reference_hydrology_daily.csv"

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(df):
    data = df.copy()

    # --------------------------------------------------------
    # Calendar / seasonality
    # --------------------------------------------------------

    day_of_year = data["date"].dt.dayofyear

    data["sin_doy"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    data["cos_doy"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    # --------------------------------------------------------
    # Precipitation memory
    # --------------------------------------------------------

    for lag in [1, 2, 3, 7, 14]:
        data[f"precip_lag_{lag}"] = (
            data["precipitation_mm"].shift(lag)
        )

    for window in [3, 7, 14]:
        data[f"precip_sum_{window}d"] = (
            data["precipitation_mm"]
            .shift(1)
            .rolling(window)
            .sum()
        )

    # --------------------------------------------------------
    # Hydrological memory
    # --------------------------------------------------------

    data["flow_lag_1"] = data["streamflow_mm"].shift(1)
    data["flow_lag_2"] = data["streamflow_mm"].shift(2)
    data["flow_lag_3"] = data["streamflow_mm"].shift(3)

    data["soil_storage_lag1"] = (
        data["soil_storage_mm"].shift(1)
    )

    data["groundwater_storage_lag1"] = (
        data["groundwater_storage_mm"].shift(1)
    )

    # --------------------------------------------------------
    # PHYSICALLY INFORMED SATURATION FEATURES
    # --------------------------------------------------------

    # Previous-day relative soil saturation.
    data["soil_saturation_lag1"] = (
        data["soil_storage_lag1"] / SOIL_CAPACITY
    )

    # Keep physically meaningful range.
    data["soil_saturation_lag1"] = (
        data["soil_saturation_lag1"].clip(0, 1)
    )

    # Previous-day storage deficit.
    data["soil_storage_deficit_lag1"] = (
        SOIL_CAPACITY - data["soil_storage_lag1"]
    ).clip(lower=0)

    # Antecedent wetness relative to soil capacity.
    data["antecedent_wetness_7d"] = (
        data["precip_sum_7d"] / SOIL_CAPACITY
    )

    data["antecedent_wetness_14d"] = (
        data["precip_sum_14d"] / SOIL_CAPACITY
    )

    # --------------------------------------------------------
    # Interaction terms
    # --------------------------------------------------------

    # Current precipitation under previous wetness conditions.
    data["precip_x_soil_saturation"] = (
        data["precipitation_mm"]
        * data["soil_saturation_lag1"]
    )

    data["precip_x_antecedent_wetness"] = (
        data["precipitation_mm"]
        * data["antecedent_wetness_7d"]
    )

    # Rainfall relative to remaining soil storage capacity.
    data["precip_over_storage_deficit"] = (
        data["precipitation_mm"]
        / (data["soil_storage_deficit_lag1"] + 1.0)
    )

    return data


# ============================================================
# FEATURE SETS
# ============================================================

def get_feature_sets():
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
        "precip_sum_14d",
    ]

    memory_features = climate_features + [
        "flow_lag_1",
        "flow_lag_2",
        "flow_lag_3",
        "soil_storage_lag1",
        "groundwater_storage_lag1",
    ]

    saturation_features = memory_features + [
        "soil_saturation_lag1",
        "soil_storage_deficit_lag1",
        "antecedent_wetness_7d",
        "antecedent_wetness_14d",
        "precip_x_soil_saturation",
        "precip_x_antecedent_wetness",
        "precip_over_storage_deficit",
    ]

    return {
        "Climate only": climate_features,
        "Climate + hydrological memory": memory_features,
        "Saturation-aware": saturation_features,
    }


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(train_df, test_df, features):

    train_clean = train_df.dropna(
        subset=features + ["streamflow_mm"]
    )

    test_clean = test_df.dropna(
        subset=features + ["streamflow_mm"]
    )

    X_train = train_clean[features]
    y_train = train_clean["streamflow_mm"]

    X_test = test_clean[features]
    y_test = test_clean["streamflow_mm"]

    model = RandomForestRegressor(**RF_PARAMS)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    return model, test_clean, y_test, prediction


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, prediction):

    return {
        "MAE": mean_absolute_error(y_true, prediction),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, prediction)
        ),
        "R2": r2_score(y_true, prediction),
    }


# ============================================================
# EXTREME CONDITIONS
# ============================================================

def classify_events(test_df, p95_threshold, p99_threshold):

    conditions = np.where(
        test_df["precipitation_mm"] < p95_threshold,
        "Non-extreme",
        np.where(
            test_df["precipitation_mm"] < p99_threshold,
            "Moderate extreme",
            "Top 1%",
        ),
    )

    return conditions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SATURATION-AWARE ML HYDROLOGICAL EMULATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()
    df = create_features(df)

    # --------------------------------------------------------
    # Temporal split
    # --------------------------------------------------------

    train = df[df["date"] < SPLIT_DATE].copy()
    test = df[df["date"] >= SPLIT_DATE].copy()

    print()
    print(f"Training observations: {len(train):,}")
    print(f"Testing observations:  {len(test):,}")

    # --------------------------------------------------------
    # Training-derived precipitation thresholds
    # --------------------------------------------------------

    p95_threshold = train["precipitation_mm"].quantile(P95)
    p99_threshold = train["precipitation_mm"].quantile(P99)

    print()
    print("TRAINING-DERIVED THRESHOLDS")
    print("-" * 45)
    print(f"95th percentile: {p95_threshold:.2f} mm/day")
    print(f"99th percentile: {p99_threshold:.2f} mm/day")

    # --------------------------------------------------------
    # Feature sets
    # --------------------------------------------------------

    feature_sets = get_feature_sets()

    all_results = []
    prediction_frames = []

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    for model_name, features in feature_sets.items():

        print()
        print("=" * 70)
        print(model_name.upper())
        print("=" * 70)

        model, test_clean, y_test, prediction = train_model(
            train,
            test,
            features,
        )

        conditions = classify_events(
            test_clean,
            p95_threshold,
            p99_threshold,
        )

        # Overall metrics
        metrics = calculate_metrics(
            y_test,
            prediction,
        )

        all_results.append({
            "model": model_name,
            "condition": "Overall",
            "n": len(y_test),
            **metrics,
        })

        print()
        print("Overall")
        print(
            f"MAE  = {metrics['MAE']:.4f}"
        )
        print(
            f"RMSE = {metrics['RMSE']:.4f}"
        )
        print(
            f"R²   = {metrics['R2']:.4f}"
        )

        # ----------------------------------------------------
        # Event-specific metrics
        # ----------------------------------------------------

        for condition in [
            "Non-extreme",
            "Moderate extreme",
            "Top 1%",
        ]:

            mask = conditions == condition

            if mask.sum() == 0:
                continue

            event_metrics = calculate_metrics(
                y_test.iloc[np.where(mask)[0]],
                prediction[mask],
            )

            all_results.append({
                "model": model_name,
                "condition": condition,
                "n": int(mask.sum()),
                **event_metrics,
            })

            print()
            print(condition)

            print(
                f"N    = {mask.sum()}"
            )

            print(
                f"MAE  = {event_metrics['MAE']:.4f}"
            )

            print(
                f"RMSE = {event_metrics['RMSE']:.4f}"
            )

            print(
                f"R²   = {event_metrics['R2']:.4f}"
            )

        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        pred_df = test_clean[
            [
                "date",
                "precipitation_mm",
                "streamflow_mm",
            ]
        ].copy()

        pred_df["condition"] = conditions

        pred_df["prediction"] = prediction

        pred_df["model"] = model_name

        prediction_frames.append(pred_df)

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(all_results)

    print()
    print("=" * 70)
    print("SATURATION-AWARE MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_path = DATA_DIR / "saturation_aware_results.csv"

    results_df.to_csv(
        results_path,
        index=False,
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions_path = DATA_DIR / "saturation_aware_predictions.csv"

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # ========================================================
    # PLOT R² BY CONDITION
    # ========================================================

    r2_plot = (
        results_df[
            results_df["condition"] != "Overall"
        ]
        .pivot(
            index="condition",
            columns="model",
            values="R2",
        )
    )

    ax = r2_plot.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_ylabel("R²")
    ax.set_xlabel("")
    ax.set_title(
        "Hydrological Emulator Performance Across Precipitation Regimes"
    )

    ax.legend(
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()

    r2_path = FIG_DIR / "saturation_aware_r2.png"

    plt.savefig(
        r2_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # PLOT TOP 1% PERFORMANCE
    # ========================================================

    top1 = predictions_df[
        predictions_df["condition"] == "Top 1%"
    ].copy()

    plt.figure(figsize=(10, 6))

    observed = (
        top1[
            top1["model"] == "Climate only"
        ]
        .sort_values("date")
    )

    plt.plot(
        observed["date"],
        observed["streamflow_mm"],
        marker="o",
        label="Observed / reference",
    )

    for model_name in [
        "Climate only",
        "Climate + hydrological memory",
        "Saturation-aware",
    ]:

        model_data = (
            top1[
                top1["model"] == model_name
            ]
            .sort_values("date")
        )

        plt.plot(
            model_data["date"],
            model_data["prediction"],
            marker="o",
            linestyle="--",
            label=model_name,
        )

    plt.ylabel("Streamflow (mm/day)")
    plt.xlabel("Date")
    plt.title(
        "Top 1% Precipitation Events: ML Representation"
    )

    plt.legend()

    plt.xticks(rotation=45)

    plt.tight_layout()

    extreme_path = (
        FIG_DIR / "saturation_aware_top1_events.png"
    )

    plt.savefig(
        extreme_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 70)
    print("SATURATION-AWARE EXPERIMENT COMPLETE")
    print("=" * 70)

    print()
    print("Saved:")
    print(f"- {results_path}")
    print(f"- {predictions_path}")
    print(f"- {r2_path}")
    print(f"- {extreme_path}")


if __name__ == "__main__":
    main()