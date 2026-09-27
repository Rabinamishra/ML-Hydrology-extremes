"""
Feature ablation experiment for the synthetic hydrological ML emulator.

Models:
1. Climate only
2. Climate + flow memory
3. Climate + catchment states
4. Full model

Temporal split:
Training: 2015-2021
Testing:  2022-2024

Extreme thresholds are calculated from training-period precipitation only.
Extreme classes are mutually exclusive:
- Non-extreme: < 95th percentile
- Moderate extreme: 95th-99th percentile
- Top 1% extreme: >= 99th percentile
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

HYDRO_FILE = DATA_DIR / "reference_hydrology_daily.csv"
OUTPUT_FILE = DATA_DIR / "feature_ablation_results.csv"
FIGURE_FILE = FIG_DIR / "feature_ablation_r2.png"


# ============================================================
# MODEL SETTINGS
# ============================================================

SPLIT_DATE = "2022-01-01"

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 18,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE and R²."""

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):
    """
    Create the same feature groups used by the main ML emulator.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    # Day-of-year seasonality
    day_of_year = df["date"].dt.dayofyear

    df["sin_doy"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    df["cos_doy"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    # --------------------------------------------------------
    # Precipitation lags
    # --------------------------------------------------------

    for lag in [1, 2, 3, 7, 14]:
        df[f"precip_lag_{lag}"] = (
            df["precipitation_mm"].shift(lag)
        )

    # --------------------------------------------------------
    # Antecedent precipitation
    # --------------------------------------------------------

    for window in [3, 7, 14]:
        df[f"precip_sum_{window}"] = (
            df["precipitation_mm"]
            .rolling(window)
            .sum()
        )

    # --------------------------------------------------------
    # Hydrological memory
    # --------------------------------------------------------

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

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FEATURE ABLATION EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    if not HYDRO_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{HYDRO_FILE}"
        )

    df = pd.read_csv(HYDRO_FILE)

    print(f"\nLoaded observations: {len(df):,}")

    # --------------------------------------------------------
    # Create features
    # --------------------------------------------------------

    df = create_features(df)

    # --------------------------------------------------------
    # Define feature groups
    # --------------------------------------------------------

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
        "precip_sum_3",
        "precip_sum_7",
        "precip_sum_14",
    ]

    flow_memory_features = [
        "flow_lag_1",
        "flow_lag_2",
        "flow_lag_3",
    ]

    state_features = [
        "soil_storage_lag1",
        "groundwater_storage_lag1",
    ]

    feature_sets = {
        "Climate only": climate_features,

        "Climate + flow memory": (
            climate_features
            + flow_memory_features
        ),

        "Climate + catchment states": (
            climate_features
            + state_features
        ),

        "Full model": (
            climate_features
            + flow_memory_features
            + state_features
        ),
    }

    # --------------------------------------------------------
    # Keep rows where all possible features exist.
    #
    # This ensures every model is evaluated on exactly
    # the same observations.
    # --------------------------------------------------------

    all_features = (
        climate_features
        + flow_memory_features
        + state_features
    )

    required_columns = (
        ["date", "streamflow_mm"]
        + all_features
    )

    df_model = df[required_columns].dropna().copy()

    print(
        f"Usable observations after feature construction: "
        f"{len(df_model):,}"
    )

    # --------------------------------------------------------
    # Temporal split
    # --------------------------------------------------------

    train = df_model[
        df_model["date"] < SPLIT_DATE
    ].copy()

    test = df_model[
        df_model["date"] >= SPLIT_DATE
    ].copy()

    print(f"\nTraining observations: {len(train):,}")
    print(f"Testing observations:  {len(test):,}")

    # --------------------------------------------------------
    # Training-derived precipitation thresholds
    # --------------------------------------------------------

    p95 = train["precipitation_mm"].quantile(0.95)
    p99 = train["precipitation_mm"].quantile(0.99)

    print("\nTRAINING-DERIVED PRECIPITATION THRESHOLDS")
    print("-" * 45)
    print(f"95th percentile: {p95:.2f} mm/day")
    print(f"99th percentile: {p99:.2f} mm/day")

    # --------------------------------------------------------
    # Define mutually exclusive test conditions
    # --------------------------------------------------------

    test["condition"] = np.select(
        [
            test["precipitation_mm"] >= p99,
            test["precipitation_mm"] >= p95,
        ],
        [
            "Top 1%",
            "Moderate extreme",
        ],
        default="Non-extreme",
    )

    # --------------------------------------------------------
    # Train and evaluate each model
    # --------------------------------------------------------

    results = []

    for model_name, features in feature_sets.items():

        print("\n" + "=" * 70)
        print(model_name)
        print("=" * 70)

        X_train = train[features]
        y_train = train["streamflow_mm"]

        X_test = test[features]
        y_test = test["streamflow_mm"]

        model = RandomForestRegressor(
            **RF_PARAMS
        )

        print("Training model...")

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # Overall performance
        # ----------------------------------------------------

        mae, rmse, r2 = calculate_metrics(
            y_test,
            predictions
        )

        print("\nOverall test performance")
        print("-" * 40)
        print(f"MAE  : {mae:.4f} mm/day")
        print(f"RMSE : {rmse:.4f} mm/day")
        print(f"R²   : {r2:.4f}")

        results.append({
            "model": model_name,
            "condition": "Overall",
            "n": len(y_test),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        # ----------------------------------------------------
        # Extreme-condition performance
        # ----------------------------------------------------

        for condition in [
            "Non-extreme",
            "Moderate extreme",
            "Top 1%",
        ]:

            mask = (
                test["condition"] == condition
            )

            y_condition = y_test[mask]
            pred_condition = predictions[mask]

            n = len(y_condition)

            if n == 0:
                print(
                    f"\n{condition}: no observations"
                )
                continue

            mae_c = mean_absolute_error(
                y_condition,
                pred_condition
            )

            rmse_c = np.sqrt(
                mean_squared_error(
                    y_condition,
                    pred_condition
                )
            )

            # R² requires variation in observations
            if n >= 2 and y_condition.nunique() > 1:
                r2_c = r2_score(
                    y_condition,
                    pred_condition
                )
            else:
                r2_c = np.nan

            print(f"\n{condition}")
            print("-" * 40)
            print(f"N    : {n}")
            print(f"MAE  : {mae_c:.4f}")
            print(f"RMSE : {rmse_c:.4f}")
            print(f"R²   : {r2_c:.4f}")

            results.append({
                "model": model_name,
                "condition": condition,
                "n": n,
                "MAE": mae_c,
                "RMSE": rmse_c,
                "R2": r2_c,
            })

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\n\n")
    print("=" * 70)
    print("FEATURE ABLATION SUMMARY")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # FIGURE: R² ACROSS CONDITIONS
    # ========================================================

    conditions = [
        "Overall",
        "Non-extreme",
        "Moderate extreme",
        "Top 1%",
    ]

    models = list(feature_sets.keys())

    x = np.arange(len(conditions))

    width = 0.20

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for i, model_name in enumerate(models):

        model_values = []

        for condition in conditions:

            row = results_df[
                (results_df["model"] == model_name)
                & (
                    results_df["condition"]
                    == condition
                )
            ]

            if len(row) == 0:
                value = np.nan
            else:
                value = row["R2"].iloc[0]

            model_values.append(value)

        ax.bar(
            x + (i - 1.5) * width,
            model_values,
            width,
            label=model_name,
        )

    ax.set_xlabel(
        "Evaluation condition"
    )

    ax.set_ylabel(
        "R²"
    )

    ax.set_title(
        "Feature Ablation: Hydrological Memory and Model Performance"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        conditions
    )

    ax.axhline(
        0,
        linewidth=0.8
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # FINISH
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FEATURE ABLATION COMPLETE")
    print("=" * 70)

    print("\nSaved:")
    print(f"- {OUTPUT_FILE}")
    print(f"- {FIGURE_FILE}")


if __name__ == "__main__":
    main()