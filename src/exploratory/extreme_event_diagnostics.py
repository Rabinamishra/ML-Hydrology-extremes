"""
Extreme-event diagnostic analysis.

Purpose:
    Examine how the ML emulator behaves during the rarest
    precipitation events.

The 99th-percentile threshold is calculated from the full
reference hydrological dataset for the training period
(2015-2021), while predictions are evaluated only for the
test period (2022-2024).

Outputs:
    data/extreme_event_diagnostics.csv
    figures/extreme_event_diagnostics.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

HYDRO_FILE = DATA_DIR / "reference_hydrology_daily.csv"
PREDICTION_FILE = DATA_DIR / "ml_predictions.csv"

OUTPUT_FILE = DATA_DIR / "extreme_event_diagnostics.csv"
FIGURE_FILE = FIG_DIR / "extreme_event_diagnostics.png"

SPLIT_DATE = "2022-01-01"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXTREME-EVENT DIAGNOSTIC ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load reference hydrological data
    # --------------------------------------------------------

    hydro = pd.read_csv(HYDRO_FILE)

    hydro["date"] = pd.to_datetime(
        hydro["date"]
    )

    print(
        f"\nReference hydrology observations: "
        f"{len(hydro):,}"
    )

    # --------------------------------------------------------
    # Load ML predictions
    # --------------------------------------------------------

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"]
    )

    print(
        f"ML prediction observations: "
        f"{len(predictions):,}"
    )

    # --------------------------------------------------------
    # TRAINING DATA
    #
    # Important:
    # ml_predictions.csv contains only test-period
    # predictions, so training thresholds must come directly
    # from the reference hydrology dataset.
    # --------------------------------------------------------

    train = hydro[
        hydro["date"] < SPLIT_DATE
    ].copy()

    test_reference = hydro[
        hydro["date"] >= SPLIT_DATE
    ].copy()

    print(
        f"\nTraining observations: "
        f"{len(train):,}"
    )

    print(
        f"Testing reference observations: "
        f"{len(test_reference):,}"
    )

    # --------------------------------------------------------
    # Training-derived precipitation thresholds
    # --------------------------------------------------------

    p95 = train[
        "precipitation_mm"
    ].quantile(0.95)

    p99 = train[
        "precipitation_mm"
    ].quantile(0.99)

    print("\nTRAINING-DERIVED THRESHOLDS")
    print("-" * 45)
    print(
        f"95th percentile: "
        f"{p95:.2f} mm/day"
    )

    print(
        f"99th percentile: "
        f"{p99:.2f} mm/day"
    )

    # --------------------------------------------------------
    # Merge test reference data with ML predictions
    #
    # Only merge prediction columns that are actually needed.
    # This avoids duplicate streamflow columns.
    # --------------------------------------------------------

    prediction_columns = [
        "date",
        "climate_only_prediction_mm",
        "predicted_streamflow_mm",
    ]

    predictions_for_merge = predictions[
        [
            column
            for column in prediction_columns
            if column in predictions.columns
        ]
    ].copy()

    test = test_reference.merge(
        predictions_for_merge,
        on="date",
        how="inner"
    )

    print(
        f"\nTest observations with predictions: "
        f"{len(test):,}"
    )

    # --------------------------------------------------------
    # Select top 1% precipitation events
    # --------------------------------------------------------

    extreme = test[
        test["precipitation_mm"] >= p99
    ].copy()

    extreme = extreme.sort_values(
        "precipitation_mm",
        ascending=False
    )

    print(
        f"\nTop 1% events in test period: "
        f"{len(extreme)}"
    )

    if len(extreme) == 0:
        raise ValueError(
            "No top-1% events were found. "
            "Check the precipitation threshold and data."
        )

    # ========================================================
    # DIAGNOSTIC TABLE
    # ========================================================

    diagnostic_columns = [
        "date",
        "precipitation_mm",
        "temperature_C",
        "pet_mm",
        "soil_storage_mm",
        "groundwater_storage_mm",
        "streamflow_mm",
    ]

    diagnostics = extreme[
        diagnostic_columns
    ].copy()

    # --------------------------------------------------------
    # Calculate diagnostics for each model
    # --------------------------------------------------------

    model_definitions = []

    if "climate_only_prediction_mm" in extreme.columns:

        model_definitions.append(
            (
                "Climate only",
                "climate_only_prediction_mm",
                "climate_only"
            )
        )

    if "predicted_streamflow_mm" in extreme.columns:

        model_definitions.append(
            (
                "Climate + memory",
                "predicted_streamflow_mm",
                "climate_memory"
            )
        )

    # --------------------------------------------------------
    # Model diagnostics
    # --------------------------------------------------------

    for model_name, prediction_column, prefix in model_definitions:

        observed = extreme[
            "streamflow_mm"
        ].to_numpy()

        predicted = extreme[
            prediction_column
        ].to_numpy()

        error = predicted - observed

        diagnostics[
            f"{prefix}_prediction_mm"
        ] = predicted

        diagnostics[
            f"{prefix}_error_mm"
        ] = error

        diagnostics[
            f"{prefix}_absolute_error_mm"
        ] = np.abs(error)

        diagnostics[
            f"{prefix}_underprediction_mm"
        ] = observed - predicted

        diagnostics[
            f"{prefix}_prediction_ratio"
        ] = np.where(
            observed != 0,
            predicted / observed,
            np.nan
        )

    # --------------------------------------------------------
    # Save detailed diagnostics
    # --------------------------------------------------------

    diagnostics.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # EVENT TABLE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TOP 1% EXTREME EVENTS")
    print("=" * 70)

    display_columns = [
        "date",
        "precipitation_mm",
        "streamflow_mm",
    ]

    for model_name, prediction_column, prefix in model_definitions:

        display_columns.extend([
            f"{prefix}_prediction_mm",
            f"{prefix}_error_mm",
        ])

    print(
        diagnostics[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # ========================================================
    # SUMMARY STATISTICS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EXTREME-EVENT DIAGNOSTIC SUMMARY")
    print("=" * 70)

    for model_name, prediction_column, prefix in model_definitions:

        observed = extreme[
            "streamflow_mm"
        ].to_numpy()

        predicted = extreme[
            prediction_column
        ].to_numpy()

        error = predicted - observed

        # ----------------------------------------------------
        # Mean statistics
        # ----------------------------------------------------

        mae = np.mean(
            np.abs(error)
        )

        rmse = np.sqrt(
            np.mean(error ** 2)
        )

        mean_bias = np.mean(error)

        mean_ratio = np.mean(
            predicted / observed
        )

        # ----------------------------------------------------
        # Peak event
        # ----------------------------------------------------

        peak_index = np.argmax(
            observed
        )

        peak_date = extreme[
            "date"
        ].iloc[peak_index]

        peak_precipitation = extreme[
            "precipitation_mm"
        ].iloc[peak_index]

        observed_peak = observed[
            peak_index
        ]

        predicted_peak = predicted[
            peak_index
        ]

        peak_error = (
            predicted_peak
            - observed_peak
        )

        peak_ratio = (
            predicted_peak
            / observed_peak
            if observed_peak != 0
            else np.nan
        )

        print("\n" + model_name)
        print("-" * 50)

        print(
            f"Mean absolute error  : "
            f"{mae:.3f} mm/day"
        )

        print(
            f"RMSE                 : "
            f"{rmse:.3f} mm/day"
        )

        print(
            f"Mean bias            : "
            f"{mean_bias:.3f} mm/day"
        )

        print(
            f"Mean prediction ratio: "
            f"{mean_ratio:.3f}"
        )

        print("\nLargest observed event")
        print(
            f"Date                 : "
            f"{peak_date.date()}"
        )

        print(
            f"Precipitation        : "
            f"{peak_precipitation:.2f} mm/day"
        )

        print(
            f"Observed streamflow  : "
            f"{observed_peak:.2f} mm/day"
        )

        print(
            f"Predicted streamflow : "
            f"{predicted_peak:.2f} mm/day"
        )

        print(
            f"Prediction error     : "
            f"{peak_error:.2f} mm/day"
        )

        print(
            f"Peak prediction ratio: "
            f"{peak_ratio:.3f}"
        )

    # ========================================================
    # FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9)
    )

    # --------------------------------------------------------
    # Panel 1: precipitation
    # --------------------------------------------------------

    ax1 = axes[0]

    ax1.plot(
        extreme["date"],
        extreme["precipitation_mm"],
        marker="o",
        linewidth=1.5,
        label="Precipitation"
    )

    ax1.set_ylabel(
        "Precipitation (mm/day)"
    )

    ax1.set_title(
        "Top 1% Precipitation Events"
    )

    ax1.grid(
        alpha=0.3
    )

    ax1.legend()

    # --------------------------------------------------------
    # Panel 2: observed vs predicted streamflow
    # --------------------------------------------------------

    ax2 = axes[1]

    ax2.plot(
        extreme["date"],
        extreme["streamflow_mm"],
        marker="o",
        linewidth=2,
        label="Reference streamflow"
    )

    for model_name, prediction_column, prefix in model_definitions:

        ax2.plot(
            extreme["date"],
            extreme[prediction_column],
            marker="x",
            linewidth=1.5,
            label=model_name
        )

    ax2.set_xlabel(
        "Date"
    )

    ax2.set_ylabel(
        "Streamflow (mm/day)"
    )

    ax2.set_title(
        "ML Representation of Top 1% Events"
    )

    ax2.grid(
        alpha=0.3
    )

    ax2.legend()

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
    print("EXTREME-EVENT DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print("\nSaved:")
    print(
        f"- {OUTPUT_FILE}"
    )

    print(
        f"- {FIGURE_FILE}"
    )


if __name__ == "__main__":
    main()