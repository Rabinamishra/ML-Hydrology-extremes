"""
Hydrological response under precipitation extremes.

Compares the internal conceptual hydrological model response across:
1. Non-extreme precipitation
2. Moderate extreme (95th-99th percentile)
3. Top 1% precipitation

Outputs:
    data/hydrological_response_extremes.csv
    figures/hydrological_response_extremes.png
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

OUTPUT_FILE = (
    DATA_DIR
    / "hydrological_response_extremes.csv"
)

FIGURE_FILE = (
    FIG_DIR
    / "hydrological_response_extremes.png"
)

SPLIT_DATE = "2022-01-01"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HYDROLOGICAL RESPONSE UNDER PRECIPITATION EXTREMES")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        HYDRO_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # --------------------------------------------------------
    # Training / testing split
    # --------------------------------------------------------

    train = df[
        df["date"] < SPLIT_DATE
    ].copy()

    test = df[
        df["date"] >= SPLIT_DATE
    ].copy()

    print(
        f"\nTraining observations: "
        f"{len(train):,}"
    )

    print(
        f"Testing observations: "
        f"{len(test):,}"
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
    # Classification
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
    # Variables to analyse
    # --------------------------------------------------------

    variables = [
        "precipitation_mm",
        "infiltration_mm",
        "recharge_mm",
        "soil_storage_mm",
        "groundwater_storage_mm",
        "quickflow_mm",
        "baseflow_mm",
        "streamflow_mm",
    ]

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    summary = (
        test
        .groupby("condition")[variables]
        .mean()
        .reindex(
            [
                "Non-extreme",
                "Moderate extreme",
                "Top 1%",
            ]
        )
    )

    print("\n")
    print("=" * 70)
    print("MEAN HYDROLOGICAL RESPONSE")
    print("=" * 70)

    print(
        summary.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --------------------------------------------------------
    # Add precipitation-to-streamflow response ratio
    # --------------------------------------------------------

    summary["streamflow_precip_ratio"] = (
        summary["streamflow_mm"]
        / summary["precipitation_mm"]
    )

    summary["quickflow_precip_ratio"] = (
        summary["quickflow_mm"]
        / summary["precipitation_mm"]
    )

    summary["baseflow_precip_ratio"] = (
        summary["baseflow_mm"]
        / summary["precipitation_mm"]
    )

    print("\n")
    print("=" * 70)
    print("RESPONSE RATIOS")
    print("=" * 70)

    ratio_columns = [
        "streamflow_precip_ratio",
        "quickflow_precip_ratio",
        "baseflow_precip_ratio",
    ]

    print(
        summary[ratio_columns].to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary.reset_index().to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # FIGURE
    # ========================================================

    figure_variables = [
        "precipitation_mm",
        "infiltration_mm",
        "quickflow_mm",
        "baseflow_mm",
        "streamflow_mm",
    ]

    labels = [
        "Precipitation",
        "Infiltration",
        "Quickflow",
        "Baseflow",
        "Streamflow",
    ]

    conditions = [
        "Non-extreme",
        "Moderate extreme",
        "Top 1%",
    ]

    x = np.arange(
        len(conditions)
    )

    width = 0.15

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    for i, variable in enumerate(
        figure_variables
    ):

        values = summary[
            variable
        ].values

        ax.bar(
            x + (i - 2) * width,
            values,
            width,
            label=labels[i],
        )

    ax.set_xlabel(
        "Precipitation regime"
    )

    ax.set_ylabel(
        "Mean daily water flux (mm/day)"
    )

    ax.set_title(
        "Hydrological Response Across Precipitation Extremes"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        conditions
    )

    ax.legend(
        ncol=3
    )

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
    # EXTREME EVENT DETAILS
    # ========================================================

    extreme = test[
        test["condition"] == "Top 1%"
    ].sort_values(
        "precipitation_mm",
        ascending=False
    )

    print("\n")
    print("=" * 70)
    print("TOP 1% EVENT DETAILS")
    print("=" * 70)

    detail_columns = [
        "date",
        "precipitation_mm",
        "infiltration_mm",
        "recharge_mm",
        "soil_storage_mm",
        "groundwater_storage_mm",
        "quickflow_mm",
        "baseflow_mm",
        "streamflow_mm",
    ]

    print(
        extreme[
            detail_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # ========================================================
    # FINISH
    # ========================================================

    print("\n")
    print("=" * 70)
    print("HYDROLOGICAL RESPONSE ANALYSIS COMPLETE")
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