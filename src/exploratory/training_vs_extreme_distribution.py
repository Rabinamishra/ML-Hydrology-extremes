"""
Training vs. extreme-event distribution diagnostic.

Purpose:
    Determine whether the extreme precipitation events in the
    test period fall outside the precipitation and streamflow
    ranges represented in the training period.

Outputs:
    data/training_vs_extreme_distribution.csv
    figures/training_vs_extreme_distribution.png
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
    / "training_vs_extreme_distribution.csv"
)

FIGURE_FILE = (
    FIG_DIR
    / "training_vs_extreme_distribution.png"
)

SPLIT_DATE = "2022-01-01"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRAINING VS EXTREME-EVENT DISTRIBUTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load reference hydrology
    # --------------------------------------------------------

    df = pd.read_csv(
        HYDRO_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # --------------------------------------------------------
    # Temporal split
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
        f"Testing observations:  "
        f"{len(test):,}"
    )

    # ========================================================
    # PRECIPITATION DISTRIBUTION
    # ========================================================

    p95 = train[
        "precipitation_mm"
    ].quantile(0.95)

    p99 = train[
        "precipitation_mm"
    ].quantile(0.99)

    p999 = train[
        "precipitation_mm"
    ].quantile(0.999
    )

    train_max_precip = train[
        "precipitation_mm"
    ].max()

    test_max_precip = test[
        "precipitation_mm"
    ].max()

    print("\n")
    print("=" * 70)
    print("PRECIPITATION DISTRIBUTION")
    print("=" * 70)

    print(
        f"Training 95th percentile : "
        f"{p95:.2f} mm/day"
    )

    print(
        f"Training 99th percentile : "
        f"{p99:.2f} mm/day"
    )

    print(
        f"Training 99.9th percentile: "
        f"{p999:.2f} mm/day"
    )

    print(
        f"Training maximum          : "
        f"{train_max_precip:.2f} mm/day"
    )

    print(
        f"Test maximum              : "
        f"{test_max_precip:.2f} mm/day"
    )

    # ========================================================
    # STREAMFLOW DISTRIBUTION
    # ========================================================

    q95 = train[
        "streamflow_mm"
    ].quantile(0.95)

    q99 = train[
        "streamflow_mm"
    ].quantile(0.99)

    train_max_flow = train[
        "streamflow_mm"
    ].max()

    test_max_flow = test[
        "streamflow_mm"
    ].max()

    print("\n")
    print("=" * 70)
    print("STREAMFLOW DISTRIBUTION")
    print("=" * 70)

    print(
        f"Training 95th percentile : "
        f"{q95:.2f} mm/day"
    )

    print(
        f"Training 99th percentile : "
        f"{q99:.2f} mm/day"
    )

    print(
        f"Training maximum          : "
        f"{train_max_flow:.2f} mm/day"
    )

    print(
        f"Test maximum              : "
        f"{test_max_flow:.2f} mm/day"
    )

    # ========================================================
    # TOP 1% TEST EVENTS
    # ========================================================

    extreme = test[
        test["precipitation_mm"] >= p99
    ].copy()

    extreme = extreme.sort_values(
        "precipitation_mm",
        ascending=False
    )

    # --------------------------------------------------------
    # Calculate distribution metrics
    # --------------------------------------------------------

    extreme["precip_excess_over_train_p99_mm"] = (
        extreme["precipitation_mm"]
        - p99
    )

    extreme["precip_ratio_to_train_p99"] = (
        extreme["precipitation_mm"]
        / p99
    )

    extreme["precip_ratio_to_train_max"] = (
        extreme["precipitation_mm"]
        / train_max_precip
    )

    extreme["streamflow_ratio_to_train_p99"] = (
        extreme["streamflow_mm"]
        / q99
    )

    extreme["streamflow_ratio_to_train_max"] = (
        extreme["streamflow_mm"]
        / train_max_flow
    )

    extreme["outside_training_precip_range"] = (
        extreme["precipitation_mm"]
        > train_max_precip
    )

    extreme["outside_training_streamflow_range"] = (
        extreme["streamflow_mm"]
        > train_max_flow
    )

    # --------------------------------------------------------
    # Save event table
    # --------------------------------------------------------

    extreme.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # PRINT EXTREME EVENT TABLE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TOP 1% TEST EVENTS")
    print("=" * 70)

    display_columns = [
        "date",
        "precipitation_mm",
        "streamflow_mm",
        "precip_excess_over_train_p99_mm",
        "precip_ratio_to_train_p99",
        "precip_ratio_to_train_max",
        "streamflow_ratio_to_train_p99",
        "streamflow_ratio_to_train_max",
        "outside_training_precip_range",
        "outside_training_streamflow_range",
    ]

    print(
        extreme[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    n_extreme = len(extreme)

    n_outside_precip = (
        extreme[
            "outside_training_precip_range"
        ].sum()
    )

    n_outside_flow = (
        extreme[
            "outside_training_streamflow_range"
        ].sum()
    )

    print("\n")
    print("=" * 70)
    print("DISTRIBUTION-SHIFT SUMMARY")
    print("=" * 70)

    print(
        f"\nTop 1% test events: "
        f"{n_extreme}"
    )

    print(
        f"Events above training maximum precipitation: "
        f"{n_outside_precip}/{n_extreme}"
    )

    print(
        f"Events above training maximum streamflow: "
        f"{n_outside_flow}/{n_extreme}"
    )

    print(
        f"\nLargest test precipitation / "
        f"training maximum: "
        f"{test_max_precip / train_max_precip:.2f}×"
    )

    print(
        f"Largest test streamflow / "
        f"training maximum: "
        f"{test_max_flow / train_max_flow:.2f}×"
    )

    # ========================================================
    # FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9)
    )

    # --------------------------------------------------------
    # Panel 1: precipitation
    # --------------------------------------------------------

    ax1 = axes[0]

    ax1.hist(
        train["precipitation_mm"],
        bins=60,
        alpha=0.7,
        label="Training precipitation"
    )

    ax1.scatter(
        extreme["precipitation_mm"],
        np.zeros(len(extreme)),
        marker="x",
        s=60,
        label="Top 1% test events"
    )

    ax1.axvline(
        p99,
        linestyle="--",
        linewidth=1.5,
        label="Training 99th percentile"
    )

    ax1.axvline(
        train_max_precip,
        linestyle=":",
        linewidth=1.5,
        label="Training maximum"
    )

    ax1.set_xlabel(
        "Precipitation (mm/day)"
    )

    ax1.set_ylabel(
        "Training frequency"
    )

    ax1.set_title(
        "Training Precipitation Distribution vs Test Extremes"
    )

    ax1.legend()

    ax1.grid(
        alpha=0.3
    )

    # --------------------------------------------------------
    # Panel 2: streamflow
    # --------------------------------------------------------

    ax2 = axes[1]

    ax2.hist(
        train["streamflow_mm"],
        bins=60,
        alpha=0.7,
        label="Training streamflow"
    )

    ax2.scatter(
        extreme["streamflow_mm"],
        np.zeros(len(extreme)),
        marker="x",
        s=60,
        label="Streamflow during top 1% events"
    )

    ax2.axvline(
        q99,
        linestyle="--",
        linewidth=1.5,
        label="Training 99th percentile"
    )

    ax2.axvline(
        train_max_flow,
        linestyle=":",
        linewidth=1.5,
        label="Training maximum"
    )

    ax2.set_xlabel(
        "Streamflow (mm/day)"
    )

    ax2.set_ylabel(
        "Training frequency"
    )

    ax2.set_title(
        "Training Streamflow Distribution vs Extreme Events"
    )

    ax2.legend()

    ax2.grid(
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
    print("DISTRIBUTION DIAGNOSTIC COMPLETE")
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