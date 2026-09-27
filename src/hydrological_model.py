import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

INPUT_FILE = DATA_DIR / "synthetic_climate_daily.csv"
OUTPUT_FILE = DATA_DIR / "reference_hydrology_daily.csv"


# =========================================================
# Load climate forcing
# =========================================================

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])


# =========================================================
# Model parameters
# =========================================================

# Soil storage
SOIL_CAPACITY = 150.0

# Maximum fraction of available water entering soil
INFILTRATION_RATE = 0.75

# Groundwater recharge fraction
RECHARGE_FRACTION = 0.20

# Maximum evapotranspiration scaling
ET_FACTOR = 0.85

# Groundwater recession coefficient
BASEFLOW_COEFFICIENT = 0.035

# Fraction of soil excess becoming quick runoff
QUICKFLOW_FRACTION = 0.80


# =========================================================
# Initial states
# =========================================================

soil_storage = 75.0
groundwater_storage = 100.0


# Storage and flux arrays
soil_storage_series = []
groundwater_storage_series = []

actual_et_series = []
infiltration_series = []
recharge_series = []
quickflow_series = []
baseflow_series = []
streamflow_series = []


# =========================================================
# Daily simulation
# =========================================================

for _, row in df.iterrows():

    precipitation = row["precipitation_mm"]
    pet = row["pet_mm"]

    # -----------------------------------------------------
    # 1. Available soil storage
    # -----------------------------------------------------

    available_soil_capacity = (
        SOIL_CAPACITY - soil_storage
    )

    # -----------------------------------------------------
    # 2. Infiltration
    # -----------------------------------------------------

    potential_infiltration = (
        precipitation * INFILTRATION_RATE
    )

    infiltration = min(
        potential_infiltration,
        available_soil_capacity
    )

    soil_storage += infiltration

    # Water that cannot infiltrate
    direct_excess = max(
        precipitation - infiltration,
        0
    )

    # -----------------------------------------------------
    # 3. Evapotranspiration
    # -----------------------------------------------------

    actual_et = min(
        soil_storage,
        pet * ET_FACTOR
    )

    soil_storage -= actual_et

    # -----------------------------------------------------
    # 4. Soil-water drainage / recharge
    # -----------------------------------------------------

    # Drainage increases as soil becomes wetter.
    relative_saturation = (
        soil_storage / SOIL_CAPACITY
    )

    drainage = (
        max(soil_storage - 80.0, 0)
        * relative_saturation
        * 0.10
    )

    drainage = min(
        drainage,
        soil_storage
    )

    soil_storage -= drainage

    recharge = drainage * RECHARGE_FRACTION

    quickflow_from_drainage = (
        drainage - recharge
    )

    # -----------------------------------------------------
    # 5. Quick runoff
    # -----------------------------------------------------

    quickflow = (
        direct_excess * QUICKFLOW_FRACTION
        + quickflow_from_drainage
    )

    # -----------------------------------------------------
    # 6. Groundwater storage
    # -----------------------------------------------------

    groundwater_storage += recharge

    # -----------------------------------------------------
    # 7. Baseflow
    # -----------------------------------------------------

    baseflow = (
        BASEFLOW_COEFFICIENT
        * groundwater_storage
    )

    baseflow = min(
        baseflow,
        groundwater_storage
    )

    groundwater_storage -= baseflow

    # -----------------------------------------------------
    # 8. Total streamflow
    # -----------------------------------------------------

    streamflow = quickflow + baseflow

    # -----------------------------------------------------
    # 9. Save states and fluxes
    # -----------------------------------------------------

    soil_storage_series.append(soil_storage)
    groundwater_storage_series.append(
        groundwater_storage
    )

    actual_et_series.append(actual_et)
    infiltration_series.append(infiltration)
    recharge_series.append(recharge)
    quickflow_series.append(quickflow)
    baseflow_series.append(baseflow)
    streamflow_series.append(streamflow)


# =========================================================
# Add model outputs
# =========================================================

df["soil_storage_mm"] = soil_storage_series
df["groundwater_storage_mm"] = groundwater_storage_series

df["actual_et_mm"] = actual_et_series
df["infiltration_mm"] = infiltration_series
df["recharge_mm"] = recharge_series

df["quickflow_mm"] = quickflow_series
df["baseflow_mm"] = baseflow_series

df["streamflow_mm"] = streamflow_series


# =========================================================
# Save
# =========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# Diagnostics
# =========================================================

print("\nREFERENCE HYDROLOGICAL MODEL COMPLETE")
print("=" * 50)

print(f"Days: {len(df):,}")

print(
    f"Total precipitation: "
    f"{df['precipitation_mm'].sum():,.1f} mm"
)

print(
    f"Total streamflow: "
    f"{df['streamflow_mm'].sum():,.1f} mm"
)

print(
    f"Mean streamflow: "
    f"{df['streamflow_mm'].mean():.3f} mm/day"
)

print(
    f"Maximum streamflow: "
    f"{df['streamflow_mm'].max():.2f} mm/day"
)

print(
    f"Maximum soil storage: "
    f"{df['soil_storage_mm'].max():.2f} mm"
)

print(
    f"Maximum groundwater storage: "
    f"{df['groundwater_storage_mm'].max():.2f} mm"
)


# =========================================================
# Figure 1 — Long-term hydrological response
# =========================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(13, 10),
    sharex=True
)

axes[0].plot(
    df["date"],
    df["precipitation_mm"],
    linewidth=0.7
)

axes[0].set_ylabel("P (mm/day)")
axes[0].set_title(
    "Climate Forcing and Reference Hydrological Response"
)


axes[1].plot(
    df["date"],
    df["soil_storage_mm"],
    linewidth=0.8,
    label="Soil storage"
)

axes[1].plot(
    df["date"],
    df["groundwater_storage_mm"],
    linewidth=0.8,
    label="Groundwater storage"
)

axes[1].set_ylabel("Storage (mm)")
axes[1].legend()


axes[2].plot(
    df["date"],
    df["streamflow_mm"],
    linewidth=0.7
)

axes[2].set_ylabel("Streamflow (mm/day)")
axes[2].set_xlabel("Date")


plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "reference_hydrological_response.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# Figure 2 — Example extreme event
# =========================================================

threshold = df["precipitation_mm"].quantile(0.99)

extreme_days = df.index[
    df["precipitation_mm"] >= threshold
]

if len(extreme_days) > 0:

    event_index = extreme_days[
        np.argmax(
            df.loc[
                extreme_days,
                "precipitation_mm"
            ].values
        )
    ]

    start = max(event_index - 10, 0)
    end = min(event_index + 11, len(df))

    event = df.iloc[start:end]

    fig, ax1 = plt.subplots(
        figsize=(12, 5)
    )

    ax1.bar(
        event["date"],
        event["precipitation_mm"],
        width=0.8,
        alpha=0.5
    )

    ax1.set_ylabel(
        "Precipitation (mm/day)"
    )

    ax2 = ax1.twinx()

    ax2.plot(
        event["date"],
        event["streamflow_mm"],
        linewidth=2
    )

    ax2.set_ylabel(
        "Streamflow (mm/day)"
    )

    plt.title(
        "Reference Response to an Extreme Precipitation Event"
    )

    fig.tight_layout()

    plt.savefig(
        FIGURES_DIR / "extreme_event_response.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


print(
    f"\nSaved: {OUTPUT_FILE}"
)

print(
    "Saved figures to figures/"
)