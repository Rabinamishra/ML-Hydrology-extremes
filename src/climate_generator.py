import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

RANDOM_SEED = 42


# ---------------------------------------------------------
# 2. Create output directories
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 3. Create daily time series
# ---------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

dates = pd.date_range(
    start=START_DATE,
    end=END_DATE,
    freq="D"
)

df = pd.DataFrame({"date": dates})

df["day_of_year"] = df["date"].dt.dayofyear


# ---------------------------------------------------------
# 4. Temperature
# ---------------------------------------------------------

# Annual seasonal cycle
seasonal_temperature = (
    10
    + 10 * np.sin(
        2 * np.pi * (df["day_of_year"] - 100) / 365.25
    )
)

# Daily weather variability
temperature_noise = rng.normal(
    loc=0,
    scale=3,
    size=len(df)
)

df["temperature_C"] = (
    seasonal_temperature
    + temperature_noise
)


# ---------------------------------------------------------
# 5. Precipitation occurrence
# ---------------------------------------------------------

# Seasonal probability of a wet day.
# Higher probability during the middle of the year.

seasonal_wet_probability = (
    0.20
    + 0.25
    * (
        0.5
        + 0.5
        * np.sin(
            2 * np.pi * (df["day_of_year"] - 100) / 365.25
        )
    )
)

wet_day = (
    rng.random(len(df))
    < seasonal_wet_probability
)


# ---------------------------------------------------------
# 6. Rainfall amount
# ---------------------------------------------------------

# Gamma distribution produces positive,
# right-skewed rainfall amounts.

rainfall_amount = rng.gamma(
    shape=1.5,
    scale=10,
    size=len(df)
)

precipitation = np.where(
    wet_day,
    rainfall_amount,
    0.0
)


# ---------------------------------------------------------
# 7. Add occasional extreme precipitation events
# ---------------------------------------------------------

extreme_event_probability = 0.004

extreme_event = (
    rng.random(len(df))
    < extreme_event_probability
)

extreme_rainfall = rng.gamma(
    shape=2.0,
    scale=35,
    size=len(df)
)

precipitation = np.where(
    extreme_event,
    precipitation + extreme_rainfall,
    precipitation
)


# ---------------------------------------------------------
# 8. Add a small persistence effect
# ---------------------------------------------------------

for i in range(1, len(precipitation)):
    if precipitation[i - 1] > 20 and rng.random() < 0.35:
        precipitation[i] += rng.gamma(
            shape=1.5,
            scale=8
        )


df["precipitation_mm"] = precipitation


# ---------------------------------------------------------
# 9. Potential evapotranspiration
# ---------------------------------------------------------

temperature_effect = np.maximum(
    df["temperature_C"] + 2,
    0
)

seasonal_pet = (
    0.7
    + 0.8
    * (
        0.5
        + 0.5
        * np.sin(
            2 * np.pi * (df["day_of_year"] - 100) / 365.25
        )
    )
)

df["pet_mm"] = (
    seasonal_pet
    + 0.08 * temperature_effect
)

df["pet_mm"] = df["pet_mm"].clip(
    lower=0
)


# ---------------------------------------------------------
# 10. Keep useful columns
# ---------------------------------------------------------

df = df[
    [
        "date",
        "temperature_C",
        "precipitation_mm",
        "pet_mm"
    ]
]


# ---------------------------------------------------------
# 11. Save dataset
# ---------------------------------------------------------

output_file = DATA_DIR / "synthetic_climate_daily.csv"

df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# 12. Basic summary
# ---------------------------------------------------------

print("\nClimate dataset created successfully.")
print(f"Observations: {len(df):,}")
print(f"Start: {df['date'].min().date()}")
print(f"End:   {df['date'].max().date()}")
print("\nSummary:")
print(df.describe())


# ---------------------------------------------------------
# 13. Plot climate forcing
# ---------------------------------------------------------

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 10),
    sharex=True
)

axes[0].plot(
    df["date"],
    df["precipitation_mm"],
    linewidth=0.7
)

axes[0].set_ylabel("Precipitation (mm/day)")
axes[0].set_title(
    "Synthetic Daily Climate Forcing"
)


axes[1].plot(
    df["date"],
    df["temperature_C"],
    linewidth=0.7
)

axes[1].set_ylabel("Temperature (°C)")


axes[2].plot(
    df["date"],
    df["pet_mm"],
    linewidth=0.7
)

axes[2].set_ylabel("PET (mm/day)")
axes[2].set_xlabel("Date")


plt.tight_layout()

figure_file = FIGURES_DIR / "climate_forcing.png"

plt.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nFigure saved to: {figure_file}")
print(f"Data saved to:   {output_file}")