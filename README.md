# Machine-Learning Emulation of Hydrological Response to Climate Variability and Extremes

## Overview

This project develops a Python-based proof-of-concept for machine-learning emulation of hydrological response to climate forcing.

The workflow combines:

**climate forcing → conceptual rainfall–runoff model → machine-learning emulator → temporal validation → extreme-event evaluation → physical interpretation**

The current stage uses a controlled synthetic experiment to investigate whether machine-learning models can reproduce nonlinear hydrological responses and how their performance changes under precipitation extremes.

The project is being developed as a foundation for a subsequent experiment using observed climate and hydrological data.

---

## Research Question

> Can machine-learning models emulate nonlinear hydrological responses to climate forcing, and how does their performance change during precipitation extremes?

A secondary question is:

> Can physically informed representations of antecedent hydrological state improve the emulation of extreme hydrological responses?

---

## Methodology

### 1. Synthetic climate forcing

Daily precipitation, temperature and potential evapotranspiration are generated for 2015–2024.

The synthetic forcing includes:

* seasonal variability
* stochastic precipitation
* wet and dry periods
* occasional high-precipitation events
* temperature-dependent potential evapotranspiration

The synthetic dataset is used as a controlled proof-of-concept and is not intended to represent a specific geographic location.

### 2. Conceptual hydrological model

The climate forcing is passed through a conceptual rainfall–runoff model representing:

* soil storage
* infiltration
* evapotranspiration
* groundwater storage
* recharge
* quickflow
* baseflow
* total streamflow

This produces a reference hydrological response against which the machine-learning emulator can be evaluated.

### 3. Machine-learning emulation

Random Forest regression is used to emulate daily streamflow.

The main experiments compare:

* climate forcing alone
* climate forcing with hydrological memory
* physically informed saturation/antecedent-state features

The evaluation uses a chronological split rather than random cross-validation:

**Training:** 2015–2021
**Testing:** 2022–2024

This preserves the temporal structure of the experiment and tests forward generalization.

---

## Extreme-Event Evaluation

Precipitation thresholds are calculated from the training period only:

* 95th percentile: **27.92 mm/day**
* 99th percentile: **51.57 mm/day**

The test period is divided into:

* non-extreme events
* moderate extremes (95th–99th percentile)
* top 1% precipitation events

This allows model performance to be evaluated separately for ordinary and extreme conditions.

---

## Main Findings

### Hydrological response

The conceptual model shows a nonlinear increase in quickflow during high-precipitation events.

For the test-period event classes:

| Condition        | Mean precipitation | Mean quickflow | Mean streamflow |
| ---------------- | -----------------: | -------------: | --------------: |
| Non-extreme      |        3.17 mm/day |    2.06 mm/day |     2.44 mm/day |
| Moderate extreme |       37.00 mm/day |   12.44 mm/day |    12.81 mm/day |
| Top 1%           |      100.57 mm/day |   45.17 mm/day |    45.50 mm/day |

This indicates an increasing shift from infiltration/recharge toward rapid runoff during intense precipitation.

### Machine-learning performance

The baseline emulator reproduced ordinary hydrological behavior substantially better than the rarest extreme events.

Adding hydrological memory improved performance under ordinary conditions, but did not resolve the largest precipitation extremes.

The saturation-aware experiment substantially improved representation of moderate extremes:

| Model                         | Non-extreme R² | Moderate extreme R² | Top 1% R² |
| ----------------------------- | -------------: | ------------------: | --------: |
| Climate only                  |          0.826 |               0.716 |     0.011 |
| Climate + hydrological memory |          0.900 |               0.716 |    -0.103 |
| Saturation-aware              |      **0.992** |           **0.965** | **0.168** |

The rarest events remained difficult to reproduce. For the top 1% events, the saturation-aware model achieved:

* MAE: **22.88 mm/day**
* RMSE: **57.87 mm/day**
* R²: **0.168**

The largest test event occurred on 10 March 2023, with approximately 323.95 mm/day of precipitation and 220.76 mm/day of reference streamflow. This event exceeded the maximum precipitation and streamflow values observed in the training period, indicating substantial distribution shift.

However, distribution shift alone did not explain all top-1% errors, since most top-1% test events remained within the training precipitation range.

---

## Interpretation

The synthetic experiment suggests that machine-learning emulation benefits from information describing the antecedent state of the hydrological system.

Physically informed saturation and wetness features substantially improved the representation of moderate extremes, indicating that hydrological state can help the emulator represent nonlinear rainfall–runoff behavior.

However, the rarest extreme events remained difficult to reproduce. This highlights the challenges associated with sparse extremes, nonlinear runoff generation and distribution shift.

These results motivate extending the workflow to observed climate and hydrological datasets.

---

## Important Limitations

This is a **synthetic proof-of-concept**, not a calibrated hydrological model for a real catchment.

The synthetic climate forcing does not represent Bergen, Nepal, or another specific location.

The conceptual rainfall–runoff model is intentionally simplified.

The saturation-aware model also uses hydrological state information derived from the conceptual reference model. Therefore, its performance should be interpreted as an experiment in physically informed emulation rather than evidence of operational forecasting skill.

The top-1% evaluation contains only nine test events, so those statistics should not be overinterpreted.

---

## Next Stage

The next stage will transfer the workflow to an observed climate–hydrology dataset.

The intended workflow is:

**observed climate forcing → hydrological response → temporal ML emulation → extreme-event evaluation → transfer/generalization analysis**

The real-world experiment will provide an opportunity to evaluate whether the patterns identified in the synthetic proof-of-concept persist under observed hydroclimatic variability.
