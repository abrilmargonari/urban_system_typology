# Urban System Typology Classification with Random Forest and Spatial Validation

This repository contains the code for a doctoral research project that classifies urban typologies using a **Random Forest** model, with a strong focus on **spatial cross‑validation** and **residual autocorrelation analysis**. The study area covers a heterogeneous region (large cities, rural areas, mountains).

## Features

- **Random Forest** optimized with `RandomizedSearchCV`.
- **Gradual class weighting** to handle imbalanced typologies.
- **Spatial cross‑validation** using DBSCAN clustering (ε=500 m, min_samples=5) and `GroupKFold`.
- **Moran's I (global and local)** to test residual spatial autocorrelation.
- **Confidence calculation** for 7 typology classes (without age suffix).
- **Prediction** on large unlabeled datasets (GeoPackage with millions of objects).

## Input Data

The model expects a **GeoPackage** (`.gpkg`) with the following 13 predictor variables (all numeric variables are scaled):

| Variable | Description |
|----------|-------------|
| `code_osm` | OSM land use code (categorical) |
| `level` | Number of floors (scaled) |
| `built_area_block` | Built area per block (scaled) |
| `dist_informal` | Distance to informal settlements (scaled) |
| `area_block` | Block area (scaled) |
| `dist_block` | Distance to nearest block (scaled) |
| `dist_median_build` | Median distance to buildings (scaled) |
| `dist_river` | Distance to river (scaled) |
| `dist_min_build` | Distance to nearest building (scaled) |
| `dist_center` | Distance to city center (scaled) |
| `cluster_block` | Block cluster ID (categorical, based on shape/orientation) |
| `code_dist_via` | Road OSM code (categorical) |
| `dist_via` | Distance to road (scaled) |

Target variable: `class` (typology with age suffix, e.g., `"residential_1980"`). The model also works with **7 typology classes without age**.

## Workflow

1. Load GeoPackage and extract coordinates.
2. Clean and encode target variable.
3. Split data (80% train, 20% test, stratified).
4. Hyperparameter optimization (RandomizedSearchCV, 3‑fold CV, F1‑weighted).
5. Train final Random Forest.
6. Evaluate on test set (accuracy, F1 with/without age).
7. Spatial cross‑validation (DBSCAN + GroupKFold, 5 folds).
8. Moran's I test on residuals.
9. Predict on unlabeled data (optional, requires separate GeoPackage).
10. Generate plots: F1 comparison, CV results, feature importance, confidence distribution, LISA map.

## Results (summary)

| Validation type | F1 (with age) | F1 (without age) |
|----------------|---------------|------------------|
| Test set (random split) | ~0.78 | ~0.90 |
| Stratified CV (random) | ~0.64 | ~0.83 |
| Spatial CV (DBSCAN) | ~0.35 | ~0.68 (±0.06) |

Global Moran's I on residuals: **-0.0206 (p=0.142)** → no significant spatial autocorrelation.


## From typology to economic valuation: local adaptation
The typology classes predicted by this model are aligned with internationally recognized taxonomies for exposure and vulnerability assessment (GEM and PAGER). However, the economic valuation (replacement costs, depreciation curves, etc.) is highly context‑dependent. Therefore, the methodology is designed to be locally calibrated using:

Local construction cost databases (public or market surveys)

Age‑depreciation curves that reflect actual building deterioration patterns in the region

Specific coefficients for heritage buildings, special investment zones, etc.

The following steps illustrate how the predicted typology can be converted into a current built‑up valuation (e.g., for risk assessment or cadastral purposes). All numerical values in the table below are examples only; they must be replaced with locally validated figures.

## Workflow

Typology Prediction
        ↓
Unit Replacement Cost (USD/m²) ← Local database
        ↓
Building Area (m²) ← from footprint
        ↓
Replacement Value = Unit Cost × Area
        ↓
Age Category (decade) ← predicted by model
        ↓
Depreciation Factor ← local age‑depreciation curve
        ↓
Current Built‑Up Valuation = Replacement Value × Depreciation Factor


## Indicative replacement costs (for illustration only)
Typology	Replacement Cost (USD/m², 2020)
Residential (high)	2000‑4000+
Residential (medium)	1200‑2000
Residential (low)	1000‑1600
Informal	200‑400
High‑rise mixed‑use (C3)	2300‑3500+
Industrial / Agro‑industrial	1000‑1500
Extractive	5500‑7000+

⚠️ Important: The numbers above are not universal. They are derived from a specific local construction cost survey (Central‑Pampean region, Argentina, 2020). If you apply this methodology elsewhere, you must obtain your own local replacement costs and depreciation curves.

The depreciation factor for each age decade is computed as:

Factor = Depreciated value in decade / Replacement cost (2020)

Once the current built‑up valuation is obtained per building, it can be aggregated at any spatial scale (parcel, block, neighbourhood, watershed) to support risk models, land‑use planning, or insurance portfolio assessments.

By making the entire chain (typology estimation → replacement cost → depreciation) open‑source and transparent, we empower local institutions to carry out their own risk and capital assessments without relying on proprietary black‑box models. The code and data used in this research are entirely open, demonstrating that high‑quality territorial knowledge can be generated without dependence on proprietary tools.


⚠️ Note on result variability: The metrics shown above were computed on a reduced sample (1,374 observations). 
With a smaller or differently distributed dataset, you may obtain higher variance in spatial 
cross‑validation and significant residual spatial autocorrelation (as seen here: Moran's I = 0.1426, p = 0.003). 
For reliable results, use a larger, spatially balanced sample. 
The original research (with >3,800 samples) found no significant autocorrelation and more stable spatial folds.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abrilmargonari/urban_system_typology.git
   cd urban_system_typology