# Spatiotemporal Crime Classification with Transfer Learning

> Grid-level crime hotspot prediction across **14 cities worldwide**, with cross-city zero-shot transfer learning and an interactive browser map.

---

## Overview

This project predicts the **dominant crime type** (violent / property / other) for geographic grid cells across 14 cities, and studies whether a model trained in one city can transfer to another without retraining.

**Key findings:**
- Zero-shot transfer **NYC → Chicago** achieves precision **0.614**, outperforming Chicago's locally-trained model (0.439) by **+17.5 percentage points**
- Just **3 historical composition features** (`hist_violent`, `hist_property`, `hist_other`) outperform the full 26-feature model
- Cross-cultural transfer (NYC → Karachi) fails due to structural differences in crime reporting — confirmed identical results from two source cities
- Fine-tuning with target city data causes **negative transfer** in all tested scenarios

---

## Cities & Data

| City | Country | Source | Records | Years |
|------|---------|--------|---------|-------|
| New York City | USA | NYC OpenData (Socrata) | 9,481,523 | 2006–2024 |
| Chicago | USA | Chicago Data Portal (Socrata) | 6,626,073 | 2001–2024 |
| Los Angeles | USA | LA Open Data | 3,134,838 | 2010–2024 |
| London | UK | data.police.uk | 3,416,746 | 2015–2024 |
| West Yorkshire | UK | data.police.uk | 916,815 | 2023–2026 |
| Philadelphia | USA | OpenDataPhilly | 899,422 | 2006–2024 |
| Detroit | USA | ArcGIS FeatureServer | 625,766 | 2009–2024 |
| Kansas City | USA | KCMO Open Data (Socrata) | 569,639 | 2016–2024 |
| Washington DC | USA | Open Data DC | 180,924 | 2008–2024 |
| Karachi | Pakistan | Kaggle synthetic dataset | 100,000 | 2020–2025 |
| Salt Lake City | USA | Utah Open Data | 82,812 | 2021–2024 |
| Peoria | USA | ArcGIS FeatureServer | 67,086 | 2023–2026 |
| Cambridge | UK | data.police.uk | 38,153 | 2021–2024 |
| Birmingham (AL) | USA | ArcGIS FeatureServer | 20,702 | 2021–2024 |

**Total: ~26 million records across 14 cities**

---

## Model Performance

| City | Grids | Map Accuracy | Best Precision (macro) | Avg Confidence |
|------|-------|-------------|----------------------|---------------|
| NYC | 3,503 | 71.2% | 0.724 | 0.786 |
| Chicago | 2,740 | 81.8% | 0.474 (local) / **0.614** (zero-shot from NYC) | 0.743 |
| LA | 4,668 | 85.2% | 0.520 | 0.791 |
| London | 7,348 | 58.4% | 0.550 | 0.583 |
| West Yorkshire | 4,940 | 69.3% | 0.548 | 0.685 |
| Philadelphia | 1,495 | 81.5% | 0.558 | 0.876 |
| Detroit | 1,774 | 97.6% | 0.826 | 0.989 |
| Kansas City | 2,936 | 74.9% | 0.678 | 0.681 |
| DC | 633 | 96.5% | 0.531 | 0.590 |
| Karachi | 153 | 80.4% | 0.939 | 0.817 |
| Salt Lake City | 548 | 97.4% | 0.937 | 0.979 |
| Peoria | 847 | 88.1% | 0.850 | 0.902 |
| Cambridge | 540 | 92.8% | 1.000 | 0.972 |
| Birmingham | 465 | 94.4% | 1.000 | 0.997 |

---

## Transfer Learning Results

### Same-country: NYC → Chicago

| Scenario | Precision Macro | Note |
|----------|----------------|------|
| Chicago local baseline | 0.439 | Trained from scratch on Chicago data |
| **Zero-shot NYC → Chicago** | **0.614** | No Chicago data used at all |
| Fine-tune 10% Chicago data | 0.437 | Negative transfer |
| Fine-tune 50% Chicago data | 0.439 | Negative transfer |
| Teacher-Student (T=3.0) | ~0.422 | Soft label distillation |

### Cross-cultural: NYC / Chicago → Karachi

| Scenario | Precision Macro |
|----------|----------------|
| Karachi local baseline | 0.625 |
| Zero-shot NYC → Karachi | 0.389 |
| Zero-shot Chicago → Karachi | 0.389 |

Both US source cities produce identical results — the barrier is structural/cultural, not city-specific.

---

## Feature Ablation Study (NYC)

| Feature Group | # Features | Precision Macro |
|--------------|-----------|----------------|
| **hist_* only** | **3** | **0.649** |
| All features | 26 | 0.567 |
| hist_* + lag_* | 6 | 0.566 |
| No hist_* | 23 | 0.520 |
| Spatial + Temporal only | 13 | 0.464 |

`hist_*` features alone beat the full 26-feature model.

---

## Methodology

### Task Definition

Each prediction unit is a **grid cell × time slot** pair:
- **Spatial grid:** 0.01° × 0.01° (~1 km²), minimum 3 incidents per cell
- **Time slots:** 4 per day — midnight (0–5h), morning (6–11h), afternoon (12–17h), night (18–23h)
- **Target:** dominant crime category (most frequent type) in that grid-timeslot

### Data Split (Temporal — no leakage)

```
Train  ─── earliest 74% by datetime  ← hist_* features computed HERE only
Val    ─── next 13%                  ← calibration, early stopping
Test   ─── most recent 13%           ← final evaluation, not touched during training
```

Short-window cities (< 4 years): 60/20/20 split.

### Feature Engineering

| Group | Features | Description |
|-------|----------|-------------|
| Historical composition | `hist_violent`, `hist_property`, `hist_other` | Grid's crime type ratios in training period |
| Spatial lag | `lag_violent`, `lag_property`, `lag_other` | Neighboring grids' average composition (KDTree k=4) |
| Relative percentile | `violent_pct`, `density_pct` | Rank within the city |
| Temporal | `time_slot`, `ts_sin`, `ts_cos`, `month_sin`, `month_cos`, `weekday_sin`, `weekday_cos`, `is_weekend` | Cyclical time encoding |
| Spatial | `lat_bin`, `lon_bin`, `lat_norm`, `lon_norm` | Grid coordinates |

### Crime Category Mapping

All cities are mapped to 3 unified categories:

| Category | Examples |
|----------|---------|
| `violent` | murder, assault, robbery, rape, kidnapping |
| `property` | theft, burglary, fraud, arson, vandalism |
| `other` | drug offenses, public disorder, traffic violations |

### Model Pipeline

1. **CatBoost + LightGBM** ensemble with balanced class weights
2. **Isotonic regression** probability calibration on validation set
3. Ensemble: average of calibrated CatBoost and LightGBM probabilities
4. Final prediction: argmax of calibrated ensemble probabilities

---

## Project Structure

```
model-predict-crime/
├── src/
│   ├── 01_download.py                    # NYC, Chicago, LA auto-download
│   ├── 02_preprocess.py                  # Standardize to unified CSV schema
│   ├── download_peoria.py                # ArcGIS FeatureServer download
│   ├── download_kansas_city.py           # Socrata multi-year download
│   ├── download_detroit.py
│   ├── download_birmingham.py
│   ├── download_slc.py
│   └── download_cambridge.py
├── notebook/
│   ├── clean_*.py                        # City-specific cleaning scripts
│   ├── crime_classification_full_NYC.py  # Main pipeline (all cities)
│   ├── crime_classification_full_*.py    # Per-city classification scripts
│   ├── dann_crime.py                     # DANN v1 domain adaptation
│   └── dann_v2.py                        # DANN v2 (4 improvements)
├── build_all_cities.py                   # Merge all city CSVs → all_cities.csv
├── update_map_v*.py                      # Map update scripts
├── data/
│   ├── raw/                              # Downloaded CSVs (not committed)
│   └── processed/                        # Cleaned CSVs per city + all_cities.csv
└── outputs/
    ├── models/                           # Trained models, calibrators, grid_risk CSVs
    ├── eda/                              # Confusion matrices, SHAP, ablation charts
    └── maps/
        └── crime_map_v6.html             # Interactive map — 13 cities, no backend needed
```

---

## Interactive Map

`outputs/maps/crime_map_v6.html` — standalone HTML, open directly in any browser.

**Features:**
- 13 cities switchable via tabs
- 4 time slot animation (auto-cycle every 1.8s)
- Click any grid cell for details: predicted category, confidence, risk score, probability bar chart
- Alert threshold slider: flag high-risk grids
- Route risk query: enter start/end coordinates, get average risk along path
- Top-10 highest-risk grids panel with map fly-to
- Dark / light theme toggle

**Grid color coding:**
- Red = violent, Blue = property, Green = other
- Opacity = confidence (high ≥ 0.80, medium 0.55–0.80, low < 0.55)

---

## Setup

```bash
conda create -n crime-tl python=3.10 -y
conda activate crime-tl
pip install pandas==2.2.2 numpy==1.26.4 scikit-learn==1.4.2 \
            catboost lightgbm xgboost==2.0.3 scipy joblib \
            folium==0.16.0 matplotlib==3.9.0 seaborn==0.13.2 \
            tqdm==4.66.4 jupyter torch
```

> `requirements.txt` is incomplete — install from the command above instead.

### Run the pipeline

```bash
# Step 1: Download raw data
python src/01_download.py          # NYC, Chicago, LA (automatic)
# Other cities: run src/download_*.py individually

# Step 2: Clean each city
python notebook/clean_kansas_city.py
python notebook/clean_peoria.py
# ... etc.

# Step 3: Merge all cities
python build_all_cities.py

# Step 4: Train models (one city at a time)
python notebook/crime_classification_full_NYC.py
python notebook/crime_classification_full_KansasCity.py
# ... etc.

# Step 5: Update interactive map
python update_map_v6.py
```

---

## DANN (Domain Adversarial Neural Network)

Two versions for cross-city domain adaptation:

**DANN v1** (`notebook/dann_crime.py`):
- Architecture: 26-dim → 128 → 64 → 3 classes
- Gradient Reversal Layer aligns source and target feature distributions
- Loss: `L = L_class(F,C) − λ·L_domain(F,D)`

**DANN v2** (`notebook/dann_v2.py`) — 4 improvements over v1:
1. Random Forest pre-training initializes feature extractor
2. RF feature importance selects top-K features to reduce alignment dimension
3. Ensemble teacher (RF + LightGBM + XGBoost) provides soft labels
4. JSD-adaptive λ scheduling: `lambda_max = 0.5 × exp(−mean_JSD)`

```bash
python notebook/dann_crime.py --source NYC --target Chicago
python notebook/dann_v2.py --source NYC --target Chicago --epochs 100 --top_k 12
python notebook/dann_v2.py --all --epochs 100   # all city pairs
```

---

## Key Insights

1. **Grid framing enables high precision** — event-level prediction ceiling ~0.35; grid-level achieves 0.65–0.99
2. **Historical composition is everything** — 3 `hist_*` features beat 26 features combined
3. **Zero-shot same-country transfer works** — crime patterns are geographically transferable within similar urban contexts
4. **Fine-tuning hurts** — adding local data reintroduces city-specific noise, causing negative transfer
5. **Cultural barrier is real** — cross-cultural transfer fails regardless of source city; the `hist_*` features that drive success are also the least transferable across cultures
6. **Unbalanced cities have misleadingly high accuracy** — DC (97% property) and Birmingham (89% property) achieve high map accuracy by predicting the dominant class; London (balanced distribution) is genuinely harder
