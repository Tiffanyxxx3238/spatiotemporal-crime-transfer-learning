# Spatiotemporal Crime Classification with Transfer Learning

> Grid-level crime hotspot prediction across **14 cities worldwide**, with cross-city zero-shot transfer learning, seasonal pattern analysis, and an interactive browser map.

---

## Overview

This project predicts the **dominant crime type** (violent / property / other) for geographic grid cells across 14 cities, and studies whether a model trained in one city can transfer to another without retraining.

Each prediction unit is a **(grid cell × time slot × month)** triple — capturing both diurnal and seasonal crime rhythms.

**Key findings:**
- Zero-shot transfer **NYC → Chicago** achieves precision **0.614**, outperforming Chicago's locally-trained model (0.439) by **+17.5 percentage points**
- Just **3 historical composition features** (`hist_violent`, `hist_property`, `hist_other`) outperform the full 26-feature model
- Adding **month** to the grid groupby (Method B) improves Kansas City precision by **+3.0 pp** and enables season-level filtering in the interactive map
- Cross-cultural transfer (NYC → Karachi) fails due to structural differences in crime reporting — confirmed identical results from two source cities
- Fine-tuning with target city data causes **negative transfer** in all tested scenarios

---

## Cities & Data

| City | Country | Source | Records | Years |
|------|---------|--------|--------:|-------|
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

## Model Performance (Method B — with Month)

Grid cells are grouped by `(lat_bin, lon_bin, time_slot, month)`. Compared to the original time-slot-only grouping, grid counts are 4–10× larger, capturing seasonal variation.

| City | Grids (w/ month) | Map Accuracy | Note |
|------|----------------:|------------:|------|
| NYC | 36,251 | 62.1% | |
| Chicago | 28,194 | 69.2% | |
| LA | 37,716 | 67.3% | |
| London | 41,946 | 45.9% | Balanced crime distribution → harder task |
| West Yorkshire | 17,329 | 63.3% | |
| Philadelphia | 13,008 | 74.2% | |
| Detroit | 15,856 | 93.2% | |
| Kansas City | 20,123 | 67.7% | |
| DC | 3,783 | 92.9% | |
| Karachi | 1,677 | 56.3% | Synthetic data |
| Salt Lake City | 1,242 | 99.8% | |
| Peoria | 3,889 | 82.6% | |
| Cambridge | 854 | 100.0% | Small dataset, highly separable |
| Birmingham | 284 | 98.9% | Dominant property class |

> **Map Accuracy** = fraction of grids where model prediction matches true dominant category.  
> High accuracy for Cambridge / SLC / Birmingham partly reflects class imbalance (one dominant crime type).

---

## Transfer Learning Results

### Same-country: NYC → Chicago

| Scenario | Precision Macro | Note |
|----------|----------------:|------|
| Chicago local baseline | 0.439 | Trained from scratch on Chicago data |
| **Zero-shot NYC → Chicago** | **0.614** | No Chicago data used at all |
| Fine-tune 10% Chicago data | 0.437 | Negative transfer |
| Fine-tune 50% Chicago data | 0.439 | Negative transfer |
| Teacher-Student (T=3.0) | ~0.422 | Soft label distillation |

### Cross-cultural: NYC / Chicago → Karachi

| Scenario | Precision Macro |
|----------|----------------:|
| Karachi local baseline | 0.625 |
| Zero-shot NYC → Karachi | 0.389 |
| Zero-shot Chicago → Karachi | 0.389 |

Both US source cities produce identical results — the barrier is structural/cultural, not city-specific.

---

## Feature Ablation Study (NYC)

| Feature Group | # Features | Precision Macro |
|--------------|----------:|----------------:|
| **hist_* only** | **3** | **0.649** |
| All features | 26 | 0.567 |
| hist_* + lag_* | 6 | 0.566 |
| No hist_* | 23 | 0.520 |
| Spatial + Temporal only | 13 | 0.464 |

`hist_*` features (grid's historical crime composition from training data) alone beat the full 26-feature model.

---

## Methodology

### Task Definition

Each prediction unit: **(lat_bin × lon_bin) × time_slot × month**

| Dimension | Detail |
|-----------|--------|
| Spatial grid | 0.01° × 0.01° ≈ 1 km², minimum 3 incidents per cell |
| Time slots | 4 per day — midnight (0–5h), morning (6–11h), afternoon (12–17h), night (18–23h) |
| Month | 1–12 (Method B — enables seasonal analysis) |
| Target | Dominant crime category (most frequent type) in that grid-slot-month |

### Data Split (Temporal — no leakage)

```
Train  ─── earliest 74% by datetime  ← hist_* features computed HERE only
Val    ─── next 13%                  ← calibration, early stopping
Test   ─── most recent 13%           ← final evaluation, not touched during training
```

Short-window cities (< 4 years): 60/20/20 split.

### Feature Engineering (26 features total)

| Group | Features | Description |
|-------|----------|-------------|
| Historical composition | `hist_violent`, `hist_property`, `hist_other` | Grid's crime type ratios in **training period only** |
| Spatial lag | `lag_violent`, `lag_property`, `lag_other` | Neighboring grids' average composition (KDTree k=4) |
| Relative percentile | `violent_pct`, `density_pct`, `entropy_pct`, `dom_gap_pct` | Rank within city |
| Temporal | `time_slot`, `ts_sin/cos`, `month`, `month_sin/cos`, `weekday_sin/cos`, `is_weekend` | Cyclical encoding |
| Spatial | `lat_bin`, `lon_bin`, `lat_norm`, `lon_norm` | Grid coordinates |

### Crime Category Mapping

All cities are unified to 3 categories:

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

## Interactive Map

`outputs/maps/crime_map_v8.html` — standalone HTML (42 MB), open directly in any browser. No backend required. All 14-city data is embedded as static JSON.

### Features

| Feature | Description |
|---------|-------------|
| 14-city tabs | NYC / Chicago / LA / London / Philadelphia / DC / West Yorkshire / Detroit / Kansas City / Peoria / Cambridge / Salt Lake City / Birmingham / Karachi |
| Time slot animation | 4 slots auto-cycle every 1.8s, or click to select |
| **Season filter** | 春 Spring [3–5] / 夏 Summer [6–8] / 秋 Fall [9–11] / 冬 Winter [12,1,2] / 全年 All |
| Grid click details | coordinates, true category, predicted category (✓/✗), confidence, count, risk score, probability bar chart |
| Alert threshold | Slider sets risk threshold; grids above it are flagged with badges |
| Route risk query | Enter start/end coordinates → average/max risk and violent fraction along route |
| Top-10 risk panel | Ranked high-risk grids for current city+timeslot; click → fly to grid |
| Time distribution chart | Per-timeslot violent/property/other stacked bars; click → switch map timeslot |
| Dark/light theme | Switches CartoDB dark_all ↔ light_all basemap |

### Grid Color Coding

- **Red** = violent, **Blue** = property, **Green** = other
- **Opacity** = confidence (high ≥ 0.80, medium 0.55–0.80, low < 0.55)

---

## Project Structure

```
model-predict-crime/
├── src/
│   ├── 01_download.py                    # NYC, Chicago, LA auto-download (Socrata)
│   ├── 02_preprocess.py                  # Standardize all cities to unified CSV schema
│   ├── download_peoria.py                # ArcGIS FeatureServer download
│   ├── download_kansas_city.py           # Socrata multi-year download
│   ├── download_detroit.py
│   ├── download_birmingham.py
│   ├── download_slc.py
│   └── download_cambridge.py
├── notebook/
│   ├── clean_*.py                        # City-specific cleaning scripts
│   ├── crime_classification_full_NYC.py  # Main pipeline template (all large cities)
│   ├── crime_classification_full_KansasCity.py
│   ├── crime_classification_full_Peoria.py
│   ├── crime_classification_full_*.py    # One script per city
│   ├── dann_crime.py                     # DANN v1 domain adaptation
│   └── dann_v2.py                        # DANN v2 (4 improvements)
├── build_all_cities.py                   # Merge per-city CSVs → all_cities.csv
├── patch_method_b_all.py                 # Batch-patched all scripts to Method B
├── update_map_v8.py                      # Regenerate interactive map from grid_risk CSVs
├── data/
│   ├── raw/                              # Downloaded CSVs (not committed to git)
│   └── processed/
│       ├── all_cities.csv               # All 14 cities merged (~2 GB)
│       └── *_clean.csv                  # Per-city cleaned CSV
└── outputs/
    ├── models/
    │   ├── model_*_catboost.cbm         # Trained CatBoost models
    │   ├── model_*_lgb.pkl              # Trained LightGBM models
    │   ├── cal_iso_*.pkl                # Isotonic calibrators
    │   ├── label_encoder_*.pkl          # Label encoders
    │   └── grid_risk_*.csv             # Per-city grid risk scores (with month)
    ├── eda/                             # Confusion matrices, SHAP, ablation charts
    └── maps/
        └── crime_map_v8.html           # Interactive map — 14 cities, season filter
```

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

### Run the Full Pipeline

```bash
# Step 1: Download raw data
python src/01_download.py          # NYC, Chicago, LA (automatic via Socrata API)
python src/download_kansas_city.py # Other cities individually
python src/download_peoria.py
# ... (see src/ for each city's downloader)

# Step 2: Clean each city
python notebook/clean_dc.py
python notebook/clean_philadelphia.py
python notebook/clean_westyorkshire.py
python notebook/clean_kansas_city.py
python notebook/clean_peoria.py
# ... (see notebook/clean_*.py for each city)

# Step 3: Merge all cities into one CSV
python build_all_cities.py         # → data/processed/all_cities.csv (~2 GB)

# Step 4: Train models per city (run from project root)
python notebook/crime_classification_full_NYC.py
python notebook/crime_classification_full_KansasCity.py
# ... (repeat for all 14 cities)

# Step 5: Regenerate interactive map
python update_map_v8.py            # → outputs/maps/crime_map_v8.html
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
2. RF feature importance selects top-K features (reduces alignment dimension)
3. Ensemble teacher (RF + LightGBM + XGBoost) provides soft labels
4. JSD-adaptive λ scheduling: `lambda_max = 0.5 × exp(−mean_JSD)`

```bash
python notebook/dann_crime.py --source NYC --target Chicago
python notebook/dann_v2.py --source NYC --target Chicago --epochs 100 --top_k 12
python notebook/dann_v2.py --all --epochs 100   # all city pairs
```

---

## Key Insights

1. **Grid framing enables high precision** — event-level prediction ceiling ~0.35; grid-level achieves 0.62–1.00 depending on city
2. **Historical composition is everything** — 3 `hist_*` features beat 26 features combined (0.649 vs 0.567)
3. **Seasonal patterns matter** — adding `month` to groupby (Method B) improves precision +3 pp and makes seasonal crime trends visible
4. **Zero-shot same-country transfer works** — NYC patterns transfer to Chicago out-of-the-box, outperforming local training by +17.5 pp
5. **Fine-tuning hurts** — adding local data reintroduces city-specific noise, causing negative transfer
6. **Cultural barrier is real** — cross-cultural transfer fails regardless of source city; `hist_*` features that drive in-country success are the least portable across cultures
7. **Class imbalance inflates accuracy** — DC (97% property) and Cambridge (uniform area) achieve near-perfect map accuracy by class dominance; London (balanced crime types) is the genuinely hardest task
