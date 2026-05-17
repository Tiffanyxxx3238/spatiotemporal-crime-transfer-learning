# Spatiotemporal Crime Classification with Transfer Learning

> Grid-level crime hotspot prediction across **17 cities worldwide**, with cross-city zero-shot transfer learning, seasonal pattern analysis, and an interactive browser map.

---

## Overview

This project predicts the **dominant crime type** (violent / property / other) for geographic grid cells across 17 cities, and studies whether a model trained in one city can transfer to another without retraining.

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
| Dallas | USA | Dallas Open Data (Socrata) | 1,468,929 | 2014–2024 |
| San Francisco | USA | SF Open Data (Socrata) | 1,073,954 | 2003–2024 |
| Seattle | USA | Seattle Open Data (Socrata) | 1,138,618 | 2008–2024 |
| Washington DC | USA | Open Data DC | 180,924 | 2008–2024 |
| Karachi | Pakistan | Kaggle synthetic dataset | 100,000 | 2020–2025 |
| Salt Lake City | USA | Utah Open Data | 82,812 | 2021–2024 |
| Peoria | USA | ArcGIS FeatureServer | 67,086 | 2023–2026 |
| Cambridge | UK | data.police.uk | 38,153 | 2021–2024 |
| Birmingham (AL) | USA | ArcGIS FeatureServer | 20,702 | 2021–2024 |

**Total: ~30 million records across 17 cities**

> **Karachi note**: The Kaggle synthetic dataset has `hour=0` for all records — no time-of-day dimension. The map displays all Karachi grids regardless of selected time slot, and the time slider is disabled for this city.

### Planned additions (download scripts ready)

| City | State | Source | Status |
|------|-------|--------|--------|
| Lansing | MI | ArcGIS FeatureServer | Script ready — `src/download_lansing_dayton_littlerock.py` |
| Dayton | OH | ArcGIS FeatureServer | Script ready |
| Little Rock | AR | ArcGIS FeatureServer | Script ready |

---

## Model Performance (Method B — with Month)

Grid cells are grouped by `(lat_bin, lon_bin, time_slot, month)`. Compared to the original time-slot-only grouping, grid counts are 4–10× larger, capturing seasonal variation.

| City | Grid-Slot-Month Rows | Unique Grid Cells | Map Accuracy | Note |
|------|--------------------:|------------------:|------------:|------|
| NYC | 36,251 | 869 | 62.1% | |
| Chicago | 28,194 | 672 | 69.2% | |
| LA | 37,716 | 1,134 | 67.3% | |
| London | 41,946 | 1,783 | 45.9% | Balanced crime distribution → harder task |
| West Yorkshire | 17,329 | 1,108 | 63.3% | |
| Philadelphia | 13,008 | 378 | 74.2% | |
| Detroit | 15,856 | 442 | 93.2% | |
| Kansas City | 20,123 | 842 | 67.7% | |
| Dallas | 18,756 | 804 | 70.5% | Class collapse — model predicts "other" for all grids |
| San Francisco | 4,893 | 133 | 65.7% | Strong property dominance |
| Seattle | 9,094 | 304 | 71.9% | |
| DC | 3,783 | 151 | 92.9% | |
| Karachi | 1,677 | 153 | 56.3% | Synthetic data, no time-of-day dimension |
| Salt Lake City | 1,242 | 134 | 99.8% | |
| Peoria | 3,889 | 245 | 82.6% | |
| Cambridge | 854 | 122 | 100.0% | Small dataset, highly separable |
| Birmingham | 284 | 96 | 98.9% | Dominant property class |

> **Map Accuracy** = fraction of grid-slot-month rows where model prediction matches true dominant category.  
> High accuracy for Cambridge / SLC / Birmingham / DC partly reflects class imbalance (one dominant crime type).

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

**File:** `outputs/maps/crime_map_v8.html` — ~46 MB standalone HTML, no backend required.  
**Open:** double-click the file in any modern browser (Chrome / Firefox / Edge). All 17-city data is embedded as static JSON.

---

### Getting Started

When you open the map you will see three sequential screens:

1. **Intro animation** — a police car chases a thief across the screen while data loads.
2. **Role selection** — choose your view before entering the map:
   - **Police View 警政署視角** — reveals the Patrol tab, high-risk grid ranking, threshold presets, and decision-support text.
   - **Public View 一般民眾視角** — shows a citizen-friendly safety summary and a list of lower-risk areas to visit.
3. **Main map** — the full interactive interface described below.

You can switch roles at any time using the **「選擇視角 / Choose View」** chip in the top-right corner of the sidebar.

---

### Sidebar Layout

The sidebar has up to **six tabs** (some are role-specific):

| Tab | Available to | Content |
|-----|-------------|---------|
| **總覽 / Overview** | Everyone | City selector, time slider, auto-play, season/month filter |
| **篩選 / Filters** | Everyone | Season buttons (Spring / Summer / Fall / Winter / All Year) and month buttons (Jan – Dec) |
| **分析 / Analysis** | Everyone | City statistics, confidence breakdown, legend, layer toggles, alert threshold, route risk query |
| **巡邏 / Patrol** | Police View only | Patrol priority list, threshold presets, max-risk summary |
| **民眾 / Public** | Public View only | Area safety summary, lower-risk grid list, "find safest area" button |
| **工具 / Tools** | Everyone | Top-10 high-risk grids, time-slot distribution chart, grid detail panel |

---

### City & Time Selection

- **City tabs** (top of sidebar): 17 cities arranged in a 2-column scrollable grid. Active city is highlighted in blue.
- **Time slider**: drag or click one of four positions — 深夜/Night (00–06), 早晨/Morning (06–12), 下午/Afternoon (12–18), 夜晚/Evening (18–24).
- **Auto Play**: click ▶ to cycle through all four time slots every 1.8 s. Click ⏹ to stop.
- **Season filter**: narrows grids to months in that season. Stacks with the time slider.
- **Month filter**: drill down to a single calendar month (1–12). Selecting a month clears the season filter, and vice versa.

> **Karachi exception**: the synthetic dataset has `hour=0` for all records. The time slider is disabled for Karachi and all grids are shown regardless of selected slot.

---

### Reading the Map

#### Grid colour
| Colour | Crime category |
|--------|---------------|
| 🔴 Red | Violent (assault, robbery, murder, rape) |
| 🔵 Blue | Property (theft, burglary, fraud, arson) |
| 🟢 Green | Other (drug offences, public disorder) |

#### Grid opacity
Opacity encodes **model confidence**, calibrated per city so every city has a meaningful range:

| Opacity | Tier | Condition |
|---------|------|-----------|
| Solid (≥ p80) | High confidence | Model is highly certain of its prediction |
| Medium (p50–p80) | Mid confidence | Moderate certainty |
| Faint (< p50) | Uncertain | Low certainty; treat with caution |

Percentile thresholds (p80 / p50) are computed **per city** from calibrated probabilities, so cities with compressed confidence ranges (e.g., London max ≈ 0.50) still display useful opacity variation.

#### Decision Support panel (bottom-left overlay)
Shows a plain-language summary of the current view: how many grids are visible, how many exceed the alert threshold, and the highest-risk prediction. Updates whenever city, time, season, or month changes.

#### Model Accuracy badge (top-right overlay)
Displays the map accuracy (fraction of grids where prediction = true dominant category) and the current city + time subtitle. This is a training-set metric, not a live prediction.

---

### Interacting with the Map

#### Clicking grids
Click any coloured grid cell to:
- Fly the map to that grid (zoom 14) with a white pulse ring
- Open the **Grid Detail** panel in the Tools tab showing:
  - Coordinates, true category, predicted category (✓ correct / ✗ wrong)
  - Confidence %, event count, risk score (0–100)
  - Dominance gap and entropy
  - Probability bar chart for all three categories
  - Patrol suggestion text (Police View) or safety note (Public View)

#### Clicking empty map area
Click anywhere on the basemap (not a grid) to zoom in one level with a blue pulse ring, centering on the clicked point.

---

### Alert Threshold

The **風險分數 / Risk Score** slider (Analysis tab) sets a threshold from 0 to 100:

- Grids **above** the threshold display an ⚠ alert marker on the map.
- The **⚠ N 個高風險格子 / N high-risk grids** badge appears top-left when any grids exceed the threshold.
- The count below the slider shows exactly how many grids are above threshold.

Risk scores are **city-normalised**: the highest-risk grid in each city scores 100, so scores are comparable within a city but not across cities.

**Police View threshold presets** (Patrol tab):

| Preset | Score | Use case |
|--------|-------|----------|
| 日常 / Routine | 50 | Everyday patrol planning |
| 加強 / Enhanced | 65 | Heightened attention periods |
| 緊急 / Urgent | 80 | Limited resources or special events |

---

### Route Risk Query (Analysis tab)

Enter the latitude/longitude of a start and end point. The system finds all grid cells within 0.01° of the straight-line route and reports:

| Output | Meaning |
|--------|---------|
| 沿途格子數 / Route Grids | Number of grids along the path |
| 平均風險 / Avg Risk | Mean risk score of route grids |
| 最高風險 / Max Risk | Peak risk score on the route |
| 暴力犯罪佔比 / Violent Share | Fraction of route grids predicted as violent |

---

### Top-10 High-Risk Grids (Tools tab)

Lists the 10 highest-risk grids for the current city and time slot, sorted by risk score. Each row shows:
- Rank, predicted category, risk score
- Coordinates and confidence

Click any row to fly the map to that grid and open its detail panel.

---

### Patrol Priority List (Patrol tab — Police View only)

Automatically populated from grids that exceed the current alert threshold (up to top 40, sorted by risk descending). Each entry shows:
- Risk score, coordinates, confidence, event count
- Suggested action level: 一般 / 注意巡視 / 加強巡邏 / 緊急
- Contextual patrol note based on crime category and current time/season

Click any entry to fly the map to that grid.

#### Patrol Check-In (打卡)

Each patrol list item has a **「打卡 / Check In」** button:

| Action | Result |
|--------|--------|
| Click **打卡** | Marks the grid as visited; button turns green (✓ 已打卡 / Done); list card dims with a green tint |
| Click again | Removes the check-in |
| **Progress bar** | Shows "已打卡 X / Y · N%" at the top of the list; updates live |
| **清除 / Clear** | Resets all check-ins for the current session |
| **Map overlay** | Checked-in grids display a green dashed border and a ✓ marker on the map so already-visited areas are visually distinct |

Check-in state is in-memory only and resets when the page is refreshed.

---

### Public Safety Panel (Public tab — Public View only)

- **Area summary**: plain-language description of current city risk level (average risk, number of high-risk areas)
- **Lower-risk areas list**: up to 8 grids with risk below threshold, sorted safest-first
- **「查看目前較低風險區域 / View lower-risk area」** button: flies map to the safest grid

---

### UI Controls

| Control | Location | Action |
|---------|----------|--------|
| **中 / EN** | Sidebar top-right | Toggle Chinese ↔ English for all labels |
| **☀ 亮色 / 🌙 暗色** | Sidebar top-right | Switch light ↔ dark basemap theme |
| **選擇視角 / Choose View** | Sidebar top-right | Re-open role selection |
| Layer toggles (3×) | Analysis tab | Show/hide: Prediction grids / Violence heatmap / Alert markers independently |
| Time-slot distribution bars | Tools tab | Click any bar to switch the map to that time slot |

---

### Grid Color Coding (Summary)

- **Red** = violent, **Blue** = property, **Green** = other
- **Opacity** = confidence tier, defined **per city** using the 80th / 50th percentile of calibrated confidence (city-relative, so even London's compressed [0.34–0.50] range gets meaningful high/medium/uncertain tiers)

---

### Known Limitations

| City | Issue |
|------|-------|
| Karachi | Synthetic dataset has no time-of-day info (all `hour=0`). Time slot slider is disabled; all grids shown at once |
| DC | 97% property dominance — model predicts property for nearly all grids; risk scores derived from log-normalised event count (proba_violent = 0 for all grids) |
| London | Calibrated confidence max ≈ 0.498 — the most balanced crime distribution makes all predictions genuinely uncertain |
| Cambridge / SLC / Birmingham | Near-perfect map accuracy reflects class dominance, not true model power |
| Dallas | Class collapse — model predicts "other" for 100% of grids (70.5% map acc reflects majority-class dominance) |
| San Francisco | Strong property bias — model predicts violent for 0 grids |

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
│   ├── download_cambridge.py
│   ├── download_seattle.py               # Seattle Open Data (Socrata, 2008–present)
│   ├── download_sf.py                    # SF Open Data (Socrata, 2003–2024, two datasets)
│   ├── download_dallas.py                # Dallas Open Data (Socrata, WGS84 geocoded)
│   └── download_lansing_dayton_littlerock.py  # ArcGIS (--city flag)
├── notebook/
│   ├── clean_*.py                        # City-specific cleaning scripts
│   ├── crime_classification_full_NYC.py  # Main pipeline template
│   ├── crime_classification_full_*.py    # One script per city (17 cities)
│   ├── dann_crime.py                     # DANN v1 domain adaptation
│   └── dann_v2.py                        # DANN v2 (4 improvements)
├── build_all_cities.py                   # Merge per-city CSVs → all_cities.csv
├── patch_method_b_all.py                 # Batch-patched all scripts to Method B
├── patch_add_cities_v8.py                # Add new city tabs + data to crime_map_v8.html
├── update_map_v8.py                      # Regenerate interactive map from grid_risk CSVs
├── upgrade_v8.py                         # Inject real CITY_DATA into template.html
├── data/
│   ├── raw/                              # Downloaded CSVs (not committed to git)
│   └── processed/
│       ├── all_cities.csv               # All cities merged (~2 GB)
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
        └── crime_map_v8.html           # Interactive map — 17 cities, ~46 MB
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
python src/01_download.py              # NYC, Chicago, LA (automatic via Socrata API)
python src/download_kansas_city.py
python src/download_peoria.py
python src/download_detroit.py
python src/download_birmingham.py
python src/download_slc.py
python src/download_cambridge.py
python src/download_seattle.py
python src/download_sf.py
python src/download_dallas.py
python src/download_lansing_dayton_littlerock.py --city lansing
python src/download_lansing_dayton_littlerock.py --city dayton
python src/download_lansing_dayton_littlerock.py --city littlerock

# Step 2: Clean each city
python notebook/clean_dc.py
python notebook/clean_philadelphia.py
python notebook/clean_westyorkshire.py
python notebook/clean_kansas_city.py
python notebook/clean_peoria.py
python notebook/clean_dallas.py
# ... (see notebook/clean_*.py for each city)

# Step 3: Merge all cities
python build_all_cities.py             # → data/processed/all_cities.csv (~2 GB)

# Step 4: Train models per city
python notebook/crime_classification_full_NYC.py
python notebook/crime_classification_full_KansasCity.py
python notebook/crime_classification_full_Seattle.py
# ... (repeat for all cities)

# Step 5: Regenerate interactive map
python update_map_v8.py                # → outputs/maps/crime_map_v8.html
# Or to add new cities to existing map without full rebuild:
python patch_add_cities_v8.py
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
8. **Class collapse is a real risk** — cities with extreme category skew (Dallas: 70% `other`) cause the model to collapse to a single prediction class; calibration and balanced weighting reduce but do not eliminate this
9. **Confidence calibration compresses probabilities** — isotonic regression pushes calibrated probabilities toward the center; city-relative percentile tiers (p80/p50) are needed for meaningful opacity display across all cities

---

## Data Sources & Acknowledgements

This project would not be possible without the following open data initiatives. We are grateful to every agency and community that makes crime data publicly available.

| City | Dataset | Provider | License |
|------|---------|----------|---------|
| **New York City** | [NYPD Complaint Data Historic](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i) | NYC OpenData / NYPD | City Open Data Terms |
| **Chicago** | [Crimes – 2001 to Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) | Chicago Data Portal | City Open Data Terms |
| **Los Angeles** | [Crime Data from 2020 to Present](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8) | LA Open Data / LAPD | City Open Data Terms |
| **London** | [Police recorded crime](https://data.police.uk/data/) | data.police.uk / Metropolitan Police | [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) |
| **West Yorkshire** | [Police recorded crime](https://data.police.uk/data/) | data.police.uk / West Yorkshire Police | [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) |
| **Cambridge (UK)** | [Police recorded crime](https://data.police.uk/data/) | data.police.uk / Cambridgeshire Constabulary | [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) |
| **Philadelphia** | [Crime Incidents](https://www.opendataphilly.org/datasets/crime-incidents/) | OpenDataPhilly / Philadelphia Police Dept | City Open Data Terms |
| **Washington DC** | [Crime Incidents](https://opendata.dc.gov/datasets/crime-incidents-in-2023) | Open Data DC / Metropolitan Police Dept | City Open Data Terms |
| **Detroit** | [RMS Crime Incidents](https://data.detroitmi.gov/) | Detroit Open Data / Detroit Police Dept | City Open Data Terms |
| **Kansas City** | [Crime Data](https://data.kcmo.org/) | KCMO Open Data / KCPD | City Open Data Terms |
| **Dallas** | [Police Incidents](https://www.dallasopendata.com/) | Dallas Open Data / Dallas Police Dept | City Open Data Terms |
| **San Francisco** | [Police Department Incident Reports](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783) | SF Open Data / SFPD | City Open Data Terms |
| **Seattle** | [SPD Crime Data](https://data.seattle.gov/Public-Safety/SPD-Crime-Data-2008-Present/tazs-3rd5) | Seattle Open Data / Seattle Police Dept | City Open Data Terms |
| **Salt Lake City** | [Crime Data](https://opendata.utah.gov/) | Utah Open Data / SLCPD | City Open Data Terms |
| **Peoria (IL)** | [Crimes Public](https://police-transparency-1-peoria-il.hub.arcgis.com/datasets/crimes-public) | Peoria Police Dept (ArcGIS Hub) | City Open Data Terms |
| **Birmingham (AL)** | [Crime Data](https://data.birminghamal.gov/) | City of Birmingham, AL (CKAN Open Data Portal) | City Open Data Terms |
| **Karachi** | [Karachi Crime Dataset](https://www.kaggle.com/datasets) (synthetic) | Kaggle contributor dataset | Kaggle Dataset License (per author) |

> **Licensing note:** The summaries above are informal convenience labels. Please refer to each portal's official Terms of Use for full legal details. All datasets in this project are used solely for non-commercial academic research.
>
> City open data portals typically publish incident-level records that have been anonymised (e.g., block-level addresses or street-snapped coordinates) to protect individual privacy. This project further aggregates all records into 0.01° × 0.01° grid cells (~1 km²), adding an additional layer of spatial anonymisation.
>
> If you are a data provider and have concerns about this usage, please open a GitHub issue.

---

## License

This project's code is released under the [MIT License](LICENSE).  
Raw and processed data files are **not** redistributed in this repository — original download links and scripts are provided above and in [`src/`](src/).
