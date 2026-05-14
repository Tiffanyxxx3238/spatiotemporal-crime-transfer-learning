#!/usr/bin/env python
# coding: utf-8
# Cross-City Crime Pattern Analysis & Transfer Learning — Cambridge

import os, warnings
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import joblib
warnings.filterwarnings('ignore')

_BASE     = os.path.dirname(os.path.abspath(__file__))
PROC_DIR  = os.path.join(_BASE, '..', 'data', 'processed')
MODEL_DIR = os.path.join(_BASE, '..', 'outputs', 'models')
EDA_DIR   = os.path.join(_BASE, '..', 'outputs', 'eda')
MAP_DIR   = os.path.join(_BASE, '..', 'outputs', 'maps')
for d in [MODEL_DIR, EDA_DIR, MAP_DIR]:
    os.makedirs(d, exist_ok=True)

CATEGORIES  = ['violent', 'property', 'other']
CITY        = 'Cambridge'
GRID_SIZE   = 0.005   # finer grid for small city
plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})
print('Setup complete')

# ============================================================
# CELL 1: Load data
# ============================================================
_clean_path = os.path.join(PROC_DIR, 'cambridge_clean.csv')
print(f"Reading: {_clean_path}")
df = pd.read_csv(_clean_path, low_memory=False)
df['crime_category'] = df['crime_category'].replace({'drug': 'other', 'public_order': 'other'})
df = df.dropna(subset=['latitude', 'longitude', 'crime_category'])
df = df[df['crime_category'].isin(CATEGORIES)]
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime'])
print(f'Total records: {len(df):,}')
print(f'Date range: {df["datetime"].min()} to {df["datetime"].max()}')

# ============================================================
# CELL 2: Grid builder
# ============================================================
def build_grid_from_events(events, grid_size=0.005, min_count=3):
    d = events.copy()
    d['lat_bin']   = (d['latitude']  / grid_size).round() * grid_size
    d['lon_bin']   = (d['longitude'] / grid_size).round() * grid_size
    d['time_slot'] = pd.cut(d['hour'], bins=[-1,5,11,17,23], labels=[0,1,2,3]).astype(int)
    d['crime_category'] = d['crime_category'].replace({'drug': 'other', 'public_order': 'other'})
    for cat in CATEGORIES:
        d[f'cnt_{cat}'] = (d['crime_category'] == cat).astype(int)
    agg = {'total_count': ('crime_category', 'size'),
           'hour_mean': ('hour', 'mean'), 'month_mean': ('month', 'mean'),
           'weekday_mean': ('weekday', 'mean')}
    for cat in CATEGORIES:
        agg[f'cnt_{cat}'] = (f'cnt_{cat}', 'sum')
    grid = d.groupby(['lat_bin','lon_bin','time_slot']).agg(**agg).reset_index()
    cnt_cols = [f'cnt_{cat}' for cat in CATEGORIES]
    for cat in CATEGORIES:
        grid[f'hist_{cat}'] = grid[f'cnt_{cat}'] / grid['total_count']
    grid['dominant_category'] = grid[cnt_cols].idxmax(axis=1).str.replace('cnt_', '', regex=False)
    counts_arr = grid[cnt_cols].values
    grid['top1_ratio']    = counts_arr.max(axis=1) / grid['total_count']
    sorted_c              = np.sort(counts_arr, axis=1)[:,::-1]
    grid['dominance_gap'] = (sorted_c[:,0] - sorted_c[:,1]) / grid['total_count']
    hist_arr              = np.clip(counts_arr / grid['total_count'].values.reshape(-1,1), 1e-9, 1)
    grid['entropy']       = scipy_entropy(hist_arr.T)
    grid['log_count']     = np.log1p(grid['total_count'])
    return grid[grid['total_count'] >= min_count].copy()

# ============================================================
# CELL 3: Temporal split
# ============================================================
city_events = df.copy()
max_date   = city_events['datetime'].max()
min_date   = city_events['datetime'].min()
total_days = (max_date - min_date).days

print(f'\nCity: {CITY}  grid_size={GRID_SIZE}')
print(f'Date range: {min_date.date()} to {max_date.date()} ({total_days} days)')

test_start = max_date - pd.Timedelta(days=int(total_days * 0.20))
val_start  = max_date - pd.Timedelta(days=int(total_days * 0.40))

train_events = city_events[city_events['datetime'] <  val_start]
val_events   = city_events[(city_events['datetime'] >= val_start) & (city_events['datetime'] < test_start)]
test_events  = city_events[city_events['datetime'] >= test_start]
print(f'Train: {len(train_events):,}  Val: {len(val_events):,}  Test: {len(test_events):,}')

grid_train = build_grid_from_events(train_events, grid_size=GRID_SIZE)
grid_val   = build_grid_from_events(val_events,   grid_size=GRID_SIZE)
grid_test  = build_grid_from_events(test_events,  grid_size=GRID_SIZE)
print(f'Train grids: {len(grid_train):,}  Val: {len(grid_val):,}  Test: {len(grid_test):,}')

# ============================================================
# CELL 4: Features
# ============================================================
def add_spatial_lag(grid, k=8):
    from sklearn.neighbors import BallTree
    coords = grid[['lat_bin','lon_bin']].values
    tree   = BallTree(np.radians(coords), metric='haversine')
    dists, idxs = tree.query(np.radians(coords), k=min(k+1, len(grid)))
    for cat in CATEGORIES:
        vals = grid[f'hist_{cat}'].values
        grid[f'lag_{cat}'] = np.array([vals[idx[1:]].mean() for idx in idxs])
    return grid

def make_percentile_features(grid, ref_grid=None):
    for col, out in [('total_count','density_pct'), ('hist_violent','violent_pct'),
                     ('entropy','entropy_pct'), ('dominance_gap','dom_gap_pct')]:
        grid[out] = grid[col].rank(pct=True)
    return grid

FEATURE_COLS = [
    'lat_bin','lon_bin','lat_norm','lon_norm',
    'density_pct','violent_pct','entropy_pct','dom_gap_pct',
    'time_slot','is_weekend','log_count',
    'hour_sin','hour_cos','month_sin','month_cos','weekday_sin','weekday_cos',
    'top1_ratio','dominance_gap','entropy',
    'hist_violent','hist_property','hist_other',
    'lag_violent','lag_property','lag_other',
]
CAT_COLS = ['lat_bin','lon_bin','time_slot']

def make_features(grid, ref_grid=None):
    d = grid.copy()
    d = make_percentile_features(d, ref_grid)
    lat_vals = d['lat_bin']
    lon_vals = d['lon_bin']
    d['lat_norm']    = (lat_vals - lat_vals.min()) / max(lat_vals.max() - lat_vals.min(), 1e-9)
    d['lon_norm']    = (lon_vals - lon_vals.min()) / max(lon_vals.max() - lon_vals.min(), 1e-9)
    d['is_weekend']  = (d['weekday_mean'].round().astype(int) >= 5).astype(int)
    d['hour_sin']    = np.sin(2*np.pi * d['hour_mean']    / 24)
    d['hour_cos']    = np.cos(2*np.pi * d['hour_mean']    / 24)
    d['month_sin']   = np.sin(2*np.pi * d['month_mean']   / 12)
    d['month_cos']   = np.cos(2*np.pi * d['month_mean']   / 12)
    d['weekday_sin'] = np.sin(2*np.pi * d['weekday_mean'] / 7)
    d['weekday_cos'] = np.cos(2*np.pi * d['weekday_mean'] / 7)
    for c in CATEGORIES:
        if f'lag_{c}' not in d.columns:
            d[f'lag_{c}'] = d[f'hist_{c}']
    return d[FEATURE_COLS].fillna(0)

def prep_cat(Xdf):
    Xc = Xdf.copy()
    for c in CAT_COLS:
        if c in Xc.columns:
            Xc[c] = Xc[c].astype(str)
    return Xc

print('Computing spatial lag...')
grid_train = add_spatial_lag(grid_train)
print('Done')

le = LabelEncoder()
le.fit(CATEGORIES)

X_train = make_features(grid_train)
X_val   = make_features(grid_val,   ref_grid=grid_train)
X_test  = make_features(grid_test,  ref_grid=grid_train)

def filter_known_cats(grid, X, le):
    known = set(le.classes_)
    mask  = grid['dominant_category'].isin(known)
    return X[mask], grid['dominant_category'][mask].values

X_train, y_train_raw = filter_known_cats(grid_train, X_train, le)
X_val,   y_val_raw   = filter_known_cats(grid_val,   X_val,   le)
X_test,  y_test_raw  = filter_known_cats(grid_test,  X_test,  le)

y_train = le.transform(y_train_raw)
y_val   = le.transform(y_val_raw)
y_test  = le.transform(y_test_raw)

cat_idx = [list(X_train.columns).index(c) for c in CAT_COLS if c in X_train.columns]
print(f'Features: {X_train.shape[1]}  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}')
print(f'Classes: {le.classes_}')

# ============================================================
# CELL 5: CatBoost + LightGBM
# ============================================================
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb

cb = CatBoostClassifier(
    iterations=800, depth=6, learning_rate=0.05,
    loss_function='MultiClass', eval_metric='TotalF1',
    auto_class_weights='Balanced',
    cat_features=cat_idx, random_seed=42,
    early_stopping_rounds=80, verbose=100,
)
cb.fit(
    Pool(prep_cat(X_train), y_train, cat_features=cat_idx),
    eval_set=Pool(prep_cat(X_val), y_val, cat_features=cat_idx),
)
cb.save_model(f'{MODEL_DIR}/model_{CITY.lower()}_catboost.cbm')
print('CatBoost saved')

y_pred_cb = cb.predict(prep_cat(X_test)).flatten()
fig, ax = plt.subplots(figsize=(6,5))
present = sorted(np.unique(np.concatenate([y_test, y_pred_cb])))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_cb, labels=present),
    display_labels=[le.classes_[i] for i in present]
).plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'CatBoost [{CITY}]')
plt.tight_layout()
plt.savefig(f'{EDA_DIR}/cm_{CITY.lower()}_catboost.png', dpi=120)
plt.close()

pm_cb = precision_score(y_test, y_pred_cb, average='macro', zero_division=0)
f1_cb = f1_score(y_test, y_pred_cb, average='macro', zero_division=0)
print(f'\n===== CatBoost [{CITY}] =====')
print(f'Precision macro={pm_cb:.4f}  F1={f1_cb:.4f}')
print(classification_report(y_test, y_pred_cb, labels=present,
      target_names=[le.classes_[i] for i in present], zero_division=0))

lgb_params = dict(
    objective='multiclass', num_class=len(CATEGORIES),
    metric='multi_logloss', learning_rate=0.05,
    num_leaves=63, n_estimators=800, random_state=42,
    class_weight='balanced', verbose=-1,
)
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(100)],
)
joblib.dump(lgb_model, f'{MODEL_DIR}/model_{CITY.lower()}_lgb.pkl')
print('LightGBM saved')

y_pred_lgb = lgb_model.predict(X_test)
pm_lgb = precision_score(y_test, y_pred_lgb, average='macro', zero_division=0)
f1_lgb = f1_score(y_test, y_pred_lgb, average='macro', zero_division=0)
print(f'\n===== LightGBM [{CITY}] =====')
print(f'Precision macro={pm_lgb:.4f}  F1={f1_lgb:.4f}')
print(classification_report(y_test, y_pred_lgb, labels=present,
      target_names=[le.classes_[i] for i in present], zero_division=0))

fi = pd.DataFrame({'feature': X_train.columns,
                   'importance': lgb_model.feature_importances_}).sort_values('importance', ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=fi.head(15), x='importance', y='feature', ax=ax)
ax.set_title(f'Feature Importance LightGBM [{CITY}]')
plt.tight_layout()
plt.savefig(f'{EDA_DIR}/feature_importance_{CITY.lower()}.png', dpi=120)
plt.close()

# ============================================================
# CELL 6: Calibration & save
# ============================================================
proba_cb_val = cb.predict_proba(prep_cat(X_val))

cal_platt = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial')
cal_platt.fit(proba_cb_val, y_val)
joblib.dump(cal_platt, f'{MODEL_DIR}/cal_platt_{CITY.lower()}.pkl')

cal_iso_list = []
for c in range(len(CATEGORIES)):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(proba_cb_val[:, c], (y_val == c).astype(int))
    cal_iso_list.append(iso)
joblib.dump(cal_iso_list, f'{MODEL_DIR}/cal_iso_{CITY.lower()}.pkl')
joblib.dump(le, f'{MODEL_DIR}/label_encoder_{CITY.lower()}.pkl')
print('Calibration & labels saved.')

# ============================================================
# CELL 7: Grid risk scores
# ============================================================
proba_test_lgb = lgb_model.predict_proba(X_test)
risk_df = grid_test[['lat_bin','lon_bin','time_slot','total_count',
                      'dominant_category']].copy().reset_index(drop=True)
risk_df['pred']  = le.inverse_transform(y_pred_lgb)
risk_df['conf']  = proba_test_lgb.max(axis=1)
risk_df['ok']    = (risk_df['dominant_category'] == risk_df['pred'])
for i, cat in enumerate(le.classes_):
    risk_df[f'p_{cat}'] = proba_test_lgb[:, i]

raw_risk = risk_df['conf'] * np.log1p(risk_df['total_count'])
risk_df['risk'] = (raw_risk / raw_risk.max() * 100).round(1)

risk_out = f'{MODEL_DIR}/grid_risk_{CITY.lower()}.csv'
risk_df.to_csv(risk_out, index=False)
acc = risk_df['ok'].mean()
print(f'[{CITY}] Saved: {risk_out}  ({len(risk_df):,} grids)  Accuracy: {acc:.4f}')

# ============================================================
# CELL 8: Method comparison
# ============================================================
cmp_df = pd.DataFrame([
    {'Method': 'CatBoost', 'Precision Macro': round(pm_cb, 4), 'F1 Macro': round(f1_cb, 4), 'Coverage': 1.0},
    {'Method': 'LightGBM', 'Precision Macro': round(pm_lgb, 4), 'F1 Macro': round(f1_lgb, 4), 'Coverage': 1.0},
])
cmp_df.to_csv(f'{MODEL_DIR}/method_comparison_{CITY.lower()}.csv', index=False)
print(f'\n===== Method Comparison [{CITY}] =====')
print(cmp_df.to_string(index=False))

# ============================================================
# CELL 9: Transfer learning — London → Cambridge
# ============================================================
try:
    lon_cb  = CatBoostClassifier()
    lon_cb.load_model(f'{MODEL_DIR}/model_london_catboost.cbm')
    lon_lgb = joblib.load(f'{MODEL_DIR}/model_london_lgb.pkl')
    lon_le  = joblib.load(f'{MODEL_DIR}/label_encoder_london.pkl')

    lon_order = list(lon_le.classes_)
    cam_order = list(le.classes_)
    idx_map   = [lon_order.index(c) for c in cam_order if c in lon_order]

    proba_zs = lon_cb.predict_proba(prep_cat(X_test))[:, idx_map]
    y_zs     = np.argmax(proba_zs, axis=1)
    pm_zs    = precision_score(y_test, y_zs, average='macro', zero_division=0)
    f1_zs    = f1_score(y_test, y_zs, average='macro', zero_division=0)

    T = 3.0
    lon_proba_train = lon_cb.predict_proba(prep_cat(X_train))[:, idx_map]
    soft_labels = np.exp(np.log(lon_proba_train + 1e-9) / T)
    soft_labels /= soft_labels.sum(axis=1, keepdims=True)
    y_soft = np.argmax(soft_labels, axis=1)

    ts_model = CatBoostClassifier(
        iterations=500, depth=5, learning_rate=0.05,
        loss_function='MultiClass', auto_class_weights='Balanced',
        cat_features=cat_idx, random_seed=42, verbose=0,
    )
    ts_model.fit(Pool(prep_cat(X_train), y_soft, cat_features=cat_idx))
    y_ts = ts_model.predict(prep_cat(X_test)).flatten()
    pm_ts = precision_score(y_test, y_ts, average='macro', zero_division=0)
    f1_ts = f1_score(y_test, y_ts, average='macro', zero_division=0)

    # Also try NYC → Cambridge (cross-cultural)
    nyc_cb2 = CatBoostClassifier()
    nyc_cb2.load_model(f'{MODEL_DIR}/model_nyc_catboost.cbm')
    nyc_le2  = joblib.load(f'{MODEL_DIR}/label_encoder_nyc.pkl')
    nyc_idx  = [list(nyc_le2.classes_).index(c) for c in cam_order if c in nyc_le2.classes_]
    proba_nyc = nyc_cb2.predict_proba(prep_cat(X_test))[:, nyc_idx]
    y_nyc     = np.argmax(proba_nyc, axis=1)
    pm_nyc    = precision_score(y_test, y_nyc, average='macro', zero_division=0)
    f1_nyc    = f1_score(y_test, y_nyc, average='macro', zero_division=0)

    transfer_df = pd.DataFrame([
        {'Scenario': f'{CITY} baseline',          'Precision Macro': round(pm_lgb, 4), 'F1 Macro': round(f1_lgb, 4)},
        {'Scenario': f'Zero-shot London->{CITY}',  'Precision Macro': round(pm_zs, 4),  'F1 Macro': round(f1_zs, 4)},
        {'Scenario': 'Teacher-Student London→Cam', 'Precision Macro': round(pm_ts, 4),  'F1 Macro': round(f1_ts, 4)},
        {'Scenario': f'Zero-shot NYC->{CITY}',     'Precision Macro': round(pm_nyc, 4), 'F1 Macro': round(f1_nyc, 4)},
    ])
    transfer_df.to_csv(f'{MODEL_DIR}/transfer_london_{CITY.lower()}.csv', index=False)
    print(f'\n===== Transfer → {CITY} =====')
    print(transfer_df.to_string(index=False))
except Exception as e:
    print(f'[Transfer] Skipped: {e}')

print(f'\n[{CITY}] Pipeline complete.')
