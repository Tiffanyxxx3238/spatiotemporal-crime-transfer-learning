#!/usr/bin/env python
# coding: utf-8
# Cross-City Crime Pattern Analysis — San Francisco
# Template: Detroit-style (simple column format, relative paths)

import os, warnings
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import precision_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
warnings.filterwarnings('ignore')

_BASE     = os.path.dirname(os.path.abspath(__file__))
PROC_DIR  = os.path.join(_BASE, '..', 'data', 'processed')
MODEL_DIR = os.path.join(_BASE, '..', 'outputs', 'models')
EDA_DIR   = os.path.join(_BASE, '..', 'outputs', 'eda')
for d in [MODEL_DIR, EDA_DIR]:
    os.makedirs(d, exist_ok=True)

CATEGORIES = ['violent', 'property', 'other']
CITY       = 'SF'
GRID_SIZE  = 0.01
print('Setup complete')

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(PROC_DIR, 'san_francisco_clean.csv'), low_memory=False)
df['crime_category'] = df['crime_category'].replace({'drug': 'other', 'public_order': 'other'})
df = df.dropna(subset=['latitude', 'longitude', 'crime_category'])
df = df[df['crime_category'].isin(CATEGORIES)]
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime'])
print(f'Total records: {len(df):,}')
print(f'Date range: {df["datetime"].min()} to {df["datetime"].max()}')

# ── Grid builder ──────────────────────────────────────────────────────────────
def build_grid(events, grid_size=0.01, min_count=3):
    d = events.copy()
    d['lat_bin']   = (d['latitude']  / grid_size).round() * grid_size
    d['lon_bin']   = (d['longitude'] / grid_size).round() * grid_size
    d['time_slot'] = pd.cut(d['hour'], bins=[-1,5,11,17,23], labels=[0,1,2,3]).astype(int)
    d['crime_category'] = d['crime_category'].replace({'drug':'other','public_order':'other'})
    for cat in CATEGORIES:
        d[f'cnt_{cat}'] = (d['crime_category'] == cat).astype(int)
    agg = {'total_count': ('crime_category','size'),
           'hour_mean':   ('hour','mean'),
           'month_mean':  ('month','mean'),
           'weekday_mean':('weekday','mean')}
    for cat in CATEGORIES:
        agg[f'cnt_{cat}'] = (f'cnt_{cat}','sum')
    grid = d.groupby(['lat_bin','lon_bin','time_slot','month']).agg(**agg).reset_index()
    cnt_cols = [f'cnt_{cat}' for cat in CATEGORIES]
    for cat in CATEGORIES:
        grid[f'hist_{cat}'] = grid[f'cnt_{cat}'] / grid['total_count']
    grid['dominant_category'] = grid[cnt_cols].idxmax(axis=1).str.replace('cnt_','',regex=False)
    counts_arr = grid[cnt_cols].values
    grid['top1_ratio']    = counts_arr.max(axis=1) / grid['total_count']
    sorted_c              = np.sort(counts_arr,axis=1)[:,::-1]
    grid['dominance_gap'] = (sorted_c[:,0]-sorted_c[:,1]) / grid['total_count']
    hist_arr              = np.clip(counts_arr/grid['total_count'].values.reshape(-1,1),1e-9,1)
    grid['entropy']       = scipy_entropy(hist_arr.T)
    grid['log_count']     = np.log1p(grid['total_count'])
    return grid[grid['total_count'] >= min_count].copy()

# ── Feature engineering ───────────────────────────────────────────────────────
def make_features(grid, ref_grid=None):
    from scipy.spatial import KDTree
    d = grid.copy().reset_index(drop=True)
    if ref_grid is None:
        ref_grid = d
    ref = ref_grid.groupby(['lat_bin','lon_bin','time_slot','month'])[
        ['hist_violent','hist_property','hist_other']].mean().reset_index()
    ref = ref.rename(columns={'hist_violent':'lag_violent','hist_property':'lag_property','hist_other':'lag_other'})
    d = d.merge(ref, on=['lat_bin','lon_bin','time_slot','month'], how='left').reset_index(drop=True)
    d['lag_violent']  = d['lag_violent'].fillna(d['lag_violent'].median())
    d['lag_property'] = d['lag_property'].fillna(d['lag_property'].median())
    d['lag_other']    = d['lag_other'].fillna(d['lag_other'].median())
    for pct_col, src_col in [('violent_pct','hist_violent'),('density_pct','log_count'),
                              ('entropy_pct','entropy'),('dom_gap_pct','dominance_gap')]:
        d[pct_col] = d[src_col].rank(pct=True)
    d['ts_sin']  = np.sin(2*np.pi*d['time_slot']/4)
    d['ts_cos']  = np.cos(2*np.pi*d['time_slot']/4)
    d['month_sin']    = np.sin(2*np.pi*d['month']/12)
    d['month_cos']    = np.cos(2*np.pi*d['month']/12)
    d['weekday_sin']  = np.sin(2*np.pi*d['weekday_mean']/7)
    d['weekday_cos']  = np.cos(2*np.pi*d['weekday_mean']/7)
    d['is_weekend']   = (d['weekday_mean'] >= 5).astype(int)
    d['lat_norm'] = (d['lat_bin']-d['lat_bin'].mean())/d['lat_bin'].std().clip(1e-6)
    d['lon_norm'] = (d['lon_bin']-d['lon_bin'].mean())/d['lon_bin'].std().clip(1e-6)
    return d

FEATURES = [
    'hist_violent','hist_property','hist_other',
    'lag_violent','lag_property','lag_other',
    'violent_pct','density_pct','entropy_pct','dom_gap_pct',
    'time_slot','ts_sin','ts_cos',
    'month','month_sin','month_cos',
    'weekday_sin','weekday_cos','is_weekend',
    'lat_bin','lon_bin','lat_norm','lon_norm',
    'top1_ratio','dominance_gap','entropy',
]

# ── Temporal split ────────────────────────────────────────────────────────────
df_sorted = df.sort_values('datetime')
n = len(df_sorted)
# Use 60/20/20 if < 4 years of data, else 74/13/13
span_years = (df_sorted['datetime'].max()-df_sorted['datetime'].min()).days / 365
if span_years < 4:
    t1, t2 = int(n*0.60), int(n*0.80)
else:
    t1, t2 = int(n*0.74), int(n*0.87)

df_train = df_sorted.iloc[:t1]
df_val   = df_sorted.iloc[t1:t2]
df_test  = df_sorted.iloc[t2:]
print(f'Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}')

grid_train = build_grid(df_train)
grid_val   = build_grid(df_val)
grid_test  = build_grid(df_test)
print(f'Grids — Train: {len(grid_train):,} | Val: {len(grid_val):,} | Test: {len(grid_test):,}')

# Compute hist_* from training grids only, then join to val/test
hist_ref = grid_train.groupby(['lat_bin','lon_bin','time_slot','month'])[
    ['hist_violent','hist_property','hist_other']].mean().reset_index()

def attach_hist(grid, ref):
    g = grid.copy().reset_index(drop=True)
    g = g.drop(columns=[c for c in ['hist_violent','hist_property','hist_other'] if c in g.columns], errors='ignore')
    g = g.merge(ref, on=['lat_bin','lon_bin','time_slot','month'], how='left').reset_index(drop=True)
    for c in ['hist_violent','hist_property','hist_other']:
        g[c] = g[c].fillna(1/3)
    return g

grid_val  = attach_hist(grid_val,  hist_ref)
grid_test = attach_hist(grid_test, hist_ref)

grid_train_ft = make_features(grid_train)
grid_val_ft   = make_features(grid_val,  ref_grid=grid_train)
grid_test_ft  = make_features(grid_test, ref_grid=grid_train)

le = LabelEncoder().fit(CATEGORIES)
def get_Xy(g):
    mask = g['dominant_category'].isin(CATEGORIES)
    g2 = g[mask].reset_index(drop=True)
    X  = g2[FEATURES].fillna(0)
    y  = le.transform(g2['dominant_category'])
    return X, y, g2

X_tr, y_tr, _ = get_Xy(grid_train_ft)
X_va, y_va, _ = get_Xy(grid_val_ft)
X_te, y_te, g_test = get_Xy(grid_test_ft)
print(f'Feature matrix: train={X_tr.shape}, val={X_va.shape}, test={X_te.shape}')

# ── Train CatBoost ────────────────────────────────────────────────────────────
from catboost import CatBoostClassifier
cb = CatBoostClassifier(
    iterations=600, learning_rate=0.05, depth=6,
    loss_function='MultiClass', eval_metric='Accuracy',
    class_weights={i: max(len(y_tr)/(3*np.bincount(y_tr)[i]),1) for i in range(3)},
    random_seed=42, verbose=100,
)
cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
cb.save_model(os.path.join(MODEL_DIR, f'model_sf_catboost.cbm'))

# ── Train LightGBM ────────────────────────────────────────────────────────────
import lightgbm as lgb
counts = np.bincount(y_tr)
w_tr = np.array([len(y_tr)/(3*counts[y]) for y in y_tr])
lgb_train = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
lgb_val   = lgb.Dataset(X_va, label=y_va, reference=lgb_train)
params = dict(objective='multiclass', num_class=3, num_leaves=63,
              learning_rate=0.05, n_estimators=600, verbose=-1)
lgbm = lgb.train(params, lgb_train, valid_sets=[lgb_val],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
joblib.dump(lgbm, os.path.join(MODEL_DIR, f'model_{CITY.lower()}_lgb.pkl'))

# ── Calibrate & evaluate ──────────────────────────────────────────────────────
prob_cb_val  = cb.predict_proba(X_va)
prob_lgb_val = lgbm.predict(X_va)
prob_ens_val = (prob_cb_val + prob_lgb_val) / 2

cal = [IsotonicRegression(out_of_bounds='clip').fit(prob_ens_val[:,i], (y_va==i).astype(float))
       for i in range(3)]
joblib.dump(cal, os.path.join(MODEL_DIR, f'cal_iso_{CITY.lower()}.pkl'))
joblib.dump(le,  os.path.join(MODEL_DIR, f'label_encoder_{CITY.lower()}.pkl'))

prob_cb_te  = cb.predict_proba(X_te)
prob_lgb_te = lgbm.predict(X_te)
prob_ens_te = (prob_cb_te + prob_lgb_te) / 2
prob_cal_te = np.column_stack([cal[i].predict(prob_ens_te[:,i]) for i in range(3)])
prob_cal_te = prob_cal_te / prob_cal_te.sum(axis=1, keepdims=True).clip(1e-9)
y_pred_te   = prob_cal_te.argmax(axis=1)

prec = precision_score(y_te, y_pred_te, average='macro', zero_division=0)
print(f'\n=== {CITY} Test Precision (macro): {prec:.3f} ===')
print(pd.DataFrame({'true': le.inverse_transform(y_te),
                    'pred': le.inverse_transform(y_pred_te)}).value_counts().head(12))

# ── Save grid_risk CSV ────────────────────────────────────────────────────────
violent_idx = list(le.classes_).index('violent')
g_test = g_test.reset_index(drop=True)
g_test['pred']       = le.inverse_transform(y_pred_te)
g_test['conf']       = prob_cal_te.max(axis=1).round(4)
g_test['ok']         = g_test['pred'] == g_test['dominant_category']
g_test['p_violent']  = prob_cal_te[:,violent_idx].round(4)
g_test['p_property'] = prob_cal_te[:,list(le.classes_).index('property')].round(4)
g_test['p_other']    = prob_cal_te[:,list(le.classes_).index('other')].round(4)
g_test['risk']       = (prob_cal_te[:,violent_idx]*100).round(1)
g_test['city']       = CITY

risk_df = g_test[['lat_bin','lon_bin','time_slot','month','total_count',
                   'dominant_category','pred','conf','ok',
                   'p_violent','p_property','p_other','risk','city']].copy()
out_path = os.path.join(MODEL_DIR, f'grid_risk_{CITY.lower()}.csv')
risk_df.to_csv(out_path, index=False)
print(f'Saved {len(risk_df):,} grid rows → {out_path}')

map_acc = g_test['ok'].mean()
print(f'Map accuracy: {map_acc:.1%}')
print('Done.')
