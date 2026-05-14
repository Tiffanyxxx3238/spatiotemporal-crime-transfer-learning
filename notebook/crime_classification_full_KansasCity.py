"""
Crime Classification — Kansas City, MO
Data: 2016, 2018, 2022-2024 (~570k records, non-contiguous years)
Temporal split: 74/13/13 using sorted datetime (standard)
Grid size: 0.01° (~1km²)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_score, accuracy_score
from scipy.spatial import KDTree
import catboost as cb
import lightgbm as lgb
import joblib, warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
CITY       = 'Kansas City'
CITY_KEY   = 'kansas_city'
GRID_SIZE  = 0.01
MIN_CNT    = 3
CATEGORIES = ['violent', 'property', 'other']
PROC       = r'C:\Users\user\GitHub\model-predict-crime\data\processed'
MODEL_DIR  = r'C:\Users\user\GitHub\model-predict-crime\outputs\models'

# ── Load data ─────────────────────────────────────────────────────────────────
print(f'Loading {CITY}...')
df = pd.read_csv(f'{PROC}/kansas_city_clean.csv', low_memory=False)
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime'])
df = df[df['crime_type'].isin(CATEGORIES)].copy()
# Keep only records with plausible years
df = df[df['year'].between(2010, 2026)]
df['day_of_week'] = df['datetime'].dt.dayofweek   # 0=Mon … 6=Sun
print(f'  {len(df):,} records  years: {sorted(df["year"].unique())}')

# ── Grid binning ──────────────────────────────────────────────────────────────
df['lat_bin'] = (df['lat'] / GRID_SIZE).round() * GRID_SIZE
df['lon_bin'] = (df['lon'] / GRID_SIZE).round() * GRID_SIZE

# ── Temporal split 74/13/13 ───────────────────────────────────────────────────
df_sorted = df.sort_values('datetime')
n = len(df_sorted)
i_train = int(n * 0.74)
i_val   = int(n * 0.87)
train_df = df_sorted.iloc[:i_train].copy()
val_df   = df_sorted.iloc[i_train:i_val].copy()
test_df  = df_sorted.iloc[i_val:].copy()
print(f'  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}')
print(f'  Train period: {train_df["datetime"].min().date()} ~ {train_df["datetime"].max().date()}')
print(f'  Test  period: {test_df["datetime"].min().date()}  ~ {test_df["datetime"].max().date()}')

# ── Grid aggregation ──────────────────────────────────────────────────────────
def agg_grids(part):
    g = part.groupby(['lat_bin', 'lon_bin', 'time_slot', 'crime_type']).size().reset_index(name='cnt')
    g2 = g.groupby(['lat_bin', 'lon_bin', 'time_slot']).apply(
        lambda x: pd.Series({'total_count': x['cnt'].sum(),
                              'dominant_category': x.loc[x['cnt'].idxmax(), 'crime_type'],
                              **{f'p_{c}': x.loc[x['crime_type']==c,'cnt'].sum()/x['cnt'].sum()
                                 for c in CATEGORIES}})
    ).reset_index()
    return g2[g2['total_count'] >= MIN_CNT].copy()

train_g = agg_grids(train_df)
val_g   = agg_grids(val_df)
test_g  = agg_grids(test_df)
print(f'  Grid rows — train: {len(train_g):,}  val: {len(val_g):,}  test: {len(test_g):,}')

# ── hist_* features (train only) ──────────────────────────────────────────────
hist = train_df.groupby(['lat_bin','lon_bin','crime_type']).size().unstack(fill_value=0)
for c in CATEGORIES:
    if c not in hist.columns: hist[c] = 0
hist = hist[CATEGORIES]
hist = hist.div(hist.sum(axis=1).clip(lower=1), axis=0)
hist.columns = [f'hist_{c}' for c in CATEGORIES]
hist = hist.reset_index()

def add_hist(g):
    return g.merge(hist, on=['lat_bin','lon_bin'], how='left').fillna(1/len(CATEGORIES))

train_g = add_hist(train_g)
val_g   = add_hist(val_g)
test_g  = add_hist(test_g)

# ── Temporal hist features (train only) ───────────────────────────────────────
temp_hist = train_df.groupby(['lat_bin','lon_bin']).agg(
    weekend_ratio = ('day_of_week', lambda x: (x >= 5).mean()),
    month_sin_avg = ('month', lambda x: np.sin(2*np.pi*x/12).mean()),
    month_cos_avg = ('month', lambda x: np.cos(2*np.pi*x/12).mean()),
).reset_index()

def add_temporal_hist(g):
    return g.merge(temp_hist, on=['lat_bin','lon_bin'], how='left').fillna(
        {'weekend_ratio': 0.29, 'month_sin_avg': 0.0, 'month_cos_avg': 0.0}
    )

train_g = add_temporal_hist(train_g)
val_g   = add_temporal_hist(val_g)
test_g  = add_temporal_hist(test_g)

# ── Spatial lag features ──────────────────────────────────────────────────────
def add_spatial_lag(g, k=4):
    coords = g[['lat_bin','lon_bin']].values
    if len(coords) < 2:
        for c in CATEGORIES: g[f'lag_{c}'] = 1/len(CATEGORIES)
        return g
    tree = KDTree(coords)
    lags = {f'lag_{c}': [] for c in CATEGORIES}
    for i, pt in enumerate(coords):
        k_eff = min(k+1, len(coords))
        _, idxs = tree.query(pt, k=k_eff)
        neighbors = [j for j in idxs if j != i]
        if not neighbors: neighbors = [i]
        for c in CATEGORIES:
            lags[f'lag_{c}'].append(g[f'hist_{c}'].iloc[neighbors].mean())
    for key, vals in lags.items():
        g[key] = vals
    return g

train_g = add_spatial_lag(train_g)
val_g   = add_spatial_lag(val_g)
test_g  = add_spatial_lag(test_g)

# ── Percentile + spatial features ────────────────────────────────────────────
def add_features(g, ref_df):
    g['violent_pct'] = g['p_violent'].rank(pct=True)
    density = ref_df.groupby(['lat_bin','lon_bin'])['crime_type'].count().reset_index(name='density')
    g = g.merge(density, on=['lat_bin','lon_bin'], how='left')
    g['density_pct'] = g['density'].fillna(0).rank(pct=True)
    g['lat_norm'] = (g['lat_bin'] - g['lat_bin'].mean()) / (g['lat_bin'].std() + 1e-9)
    g['lon_norm'] = (g['lon_bin'] - g['lon_bin'].mean()) / (g['lon_bin'].std() + 1e-9)
    g['ts_sin']   = np.sin(2*np.pi*g['time_slot']/4)
    g['ts_cos']   = np.cos(2*np.pi*g['time_slot']/4)
    return g

train_g = add_features(train_g, train_df)
val_g   = add_features(val_g,   train_df)
test_g  = add_features(test_g,  train_df)

FEATURES = ([f'hist_{c}' for c in CATEGORIES] +
            [f'lag_{c}'  for c in CATEGORIES] +
            ['violent_pct','density_pct','lat_bin','lon_bin',
             'lat_norm','lon_norm','time_slot','ts_sin','ts_cos',
             'weekend_ratio','month_sin_avg','month_cos_avg'])

le = LabelEncoder()
le.fit(CATEGORIES)
y_train = le.transform(train_g['dominant_category'])
y_val   = le.transform(val_g['dominant_category'])
y_test  = le.transform(test_g['dominant_category'])

X_train = train_g[FEATURES].fillna(0)
X_val   = val_g[FEATURES].fillna(0)
X_test  = test_g[FEATURES].fillna(0)

# ── CatBoost ──────────────────────────────────────────────────────────────────
print('\nTraining CatBoost...')
weights = {i: len(y_train)/(len(CATEGORIES)*max(np.bincount(y_train)[i],1))
           for i in range(len(CATEGORIES))}

cb_model = cb.CatBoostClassifier(
    iterations=600, learning_rate=0.05, depth=6,
    loss_function='MultiClass', eval_metric='Accuracy',
    class_weights=list(weights.values()),
    verbose=0, random_seed=42
)
cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
cb_model.save_model(f'{MODEL_DIR}/model_{CITY_KEY}_catboost.cbm')

# ── LightGBM ──────────────────────────────────────────────────────────────────
print('Training LightGBM...')
lgb_model = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.05, num_leaves=63,
    class_weight='balanced', random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
joblib.dump(lgb_model, f'{MODEL_DIR}/model_{CITY_KEY}_lgb.pkl')

# ── Calibration ───────────────────────────────────────────────────────────────
def pad_proba(proba, n_classes):
    if proba.shape[1] == n_classes: return proba
    padded = np.zeros((len(proba), n_classes))
    padded[:, :proba.shape[1]] = proba
    return padded

p_cb_val  = pad_proba(cb_model.predict_proba(X_val),  len(CATEGORIES))
p_lgb_val = pad_proba(lgb_model.predict_proba(X_val), len(CATEGORIES))
p_val_ens = (p_cb_val + p_lgb_val) / 2

iso_list = []
for c in range(len(CATEGORIES)):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_val_ens[:, c], (y_val == c).astype(int))
    iso_list.append(iso)
joblib.dump(iso_list, f'{MODEL_DIR}/cal_iso_{CITY_KEY}.pkl')
joblib.dump(le,       f'{MODEL_DIR}/label_encoder_{CITY_KEY}.pkl')

# ── Test evaluation ───────────────────────────────────────────────────────────
p_cb_test  = pad_proba(cb_model.predict_proba(X_test),  len(CATEGORIES))
p_lgb_test = pad_proba(lgb_model.predict_proba(X_test), len(CATEGORIES))
p_test_ens = (p_cb_test + p_lgb_test) / 2
p_test_cal = np.stack([iso_list[c].predict(p_test_ens[:, c])
                       for c in range(len(CATEGORIES))], axis=1)
p_test_cal = p_test_cal / p_test_cal.sum(axis=1, keepdims=True).clip(1e-9)

y_pred = p_test_cal.argmax(axis=1)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
acc  = accuracy_score(y_test, y_pred)
print(f'\nTest  Precision (macro): {prec:.4f}')
print(f'Test  Accuracy:          {acc:.4f}')

# ── Grid risk CSV ─────────────────────────────────────────────────────────────
print('\nBuilding grid_risk CSV...')
all_g = pd.concat([train_g, val_g, test_g], ignore_index=True)
all_g2 = all_g.groupby(['lat_bin','lon_bin','time_slot']).agg(
    total_count=('total_count','sum'),
    dominant_category=('dominant_category', lambda x: x.value_counts().index[0]),
    p_violent=('p_violent','mean'),
    p_property=('p_property','mean'),
    p_other=('p_other','mean'),
).reset_index()
all_g2 = add_hist(all_g2)
all_g2 = add_spatial_lag(all_g2)
all_g2 = add_temporal_hist(all_g2)
all_g2 = add_features(all_g2, df)

X_all = all_g2[FEATURES].fillna(0)
p_cb  = pad_proba(cb_model.predict_proba(X_all),  len(CATEGORIES))
p_lgb = pad_proba(lgb_model.predict_proba(X_all), len(CATEGORIES))
p_ens = (p_cb + p_lgb) / 2
p_cal = np.stack([iso_list[c].predict(p_ens[:, c]) for c in range(len(CATEGORIES))], axis=1)
p_cal = p_cal / p_cal.sum(axis=1, keepdims=True).clip(1e-9)

pred_idx = p_cal.argmax(axis=1)
conf     = p_cal.max(axis=1)
pred_lbl = le.inverse_transform(pred_idx)
true_lbl = all_g2['dominant_category'].values
ok       = (pred_lbl == true_lbl)
risk     = (conf * 100 * (pred_lbl == 'violent')).round(1)

out = pd.DataFrame({
    'lat_bin':           all_g2['lat_bin'].round(4),
    'lon_bin':           all_g2['lon_bin'].round(4),
    'time_slot':         all_g2['time_slot'].astype(int),
    'total_count':       all_g2['total_count'].astype(int),
    'dominant_category': true_lbl,
    'pred':              pred_lbl,
    'conf':              conf.round(3),
    'ok':                ok,
    'p_violent':         p_cal[:, le.transform(['violent'])[0]].round(4),
    'p_property':        p_cal[:, le.transform(['property'])[0]].round(4),
    'p_other':           p_cal[:, le.transform(['other'])[0]].round(4),
    'risk':              risk,
})
out.to_csv(f'{MODEL_DIR}/grid_risk_{CITY_KEY}.csv', index=False)
map_acc = ok.mean() * 100
print(f'Grid risk saved: {len(out):,} rows  map_acc={map_acc:.1f}%')

# ── Transfer learning ─────────────────────────────────────────────────────────
print('\n--- Transfer: Zero-shot NYC → Kansas City ---')
try:
    nyc_lgb = joblib.load(f'{MODEL_DIR}/model_nyc_lgb.pkl')
    p_nyc = nyc_lgb.predict_proba(X_test)
    y_zs = p_nyc.argmax(axis=1)
    prec_zs = precision_score(y_test, y_zs, average='macro', zero_division=0)
    print(f'Zero-shot NYC→KC Precision: {prec_zs:.4f}')
except Exception as e:
    print(f'Transfer skip: {e}')

print('\nDone!')
