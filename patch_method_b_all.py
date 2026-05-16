"""
Patch all city classification scripts to Method B:
- Add 'month' to grid groupby  →  prediction unit = grid × time_slot × month
- Add 'month' to grid_risk output CSV
Run this once, then re-run each city script.
"""
import re, os

NOTEBOOK = r'C:\Users\user\GitHub\model-predict-crime\notebook'

# ─────────────────────────────────────────────────────────────────────────────
# Pattern A  —  old/medium scripts (NYC, Chicago, LA, London, Philly, DC, WY, Karachi)
# build_grid_from_events uses d.groupby(['lat_bin','lon_bin','time_slot'])
# risk_df selected from grid_test with explicit column list
# ─────────────────────────────────────────────────────────────────────────────
OLD_PATTERN_FILES = [
    'crime_classification_full_NYC.py',
    'crime_classification_full_Chicage_final_step.py',
    'crime_classification_full_LA_copy.py',
    'crime_classification_full_London.py',
    'crime_classification_full_Philadelphia.py',
    'crime_classification_full_DC.py',
    'crime_classification_full_WestYorkshire.py',
    'crime_classification_full_Karachi.py',
]

# Pattern B  —  Detroit-style scripts
DETROIT_PATTERN_FILES = [
    'crime_classification_full_Detroit.py',
    'crime_classification_full_Cambridge.py',
    'crime_classification_full_SaltLakeCity.py',
    'crime_classification_full_Birmingham.py',
]

# Pattern C  —  Peoria-style (newest, same as KC already done)
PEORIA_PATTERN_FILES = [
    'crime_classification_full_Peoria.py',
]

def patch_old_pattern(path):
    with open(path, encoding='utf-8') as f:
        src = f.read()

    changed = False

    # 1. Add 'month' to build_grid_from_events groupby
    old_gb = "d.groupby(['lat_bin','lon_bin','time_slot']).agg(**agg).reset_index()"
    new_gb = "d.groupby(['lat_bin','lon_bin','time_slot','month']).agg(**agg).reset_index()"
    if old_gb in src:
        src = src.replace(old_gb, new_gb, 1)
        changed = True

    # 2. Add 'month' to risk_df column selection (NYC/London/etc style)
    old_risk = ("risk_df = grid_test[['lat_bin','lon_bin','time_slot','total_count',\n"
                "                      'dominant_category','dominance_gap','entropy']].copy()")
    new_risk = ("risk_df = grid_test[['lat_bin','lon_bin','time_slot','month','total_count',\n"
                "                      'dominant_category','dominance_gap','entropy']].copy()")
    if old_risk in src:
        src = src.replace(old_risk, new_risk, 1)
        changed = True

    # Philadelphia / DC / WY / Karachi variants may differ slightly — try regex
    if "'time_slot','total_count'" in src and "'month'" not in src[src.find("risk_df = grid_test"):src.find("risk_df = grid_test")+300]:
        src = src.replace(
            "risk_df = grid_test[['lat_bin','lon_bin','time_slot','total_count',",
            "risk_df = grid_test[['lat_bin','lon_bin','time_slot','month','total_count',",
            1
        )
        changed = True

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(src)
        print(f'  ✓ patched (old-pattern): {os.path.basename(path)}')
    else:
        print(f'  ~ skipped (already patched or pattern not found): {os.path.basename(path)}')


def patch_detroit_pattern(path):
    with open(path, encoding='utf-8') as f:
        src = f.read()

    changed = False

    # 1. groupby
    old_gb = "d.groupby(['lat_bin','lon_bin','time_slot']).agg(**agg).reset_index()"
    new_gb = "d.groupby(['lat_bin','lon_bin','time_slot','month']).agg(**agg).reset_index()"
    if old_gb in src:
        src = src.replace(old_gb, new_gb, 1)
        changed = True

    # 2. risk_df column selection  (Detroit style)
    old_risk = ("risk_df = grid_test[['lat_bin','lon_bin','time_slot','total_count',\n"
                "                      'dominant_category']].copy().reset_index(drop=True)")
    new_risk = ("risk_df = grid_test[['lat_bin','lon_bin','time_slot','month','total_count',\n"
                "                      'dominant_category']].copy().reset_index(drop=True)")
    if old_risk in src:
        src = src.replace(old_risk, new_risk, 1)
        changed = True

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(src)
        print(f'  ✓ patched (detroit-pattern): {os.path.basename(path)}')
    else:
        print(f'  ~ skipped: {os.path.basename(path)}')


def patch_peoria_pattern(path):
    """Same changes as Kansas City — agg_grids + FEATURES + all_g2 + output CSV."""
    with open(path, encoding='utf-8') as f:
        src = f.read()

    if "'month'" in src and "groupby(['lat_bin', 'lon_bin', 'time_slot', 'month'" in src:
        print(f'  ~ already Method B: {os.path.basename(path)}')
        return

    changed = False

    # 1. agg_grids groupby
    old_agg = ("g = part.groupby(['lat_bin', 'lon_bin', 'time_slot', 'crime_type']).size().reset_index(name='cnt')\n"
               "    g2 = g.groupby(['lat_bin', 'lon_bin', 'time_slot']).apply(")
    new_agg = ("g = part.groupby(['lat_bin', 'lon_bin', 'time_slot', 'month', 'crime_type']).size().reset_index(name='cnt')\n"
               "    g2 = g.groupby(['lat_bin', 'lon_bin', 'time_slot', 'month']).apply(")
    if old_agg in src:
        src = src.replace(old_agg, new_agg, 1)
        changed = True

    # 2. month_sin/cos in add_pct or add_features
    old_ts = ("    g['ts_sin']   = np.sin(2*np.pi*g['time_slot']/4)\n"
              "    g['ts_cos']   = np.cos(2*np.pi*g['time_slot']/4)\n"
              "    return g")
    new_ts = ("    g['ts_sin']    = np.sin(2*np.pi*g['time_slot']/4)\n"
              "    g['ts_cos']    = np.cos(2*np.pi*g['time_slot']/4)\n"
              "    g['month_sin'] = np.sin(2*np.pi*g['month']/12)\n"
              "    g['month_cos'] = np.cos(2*np.pi*g['month']/12)\n"
              "    return g")
    if old_ts in src:
        src = src.replace(old_ts, new_ts, 1)
        changed = True

    # 3. FEATURES list
    old_feat = ("             'lat_norm','lon_norm','time_slot','ts_sin','ts_cos',\n"
                "             'weekend_ratio','month_sin_avg','month_cos_avg'])")
    new_feat = ("             'lat_norm','lon_norm','time_slot','ts_sin','ts_cos',\n"
                "             'month','month_sin','month_cos'])")
    if old_feat in src:
        src = src.replace(old_feat, new_feat, 1)
        changed = True
    else:
        old_feat2 = ("             'lat_norm','lon_norm','time_slot','ts_sin','ts_cos'])")
        new_feat2  = ("             'lat_norm','lon_norm','time_slot','ts_sin','ts_cos',\n"
                      "             'month','month_sin','month_cos'])")
        if old_feat2 in src:
            src = src.replace(old_feat2, new_feat2, 1)
            changed = True

    # 4. all_g2 groupby in grid_risk section
    old_allg = "all_g2 = all_g.groupby(['lat_bin','lon_bin','time_slot']).agg("
    new_allg = "all_g2 = all_g.groupby(['lat_bin','lon_bin','time_slot','month']).agg("
    if old_allg in src:
        src = src.replace(old_allg, new_allg, 1)
        changed = True

    # 5. Output CSV: add month column
    old_out = ("    'time_slot':         all_g2['time_slot'].astype(int),\n"
               "    'total_count':       all_g2['total_count'].astype(int),")
    new_out = ("    'time_slot':         all_g2['time_slot'].astype(int),\n"
               "    'month':             all_g2['month'].astype(int),\n"
               "    'total_count':       all_g2['total_count'].astype(int),")
    if old_out in src and "'month'" not in src[src.find("out = pd.DataFrame"):src.find("out = pd.DataFrame")+500]:
        src = src.replace(old_out, new_out, 1)
        changed = True

    # 6. Remove temp_hist if still present (Method A leftovers)
    if 'add_temporal_hist' in src:
        src = re.sub(
            r'# ── Temporal hist features.*?train_g = add_temporal_hist\(train_g\)\nval_g.*?test_g.*?\n',
            '', src, flags=re.DOTALL
        )
        src = src.replace('all_g2 = add_temporal_hist(all_g2)\n', '')
        changed = True

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(src)
        print(f'  ✓ patched (peoria-pattern): {os.path.basename(path)}')
    else:
        print(f'  ~ skipped: {os.path.basename(path)}')


print('=== Patching old-pattern scripts ===')
for fname in OLD_PATTERN_FILES:
    patch_old_pattern(os.path.join(NOTEBOOK, fname))

print('\n=== Patching detroit-pattern scripts ===')
for fname in DETROIT_PATTERN_FILES:
    patch_detroit_pattern(os.path.join(NOTEBOOK, fname))

print('\n=== Patching peoria-pattern scripts ===')
for fname in PEORIA_PATTERN_FILES:
    patch_peoria_pattern(os.path.join(NOTEBOOK, fname))

print('\nDone. Now re-run each city script to regenerate grid_risk CSVs.')
print('Suggested order (fast→slow):')
print('  Cambridge, Birmingham, SLC, Peoria, DC, Karachi, Detroit, KC (done), Philadelphia, WY, LA, London, Chicago, NYC')
