"""
Update crime_map_v7.html → crime_map_v8.html
- Rebuild ALL 14 cities with month (mo) field from latest Method B grid_risk CSVs
- Add Karachi tab (new city, not in v7)
- Season selector already in v7 JS/HTML
- Handles both column naming formats:
    Simple format: pred, conf, ok, p_violent, p_property, p_other, risk
    DC-pattern:    predicted_category, confidence_calibrated, correct,
                   proba_violent, proba_property, proba_other, risk_score
"""
import json, re
import pandas as pd
import numpy as np

HTML_IN   = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\crime_map_v7.html'
HTML_OUT  = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\crime_map_v8.html'
MODEL_DIR = r'C:\Users\user\GitHub\model-predict-crime\outputs\models'

# ── City config ───────────────────────────────────────────────────────────────
CITIES = [
    # (key, display_name, center, zoom, csv_path)
    ('NYC',           'NYC',            [40.73, -73.99], 11, f'{MODEL_DIR}/grid_risk_nyc.csv'),
    ('Chicago',       'Chicago',        [41.85, -87.65], 11, f'{MODEL_DIR}/grid_risk_chicago.csv'),
    ('LA',            'LA',             [34.05,-118.25], 10, f'{MODEL_DIR}/grid_risk_la.csv'),
    ('London',        'London',         [51.51,  -0.12], 11, f'{MODEL_DIR}/grid_risk_london.csv'),
    ('Philadelphia',  'Philadelphia',   [39.95, -75.16], 12, f'{MODEL_DIR}/grid_risk_philadelphia.csv'),
    ('DC',            'DC',             [38.90, -77.04], 12, f'{MODEL_DIR}/grid_risk_dc.csv'),
    ('West Yorkshire','West Yorkshire', [53.80,  -1.55], 11, f'{MODEL_DIR}/grid_risk_west yorkshire.csv'),
    ('Detroit',       'Detroit',        [42.33, -83.05], 12, f'{MODEL_DIR}/grid_risk_detroit.csv'),
    ('Cambridge',     'Cambridge',      [52.20,   0.12], 13, f'{MODEL_DIR}/grid_risk_cambridge.csv'),
    ('Salt Lake City','Salt Lake City', [40.76,-111.89], 12, f'{MODEL_DIR}/grid_risk_salt_lake_city.csv'),
    ('Birmingham',    'Birmingham',     [33.52, -86.80], 12, f'{MODEL_DIR}/grid_risk_birmingham.csv'),
    ('Peoria',        'Peoria',         [40.69, -89.59], 12, f'{MODEL_DIR}/grid_risk_peoria.csv'),
    ('Kansas City',   'Kansas City',    [39.10, -94.58], 12, f'{MODEL_DIR}/grid_risk_kansas_city.csv'),
    ('Karachi',       'Karachi',        [24.86,  67.01], 12, f'{MODEL_DIR}/grid_risk_karachi.csv'),
]

print("Reading HTML...")
with open(HTML_IN, encoding='utf-8') as f:
    html = f.read()

# ── Parse existing CITY_DATA ──────────────────────────────────────────────────
idx   = html.index('const CITY_DATA=')
start = idx + len('const CITY_DATA=')
end_m = re.search(r'\nconst [A-Z]', html[start:])
end   = start + end_m.start()
city_data = json.loads(html[start:end].rstrip(';\n\r '))
print(f"Existing cities in v7: {list(city_data.keys())}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def entropy3(pv, pp, po):
    probs = np.clip([pv, pp, po], 1e-9, 1)
    probs /= probs.sum()
    return float(-np.sum(probs * np.log(probs)))

def conf_tier(c):
    if c >= 0.80: return 'high'
    if c >= 0.55: return 'medium'
    return 'uncertain'

def read_city_csv(csv_path):
    """Read grid_risk CSV and normalise column names to simple format."""
    df = pd.read_csv(csv_path)
    # Normalise DC-pattern → simple format
    rename = {
        'predicted_category':   'pred',
        'confidence_calibrated':'conf',
        'correct':              'ok',
        'proba_violent':        'p_violent',
        'proba_property':       'p_property',
        'proba_other':          'p_other',
        'risk_score':           'risk',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df

def build_city_entry(csv_path, center, zoom):
    df = read_city_csv(csv_path)
    has_month = 'month' in df.columns

    grids = []
    for _, r in df.iterrows():
        pv = round(float(r.get('p_violent',  0)), 3)
        pp = round(float(r.get('p_property', 0)), 3)
        po = round(float(r.get('p_other',    0)), 3)
        cf = round(float(r.get('conf',        0)), 3)
        probs_sorted = sorted([pv, pp, po], reverse=True)
        gap = round(probs_sorted[0] - probs_sorted[1], 3)
        ent = round(entropy3(pv, pp, po), 3)

        g = {
            'lat':  round(float(r['lat_bin']),  4),
            'lon':  round(float(r['lon_bin']),  4),
            'ts':   int(r['time_slot']),
            'cnt':  int(r['total_count']),
            'dom':  str(r['dominant_category']),
            'pred': str(r.get('pred', r.get('dominant_category', ''))),
            'conf': cf,
            'tier': conf_tier(cf),
            'ok':   bool(r.get('ok', r.get('correct', False))),
            'risk': round(float(r.get('risk', r.get('risk_score', 0))), 1),
            'pv': pv, 'pp': pp, 'po': po,
            'gap': gap, 'ent': ent,
        }
        if has_month:
            g['mo'] = int(r['month'])
        grids.append(g)

    ok_arr = [g['ok']   for g in grids]
    cf_arr = [g['conf'] for g in grids]
    rk_arr = [g['risk'] for g in grids]
    acc    = round(sum(ok_arr) / max(len(ok_arr), 1) * 100, 1)

    return {
        'meta':  {'center': center, 'zoom': zoom, 'has_month': has_month},
        'stats': {
            'total':    len(grids),
            'acc':      acc,
            'high':     sum(1 for c in cf_arr if c >= 0.80),
            'med':      sum(1 for c in cf_arr if 0.55 <= c < 0.80),
            'unc':      sum(1 for c in cf_arr if c < 0.55),
            'avg_risk': round(float(np.mean(rk_arr)), 1) if rk_arr else 0,
            'max_risk': round(float(np.max(rk_arr)),  1) if rk_arr else 0,
        },
        'grids': grids,
    }

# ── Rebuild all 14 cities ─────────────────────────────────────────────────────
import os
new_city_data = {}
for key, display, center, zoom, csv_path in CITIES:
    if not os.path.exists(csv_path):
        print(f"  SKIP {key}: CSV not found")
        if key in city_data:
            new_city_data[key] = city_data[key]
        continue
    print(f"  Building {key}...")
    entry = build_city_entry(csv_path, center, zoom)
    s = entry['stats']
    new_city_data[key] = entry
    print(f"    {s['total']:,} grids, acc={s['acc']}%, has_month={entry['meta']['has_month']}")

# ── Add Karachi tab if not already in HTML ────────────────────────────────────
if "switchCity('Karachi')" not in html:
    print("Adding Karachi tab button...")
    # Insert after Kansas City tab
    kc_tab_pattern = r"(<button[^>]*onclick=[\"']switchCity\('Kansas City'\)[\"'][^>]*>[^<]*</button>)"
    kc_match = re.search(kc_tab_pattern, html)
    if kc_match:
        karachi_btn = (
            "<button class=\"city-tab\" onclick=\"switchCity('Karachi')\" "
            "style=\"white-space:nowrap\">卡拉奇</button>"
        )
        html = html[:kc_match.end()] + karachi_btn + html[kc_match.end():]
        print("  Karachi tab added after Kansas City")
    else:
        print("  WARNING: Could not find Kansas City tab to insert after")

# ── Inject updated CITY_DATA ──────────────────────────────────────────────────
print("Serialising CITY_DATA...")
new_json = json.dumps(new_city_data, ensure_ascii=False, separators=(',', ':'))
html = html[:start] + new_json + html[end:]

# ── Fix season-wrap show/hide for initial city (NYC) ─────────────────────────
# Ensure NYC shows season-wrap if has_month
nyc_has_month = new_city_data.get('NYC', {}).get('meta', {}).get('has_month', False)
if nyc_has_month:
    # Show season-wrap on load
    html = html.replace(
        "id=\"season-wrap\" style=\"display:none;",
        "id=\"season-wrap\" style=\"display:block;"
    )
    print("NYC has_month=True → season-wrap shown on load")

# ── Write ─────────────────────────────────────────────────────────────────────
print(f"Writing {HTML_OUT}...")
with open(HTML_OUT, 'w', encoding='utf-8') as f:
    f.write(html)

size_mb = len(html.encode('utf-8')) / 1e6
print(f"Done! {size_mb:.1f} MB")
print(f"Total cities: {len(new_city_data)}")
for k, v in new_city_data.items():
    print(f"  {k:20s}: {v['stats']['total']:6,} grids, has_month={v['meta']['has_month']}")
