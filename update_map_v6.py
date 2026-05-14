"""
Update crime_map_v5_fixed_city_zh.html → crime_map_v6.html
- Add Peoria and Kansas City
"""
import json, re
import pandas as pd
import numpy as np

HTML_IN   = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\crime_map_v5_fixed_city_zh.html'
HTML_OUT  = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\crime_map_v6.html'
MODEL_DIR = r'C:\Users\user\GitHub\model-predict-crime\outputs\models'

print("Reading HTML...")
with open(HTML_IN, encoding='utf-8') as f:
    html = f.read()

# ── Parse existing CITY_DATA ──────────────────────────────────────────────────
idx   = html.index('const CITY_DATA=')
start = idx + len('const CITY_DATA=')
end_m = re.search(r'\nconst [A-Z]', html[start:])
end   = start + end_m.start()
city_data = json.loads(html[start:end].rstrip(';\n\r '))
print(f"Existing cities: {list(city_data.keys())}")

# ── Helper ────────────────────────────────────────────────────────────────────
def entropy3(pv, pp, po):
    probs = np.clip([pv, pp, po], 1e-9, 1)
    probs /= probs.sum()
    return float(-np.sum(probs * np.log(probs)))

def tier(conf):
    if conf >= 0.80: return 'high'
    if conf >= 0.55: return 'medium'
    return 'uncertain'

def build_city_entry(csv_path, center, zoom):
    df = pd.read_csv(csv_path)
    grids = []
    for _, r in df.iterrows():
        pv  = round(float(r.get('p_violent',  0)), 3)
        pp  = round(float(r.get('p_property', 0)), 3)
        po  = round(float(r.get('p_other',    0)), 3)
        cf  = round(float(r['conf']), 3)
        probs_sorted = sorted([pv, pp, po], reverse=True)
        gap = round(probs_sorted[0] - probs_sorted[1], 3)
        ent = round(entropy3(pv, pp, po), 3)
        grids.append({
            'lat':  round(float(r['lat_bin']),  4),
            'lon':  round(float(r['lon_bin']),  4),
            'ts':   int(r['time_slot']),
            'cnt':  int(r['total_count']),
            'dom':  str(r['dominant_category']),
            'pred': str(r['pred']),
            'conf': cf,
            'tier': tier(cf),
            'ok':   bool(r['ok']),
            'risk': round(float(r['risk']), 1),
            'pv':   pv, 'pp': pp, 'po': po,
            'gap':  gap, 'ent': ent,
        })
    ok_arr = [g['ok']   for g in grids]
    cf_arr = [g['conf'] for g in grids]
    rk_arr = [g['risk'] for g in grids]
    acc    = round(sum(ok_arr) / max(len(ok_arr), 1) * 100, 1)
    return {
        'meta':  {'center': center, 'zoom': zoom},
        'stats': {
            'total':    len(grids),
            'acc':      acc,
            'high':     sum(1 for c in cf_arr if c >= 0.80),
            'med':      sum(1 for c in cf_arr if 0.55 <= c < 0.80),
            'unc':      sum(1 for c in cf_arr if c < 0.55),
            'avg_risk': round(float(np.mean(rk_arr)), 1),
            'max_risk': round(float(np.max(rk_arr)),  1),
        },
        'grids': grids,
    }

# ── Build new city entries ────────────────────────────────────────────────────
print("Building Peoria entry...")
city_data['Peoria'] = build_city_entry(
    f'{MODEL_DIR}/grid_risk_peoria.csv',
    center=[40.69, -89.59], zoom=12
)
s = city_data['Peoria']['stats']
print(f"  Peoria: {s['total']} grids, acc={s['acc']}%")

print("Building Kansas City entry...")
city_data['Kansas City'] = build_city_entry(
    f'{MODEL_DIR}/grid_risk_kansas_city.csv',
    center=[39.10, -94.58], zoom=12
)
s = city_data['Kansas City']['stats']
print(f"  Kansas City: {s['total']} grids, acc={s['acc']}%")

# ── Inject updated CITY_DATA ──────────────────────────────────────────────────
print("Serialising CITY_DATA...")
new_json = json.dumps(city_data, ensure_ascii=False, separators=(',', ':'))
html = html[:start] + new_json + html[end:]

# ── Add city tabs ─────────────────────────────────────────────────────────────
old_tab = """onclick="switchCity('Birmingham')">伯明罕</button>"""
new_tab = """onclick="switchCity('Birmingham')">伯明罕</button>
    <button class="city-tab" onclick="switchCity('Peoria')">皮奧里亞</button>
    <button class="city-tab" onclick="switchCity('Kansas City')">堪薩斯城</button>"""
html = html.replace(old_tab, new_tab, 1)

# ── Write ─────────────────────────────────────────────────────────────────────
print(f"Writing {HTML_OUT}...")
with open(HTML_OUT, 'w', encoding='utf-8') as f:
    f.write(html)

size_mb = len(html.encode('utf-8')) / 1e6
print(f"Done! {size_mb:.1f} MB")
print(f"Cities: {list(city_data.keys())}")
