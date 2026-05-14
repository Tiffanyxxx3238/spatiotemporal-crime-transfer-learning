"""
Update crime_map_v4.html → crime_map_v5.html
- Add Salt Lake City and Birmingham city data
- Add city tabs for both new cities
"""
import json, re
import pandas as pd
import numpy as np

HTML_IN  = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\crime_map_v4.html'
HTML_OUT = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\crime_map_v5.html'
MODEL_DIR = r'C:\Users\user\GitHub\model-predict-crime\outputs\models'

# ── 1. Load existing HTML ─────────────────────────────────────────────────────
print("Reading HTML...")
with open(HTML_IN, encoding='utf-8') as f:
    html = f.read()

# ── 2. Parse existing CITY_DATA ───────────────────────────────────────────────
idx   = html.index('const CITY_DATA=')
start = idx + len('const CITY_DATA=')
end_m = re.search(r'\nconst [A-Z]', html[start:])
end   = start + end_m.start()
city_data = json.loads(html[start:end].rstrip(';\n\r '))
print(f"Existing cities: {list(city_data.keys())}")

# ── 3. Helper: build city entry from grid_risk CSV ────────────────────────────
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
    ok_arr  = [g['ok']   for g in grids]
    cf_arr  = [g['conf'] for g in grids]
    rk_arr  = [g['risk'] for g in grids]
    acc     = round(sum(ok_arr) / max(len(ok_arr), 1) * 100, 1)
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

# ── 4. Generate SLC & Birmingham entries ──────────────────────────────────────
print("Building Salt Lake City entry...")
city_data['Salt Lake City'] = build_city_entry(
    f'{MODEL_DIR}/grid_risk_salt_lake_city.csv',
    center=[40.76, -111.89], zoom=12
)
slc_stats = city_data['Salt Lake City']['stats']
print(f"  SLC: {slc_stats['total']} grids, acc={slc_stats['acc']}%")

print("Building Birmingham entry...")
city_data['Birmingham'] = build_city_entry(
    f'{MODEL_DIR}/grid_risk_birmingham.csv',
    center=[33.52, -86.80], zoom=12
)
brm_stats = city_data['Birmingham']['stats']
print(f"  Birmingham: {brm_stats['total']} grids, acc={brm_stats['acc']}%")

# ── 5. Serialize updated CITY_DATA ────────────────────────────────────────────
print("Serialising CITY_DATA...")
new_json = json.dumps(city_data, ensure_ascii=False, separators=(',', ':'))
html = html[:start] + new_json + html[end:]

# ── 6. Add city tabs for SLC & Birmingham ─────────────────────────────────────
old_tabs = """    <button class="city-tab" onclick="switchCity('Cambridge')">Cambridge</button>
  </div>"""
new_tabs = """    <button class="city-tab" onclick="switchCity('Cambridge')">Cambridge</button>
    <button class="city-tab" onclick="switchCity('Salt Lake City')">SLC</button>
    <button class="city-tab" onclick="switchCity('Birmingham')">Birmingham</button>
  </div>"""
html = html.replace(old_tabs, new_tabs, 1)

# ── 7. Write output ───────────────────────────────────────────────────────────
print("Writing crime_map_v5.html...")
with open(HTML_OUT, 'w', encoding='utf-8') as f:
    f.write(html)

size_mb = len(html.encode('utf-8')) / 1e6
print(f"Done! {HTML_OUT} ({size_mb:.1f} MB)")
print(f"Cities: {list(city_data.keys())}")
