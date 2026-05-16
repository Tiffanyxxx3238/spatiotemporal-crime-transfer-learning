"""
Patch crime_map_v8.html in-place: add Seattle / SF / Dallas entries.
Reads the 3 new grid_risk CSVs, builds CITY_DATA entries, inserts them
into the existing CITY_DATA block, and adds tab buttons.
Does NOT touch the 14 existing cities or any JS patches.
"""
import json, re, os
import pandas as pd
import numpy as np

HTML_PATH = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\crime_map_v8.html'
MODEL_DIR = r'C:\Users\user\GitHub\model-predict-crime\outputs\models'

NEW_CITIES = [
    ('Seattle',       'Seattle',        [47.60,-122.33], 11, f'{MODEL_DIR}/grid_risk_seattle.csv'),
    ('San Francisco', 'San Francisco',  [37.77,-122.42], 12, f'{MODEL_DIR}/grid_risk_sf.csv'),
    ('Dallas',        'Dallas',         [32.78, -96.80], 11, f'{MODEL_DIR}/grid_risk_dallas.csv'),
]

# ── Helpers (copied from update_map_v8.py) ────────────────────────────────────
def parse_city_data(html_text):
    idx   = html_text.index('const CITY_DATA=')
    start = idx + len('const CITY_DATA=')
    end_m = re.search(r'\nconst [A-Z]', html_text[start:])
    end   = start + end_m.start()
    decoder = json.JSONDecoder()
    data, json_len = decoder.raw_decode(html_text[start:end])
    return data, start, start + json_len

def entropy3(pv, pp, po):
    probs = np.clip([pv, pp, po], 1e-9, 1)
    probs /= probs.sum()
    return float(-np.sum(probs * np.log(probs)))

def make_conf_tier_fn(conf_values):
    p80 = float(np.percentile(conf_values, 80))
    p50 = float(np.percentile(conf_values, 50))
    def tier(c):
        if c >= p80: return 'high'
        if c >= p50: return 'medium'
        return 'uncertain'
    return tier, p80, p50

def build_city_entry(csv_path, center, zoom):
    df = pd.read_csv(csv_path)
    has_month    = 'month' in df.columns
    has_timeslot = int(df['time_slot'].nunique()) > 1

    conf_col = df['conf'].astype(float) if 'conf' in df.columns else pd.Series([0.5])
    conf_rounded = conf_col.round(3).values
    tier_fn, p80_conf, p50_conf = make_conf_tier_fn(conf_rounded)

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
            'tier': tier_fn(cf),
            'ok':   bool(r.get('ok', False)),
            'risk': round(float(r.get('risk', 0)), 1),
            'pv': pv, 'pp': pp, 'po': po,
            'gap': gap, 'ent': ent,
        }
        if has_month:
            g['mo'] = int(r['month'])
        grids.append(g)

    ok_arr  = [g['ok']   for g in grids]
    raw_arr = [g['risk'] for g in grids]
    acc     = round(sum(ok_arr) / max(len(ok_arr), 1) * 100, 1)

    r_min   = float(np.min(raw_arr)) if raw_arr else 0
    r_max   = float(np.max(raw_arr)) if raw_arr else 0
    r_range = r_max - r_min
    for g in grids:
        g['risk'] = round((g['risk'] - r_min) / r_range * 100, 1) if r_range > 0 else 0.0

    rk_arr = [g['risk'] for g in grids]
    return {
        'meta':  {'center': center, 'zoom': zoom, 'has_month': has_month,
                  'has_timeslot': has_timeslot,
                  'risk_min': round(r_min, 1), 'risk_max': round(r_max, 1)},
        'stats': {
            'total':    len(grids),
            'acc':      acc,
            'high':     sum(1 for g in grids if g['tier'] == 'high'),
            'med':      sum(1 for g in grids if g['tier'] == 'medium'),
            'unc':      sum(1 for g in grids if g['tier'] == 'uncertain'),
            'avg_risk': round(float(np.mean(rk_arr)), 1) if rk_arr else 0,
            'max_risk': 100.0,
            'conf_p80': round(p80_conf, 3),
            'conf_p50': round(p50_conf, 3),
        },
        'grids': grids,
    }

# ── Load v8 ───────────────────────────────────────────────────────────────────
print("Reading crime_map_v8.html...")
with open(HTML_PATH, encoding='utf-8') as f:
    html = f.read()

city_data, _, _ = parse_city_data(html)
print(f"Existing cities: {list(city_data.keys())}")

# ── Build new city entries ────────────────────────────────────────────────────
added = []
for key, display, center, zoom, csv_path in NEW_CITIES:
    if key in city_data:
        print(f"  SKIP {key}: already in map")
        continue
    if not os.path.exists(csv_path):
        print(f"  SKIP {key}: CSV not found ({csv_path})")
        continue
    print(f"  Building {key}...")
    entry = build_city_entry(csv_path, center, zoom)
    s = entry['stats']
    city_data[key] = entry
    added.append((key, display, s))
    print(f"    {s['total']:,} grids, acc={s['acc']}%, has_timeslot={entry['meta']['has_timeslot']}")

if not added:
    print("Nothing to add — exiting.")
    exit(0)

# ── Add tab buttons for new cities ───────────────────────────────────────────
# Insert after the last existing tab (Karachi), before the closing </div> of tab bar
for key, display, s in added:
    btn_id  = f"switchCity('{key}')"
    if btn_id in html:
        print(f"  Tab already exists: {key}")
        continue
    # Find Karachi tab button and insert after it
    karachi_pattern = r"(<button[^>]*onclick=[\"']switchCity\('Karachi'\)[\"'][^>]*>[^<]*</button>)"
    m = re.search(karachi_pattern, html)
    if m:
        new_btn = (
            f"<button class=\"city-tab\" onclick=\"switchCity('{key}')\" "
            f"style=\"white-space:nowrap\">{display}</button>"
        )
        html = html[:m.end()] + new_btn + html[m.end():]
        print(f"  Tab added: {key}")
    else:
        print(f"  WARNING: could not find Karachi tab to insert {key} after")

# ── Inject updated CITY_DATA ──────────────────────────────────────────────────
print("Serialising updated CITY_DATA...")
new_json = json.dumps(city_data, ensure_ascii=False, separators=(',', ':'))
_, start, json_end = parse_city_data(html)
html = html[:start] + new_json + html[json_end:]

# ── Write back to v8 ─────────────────────────────────────────────────────────
with open(HTML_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

size_mb = os.path.getsize(HTML_PATH) / 1024 / 1024
print(f"\nDone. crime_map_v8.html updated → {size_mb:.1f} MB")
print(f"Added cities: {[k for k,_,_ in added]}")
