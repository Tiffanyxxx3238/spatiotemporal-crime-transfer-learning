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
    # ── new cities (added after initial 14) ───────────────────────────────────
    ('Seattle',       'Seattle',        [47.60,-122.33], 11, f'{MODEL_DIR}/grid_risk_seattle.csv'),
    ('San Francisco', 'San Francisco',  [37.77,-122.42], 12, f'{MODEL_DIR}/grid_risk_sf.csv'),
    ('Dallas',        'Dallas',         [32.78, -96.80], 11, f'{MODEL_DIR}/grid_risk_dallas.csv'),
    ('Lansing',       'Lansing',        [42.73, -84.55], 13, f'{MODEL_DIR}/grid_risk_lansing.csv'),
    ('Dayton',        'Dayton',         [39.76, -84.19], 13, f'{MODEL_DIR}/grid_risk_dayton.csv'),
    ('Little Rock',   'Little Rock',    [34.75, -92.29], 13, f'{MODEL_DIR}/grid_risk_little_rock.csv'),
]

print("Reading HTML...")
with open(HTML_IN, encoding='utf-8') as f:
    html = f.read()

# ── Parse existing CITY_DATA ──────────────────────────────────────────────────
def parse_city_data(html_text):
    idx   = html_text.index('const CITY_DATA=')
    start = idx + len('const CITY_DATA=')
    end_m = re.search(r'\nconst [A-Z]', html_text[start:])
    end   = start + end_m.start()
    data  = json.loads(html_text[start:end].rstrip(';\n\r '))
    return data, start, end

city_data, _, _ = parse_city_data(html)
print(f"Existing cities in v7: {list(city_data.keys())}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def entropy3(pv, pp, po):
    probs = np.clip([pv, pp, po], 1e-9, 1)
    probs /= probs.sum()
    return float(-np.sum(probs * np.log(probs)))

def make_conf_tier_fn(conf_values):
    """Return a tier function based on city-relative percentiles.

    Isotonic-calibrated probabilities are often compressed into a narrow band
    (e.g. all NYC grids land in [0.40, 0.71]), so a fixed 0.80 threshold gives
    zero 'high' grids. Instead, use the city's own 80th / 50th percentiles so
    every city gets a meaningful three-tier split.
    """
    p80 = float(np.percentile(conf_values, 80))
    p50 = float(np.percentile(conf_values, 50))
    def tier(c):
        if c >= p80: return 'high'
        if c >= p50: return 'medium'
        return 'uncertain'
    return tier, p80, p50

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
    # Cities like Karachi have hour=0 for all records → only time_slot 0 exists.
    # Mark has_timeslot=False so the map shows all grids regardless of selected slot.
    has_timeslot = int(df['time_slot'].nunique()) > 1

    # Build city-relative confidence tier thresholds.
    # Use rounded values (3dp) to match what gets stored in each grid object,
    # avoiding float precision mismatches between np.percentile and the tier check.
    conf_col = df['conf'].astype(float) if 'conf' in df.columns else pd.Series([0.5])
    conf_values_rounded = conf_col.round(3).values
    tier_fn, p80_conf, p50_conf = make_conf_tier_fn(conf_values_rounded)

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
            'ok':   bool(r.get('ok', r.get('correct', False))),
            'risk': round(float(r.get('risk', r.get('risk_score', 0))), 1),
            'pv': pv, 'pp': pp, 'po': po,
            'gap': gap, 'ent': ent,
        }
        if has_month:
            g['mo'] = int(r['month'])
        grids.append(g)

    ok_arr  = [g['ok']   for g in grids]
    cf_arr  = [g['conf'] for g in grids]
    raw_arr = [g['risk'] for g in grids]
    acc     = round(sum(ok_arr) / max(len(ok_arr), 1) * 100, 1)

    # Normalise risk to 0-100 within city so the alert-threshold slider works
    # consistently regardless of how violent the city is.
    # raw risk = proba_violent * 100, which maxes at 23 for LA, 34 for NYC, etc.
    # After normalisation, every city's highest-risk grid = 100.
    r_min = float(np.min(raw_arr)) if raw_arr else 0
    r_max = float(np.max(raw_arr)) if raw_arr else 0
    r_range = r_max - r_min
    for g in grids:
        if r_range > 0:
            g['risk'] = round((g['risk'] - r_min) / r_range * 100, 1)
        else:
            g['risk'] = 0.0

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
# Re-parse positions from the (possibly modified) html so tab insertion doesn't corrupt offsets
print("Serialising CITY_DATA...")
new_json = json.dumps(new_city_data, ensure_ascii=False, separators=(',', ':'))
_, start, end = parse_city_data(html)
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

# (no per-city threshold patch needed — risk is now normalised 0-100 per city)

# ── has_timeslot: skip time-slot filter for cities where all records share one slot ──
# Karachi synthetic data has hour=0 for all records, producing only time_slot 0.
# Patch renderCity to skip the ts filter when meta.has_timeslot is false.
_old_ts_filter = (
    "data.grids.filter(g=>{"
    "if(g.ts!==timeSlot)return false;"
    "if(currentSeason===null||!data.meta.has_month)return true;"
    "return currentSeason.includes(g.mo);})"
)
_new_ts_filter = (
    "data.grids.filter(g=>{"
    "if(data.meta.has_timeslot!==false&&g.ts!==timeSlot)return false;"
    "if(currentSeason===null||!data.meta.has_month)return true;"
    "return currentSeason.includes(g.mo);})"
)
if _old_ts_filter in html:
    html = html.replace(_old_ts_filter, _new_ts_filter, 1)
    print("Patched renderCity: time-slot filter skipped for has_timeslot=false cities")
else:
    print("WARNING: time-slot filter patch not found")

# ── Dim time-slider and disable play when has_timeslot=false (e.g. Karachi) ───
# The slider section has no meaningful use when all grids share one time slot.
_old_switch_render = (
    "document.getElementById('route-result').style.display='none';\n"
    "  renderCity(city,currentTime);"
)
_new_switch_render = (
    "document.getElementById('route-result').style.display='none';\n"
    "  const _hasTs=!(CITY_DATA[city]&&CITY_DATA[city].meta&&CITY_DATA[city].meta.has_timeslot===false);\n"
    "  const _tsWrap=document.getElementById('time-slider-wrap');\n"
    "  if(_tsWrap)_tsWrap.parentElement.style.opacity=_hasTs?'1':'0.35';\n"
    "  if(_tsWrap)_tsWrap.parentElement.style.pointerEvents=_hasTs?'':'none';\n"
    "  if(!_hasTs&&playTimer){clearInterval(playTimer);playTimer=null;"
    "document.getElementById('play-btn').classList.remove('playing');}\n"
    "  renderCity(city,currentTime);"
)
if _old_switch_render in html:
    html = html.replace(_old_switch_render, _new_switch_render, 1)
    print("Patched switchCity: time-slider dims for has_timeslot=false cities")
else:
    print("WARNING: switchCity time-slider dim patch not found")

# ── renderTimeDist: show 全天 notice instead of 4 empty slots for Karachi ────
_old_ts_dist = (
    "function renderTimeDist(city){\n"
    "  const TIME_NAMES=getTimeNames();\n"
    "  const allGrids=CITY_DATA[city].grids;"
)
_new_ts_dist = (
    "function renderTimeDist(city){\n"
    "  const _d=CITY_DATA[city];\n"
    "  if(_d&&_d.meta&&_d.meta.has_timeslot===false){\n"
    "    document.getElementById('ts-dist').innerHTML="
    "`<div class='ts-note' style='text-align:center;padding:10px 0;color:var(--muted)'>"
    "此城市無時間維度資料<br><span style='font-size:9px'>No time-of-day info · all records shown as a single slot</span></div>`;"
    "return;}\n"
    "  const TIME_NAMES=getTimeNames();\n"
    "  const allGrids=CITY_DATA[city].grids;"
)
if _old_ts_dist in html:
    html = html.replace(_old_ts_dist, _new_ts_dist, 1)
    print("Patched renderTimeDist: full-day notice for has_timeslot=false cities")
else:
    print("WARNING: renderTimeDist patch not found")

# ── Auto-scroll sidebar to show detail panel when grid is clicked ─────────────
# detail-panel is the last element in the sidebar; without scrollIntoView the
# user has to manually scroll down to see it after clicking a grid.
_old_detail = (
    "  document.getElementById('pn-v').textContent=(g.pv*100).toFixed(1)+'%';\n"
    "  document.getElementById('pn-p').textContent=(g.pp*100).toFixed(1)+'%';\n"
    "  document.getElementById('pn-o').textContent=(g.po*100).toFixed(1)+'%';\n"
    "}"
)
_new_detail = (
    "  document.getElementById('pn-v').textContent=(g.pv*100).toFixed(1)+'%';\n"
    "  document.getElementById('pn-p').textContent=(g.pp*100).toFixed(1)+'%';\n"
    "  document.getElementById('pn-o').textContent=(g.po*100).toFixed(1)+'%';\n"
    "  document.getElementById('detail-panel').scrollIntoView({behavior:'smooth',block:'nearest'});\n"
    "}"
)
if _old_detail in html:
    html = html.replace(_old_detail, _new_detail, 1)
    print("Patched showDetail: auto-scroll to detail panel on grid click")
else:
    print("WARNING: showDetail scroll patch not found")

# ── Write ─────────────────────────────────────────────────────────────────────
print(f"Writing {HTML_OUT}...")
with open(HTML_OUT, 'w', encoding='utf-8') as f:
    f.write(html)

size_mb = len(html.encode('utf-8')) / 1e6
print(f"Done! {size_mb:.1f} MB")
print(f"Total cities: {len(new_city_data)}")
for k, v in new_city_data.items():
    print(f"  {k:20s}: {v['stats']['total']:6,} grids, has_month={v['meta']['has_month']}")
