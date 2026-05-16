"""
Update crime_map_v6.html → crime_map_v7.html
- Rebuild Kansas City data with month (mo) field from Method B grid_risk
- Add season selector UI (visible only for cities with month data)
- Patch renderCity JS to filter by season
"""
import json, re
import pandas as pd
import numpy as np

HTML_IN   = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\crime_map_v6.html'
HTML_OUT  = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\crime_map_v7.html'
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

def build_city_entry_with_month(csv_path, center, zoom):
    """Build city entry with mo field (Method B grid_risk has month column)."""
    df = pd.read_csv(csv_path)
    has_month = 'month' in df.columns
    grids = []
    for _, r in df.iterrows():
        pv  = round(float(r.get('p_violent',  0)), 3)
        pp  = round(float(r.get('p_property', 0)), 3)
        po  = round(float(r.get('p_other',    0)), 3)
        cf  = round(float(r['conf']), 3)
        probs_sorted = sorted([pv, pp, po], reverse=True)
        gap = round(probs_sorted[0] - probs_sorted[1], 3)
        ent = round(entropy3(pv, pp, po), 3)
        g = {
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
            'avg_risk': round(float(np.mean(rk_arr)), 1),
            'max_risk': round(float(np.max(rk_arr)),  1),
        },
        'grids': grids,
    }

# ── Rebuild Kansas City with month data ───────────────────────────────────────
print("Rebuilding Kansas City entry (Method B, with month)...")
city_data['Kansas City'] = build_city_entry_with_month(
    f'{MODEL_DIR}/grid_risk_kansas_city.csv',
    center=[39.10, -94.58], zoom=12
)
s = city_data['Kansas City']['stats']
print(f"  Kansas City: {s['total']} grids, acc={s['acc']}%")

# ── Inject updated CITY_DATA ──────────────────────────────────────────────────
print("Serialising CITY_DATA...")
new_json = json.dumps(city_data, ensure_ascii=False, separators=(',', ':'))
html = html[:start] + new_json + html[end:]

# ── Patch 1: Add currentSeason state variable ─────────────────────────────────
html = html.replace(
    "let map,currentCity='NYC',currentTime=0;",
    "let map,currentCity='NYC',currentTime=0,currentSeason=null;"
    # null = all months shown
)

# ── Patch 2: Update renderCity to filter by season ───────────────────────────
old_render = "const grids=data.grids.filter(g=>g.ts===timeSlot);"
new_render = (
    "const grids=data.grids.filter(g=>{"
    "if(g.ts!==timeSlot)return false;"
    "if(currentSeason===null||!data.meta.has_month)return true;"
    "return currentSeason.includes(g.mo);"
    "});"
)
html = html.replace(old_render, new_render, 1)

# ── Patch 3: Add switchSeason function before renderCity ─────────────────────
season_fn = """
function setSeasonBtnStyle(b,active){
  b.style.background=active?'var(--accent)':'var(--panel2)';
  b.style.color=active?'#fff':'var(--text)';
  b.style.borderColor=active?'var(--accent)':'var(--border)';
  b.classList.toggle('active',active);
}
function switchSeason(months){
  currentSeason=months;
  document.querySelectorAll('.season-btn').forEach(b=>{
    const active=months===null?b.dataset.season==='all'
      :b.dataset.months===JSON.stringify(months);
    setSeasonBtnStyle(b,active);
  });
  renderCity(currentCity,currentTime);
}
"""
html = html.replace(
    "function renderCity(city,timeSlot){",
    season_fn + "function renderCity(city,timeSlot){"
)

# ── Patch 4: Reset season when switching city ─────────────────────────────────
old_switch = "function switchCity(city){"
new_switch = (
    "function switchCity(city){"
    "currentSeason=null;"
    "const hasMonth=CITY_DATA[city]&&CITY_DATA[city].meta&&CITY_DATA[city].meta.has_month;"
    "const sw=document.getElementById('season-wrap');"
    "if(sw)sw.style.display=hasMonth?'block':'none';"
    "document.querySelectorAll('.season-btn').forEach(b=>setSeasonBtnStyle(b,b.dataset.season==='all'));"
)
html = html.replace(old_switch, new_switch, 1)

# ── Patch 4b: After initial renderCity call, highlight 全年 button ────────────
html = html.replace(
    "renderCity('NYC',0);",
    "renderCity('NYC',0);"
    "document.querySelectorAll('.season-btn').forEach(b=>setSeasonBtnStyle(b,b.dataset.season==='all'));"
)

# ── Patch 5: Inject season selector HTML after time-slider-wrap ──────────────
season_html = """
<div id="season-wrap" style="display:none;padding:0 14px 12px;">
  <div style="font-size:11px;color:var(--muted);margin-bottom:5px;" id="season-label">季節篩選 Season Filter</div>
  <div style="display:flex;gap:5px;flex-wrap:wrap;">
    <button class="season-btn active" data-season="all" data-months="null"
      onclick="switchSeason(null)"
      style="flex:1;padding:5px 4px;background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:5px;font-size:11px;cursor:pointer;">
      全年 All
    </button>
    <button class="season-btn" data-season="spring" data-months="[3,4,5]"
      onclick="switchSeason([3,4,5])"
      style="flex:1;padding:5px 4px;background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:5px;font-size:11px;cursor:pointer;">
      春 Spring
    </button>
    <button class="season-btn" data-season="summer" data-months="[6,7,8]"
      onclick="switchSeason([6,7,8])"
      style="flex:1;padding:5px 4px;background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:5px;font-size:11px;cursor:pointer;">
      夏 Summer
    </button>
    <button class="season-btn" data-season="fall" data-months="[9,10,11]"
      onclick="switchSeason([9,10,11])"
      style="flex:1;padding:5px 4px;background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:5px;font-size:11px;cursor:pointer;">
      秋 Fall
    </button>
    <button class="season-btn" data-season="winter" data-months="[12,1,2]"
      onclick="switchSeason([12,1,2])"
      style="flex:1;padding:5px 4px;background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:5px;font-size:11px;cursor:pointer;">
      冬 Winter
    </button>
  </div>
</div>
"""
# Insert after the closing </div> of time-slider-wrap
html = html.replace(
    '#time-slider-wrap{padding:0 14px 12px;}',
    '#time-slider-wrap{padding:0 14px 12px;}#season-wrap .season-btn.active{background:var(--accent)!important;color:#fff!important;border-color:var(--accent)!important;}'
)

# Find the play-btn closing area and insert season panel after it
play_btn_close = 'id="play-btn"'
idx_play = html.rfind(play_btn_close)
# Find the next </div> after play-btn
close_div_idx = html.find('</div>', idx_play)
html = html[:close_div_idx+6] + season_html + html[close_div_idx+6:]

# ── Write ─────────────────────────────────────────────────────────────────────
print(f"Writing {HTML_OUT}...")
with open(HTML_OUT, 'w', encoding='utf-8') as f:
    f.write(html)

size_mb = len(html.encode('utf-8')) / 1e6
print(f"Done! {size_mb:.1f} MB")
print("Season selector added — visible only for Kansas City")
