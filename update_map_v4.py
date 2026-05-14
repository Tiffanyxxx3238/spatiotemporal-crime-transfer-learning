"""
Update crime_map_v3.html → crime_map_v4.html
- Add Detroit and Cambridge city data
- Add Chinese/English language toggle
- Add city tabs for Detroit and Cambridge
"""
import json, re, math
import pandas as pd
import numpy as np

HTML_IN  = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\crime_map_v3.html'
HTML_OUT = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\crime_map_v4.html'
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

# ── 4. Generate Detroit & Cambridge entries ───────────────────────────────────
print("Building Detroit entry...")
city_data['Detroit']   = build_city_entry(
    f'{MODEL_DIR}/grid_risk_detroit.csv',
    center=[42.33, -83.05], zoom=12
)
print(f"  Detroit: {city_data['Detroit']['stats']['total']} grids, acc={city_data['Detroit']['stats']['acc']}%")

print("Building Cambridge entry...")
city_data['Cambridge'] = build_city_entry(
    f'{MODEL_DIR}/grid_risk_cambridge.csv',
    center=[52.205, 0.120], zoom=14
)
print(f"  Cambridge: {city_data['Cambridge']['stats']['total']} grids, acc={city_data['Cambridge']['stats']['acc']}%")

# ── 5. Serialize updated CITY_DATA ────────────────────────────────────────────
print("Serialising CITY_DATA...")
new_json = json.dumps(city_data, ensure_ascii=False, separators=(',', ':'))
html = html[:start] + new_json + html[end:]

# ── 6. Add city tabs for Detroit & Cambridge ──────────────────────────────────
old_tabs = """    <button class="city-tab" onclick="switchCity('West Yorkshire')">W.Yorks</button>
  </div>"""
new_tabs = """    <button class="city-tab" onclick="switchCity('West Yorkshire')">W.Yorks</button>
    <button class="city-tab" onclick="switchCity('Detroit')">Detroit</button>
    <button class="city-tab" onclick="switchCity('Cambridge')">Cambridge</button>
  </div>"""
html = html.replace(old_tabs, new_tabs, 1)

# ── 7. Add language toggle button ────────────────────────────────────────────
# Replace the theme button line to add lang button next to it
old_header = '''    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
      <h1 style="margin-bottom:0">Crime Pattern Analysis</h1>
      <button id="theme-btn" onclick="toggleTheme()">☀ 亮色</button>
    </div>
    <p>跨城市犯罪預測 · 時空遷移學習框架</p>'''
new_header = '''    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
      <h1 style="margin-bottom:0" id="i18n-title">Crime Pattern Analysis</h1>
      <div style="display:flex;gap:5px;">
        <button id="lang-btn" onclick="toggleLang()" style="background:none;border:1px solid var(--border);color:var(--muted);font-family:var(--sans);font-size:10px;padding:3px 7px;border-radius:5px;cursor:pointer;transition:all .15s;">EN</button>
        <button id="theme-btn" onclick="toggleTheme()">☀ 亮色</button>
      </div>
    </div>
    <p id="i18n-subtitle">跨城市犯罪預測 · 時空遷移學習框架</p>'''
html = html.replace(old_header, new_header, 1)

# ── 8. Add language toggle CSS ────────────────────────────────────────────────
old_theme_css = '/* ── Theme toggle button ── */'
new_lang_css = '''/* ── Lang toggle ── */
#lang-btn:hover{border-color:var(--accent);color:var(--accent);}
/* ── Theme toggle button ── */'''
html = html.replace(old_theme_css, new_lang_css, 1)

# ── 9. Add i18n spans to time slot labels ────────────────────────────────────
old_time_labels = '<div class="slider-labels"><span>深夜</span><span>早晨</span><span>下午</span><span>夜晚</span></div>'
new_time_labels = '<div class="slider-labels"><span data-zh="深夜" data-en="Night">深夜</span><span data-zh="早晨" data-en="Morning">早晨</span><span data-zh="下午" data-en="Afternoon">下午</span><span data-zh="夜晚" data-en="Evening">夜晚</span></div>'
html = html.replace(old_time_labels, new_time_labels, 1)

# ── 10. Add i18n to play button ──────────────────────────────────────────────
old_play = '<button id="play-btn" onclick="togglePlay()">▶ 自動播放時間演變</button>'
new_play = '<button id="play-btn" onclick="togglePlay()" data-zh="▶ 自動播放時間演變" data-en="▶ Auto Play">▶ 自動播放時間演變</button>'
html = html.replace(old_play, new_play, 1)

# ── 11. Add i18n JavaScript ──────────────────────────────────────────────────
LANG_JS = r"""
// ── Language toggle ──────────────────────────────────────────────────────────
let currentLang = 'zh';
const I18N = {
  zh: {
    title:    'Crime Pattern Analysis',
    subtitle: '跨城市犯罪預測 · 時空遷移學習框架',
    'sec-time':   '時間段 — 動態播放',
    'sec-stats':  '城市統計',
    'sec-conf':   '信心度分佈',
    'sec-legend': '圖例',
    'sec-layers': '圖層',
    'sec-alert':  '告警閾值',
    'sec-route':  '路徑風險查詢',
    'sec-top10':  'Top 10 高風險格子',
    'sec-tsdist': '時段分佈',
    'stat-total': '格子總數',
    'stat-acc':   '預測準確率',
    'stat-avgrisk': '平均風險',
    'stat-maxrisk': '最高風險',
    'conf-high':  '高信心',
    'conf-med':   '中信心',
    'conf-unc':   '不確定',
    'layer-pred': '預測類別',
    'layer-heat': '暴力熱力圖',
    'layer-alert':'告警標記',
    'route-btn':  '計算路徑風險',
    'theme-btn':  '☀ 亮色',
    'theme-btn-light': '🌙 暗色',
    'detail-coord': '座標',
    'detail-true':  '真實類別',
    'detail-pred':  '預測類別',
    'detail-conf':  '信心度',
    'detail-cnt':   '事件數',
    'detail-risk':  '風險分數',
    'detail-prob':  '各類別機率',
  },
  en: {
    title:    'Crime Pattern Analysis',
    subtitle: 'Cross-City Crime Prediction · Spatiotemporal Transfer Learning',
    'sec-time':   'Time Slot — Auto Play',
    'sec-stats':  'City Statistics',
    'sec-conf':   'Confidence Distribution',
    'sec-legend': 'Legend',
    'sec-layers': 'Layers',
    'sec-alert':  'Alert Threshold',
    'sec-route':  'Route Risk Query',
    'sec-top10':  'Top 10 High-Risk Grids',
    'sec-tsdist': 'Time Distribution',
    'stat-total': 'Total Grids',
    'stat-acc':   'Accuracy',
    'stat-avgrisk': 'Avg Risk',
    'stat-maxrisk': 'Max Risk',
    'conf-high':  'High Conf',
    'conf-med':   'Mid Conf',
    'conf-unc':   'Uncertain',
    'layer-pred': 'Prediction',
    'layer-heat': 'Violence Heatmap',
    'layer-alert':'Alert Marks',
    'route-btn':  'Calculate Route Risk',
    'theme-btn':  '☀ Light',
    'theme-btn-light': '🌙 Dark',
    'detail-coord': 'Coordinates',
    'detail-true':  'True Category',
    'detail-pred':  'Prediction',
    'detail-conf':  'Confidence',
    'detail-cnt':   'Event Count',
    'detail-risk':  'Risk Score',
    'detail-prob':  'Category Probabilities',
  }
};

// Mapping of i18n keys → element IDs or selectors
const I18N_IDS = {
  'title':       {id:'i18n-title'},
  'subtitle':    {id:'i18n-subtitle'},
};

// Elements with sec-title class that need translation (in order they appear)
const SEC_TITLES_ZH = ['時間段 — 動態播放','城市統計','信心度分佈','圖例','圖層','告警閾值','路徑風險查詢','Top 10 高風險格子','時段分佈'];
const SEC_TITLES_EN = ['Time Slot — Auto Play','City Statistics','Confidence Distribution','Legend','Layers','Alert Threshold','Route Risk Query','Top 10 High-Risk Grids','Time Distribution'];

const STAT_LABELS_ZH = ['格子總數','預測準確率','平均風險','最高風險'];
const STAT_LABELS_EN = ['Total Grids','Accuracy','Avg Risk','Max Risk'];
const CONF_LABELS_ZH = ['高信心','中信心','不確定'];
const CONF_LABELS_EN = ['High Conf','Mid Conf','Uncertain'];
const LAYER_LABELS_ZH = ['預測類別','暴力熱力圖','告警標記'];
const LAYER_LABELS_EN = ['Prediction','Violence Heatmap','Alert Marks'];

function toggleLang(){
  currentLang = (currentLang === 'zh') ? 'en' : 'zh';
  document.getElementById('lang-btn').textContent = (currentLang === 'zh') ? 'EN' : '中';
  applyLang();
}

function applyLang(){
  const L = I18N[currentLang];
  // Title & subtitle
  document.getElementById('i18n-title').textContent = L.title;
  document.getElementById('i18n-subtitle').textContent = L.subtitle;

  // data-zh / data-en elements (time labels, play button)
  document.querySelectorAll('[data-zh]').forEach(el=>{
    el.textContent = el.getAttribute('data-' + currentLang);
  });

  // Section titles
  const srcTitles = currentLang==='zh' ? SEC_TITLES_EN : SEC_TITLES_ZH;
  const dstTitles = currentLang==='zh' ? SEC_TITLES_ZH : SEC_TITLES_EN;
  document.querySelectorAll('.sec-title').forEach(el=>{
    const idx = srcTitles.indexOf(el.textContent.trim());
    if(idx >= 0) el.textContent = dstTitles[idx];
  });

  // Stat card labels
  const srcStat = currentLang==='zh' ? STAT_LABELS_EN : STAT_LABELS_ZH;
  const dstStat = currentLang==='zh' ? STAT_LABELS_ZH : STAT_LABELS_EN;
  document.querySelectorAll('.stat-card .label').forEach(el=>{
    const idx = srcStat.indexOf(el.textContent.trim());
    if(idx >= 0) el.textContent = dstStat[idx];
  });

  // Conf pill labels
  const srcConf = currentLang==='zh' ? CONF_LABELS_EN : CONF_LABELS_ZH;
  const dstConf = currentLang==='zh' ? CONF_LABELS_ZH : CONF_LABELS_EN;
  document.querySelectorAll('.conf-pill .cl').forEach(el=>{
    const idx = srcConf.indexOf(el.textContent.trim());
    if(idx >= 0) el.textContent = dstConf[idx];
  });

  // Layer labels
  const srcLayer = currentLang==='zh' ? LAYER_LABELS_EN : LAYER_LABELS_ZH;
  const dstLayer = currentLang==='zh' ? LAYER_LABELS_ZH : LAYER_LABELS_EN;
  document.querySelectorAll('.layer-row').forEach(el=>{
    const span = el.querySelector('span');
    if(!span) return;
    const idx = srcLayer.indexOf(span.textContent.trim());
    if(idx >= 0) span.textContent = dstLayer[idx];
  });

  // Route button
  const routeBtn = document.getElementById('route-btn');
  if(routeBtn) routeBtn.textContent = L['route-btn'];

  // Time slot names in renderTimeDist
  renderTimeDist(currentCity);
}

// Override TIME_NAMES to use current lang
function getTimeNames(){
  return currentLang==='zh'
    ? ['深夜','早晨','下午','夜晚']
    : ['Night','Morning','Afternoon','Evening'];
}
"""

# Insert LANG_JS before the closing </script> tag
old_script_end = '</script>\n</body>'
new_script_end = LANG_JS + '\n</script>\n</body>'
html = html.replace(old_script_end, new_script_end, 1)

# ── 12. Patch renderTimeDist to use getTimeNames() ────────────────────────────
# Replace the hardcoded TIME_NAMES array
old_time_names = "const TIME_NAMES=['深夜','早晨','下午','夜晚'];"
new_time_names = "const TIME_NAMES=getTimeNames();"
html = html.replace(old_time_names, new_time_names, 1)

# ── 13. Write output ──────────────────────────────────────────────────────────
print("Writing crime_map_v4.html...")
with open(HTML_OUT, 'w', encoding='utf-8') as f:
    f.write(html)

size_mb = len(html.encode('utf-8')) / 1e6
print(f"Done! {HTML_OUT} ({size_mb:.1f} MB)")
print(f"Cities: {list(city_data.keys())}")
