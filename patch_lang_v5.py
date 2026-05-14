"""
Patch crime_map_v5.html to fix the EN language toggle:
1. Correct SEC_TITLES_ZH/EN to match actual HTML sec-title texts
2. Correct LAYER_LABELS_ZH/EN to match actual HTML layer-row texts
3. Make switchCity() re-apply lang after re-rendering so translations persist
"""

HTML_IN  = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\crime_map_v5.html'
HTML_OUT = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\crime_map_v5.html'

print("Reading...")
with open(HTML_IN, encoding='utf-8') as f:
    html = f.read()

# ── Fix 1: Correct SEC_TITLES arrays ─────────────────────────────────────────
# Actual sec-title texts in the HTML (verified by inspection):
#   '時間段 — 動態播放', '城市統計', '告警閾值設定', '路徑風險查詢',
#   '圖例 — 預測類別', '透明度 — 信心程度', '圖層控制',
#   'Top 10 高風險格子', 'Time Slot Distribution', '格子詳情'

old_sec_zh = "const SEC_TITLES_ZH = ['時間段 — 動態播放','城市統計','信心度分佈','圖例','圖層','告警閾值','路徑風險查詢','Top 10 高風險格子','時段分佈'];"
new_sec_zh = "const SEC_TITLES_ZH = ['時間段 — 動態播放','城市統計','告警閾值設定','路徑風險查詢','圖例 — 預測類別','透明度 — 信心程度','圖層控制','Top 10 高風險格子','Time Slot Distribution','格子詳情'];"

old_sec_en = "const SEC_TITLES_EN = ['Time Slot — Auto Play','City Statistics','Confidence Distribution','Legend','Layers','Alert Threshold','Route Risk Query','Top 10 High-Risk Grids','Time Distribution'];"
new_sec_en = "const SEC_TITLES_EN = ['Time Slot — Auto Play','City Statistics','Alert Threshold','Route Risk Query','Legend — Prediction','Opacity — Confidence','Layer Control','Top 10 High-Risk Grids','Time Slot Distribution','Grid Details'];"

n = html.count(old_sec_zh)
html = html.replace(old_sec_zh, new_sec_zh, 1)
print(f"Fix 1a SEC_TITLES_ZH: replaced {n} occurrence(s)")

n = html.count(old_sec_en)
html = html.replace(old_sec_en, new_sec_en, 1)
print(f"Fix 1b SEC_TITLES_EN: replaced {n} occurrence(s)")

# ── Fix 2: Correct LAYER_LABELS arrays ───────────────────────────────────────
# Actual layer-row span texts: '預測類別', '暴力風險熱力圖', '告警格子標記'

old_layer_zh = "const LAYER_LABELS_ZH = ['預測類別','暴力熱力圖','告警標記'];"
new_layer_zh = "const LAYER_LABELS_ZH = ['預測類別','暴力風險熱力圖','告警格子標記'];"

old_layer_en = "const LAYER_LABELS_EN = ['Prediction','Violence Heatmap','Alert Marks'];"
new_layer_en = "const LAYER_LABELS_EN = ['Prediction','Violence Heatmap','Alert Markers'];"

n = html.count(old_layer_zh)
html = html.replace(old_layer_zh, new_layer_zh, 1)
print(f"Fix 2a LAYER_LABELS_ZH: replaced {n} occurrence(s)")

n = html.count(old_layer_en)
html = html.replace(old_layer_en, new_layer_en, 1)
print(f"Fix 2b LAYER_LABELS_EN: replaced {n} occurrence(s)")

# ── Fix 3: Make switchCity() re-apply lang after rendering ───────────────────
old_switch = "  renderCity(city,currentTime);\n}"
new_switch = "  renderCity(city,currentTime);\n  if(typeof currentLang!=='undefined'&&currentLang==='en') applyLang();\n}"

n = html.count(old_switch)
html = html.replace(old_switch, new_switch, 1)
print(f"Fix 3 switchCity lang persist: replaced {n} occurrence(s)")

# ── Fix 4: Also re-apply lang after switchTimeSlot ───────────────────────────
old_ts = "function switchTimeSlot(ts){\n  currentTime=ts;\n  renderCity(currentCity,ts);\n}"
new_ts  = "function switchTimeSlot(ts){\n  currentTime=ts;\n  renderCity(currentCity,ts);\n  if(typeof currentLang!=='undefined'&&currentLang==='en') applyLang();\n}"

n = html.count(old_ts)
html = html.replace(old_ts, new_ts, 1)
print(f"Fix 4 switchTimeSlot lang persist: replaced {n} occurrence(s)")

# ── Write ─────────────────────────────────────────────────────────────────────
with open(HTML_OUT, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nDone! Written to {HTML_OUT}")
