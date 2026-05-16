#!/usr/bin/env python3
"""
upgrade_v8.py — Injects real CITY_DATA from crime_map_v8.html into template.html
(template.html contains the full upgraded UI layout; v8.html has the real 17-city data)
"""
import re, os

V8_PATH       = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\crime_map_v8.html'
TEMPLATE_PATH = r'C:\Users\user\GitHub\model-predict-crime\outputs\maps\template.html'

print('Reading crime_map_v8.html (source of real CITY_DATA)...')
with open(V8_PATH, encoding='utf-8') as f:
    old = f.read()

# ── Extract CITY_DATA from v8 ─────────────────────────────────────────────────
idx   = old.index('const CITY_DATA=')
start = idx + len('const CITY_DATA=')
end_m = re.search(r'\nconst [A-Z]', old[start:])
end   = start + end_m.start()
CD    = old[start:end].rstrip(';\n\r ')
print(f'Extracted CITY_DATA from v8: {len(CD)//1024} KB')

print('Reading template.html (source of UI layout)...')
with open(TEMPLATE_PATH, encoding='utf-8') as f:
    template = f.read()

# ── Build new HTML ────────────────────────────────────────────────────────────
PLACEHOLDER = '/*__CITY_DATA_HERE__*/'

# ── Replace template's demo CITY_DATA with real data ─────────────────────────
t_idx   = template.index('const CITY_DATA=')
t_start = t_idx + len('const CITY_DATA=')
# find the next `const` or `</script>` to know where demo data ends
t_end_m = re.search(r'(;\s*\n(?:const |let |var |function |//)|\s*</script>)', template[t_start:])
t_end   = t_start + t_end_m.start()

new_html = template[:t_start] + CD + template[t_end:]

print('Writing upgraded crime_map_v8.html ...')
with open(V8_PATH, 'w', encoding='utf-8') as f:
    f.write(new_html)

size_mb = os.path.getsize(V8_PATH) / 1024 / 1024
print(f'Done. {size_mb:.1f} MB')
