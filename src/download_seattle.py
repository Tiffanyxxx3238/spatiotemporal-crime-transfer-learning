"""
Download Seattle, WA crime data from Seattle Open Data (Socrata)
Dataset: SPD Crime Data 2008–Present
Portal:  data.seattle.gov  resource id: tazs-3rd5
Fields:  report_date_time, latitude, longitude, offense_category
"""
import urllib.request, urllib.parse, json, time
import pandas as pd

OUT_CSV   = r'C:\Users\user\GitHub\model-predict-crime\data\raw\seattle_raw.csv'
PAGE_SIZE = 50000
BASE_URL  = 'https://data.seattle.gov/resource/tazs-3rd5.json'

def map_cat(cat):
    c = str(cat).upper()
    if 'VIOLENT' in c: return 'violent'
    if 'PROPERTY' in c: return 'property'
    return 'other'

records = []
offset  = 0
print('Downloading Seattle crime data...')
while True:
    params = {
        '$limit':  PAGE_SIZE,
        '$offset': offset,
        '$where':  "latitude IS NOT NULL AND longitude IS NOT NULL",
        '$order':  'report_date_time ASC',
    }
    url = BASE_URL + '?' + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            batch = json.loads(r.read().decode())
    except Exception as e:
        print(f'  Error at offset {offset}: {e}'); break
    if not batch:
        break
    for row in batch:
        try:
            lat = float(row.get('latitude',  0) or 0)
            lon = float(row.get('longitude', 0) or 0)
            if lat == 0 or lon == 0:
                continue
            dt_str = row.get('report_date_time', '')
            dt = pd.to_datetime(dt_str, errors='coerce')
            if pd.isna(dt):
                continue
            cat = row.get('offense_category', 'ALL OTHER')
            records.append({
                'datetime':        dt,
                'hour':            dt.hour,
                'month':           dt.month,
                'weekday':         dt.weekday(),
                'latitude':        lat,
                'longitude':       lon,
                'crime_category':  map_cat(cat),
                'raw_offense':     cat,
                'city':            'Seattle',
            })
        except Exception:
            continue
    offset += len(batch)
    print(f'  offset={offset:,}  records={len(records):,}')
    if len(batch) < PAGE_SIZE:
        break
    time.sleep(0.3)

df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
print(f'Saved {len(df):,} rows → {OUT_CSV}')
print(df['crime_category'].value_counts())
