"""
Download Dallas, TX crime data from Dallas Open Data (Socrata)
Dataset: Police Incidents
Portal:  www.dallasopendata.com  resource id: qv6i-rri7
Fields:  date1, y_cordinate, x_coordinate, offincident
"""
import urllib.request, urllib.parse, json, time
import pandas as pd

OUT_CSV   = r'C:\Users\user\GitHub\model-predict-crime\data\raw\dallas_raw.csv'
PAGE_SIZE = 50000
BASE_URL  = 'https://www.dallasopendata.com/resource/qv6i-rri7.json'

VIOLENT  = {'ASSAULT','AGGRAVATED ASSAULT','ROBBERY','MURDER & NONNEGLIGENT MANSLAUGHTER',
            'RAPE','FORCIBLE RAPE','KIDNAPPING','HUMAN TRAFFICKING'}
PROPERTY = {'BURGLARY','THEFT','MOTOR VEHICLE THEFT','ARSON','FRAUD','FORGERY & COUNTERFEITING',
            'EMBEZZLEMENT','STOLEN PROPERTY','VANDALISM','CRIMINAL MISCHIEF'}

def map_cat(off):
    o = str(off).upper()
    for v in VIOLENT:
        if v in o: return 'violent'
    for p in PROPERTY:
        if p in o: return 'property'
    return 'other'

records = []
offset  = 0
print('Downloading Dallas crime data...')
while True:
    params = {
        '$limit':  PAGE_SIZE,
        '$offset': offset,
        '$where':  'y_cordinate IS NOT NULL AND x_coordinate IS NOT NULL',
        '$order':  'date1 ASC',
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
            lat = float(row.get('y_cordinate', 0) or 0)
            lon = float(row.get('x_coordinate', 0) or 0)
            if lat == 0 or lon == 0:
                continue
            dt = pd.to_datetime(row.get('date1', ''), errors='coerce')
            if pd.isna(dt):
                continue
            off = row.get('offincident', 'OTHER')
            records.append({
                'datetime':       dt,
                'hour':           dt.hour,
                'month':          dt.month,
                'weekday':        dt.weekday(),
                'latitude':       lat,
                'longitude':      lon,
                'crime_category': map_cat(off),
                'raw_offense':    off,
                'city':           'Dallas',
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
