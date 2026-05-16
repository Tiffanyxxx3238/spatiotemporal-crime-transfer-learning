"""
Download San Francisco, CA crime data from SF Open Data (Socrata)
Dataset: Police Department Incident Reports (2018–Present)
Portal:  data.sfgov.org  resource id: wg3w-h783
Fields:  incident_datetime, latitude, longitude, incident_category
Historical (2003-2017): resource id = tmnf-yvry (different schema)
"""
import urllib.request, urllib.parse, json, time
import pandas as pd

OUT_CSV   = r'C:\Users\user\GitHub\model-predict-crime\data\raw\sf_raw.csv'
PAGE_SIZE = 50000

DATASETS = {
    'new': {   # 2018–present
        'url':       'https://data.sfgov.org/resource/wg3w-h783.json',
        'date_fld':  'incident_datetime',
        'lat_fld':   'latitude',
        'lon_fld':   'longitude',
        'cat_fld':   'incident_category',
    },
    'old': {   # 2003–2017
        'url':       'https://data.sfgov.org/resource/tmnf-yvry.json',
        'date_fld':  'date',
        'lat_fld':   'y',
        'lon_fld':   'x',
        'cat_fld':   'category',
    },
}

VIOLENT  = {'ASSAULT','ROBBERY','RAPE','HOMICIDE','KIDNAPPING','SEX OFFENSES FORCIBLE',
            'HUMAN TRAFFICKING (A), COMMERCIAL SEX ACTS',
            'HUMAN TRAFFICKING, COMMERCIAL SEX ACTS','WEAPONS OFFENSE','WEAPONS OFFENSES'}
PROPERTY = {'LARCENY THEFT','BURGLARY','MOTOR VEHICLE THEFT','ARSON','FRAUD',
            'EMBEZZLEMENT','STOLEN PROPERTY','VANDALISM','EXTORTION',
            'VEHICLE BREAK-IN/THEFT','MALICIOUS MISCHIEF'}

def map_cat(cat):
    c = str(cat).upper()
    if c in VIOLENT:  return 'violent'
    if c in PROPERTY: return 'property'
    return 'other'

all_records = []
for name, cfg in DATASETS.items():
    url_base = cfg['url']
    offset = 0
    print(f'Downloading SF ({name})...')
    while True:
        params = {
            '$limit':  PAGE_SIZE,
            '$offset': offset,
            '$where':  f"{cfg['lat_fld']} IS NOT NULL",
            '$order':  f"{cfg['date_fld']} ASC",
        }
        url = url_base + '?' + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                batch = json.loads(r.read().decode())
        except Exception as e:
            print(f'  Error at offset {offset}: {e}'); break
        if not batch:
            break
        for row in batch:
            try:
                lat = float(row.get(cfg['lat_fld'], 0) or 0)
                lon = float(row.get(cfg['lon_fld'], 0) or 0)
                if lat == 0 or lon == 0:
                    continue
                dt = pd.to_datetime(row.get(cfg['date_fld'], ''), errors='coerce')
                if pd.isna(dt):
                    continue
                cat = row.get(cfg['cat_fld'], 'OTHER')
                all_records.append({
                    'datetime':       dt,
                    'hour':           dt.hour,
                    'month':          dt.month,
                    'weekday':        dt.weekday(),
                    'latitude':       lat,
                    'longitude':      lon,
                    'crime_category': map_cat(cat),
                    'raw_offense':    cat,
                    'city':           'San Francisco',
                })
            except Exception:
                continue
        offset += len(batch)
        print(f'  offset={offset:,}  total={len(all_records):,}')
        if len(batch) < PAGE_SIZE:
            break
        time.sleep(0.3)

df = pd.DataFrame(all_records)
df.to_csv(OUT_CSV, index=False)
print(f'Saved {len(df):,} rows → {OUT_CSV}')
print(df['crime_category'].value_counts())
