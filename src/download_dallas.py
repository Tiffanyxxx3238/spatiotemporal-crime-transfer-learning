"""
Download Dallas, TX crime data from Dallas Open Data (Socrata)
Dataset: Police Incidents
Portal:  www.dallasopendata.com  resource id: qv6i-rri7
Fields:  date1, time1, geocoded_column (WGS84 lat/lon), offincident
"""
import urllib.request, urllib.parse, json, time, os
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

PROGRESS_FILE = OUT_CSV + '.offset'

def fetch_page(offset, retries=5):
    params = {
        '$limit':  PAGE_SIZE,
        '$offset': offset,
        '$where':  'geocoded_column IS NOT NULL',
        '$order':  'date1 ASC',
    }
    url = BASE_URL + '?' + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f'  Retry {attempt+1}/{retries} (offset={offset:,}): {e} — waiting {wait}s')
            time.sleep(wait)
    raise RuntimeError(f'Failed after {retries} retries at offset {offset}')

# Resume from last saved offset if available
start_offset = 0
write_mode   = 'w'
header       = True
if os.path.exists(PROGRESS_FILE) and os.path.exists(OUT_CSV):
    with open(PROGRESS_FILE) as f:
        start_offset = int(f.read().strip())
    write_mode = 'a'
    header     = False
    print(f'Resuming from offset {start_offset:,}')

offset = start_offset
total  = 0
print('Downloading Dallas crime data...')
while True:
    batch = fetch_page(offset)
    if not batch:
        break

    rows = []
    for row in batch:
        try:
            geo = row.get('geocoded_column', {}) or {}
            lat = float(geo.get('latitude', 0) or 0)
            lon = float(geo.get('longitude', 0) or 0)
            if lat == 0 or lon == 0:
                continue
            dt = pd.to_datetime(row.get('date1', ''), errors='coerce')
            if pd.isna(dt):
                continue
            t1 = row.get('time1', '')
            try:
                hour = int(t1.split(':')[0]) if t1 else 0
            except Exception:
                hour = 0
            off = row.get('offincident', 'OTHER')
            rows.append({
                'datetime':       dt.strftime('%Y-%m-%d') + (f' {t1}:00' if t1 else ' 00:00:00'),
                'hour':           hour,
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

    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, mode=write_mode, header=header, index=False)
        write_mode = 'a'
        header     = False

    offset += len(batch)
    total  += len(rows)
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(offset))
    print(f'  offset={offset:,}  saved={total:,}')

    if len(batch) < PAGE_SIZE:
        break
    time.sleep(0.5)

# Clean up progress file on success
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)

df = pd.read_csv(OUT_CSV)
print(f'\nDone. {len(df):,} rows → {OUT_CSV}')
print(df['crime_category'].value_counts())
