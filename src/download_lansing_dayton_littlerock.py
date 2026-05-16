"""
Download Lansing MI / Dayton OH / Little Rock AR crime data from ArcGIS FeatureServer.

Run individual city by passing --city flag:
  python download_lansing_dayton_littlerock.py --city lansing
  python download_lansing_dayton_littlerock.py --city dayton
  python download_lansing_dayton_littlerock.py --city littlerock

ArcGIS endpoints (verify if not working — city portals occasionally change):
  Lansing:    https://maps.lansingmi.gov/arcgis/rest/services/OpenData/CrimeData/FeatureServer/0
  Dayton:     https://gis.daytonohio.gov/arcgis/rest/services/OpenData/Police_Incidents/FeatureServer/0
  Little Rock: https://maps.littlerock.gov/arcgis/rest/services/OpenData/Crime/FeatureServer/0
              (alt: search LittleRock on hub.arcgis.com)
"""
import urllib.request, urllib.parse, json, time, argparse
import pandas as pd

BASE = r'C:\Users\user\GitHub\model-predict-crime\data\raw'

CITIES = {
    'lansing': {
        'url':      'https://maps.lansingmi.gov/arcgis/rest/services/OpenData/CrimeData/FeatureServer/0',
        'out':      f'{BASE}/lansing_raw.csv',
        'date_fld': 'CrimeDate',        # adjust if different
        'cat_fld':  'CrimeType',
        'lat_fld':  None,               # use geometry
        'lon_fld':  None,
        'city_name':'Lansing',
    },
    'dayton': {
        'url':      'https://gis.daytonohio.gov/arcgis/rest/services/OpenData/Police_Incidents/FeatureServer/0',
        'out':      f'{BASE}/dayton_raw.csv',
        'date_fld': 'INCIDENT_DATE',
        'cat_fld':  'OFFENSE_CATEGORY',
        'lat_fld':  None,
        'lon_fld':  None,
        'city_name':'Dayton',
    },
    'littlerock': {
        'url':      'https://maps.littlerock.gov/arcgis/rest/services/OpenData/Crime/FeatureServer/0',
        'out':      f'{BASE}/little_rock_raw.csv',
        'date_fld': 'DATE_OCCUR',
        'cat_fld':  'OFFENSE',
        'lat_fld':  None,
        'lon_fld':  None,
        'city_name':'Little Rock',
    },
}

VIOLENT_KEYWORDS  = ['assault','homicide','murder','robbery','rape','kidnap','sex offense','human traf']
PROPERTY_KEYWORDS = ['theft','burglary','larceny','vehicle','arson','fraud','vandal','stolen','embez']

def map_cat(text):
    t = str(text).lower()
    for kw in VIOLENT_KEYWORDS:
        if kw in t: return 'violent'
    for kw in PROPERTY_KEYWORDS:
        if kw in t: return 'property'
    return 'other'

def fetch_arcgis(base_url, date_fld, cat_fld, city_name, page_size=2000):
    query_url = base_url + '/query'
    records = []
    offset  = 0
    print(f'Downloading {city_name}...')
    while True:
        params = {
            'where':         '1=1',
            'outFields':     '*',
            'returnGeometry':'true',
            'f':             'json',
            'resultOffset':  offset,
            'resultRecordCount': page_size,
        }
        url = query_url + '?' + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.loads(r.read().decode())
        except Exception as e:
            print(f'  Error at offset {offset}: {e}')
            break
        features = data.get('features', [])
        if not features:
            break
        for feat in features:
            attrs = feat.get('attributes', {})
            geom  = feat.get('geometry', {})
            try:
                # Geometry: ArcGIS typically returns {x, y} in WGS84 when outSR=4326
                # If not WGS84, you'll see large numbers (state plane coords) — needs reprojection
                lon = geom.get('x')
                lat = geom.get('y')
                if lon is None or lat is None:
                    continue
                lat, lon = float(lat), float(lon)
                if abs(lat) > 90 or abs(lon) > 180:
                    continue  # non-WGS84 — skip (need reprojection)

                raw_date = attrs.get(date_fld)
                if raw_date is None:
                    continue
                # ArcGIS timestamps are epoch ms
                if isinstance(raw_date, (int, float)):
                    dt = pd.to_datetime(raw_date, unit='ms', errors='coerce')
                else:
                    dt = pd.to_datetime(str(raw_date), errors='coerce')
                if pd.isna(dt):
                    continue

                cat_raw = attrs.get(cat_fld, 'OTHER')
                records.append({
                    'datetime':       dt,
                    'hour':           dt.hour,
                    'month':          dt.month,
                    'weekday':        dt.weekday(),
                    'latitude':       lat,
                    'longitude':      lon,
                    'crime_category': map_cat(str(cat_raw)),
                    'raw_offense':    cat_raw,
                    'city':           city_name,
                })
            except Exception:
                continue
        offset += len(features)
        print(f'  offset={offset:,}  records={len(records):,}')
        if not data.get('exceededTransferLimit', False) and len(features) < page_size:
            break
        time.sleep(0.2)
    return records

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', required=True, choices=list(CITIES.keys()))
    args = parser.parse_args()

    cfg = CITIES[args.city]
    rows = fetch_arcgis(cfg['url'], cfg['date_fld'], cfg['cat_fld'], cfg['city_name'])

    if not rows:
        print('No records downloaded — check the ArcGIS URL and field names.')
    else:
        df = pd.DataFrame(rows)
        df.to_csv(cfg['out'], index=False)
        print(f'Saved {len(df):,} rows → {cfg["out"]}')
        print(df['crime_category'].value_counts())
