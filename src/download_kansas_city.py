"""
Download Kansas City, MO crime data from Socrata (KCMO Open Data)
Years with lat/lon: 2016, 2018, 2022-2024 (2017 has no coordinates)
Location schemas:
  2016:       latitude / longitude string attributes directly on row
  2018:       location dict {'latitude': str, 'longitude': str, 'human_address': ...}
  2022-2024:  location GeoJSON {'type':'Point','coordinates':[lon,lat]}
Offense field: 'description' (text) for all years; numeric 'offense' code ignored
"""
import urllib.request, urllib.parse, json, time
import pandas as pd

OUT_CSV = r'C:\Users\user\GitHub\model-predict-crime\data\raw\kansas_city_raw.csv'
PAGE_SIZE = 50000

# 2017 excluded: location_1 has only human_address, no lat/lon
DATASETS = {
    '2016': {'id': 'wbz8-pdv7', 'loc_type': 'latlon',  'date_field': 'reported_date', 'time_field': 'reported_time'},
    '2018': {'id': 'dmjw-d28i', 'loc_type': 'loc1',    'date_field': 'reported_date', 'time_field': 'reported_time'},
    '2022': {'id': 'x39y-7d3m', 'loc_type': 'geojson', 'date_field': 'report_date',   'time_field': 'report_time'},
    '2023': {'id': 'bfyq-5nh6', 'loc_type': 'geojson', 'date_field': 'report_date',   'time_field': 'reported_time'},
    '2024': {'id': 'isbe-v4d8', 'loc_type': 'geojson', 'date_field': 'reported_date', 'time_field': 'reported_time'},
}

def get_loc(row, loc_type):
    if loc_type == 'latlon':
        lat = row.get('latitude')
        lon = row.get('longitude')
        try:
            lat, lon = float(lat), float(lon)
            return (lat, lon) if lat != 0.0 and lon != 0.0 else (None, None)
        except (TypeError, ValueError):
            return (None, None)
    elif loc_type == 'loc1':
        loc = row.get('location', {}) or {}
        lat = loc.get('latitude')
        lon = loc.get('longitude')
        try:
            lat, lon = float(lat), float(lon)
            return (lat, lon) if lat != 0.0 and lon != 0.0 else (None, None)
        except (TypeError, ValueError):
            return (None, None)
    elif loc_type == 'geojson':
        loc = row.get('location', {}) or {}
        coords = loc.get('coordinates', [])
        return (coords[1], coords[0]) if len(coords) == 2 else (None, None)
    return (None, None)

all_records = []
for year, cfg in sorted(DATASETS.items()):
    dsid      = cfg['id']
    loc_type  = cfg['loc_type']
    date_fld  = cfg['date_field']
    time_fld  = cfg['time_field']
    base_url  = f'https://data.kcmo.org/resource/{dsid}.json'
    offset = 0
    year_records = []
    while True:
        params = {
            '$limit': PAGE_SIZE,
            '$offset': offset,
            '$order': ':id',
        }
        url = base_url + '?' + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as r:
            rows = json.load(r)
        if not rows:
            break
        for row in rows:
            lat, lon = get_loc(row, loc_type)
            # Use 'description' (text) field; fall back to 'offense' for newer years
            offense = row.get('description') or row.get('offense', '')
            year_records.append({
                'year':    year,
                'offense': offense,
                'date':    row.get(date_fld, ''),
                'time':    row.get(time_fld, ''),
                'lat':     lat,
                'lon':     lon,
            })
        offset += len(rows)
        print(f'  {year}: {offset:,}...', end='\r')
        if len(rows) < PAGE_SIZE:
            break
        time.sleep(0.2)
    print(f'  {year}: {len(year_records):,} records')
    all_records.extend(year_records)

df = pd.DataFrame(all_records)
df.to_csv(OUT_CSV, index=False)
print(f'\nTotal: {len(df):,} records → {OUT_CSV}')
print(df.groupby('year').size())
print(f'Null lat: {df["lat"].isna().sum():,}')
