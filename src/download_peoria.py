"""
Download Peoria, IL crime data from ArcGIS FeatureServer
Service: https://police-transparency-1-peoria-il.hub.arcgis.com/datasets/crimes-public
Fields: nibrsdesc, nibrsgroup, reportdate, reportyear, reporthour
Geometry: WGS84 (outSR=4326), geometry.x=lon, geometry.y=lat
Date range: 2023-2026 (~69k records)
"""
import urllib.request, urllib.parse, json, time
import pandas as pd

BASE_URL = (
    'https://services1.arcgis.com/Vm4J3EDyqMzmDYgP/arcgis/rest/services'
    '/Crimes_public_b259ad13665440579e8fa083818cdd9f/FeatureServer/0/query'
)
OUT_CSV = r'C:\Users\user\GitHub\model-predict-crime\data\raw\peoria_raw.csv'
PAGE_SIZE = 2000

params_base = {
    'where': '1=1',
    'outFields': 'nibrsdesc,nibrsgroup,reportdate,reportyear,reporthour',
    'returnGeometry': 'true',
    'outSR': '4326',
    'resultRecordCount': PAGE_SIZE,
    'f': 'json',
}

records = []
offset = 0
while True:
    params = {**params_base, 'resultOffset': offset}
    url = BASE_URL + '?' + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    features = data.get('features', [])
    if not features:
        break
    for feat in features:
        attr = feat['attributes']
        geom = feat.get('geometry') or {}
        records.append({
            'lon':        geom.get('x'),
            'lat':        geom.get('y'),
            'nibrsdesc':  attr.get('nibrsdesc'),
            'nibrsgroup': attr.get('nibrsgroup'),
            'reportdate': attr.get('reportdate'),
            'reportyear': attr.get('reportyear'),
            'reporthour': attr.get('reporthour'),
        })
    offset += len(features)
    print(f'  Downloaded {offset:,} records...', end='\r')
    if not data.get('exceededTransferLimit', False) and len(features) < PAGE_SIZE:
        break
    time.sleep(0.3)

df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
print(f'\nDone: {len(df):,} records → {OUT_CSV}')
print(df.dtypes)
print(df[['lat','lon','nibrsdesc','reportyear']].head(3))
