"""Download SLC Police crime data (Jan 2020 – Jun 2021) via ArcGIS Feature Service."""
import requests
import pandas as pd
import time
import os

OUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'slc_raw.csv'))

SERVICE_URL = (
    'https://services.arcgis.com/mMBpeYj0vPFotzbe/arcgis/rest/services'
    '/SLCPD_Crime_Data_1Jan2020_30Jun2021/FeatureServer/0/query'
)

PAGE_SIZE = 2000

FIELDS = 'Occ_Date,Occ_Time,Rucr_Ext_D,Location'


def fetch_page(offset: int) -> list:
    params = {
        'where': '1=1',
        'outFields': FIELDS,
        'resultRecordCount': PAGE_SIZE,
        'resultOffset': offset,
        'outSR': '4326',
        'f': 'json',
    }
    for attempt in range(4):
        try:
            r = requests.get(SERVICE_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if 'error' in data:
                raise ValueError(data['error'])
            return data.get('features', [])
        except Exception as e:
            print(f'  offset={offset} attempt={attempt+1} error: {e}')
            time.sleep(2 ** attempt)
    return []


def main():
    print('[SLC] Downloading crime data via ArcGIS Feature Service...')
    all_rows = []
    offset = 0
    while True:
        features = fetch_page(offset)
        if not features:
            break
        for feat in features:
            attr = feat['attributes']
            geom = feat.get('geometry', {})
            lon = geom.get('x')
            lat = geom.get('y')
            if lat is None or lon is None:
                continue
            occ_ms = attr.get('Occ_Date')
            if occ_ms is None:
                continue
            dt = pd.to_datetime(occ_ms, unit='ms', utc=True).tz_convert(None)
            occ_time = attr.get('Occ_Time', 0) or 0
            hour = int(occ_time) // 100
            dt = dt.replace(hour=min(hour, 23))
            all_rows.append({
                'latitude':  lat,
                'longitude': lon,
                'datetime':  dt,
                'crime_type': str(attr.get('Rucr_Ext_D', '')).strip(),
                'location':  str(attr.get('Location', '')).strip(),
            })
        print(f'  offset={offset}: fetched {len(features)}, total so far={len(all_rows)}')
        offset += PAGE_SIZE
        if len(features) < PAGE_SIZE:
            break
        time.sleep(0.2)

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f'\n[SLC] Done: {len(df):,} records -> {OUT_PATH}')


if __name__ == '__main__':
    main()
