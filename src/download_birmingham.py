"""Download Birmingham AL crime data from CKAN open data portal (2023-2026, 4 precincts)."""
import requests
import pandas as pd
import io
import os
import time

OUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'birmingham_raw.csv'))

CKAN_BASE = 'https://data.birminghamal.gov/api/3/action'
PRECINCTS = {
    'East':  'east-precinct-crime-data',
    'North': 'north-precinct-crime-data',
    'South': 'south-precinct-crime-data',
    'West':  'west-precinct-crime-data',
}
# Only years 2023+ have lat/lon columns
YEARS_WITH_LATLON = {'2023', '2024', '2025', '2026'}


def get_csv_resources() -> list[dict]:
    """Return list of {precinct, year, url} for CSV resources with lat/lon."""
    resources = []
    for precinct, pkg_id in PRECINCTS.items():
        r = requests.get(f'{CKAN_BASE}/package_show?id={pkg_id}', timeout=15)
        r.raise_for_status()
        for res in r.json().get('result', {}).get('resources', []):
            if res.get('format') != 'CSV':
                continue
            name = res.get('name', '')
            year = next((y for y in YEARS_WITH_LATLON if y in name), None)
            if year is None:
                continue
            resources.append({'precinct': precinct, 'year': year, 'url': res['url']})
    return resources


def download_csv(url: str) -> pd.DataFrame | None:
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content))
            df.columns = df.columns.str.strip().str.lstrip('﻿')
            return df
        except Exception as e:
            print(f'    attempt={attempt+1} error: {e}')
            time.sleep(2 ** attempt)
    return None


def main():
    print('[Birmingham] Discovering CSV resources...')
    resources = get_csv_resources()
    print(f'  Found {len(resources)} CSV files with lat/lon (2023-2026)')

    frames = []
    for res in sorted(resources, key=lambda x: (x['precinct'], x['year'])):
        precinct, year, url = res['precinct'], res['year'], res['url']
        print(f'  Downloading {precinct} {year}...', end=' ', flush=True)
        df = download_csv(url)
        if df is None:
            print('FAILED')
            continue
        # Standardise columns
        lat_col = next((c for c in df.columns if 'Latitude' in c), None)
        lon_col = next((c for c in df.columns if 'Longitude' in c), None)
        date_col = next((c for c in df.columns if 'Occurred' in c), None)
        off_col  = next((c for c in df.columns if 'Offense' in c and 'Description' in c), None)
        if not all([lat_col, lon_col, date_col, off_col]):
            print(f'SKIP (missing cols: {list(df.columns)})')
            continue
        df = df[[lat_col, lon_col, date_col, off_col]].copy()
        df.columns = ['latitude', 'longitude', 'datetime', 'crime_type']
        df['precinct'] = precinct
        frames.append(df)
        print(f'{len(df):,} rows')
        time.sleep(0.3)

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f'\n[Birmingham] Done: {len(combined):,} records -> {OUT_PATH}')


if __name__ == '__main__':
    main()
