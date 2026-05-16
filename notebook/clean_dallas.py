"""
Dallas Crime Data cleaner
Input:  data/raw/dallas_raw.csv  (from src/download_dallas.py)
Output: data/processed/dallas_clean.csv

Coordinates are already WGS84 (extracted from geocoded_column by download script).
"""
import os
import pandas as pd

BASE     = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE, '..', 'data', 'raw',       'dallas_raw.csv')
OUT_PATH = os.path.join(BASE, '..', 'data', 'processed', 'dallas_clean.csv')

LAT_MIN, LAT_MAX = 32.62, 33.02
LON_MIN, LON_MAX = -97.00, -96.55

def clean():
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f'Raw records: {len(df):,}')

    df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude', 'crime_category', 'datetime'])

    df = df[
        (df['latitude']  >= LAT_MIN) & (df['latitude']  <= LAT_MAX) &
        (df['longitude'] >= LON_MIN) & (df['longitude'] <= LON_MAX)
    ]
    print(f'After bbox filter: {len(df):,}')

    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])

    out = pd.DataFrame({
        'city':           'Dallas',
        'datetime':       df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
        'hour':           df['hour'].astype(int),
        'month':          df['month'].astype(int),
        'weekday':        df['weekday'].astype(int),
        'crime_category': df['crime_category'],
        'latitude':       df['latitude'].round(6),
        'longitude':      df['longitude'].round(6),
    })

    print(f'Clean records: {len(out):,}')
    print(f'Date range: {out["datetime"].min()} to {out["datetime"].max()}')
    print(out['crime_category'].value_counts())

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f'Saved → {OUT_PATH}')

if __name__ == '__main__':
    clean()
