"""
Clean Kansas City, MO crime data → data/processed/kansas_city_clean.csv
Source: data/raw/kansas_city_raw.csv (Socrata, years 2016-2018, 2022-2024)
Standard 74/13/13 temporal split (7 years of data)
"""
import pandas as pd
import numpy as np

RAW = r'C:\Users\user\GitHub\model-predict-crime\data\raw\kansas_city_raw.csv'
OUT = r'C:\Users\user\GitHub\model-predict-crime\data\processed\kansas_city_clean.csv'

# Bbox: Kansas City, MO (city proper)
LAT_MIN, LAT_MAX = 38.85, 39.35
LON_MIN, LON_MAX = -94.75, -94.25

def classify(offense):
    if not offense:
        return None
    o = str(offense).lower()
    if any(k in o for k in ['murder','rape','robbery','assault','kidnap','trafficking',
                             'molest','sodomy','sexual abuse','sexual misconduct','incest',
                             'stalking','terroristic','homicide','fondling','statutory rape',
                             'enticement','human trafficking','shooting - fatal']):
        return 'violent'
    if any(k in o for k in ['burglary','stealing','theft','arson','embezzle','forgery',
                             'fraud','identity theft','counterfeit','stolen','vandal',
                             'property damage','bad check','passing bad','tampering',
                             'extortion','receiving stolen','possession of stolen',
                             'possession of burglar','burning or exploding']):
        return 'property'
    if any(k in o for k in ['drug','narcotic','controlled substance','paraphernalia',
                             'marijuana','interdiction']):
        return 'drug'
    return 'public_order'

print('Reading raw data...')
df = pd.read_csv(RAW)
print(f'  Raw: {len(df):,} rows  years={sorted(df["year"].unique())}')

# Drop rows without coordinates
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
df = df.dropna(subset=['lat', 'lon'])

# Bbox filter
df = df[(df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) &
        (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX)]
print(f'  After bbox: {len(df):,}')

# Parse datetime: 'date' field format varies (ISO string) + 'time' HH:MM
def parse_dt(row):
    date_str = str(row.get('date', ''))
    time_str = str(row.get('time', ''))
    if not date_str or date_str == 'nan':
        return pd.NaT
    try:
        dt = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(dt):
            return pd.NaT
        # Apply time component if available and date has no time
        if time_str and time_str != 'nan' and ':' in time_str:
            try:
                h, m = int(time_str[:2]), int(time_str[3:5])
                dt = dt.replace(hour=h, minute=m)
            except (ValueError, IndexError):
                pass
        return dt
    except Exception:
        return pd.NaT

print('Parsing datetimes...')
df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
# Apply time field for hour precision
def apply_time(row):
    dt = row['datetime']
    t  = str(row.get('time', ''))
    if pd.isna(dt) or not t or t == 'nan' or ':' not in t:
        return dt
    try:
        h, m = int(t[:2]), int(t[3:5])
        return dt.replace(hour=h, minute=m)
    except Exception:
        return dt

df['datetime'] = df.apply(apply_time, axis=1)
df = df.dropna(subset=['datetime'])

df['year']  = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['hour']  = df['datetime'].dt.hour

def time_slot(h):
    if h < 6:  return 0
    if h < 12: return 1
    if h < 18: return 2
    return 3

df['time_slot'] = df['hour'].apply(time_slot)

# Crime category
df['crime_type4'] = df['offense'].apply(classify)
df = df.dropna(subset=['crime_type4'])

MERGE_MAP = {'drug': 'other', 'public_order': 'other'}
df['crime_type'] = df['crime_type4'].map(lambda x: MERGE_MAP.get(x, x))

print('\nCrime type distribution:')
print(df['crime_type4'].value_counts())
print('\nAfter merge:')
print(df['crime_type'].value_counts())

# Standardise output
out = pd.DataFrame({
    'city':       'Kansas City',
    'lat':        df['lat'].round(5),
    'lon':        df['lon'].round(5),
    'datetime':   df['datetime'],
    'year':       df['year'],
    'month':      df['month'],
    'hour':       df['hour'],
    'time_slot':  df['time_slot'],
    'crime_type': df['crime_type'],
})

out.to_csv(OUT, index=False)
print(f'\nSaved: {len(out):,} records → {OUT}')
print(out['year'].value_counts().sort_index())
