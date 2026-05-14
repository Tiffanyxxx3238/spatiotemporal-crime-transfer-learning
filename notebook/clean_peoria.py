"""
Clean Peoria, IL crime data → data/processed/peoria_clean.csv
Source: data/raw/peoria_raw.csv (ArcGIS FeatureServer)
Date range: 2023-2026 (~3 years) → use 60/20/20 split in classification
"""
import pandas as pd
import numpy as np

RAW  = r'C:\Users\user\GitHub\model-predict-crime\data\raw\peoria_raw.csv'
OUT  = r'C:\Users\user\GitHub\model-predict-crime\data\processed\peoria_clean.csv'

# Bbox: Peoria, IL
LAT_MIN, LAT_MAX = 40.55, 40.85
LON_MIN, LON_MAX = -89.80, -89.45

CRIME_MAP = {
    'AGGRAVATED ASSAULT':                      'violent',
    'SIMPLE ASSAULT':                          'violent',
    'RAPE':                                    'violent',
    'STATUTORY RAPE':                          'violent',
    'FONDLING':                                'violent',
    'INTIMIDATION':                            'violent',
    'WEAPON LAW VIOLATIONS':                   'violent',
    'KIDNAPING/ABDUCTION':                     'violent',
    'HUMAN TRAFFICKING, COMMERCIAL SEX ACTS':  'violent',
    'MURDER AND NONNEGLIGENT MANSLAUGHTER':    'violent',
    'ROBBERY':                                 'violent',
    'BURGLARY/BREAKING AND ENTERING':          'property',
    'ALL OTHER LARCENY':                       'property',
    'THEFT FROM MOTOR VEHICLE':                'property',
    'MOTOR VEHICLE THEFT':                     'property',
    'ARSON':                                   'property',
    'FALSE PRETENSES/SWINDLE/CONFIDENCE GAME': 'property',
    'DESTRUCTION/DAMAGE/VANDALISM OF PROPERTY':'property',
    'EMBEZZLEMENT':                            'property',
    'SHOPLIFTING':                             'property',
    'THEFT FROM BUILDING':                     'property',
    'POCKET-PICKING':                          'property',
    'STOLEN PROPERTY OFFENSES':                'property',
    'DRUG/NARCOTIC VIOLATIONS':                'drug',
    'DRUG/NARCOTIC VIOLATION':                 'drug',
    'DRUG EQUIPMENT VIOLATIONS':               'drug',
    'DRUG EQUIPMENT VIOLATION':                'drug',
    'DISORDERLY CONDUCT':                      'public_order',
    'ALL OTHER OFFENSES':                      'public_order',
    'TRESPASS OF REAL PROPERTY':               'public_order',
    'RUNAWAY':                                 'public_order',
    'LIQUOR LAW VIOLATIONS':                   'public_order',
    'PORNOGRAPHY/OBSCENE MATERIAL':            'public_order',
    'PEEPING TOM':                             'public_order',
    'CURFEW/LOITERING/VAGRANCY VIOLATIONS':    'public_order',
}

def classify(desc):
    if not desc:
        return None
    desc = str(desc).upper().strip()
    if desc in CRIME_MAP:
        return CRIME_MAP[desc]
    if any(k in desc for k in ['ASSAULT','ROBBERY','MURDER','RAPE','FONDLING','KIDNAP','TRAFFICKING','WEAPON','HOMICIDE','SEX OFFENSE']):
        return 'violent'
    if any(k in desc for k in ['BURGLARY','LARCENY','THEFT','ARSON','EMBEZZLE','FRAUD','FORGERY','STOLEN','VANDAL','PROPERTY DAMAGE']):
        return 'property'
    if any(k in desc for k in ['DRUG','NARCOTIC','CONTROLLED SUBSTANCE','PARAPHERNALIA']):
        return 'drug'
    return 'public_order'

print('Reading raw data...')
df = pd.read_csv(RAW)
print(f'  Raw: {len(df):,} rows')

# Drop rows without coordinates
df = df.dropna(subset=['lat', 'lon'])
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
df = df.dropna(subset=['lat', 'lon'])

# Bbox filter
df = df[(df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) &
        (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX)]
print(f'  After bbox: {len(df):,}')

# Parse datetime: reportdate is ms epoch (midnight) + reporthour for hour
df['reportdate'] = pd.to_numeric(df['reportdate'], errors='coerce')
df['reporthour'] = pd.to_numeric(df['reporthour'], errors='coerce').fillna(0).astype(int)
df = df.dropna(subset=['reportdate'])
df['datetime'] = pd.to_datetime(df['reportdate'], unit='ms', utc=True).dt.tz_localize(None)
df['datetime'] = df['datetime'].dt.normalize() + pd.to_timedelta(df['reporthour'], unit='h')
df['year']  = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['hour']  = df['reporthour']

# Time slot: 0=深夜0-5, 1=早晨6-11, 2=下午12-17, 3=夜晚18-23
def time_slot(h):
    if h < 6:  return 0
    if h < 12: return 1
    if h < 18: return 2
    return 3

df['time_slot'] = df['hour'].apply(time_slot)

# Crime category
df['crime_type4'] = df['nibrsdesc'].apply(classify)
df = df.dropna(subset=['crime_type4'])

MERGE_MAP = {'drug': 'other', 'public_order': 'other'}
df['crime_type'] = df['crime_type4'].map(lambda x: MERGE_MAP.get(x, x))

print('\nCrime type distribution:')
print(df['crime_type4'].value_counts())
print('\nAfter merge:')
print(df['crime_type'].value_counts())

# Standardise output
out = pd.DataFrame({
    'city':       'Peoria',
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
print(out.dtypes)
print(out.head(3))
