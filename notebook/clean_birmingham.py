"""Clean raw Birmingham AL crime data → data/processed/birmingham_clean.csv"""
import os
import re
import pandas as pd

RAW  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw',  'birmingham_raw.csv'))
PROC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'birmingham_clean.csv'))

# Bounding box: Birmingham, Alabama
LAT_MIN, LAT_MAX = 33.40, 33.70
LON_MIN, LON_MAX = -86.95, -86.60

# Keyword-based mapping: priority order violent > drug > property > public_order > other
VIOLENT_KW = [
    'murder', 'homicide', 'manslaughter', 'assault', 'battery', 'robbery',
    'rape', 'sexual', 'sex offense', 'kidnap', 'abduct', 'carjack',
    'strangul', 'suffoc', 'attempt to commit', 'disch firearm',
    'discharge', 'shooting', 'shot', 'aggravated',
]
DRUG_KW = [
    'drug', 'narcotic', 'controlled substance', 'marijuana', 'cocaine',
    'methamphetamine', 'heroin', 'opiate', 'cannabis',
]
PROPERTY_KW = [
    'theft', 'burglary', 'larceny', 'shoplifting', 'stolen', 'steal',
    'fraud', 'forgery', 'arson', 'damage to property', 'vandal',
    'embezzle', 'extort', 'counterfeit', 'receiving stolen',
    'criminal damage', 'break', 'vehicle parts', 'auto theft', 'deception',
]
PUBLIC_ORDER_KW = [
    'dui', 'dwi', 'driving under', 'traffic', 'hit and run',
    'disorderly', 'trespass', 'public intox', 'liquor', 'prostitut',
    'resist', 'obstruct', 'escape', 'contempt', 'public order',
    'weapon possession', 'firearm possession', 'carry',
]


def classify(desc: str) -> str:
    d = desc.lower()
    if any(k in d for k in VIOLENT_KW):
        return 'violent'
    if any(k in d for k in DRUG_KW):
        return 'drug'
    if any(k in d for k in PROPERTY_KW):
        return 'property'
    if any(k in d for k in PUBLIC_ORDER_KW):
        return 'public_order'
    return 'other'


def main():
    print(f'Reading: {RAW}')
    df = pd.read_csv(RAW, low_memory=False)
    print(f'Raw records: {len(df):,}')

    # Bbox filter
    df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df[(df['latitude']  >= LAT_MIN) & (df['latitude']  <= LAT_MAX) &
            (df['longitude'] >= LON_MIN) & (df['longitude'] <= LON_MAX)]
    print(f'After bbox filter: {len(df):,}')

    # Parse datetime (format: "12/31/2024 23:50")
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', format='mixed', dayfirst=False)
    df = df.dropna(subset=['datetime'])

    # Map crime category
    df['crime_category'] = df['crime_type'].fillna('').apply(classify)

    df['city'] = 'Birmingham'
    out = df[['city', 'latitude', 'longitude', 'datetime', 'crime_category']].copy()
    out = out.dropna(subset=['latitude', 'longitude'])
    print(f'Clean records: {len(out):,}')
    print(f'Date range: {out["datetime"].min()} to {out["datetime"].max()}')
    print('\ncrime_category:')
    print(out['crime_category'].value_counts().to_string())

    os.makedirs(os.path.dirname(PROC), exist_ok=True)
    out.to_csv(PROC, index=False)
    print(f'\nSaved to: {PROC}')


if __name__ == '__main__':
    main()
