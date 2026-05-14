"""Clean raw SLC crime data → data/processed/slc_clean.csv"""
import os
import pandas as pd
import numpy as np

RAW  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw',  'slc_raw.csv'))
PROC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'slc_clean.csv'))

# Bounding box: Salt Lake City, UT
LAT_MIN, LAT_MAX = 40.68, 40.83
LON_MIN, LON_MAX = -112.10, -111.75

CRIME_MAP = {
    'ASSAULT':        'violent',
    'ROBBERY':        'violent',
    'SEXUAL ASSAULT': 'violent',
    'SEX OFFENSES':   'violent',
    'SEXUAL OFFENSE': 'violent',
    'SEX ASSAULT':    'violent',
    'WEAPON OFFENSE': 'violent',
    'KIDNAP':         'violent',
    'ENTICEMENT':     'violent',
    'HOMICIDE':       'violent',
    'WEAPONS':        'violent',
    'THREATS':        'violent',
    'EXPLOITATION':   'violent',
    'EXPL-HUMAN TRF': 'violent',

    'EMBEZZLEMENT':   'property',
    'PROPERTY CRIME': 'property',

    'DUI - FEL3':     'public_order',
    'DUI - CLS A':    'public_order',
    'DUI < .05 ALC':  'public_order',
    'OBST JUDICIAL':  'public_order',
    'GAMBLING':       'public_order',
    'CIVIL RIGHTS':   'public_order',

    'BURGLARY':       'property',
    'LARCENY':        'property',
    'STOLEN VEHICLE': 'property',
    'STOLEN PROP':    'property',
    'DAMAGED PROP':   'property',
    'COUNTERFEITING': 'property',
    'FORGERY':        'property',
    'FRAUD':          'property',
    'HIT AND RUN':    'property',
    'ARSON':          'property',
    'EXTORTION':      'property',
    'ARPRT TITLE 16': 'property',

    'DRUGS':          'drug',
    'DUI DRUGS':      'drug',

    'DUI ALCOHOL':    'public_order',
    'LIQUOR':         'public_order',
    'ALCOHOL IN VEH': 'public_order',
    'PUBLIC ORDER':   'public_order',
    'PUBLIC PEACE':   'public_order',
    'OBST POLICE':    'public_order',
    'ESCAPE':         'public_order',
    'FAMILY OFFENSES':'public_order',
    'FLEEING':        'public_order',
    'MORALS-DECENCY': 'public_order',
    'COMMERCIAL SEX': 'public_order',
    'PORNOGRAPHY':    'public_order',

    'TRAFFIC':        'other',
    'MOV TRAF VIOL':  'other',
    'NON MOV TRAF':   'other',
    'CONSERVATION':   'other',
    'IMP/ABAND VEH':  'other',
    'INV OF PRIVACY': 'other',
    'NONREPTABL TA':  'other',
    'REPORTABLE TA':  'other',
    'JUVENILE OFF':   'other',
    'TA- CITY EQUIP': 'other',
    'OFCR INV TA':    'other',
    'HEALTH/SAFETY':  'other',
}

def main():
    print(f'Reading: {RAW}')
    df = pd.read_csv(RAW, low_memory=False)
    print(f'Raw records: {len(df):,}')

    # Bbox filter
    df = df[(df['latitude']  >= LAT_MIN) & (df['latitude']  <= LAT_MAX) &
            (df['longitude'] >= LON_MIN) & (df['longitude'] <= LON_MAX)]
    print(f'After bbox filter: {len(df):,}')

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])

    # Map crime category
    df['crime_category'] = df['crime_type'].str.upper().str.strip().map(CRIME_MAP)
    unmapped = df[df['crime_category'].isna()]['crime_type'].value_counts()
    if len(unmapped):
        print('Unmapped crime types (→ other):')
        print(unmapped.head(20).to_string())
    df['crime_category'] = df['crime_category'].fillna('other')

    # Rename/select columns to standard format
    df = df.rename(columns={'latitude': 'latitude', 'longitude': 'longitude'})
    df['city'] = 'Salt Lake City'

    out = df[['city', 'latitude', 'longitude', 'datetime', 'crime_category']].copy()
    out = out.dropna(subset=['latitude', 'longitude', 'crime_category'])
    print(f'Clean records: {len(out):,}')
    print(f'Date range: {out["datetime"].min()} to {out["datetime"].max()}')
    print('\ncrime_category:')
    print(out['crime_category'].value_counts().to_string())

    os.makedirs(os.path.dirname(PROC), exist_ok=True)
    out.to_csv(PROC, index=False)
    print(f'\nSaved to: {PROC}')


if __name__ == '__main__':
    main()
