import pandas as pd

# Read just Karachi rows
print("Reading all_cities.csv...")
df = pd.read_csv(r"C:\Users\user\GitHub\model-predict-crime\data\processed\all_cities.csv",
                 low_memory=False)
print(f"Total rows: {len(df):,}")
print(f"\nCities:\n{df['city'].value_counts().to_string()}")

kar = df[df['city'] == 'Karachi']
print(f"\nKarachi rows: {len(kar):,}")
if len(kar) > 0:
    print(f"crime_category values:\n{kar['crime_category'].value_counts()}")
    print(f"Sample crime_category repr: {repr(kar['crime_category'].iloc[0])}")
    print(f"datetime sample: {kar['datetime'].head(3).tolist()}")
    print(f"lat/lon nulls: {kar[['latitude','longitude']].isna().sum().to_string()}")

CATEGORIES = ['violent', 'property', 'other']
kar_filtered = kar[kar['crime_category'].isin(CATEGORIES)]
print(f"\nAfter CATEGORIES filter: {len(kar_filtered):,}")
