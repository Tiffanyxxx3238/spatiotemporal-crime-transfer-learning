"""
Merge all city clean CSVs into data/processed/all_cities.csv
"""
import pandas as pd
import os

PROC = r"C:\Users\user\GitHub\model-predict-crime\data\processed"

files = {
    "NYC":            "nyc_clean.csv",
    "Chicago":        "chicago_clean.csv",
    "LA":             "la_clean.csv",
    "Karachi":        "karachi_clean.csv",
    "London":         "london_clean.csv",
    "Philadelphia":   "philadelphia_clean.csv",
    "DC":             "dc_clean.csv",
    "West Yorkshire": "west_yorkshire_clean.csv",
    "Detroit":        "detroit_clean.csv",
    "Cambridge":      "cambridge_clean.csv",
    "Salt Lake City": "slc_clean.csv",
    "Birmingham":     "birmingham_clean.csv",
    "Peoria":         "peoria_clean.csv",
    "Kansas City":    "kansas_city_clean.csv",
}

dfs = []
for city, fname in files.items():
    path = os.path.join(PROC, fname)
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        print(f"  {city}: {len(df):,} records")
        dfs.append(df)
    else:
        print(f"  {city}: MISSING ({path})")

combined = pd.concat(dfs, ignore_index=True)
out = os.path.join(PROC, "all_cities.csv")
combined.to_csv(out, index=False)
print(f"\nTotal: {len(combined):,} records -> {out}")
print(combined["city"].value_counts().to_string())
