import requests
import pandas as pd

LA_API_OLD = "https://data.lacity.org/resource/63jg-8b9z.json"
SELECT = "date_occ,time_occ,crm_cd_desc,area_name,lat,lon"
LA_RAW = r"C:\Users\user\GitHub\model-predict-crime\data\raw\la_raw.csv"

print("Downloading LA 2010-2019...")
all_dfs = []
for year in range(2010, 2020):
    print(f"  {year}...", end=" ", flush=True)
    params = {
        "$limit": 400000,
        "$where": f"date_occ >= '{year}-01-01T00:00:00' AND date_occ <= '{year}-12-31T23:59:59'",
        "$select": SELECT,
    }
    try:
        r = requests.get(LA_API_OLD, params=params, timeout=120)
        r.raise_for_status()
        df_yr = pd.DataFrame(r.json())
        print(f"{len(df_yr):,}")
        all_dfs.append(df_yr)
    except Exception as e:
        print(f"Error: {e}")

old = pd.concat(all_dfs, ignore_index=True)
print(f"\n2010-2019 downloaded: {len(old):,} records")

existing = pd.read_csv(LA_RAW, low_memory=False)
print(f"Existing 2020-2024: {len(existing):,} records")

combined = pd.concat([old, existing], ignore_index=True)
combined.to_csv(LA_RAW, index=False)
print(f"Merged total: {len(combined):,} records -> {LA_RAW}")
