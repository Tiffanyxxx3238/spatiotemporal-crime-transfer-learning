"""
Download Detroit RMS Crime Incidents (2017–2026)
Source: data.detroitmi.gov (ArcGIS Hub)
Output: data/raw/detroit_raw.csv
"""
import os
import requests
import pandas as pd
from io import StringIO

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

YEAR_URLS = {
    2017: "https://data.detroitmi.gov/api/download/v1/items/9d4040a2b02343dbbe59331e2bd29e84/csv?layers=0",
    2018: "https://data.detroitmi.gov/api/download/v1/items/bd5ecbffdbdb4db1bc8e2ec9123e7861/csv?layers=0",
    2019: "https://data.detroitmi.gov/api/download/v1/items/332b28b9ff0e4777aa266704359246ca/csv?layers=0",
    2020: "https://data.detroitmi.gov/api/download/v1/items/20a3c63d31a44088b143e925fc71a5a1/csv?layers=0",
    2021: "https://data.detroitmi.gov/api/download/v1/items/63a224e108914937a6a4cd9f96d25e81/csv?layers=0",
    2022: "https://data.detroitmi.gov/api/download/v1/items/e461023806dc4ab79746afd2a3e41e25/csv?layers=0",
    2023: "https://data.detroitmi.gov/api/download/v1/items/43e793425d1d486a807c731d88648ac7/csv?layers=0",
    2024: "https://data.detroitmi.gov/api/download/v1/items/ed7646f5c75c4de4ae0281054b9300a9/csv?layers=0",
    2025: "https://data.detroitmi.gov/api/download/v1/items/b794a57159204676a690cb2dd736181c/csv?layers=0",
    2026: "https://data.detroitmi.gov/api/download/v1/items/fc701d4b3e14413db2d9f78f0ae0a105/csv?layers=0",
}

KEEP_COLS = [
    "offense_category",
    "incident_occurred_at",
    "incident_hour_of_day",
    "latitude",
    "longitude",
]


def download_detroit():
    out_path = os.path.join(RAW_DIR, "detroit_raw.csv")
    if os.path.exists(out_path):
        print(f"[Detroit] Already exists, skipping. Delete {out_path} to re-download.")
        return

    all_dfs = []
    for year, url in sorted(YEAR_URLS.items()):
        print(f"  {year}...", end=" ", flush=True)
        try:
            r = requests.get(url, timeout=300, stream=True)
            r.raise_for_status()
            content = r.content.decode("utf-8", errors="replace")
            df = pd.read_csv(StringIO(content), low_memory=False)
            # Keep only needed columns (ignore missing ones gracefully)
            cols = [c for c in KEEP_COLS if c in df.columns]
            df = df[cols]
            df["year"] = year
            print(f"{len(df):,}")
            all_dfs.append(df)
        except Exception as e:
            print(f"Error: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(out_path, index=False)
        print(f"\n[Detroit] Done: {len(combined):,} records -> {out_path}")
    else:
        print("[Detroit] No data downloaded.")


if __name__ == "__main__":
    print("[Detroit] Downloading 2017-2026 RMS Crime Incidents...")
    download_detroit()
