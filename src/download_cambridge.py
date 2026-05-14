"""
Download Cambridge (Cambridgeshire) crime data via data.police.uk API
Covers Cambridge city (polygon boundary) from 2019-04 to 2025-03
Output: data/raw/cambridge_raw.csv

Uses the crimes-street API with a polygon approximating Cambridge city.
API docs: https://data.police.uk/docs/method/crime-street/
"""
import os
import time
import requests
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Cambridge city bounding polygon (lat,lng pairs, clockwise)
# Covers central Cambridge and surrounding city area
CAMBRIDGE_POLY = (
    "52.240,0.060:"
    "52.240,0.210:"
    "52.155,0.210:"
    "52.155,0.060"
)

POLICE_UK_API = "https://data.police.uk/api/crimes-street/all-crime"

CRIME_MAP = {
    "anti-social-behaviour":          "other",
    "bicycle-theft":                  "property",
    "burglary":                       "property",
    "criminal-damage-arson":          "property",
    "drugs":                          "drug",
    "other-crime":                    "other",
    "other-theft":                    "property",
    "possession-of-weapons":          "violent",
    "public-order":                   "public_order",
    "robbery":                        "violent",
    "shoplifting":                    "property",
    "theft-from-the-person":          "property",
    "vehicle-crime":                  "property",
    "violent-crime":                  "violent",
    "violence-and-sexual-offences":   "violent",
    "stalking-and-harassment":        "other",
    "miscellaneous-crimes-against-society": "other",
}


def get_months(start_year, start_month, end_year, end_month):
    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def download_cambridge():
    out_path = os.path.join(RAW_DIR, "cambridge_raw.csv")
    if os.path.exists(out_path):
        print(f"[Cambridge] Already exists, skipping. Delete {out_path} to re-download.")
        return

    months = get_months(2019, 4, 2025, 3)
    print(f"[Cambridge] Downloading {len(months)} months ({months[0]} to {months[-1]})...")

    all_records = []
    for date_str in months:
        try:
            r = requests.get(
                POLICE_UK_API,
                params={"poly": CAMBRIDGE_POLY, "date": date_str},
                timeout=60,
            )
            if r.status_code == 200:
                data = r.json()
                for item in data:
                    loc = item.get("location", {})
                    all_records.append({
                        "category":  item.get("category", ""),
                        "date":      item.get("month", date_str),
                        "latitude":  loc.get("latitude"),
                        "longitude": loc.get("longitude"),
                    })
                print(f"  {date_str}: {len(data):,} records", flush=True)
            else:
                print(f"  {date_str}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  {date_str}: Error - {e}")
        time.sleep(0.4)   # respect rate limit

    if all_records:
        df = pd.DataFrame(all_records)
        df["crime_category"] = df["category"].map(CRIME_MAP).fillna("other")
        df.to_csv(out_path, index=False)
        print(f"\n[Cambridge] Done: {len(df):,} records -> {out_path}")
    else:
        print("[Cambridge] No data downloaded.")


if __name__ == "__main__":
    download_cambridge()
