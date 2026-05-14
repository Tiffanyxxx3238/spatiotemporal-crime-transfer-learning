"""
Cambridge (Cambridgeshire) crime data cleaner
Input:  data/raw/cambridge_raw.csv  (from src/download_cambridge.py)
Output: data/processed/cambridge_clean.csv

The raw file has columns: category, date (YYYY-MM), latitude, longitude, crime_category
(crime_category already mapped in the download script, but we re-map here for consistency)

Usage:
  python clean_cambridge.py
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE, "..", "data", "raw", "cambridge_raw.csv")
OUT_PATH = os.path.join(BASE, "..", "data", "processed", "cambridge_clean.csv")

# Cambridge city bounding box
LAT_MIN, LAT_MAX = 52.15, 52.28
LON_MIN, LON_MAX =  0.05,  0.22

CRIME_MAP = {
    "anti-social-behaviour":              "other",
    "bicycle-theft":                      "property",
    "burglary":                           "property",
    "criminal-damage-arson":              "property",
    "drugs":                              "drug",
    "other-crime":                        "other",
    "other-theft":                        "property",
    "possession-of-weapons":              "violent",
    "public-order":                       "public_order",
    "robbery":                            "violent",
    "shoplifting":                        "property",
    "theft-from-the-person":              "property",
    "vehicle-crime":                      "property",
    "violent-crime":                      "violent",
    "violence-and-sexual-offences":       "violent",
    "stalking-and-harassment":            "other",
    "miscellaneous-crimes-against-society": "other",
}


def clean():
    print(f"Reading: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"Raw records: {len(df):,}")

    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "category"])

    # Bounding box
    df = df[
        (df["latitude"]  >= LAT_MIN) & (df["latitude"]  <= LAT_MAX) &
        (df["longitude"] >= LON_MIN) & (df["longitude"] <= LON_MAX)
    ]
    print(f"After bbox filter: {len(df):,}")

    df["crime_category"] = df["category"].map(CRIME_MAP).fillna("other")

    # date column is YYYY-MM — randomise day and hour for temporal granularity
    df["date_parsed"] = pd.to_datetime(df["date"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["date_parsed"])

    np.random.seed(42)
    n = len(df)
    df["datetime"] = (
        df["date_parsed"]
        + pd.to_timedelta(np.random.randint(0, 28, n), unit="D")
        + pd.to_timedelta(np.random.randint(0, 24, n), unit="h")
    )

    out = pd.DataFrame({
        "city":           "Cambridge",
        "datetime":       df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "hour":           df["datetime"].dt.hour,
        "month":          df["datetime"].dt.month,
        "weekday":        df["datetime"].dt.weekday,
        "crime_category": df["crime_category"],
        "latitude":       df["latitude"],
        "longitude":      df["longitude"],
    })

    print(f"Clean records: {len(out):,}")
    print(f"Date range: {out['datetime'].min()} to {out['datetime'].max()}")
    print(f"\ncrime_category:\n{out['crime_category'].value_counts()}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    clean()
