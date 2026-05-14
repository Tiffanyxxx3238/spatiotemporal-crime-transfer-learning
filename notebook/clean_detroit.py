"""
Detroit RMS Crime Incidents cleaner
Input:  data/raw/detroit_raw.csv  (from src/download_detroit.py)
Output: data/processed/detroit_clean.csv

Usage:
  python clean_detroit.py
"""
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE, "..", "data", "raw", "detroit_raw.csv")
OUT_PATH = os.path.join(BASE, "..", "data", "processed", "detroit_clean.csv")

# Detroit lat/lon bounds (city only — filter out outliers)
LAT_MIN, LAT_MAX = 42.25, 42.55
LON_MIN, LON_MAX = -83.30, -82.90

CRIME_MAP = {
    "ASSAULT":                    "violent",
    "AGGRAVATED ASSAULT":         "violent",
    "ROBBERY":                    "violent",
    "SEXUAL ASSAULT":             "violent",
    "HOMICIDE":                   "violent",
    "WEAPONS OFFENSES":           "violent",
    "KIDNAPPING":                 "violent",
    "SEX OFFENSES":               "violent",
    "LARCENY":                    "property",
    "STOLEN VEHICLE":             "property",
    "BURGLARY":                   "property",
    "STOLEN PROPERTY":            "property",
    "FRAUD":                      "property",
    "ARSON":                      "property",
    "DAMAGE TO PROPERTY":         "property",
    "FORGERY":                    "property",
    "DANGEROUS DRUGS":            "drug",
    "DISORDERLY CONDUCT":         "public_order",
    "OBSTRUCTING THE POLICE":     "public_order",
    "OBSTRUCTING JUDICIARY":      "public_order",
    "OUIL":                       "public_order",
    "LIQUOR":                     "public_order",
    "FAMILY OFFENSE":             "other",
    "RUNAWAY":                    "other",
    "OTHER":                      "other",
    "MISCELLANEOUS":              "other",
    "EXTORTION":                  "other",
    "HEALTH AND SAFETY":          "other",
    "JUSTIFIABLE HOMICIDE":       "other",
    "SOLICITATION":               "other",
    "GAMBLING":                   "other",
    "INVASION OF PRIVACY -OTHER": "other",
}


def clean():
    print(f"Reading: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"Raw records: {len(df):,}")

    df = df.dropna(subset=["offense_category", "incident_occurred_at",
                            "latitude", "longitude"])
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    # Bounding box filter
    df = df[
        (df["latitude"]  >= LAT_MIN) & (df["latitude"]  <= LAT_MAX) &
        (df["longitude"] >= LON_MIN) & (df["longitude"] <= LON_MAX)
    ]
    print(f"After bbox filter: {len(df):,}")

    df["crime_category"] = df["offense_category"].str.strip().map(CRIME_MAP).fillna("other")

    df["datetime"] = pd.to_datetime(df["incident_occurred_at"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Use incident_hour_of_day if available, else derive from datetime
    if "incident_hour_of_day" in df.columns:
        df["hour"] = pd.to_numeric(df["incident_hour_of_day"], errors="coerce").fillna(
            df["datetime"].dt.hour
        ).astype(int)
    else:
        df["hour"] = df["datetime"].dt.hour

    out = pd.DataFrame({
        "city":           "Detroit",
        "datetime":       df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "hour":           df["hour"],
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
