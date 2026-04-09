"""
02_preprocess.py
----------------
把三個城市的原始資料標準化成統一格式。

輸出欄位（每個城市都一樣）：
  city, datetime, hour, month, weekday,
  crime_category, latitude, longitude, district
"""

import os
import pandas as pd
import numpy as np

RAW_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════
# 犯罪類別映射表（這是整個專題的核心設計決策，作品集亮點之一）
# ════════════════════════════════════════════════════════════

NYC_CATEGORY_MAP = {
    # 暴力犯罪
    "ASSAULT 3 & RELATED OFFENSES": "violent",
    "FELONY ASSAULT":               "violent",
    "MURDER & NON-NEGL. MANSLAUGHTER": "violent",
    "RAPE":                         "violent",
    "ROBBERY":                      "violent",
    "KIDNAPPING & RELATED OFFENSES":"violent",
    # 財產犯罪
    "BURGLARY":                     "property",
    "GRAND LARCENY":                "property",
    "PETIT LARCENY":                "property",
    "GRAND LARCENY OF MOTOR VEHICLE":"property",
    "THEFT OF SERVICES":            "property",
    "THEFT-FRAUD":                  "property",
    "FRAUDS":                       "property",
    # 毒品相關
    "DANGEROUS DRUGS":              "drug",
    "CANNABIS RELATED OFFENSES":    "drug",
    # 公共秩序
    "DISORDERLY CONDUCT":           "public_order",
    "ADMINISTRATIVE CODE":          "public_order",
    "MISCELLANEOUS PENAL LAW":      "public_order",
    "OFFENSES AGAINST PUBLIC ADMINI":"public_order",
    "VEHICLE AND TRAFFIC LAWS":     "public_order",
}

CHICAGO_CATEGORY_MAP = {
    # 暴力犯罪
    "ASSAULT":                  "violent",
    "BATTERY":                  "violent",
    "HOMICIDE":                 "violent",
    "CRIM SEXUAL ASSAULT":      "violent",
    "ROBBERY":                  "violent",
    "KIDNAPPING":               "violent",
    "INTIMIDATION":             "violent",
    "HUMAN TRAFFICKING":        "violent",
    # 財產犯罪
    "BURGLARY":                 "property",
    "THEFT":                    "property",
    "MOTOR VEHICLE THEFT":      "property",
    "ARSON":                    "property",
    "CRIMINAL DAMAGE":          "property",
    "FRAUD":                    "property",
    "FORGERY & COUNTERFEITING": "property",
    # 毒品相關
    "NARCOTICS":                "drug",
    "OTHER NARCOTIC VIOLATION": "drug",
    "CANNABIS POSSESSION":      "drug",
    # 公共秩序
    "DECEPTIVE PRACTICE":       "public_order",
    "CRIMINAL TRESPASS":        "public_order",
    "WEAPONS VIOLATION":        "public_order",
    "LIQUOR LAW VIOLATION":     "public_order",
    "PROSTITUTION":             "public_order",
    "GAMBLING":                 "public_order",
    "PUBLIC PEACE VIOLATION":   "public_order",
    "INTERFERENCE WITH PUBLIC OFFICER": "public_order",
}

KARACHI_CATEGORY_MAP = {
    # 暴力犯罪（依 Kaggle 資料集欄位調整）
    "Robbery":          "violent",
    "Violence":         "violent",
    "Murder":           "violent",
    "Assault":          "violent",
    "Kidnapping":       "violent",
    "Rape":             "violent",
    # 財產犯罪
    "Theft":            "property",
    "Burglary":         "property",
    "Vehicle Theft":    "property",
    "Fraud":            "property",
    # 毒品相關
    "Drug offenses":    "drug",
    "Drug Trafficking": "drug",
    # 公共秩序
    "Other":            "public_order",
    "Public Order":     "public_order",
}

CATEGORY_LABELS = {
    "violent":      "暴力犯罪",
    "property":     "財產犯罪",
    "drug":         "毒品相關",
    "public_order": "公共秩序",
}

# ════════════════════════════════════════════════════════════
# NYC 前處理
# ════════════════════════════════════════════════════════════
def process_nyc():
    path = os.path.join(RAW_DIR, "nyc_raw.csv")
    if not os.path.exists(path):
        print("[NYC] 找不到原始資料，請先執行 01_download.py")
        return None

    print("[NYC] 前處理中...")
    df = pd.read_csv(path, low_memory=False)

    # 時間解析
    df["datetime"] = pd.to_datetime(
        df["cmplnt_fr_dt"].str[:10] + " " + df["cmplnt_fr_tm"].fillna("00:00:00"),
        errors="coerce"
    )
    df = df.dropna(subset=["datetime"])

    # 類別映射
    df["crime_category"] = df["ofns_desc"].str.strip().str.upper().map(
        {k.upper(): v for k, v in NYC_CATEGORY_MAP.items()}
    ).fillna("other")

    # 座標
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"] > 40.4) & (df["latitude"] < 41.0)]
    df = df[(df["longitude"] > -74.3) & (df["longitude"] < -73.6)]

    out = pd.DataFrame({
        "city":           "NYC",
        "datetime":       df["datetime"],
        "hour":           df["datetime"].dt.hour,
        "month":          df["datetime"].dt.month,
        "weekday":        df["datetime"].dt.weekday,
        "crime_category": df["crime_category"],
        "latitude":       df["latitude"],
        "longitude":      df["longitude"],
        "district":       df["boro_nm"].fillna("UNKNOWN"),
    })
    out_path = os.path.join(PROC_DIR, "nyc_clean.csv")
    out.to_csv(out_path, index=False)
    print(f"[NYC] 完成！{len(out):,} 筆 → {out_path}")
    print(f"      類別分布：\n{out['crime_category'].value_counts().to_string()}")
    return out

# ════════════════════════════════════════════════════════════
# Chicago 前處理
# ════════════════════════════════════════════════════════════
def process_chicago():
    path = os.path.join(RAW_DIR, "chicago_raw.csv")
    if not os.path.exists(path):
        print("[Chicago] 找不到原始資料，請先執行 01_download.py")
        return None

    print("[Chicago] 前處理中...")
    df = pd.read_csv(path, low_memory=False)

    df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df["crime_category"] = df["primary_type"].str.strip().str.upper().map(
        {k.upper(): v for k, v in CHICAGO_CATEGORY_MAP.items()}
    ).fillna("other")

    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"] > 41.6) & (df["latitude"] < 42.1)]
    df = df[(df["longitude"] > -87.95) & (df["longitude"] < -87.5)]

    out = pd.DataFrame({
        "city":           "Chicago",
        "datetime":       df["datetime"],
        "hour":           df["datetime"].dt.hour,
        "month":          df["datetime"].dt.month,
        "weekday":        df["datetime"].dt.weekday,
        "crime_category": df["crime_category"],
        "latitude":       df["latitude"],
        "longitude":      df["longitude"],
        "district":       df["district"].fillna("UNKNOWN").astype(str),
    })
    out_path = os.path.join(PROC_DIR, "chicago_clean.csv")
    out.to_csv(out_path, index=False)
    print(f"[Chicago] 完成！{len(out):,} 筆 → {out_path}")
    print(f"          類別分布：\n{out['crime_category'].value_counts().to_string()}")
    return out

# ════════════════════════════════════════════════════════════
# Karachi 前處理
# ════════════════════════════════════════════════════════════
def process_karachi():
    path = os.path.join(RAW_DIR, "karachi_raw.csv")
    if not os.path.exists(path):
        print("[Karachi] 找不到原始資料，請先手動下載後放到 data/raw/karachi_raw.csv")
        return None

    print("[Karachi] 前處理中...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  原始欄位：{list(df.columns)}")

    # 嘗試自動找時間欄位（Kaggle 資料集欄位名稱可能不固定）
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        print("  [警告] 找不到日期欄位，請手動指定")
        return None
    df["datetime"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # 找犯罪類型欄位
    type_col = next((c for c in df.columns if "crime" in c.lower() or "type" in c.lower()), None)
    if type_col is None:
        print("  [警告] 找不到犯罪類型欄位，請手動指定")
        return None

    df["crime_category"] = df[type_col].str.strip().map(KARACHI_CATEGORY_MAP).fillna("other")

    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    district_col = next((c for c in df.columns if "district" in c.lower() or "area" in c.lower()), None)

    out = pd.DataFrame({
        "city":           "Karachi",
        "datetime":       df["datetime"],
        "hour":           df["datetime"].dt.hour,
        "month":          df["datetime"].dt.month,
        "weekday":        df["datetime"].dt.weekday,
        "crime_category": df["crime_category"],
        "latitude":       pd.to_numeric(df[lat_col], errors="coerce") if lat_col else np.nan,
        "longitude":      pd.to_numeric(df[lon_col], errors="coerce") if lon_col else np.nan,
        "district":       df[district_col].fillna("UNKNOWN") if district_col else "UNKNOWN",
    })

    out_path = os.path.join(PROC_DIR, "karachi_clean.csv")
    out.to_csv(out_path, index=False)
    print(f"[Karachi] 完成！{len(out):,} 筆 → {out_path}")
    print(f"          類別分布：\n{out['crime_category'].value_counts().to_string()}")
    return out

# ════════════════════════════════════════════════════════════
# 執行
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    nyc = process_nyc()
    chi = process_chicago()
    kar = process_karachi()

    # 合併三城市（之後 EDA 和建模都用這個）
    dfs = [d for d in [nyc, chi, kar] if d is not None]
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(PROC_DIR, "all_cities.csv")
        combined.to_csv(out_path, index=False)
        print(f"\n合併完成！共 {len(combined):,} 筆 → {out_path}")
        print(f"各城市筆數：\n{combined['city'].value_counts().to_string()}")

    print("\n下一步：執行 03_eda.py 做探索性分析")
