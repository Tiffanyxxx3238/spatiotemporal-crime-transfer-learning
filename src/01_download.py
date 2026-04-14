"""
01_download.py
--------------
下載三個城市的犯罪資料。

NYC 和 Chicago 用官方 API 自動下載（近 5 年）。
Karachi 需手動從 Kaggle 下載後放到 data/raw/。
"""

import os
import requests
import pandas as pd
from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ── 共用設定 ────────────────────────────────────────────────
START_YEAR = 2006   # 抓近 5 年
LIMIT = 2_000_000     # 每次最多拉幾筆（NYC/Chicago API 上限）

# ── NYC：NYPD Complaint Data ─────────────────────────────────
NYC_API = "https://data.cityofnewyork.us/resource/qgea-i56i.json"

def download_nyc():
    out_path = os.path.join(RAW_DIR, "nyc_raw.csv")
    if os.path.exists(out_path):
        print(f"[NYC] 已存在，跳過：{out_path}")
        return

    print("[NYC] 開始分年下載 (2015-2023)...")
    all_dfs = []
    for year in range(2006, 2025):
        print(f"  下載 {year}...")
        params = {
            "$limit": 600000,
            "$where": (f"cmplnt_fr_dt >= '{year}-01-01T00:00:00' AND "
                       f"cmplnt_fr_dt <= '{year}-12-31T23:59:59'"),
            "$select": (
                "cmplnt_fr_dt,cmplnt_fr_tm,"
                "ofns_desc,law_cat_cd,"
                "boro_nm,addr_pct_cd,"
                "latitude,longitude"
            ),
        }
        try:
            r = requests.get(NYC_API, params=params, timeout=120)
            r.raise_for_status()
            df_yr = pd.DataFrame(r.json())
            print(f"    {len(df_yr):,} 筆")
            all_dfs.append(df_yr)
        except Exception as e:
            print(f'    {year} Error: {e}, skipping...')
            continue

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"[NYC] 完成！共 {len(df):,} 筆 -> {out_path}")

# ── Chicago：Crimes 2001 to Present ─────────────────────────
CHI_API = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"

def download_chicago():
    out_path = os.path.join(RAW_DIR, "chicago_raw.csv")
    if os.path.exists(out_path):
        print(f"[Chicago] 已存在，跳過：{out_path}")
        return

    print("[Chicago] 開始分年下載 (2015-2023)...")
    all_dfs = []
    for year in range(2001, 2025):
        print(f"  下載 {year}...")
        params = {
            "$limit": 600000,
            "$where": f"year = {year}",
            "$select": (
                "date,"
                "primary_type,description,"
                "community_area,district,"
                "latitude,longitude"
            ),
        }
        try:
            r = requests.get(CHI_API, params=params, timeout=120)
            r.raise_for_status()
            df_yr = pd.DataFrame(r.json())
            print(f"    {len(df_yr):,} 筆")
            all_dfs.append(df_yr)
        except Exception as e:
            print(f'    {year} Error: {e}, skipping...')
            continue

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"[Chicago] 完成！共 {len(df):,} 筆 -> {out_path}")

# ── Karachi：手動下載說明 ────────────────────────────────────
def check_karachi():
    kar_path = os.path.join(RAW_DIR, "karachi_raw.csv")
    if os.path.exists(kar_path):
        print(f"[Karachi] 找到檔案：{kar_path}")
        return
    print("""
[Karachi] 尚未找到資料，請手動下載：
  1. 前往 https://www.kaggle.com/datasets/sarcasmos/karachi-urban-crime-analysis-with-demographic
  2. 點 Download，解壓縮後找到 .csv 檔
  3. 重新命名為 karachi_raw.csv
  4. 放到 data/raw/ 資料夾
""")

# ── 執行 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    download_nyc()
    download_chicago()
    check_karachi()
    print("\n全部完成！接著執行 02_preprocess.py")
