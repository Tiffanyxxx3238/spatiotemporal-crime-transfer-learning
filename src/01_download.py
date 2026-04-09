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
START_YEAR = 2019   # 抓近 5 年
LIMIT = 500_000     # 每次最多拉幾筆（NYC/Chicago API 上限）

# ── NYC：NYPD Complaint Data ─────────────────────────────────
NYC_API = "https://data.cityofnewyork.us/resource/qgea-i56i.json"

def download_nyc():
    out_path = os.path.join(RAW_DIR, "nyc_raw.csv")
    if os.path.exists(out_path):
        print(f"[NYC] 已存在，跳過下載：{out_path}")
        return

    print("[NYC] 開始下載...")
    params = {
        "$limit": LIMIT,
        "$where": f"cmplnt_fr_dt >= '{START_YEAR}-01-01T00:00:00'",
        "$select": (
            "cmplnt_fr_dt,cmplnt_fr_tm,"   # 日期、時間
            "ofns_desc,law_cat_cd,"         # 犯罪描述、等級
            "boro_nm,addr_pct_cd,"          # 行政區、precinct
            "latitude,longitude"            # 座標
        ),
    }
    r = requests.get(NYC_API, params=params, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df.to_csv(out_path, index=False)
    print(f"[NYC] 完成！共 {len(df):,} 筆 → {out_path}")

# ── Chicago：Crimes 2001 to Present ─────────────────────────
CHI_API = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"

def download_chicago():
    out_path = os.path.join(RAW_DIR, "chicago_raw.csv")
    if os.path.exists(out_path):
        print(f"[Chicago] 已存在，跳過下載：{out_path}")
        return

    print("[Chicago] 開始下載...")
    params = {
        "$limit": LIMIT,
        "$where": f"year >= {START_YEAR}",
        "$select": (
            "date,"                         # 日期時間
            "primary_type,description,"     # 犯罪類型
            "community_area,district,"      # 行政區
            "latitude,longitude"            # 座標
        ),
    }
    r = requests.get(CHI_API, params=params, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df.to_csv(out_path, index=False)
    print(f"[Chicago] 完成！共 {len(df):,} 筆 → {out_path}")

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
