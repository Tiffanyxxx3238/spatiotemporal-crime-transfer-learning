"""
03_eda.py
---------
探索性分析：時間分布、類別分布、跨城市比較。
產出圖表存到 outputs/eda/。
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

CITY_COLORS = {
    "NYC":     "#378ADD",
    "Chicago": "#1D9E75",
    "Karachi": "#D85A30",
}

CAT_ORDER = ["violent", "property", "drug", "public_order", "other"]
CAT_LABELS = {
    "violent":      "暴力犯罪",
    "property":     "財產犯罪",
    "drug":         "毒品相關",
    "public_order": "公共秩序",
    "other":        "其他",
}

def load_data():
    path = os.path.join(PROC_DIR, "all_cities.csv")
    if not os.path.exists(path):
        print("找不到 all_cities.csv，請先執行 02_preprocess.py")
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    print(f"載入 {len(df):,} 筆資料")
    return df

# ── 圖1：24小時犯罪分布 ──────────────────────────────────────
def plot_hourly(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    for city, grp in df.groupby("city"):
        hourly = grp.groupby("hour").size()
        hourly = hourly / hourly.sum() * 100   # 百分比
        ax.plot(hourly.index, hourly.values,
                label=city, color=CITY_COLORS.get(city, "gray"),
                linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("小時（0–23）")
    ax.set_ylabel("犯罪佔比 (%)")
    ax.set_title("三城市 24 小時犯罪分布")
    ax.legend()
    ax.set_xticks(range(0, 24, 2))
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "01_hourly_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"圖1 儲存：{path}")

# ── 圖2：月份趨勢 ───────────────────────────────────────────
def plot_monthly(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    for city, grp in df.groupby("city"):
        monthly = grp.groupby("month").size()
        monthly = monthly / monthly.sum() * 100
        ax.plot(monthly.index, monthly.values,
                label=city, color=CITY_COLORS.get(city, "gray"),
                linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("月份")
    ax.set_ylabel("犯罪佔比 (%)")
    ax.set_title("三城市月份犯罪趨勢")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["1月","2月","3月","4月","5月","6月",
                         "7月","8月","9月","10月","11月","12月"])
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "02_monthly_trend.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"圖2 儲存：{path}")

# ── 圖3：犯罪類別比較（grouped bar）──────────────────────────
def plot_categories(df):
    cities = df["city"].unique()
    cat_pct = {}
    for city in cities:
        grp = df[df["city"] == city]
        pct = grp["crime_category"].value_counts(normalize=True) * 100
        cat_pct[city] = pct

    cat_df = pd.DataFrame(cat_pct).T.fillna(0)
    # 只保留已定義的類別
    cols = [c for c in CAT_ORDER if c in cat_df.columns]
    cat_df = cat_df[cols].rename(columns=CAT_LABELS)

    ax = cat_df.plot(kind="bar", figsize=(10, 5), width=0.7,
                     color=["#E24B4A","#378ADD","#1D9E75","#888780","#B4B2A9"])
    ax.set_xlabel("")
    ax.set_ylabel("佔比 (%)")
    ax.set_title("三城市犯罪類別比較")
    ax.set_xticklabels(cat_df.index, rotation=0)
    ax.legend(title="犯罪類別", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "03_category_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"圖3 儲存：{path}")

# ── 圖4：星期幾分布（heatmap）───────────────────────────────
def plot_weekday_hour_heatmap(df, city="NYC"):
    sub = df[df["city"] == city].copy()
    if len(sub) == 0:
        return
    pivot = sub.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    pivot = pivot.div(pivot.values.sum()) * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0,
                xticklabels=range(0, 24, 1),
                yticklabels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                cbar_kws={"label": "佔比 (%)"})
    ax.set_title(f"{city} — 星期 × 小時犯罪熱圖")
    ax.set_xlabel("小時")
    ax.set_ylabel("")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"04_weekday_hour_heatmap_{city.lower()}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"圖4 ({city}) 儲存：{path}")

# ── 執行 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    if df is not None:
        plot_hourly(df)
        plot_monthly(df)
        plot_categories(df)
        for city in df["city"].unique():
            plot_weekday_hour_heatmap(df, city)
        print(f"\n所有圖表已存到 outputs/eda/")
        print("下一步：執行 04_baseline_model.py 建立基準模型")
