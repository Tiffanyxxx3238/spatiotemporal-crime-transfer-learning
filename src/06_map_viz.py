"""
06_map_viz.py
-------------
用模型輸出的風險分數產出互動式 Folium 地圖。
執行後在 outputs/maps/ 產出 HTML，瀏覽器直接開啟即可。
"""

import os
import json
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "maps")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 城市設定 ─────────────────────────────────────────────────
CITY_CONFIG = {
    "NYC":     {"center": [40.730, -73.935], "zoom": 11},
    "Chicago": {"center": [41.850, -87.680], "zoom": 11},
    "Karachi": {"center": [24.860,  67.010], "zoom": 11},
}

TIME_SLOTS = {
    "深夜 (00–06h)":   (0, 6),
    "早晨 (06–12h)":   (6, 12),
    "下午 (12–18h)":   (12, 18),
    "夜晚 (18–24h)":   (18, 24),
}

RISK_COLOR = {
    "high":   "#E24B4A",
    "medium": "#EF9F27",
    "low":    "#1D9E75",
}

def risk_level(score):
    if score >= 75: return "high"
    if score >= 45: return "medium"
    return "low"

def risk_label(score):
    lvl = risk_level(score)
    return {"high": "高風險", "medium": "中風險", "low": "低風險"}[lvl]

# ── 計算每個網格的風險分數 ──────────────────────────────────
def compute_grid_risk(df, city, time_slot):
    sub = df[df["city"] == city].copy()
    h_start, h_end = TIME_SLOTS[time_slot]
    sub = sub[(sub["hour"] >= h_start) & (sub["hour"] < h_end)]
    sub = sub.dropna(subset=["latitude", "longitude"])

    if len(sub) == 0:
        return pd.DataFrame()

    # 0.01 度 ≈ 1 km，把座標 bin 到格子
    sub["lat_bin"] = (sub["latitude"]  / 0.01).round() * 0.01
    sub["lon_bin"] = (sub["longitude"] / 0.01).round() * 0.01

    grid = sub.groupby(["lat_bin", "lon_bin"]).agg(
        count=("crime_category", "size"),
        violent_pct=("crime_category", lambda x: (x == "violent").mean() * 100),
    ).reset_index()

    # 風險分數：用 count 正規化到 0–100
    if grid["count"].max() > 0:
        grid["risk_score"] = (grid["count"] / grid["count"].max() * 100).round(1)
    else:
        grid["risk_score"] = 0

    return grid

# ── 建立單城市互動地圖 ───────────────────────────────────────
def build_city_map(df, city):
    cfg = CITY_CONFIG[city]
    m = folium.Map(
        location=cfg["center"],
        zoom_start=cfg["zoom"],
        tiles="CartoDB positron",   # 簡潔底圖，適合疊加資料
    )

    # 每個時段建一個 FeatureGroup（可切換顯示）
    for slot_name in TIME_SLOTS:
        grid = compute_grid_risk(df, city, slot_name)
        if grid.empty:
            continue

        fg = folium.FeatureGroup(name=slot_name, show=(slot_name == "下午 (12–18h)"))

        for _, row in grid.iterrows():
            score = row["risk_score"]
            lvl   = risk_level(score)
            color = RISK_COLOR[lvl]

            # 每個格子畫成矩形
            folium.Rectangle(
                bounds=[
                    [row["lat_bin"] - 0.005, row["lon_bin"] - 0.005],
                    [row["lat_bin"] + 0.005, row["lon_bin"] + 0.005],
                ],
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=min(0.65, score / 100 + 0.1),
                weight=0,
                tooltip=folium.Tooltip(
                    f"<b>風險指數：{score:.0f}/100</b><br>"
                    f"等級：{risk_label(score)}<br>"
                    f"暴力犯罪佔比：{row['violent_pct']:.1f}%<br>"
                    f"案件數（此時段）：{int(row['count'])}"
                ),
            ).add_to(fg)

        fg.add_to(m)

    # 圖層切換控件
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    # 圖例
    legend_html = """
    <div style="
        position: absolute; bottom: 30px; left: 30px; z-index: 1000;
        background: white; padding: 12px 16px; border-radius: 8px;
        border: 1px solid #ddd; font-family: sans-serif; font-size: 13px;
    ">
        <b>風險等級</b><br>
        <span style="color:#E24B4A">■</span> 高風險 (75–100)<br>
        <span style="color:#EF9F27">■</span> 中風險 (45–74)<br>
        <span style="color:#1D9E75">■</span> 低風險 (0–44)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    out_path = os.path.join(OUT_DIR, f"map_{city.lower()}.html")
    m.save(out_path)
    print(f"[{city}] 地圖已儲存：{out_path}")
    return out_path

# ── HeatMap 版本（另一種視覺化風格）────────────────────────
def build_heatmap(df, city):
    cfg = CITY_CONFIG[city]
    sub = df[(df["city"] == city)].dropna(subset=["latitude", "longitude"])

    m = folium.Map(location=cfg["center"], zoom_start=cfg["zoom"],
                   tiles="CartoDB dark_matter")

    for slot_name, (h_start, h_end) in TIME_SLOTS.items():
        slot_df = sub[(sub["hour"] >= h_start) & (sub["hour"] < h_end)]
        heat_data = slot_df[["latitude", "longitude"]].values.tolist()

        fg = folium.FeatureGroup(name=slot_name, show=(slot_name == "夜晚 (18–24h)"))
        HeatMap(
            heat_data,
            radius=12,
            blur=15,
            min_opacity=0.3,
            gradient={"0.4": "#1D9E75", "0.65": "#EF9F27", "1": "#E24B4A"},
        ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    out_path = os.path.join(OUT_DIR, f"heatmap_{city.lower()}.html")
    m.save(out_path)
    print(f"[{city}] 熱力圖已儲存：{out_path}")
    return out_path

# ── 執行 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    path = os.path.join(PROC_DIR, "all_cities.csv")
    if not os.path.exists(path):
        print("找不到 all_cities.csv，請先執行 01–02 的前處理步驟")
    else:
        df = pd.read_csv(path)
        print(f"載入 {len(df):,} 筆資料")

        for city in df["city"].unique():
            build_city_map(df, city)
            build_heatmap(df, city)

        print(f"\n所有地圖已輸出到 outputs/maps/")
        print("用瀏覽器直接打開 .html 檔案即可看到互動地圖！")
