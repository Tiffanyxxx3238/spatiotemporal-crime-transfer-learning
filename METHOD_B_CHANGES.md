# Method B 實作紀錄 — 月份季節特徵升級

**日期：** 2026-05-16  
**目標：** 將所有 14 個城市的 grid 聚合加入 `month` 維度，讓模型具備季節辨識能力，並在互動地圖新增春/夏/秋/冬篩選器。

---

## 一、為何做這個改動（Method B vs Method A）

原始 grid 聚合只以 `(lat_bin, lon_bin, time_slot)` 為鍵，忽略月份差異。

| 方法 | 作法 | 結果 |
|------|------|------|
| **Method A**（靜態平均） | 計算每格 `month_sin_avg`, `month_cos_avg` 當特徵 | Peoria −0.013、KC −0.004，**無改善** |
| **Method B**（月份入 groupby） | `groupby(['lat_bin','lon_bin','time_slot','month'])` | Kansas City Precision **+3.0 pp**（0.6775 → 0.7072）|

Method B 有效的原因：模型看到「這個格子在 7 月的犯罪組成」，而非靜態平均，真正捕捉到季節信號。

---

## 二、各城市資料量變化（Before → After）

| 城市 | 舊 grid 數 | 新 grid 數 | 倍率 | 有 month |
|------|----------:|----------:|-----:|:--------:|
| NYC | 3,503 | **36,251** | 10× | ✅ |
| Chicago | 2,740 | **28,194** | 10× | ✅ |
| LA | 4,668 | **37,716** | 8× | ✅ |
| London | 7,348 | **41,946** | 6× | ✅ |
| Philadelphia | 1,495 | **13,008** | 9× | ✅ |
| DC | 633 | **3,783** | 6× | ✅ |
| West Yorkshire | 4,940 | **17,329** | 4× | ✅ |
| Detroit | 1,774 | **15,856** | 9× | ✅ |
| Kansas City | ~4,000 | **20,123** | 5× | ✅ |
| Peoria | 847 | **3,889** | 5× | ✅ |
| Cambridge | 540 | **854** | 2× | ✅ |
| Salt Lake City | 548 | **1,242** | 2× | ✅ |
| Birmingham | 465 | **284** | — | ✅ |
| Karachi | 153 | **1,677** | 11× | ✅ |

> Grid 數量增加的原因：同一個 (lat, lon, time_slot) 現在按月份拆分，每個月各有一筆記錄。

---

## 三、地圖準確率（map_acc）

| 城市 | map_acc |
|------|--------:|
| Cambridge | 100.0% |
| Salt Lake City | 99.8% |
| Birmingham | 98.9% |
| DC | 92.9% |
| Detroit | 93.2% |
| Peoria | 82.6% |
| Philadelphia | 74.2% |
| Chicago | 69.2% |
| Kansas City | 67.7% |
| LA | 67.3% |
| West Yorkshire | 63.3% |
| NYC | 62.1% |
| Karachi | 56.3% |
| London | 45.9% |

---

## 四、腳本修改清單

所有修改均套用至 14 個城市的 `.py` 腳本：

### 4.1 核心 Method B patch（`patch_method_b_all.py`）
- `groupby(['lat_bin','lon_bin','time_slot'])` → 加入 `'month'`
- `risk_df = grid_test[['lat_bin','lon_bin','time_slot', ...]]` → 加入 `'month'`
- FEATURES list 加入 `'month'`, `'month_sin'`, `'month_cos'`

### 4.2 `make_features` 合併鍵修正（DC-pattern 腳本）
```python
# Before（錯誤：造成 many-to-many join）
d = d.merge(ref, on=['lat_bin','lon_bin','time_slot'], how='left')

# After
hist_cols = ['lat_bin','lon_bin','time_slot','month'] + ...
d = d.merge(ref, on=['lat_bin','lon_bin','time_slot','month'], how='left')
d = d.reset_index(drop=True)   # 修正 index 對齊問題
```
**影響檔案：** Philadelphia、WestYorkshire、London、LA、Chicago、NYC、Karachi

### 4.3 `filter_known_cats` index 修正（Karachi）
```python
# 加入 reset_index，避免 non-sequential index 與 RangeIndex 衝突
grid = grid.reset_index(drop=True)
X    = X.reset_index(drop=True)
```

### 4.4 `save_city_outputs` 補上 `month`
```python
# Before
risk_df = grid_test[['lat_bin','lon_bin','time_slot',
                      'total_count','dominant_category', ...]]
# After
risk_df = grid_test[['lat_bin','lon_bin','time_slot','month',
                      'total_count','dominant_category', ...]]
```
**影響檔案：** Chicago、LA、London、Philadelphia、WestYorkshire、Karachi

### 4.5 相對路徑 → 絕對路徑
```python
# Before（從 notebook/ 目錄執行才正確）
PROC_DIR  = '../data/processed'
MODEL_DIR = '../outputs/models'

# After（從專案根目錄執行）
PROC_DIR  = r'C:\Users\user\GitHub\model-predict-crime\data\processed'
MODEL_DIR = r'C:\Users\user\GitHub\model-predict-crime\outputs\models'
```
**影響檔案：** Philadelphia、LA、NYC、WestYorkshire、Chicago、London

### 4.6 Matplotlib 後端修正
```python
# 加在 import 最上方（避免無頭環境嘗試開視窗造成 exit code 5）
import matplotlib
matplotlib.use('Agg')
```
**影響檔案：** Philadelphia、LA、Chicago、NYC、London

---

## 五、地圖升級（crime_map_v8.html）

| 版本 | 大小 | 說明 |
|------|-----:|------|
| crime_map_v6.html | 5.9 MB | 加入 Peoria、Kansas City |
| crime_map_v7.html | 9.1 MB | KC 加入 month；新增季節篩選 JS |
| **crime_map_v8.html** | **42.2 MB** | **全部 14 城市 + month；Karachi 新分頁** |

### 新功能
- **季節篩選**（春/夏/秋/冬/全年）對所有 14 城市生效
- **Karachi 新增**：加入資料分頁（卡拉奇）
- Grid 數量大幅增加，每個時段×月份組合各有獨立預測

### 季節篩選運作方式
```javascript
// renderCity 過濾邏輯
const grids = data.grids.filter(g => {
    if (g.ts !== timeSlot) return false;
    if (currentSeason === null || !data.meta.has_month) return true;
    return currentSeason.includes(g.mo);  // g.mo = 月份 1~12
});
```

---

## 六、新增/修改的檔案

| 檔案 | 說明 |
|------|------|
| `patch_method_b_all.py` | 批次 patch 腳本（已完成任務，可保留參考） |
| `update_map_v8.py` | 產生 crime_map_v8.html 的腳本 |
| `outputs/maps/crime_map_v8.html` | 最新互動地圖（42.2 MB） |
| `outputs/models/grid_risk_*.csv` | 全部 14 個城市重新產生，含 month 欄位 |
| `outputs/models/model_*_catboost.cbm` | 重新訓練的模型（含月份特徵） |
| `outputs/models/model_*_lgb.pkl` | 重新訓練的模型（含月份特徵） |
