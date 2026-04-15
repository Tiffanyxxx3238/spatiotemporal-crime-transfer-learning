<div align="center">

# 🍍 鳳梨成熟度辨識系統
### Pineapple Ripeness Detection System

*電子鼻 × 邊緣運算 × 機器學習 | Electronic Nose × Edge Computing × Machine Learning*

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=flat-square&logo=python&logoColor=white)
![Arduino](https://img.shields.io/badge/Arduino-Mega_2560-00979D?style=flat-square&logo=arduino&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi-3_Model_B-A22846?style=flat-square&logo=raspberry-pi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ExtraTrees-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![React Native](https://img.shields.io/badge/React_Native-App-61DAFB?style=flat-square&logo=react&logoColor=black)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=flat-square&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Version](https://img.shields.io/badge/Version-v1.0-blue?style=flat-square)

</div>

---

## 📖 專案簡介

本系統為**非破壞性鳳梨成熟度辨識系統**，結合電子鼻氣體感測器陣列、Arduino Mega、Raspberry Pi 3 以及機器學習模型，透過量測鳳梨周圍揮發性有機化合物（VOC）訊號，輸出**四階段成熟度分類結果**，協助農民、驗收人員與消費者進行快速且客觀的成熟度判定。

### 🎯 目標使用者

| 使用者 | 使用情境 |
|--------|----------|
| 🌾 **農民** | 採收與分級前快速判斷，減少農損 |
| 🔍 **驗收人員** | 降低主觀經驗差異，提供量化依據 |
| 🛒 **消費者** | 判斷是否適合立即食用 |

### 🍍 成熟度分級說明

| Stage | 說明 | 建議 |
|-------|------|------|
| **Stage 0** | 未成熟 | 再等 3–5 天 |
| **Stage 1** | 初熟 | 再等 1–2 天 |
| **Stage 2** | 完熟 | 立即食用 ✅ |
| **Stage 3** | 過熟 | 盡快食用 ⚠️ |

---

## 🏗️ 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│            👤 使用者端  User Interface Layer                 │
│  消費端 App (React Native) │ 農民端 App │ Flask 模擬介面      │
├─────────────────────────────────────────────────────────────┤
│            🌐 應用服務層  Application Service Layer          │
│          SSH 遠端控制 │ API 服務 │ 結果紀錄 / 報表            │
├─────────────────────────────────────────────────────────────┤
│            🖥️  運算處理層  Processing Layer                  │
│   Raspberry Pi 3 │ 特徵工程 │ ExtraTrees 推論 │ 後處理邏輯   │
├─────────────────────────────────────────────────────────────┤
│            🔌 資料傳輸層  Data Transfer Layer                │
│          USB Serial (115200 baud) │ I²C (BME280)             │
├─────────────────────────────────────────────────────────────┤
│            🔬 感測層  Sensing Layer                          │
│   Arduino Mega 2560 │ MQ 系列感測器 │ TGS 系列 │ BME280      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 硬體需求

### 元件清單

| 元件 | 規格 | 數量 | 用途 |
|------|------|------|------|
| Arduino Mega 2560 | ATmega2560, 16MHz | 1 | 感測器資料擷取 |
| Raspberry Pi 3 Model B | 1.2GHz Quad-Core, 1GB RAM | 1 | 邊緣運算、模型推論 |
| MQ-2 氣體感測器 | 煙霧 / 可燃氣體 | 1 | VOC 特徵擷取 |
| MQ-3 氣體感測器 | 酒精 / 乙醇 ⭐ | 1 | 酒精類揮發物偵測 |
| MQ-9 氣體感測器 | CO / 可燃氣體 | 1 | 一氧化碳類特徵 |
| MQ-135 氣體感測器 | 空氣品質 / 氨氣 | 1 | 氨氣類揮發物 |
| TGS2602 感測器 | VOC / 氨氣 ⭐⭐ | 1 | 核心 VOC 特徵（模型權重最高） |
| TGS2620 感測器 | 酒精 / 有機溶劑 | 1 | 有機溶劑類特徵 |
| BME280 | 溫度 / 濕度 / 氣壓 | 1 | 環境監測 |
| 麵包板 + 杜邦線 | 標準尺寸 | 若干 | 電路連接 |
| USB Type A-B 連接線 | — | 1 | Arduino ↔ Raspberry Pi |
| 電源供應器 | 5V 2.5A | 1 | Raspberry Pi 供電 |

### 硬體接線說明

- **氣體感測器（MQ / TGS）**：類比訊號輸出，連接至 Arduino `A0–A5`（10-bit ADC）
- **BME280**：I²C 協定，連接至 Arduino `SDA / SCL`
- **Arduino → Raspberry Pi**：USB Serial，`/dev/ttyACM0`，115200 baud

---

## 💻 軟體需求

### 開發環境

| 類別 | 規格 |
|------|------|
| Raspberry Pi OS | Debian-based Linux（最新穩定版） |
| Python | 3.7+ |
| Arduino IDE | 1.8+ |
| 版本控制 | Git / GitHub |

### Python 套件安裝

```bash
pip install scikit-learn numpy pandas flask paramiko pyserial
```

或使用 `requirements.txt`：

```bash
pip install -r requirements.txt
```

### 專案檔案結構

```
📦 AN-ELECTRONIC-NOSE-BASED-NO...
│
├── 📁 arduino_mega_data_collection/          # Arduino 資料擷取韌體
│   ├── 🔌 air_baseline_collection.ino        # 空氣基線蒐集程式
│   ├── 🔌 pineapple_sample_collection.ino    # 鳳梨樣本蒐集程式
│   └── 📄 README.md
│
├── 📁 enose_model_training/                  # 模型訓練工作區
│   └── 📁 orkspace/
│       ├── 📁 catboost_info/                 # CatBoost 訓練紀錄
│       ├── 📁 data/                          # 原始量測資料
│       ├── 📁 deploy_rpi_catboost/           # CatBoost 部署版本
│       ├── 📁 deploy_rpi_et_10s/             # ExtraTrees 10s 版本
│       ├── 📁 deploy_rpi_et_10s_nodeay/
│       ├── 📁 deploy_rpi_et_30s_noday/
│       ├── 📁 deploy_rpi_et_30s_noday_s3_sensitive/
│       ├── 📁 deploy_rpi_et_30s_nodeay/
│       ├── 📁 deploy_rpi_student/            # Student 模型部署版本
│       ├── 📁 deploy_rpi_student_short/
│       ├── 📁 models/                        # 儲存的模型檔案
│       ├── 📁 reports/                       # 訓練報表與評估結果
│       ├── {} cutpoints.json                 # 閾值設定
│       ├── 📊 data_label_audit_report.xlsx   # 資料標注稽核報表
│       ├── {} feature_columns.json           # 特徵欄位定義
│       ├── 📓 labeling_perfect_final.ipynb   # 標注最終版 Notebook
│       ├── 📊 labeling.xlsx                  # 標注資料表
│       ├── 📄 model_final.pkl                # 最終訓練模型
│       ├── 🖼️ stage_timeline.png             # 成熟度時間軸圖
│       └── 📄 README.md
│
├── 📁 files(設計文件&簡報)/                  # 設計規格書與簡報
│
├── 📁 pineapple_deployment_system/           # Raspberry Pi 部署系統
│   ├── 🐍 app_local.py                       # Flask 模擬測試介面
│   ├── 🐍 calibrate_air_30s.py              # 空氣基線校正主程式
│   ├── 🐍 inference_30s.py                  # 30 秒窗口推論主程式
│   ├── 🔌 use_this.ino                       # Arduino 搭配推論用韌體
│   └── 📄 README.md
│
├── 🎬 114_2_midterm_demo_video.mov           # 期中成果展示影片
└── 📄 README.md
```

---

## 🚀 使用方式

### Step 1｜硬體連接

1. 依接線圖將六種氣體感測器連接至 Arduino `A0–A5`
2. 將 BME280 以 I²C 接至 Arduino `SDA / SCL`
3. 以 USB 線連接 Arduino Mega 與 Raspberry Pi 3
4. 確認裝置路徑（通常為 `/dev/ttyACM0`）

### Step 2｜部署環境設定（Raspberry Pi）

```bash
# 建立 Python 虛擬環境
python3 -m venv venv
source venv/bin/activate

# 安裝相依套件
pip install -r requirements.txt

# 確認模型檔案與 JSON 設定檔皆在同一目錄
ls deploystudent.pkl featurecolumns.json deploymeta.json
```

### Step 3｜空氣基線校正

> ⚠️ 每次量測前，或環境條件明顯改變時，須重新執行校正

```bash
# 預設暖機 60 秒，校正 30 秒
python3 calibrate_air_30s.py

# 首次使用建議暖機 180 秒
python3 calibrate_air_30s.py --warmup-sec 180
```

校正完成後產生 `airbase.json`，格式如下：

```json
{
  "MQ2":     { "mean": 450.2, "std": 12.3, "min": 420, "max": 480 },
  "MQ3":     { "mean": 380.5, "std": 15.1, "min": 350, "max": 410 },
  "MQ9":     { "mean": 312.0, "std": 10.2, "min": 290, "max": 335 },
  "MQ135":   { "mean": 520.1, "std": 18.4, "min": 490, "max": 555 },
  "TGS2602": { "mean": 280.3, "std":  9.7, "min": 260, "max": 300 },
  "TGS2620": { "mean": 410.6, "std": 13.5, "min": 385, "max": 435 }
}
```

### Step 4｜成熟度推論

```bash
# 將鳳梨置於感測器前約 5 cm，執行推論
python3 inference_30s.py
```

系統累積 30 秒感測窗口後輸出：

```
[推論結果]
Stage 0 機率: 0.03
Stage 1 機率: 0.12
Stage 2 機率: 0.79   ← argmax
Stage 3 機率: 0.06

✅ 最終判定：Stage 2（完熟）— 建議立即食用
```

### Step 5｜Flask 模擬介面（開發 / 測試用）

```bash
python3 applocal.py
# 開啟瀏覽器前往 http://localhost:5000
```

介面功能：
- **Air Baseline 校正**：透過 SSH 遠端觸發 Raspberry Pi 執行校正
- **開始推論**：顯示四階段機率長條圖與最終預測結果
- **狀態訊息列**：顯示目前作業狀態（待機 / 校正中 / 推論中 / 錯誤）

---

## 🤖 機器學習模型

### 訓練流程

```
VOC 量測資料
     │
     ▼
特徵工程（159 項 PID-specific + 7 項 shared-by-date）
     │
     ▼
CatBoost  ──── Teacher Model ────▶  知識蒸餾
                                        │
                                        ▼
                               ExtraTrees  ──── Student Model
                                        │
                                        ▼
                          MI 篩選 11 項核心特徵 → 部署推論
```

### 訓練資料

| 項目 | 數值 |
|------|------|
| 鳳梨數量 | 14 顆（金鑽鳳梨） |
| 每日量測筆數 | 333 筆 per-day features |
| Stage 0 | 32 筆 |
| Stage 1 | 25 筆 |
| Stage 2 | 29 筆 |
| Stage 3 | 21 筆 → **69 筆**（Pseudo Labeling 補強） |

### 11 項部署特徵

| # | 特徵名稱 | 計算方式 |
|---|----------|----------|
| 1 | `MQ3stdnorm` | MQ3 標準差歸一化 |
| 2 | `MQ3rangenorm` | MQ3 範圍歸一化 |
| 3 | `MQ2MQ3ratio` | MQ2 / MQ3 濃度比值 |
| 4 | `MQ3MQ135ratio` | MQ3 / MQ135 濃度比值 |
| 5 | `MQ9slope` | MQ9 濃度變化斜率 |
| 6 | `TGS2602minnorm` | TGS2602 最小值歸一化 ⭐ |
| 7 | `MQ2aucnorm` | MQ2 曲線下面積歸一化 |
| 8 | `MQ2meannorm` | MQ2 均值歸一化 |
| 9 | `MQ9minnorm` | MQ9 最小值歸一化 |
| 10 | `TGS2602stdnorm` | TGS2602 標準差歸一化 ⭐ |
| 11 | `MQ9deltamean` | MQ9 均值變化量 |

### 模型效能指標

| 指標 | 數值 |
|------|------|
| LOGO 交叉驗證平均準確率 | **86.15%** |
| Macro F1 Score | **83.09%** |
| 部署前樣本準確率 | 79.27% |
| 含後處理樣本準確率 | **80.49%** |
| 部署模型大小 | **6.2 MB** |

### 後處理邏輯

```
Guard Baseline
  └─ 若所有感測值接近 airbase.json 基準線（誤差範圍內）
     → 強制輸出 Stage 0（避免空氣被誤判為成熟）

Override 邏輯
  └─ 若模型預測為 Stage 2
     且 TGS2602 / MQ3 特徵超過閾值
     → Override 為 Stage 3（過熟）
```

---

## 📱 App 介面

### 消費端 App

| 頁面 | 檔案 | 功能 |
|------|------|------|
| 首頁 | `Home.tsx` | 開始掃描、中英語言切換 |
| 掃描流程頁 | `ScanProcess.tsx` | 放入引導動畫、30 秒倒數進度條 |
| 結果頁 | `Result.tsx` | 熟度標籤（🟢🟡🟠🔴）、信心值、食用建議 |
| 歷史紀錄 | `History.tsx` | 過往掃描清單（日期 / 熟度 / 信心值） |

### 農民端 App

| 頁面 | 檔案 | 功能 |
|------|------|------|
| 農民首頁 | `FarmerHome.tsx` | 批次概況、統計摘要 |
| 批次建立 | `BatchCreate.tsx` | 建立批次名稱 / 數量 / 日期 |
| 批次掃描 | `BatchScan.tsx` | 連續多顆掃描、剩餘計數器 |
| 批次結果 | `BatchResult.tsx` | 熟度分布餅圖、表格摘要 |
| 報表分析 | `ReportAnalysis.tsx` | PDF / 圖表匯出 |

> 🌐 **雙語支援**：消費端與農民端皆支援中英文即時切換，語系偏好儲存於裝置本地。

---

## ⚠️ 系統限制

| 限制類別 | 說明 |
|----------|------|
| **硬體** | 感測器需暖機 60–180 秒；Raspberry Pi 3 記憶體 1GB，僅適合輕量模型 |
| **演算法** | 部署端不含 `dayratio` 等日期相關特徵；Stage 3 仍受個體差異影響 |
| **應用場景** | 目前僅支援金鑽鳳梨品種；每次單顆量測，不支援批次同時量測 |
| **安全性** | Flask 介面僅限區域網路，尚未實作 HTTPS 與完整身分驗證 |
| **操作環境** | 建議溫度 20–30°C，相對濕度 40–70% |

---

## 🗂️ 版本管理

### 分支策略

```
main        ← 正式穩定版本（可展示 / 可交付）
develop     ← 開發整合版本（功能整合測試）
feature/*   ← 功能開發分支（各模組獨立開發）
```

### API 版本規劃

| 版本 | 功能 |
|------|------|
| `v1` | 鳳梨成熟度四階段分類 |
| `v2` | 預留：多水果支援、批次分析擴充 |

---

## 👥 開發團隊

> 本專題由長庚大學(Chang Gung University)B1144143陳玟妤、B1229062林冠妤、B1229066陳怡禎、B1229068廖文歆共同開發

| 部分 | 細節 |
|------|----------|
| 硬體工程 | 感測器電路設計、Arduino 韌體開發 |
| 模型工程 | 特徵工程、Teacher-Student 模型訓練與部署 |
| 後端工程 | Raspberry Pi 部署、Flask API、SSH 整合 |
| 前端工程 | React Native App（消費端 / 農民端） |

---

## 📄 相關文件

| 文件 | 說明 |
|------|------|
| `設計規格書 v1.0` | 系統完整設計規格，含架構圖、流程圖、介面設計 |
| `featurecolumns.json` | 11 項部署特徵欄位定義 |
| `deploymeta.json` | Stage 映射與模型元資料 |
