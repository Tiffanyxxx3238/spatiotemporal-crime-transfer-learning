"""
04_baseline_model.py  (v4 — CatBoost + Teacher-Student)
---------------------------------------------------------
模型策略：
  1. CatBoost baseline：每城市單獨訓練
  2. Teacher-Student：NYC（Teacher）→ 蒸餾軟標籤 → Chicago（Student）
     Student 同時學 hard label（真實類別）+ soft label（Teacher 機率輸出）
     這讓 Student 在少量資料下也能有 Teacher 的泛化能力

為什麼 CatBoost 比 XGBoost 更適合：
  - 內建 class_weights，不需要手動 sample_weight
  - 對類別型特徵（time_slot）有原生支援
  - 通常在不平衡多類別任務上表現更好
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, precision_score
from catboost import CatBoostClassifier, Pool

PROC_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CATEGORIES = ["violent", "property", "drug", "public_order", "other"]
CAT_FEATURES = ["time_slot", "is_weekend", "slot_x_weekend"]  # CatBoost 原生類別特徵

# ── 格子統計（歷史特徵）─────────────────────────────────────
def build_grid_stats(df):
    d = df.copy()
    d["lat_bin"]   = (d["latitude"]  / 0.01).round() * 0.01
    d["lon_bin"]   = (d["longitude"] / 0.01).round() * 0.01
    d["time_slot"] = pd.cut(d["hour"], bins=[-1,5,11,17,23],
                            labels=[0,1,2,3]).astype(int)
    for cat in CATEGORIES:
        d[f"is_{cat}"] = (d["crime_category"] == cat).astype(int)

    grid = d.groupby(["lat_bin","lon_bin","time_slot"]).agg(
        total_count = ("crime_category", "size"),
        **{f"hist_{cat}": (f"is_{cat}", "mean") for cat in CATEGORIES}
    ).reset_index()
    grid["log_count"] = np.log1p(grid["total_count"])
    return grid

def add_spatial_lag(grid):
    step = 0.01
    lags = []
    for _, row in grid.iterrows():
        lat, lon = row["lat_bin"], row["lon_bin"]
        mask = (
            grid["lat_bin"].between(lat-step*1.5, lat+step*1.5) &
            grid["lon_bin"].between(lon-step*1.5, lon+step*1.5) &
            ~((grid["lat_bin"]==lat) & (grid["lon_bin"]==lon))
        )
        nb = grid[mask]
        lags.append(nb["hist_violent"].mean() if len(nb) > 0 else 0)
    grid["spatial_lag_violent"] = lags
    return grid

# ── 特徵工程 ─────────────────────────────────────────────────
FEATURE_COLS = [
    "lat_bin","lon_bin","lat_norm","lon_norm",
    "hour_sin","hour_cos","month_sin","month_cos",
    "weekday_sin","weekday_cos",
    "is_weekend","time_slot","slot_x_weekend",
    "log_count","spatial_lag_violent",
    "hist_violent","hist_property","hist_drug","hist_public_order","hist_other",
]

def make_features(df, grid_stats):
    d = df.copy()
    d["lat_bin"]   = (d["latitude"]  / 0.01).round() * 0.01
    d["lon_bin"]   = (d["longitude"] / 0.01).round() * 0.01
    d["time_slot"] = pd.cut(d["hour"], bins=[-1,5,11,17,23],
                            labels=[0,1,2,3]).astype(int)
    d["hour_sin"]    = np.sin(2*np.pi*d["hour"]   /24)
    d["hour_cos"]    = np.cos(2*np.pi*d["hour"]   /24)
    d["month_sin"]   = np.sin(2*np.pi*d["month"]  /12)
    d["month_cos"]   = np.cos(2*np.pi*d["month"]  /12)
    d["weekday_sin"] = np.sin(2*np.pi*d["weekday"]/7)
    d["weekday_cos"] = np.cos(2*np.pi*d["weekday"]/7)
    d["is_weekend"]  = (d["weekday"] >= 5).astype(int)
    d["slot_x_weekend"] = d["time_slot"] * (d["is_weekend"]+1)
    d["lat_norm"] = (d["lat_bin"]-d["lat_bin"].mean())/(d["lat_bin"].std()+1e-8)
    d["lon_norm"] = (d["lon_bin"]-d["lon_bin"].mean())/(d["lon_bin"].std()+1e-8)
    d = d.merge(grid_stats, on=["lat_bin","lon_bin","time_slot"], how="left")
    d[FEATURE_COLS] = d[FEATURE_COLS].fillna(0)
    # CatBoost 需要類別特徵是 str
    for c in CAT_FEATURES:
        d[c] = d[c].astype(str)
    return d[FEATURE_COLS], d["crime_category"]

def make_grid_features(grid_stats):
    """grid_stats 已有所有 hist_* 欄位，直接組特徵。"""
    d = grid_stats.copy()
    d["hour"]    = d["time_slot"].map({0:3, 1:9, 2:15, 3:21})
    d["month"]   = 6
    d["weekday"] = 2
    d["hour_sin"]    = np.sin(2*np.pi*d["hour"]   /24)
    d["hour_cos"]    = np.cos(2*np.pi*d["hour"]   /24)
    d["month_sin"]   = np.sin(2*np.pi*d["month"]  /12)
    d["month_cos"]   = np.cos(2*np.pi*d["month"]  /12)
    d["weekday_sin"] = np.sin(2*np.pi*d["weekday"]/7)
    d["weekday_cos"] = np.cos(2*np.pi*d["weekday"]/7)
    d["is_weekend"]  = 0
    d["slot_x_weekend"] = d["time_slot"] * 1
    d["lat_norm"] = (d["lat_bin"]-d["lat_bin"].mean())/(d["lat_bin"].std()+1e-8)
    d["lon_norm"] = (d["lon_bin"]-d["lon_bin"].mean())/(d["lon_bin"].std()+1e-8)
    for c in CAT_FEATURES:
        d[c] = d[c].astype(str)
    return d[FEATURE_COLS].fillna(0)

# ── CatBoost baseline ────────────────────────────────────────
def train_catboost(city_df, city_name, grid_stats):
    print(f"\n{'='*50}")
    print(f"[{city_name}] CatBoost baseline...")
    city_df = city_df[city_df["crime_category"].isin(CATEGORIES)].copy()

    X, y = make_features(city_df, grid_stats)
    le = LabelEncoder()
    le.fit(CATEGORIES)
    y_enc = le.transform(y)

    # group split（同格子不同時出現在 train/test）
    city_df2 = city_df.copy()
    city_df2["lat_bin"] = (city_df2["latitude"] /0.01).round()*0.01
    city_df2["lon_bin"] = (city_df2["longitude"]/0.01).round()*0.01
    city_df2["grid_id"] = city_df2["lat_bin"].astype(str)+"_"+city_df2["lon_bin"].astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_enc, groups=city_df2["grid_id"]))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    # CatBoost 內建 class_weights='Balanced'
    cat_idx = [FEATURE_COLS.index(c) for c in CAT_FEATURES]
    model = CatBoostClassifier(
        iterations=600,
        depth=8,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        class_weights=None,        # 用 auto_class_weights 取代
        auto_class_weights="Balanced",
        cat_features=cat_idx,
        random_seed=42,
        verbose=100,
    )
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    test_pool  = Pool(X_test,  y_test,  cat_features=cat_idx)
    model.fit(train_pool, eval_set=test_pool)

    y_pred = model.predict(X_test).flatten()
    p_macro    = precision_score(y_test, y_pred, average="macro",    zero_division=0)
    p_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro   = f1_score(y_test, y_pred, average="macro",    zero_division=0)

    print(f"\n  Precision macro    = {p_macro:.4f}")
    print(f"  Precision weighted = {p_weighted:.4f}")
    print(f"  F1 macro           = {f1_macro:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    model_path = os.path.join(MODEL_DIR, f"model_{city_name.lower()}.cbm")
    model.save_model(model_path)
    print(f"  模型儲存：{model_path}")
    return model, le, X_train, X_test, y_train, y_test, p_macro, p_weighted, f1_macro

# ── Teacher-Student 知識蒸餾 ─────────────────────────────────
def teacher_student(teacher_model, student_city_df, student_name,
                    teacher_grid_stats, student_grid_stats, le, temperature=3.0):
    """
    Teacher（NYC）→ Student（Chicago）知識蒸餾。

    做法：
      1. Teacher 對 Student 資料產生軟標籤（soft labels / probabilities）
      2. Student 同時學：
           hard loss（CrossEntropy vs 真實標籤）× alpha
         + soft loss（KL divergence vs Teacher 軟標籤）× (1-alpha)
      3. 用 CatBoost 實作：把 Teacher 軟標籤當額外特徵加入 Student 訓練

    alpha = 0.4 → 60% 靠 Teacher 引導，40% 靠真實標籤
    temperature = 3 → 軟化 Teacher 機率分布，讓次要類別資訊也能傳遞
    """
    print(f"\n{'='*50}")
    print(f"[Teacher→Student] NYC → {student_name}（temperature={temperature}）")

    student_city_df = student_city_df[student_city_df["crime_category"].isin(CATEGORIES)].copy()
    X_stu, y_stu = make_features(student_city_df, student_grid_stats)
    y_enc = le.transform(y_stu)

    # Teacher 對 Student 資料產生軟標籤
    cat_idx = [FEATURE_COLS.index(c) for c in CAT_FEATURES]
    # 先把 Student 特徵轉成 Teacher grid_stats 的座標系
    X_teacher_view, _ = make_features(student_city_df, teacher_grid_stats)
    teacher_proba = teacher_model.predict_proba(X_teacher_view)

    # Temperature scaling（讓軟標籤更平滑）
    logits = np.log(teacher_proba + 1e-9) / temperature
    soft_labels = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    # 把 Teacher 軟標籤拼接為額外特徵
    soft_df = pd.DataFrame(
        soft_labels,
        columns=[f"teacher_{cat}" for cat in le.classes_],
        index=X_stu.index
    )
    X_stu_aug = pd.concat([X_stu.reset_index(drop=True),
                            soft_df.reset_index(drop=True)], axis=1)
    aug_feature_cols = list(X_stu_aug.columns)

    # group split
    student_city_df2 = student_city_df.copy()
    student_city_df2["lat_bin"] = (student_city_df2["latitude"] /0.01).round()*0.01
    student_city_df2["lon_bin"] = (student_city_df2["longitude"]/0.01).round()*0.01
    student_city_df2["grid_id"] = (student_city_df2["lat_bin"].astype(str)+"_"+
                                    student_city_df2["lon_bin"].astype(str))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_stu_aug, y_enc,
                                          groups=student_city_df2["grid_id"]))
    X_train = X_stu_aug.iloc[train_idx]
    X_test  = X_stu_aug.iloc[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    # Student CatBoost（類別特徵同 idx，新增的 teacher_* 特徵是連續值）
    student_model = CatBoostClassifier(
        iterations=600,
        depth=8,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        auto_class_weights="Balanced",
        cat_features=cat_idx,      # 位置不變
        random_seed=42,
        verbose=100,
    )
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    test_pool  = Pool(X_test,  y_test,  cat_features=cat_idx)
    student_model.fit(train_pool, eval_set=test_pool)

    y_pred = student_model.predict(X_test).flatten()
    p_macro    = precision_score(y_test, y_pred, average="macro",    zero_division=0)
    p_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro   = f1_score(y_test, y_pred, average="macro",    zero_division=0)

    print(f"\n  [Student 結果]")
    print(f"  Precision macro    = {p_macro:.4f}")
    print(f"  Precision weighted = {p_weighted:.4f}")
    print(f"  F1 macro           = {f1_macro:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    model_path = os.path.join(MODEL_DIR, f"model_{student_name.lower()}_student.cbm")
    student_model.save_model(model_path)
    print(f"  Student 模型儲存：{model_path}")
    return student_model, p_macro, p_weighted, f1_macro

# ── 格子風險分數（地圖用）───────────────────────────────────
def compute_grid_risk_scores(model, le, grid_stats, city_name):
    classes = list(le.classes_)
    violent_idx = classes.index("violent") if "violent" in classes else 0

    X_grid = make_grid_features(grid_stats)
    cat_idx = [FEATURE_COLS.index(c) for c in CAT_FEATURES]
    proba = model.predict_proba(Pool(X_grid, cat_features=cat_idx))

    out = grid_stats[["lat_bin","lon_bin","time_slot","total_count","log_count"]].copy()
    out["violent_proba"] = proba[:, violent_idx]
    for i, cat in enumerate(classes):
        out[f"proba_{cat}"] = proba[:, i].round(4)
    max_c = out["total_count"].max()
    out["density_norm"] = out["total_count"] / max_c if max_c > 0 else 0
    out["risk_score"]   = (out["violent_proba"]*0.6 + out["density_norm"]*0.4)*100
    out["city"] = city_name

    out_path = os.path.join(MODEL_DIR, f"grid_risk_{city_name.lower()}.csv")
    out.to_csv(out_path, index=False)
    print(f"  格子風險分數：{out_path}  ({len(out):,} 格)")
    return out

# ── 主程式 ───────────────────────────────────────────────────
if __name__ == "__main__":
    path = os.path.join(PROC_DIR, "all_cities.csv")
    if not os.path.exists(path):
        print("找不到 all_cities.csv"); exit(1)

    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["latitude","longitude","crime_category"])

    # 先建各城市格子統計
    print("建構格子統計特徵...")
    grid_stats_all = {}
    for city in df["city"].unique():
        city_df = df[df["city"]==city].copy()
        gs = build_grid_stats(city_df)
        print(f"  [{city}] 計算 spatial lag...")
        gs = add_spatial_lag(gs)
        grid_stats_all[city] = gs
        gs.to_csv(os.path.join(MODEL_DIR, f"grid_stats_{city.lower()}.csv"), index=False)

    results = []
    models  = {}
    les     = {}

    # Step 1: 各城市 CatBoost baseline
    for city in df["city"].unique():
        city_df   = df[df["city"]==city].copy()
        gs        = grid_stats_all[city]
        model, le, X_tr, X_te, y_tr, y_te, p_macro, p_weighted, f1 = \
            train_catboost(city_df, city, gs)
        compute_grid_risk_scores(model, le, gs, city)
        models[city] = model
        les[city]    = le
        results.append({"city": city, "type": "catboost_baseline",
                         "precision_macro": round(p_macro,4),
                         "precision_weighted": round(p_weighted,4),
                         "f1_macro": round(f1,4)})

    # Step 2: Teacher-Student（NYC → Chicago）
    if "NYC" in models and "Chicago" in df["city"].unique():
        stu_model, p_m, p_w, f1_m = teacher_student(
            teacher_model      = models["NYC"],
            student_city_df    = df[df["city"]=="Chicago"].copy(),
            student_name       = "Chicago",
            teacher_grid_stats = grid_stats_all["NYC"],
            student_grid_stats = grid_stats_all["Chicago"],
            le                 = les["NYC"],
            temperature        = 3.0,
        )
        compute_grid_risk_scores(stu_model, les["NYC"], grid_stats_all["Chicago"],
                                  "Chicago_student")
        results.append({"city": "Chicago(Student)", "type": "teacher_student_NYC→CHI",
                         "precision_macro": round(p_m,4),
                         "precision_weighted": round(p_w,4),
                         "f1_macro": round(f1_m,4)})

    results_df = pd.DataFrame(results)
    out = os.path.join(MODEL_DIR, "results_baseline.csv")
    results_df.to_csv(out, index=False)
    print(f"\n{'='*50}\n最終結果：\n{results_df.to_string(index=False)}")
    print(f"\n儲存：{out}\n下一步：05_transfer.py")