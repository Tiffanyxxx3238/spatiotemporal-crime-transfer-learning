import pandas as pd
df = pd.read_csv(r"C:\Users\user\GitHub\model-predict-crime\data\raw\la_raw.csv", usecols=["date_occ"], low_memory=False)
df["date_occ"] = pd.to_datetime(df["date_occ"], errors="coerce")
print("Records:", len(df))
print("Min:", df["date_occ"].min())
print("Max:", df["date_occ"].max())
print(df["date_occ"].dt.year.value_counts().sort_index().to_string())
