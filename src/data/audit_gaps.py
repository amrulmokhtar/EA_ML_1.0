import os, glob
import pandas as pd

RAW_DIR = os.path.join("data", "raw")

# 1) find the raw file
candidates = glob.glob(os.path.join(RAW_DIR, "*.csv")) + glob.glob(os.path.join(RAW_DIR, "*.xlsx"))
assert candidates, f"No CSV or XLSX found in {RAW_DIR}"
RAW_PATH = candidates[0]
ext = os.path.splitext(RAW_PATH)[1].lower()

# 2) read the file
if ext == ".csv":
    df = pd.read_csv(RAW_PATH, engine="python")
else:
    df = pd.read_excel(RAW_PATH)

# 3) detect the timestamp column
possible = ["time", "datetime", "timestamp", "date", "Date", "Time", "Datetime", "Timestamp"]
ts_col = None
for c in df.columns:
    cname = c.strip()
    if cname in possible:
        ts_col = c
        break

if ts_col is None:
    # try any column that parses well to datetimes
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], utc=True, errors="coerce")
            ok_ratio = parsed.notna().mean()
            if ok_ratio > 0.95:
                ts_col = c
                df[c] = parsed
                break
        except Exception:
            pass
    if ts_col is None:
        raise ValueError(f"Could not find a timestamp column. Columns are: {list(df.columns)}")

# 4) normalize times and index
df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
df = df.dropna(subset=[ts_col]).sort_values(ts_col)
df = df.set_index(ts_col)

# 5) build full 5 minute grid and find gaps
full = pd.date_range(df.index[0], df.index[-1], freq="5min", inclusive="both")
missing = full.difference(df.index)

missing_df = pd.DataFrame({"time": missing})
missing_df["dow"] = missing_df["time"].dt.day_name()
missing_df["hour"] = missing_df["time"].dt.hour

print("Missing bars total:", len(missing_df))
print("\nBy day of week:")
print(missing_df.groupby("dow").size().sort_index())
print("\nBy hour (UTC):")
print(missing_df.groupby("hour").size().sort_index())

# 6) save distribution sample
os.makedirs("reports", exist_ok=True)
missing_df.head(5000).to_csv("reports/missing_distribution_sample.csv", index=False)
print("\nSaved reports/missing_distribution_sample.csv")
