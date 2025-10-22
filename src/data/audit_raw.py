# src/data/audit_raw.py
import os, sys, textwrap
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "historical_5m.csv")
REPORT_PATH = os.path.join("reports", "DATA_AUDIT.md")

# 1) load CSV, try common time column names
time_cols = ["time", "datetime", "timestamp", "date"]
df = pd.read_csv(RAW_PATH)

found = None
for c in df.columns:
    if c.strip().lower() in time_cols:
        found = c
        break
if found is None:
    print("Could not find a time column, expected one of:", time_cols)
    sys.exit(1)

df[found] = pd.to_datetime(df[found], utc=True, errors="coerce")
df = df.dropna(subset=[found]).sort_values(found).reset_index(drop=True)

# 2) basic cadence check, 5 minute freq
df = df.set_index(found)
expected = pd.date_range(df.index[0], df.index[-1], freq="5min", inclusive="both")
missing = expected.difference(df.index)

# 3) duplicate timestamps
dupes = df.index.duplicated(keep="first").sum()

# 4) write a short report
os.makedirs("reports", exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("# Data Audit, raw M5\n\n")
    f.write(f"- Date range, {df.index[0]} to {df.index[-1]}\n")
    f.write(f"- Bars present, {len(df):,}\n")
    f.write(f"- Expected bars, {len(expected):,}\n")
    f.write(f"- Missing bars, {len(missing):,}\n")
    f.write(f"- Duplicate timestamps, {dupes}\n\n")
    # show first few gaps
    if len(missing) > 0:
        sample = pd.Series(missing[:20]).astype(str).to_list()
        f.write("## Sample missing timestamps, first 20\n")
        for ts in sample:
            f.write(f"- {ts}\n")

print("Audit complete, report at", REPORT_PATH)
