# src/data/clean_pipeline.py
import os
import yaml
import numpy as np
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "historical_5m.csv")
OUT_PARQUET = os.path.join("data", "canonical", "xauusd_m5_cleaned.parquet")
FEATURE_SPEC = os.path.join("configs", "features_m5.yaml")
REPORT_PATH = os.path.join("reports", "CLEAN_SUMMARY.md")

# ---------- Tunable parameters ----------
P_WINSOR_LOW  = 0.001    # 0.1%
P_WINSOR_HIGH = 0.999    # 99.9%
HAMPEL_WIN    = 25       # 25 bars ~ 2 hours
HAMPEL_K      = 3.0
EMA_ALPHA     = 0.1      # robust EMA smoothing strength
# ---------------------------------------

def winsorize(s: pd.Series, low=P_WINSOR_LOW, high=P_WINSOR_HIGH):
    lo, hi = s.quantile([low, high])
    return s.clip(lower=lo, upper=hi)

def mad_scale(s: pd.Series):
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - med) / (1.4826 * mad)

def hampel_filter(s: pd.Series, win=HAMPEL_WIN, k=HAMPEL_K):
    med = s.rolling(win, center=True, min_periods=1).median()
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(win, center=True, min_periods=1).median()
    thresh = k * 1.4826 * mad
    outlier = abs_dev > thresh
    s_filled = s.copy()
    s_filled[outlier] = med[outlier]
    return s_filled

def robust_ema(s: pd.Series, alpha=EMA_ALPHA):
    # Huber-like weights around a rolling median
    med = s.rolling(25, min_periods=1).median()
    dev = (s - med).abs()
    scale = dev.rolling(25, min_periods=1).median().replace(0, 1e-9)
    r = dev / (2.5 * scale)
    w = 1.0 / (1.0 + r**2)  # heavier penalty for outliers
    # weighted EMA
    y = []
    prev = s.iloc[0]
    for i, (xi, wi) in enumerate(zip(s.values, w.values)):
        a = alpha * wi
        prev = a * xi + (1 - a) * prev
        y.append(prev)
    return pd.Series(y, index=s.index)

def main():
    os.makedirs("data/canonical", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    # detect timestamp column
    ts_col = None
    for c in df.columns:
        if c.strip().lower() in {"timestamp", "time", "datetime", "date"}:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError(f"Could not find a timestamp column in {list(df.columns)}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    df = df.set_index(ts_col)

    # keep core columns
    cols_expected = ["open", "high", "low", "close", "tick_volume", "spread_points", "commission_roundtrip_usd"]
    missing = [c for c in cols_expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # build base features
    out = df[cols_expected].copy()
    out.rename(columns={"tick_volume": "tick_vol"}, inplace=True)

    # simple 1 bar log return on close
    out["ret1"] = np.log(out["close"]).diff()

    # robust transforms on returns and spread
    out["ret1_winsor"]  = winsorize(out["ret1"])
    out["ret1_hampel"]  = hampel_filter(out["ret1_winsor"])
    out["ret1_robust_z"] = mad_scale(out["ret1_hampel"])

    # robust smoothing for spread and a mid price
    mid = (out["high"] + out["low"]) / 2.0
    out["mid_smooth"] = robust_ema(mid)
    out["spread_smooth"] = robust_ema(out["spread_points"].astype(float))

    # drop leading NaNs from diff
    out = out.dropna()

    # write parquet
    out.to_parquet(OUT_PARQUET, index=True)

    # write feature spec yaml
    spec = {
        "dataset": "XAUUSD M5 ICMarkets",
        "paths": {
            "raw_csv": RAW_PATH.replace("\\", "/"),
            "canonical_parquet": OUT_PARQUET.replace("\\", "/")
        },
        "timestamp_col": ts_col,
        "columns": {
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "tick_vol": "int",
            "spread_points": "float",
            "commission_roundtrip_usd": "float",
            "ret1": "log return close[t] - close[t-1]",
            "ret1_winsor": f"Winsorized ret1, p_low={P_WINSOR_LOW}, p_high={P_WINSOR_HIGH}",
            "ret1_hampel": f"Hampel filtered, win={HAMPEL_WIN}, k={HAMPEL_K}",
            "ret1_robust_z": "MAD scaled ret1_hampel",
            "mid_smooth": f"Robust EMA of mid price, alpha={EMA_ALPHA}",
            "spread_smooth": f"Robust EMA of spread_points, alpha={EMA_ALPHA}",
        },
        "notes": [
            "Keep commission_roundtrip_usd for cost modeling in backtests and EV.",
            "Gaps during weekends are not filled, which matches actual market activity."
        ],
    }
    with open(FEATURE_SPEC, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False)

    # short summary
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Clean Summary\n\n")
        f.write(f"- Input rows, {len(df):,}\n")
        f.write(f"- Output rows, {len(out):,}\n")
        f.write(f"- Saved parquet, {OUT_PARQUET}\n")
        f.write(f"- Saved feature spec, {FEATURE_SPEC}\n")

    print("Done. Wrote:")
    print(" -", OUT_PARQUET)
    print(" -", FEATURE_SPEC)
    print(" -", REPORT_PATH)

if __name__ == "__main__":
    main()
