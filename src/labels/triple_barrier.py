# src/labels/triple_barrier.py
import os
import yaml
import numpy as np
import pandas as pd

CANON_PATH = os.path.join("data", "canonical", "xauusd_m5_cleaned.parquet")
OUT_PATH   = os.path.join("data", "canonical", "xauusd_m5_labeled.parquet")
CFG_PATH   = os.path.join("configs", "labels.yaml")
REPORT     = os.path.join("reports", "LABEL_SUMMARY.md")

# Tunable params for XAUUSD M5
VERT_HORIZON = 20           # max holding time in bars, 20 bars equals 100 minutes
ATR_WINDOW   = 48           # about 4 hours on M5
TP_K         = 0.50         # take profit = 0.50 * ATR
SL_K         = 0.35         # stop loss  = 0.35 * ATR

def compute_atr(df, n=ATR_WINDOW):
    # classic ATR on mid price, avoids gaps from close to next open
    tr1 = (df["high"] - df["low"]).abs()
    prev_close = df["close"].shift(1)
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=1).mean()
    return atr

def triple_barrier_labels(df, horizon=VERT_HORIZON, tp_k=TP_K, sl_k=SL_K, atr=None):
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    n = len(df)

    if atr is None:
        atr = compute_atr(df)
    atrv = atr.values

    labels = np.zeros(n, dtype=np.int8)     # 1, -1, 0
    t1     = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    hit    = np.full(n, "none", dtype=object)

    for i in range(n - horizon):
        entry = close[i]
        up = entry + tp_k * atrv[i]
        dn = entry - sl_k * atrv[i]
        end = i + horizon

        # scan forward until one barrier hits or horizon ends
        # use intrabar high and low to catch first touch
        sl_idx = None
        tp_idx = None
        # find first index where high >= up
        tp_hits = np.where(high[i+1:end+1] >= up)[0]
        if tp_hits.size > 0:
            tp_idx = i + 1 + tp_hits[0]
        # find first index where low <= dn
        sl_hits = np.where(low[i+1:end+1] <= dn)[0]
        if sl_hits.size > 0:
            sl_idx = i + 1 + sl_hits[0]

        if tp_idx is not None and sl_idx is not None:
            # whichever happened first
            if tp_idx < sl_idx:
                labels[i] = 1
                t1[i] = df.index[tp_idx]
                hit[i] = "tp"
            elif sl_idx < tp_idx:
                labels[i] = -1
                t1[i] = df.index[sl_idx]
                hit[i] = "sl"
            else:
                # same bar, treat as neutral to avoid bias
                labels[i] = 0
                t1[i] = df.index[end]
                hit[i] = "tie"
        elif tp_idx is not None:
            labels[i] = 1
            t1[i] = df.index[tp_idx]
            hit[i] = "tp"
        elif sl_idx is not None:
            labels[i] = -1
            t1[i] = df.index[sl_idx]
            hit[i] = "sl"
        else:
            # vertical barrier
            ret = (close[end] - entry)
            if ret > 0:
                labels[i] = 1
                hit[i] = "vert_up"
            elif ret < 0:
                labels[i] = -1
                hit[i] = "vert_dn"
            else:
                labels[i] = 0
                hit[i] = "vert_flat"
            t1[i] = df.index[end]

    out = df.copy()
    out["atr"] = atr
    out["label"] = labels
    out["t1"] = t1
    out["hit"] = hit
    # drop the last horizon bars where label is undefined
    out = out.iloc[:-VERT_HORIZON]
    return out

def save_config():
    cfg = {
        "method": "triple_barrier",
        "params": {
            "horizon_bars": VERT_HORIZON,
            "atr_window": ATR_WINDOW,
            "tp_k_atr": TP_K,
            "sl_k_atr": SL_K
        },
        "notes": [
            "Barriers scale with ATR to stay volatility aware",
            "Labels are 1 for upper barrier first, -1 for lower barrier first, 0 for tie or flat",
            "commission_roundtrip_usd kept in dataset for later cost modeling"
        ]
    }
    os.makedirs("configs", exist_ok=True)
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def main():
    os.makedirs("data/canonical", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_parquet(CANON_PATH)
    labeled = triple_barrier_labels(df)

    labeled.to_parquet(OUT_PATH)
    save_config()

    # small report
    counts = labeled["label"].value_counts().reindex([1, 0, -1], fill_value=0)
    total = int(counts.sum())
    suppression_pct = 100.0 * counts.get(0, 0) / max(total, 1)

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# Label Summary\n\n")
        f.write(f"- Rows labeled, {total:,}\n")
        f.write(f"- Positives, {int(counts.get(1, 0)):,}\n")
        f.write(f"- Neutrals,  {int(counts.get(0, 0)):,}\n")
        f.write(f"- Negatives, {int(counts.get(-1, 0)):,}\n")
        f.write(f"- Suppression percent, {suppression_pct:.2f}%\n")
        f.write(f"- Output parquet, {OUT_PATH}\n")
        f.write(f"- Config, {CFG_PATH}\n")

    print("Done, labels written")
    print(" -", OUT_PATH)
    print(" -", CFG_PATH)
    print(" -", REPORT)

if __name__ == "__main__":
    main()
