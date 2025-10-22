# src/models/lgbm/train.py
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib

LABELED = os.path.join("data", "canonical", "xauusd_m5_labeled.parquet")
ARTIFACT = os.path.join("artifacts", "lgbm_model.pkl")
REPORT = os.path.join("reports", "EVAL_REPORT.md")

SEED = 42
TEST_SIZE = 0.20
THRESHOLDS = [0.50, 0.55, 0.60]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use only information available at t-1 to avoid leakage.
    """
    out = pd.DataFrame(index=df.index)

    # Base robust features from Phase 1
    out["ret1_robust_z_l1"] = df["ret1_robust_z"].shift(1)
    out["ret1_robust_z_l2"] = df["ret1_robust_z"].shift(2)
    out["ret1_robust_z_l3"] = df["ret1_robust_z"].shift(3)

    out["spread_smooth_l1"] = df["spread_smooth"].shift(1)
    out["spread_smooth_l2"] = df["spread_smooth"].shift(2)

    out["tick_vol_l1"] = df["tick_vol"].shift(1)
    out["tick_vol_ma24"] = df["tick_vol"].rolling(24, min_periods=5).mean().shift(1)
    out["tick_vol_z"] = ((df["tick_vol"] - df["tick_vol"].rolling(96, min_periods=20).median())
                         / (1.4826 * (df["tick_vol"] - df["tick_vol"].rolling(96, min_periods=20).median()).abs().rolling(96, min_periods=20).median())).shift(1)

    # From labeling step
    out["atr_l1"] = df["atr"].shift(1)

    # Optional cost proxy available later
    if "commission_roundtrip_usd" in df.columns:
        out["commission_usd_l1"] = df["commission_roundtrip_usd"].shift(1)

    out = out.dropna()
    return out

def main():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_parquet(LABELED)

    # Binary only, drop neutrals
    df = df[df["label"] != 0].copy()
    df.sort_index(inplace=True)

    X = build_features(df)
    y = df.loc[X.index, "label"].astype(int)

    # Time based split, last 20 percent is test
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LGBMClassifier(
        n_estimators=400,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Probabilities for trade gating later
    proba = model.predict_proba(X_test)  # columns ordered as [-1, 1]
    p_neg = proba[:, 0]
    p_pos = proba[:, 1]
    y_pred = model.predict(X_test)

    # Classification metrics
    cm = confusion_matrix(y_test, y_pred, labels=[1, -1])
    report = classification_report(y_test, y_pred, digits=4)

    # Simple threshold sweep for trade rate and hit rate
    sweep = []
    for thr in THRESHOLDS:
        # go long when p_pos >= thr, go short when p_neg >= thr, otherwise skip
        take_long = p_pos >= thr
        take_short = p_neg >= thr
        side = np.where(take_long & ~take_short, 1,
                        np.where(take_short & ~take_long, -1, 0))

        mask = side != 0
        trades = side[mask]
        realized = y_test[mask].values
        if len(trades) > 0:
            hit = (trades == realized).mean()
        else:
            hit = np.nan
        trade_rate = mask.mean()
        sweep.append({"threshold": thr, "trades": int(mask.sum()),
                      "trade_rate": float(trade_rate), "hit_rate": float(hit)})

    # Save model
    joblib.dump({"model": model, "features": list(X.columns)}, ARTIFACT)

    # Write report
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# LightGBM Evaluation\n\n")
        f.write(f"- Train rows, {len(X_train):,}\n")
        f.write(f"- Test rows, {len(X_test):,}\n\n")
        f.write("## Confusion matrix [rows true 1,-1 | cols pred 1,-1]\n")
        f.write(f"{cm.tolist()}\n\n")
        f.write("## Classification report\n")
        f.write(f"{report}\n")
        f.write("## Threshold sweep\n")
        for row in sweep:
            f.write(f"- thr {row['threshold']:.2f}, trades {row['trades']}, trade_rate {row['trade_rate']:.4f}, hit_rate {row['hit_rate']:.4f}\n")

    print("Done. Wrote:")
    print(" -", ARTIFACT)
    print(" -", REPORT)

if __name__ == "__main__":
    main()
