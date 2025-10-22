# Robust + ML + TCN + Rule-Based Hybrid (XAUUSD M5)

This repository contains a complete workflow to build a **robust hybrid trading system** for **XAUUSD on the 5-minute timeframe (IC Markets)**.  
It combines **robust preprocessing**, **machine-learning models** (LightGBM, Logistic Regression, TCN), and **rule-based filters** to achieve consistent and execution-aware performance.

---

## 🔷 Project Structure

tcn_scalping_project/
├── data/
│ ├── raw/ # original ICMarkets data
│ └── canonical/ # cleaned + labeled data
├── src/
│ ├── data/clean_pipeline.py
│ ├── labels/triple_barrier.py
│ ├── models/lgbm/train.py
│ └── (more models later)
├── configs/ # YAML configs for features & labels
├── reports/ # audit + evaluation summaries
├── artifacts/ # trained models (.pkl / onnx)
└── requirements.txt


---

## 🧩 Phase 1 – Data & Robust Foundation
**Goal:** Prepare clean and stable M5 data for machine learning.  

**Key steps**
- Audited 5 years of IC Markets XAUUSD M5 data (2021 – 2025).
- Applied **Winsorization, MAD scaling, Hampel filter, Huberized EMA smoothing**.
- Produced canonical dataset and feature spec.

**Outputs**
- `data/canonical/xauusd_m5_cleaned.parquet`
- `configs/features_m5.yaml`
- `reports/DATA_AUDIT.md`
- `reports/CLEAN_SUMMARY.md`

---

## 🤖 Phase 2 – Machine Learning Models
### Step 1  Triple-Barrier Labeling
- Added volatility-scaled upper/lower barriers (+0.5 ATR / –0.35 ATR, 20 bars horizon).  
- Generated realistic 1 / –1 / 0 labels for directional training.

**Outputs**
- `data/canonical/xauusd_m5_labeled.parquet`
- `configs/labels.yaml`
- `reports/LABEL_SUMMARY.md`  
  - Positive 37 %  | Negative 51 %  | Neutral 12 %

### Step 2  LightGBM Baseline (Binary Model)
- Trained on robust features with time-based split (80 / 20).  
- Excluded neutral labels for clean directional signal.  
- Evaluated accuracy and threshold sweep.

**Results (EVAL_REPORT.md)**
| Threshold | Trade Rate | Hit Rate |
|------------|-------------|-----------|
| 0.50       | 1.00        |    0.5149 |
| 0.55       | 0.11        | 0.5303    |
| 0.60       | 0.03        | 0.5421    |

**Artifacts**
- `artifacts/lgbm_model.pkl`
- `reports/EVAL_REPORT.md`

---

## 🔜 Next Phases
- Train Logistic Regression baseline  
- Add Temporal Convolutional Network (TCN)  
- Integrate rule-based filters (spread, volume, EMA slope, BOS/MSS)  
- Combine and optimize for PF ≥ 1.5 and EV > 0  
- Deploy signals to MT5 / cTrader EA and enable drift monitoring

---

## 🧠 Tech Stack
`Python 3.11 | pandas | numpy | pyarrow | lightgbm | scikit-learn | PyYAML`

---

## 🧾 Author
**Amrul Mokhtar**  
Building robust, data-driven trading systems that balance machine learning and execution discipline.
