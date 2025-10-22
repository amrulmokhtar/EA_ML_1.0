# Logistic Calibrated Evaluation

- Train rows, 208,905
- Calib rows, 29,843
- Test rows, 59,688

## Confusion matrix [rows true 1,-1 | cols pred 1,-1]
[[0, 25074], [0, 34614]]

## Classification report
              precision    recall  f1-score   support

          -1     0.5799    1.0000    0.7341     34614
           1     0.0000    0.0000    0.0000     25074

    accuracy                         0.5799     59688
   macro avg     0.2900    0.5000    0.3671     59688
weighted avg     0.3363    0.5799    0.4257     59688

## Threshold sweep
- thr 0.50, trades 59688, trade_rate 1.0000, hit_rate 0.5799
- thr 0.52, trades 59688, trade_rate 1.0000, hit_rate 0.5799
- thr 0.54, trades 59688, trade_rate 1.0000, hit_rate 0.5799
- thr 0.56, trades 59688, trade_rate 1.0000, hit_rate 0.5799
- thr 0.58, trades 764, trade_rate 0.0128, hit_rate 0.5903
- thr 0.60, trades 350, trade_rate 0.0059, hit_rate 0.6000
