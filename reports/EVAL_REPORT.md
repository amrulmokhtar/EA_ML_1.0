# LightGBM Evaluation

- Train rows, 238,721
- Test rows, 59,681

## Confusion matrix [rows true 1,-1 | cols pred 1,-1]
[[11006, 14066], [14888, 19721]]

## Classification report
              precision    recall  f1-score   support

          -1     0.5837    0.5698    0.5767     34609
           1     0.4250    0.4390    0.4319     25072

    accuracy                         0.5149     59681
   macro avg     0.5044    0.5044    0.5043     59681
weighted avg     0.5170    0.5149    0.5159     59681

## Threshold sweep
- thr 0.50, trades 59681, trade_rate 1.0000, hit_rate 0.5149
- thr 0.55, trades 6834, trade_rate 0.1145, hit_rate 0.5303
- thr 0.60, trades 2079, trade_rate 0.0348, hit_rate 0.5421
