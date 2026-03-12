# PL Predictor — Model Evaluation Report

**Training Date:** 2026-03-12 02:57

**Dataset:** 380 matches, 15 features

**Validation:** 10-fold walk-forward time-series split

## Model Comparison

| Model | Accuracy | Log-Loss | Brier Score |
|---|---|---|---|
| xgboost_primary | 0.5087 | 1.2412 | 0.6860 |
| random_forest | 0.5348 | 1.0054 | 0.6020 |
| logistic_baseline | 0.5174 | 1.1450 | 0.6393 |

## Per-Class Metrics

### xgboost_primary

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Home Win | 0.560 | 0.705 | 0.625 | 112 |
| Draw | 0.387 | 0.231 | 0.289 | 52 |
| Away Win | 0.448 | 0.394 | 0.419 | 66 |

### random_forest

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Home Win | 0.647 | 0.688 | 0.667 | 112 |
| Draw | 0.359 | 0.269 | 0.308 | 52 |
| Away Win | 0.444 | 0.485 | 0.464 | 66 |

### logistic_baseline

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Home Win | 0.670 | 0.580 | 0.622 | 112 |
| Draw | 0.278 | 0.288 | 0.283 | 52 |
| Away Win | 0.494 | 0.591 | 0.538 | 66 |

## Walk-Forward Fold Breakdown

| Fold | Accuracy | Log-Loss | Brier Score | Samples |
|---|---|---|---|---|
| 0 | 0.5217 | 1.3071 | 0.7003 | 23 |
| 1 | 0.6957 | 0.9560 | 0.4831 | 23 |
| 2 | 0.4348 | 1.3746 | 0.7871 | 23 |
| 3 | 0.4348 | 1.2799 | 0.7458 | 23 |
| 4 | 0.4783 | 1.1437 | 0.6914 | 23 |
| 5 | 0.4348 | 1.5147 | 0.7578 | 23 |
| 6 | 0.5652 | 1.1484 | 0.6302 | 23 |
| 7 | 0.6087 | 1.0388 | 0.6309 | 23 |
| 8 | 0.4783 | 1.1699 | 0.6395 | 23 |
| 9 | 0.4348 | 1.4786 | 0.7942 | 23 |
