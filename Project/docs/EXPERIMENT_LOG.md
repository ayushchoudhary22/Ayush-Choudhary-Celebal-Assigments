# RetailSense AI: MLflow-Style Experimentation & Model Benchmark Log

This document tracks all model iterations, hyperparameter experiments, feature engineering choices, and validation metrics across the development lifecycle of the **RetailSense AI** forecasting platform.

---

## 🏆 Summary of Progressive Improvements

| Phase | Best Model | MAE | RMSE | MAPE (%) | Improvement vs Baseline |
|:---|:---|:---:|:---:|:---:|:---:|
| 1. Naive Benchmark | Historical Average | 10.10 | 13.37 | 20.74% | 0.0% Baseline |
| 2. Parametric Linear | Ridge Regression | 7.73 | 10.24 | 16.70% | -23.5% MAE reduction |
| 3. Tree Ensembles | Random Forest | 6.24 | 8.19 | 13.63% | -38.2% MAE reduction |
| 4. Gradient Boosting | LightGBM Regressor | 5.93 | 7.69 | 13.05% | -41.3% MAE reduction |
| 5. Final Champion | **XGBoost Regressor** | **5.92** | **7.67** | **13.00%** | **-41.4% MAE reduction** |

---

## 🧪 Detailed Experiment Tracking Table (15 Iterations)

| Exp ID | Date | Dataset Ver | Feature Set | Model Algorithm | Key Hyperparameters | Validation Method | MAE ↓ | RMSE ↓ | MAPE (%) ↓ | Train Time | Infer Time | Observations & Action |
|:---:|:---:|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `EXP-001` | 2026-07-01 | v1.0 (Raw) | Baseline (store, item, dayofweek) | Naive Mean | N/A | Out-of-Sample Split | 12.45 | 16.20 | 26.10% | 0.01s | 0.01s | High bias; fails to capture seasonality. |
| `EXP-002` | 2026-07-02 | v1.0 (Raw) | Grouped Store-Item Historical Mean | Historical Average | N/A | Out-of-Sample Split | 10.10 | 13.37 | 20.74% | 0.05s | 0.02s | Establishes non-trivial baseline. |
| `EXP-003` | 2026-07-03 | v1.1 (Feat) | Lags (1,7) + Calendar | Linear Regression | Default (OLS) | Out-of-Sample Split | 8.42 | 10.95 | 17.80% | 0.85s | 0.04s | Linear model captures day-of-week & 7d lag. |
| `EXP-004` | 2026-07-04 | v1.1 (Feat) | Lags (1,7,14,30) + Calendar | Ridge Regression | alpha=1.0 | Out-of-Sample Split | 7.73 | 10.24 | 16.70% | 0.92s | 0.04s | Regularization improves generalizability. |
| `EXP-005` | 2026-07-05 | v1.1 (Feat) | Lags (1,7,14,30) + Calendar | Decision Tree | max_depth=10 | Out-of-Sample Split | 6.53 | 8.62 | 14.30% | 2.10s | 0.05s | Non-linear splits improve error significantly. |
| `EXP-006` | 2026-07-06 | v1.2 (Full) | Lags + Rolling Stats (7,14,30d) | Random Forest | n_estimators=30, max_depth=12 | Out-of-Sample Split | 6.24 | 8.19 | 13.63% | 45.2s | 0.35s | Ensemble variance reduction drops MAE to 6.24. |
| `EXP-007` | 2026-07-07 | v1.2 (Full) | 32 Features (Lags, Rolling, EMA) | LightGBM | n_estimators=100, lr=0.1 | 5-Fold TimeSeriesSplit | 6.08 | 7.89 | 13.25% | 3.50s | 0.08s | Fast training speed; outperforms Random Forest. |
| `EXP-008` | 2026-07-08 | v1.2 (Full) | 32 Features (Lags, Rolling, EMA) | LightGBM | n_estimators=300, lr=0.05, num_leaves=31 | 5-Fold TimeSeriesSplit | 5.93 | 7.69 | 13.05% | 8.20s | 0.12s | Lower learning rate improves fine-grain convergence. |
| `EXP-009` | 2026-07-09 | v1.2 (Full) | 32 Features (Lags, Rolling, EMA) | XGBoost | n_estimators=100, lr=0.1, max_depth=6 | 5-Fold TimeSeriesSplit | 6.01 | 7.78 | 13.15% | 6.80s | 0.09s | Strong regularized trees handle sales spikes well. |
| `EXP-010` | 2026-07-10 | v1.2 (Full) | 32 Features (Lags, Rolling, EMA) | CatBoost | iterations=300, lr=0.05, depth=6 | 5-Fold TimeSeriesSplit | 5.98 | 7.74 | 13.10% | 14.5s | 0.18s | Robust performance on Store/Item categorical signals. |
| `EXP-011` | 2026-07-11 | v1.3 (Agg) | 32 Feats + Target Encoding | LightGBM | n_estimators=300, lr=0.05 | 5-Fold TimeSeriesSplit | 5.95 | 7.71 | 13.08% | 9.10s | 0.13s | Target encoding added slight overfitting risk. |
| `EXP-012` | 2026-07-12 | v1.2 (Full) | 32 Features (Lags, Rolling, EMA) | **XGBoost (Tuned)** | **n_est=300, lr=0.05, depth=6, sub=0.8, col=0.8** | **5-Fold TimeSeriesSplit** | **5.92** | **7.67** | **13.00%** | **12.4s** | **0.15s** | **CHAMPION MODEL: Lowest MAE (5.92) & RMSE (7.67).** |
| `EXP-013` | 2026-07-13 | v1.2 (Full) | 32 Features (Lags, Rolling, EMA) | XGBoost | n_estimators=500, lr=0.02, depth=8 | 5-Fold TimeSeriesSplit | 5.94 | 7.70 | 13.04% | 28.6s | 0.22s | Higher depth caused minor validation overfit. |
| `EXP-014` | 2026-07-14 | v1.2 (Full) | Top-15 SHAP Selected Features | XGBoost | n_estimators=300, lr=0.05 | 5-Fold TimeSeriesSplit | 5.96 | 7.72 | 13.07% | 7.40s | 0.08s | Feature pruning reduced inference time by 46%. |
| `EXP-015` | 2026-07-15 | v1.2 (Full) | Ensemble (XGB + LGB + Cat) | Equal Weight Average (0.33 x 3) | N/A | 5-Fold TimeSeriesSplit | 5.91 | 7.66 | 12.98% | N/A | 0.45s | Slight improvement (0.01 MAE), but adds deployment complexity. |

---

## 📌 Experimentation Conclusions & Deployment Decision

1. **Feature Engineering Impact:** Adding rolling mean and lag features reduced forecasting error by over **35%** relative to raw sales inputs.
2. **Gradient Boosting Supremacy:** XGBoost, LightGBM, and CatBoost all achieved sub-6.0 MAE scores, outperforming Random Forest (6.24 MAE) and Linear Regression (7.73 MAE).
3. **Champion Selection:** **XGBoost (EXP-012)** was selected for production deployment due to its superior out-of-sample stability, minimal residual variance (RMSE 7.67), and optimal balance between inference latency (150ms) and model accuracy.
