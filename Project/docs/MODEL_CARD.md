# RetailSense AI: Model Card — Champion XGBoost Regressor

This Model Card follows Google's Model Card framework to provide transparent documentation for the production-deployed **XGBoost Demand Forecasting Model**.

---

## 📌 Model Details

- **Model Name:** RetailSense XGBoost Multi-Horizon Forecasting Regressor
- **Model Version:** `v1.2.0-prod`
- **Model Architecture:** Extreme Gradient Boosting (`XGBRegressor`)
- **Developer / Author:** Ayush Choudhary (Data Science Intern, Celebal Technologies)
- **Organization:** Celebal Technologies
- **Release Date:** July 2026
- **License:** MIT License

---

## 🎯 Intended Use & Business Scope

- **Primary Application:** Multi-horizon daily unit demand forecasting across 500 Store–Item series.
- **Intended Users:** Supply Chain Managers, Procurement Planners, Inventory Operations Lead.
- **Downstream Operations:** Computes Safety Stock ($SS$), Reorder Point ($ROP$), and inventory action triggers (`URGENT REORDER`, `OVERSTOCKED`, `OPTIMAL`).
- **Out-of-Scope Uses:** Financial stock market trading, macroeconomic inflation forecasting, individual customer-level behavioral profiling.

---

## 📊 Training & Evaluation Datasets

- **Source Dataset:** [Kaggle Store-Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview)
- **Training Period:** 2013-01-01 to 2017-09-30 (1,733 days · ~850,000 records)
- **Validation Period:** 2017-10-01 to 2017-12-31 (92 out-of-sample days · ~46,000 records)
- **Test Set Horizon:** 2018-01-01 to 2018-03-31 (90 forecast days · 45,000 records)
- **Total Series Count:** 500 distinct series (10 Stores × 50 Items)

---

## 🔧 Input Features & Engineering Pipeline

The model ingests 32 engineered features derived from raw `(date, store, item, sales)` tuples:

1. **Calendar Features (7):** `year`, `month`, `quarter`, `weekofyear`, `day`, `dayofweek`, `is_weekend`
2. **Autoregressive Lag Features (6):** `sales_lag_1`, `sales_lag_7`, `sales_lag_14`, `sales_lag_30`, `sales_lag_60`, `sales_lag_90`
3. **Rolling Statistics (12):** 7d, 14d, and 30d rolling `mean`, `std`, `min`, `max` (all 1-day shifted)
4. **Trend & Velocity Signals (3):** `ema_7`, `ema_30`, `sales_7d_to_30d_ratio`
5. **Categorical Encodings (4):** `store`, `item`, `store_item_id`, seasonal indicators

---

## 📈 Quantitative Performance & Model Metrics

Performance evaluated on out-of-sample temporal holdout (2017-10-01 to 2017-12-31):

| Metric | Champion XGBoost | LightGBM | CatBoost | Random Forest | Naive Baseline | Target Threshold |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MAE (units/day)** | **5.92** | 5.93 | 5.98 | 6.24 | 10.10 | **< 7.0** |
| **RMSE (units/day)** | **7.67** | 7.69 | 7.74 | 8.19 | 13.37 | **< 9.0** |
| **MAPE (%)** | **13.00%** | 13.05% | 13.10% | 13.63% | 20.74% | **< 15.0%** |
| **R² Score** | **0.912** | 0.911 | 0.909 | 0.895 | 0.720 | **> 0.85** |

---

## 🔍 Interpretability & SHAP Attributions

Based on exact SHAP (`TreeExplainer`) feature attributions:
- **`rolling_mean_7` (SHAP 8.10):** Dominant short-term demand driver.
- **`rolling_mean_14` (SHAP 5.41):** Medium-term baseline demand momentum.
- **`sales_lag_7` (SHAP 4.22):** Autoregressive weekly cycle pattern.
- **`dayofweek` (SHAP 3.88):** Systematic weekend demand uplift (25–30% increase).

---

## ⚠️ Model Limitations & Edge Cases

1. **Extreme Promotional Events:** The current dataset lacks explicit promotional markers; unannounced flash sales may result in temporary under-prediction.
2. **Cold-Start Items:** New products with less than 90 days of sales history cannot compute 90-day lag/rolling features and require default fallback heuristics.
3. **Macroeconomic Shifts:** Unprecedented supply chain disruptions (e.g., global logistics crises) fall outside historical distribution boundaries.

---

## 🛡️ Ethical Considerations & Governance

- **Fairness across Stores:** Model error rates were verified across all 10 store locations; max MAE disparity across stores is < 0.65 units, ensuring no regional allocation bias.
- **No Personal Identifiable Information (PII):** Dataset contains only aggregate unit sales numbers; zero privacy or customer tracking risks.
- **Human-in-the-Loop:** Safety Stock recommendations serve as decision-support alerts; human procurement officers retain final ordering authority.

---

## 🚀 Production Deployment & MLOps Recommendations

- **Retraining Cadence:** Re-fit model monthly with rolling 5-year historical window.
- **Data Drift Monitoring:** Monitor Kolmogorov-Smirnov (KS) test on feature distributions; alert if drift p-value < 0.05.
- **Inference Latency Target:** Batch predictions generated for 500 series in < 250 milliseconds.
