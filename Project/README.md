<div align="center">

#  RetailSense AI

### Multi-Series Forecasting for Retail Demand Optimization

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Demand%20Forecasting-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview)
[![XGBoost](https://img.shields.io/badge/Champion-XGBoost%20MAE%205.92-orange)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-red)](https://shap.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Enterprise Data Science Internship Project · Celebal Technologies**  
**Author:** Ayush Choudhary &nbsp;|&nbsp; **Domain:** Time Series Forecasting · ML · Explainable AI · Supply Chain Optimization

</div>

---

## 📌 Executive Summary

Retail enterprises manage thousands of fast-moving products across geographically distributed locations. Predicting future consumer demand accurately is vital for maintaining optimal inventory levels, reducing working capital tied up in stock, and maximizing customer satisfaction.

**RetailSense AI** is a production-grade, Fortune 500-level multi-horizon retail demand forecasting platform. It predicts daily unit sales across **500 Store–Item combinations** over a 90-day horizon using 5 years of historical Kaggle retail time-series data. 

Beyond generating point predictions, **RetailSense AI** delivers true business value through:
- **Time Series Cross-Validation:** 5-Fold expanding window `TimeSeriesSplit` to prevent temporal data leakage.
- **Explainable AI (SHAP):** Transparent feature attributions revealing seasonal, lag, and day-of-week demand drivers.
- **Inventory Decision Intelligence Engine:** Automated calculation of **Lead Time Demand ($LTD$)**, **Safety Stock ($SS$)**, **Reorder Point ($ROP$)**, and real-time inventory management triggers (`URGENT REORDER`, `OVERSTOCKED`, `OPTIMAL`).
- **Data-Driven Financial ROI Analysis:** Dynamically quantifies multi-million dollar annual holding cost savings and margin recoveries based on real model MAE outperformance.
- **Interactive Executive Streamlit Dashboard:** Commercial web application for operational exploration and scenario simulation.

---

## 🏆 Model Benchmark Summary

Models were trained on 4+ years of historical records and evaluated out-of-sample on a final 90-day holdout horizon:

| Rank | Model | MAE ↓ | RMSE ↓ | MAPE (%) ↓ | Status |
|:---:|:---|:---:|:---:|:---:|:---:|
| 🥇 **1** | **XGBoost Regressor** | **5.92** | **7.67** | **13.00%** | **Selected Champion** |
| 🥈 2 | LightGBM Regressor | 5.93 | 7.69 | 13.05% | Challenger |
| 🥉 3 | CatBoost Regressor | 5.98 | 7.74 | 13.10% | Challenger |
| 4 | Random Forest | 6.24 | 8.19 | 13.63% | Baseline |
| 5 | Decision Tree | 6.53 | 8.62 | 14.30% | Baseline |
| 6 | Linear Regression | 7.73 | 10.24 | 16.70% | Baseline |
| 7 | Historical Average | 10.10 | 13.37 | 20.74% | Naive Benchmark |

> **XGBoost achieves a 41.4% error reduction** vs. the naive Historical Average baseline (MAE: 5.92 vs. 10.10), translating to significant reductions in overstock waste and stockout events.

---

## 📓 Complete 14-Notebook Deliverable Pipeline

All 14 notebooks are pre-executed with full data-driven outputs, embedded tables, and high-quality visualizations. **They are visible directly on GitHub without running any code:**

| # | Notebook | Focus & Deliverable |
|:---:|:---|:---|
| 01 | [Business Understanding](./notebooks/01_business_understanding.ipynb) | Problem formulation, CRISP-DM alignment, pipeline architecture, business KPIs |
| 02 | [Data Loading & Inspection](./notebooks/02_data_loading.ipynb) | Memory downcasting (62.5% reduction), schema check, date range audit |
| 03 | [Data Quality Assessment](./notebooks/03_data_quality_assessment.ipynb) | Continuity validation across 500 series, missing values, IQR outlier detection |
| 04 | [Exploratory Data Analysis](./notebooks/04_exploratory_data_analysis.ipynb) | Store revenue ranking, day-of-week surge charts, monthly seasonality patterns |
| 05 | [Feature Engineering](./notebooks/05_feature_engineering.ipynb) | 32 engineered features: lags (1–90d), 1-day shifted rolling stats (7–30d), EMA trends |
| 06 | [Baseline Models](./notebooks/06_baseline_models.ipynb) | Naive Historical Average, Linear Regression, Decision Tree, Random Forest |
| 07 | [Time Series Cross Validation](./notebooks/07_time_series_cross_validation.ipynb) | 5-Fold expanding window `TimeSeriesSplit` evaluating temporal model stability |
| 08 | [Advanced ML Models](./notebooks/08_advanced_ml_models.ipynb) | State-of-the-art XGBoost, LightGBM, CatBoost gradient boosting benchmark |
| 09 | [Explainable AI (SHAP)](./notebooks/09_explainable_ai.ipynb) | Global SHAP feature ranking charts & item-level local prediction attributions |
| 10 | [Supply Chain Recommendations](./notebooks/10_supply_chain_recommendations.ipynb) | Safety Stock ($SS$), Reorder Point ($ROP$), inventory status tracking charts |
| 11 | [Business Impact Analysis](./notebooks/11_Business_Impact_Analysis.ipynb) | Data-driven financial ROI modeling, holding cost savings, 5-year value waterfall |
| 12 | [Residual & Error Analysis](./notebooks/12_Error_Analysis.ipynb) | Residual diagnostics, homoscedasticity, Store × Month error intensity heatmap |
| 13 | [Business Recommendation Engine](./notebooks/13_Business_Recommendation_Engine.ipynb) | ABC Pareto analysis (80/15/5 revenue split) & portfolio risk matrix |
| 14 | [Final Dashboard Summary](./notebooks/14_final_forecasting_dashboard.ipynb) | Platform summary, full model evaluation dashboard, and performance highlights |

---

## 📄 Technical Documentation Suite

- 📜 **[Model Card (Google Style)](./docs/MODEL_CARD.md):** Formal Model Card for Champion XGBoost Regressor covering intended use, performance, ethics, and MLOps guidelines.
- 🧪 **[Experiment Tracking Log](./docs/EXPERIMENT_LOG.md):** MLflow-style experiment tracking table documenting progressive model iterations.
- 🏗️ **[Software Architecture Document](./docs/PROJECT_ARCHITECTURE.md):** Enterprise architecture specification with ASCII system diagrams, module responsibilities, and data flow.
- 🗺️ **[Project Roadmap](./docs/project_roadmap.md):** Strategic future roadmap outlining Phase 2 MLOps, Phase 3 Deep Learning, and Phase 4 Enterprise Scaling.

---

## 🛠️ Repository Folder Structure

```text
RetailSense-AI/
│
├── 📂 configs/
│   └── config.yaml           # Centralized project parameters & model hyperconfigs
│
├── 📂 data/
│   ├── raw/                  # Kaggle train.csv, test.csv, sample_submission.csv
│   └── processed/            # Location for engineered 32-feature time series matrices
│
├── 📓 notebooks/              # 14 pre-executed GitHub-renderable Jupyter Notebooks
│
├── 🐍 src/                    # Production Modular Object-Oriented Python Package
│   ├── data/
│   │   ├── loader.py         # Memory downcasting & CSV ingestion
│   │   └── quality.py        # Data quality audit & IQR outlier detection
│   ├── features/
│   │   └── build_features.py # 32-feature time series engineering engine
│   ├── models/
│   │   ├── baseline.py       # Linear Regression & Random Forest baselines
│   │   └── advanced.py       # XGBoost, LightGBM, CatBoost model wrappers
│   ├── evaluation/
│   │   └── metrics.py        # MAE, RMSE, MAPE evaluation suite
│   ├── explainability/
│   │   └── shap_explainer.py # TreeExplainer SHAP attribution pipeline
│   ├── recommendation/
│   │   └── inventory.py      # Safety stock, ROP & decision action engine
│   └── utils/
│       ├── config.py         # Type-safe YAML configuration loader
│       └── logger.py         # Timed rotating file logging system
│
├── 📊 dashboard/
│   └── app.py                # Commercial Executive Streamlit Web Application
│
├── 🧪 tests/
│   └── test_pipeline.py      # Automated Pytest suite (100% passing)
│
├── 📝 docs/                  # Model Card, Experiment Log, Roadmap & Architecture Docs
├── requirements.txt          # Python dependencies manifest
├── main.py                   # Master end-to-end orchestration pipeline
└── README.md                 # Project homepage
```

---

## ⚙️ Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Master Orchestration Pipeline
```bash
python main.py
```
*(Executes data loading, feature engineering, model benchmarking, and inventory recommendations from the command line.)*

### 3. Launch Executive Streamlit Web App
```bash
streamlit run dashboard/app.py
```
*App opens at `http://localhost:8501`*

### 4. Run Automated Pytest Suite
```bash
pytest tests/ -v
```

---

<div align="center">

**Prepared by Ayush Choudhary · Celebal Technologies · Data Science Internship Project**

</div>
