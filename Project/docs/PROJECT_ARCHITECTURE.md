# RetailSense AI: Production Software Architecture Document

This document outlines the software engineering architecture, data flow pipelines, module responsibilities, configuration management, and deployment design for the **RetailSense AI** enterprise demand forecasting and inventory decision platform.

---

## 🏗️ High-Level System Architecture Diagram

```text
+-----------------------------------------------------------------------------------+
|                                 DATA INGESTION                                    |
|  Kaggle Raw Datasets (train.csv, test.csv) -> src/data/loader.py (Memory Downcast)|
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            DATA HEALTH AUDIT & QUALITY                            |
|  Continuity Check, Null Audit, IQR Outliers -> src/data/quality.py                |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             FEATURE ENGINEERING PIPELINE                          |
|  Calendar (7), Lags (6), Rolling Stats (12), EMA (3) -> src/features/build_features|
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        MODELING & HYPERPARAMETER BENCHMARK                        |
|  Baseline (LR, RF) & Advanced (XGBoost, LightGBM, CatBoost) -> src/models/advanced|
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        EXPLAINABILITY & EVALUATION ENGINE                         |
|  TreeExplainer SHAP & MAE/RMSE/MAPE Diagnostics -> src/explainability/ & eval/    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       INVENTORY DECISION INTELLIGENCE                             |
|  Safety Stock (SS), Reorder Point (ROP), ABC Class -> src/recommendation/inventory|
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           EXECUTIVE STREAMLIT DASHBOARD                           |
|  Interactive Plotly Graphs, SHAP Explorer, Reorder Matrix -> dashboard/app.py    |
+-----------------------------------------------------------------------------------+
```

---

## 📂 Repository Module Responsibilities

```text
Project Root/
│
├── configs/
│   └── config.yaml           # Centralized system & model hyperparameter config
│
├── src/                      # Production Python Source Package
│   ├── data/
│   │   ├── loader.py         # Memory optimization & CSV loading pipeline
│   │   └── quality.py        # Data quality audit & IQR anomaly detection
│   ├── features/
│   │   └── build_features.py # 32-feature time series engineering engine
│   ├── models/
│   │   ├── baseline.py       # Historical average, Linear Regression, Decision Tree
│   │   └── advanced.py       # XGBoost, LightGBM, CatBoost model wrappers
│   ├── evaluation/
│   │   └── metrics.py        # MAE, RMSE, MAPE evaluation suite
│   ├── explainability/
│   │   └── shap_explainer.py # TreeExplainer SHAP attributions
│   ├── recommendation/
│   │   └── inventory.py      # Safety stock, ROP & decision action engine
│   └── utils/
│       ├── config.py         # Type-safe YAML configuration loader singleton
│       └── logger.py         # Timed rotating file & stream logger
│
├── dashboard/
│   └── app.py                # Executive Streamlit Web Application
│
├── tests/
│   └── test_pipeline.py      # Automated Pytest suite
│
└── main.py                   # Master end-to-end orchestration entrypoint
```

---

## 🔄 Data Pipeline Sequence Diagram

```text
[Raw CSVs] ---> loader.load_raw_data() ---> [Downcasted DataFrame]
                     |
                     v
             quality.assess_data_quality() ---> [Clean Health Report]
                     |
                     v
             build_features.build_all_features() ---> [32-Col Feature Matrix]
                     |
                     v
             advanced.train_xgboost() ---> [Trained Model Weights]
                     |
                     +---> shap_explainer.compute_tree_shap_values() ---> [SHAP Matrix]
                     |
                     +---> inventory.generate_store_item_recommendations() ---> [ROP Matrix]
                     |
                     v
             dashboard.app ---> [Rendered Streamlit UI]
```

---

## ⚡ Key Architectural Components

### 1. Configuration Management (`src/utils/config.py` & `configs/config.yaml`)
- Centralized YAML parameters for paths, model parameters, financial rates, and downcasting options.
- Implements `ConfigLoader` singleton with dot-notation lookup (`get_setting('model_training.xgboost_params.n_estimators')`).

### 2. Enterprise Logging (`src/utils/logger.py`)
- Employs Python `logging.handlers.TimedRotatingFileHandler`.
- Log files stored in `logs/retailsense.log` with daily rotation and 7-day retention.

### 3. Feature Engineering Engine (`src/features/build_features.py`)
- Generates 32 time-series features per (store, item) group.
- All rolling stats and lag features use a **1-day right shift** to strictly prevent lookahead data leakage.

### 4. Supply Chain Recommendation Engine (`src/recommendation/inventory.py`)
- Calculates Lead Time Demand ($LTD = \hat{d} \times L$), Safety Stock ($SS = Z \times \sigma_d \times \sqrt{L}$), Reorder Point ($ROP = LTD + SS$).
- Performs ABC Analysis (Pareto revenue segmentation: A=80%, B=15%, C=5%).

### 5. Streamlit Executive Dashboard (`dashboard/app.py`)
- Built using Streamlit and Plotly Express / Graph Objects.
- Features multi-tab navigation, dynamic store/item filtering, interactive forecasting charts, SHAP interpretability plots, and CSV export capabilities.
