"""
Master Pipeline Execution Script for RetailSense AI Platform.
Runs data loading, quality checks, feature engineering, model benchmarking, and prints execution summary.
"""

import sys
from pathlib import Path
from src.data.loader import load_raw_data
from src.data.quality import assess_data_quality
from src.features.build_features import build_all_features
from src.models.baseline import HistoricalAverageModel, train_linear_regression, train_decision_tree
from src.models.advanced import train_lightgbm
from src.evaluation.metrics import evaluate_predictions, compare_models
from src.recommendation.inventory import generate_store_item_recommendations

def run_pipeline():
    print("=================================================================")
    print("::: RetailSense AI: Multi-Horizon Forecasting & Decision Intelligence :::")
    print("=================================================================\n")
    
    print("1. Loading raw Kaggle datasets...")
    train_df, test_df = load_raw_data()
    print(f"   Train Records: {len(train_df):,}, Test Records: {len(test_df):,}")
    
    print("\n2. Executing Data Quality Audit...")
    report = assess_data_quality(train_df)
    print(f"   Continuity Check: {report['continuity_status']}")
    print(f"   Total Time Series: {report['total_series']} (10 Stores x 50 Items)")
    print(f"   Zero Sales Count: {report['zero_sales_count']} ({report['zero_sales_percent']}%)")
    
    print("\n3. Engineering Time Series Features (Calendar, Lags, Rolling, Trends)...")
    sample_df = train_df.iloc[:50000].copy()
    df_feat = build_all_features(sample_df).dropna().reset_index(drop=True)
    print(f"   Feature Matrix Generated: {df_feat.shape[0]:,} rows x {df_feat.shape[1]} columns")
    
    print("\n4. Splitting Train / Out-of-Sample Validation Sets...")
    val_start = '2017-06-01'
    train_split = df_feat[df_feat['date'] < val_start]
    val_split = df_feat[df_feat['date'] >= val_start]
    
    feature_cols = [c for c in df_feat.columns if c not in ['date', 'sales']]
    X_train, y_train = train_split[feature_cols], train_split['sales']
    X_val, y_val = val_split[feature_cols], val_split['sales']
    
    results = []
    
    print("\n5. Benchmarking Models...")
    print("   Evaluating Baseline: Historical Average...")
    hist_model = HistoricalAverageModel().fit(train_split)
    hist_preds = hist_model.predict(val_split)
    results.append(evaluate_predictions(y_val, hist_preds, "Historical Average"))
    
    print("   Evaluating Baseline: Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train)
    lr_preds = lr_model.predict(X_val)
    results.append(evaluate_predictions(y_val, lr_preds, "Linear Regression"))
    
    print("   Evaluating Champion: LightGBM Regressor...")
    lgb_model = train_lightgbm(X_train, y_train)
    lgb_preds = lgb_model.predict(X_val)
    results.append(evaluate_predictions(y_val, lgb_preds, "LightGBM"))
    
    df_comp = compare_models(results)
    print("\n=== Model Benchmark Metrics Summary ===")
    print(df_comp.to_string(index=False))
    
    print("\n6. Running Inventory Decision Intelligence Engine...")
    mock_fcst = val_split.copy()
    mock_fcst['forecast'] = lgb_preds
    df_recs = generate_store_item_recommendations(mock_fcst, train_split)
    print(f"   Recommendations generated for {len(df_recs)} series combinations.")
    print("   Action Status Overview:")
    print(df_recs['Status'].value_counts().to_string())
    
    print("\n=================================================================")
    print("SUCCESS: Pipeline execution completed successfully!")
    print("INFO: Launch Streamlit web app: streamlit run dashboard/app.py")
    print("=================================================================")

if __name__ == "__main__":
    run_pipeline()
