"""
Automated Pytest Unit Test Suite for RetailSense AI Package.
Tests data downcasting, quality audit, feature engineering, evaluation metrics,
inventory engine, config loader, and enterprise logger.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Append project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data.loader import reduce_mem_usage
from src.data.quality import assess_data_quality
from src.features.build_features import build_all_features
from src.evaluation.metrics import evaluate_predictions
from src.recommendation.inventory import calculate_inventory_metrics
from src.utils.config import get_config, get_setting, ConfigLoader
from src.utils.logger import setup_logger, get_logger


def test_reduce_mem_usage():
    df = pd.DataFrame({
        "int_col": np.array([1, 2, 3, 4, 5], dtype=np.int64),
        "float_col": np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64),
    })
    res = reduce_mem_usage(df, verbose=False)
    df_opt = res[0] if isinstance(res, tuple) else res
    assert df_opt["int_col"].dtype != np.int64


def test_assess_data_quality():
    dates = pd.date_range("2020-01-01", periods=10)
    df = pd.DataFrame({
        "date": np.tile(dates, 2),
        "store": np.repeat([1, 2], 10),
        "item": np.repeat([1, 1], 10),
        "sales": np.random.randint(10, 50, 20),
    })
    report = assess_data_quality(df)
    assert report["total_records"] == 20
    assert report["missing_values"]["sales"] == 0
    assert report["continuity_status"] == "Pass"


def test_build_all_features():
    dates = pd.date_range("2020-01-01", periods=100)
    df = pd.DataFrame({
        "date": dates,
        "store": 1,
        "item": 1,
        "sales": np.random.randint(10, 50, 100),
    })
    df_feat = build_all_features(df)
    assert "sales_lag_1" in df_feat.columns
    assert "rolling_mean_7" in df_feat.columns
    assert "dayofweek" in df_feat.columns
    assert df_feat.shape[1] >= 20


def test_evaluation_metrics():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    res = evaluate_predictions(y_true, y_pred, "Test Model")
    assert res["Model"] == "Test Model"
    assert res["MAE"] == pytest.approx(2.33, rel=1e-2)
    assert res["RMSE"] == pytest.approx(2.38, rel=1e-2)


def test_inventory_metrics():
    res = calculate_inventory_metrics(
        forecast_demand_mean=50.0,
        demand_std=10.0,
        lead_time_days=7,
        service_level=0.95,
        current_inventory=200.0,
    )
    assert res["safety_stock"] > 0
    assert res["reorder_point"] > 350.0
    assert "status" in res


def test_config_loader():
    config = get_config()
    assert isinstance(config, dict)
    seed = get_setting("project.random_seed", 42)
    assert seed == 42


def test_logger_setup():
    logger = get_logger("test_module")
    assert logger is not None
    logger.info("Test log entry for Pytest verification.")
