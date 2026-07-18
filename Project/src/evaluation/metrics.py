"""
Evaluation Metrics Module.
Provides functions for computing MAE, RMSE, MAPE, and metrics overview tables.
"""

import numpy as np
import pandas as pd
from typing import Dict

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-5) -> float:
    """Computes Mean Absolute Percentage Error (MAPE) handling division by zero safely."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """Calculates all key metrics and returns dict formatted summary."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 2)
    }

def compare_models(metrics_list: list) -> pd.DataFrame:
    """Constructs sorted comparison dataframe from list of metrics dicts."""
    df_results = pd.DataFrame(metrics_list)
    return df_results.sort_values(by="MAE").reset_index(drop=True)
