"""
Advanced Machine Learning Forecasting Models.
Implements gradient boosting algorithms: XGBoost, LightGBM, and CatBoost.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, params: Dict[str, Any] = None) -> Any:
    """Trains XGBoost Regressor."""
    if xgb is None:
        raise ImportError("XGBoost library is not installed.")
    
    default_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    if params:
        default_params.update(params)
        
    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, params: Dict[str, Any] = None) -> Any:
    """Trains LightGBM Regressor."""
    if lgb is None:
        raise ImportError("LightGBM library is not installed.")
        
    default_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    if params:
        default_params.update(params)
        
    model = lgb.LGBMRegressor(**default_params)
    model.fit(X_train, y_train)
    return model

def train_catboost(X_train: pd.DataFrame, y_train: pd.Series, params: Dict[str, Any] = None) -> Any:
    """Trains CatBoost Regressor."""
    if cb is None:
        raise ImportError("CatBoost library is not installed.")
        
    default_params = {
        'iterations': 300,
        'learning_rate': 0.05,
        'depth': 6,
        'random_seed': 42,
        'verbose': 0
    }
    if params:
        default_params.update(params)
        
    model = cb.CatBoostRegressor(**default_params)
    model.fit(X_train, y_train)
    return model
