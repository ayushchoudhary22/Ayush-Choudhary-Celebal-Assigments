"""
Baseline Forecasting Models.
Implements naive, historical average, linear regression, and decision tree benchmarks.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

class HistoricalAverageModel:
    """Historical average model predicting daily store-item mean sales."""
    def __init__(self):
        self.store_item_means_ = {}
        self.global_mean_ = 0.0

    def fit(self, df: pd.DataFrame, store_col: str = "store", item_col: str = "item", target_col: str = "sales"):
        self.store_item_means_ = df.groupby([store_col, item_col])[target_col].mean().to_dict()
        self.global_mean_ = df[target_col].mean()
        return self

    def predict(self, df: pd.DataFrame, store_col: str = "store", item_col: str = "item") -> np.ndarray:
        preds = []
        for _, row in df.iterrows():
            key = (row[store_col], row[item_col])
            preds.append(self.store_item_means_.get(key, self.global_mean_))
        return np.array(preds, dtype=np.float32)

def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series, max_depth: int = 10, random_state: int = 42):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 50, max_depth: int = 12, random_state: int = 42):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=random_state)
    model.fit(X_train, y_train)
    return model
