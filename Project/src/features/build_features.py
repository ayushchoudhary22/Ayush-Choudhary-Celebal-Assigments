"""
Feature Engineering Module for Time Series Forecasting.
Generates calendar, lag, rolling window, and trend features for Store-Item series.
"""

import pandas as pd
import numpy as np
from typing import List
from src.utils.config import DATE_COL, STORE_COL, ITEM_COL, TARGET_COL, LAG_DAYS, ROLLING_WINDOWS

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts date/time attributes from datetime column."""
    df = df.copy()
    df['year'] = df[DATE_COL].dt.year.astype(np.int16)
    df['month'] = df[DATE_COL].dt.month.astype(np.int8)
    df['quarter'] = df[DATE_COL].dt.quarter.astype(np.int8)
    df['weekofyear'] = df[DATE_COL].dt.isocalendar().week.astype(np.int8)
    df['day'] = df[DATE_COL].dt.day.astype(np.int8)
    df['dayofweek'] = df[DATE_COL].dt.dayofweek.astype(np.int8)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(np.int8)
    return df

def create_lag_features(df: pd.DataFrame, lags: List[int] = LAG_DAYS, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Generates historical shift lag features per store-item group."""
    df = df.copy()
    df = df.sort_values(by=[STORE_COL, ITEM_COL, DATE_COL])
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby([STORE_COL, ITEM_COL])[target_col].shift(lag).astype(np.float32)
    return df

def create_rolling_features(df: pd.DataFrame, windows: List[int] = ROLLING_WINDOWS, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Generates rolling window statistics (mean, std, min, max) shifted by 1 day to prevent leakage."""
    df = df.copy()
    df = df.sort_values(by=[STORE_COL, ITEM_COL, DATE_COL])
    grouped = df.groupby([STORE_COL, ITEM_COL])[target_col]
    
    for window in windows:
        # Shift 1 day to rely only on strictly historical observations
        shifted = grouped.shift(1)
        df[f'rolling_mean_{window}'] = shifted.groupby([df[STORE_COL], df[ITEM_COL]]).transform(lambda x: x.rolling(window, min_periods=1).mean()).astype(np.float32)
        df[f'rolling_std_{window}'] = shifted.groupby([df[STORE_COL], df[ITEM_COL]]).transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0).astype(np.float32)
        df[f'rolling_min_{window}'] = shifted.groupby([df[STORE_COL], df[ITEM_COL]]).transform(lambda x: x.rolling(window, min_periods=1).min()).astype(np.float32)
        df[f'rolling_max_{window}'] = shifted.groupby([df[STORE_COL], df[ITEM_COL]]).transform(lambda x: x.rolling(window, min_periods=1).max()).astype(np.float32)
    return df

def create_trend_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Computes EMA and percentage change features."""
    df = df.copy()
    df = df.sort_values(by=[STORE_COL, ITEM_COL, DATE_COL])
    shifted = df.groupby([STORE_COL, ITEM_COL])[target_col].shift(1)
    
    # Exponential Moving Average (EMA)
    df['ema_7'] = shifted.groupby([df[STORE_COL], df[ITEM_COL]]).transform(lambda x: x.ewm(span=7, min_periods=1).mean()).astype(np.float32)
    df['ema_30'] = shifted.groupby([df[STORE_COL], df[ITEM_COL]]).transform(lambda x: x.ewm(span=30, min_periods=1).mean()).astype(np.float32)
    
    # Weekly ratio / trend signal
    df['sales_7d_to_30d_ratio'] = (df['rolling_mean_7'] / (df['rolling_mean_30'] + 1e-5)).astype(np.float32)
    return df

def build_all_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Executes feature pipeline and generates complete feature matrix."""
    df = create_calendar_features(df)
    df = create_lag_features(df, target_col=target_col)
    df = create_rolling_features(df, target_col=target_col)
    df = create_trend_features(df, target_col=target_col)
    return df
