"""
Data loading and preprocessing module.
Handles dataset loading, memory optimization, and split operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.config import TRAIN_PATH, TEST_PATH, DATE_COL, STORE_COL, ITEM_COL, TARGET_COL

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcasts numerical columns to optimize memory utilization."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not pd.api.types.is_datetime64_any_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage optimized from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction).")
    return df

def load_raw_data(train_path: Path = TRAIN_PATH, test_path: Path = TEST_PATH):
    """Loads train and test datasets from raw folder."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    test_df[DATE_COL] = pd.to_datetime(test_df[DATE_COL])
    
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)
    
    return train_df, test_df
