"""
Data Quality Assessment module.
Performs data integrity checks, continuity verification, zero-sales inspection, and outlier detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from src.utils.config import DATE_COL, STORE_COL, ITEM_COL, TARGET_COL

def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates a comprehensive data health audit report dict."""
    total_records = len(df)
    missing_vals = df.isnull().sum().to_dict()
    duplicate_rows = int(df.duplicated(subset=[DATE_COL, STORE_COL, ITEM_COL]).sum())
    
    unique_stores = int(df[STORE_COL].nunique())
    unique_items = int(df[ITEM_COL].nunique())
    total_series = unique_stores * unique_items
    
    date_min = df[DATE_COL].min()
    date_max = df[DATE_COL].max()
    total_days = (date_max - date_min).days + 1
    expected_records = total_series * total_days
    
    continuity_status = "Pass" if total_records == expected_records else "Gap Detected"
    zero_sales_count = int((df[TARGET_COL] == 0).sum()) if TARGET_COL in df.columns else 0
    
    report = {
        "total_records": total_records,
        "expected_records": expected_records,
        "missing_values": missing_vals,
        "duplicate_records": duplicate_rows,
        "unique_stores": unique_stores,
        "unique_items": unique_items,
        "total_series": total_series,
        "date_range": (str(date_min.date()), str(date_max.date())),
        "total_days": total_days,
        "continuity_status": continuity_status,
        "zero_sales_count": zero_sales_count,
        "zero_sales_percent": round(zero_sales_count / total_records * 100, 2) if total_records > 0 else 0.0
    }
    return report

def detect_outliers_iqr(df: pd.DataFrame, col: str = TARGET_COL, factor: float = 1.5) -> pd.DataFrame:
    """Detects rows exceeding upper IQR bounds per store-item combination."""
    q1 = df.groupby([STORE_COL, ITEM_COL])[col].transform(lambda x: x.quantile(0.25))
    q3 = df.groupby([STORE_COL, ITEM_COL])[col].transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1
    upper_bound = q3 + factor * iqr
    outliers = df[df[col] > upper_bound]
    return outliers
