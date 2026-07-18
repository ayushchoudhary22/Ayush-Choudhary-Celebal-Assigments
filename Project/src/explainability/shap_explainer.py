"""
Explainable AI (SHAP) Module.
Calculates SHAP values for global feature importance and local instance explanations.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

try:
    import shap
except ImportError:
    shap = None

def compute_tree_shap_values(model, X_sample: pd.DataFrame) -> Tuple[Any, Any]:
    """Computes TreeExplainer SHAP values for tree-based models."""
    if shap is None:
        raise ImportError("SHAP package is not installed.")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values

def get_feature_importance_df(shap_values, feature_names: list) -> pd.DataFrame:
    """Extracts global mean absolute SHAP feature importances."""
    if hasattr(shap_values, "values"):
        vals = np.abs(shap_values.values).mean(axis=0)
    else:
        vals = np.abs(shap_values).mean(axis=0)
        
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": vals
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    return df_importance
