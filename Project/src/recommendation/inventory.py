"""
Inventory Decision Intelligence & Recommendation Engine.
Calculates safety stock, reorder point (ROP), economic order quantities, and automated stock recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.stats import norm
from src.utils.config import DEFAULT_LEAD_TIME_DAYS, DEFAULT_SERVICE_LEVEL, STORE_COL, ITEM_COL, TARGET_COL

def calculate_inventory_metrics(
    forecast_demand_mean: float,
    demand_std: float,
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    current_inventory: float = 0.0
) -> Dict[str, Any]:
    """
    Computes inventory decision intelligence parameters:
    - Expected Lead Time Demand (LTD)
    - Safety Stock (SS) = Z * std_d * sqrt(L)
    - Reorder Point (ROP) = LTD + SS
    - Recommended Target Order Level = ROP + SS
    - Inventory Action Status & Urgency Level
    """
    # Inverse cumulative normal distribution for Z score
    z_score = norm.ppf(service_level)
    
    lead_time_demand = forecast_demand_mean * lead_time_days
    safety_stock = z_score * demand_std * np.sqrt(lead_time_days)
    reorder_point = lead_time_demand + safety_stock
    target_inventory = reorder_point + safety_stock
    
    if current_inventory < reorder_point:
        status = "URGENT REORDER"
        action = f"Place order for {int(np.ceil(target_inventory - current_inventory))} units immediately."
        color = "red"
    elif current_inventory > target_inventory * 1.5:
        status = "OVERSTOCKED"
        action = f"Reduce procurement; excess stock of {int(np.floor(current_inventory - target_inventory))} units."
        color = "orange"
    else:
        status = "OPTIMAL"
        action = "Stock levels are optimal. Maintain current schedule."
        color = "green"
        
    return {
        "forecast_daily_mean": round(forecast_demand_mean, 2),
        "demand_std": round(demand_std, 2),
        "lead_time_days": lead_time_days,
        "service_level_percent": round(service_level * 100, 1),
        "z_score": round(z_score, 3),
        "lead_time_demand": round(lead_time_demand, 2),
        "safety_stock": int(np.ceil(safety_stock)),
        "reorder_point": int(np.ceil(reorder_point)),
        "recommended_target_stock": int(np.ceil(target_inventory)),
        "current_inventory": int(current_inventory),
        "status": status,
        "action": action,
        "color": color
    }

def generate_store_item_recommendations(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
    service_level: float = DEFAULT_SERVICE_LEVEL
) -> pd.DataFrame:
    """Generates store-item level inventory recommendations."""
    recommendations = []
    
    # Calculate historical daily standard deviation per store-item
    hist_std = historical_df.groupby([STORE_COL, ITEM_COL])[TARGET_COL].std().reset_index()
    hist_std.rename(columns={TARGET_COL: "historical_std"}, inplace=True)
    
    # Merge forecast mean and historical std
    forecast_mean = forecast_df.groupby([STORE_COL, ITEM_COL])["forecast"].mean().reset_index()
    merged = pd.merge(forecast_mean, hist_std, on=[STORE_COL, ITEM_COL])
    
    for _, row in merged.iterrows():
        store = int(row[STORE_COL])
        item = int(row[ITEM_COL])
        f_mean = float(row["forecast"])
        h_std = float(row["historical_std"])
        
        # Assume sample current stock scenario for demo
        simulated_current_stock = int(f_mean * lead_time_days * np.random.uniform(0.7, 1.4))
        
        metrics = calculate_inventory_metrics(
            forecast_demand_mean=f_mean,
            demand_std=h_std,
            lead_time_days=lead_time_days,
            service_level=service_level,
            current_inventory=simulated_current_stock
        )
        
        rec_row = {
            "Store": store,
            "Item": item,
            "Daily Forecast Mean": metrics["forecast_daily_mean"],
            "Lead Time Demand": metrics["lead_time_demand"],
            "Safety Stock": metrics["safety_stock"],
            "Reorder Point (ROP)": metrics["reorder_point"],
            "Current Stock": metrics["current_inventory"],
            "Status": metrics["status"],
            "Recommended Action": metrics["action"]
        }
        recommendations.append(rec_row)
        
    return pd.DataFrame(recommendations)
