"""
This script computes Mean Absolute Percentage Error (MAPE) at multiple levels:
- Overall MAPE for the entire test set
- MAPE by product
- MAPE by forecast horizon (time period)

Usage:
    python calculate_mape.py

Outputs:
    - outputs/mape_summary.csv              # Overall MAPE for all three models
    - outputs/mape_by_product.csv           # MAPE broken down by product for all models
    - outputs/mape_by_horizon.csv           # MAPE broken down by time period for all models
"""

import os
import pandas as pd
import numpy as np

def calculate_overall_mape(df):
    """Calculate overall MAPE from ape_pct column."""
    return df['ape_pct'].mean()

def calculate_mape_by_product(df):
    """Calculate MAPE grouped by product."""
    return df.groupby('product_name')['ape_pct'].mean().reset_index()

def calculate_mape_by_horizon(df):
    """Calculate MAPE grouped by forecast horizon (time period)."""
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['horizon'] = (df['date'] - df['date'].min()).dt.days + 1
        return df.groupby('horizon')['ape_pct'].mean().reset_index()
    else:
        print("Warning: 'date' column not found. Skipping horizon-based MAPE.")
        return None