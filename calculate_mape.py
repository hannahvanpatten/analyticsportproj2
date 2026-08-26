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

def main():
    # Define paths for baseline, LightGBM, and RandomForest predictions
    baseline_pred_path = "baseline_seasonal_predictions.csv"  # Using seasonal baseline as example
    lgbm_pred_path = "outputs/predictions.csv"
    rf_pred_path = "outputs/predictions_rf.csv"


    # Load predictions
    print("Loading prediction files...")
    try:
        baseline_df = pd.read_csv(baseline_pred_path)
        lgbm_df = pd.read_csv(lgbm_pred_path)
        rf_df = pd.read_csv(rf_pred_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Ensure ape_pct column exists; calculate if needed
    for name, df in [("Baseline", baseline_df), ("LightGBM", lgbm_df), ("RandomForest", rf_df)]:
        if 'ape_pct' not in df.columns:
            print(f"Calculating ape_pct for {name}...")
            df['ape_pct'] = np.where(
                np.abs(df['actual_sales']) > 1e-8,
                np.abs((df['actual_sales'] - df['predicted_sales']) / df['actual_sales']) * 100.0,
                np.nan
            )

    # Calculate overall MAPE
    print("\nCalculating overall MAPE...")
    mape_summary = pd.DataFrame({
        'model': ['Baseline', 'LightGBM', 'RandomForest'],
        'overall_mape': [
            calculate_overall_mape(baseline_df),
            calculate_overall_mape(lgbm_df),
            calculate_overall_mape(rf_df)
        ]
    })


    print("\nOverall MAPE Summary:")
    print(mape_summary)