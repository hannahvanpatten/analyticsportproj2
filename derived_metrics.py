"""
Safe, forecast-aware derived metrics generator.

- All rolling / aggregate metrics are computed using only historical data by shifting measurements before aggregation:
    feature_t = f( data_{<= t-1} )

- Usage:
    from derived_metrics import add_derived_metrics
    df = pd.read_csv("final_dataset.csv", parse_dates=["date"]) 
    df = add_derived_metrics(df)
    df.to_csv("dataset_with_derived_metrics.csv", index=False)

- The function is defensive: it requires 'productid', 'date', and 'sales_units' at minimum.
"""

from typing import Optional
import pandas as pd
import numpy as np

def add_derived_metrics(df: pd.DataFrame, date_col: str = "date", group_col: str = "productid") -> pd.DataFrame:
    """
    Add forecast-safe derived metrics to a monthly product-level DataFrame.
    Requirements:
      - df must contain: group_col, date_col, 'sales_units' and preferably 'inventory_level'

    Key safety rule:
      - For any rolling feature (mean, sum, etc.) we compute the rolling on a shifted series
        so the value at time t only uses observations from t-1 and earlier.

    Returns:
      - A new DataFrame (copy) with derived columns appended.
    """
    df = df.copy()
    # Basic checks
    if group_col not in df.columns:
        raise KeyError(f"Missing required column: {group_col}")
    if date_col not in df.columns:
        raise KeyError(f"Missing required column: {date_col}")
    if "sales_units" not in df.columns:
        raise KeyError("Missing required column: sales_units")

    # Ensure date is datetime and sort
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([group_col, date_col]).reset_index(drop=True)

    # We'll compute features groupwise and assign back
    derived_list = []
    for pid, g in df.groupby(group_col, sort=False):
        g = g.sort_values(date_col).copy()

        # Shift sales so any aggregations only look at past values
        sales = g["sales_units"]
        sales_lag1 = sales.shift(1)                 # sales at t-1
        sales_lag12 = sales.shift(12)               # sales at t-12 (seasonal lag)

        # Rolling features computed on shifted sales (so window ending at t-1)
        sales_roll_3 = sales_lag1.rolling(window=3, min_periods=1).mean()
        sales_roll_6 = sales_lag1.rolling(window=6, min_periods=1).mean()
        sales_roll_12 = sales_lag1.rolling(window=12, min_periods=3).mean()  # require more history
        sales_roll_sum_12 = sales_lag1.rolling(window=12, min_periods=3).sum()

        # Inventory-based features (if present); shift inventory before aggregating
        if "inventory_level" in g.columns:
            inv = g["inventory_level"]
            inv_lag1 = inv.shift(1)
            inv_roll_12_mean = inv_lag1.rolling(window=12, min_periods=3).mean()
        else:
            inv_lag1 = pd.Series(np.nan, index=g.index)
            inv_roll_12_mean = pd.Series(np.nan, index=g.index)

        # Rate / ratio features
        # annual_sales: 12-month historical sum (ending at t-1)
        annual_sales = sales_roll_sum_12

        # simple demand volatility (std dev of last 12 months on shifted sales)
        sales_roll_std_12 = sales_lag1.rolling(window=12, min_periods=3).std()

        # month-over-month change (using lag1)
        mom_change = (sales_lag1 - sales_lag12) / np.where(np.abs(sales_lag12) < 1e-8, np.nan, sales_lag12)

        # put derived features into the group's frame
        g["_sales_lag_1"] = sales_lag1
        g["_sales_lag_12"] = sales_lag12
        g["_sales_roll_mean_3"] = sales_roll_3
        g["_sales_roll_mean_6"] = sales_roll_6
        g["_sales_roll_mean_12"] = sales_roll_12
        g["_sales_roll_sum_12"] = sales_roll_sum_12
        g["_sales_roll_std_12"] = sales_roll_std_12
        g["_inventory_lag_1"] = inv_lag1
        g["_inventory_roll_mean_12"] = inv_roll_12_mean
        g["_annual_sales"] = annual_sales
        g["_mom_change"] = mom_change

        derived_list.append(g)

    df_out = pd.concat(derived_list, axis=0).sort_index()
    # Optionally fill NaNs for newly created features with e.g., zeros or leave as NaN for modeling decisions.
    # Do not fill here automatically; leave NaN so modeler chooses imputation approach.
    return df_out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute forecast-safe derived metrics for final_dataset.csv")
    parser.add_argument("--input", "-i", default="final_dataset.csv", help="Input CSV with columns productid,date,sales_units")
    parser.add_argument("--output", "-o", default="dataset_with_derived_metrics.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    df_in = pd.read_csv(args.input, parse_dates=["date"]) 
    print("Computing features (this may take a moment)...")
    df_out = add_derived_metrics(df_in)
    print(f"Writing {args.output} ...")
    df_out.to_csv(args.output, index=False)
    print("Done.")
