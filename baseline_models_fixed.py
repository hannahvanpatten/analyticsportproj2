# ----------------Imports----------------
import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv("dataset_with_derived_metrics.csv", parse_dates=["date"])


# ---------Define Baseline Models--------
def mape(y_true, y_pred, eps=1e-8):
    # Avoid divide-by-zero; returns percentage (not fraction)
    mask = ~np.isnan(y_true)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def prepare_monthly_df(df, date_col="date", freq="MS"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    # make sure index is start-of-month, and sorted
    df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp()
    return df.sort_values([ "productid", date_col ])
df = prepare_monthly_df(df, date_col="date")

def naive_forecast(df, group_col, date_col, target_col, horizon):
    # For each product, forecast horizon steps as the last observed value
    preds = []
    for product, g in df.groupby(group_col):
        last_val = g[target_col].ffill().iloc[-1]
        # create h forecast rows for the next h months
        last_date = g[date_col].max()
        future_idx = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
        preds.append(pd.DataFrame({
            group_col: product,
            date_col: future_idx,
            'horizon': list(range(1, horizon+1)),
            'predicted_sales': [last_val]*horizon
        }))
    return pd.concat(preds, ignore_index=True)


# ------------Function Calls-------------

# -------------Save Outputs--------------

# ---------------Results-----------------
