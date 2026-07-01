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

def seasonal_naive_forecast(df, group_col, date_col, target_col, horizon, period=12):
    preds = []
    df = df.set_index(date_col)
    for product, g in df.groupby(group_col):
        g = g.sort_index()
        last_date = g.index.max()
        future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
        y_hat = [] # predicted sales
        for fd in future_dates:
            # same month last season: fd - period months
            ref_date = (fd - pd.DateOffset(months=period)).to_period("M").to_timestamp()
            val = g[target_col].get(ref_date, np.nan)
            y_hat.append(val)
        preds.append(pd.DataFrame({group_col: product, date_col: future_dates, 'horizon': list(range(1, horizon+1)), 'predicted_sales': y_hat}))
    return pd.concat(preds, ignore_index=True)

def moving_average_forecast(df, group_col, date_col, target_col, horizon, window=3): # 3-month average
    preds = []
    df = df.set_index(date_col)
    for product, g in df.groupby(group_col):
        g = g.sort_index()
        last_window_mean = g[target_col].tail(window).mean()
        last_date = g.index.max()
        future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
        preds.append(pd.DataFrame({group_col: product, date_col: future_dates, 'horizon': list(range(1, horizon+1)), 'predicted_sales': [last_window_mean]*horizon}))
    return pd.concat(preds, ignore_index=True)

def fixed_train_test_forecast(df, train_start_date, train_end_date, test_start_date, test_end_date, group_col, date_col, target_col, forecast_fn, horizon=1, **forecast_kwargs):
    df = prepare_monthly_df(df, date_col)
    
    # Partition data
    train_df = df[(df[date_col] >= train_start_date) & (df[date_col] <= train_end_date)]
    test_df = df[(df[date_col] >= test_start_date) & (df[date_col] <= test_end_date)]
    
    # Generate forecasts from training data
    preds = forecast_fn(train_df, group_col, date_col, target_col, horizon, **forecast_kwargs)
    
    # Match predictions to actuals
    results = []
    for _, row in preds.iterrows():
        prod = row[group_col]
        fd = row[date_col]
        actual_row = test_df[(test_df[group_col] == prod) & (test_df[date_col] == fd)]
        if not actual_row.empty:
            actual = actual_row[target_col].values[0]
            results.append({
                group_col: prod,
                'date': fd,
                'horizon': row['horizon'],
                'predicted_sales': row['predicted_sales'],
                'actual_sales': actual
            })
    
    res_df = pd.DataFrame(results)
    # MAPE by forecast horizon
    horizon_metrics = (res_df.groupby('horizon').apply(lambda g: mape(g['actual_sales'], g['predicted_sales'])))
    # MAPE by product
    product_metrics = (res_df.groupby(group_col).apply(lambda g: mape(g['actual_sales'], g['predicted_sales'])))
    # Overall MAPE
    overall_mape = mape(res_df['actual_sales'],res_df['predicted_sales'])
    return res_df, horizon_metrics, product_metrics, overall_mape


# ------------Function Calls-------------
results_naive, horizon_metrics_naive, product_metrics_naive, overall_mape_naive = fixed_train_test_forecast(
    df, 
    datetime(2023, 1, 1),           # train_start_date
    datetime(2024, 12, 31),         # train_end_date
    datetime(2025, 1, 1),           # test_start_date
    datetime(2025, 12, 31),         # test_end_date
    "productid",                    # group_col
    "date",                         # date_col
    "sales_units",                  # target_col
    naive_forecast,                 # forecast_fn
    horizon=12                      # keyword arg
)

results_seasonal, horizon_metrics_seasonal, product_metrics_seasonal, overall_mape_seasonal = fixed_train_test_forecast(
    df,
    datetime(2023, 1, 1),
    datetime(2024, 12, 31),
    datetime(2025, 1, 1),
    datetime(2025, 12, 31),
    "productid",
    "date",
    "sales_units",
    seasonal_naive_forecast,
    horizon=12,
    period=12  # passed via **forecast_kwargs
)

results_moving_average, horizon_metrics_moving_average, product_metrics_moving_average, overall_mape_moving_average = fixed_train_test_forecast(
    df,
    datetime(2023, 1, 1),
    datetime(2024, 12, 31),
    datetime(2025, 1, 1),
    datetime(2025, 12, 31),
    "productid",
    "date",
    "sales_units",
    moving_average_forecast,
    horizon=12,
    window=3  # passed via **forecast_kwargs
)

# Run a 2024 validation; if 2024 validation is similar to 2025, the model is likely robust
results_naive_internal_val, horizon_metrics_naive_internal_val, product_metrics_naive_internal_val, overall_mape_naive_internal_val = fixed_train_test_forecast(
    df,
    datetime(2023, 1, 1),
    datetime(2023, 12, 31),
    datetime(2024, 1, 1),
    datetime(2024, 12, 31),
    "productid",
    "date",
    "sales_units",
    naive_forecast,
    horizon=12
)


# -------------Save Outputs--------------
results_naive.to_csv("baseline_naive_predictions.csv", index=False)
results_seasonal.to_csv("baseline_seasonal_predictions.csv", index=False)
results_moving_average.to_csv("baseline_moving_average_predictions.csv", index=False)
results_naive_internal_val.to_csv("baseline_naive_internal_validation_predictions.csv", index=False)

product_metrics_naive.to_csv("baseline_naive_product_mape.csv", index=False)
product_metrics_seasonal.to_csv("baseline_seasonal_product_mape.csv", index=False)
product_metrics_moving_average.to_csv("baseline_moving_average_product_mape.csv", index=False)
product_metrics_naive_internal_val.to_csv("baseline_naive_internal_validation_product_mape.csv", index=False)

horizon_metrics_naive.to_csv("baseline_naive_horizon_mape.csv", index=False)
horizon_metrics_seasonal.to_csv("baseline_seasonal_horizon_mape.csv", index=False)
horizon_metrics_moving_average.to_csv("baseline_moving_average_horizon_mape.csv", index=False)
horizon_metrics_naive_internal_val.to_csv("baseline_naive_internal_validation_horizon_mape.csv", index=False)

overall_summary = pd.DataFrame({
    "Model": [
        "Naive",
        "Seasonal Naive",
        "Moving Average",
        "Naive (2024 Validation)"
    ],
    "Overall_MAPE": [
        overall_mape_naive,
        overall_mape_seasonal,
        overall_mape_moving_average,
        overall_mape_naive_internal_val
    ]
})
overall_summary.to_csv("baseline_model_summary.csv", index=False)


# ---------------Results-----------------
