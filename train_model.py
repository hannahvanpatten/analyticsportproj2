"""Train a demand forecasting model using the engineered feature dataset.

This script trains a global LightGBM model with simple preprocessing.
It performs a time-based split (train up to 2024-12-31, test = 2025), fits
imputers/scalers on train only (via an sklearn Pipeline), evaluates using MAPE,
and writes artifacts to disk.

Usage (from repo root):
    python train_model.py --input features_dataset.csv

Outputs (written under repository):
  - models/model_lgbm.joblib     # trained model artifact
  - outputs/predictions.csv      # test set predictions with actuals and errors
  - outputs/metrics_summary.csv  # overall diagnostics

Notes:
  - This script assumes features_dataset.csv already contains forecast-safe
    engineered columns (lags/rolls) and precomputed aggregates which are safe
    to use at forecast time.
  - Imputation and scaling are fit on training data only via the Pipeline.

"""
from __future__ import annotations
import os
import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error
import joblib


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (np.abs(y_true) > eps)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100.0


def ensure_dirs(*paths):
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)


def main(input_csv: str = "features_dataset.csv",
         model_out: str = "models/model_lgbm.joblib",
         preds_out: str = "outputs/predictions.csv",
         metrics_out: str = "outputs/metrics_summary.csv",
         train_end: str = "2024-12-31",
         test_start: str = "2025-01-01"):

    print(f"Loading features: {input_csv}")
    df = pd.read_csv(input_csv, parse_dates=["date"]) if os.path.exists(input_csv) else pd.read_csv(input_csv)

    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Convert cutoff strings to timestamps
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)

    # Split
    train_df = df[df['date'] <= train_end_dt].copy().reset_index()
    test_df = df[df['date'] >= test_start_dt].copy().reset_index()

    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    if len(train_df) == 0:
        raise SystemExit("No training rows found — check input and train_end date.")

    target_col = 'sales_units'

    # Identify features: drop obvious non-features
    drop_cols = {target_col, 'date', 'year_month'}
    # keep productid as categorical if present
    cols = [c for c in df.columns if c not in drop_cols and c not in (None, '')]

    # If productid present treat as categorical
    categorical_cols = [c for c in ['product_name'] if c in cols]
    numeric_cols = [c for c in cols if c not in categorical_cols]

    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols)>10 else ''}")
    print(f"Categorical features: {categorical_cols}")

    # Build preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    # LightGBM model (sklearn API)
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Prepare X/y
    X_train = train_df[numeric_cols + categorical_cols]
    y_train = train_df[target_col]
    X_test = test_df[numeric_cols + categorical_cols]
    y_test = test_df[target_col]

    # Fit
    print("Fitting pipeline on training data...")
    pipeline.fit(X_train, y_train)

    # Predict
    print("Predicting on test set...")
    y_pred = pipeline.predict(X_test)

    # Metrics
    overall_mape = mape(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test MAE: {mae:.4f}")
    print(f"Test MAPE: {overall_mape:.2f}%")

    # Save predictions and simple diagnostics
    preds_df = test_df[['product_name', 'date']].copy().reset_index()
    preds_df['actual_sales'] = y_test.values
    preds_df['predicted_sales'] = y_pred
    preds_df['abs_error'] = np.abs(preds_df['actual_sales'] - preds_df['predicted_sales'])
    preds_df['ape_pct'] = np.where(np.abs(preds_df['actual_sales']) > 1e-8,
                                   np.abs((preds_df['actual_sales'] - preds_df['predicted_sales']) / preds_df['actual_sales']) * 100.0,
                                   np.nan)

    ensure_dirs(os.path.dirname(model_out) or '.', os.path.dirname(preds_out) or '.')

    print(f"Saving model to: {model_out}")
    joblib.dump(pipeline, model_out)

    print(f"Saving predictions to: {preds_out}")
    preds_df.to_csv(preds_out, index=False)

    metrics_summary = pd.DataFrame({
        'metric': ['MAE', 'MAPE'],
        'value': [mae, overall_mape]
    })
    metrics_summary.to_csv(metrics_out, index=False)

    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a demand forecasting model from engineered features')
    parser.add_argument('--input', '-i', default='features_dataset.csv', help='Input features CSV')
    parser.add_argument('--model-out', default='models/model_lgbm.joblib', help='Path to save trained model')
    parser.add_argument('--preds-out', default='outputs/predictions.csv', help='Path to save test predictions')
    parser.add_argument('--metrics-out', default='outputs/metrics_summary.csv', help='Path to save metrics summary')
    parser.add_argument('--train-end', default='2024-12-31', help='Inclusive end date for training data')
    parser.add_argument('--test-start', default='2025-01-01', help='Start date for test data')
    args = parser.parse_args()
    main(input_csv=args.input, model_out=args.model_out, preds_out=args.preds_out, metrics_out=args.metrics_out, train_end=args.train_end, test_start=args.test_start)
