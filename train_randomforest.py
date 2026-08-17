"""Train a demand forecasting RandomForest model from engineered features.

- Time-based split (train <= 2024-12-31, test >= 2025-01-01 by default).
- Preprocessing pipeline (fit on training set only):
    - numeric: median imputation + StandardScaler
    - categorical: constant imputation + OneHotEncoder
- Trains a global RandomForestRegressor and saves model + predictions + metrics.

Usage:
    python train_randomforest.py --input features_dataset.csv

Outputs:
  - models/model_rf.joblib
  - outputs/predictions_rf.csv
  - outputs/metrics_rf.csv
"""
from __future__ import annotations
import os
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
         model_out: str = "models/model_rf.joblib",
         preds_out: str = "outputs/predictions_rf.csv",
         metrics_out: str = "outputs/metrics_rf.csv",
         train_end: str = "2024-12-31",
         test_start: str = "2025-01-01",
         n_estimators: int = 200,
         max_depth: int | None = None,
         random_state: int = 42):
    print(f"Loading features: {input_csv}")
    if not os.path.exists(input_csv):
        raise SystemExit(f"Input file not found: {input_csv}")
    df = pd.read_csv(input_csv, parse_dates=["date"], infer_datetime_format=True)

    # Ensure date column is datetime if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Convert cutoffs
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)

    train_df = df[df['date'] <= train_end_dt].copy()
    test_df = df[df['date'] >= test_start_dt].copy()

    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    if len(train_df) == 0:
        raise SystemExit("No training rows found — check input and train_end date.")

    target_col = 'sales_units'

    # Identify candidate feature columns (drop non-features)
    drop_cols = {target_col, 'date', 'year_month'}
    cols = [c for c in df.columns if c not in drop_cols and c not in (None, '')]

    # Treat productid as categorical if present
    categorical_cols = [c for c in ['product_name'] if c in cols]
    numeric_cols = [c for c in cols if c not in categorical_cols]

    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols)>10 else ''}")
    print(f"Categorical features: {categorical_cols}")

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    # Model
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  random_state=random_state,
                                  n_jobs=-1)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Prepare training & test matrices
    X_train = train_df[numeric_cols + categorical_cols]
    y_train = train_df[target_col]
    X_test = test_df[numeric_cols + categorical_cols]
    y_test = test_df[target_col]

    # Fit
    print("Fitting RandomForest pipeline on training data...")
    pipeline.fit(X_train, y_train)

    # Predict
    print("Predicting on test set...")
    y_pred = pipeline.predict(X_test)

    # Metrics
    overall_mape = mape(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test MAE: {mae:.4f}")
    print(f"Test MAPE: {overall_mape:.2f}%")

    # Save outputs
    preds_df = test_df[['product_name', 'date']].copy()
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
    parser = argparse.ArgumentParser(description='Train a RandomForest demand forecasting model')
    parser.add_argument('--input', '-i', default='features_dataset.csv', help='Input features CSV')
    parser.add_argument('--model-out', default='models/model_rf.joblib', help='Path to save trained model')
    parser.add_argument('--preds-out', default='outputs/predictions_rf.csv', help='Path to save test predictions')
    parser.add_argument('--metrics-out', default='outputs/metrics_rf.csv', help='Path to save metrics summary')
    parser.add_argument('--train-end', default='2024-12-31', help='Inclusive end date for training data')
    parser.add_argument('--test-start', default='2025-01-01', help='Start date for test data')
    parser.add_argument('--n-estimators', type=int, default=200, help='Number of trees for RandomForest')
    parser.add_argument('--max-depth', type=int, default=None, help='Max tree depth (None = unlimited)')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    main(input_csv=args.input,
         model_out=args.model_out,
         preds_out=args.preds_out,
         metrics_out=args.metrics_out,
         train_end=args.train_end,
         test_start=args.test_start,
         n_estimators=args.n_estimators,
         max_depth=args.max_depth,
         random_state=args.random_state)
