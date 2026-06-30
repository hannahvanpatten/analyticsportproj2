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

# ------------Function Calls-------------

# -------------Save Outputs--------------

# ---------------Results-----------------
