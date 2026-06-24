# ----------------Imports----------------
import pandas as pd
import numpy as np

df = pd.read_csv("dataset_with_derived_metrics.csv", parse_dates=["date"])

# --Separate 2025 Data from Main Dataset--
train_set = df[df["date"] <= "2024-12-31"].copy()
holdout_set = df[df["date"] >= "2025-01-01"].copy()
train_set.to_csv("training_dataset.csv", index=False)
holdout_set.to_csv("2025_holdout.csv", index=False)
