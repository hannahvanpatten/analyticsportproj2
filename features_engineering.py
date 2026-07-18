import numpy as np 

import pandas as pd 

from typing import List 

 
INPUT_CSV = "dataset_with_derived_metrics.csv" 

OUTPUT_CSV = "features_dataset.csv"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame: 

    df['date'] = pd.to_datetime(df['date']) # Ensure date is datetime 

    df['date'] = df['date'].dt.to_period('M').dt.to_timestamp() # Ensure date is normalized to start-of-month 

    # Calendar features 

    df['month'] = df['date'].dt.month # Extracts the month from the date 

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12) # Calculates the sine of the month 

    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12) # Calculates the cosine of the month 

    df['year'] = df['date'].dt.year.astype(int)  # Extracts the year from the date as an integer 

    return df 

 
