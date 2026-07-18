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

 
def add_lag_roll_features(

    df: pd.DataFrame,

    group_col: str = 'productid', # Defines the column used to group rows by product

    date_col: str = 'date', # Defines the column containing the date values

    target_col: str = 'sales_units', # Defines the sales column used to create lag and rolling features

    price_col: str = 'price_pln_per_unit', # Defines the price column used to create price-based features

    lag_list: List[int] = [1, 3, 12], # Defines the number of months to shift sales data when creating lag features

    roll_windows: List[int] = [3, 6] # Defines the window sizes used to calculate rolling sales statistics

) -> pd.DataFrame: 

    df = df.sort_values([group_col, date_col]).copy() # Sorts the data by product and date and creates a copy of the sorted DataFrame
 
    feats = [] # Creates an empty list to store the feature-engineered DataFrame for each product

    # Work per product to avoid mixing series

    for prod, g in df.groupby(group_col): # Groups the DataFrame by product and loops through each product's data

        g = g.set_index(date_col).asfreq('MS')  # enforces monthly index; will create NaNs if months missing

        g[group_col] = prod # Assigns the current product ID to all rows, including any rows created for missing months
    
    