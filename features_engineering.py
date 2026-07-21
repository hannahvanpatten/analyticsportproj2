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
     
    # Lags for sales

        for l in lag_list: # Loops through each specified sales lag period

            g[f'sales_lag_{l}'] = g[target_col].shift(l) # Creates a sales feature containing sales values from the specified number of months earlier

    # Rolling means and std for sales (use min_periods=1 so early windows produce values)

        for w in roll_windows: # Loops through each specified rolling window size

            g[f'sales_roll_mean_{w}'] = g[target_col].shift(1).rolling(window=w, min_periods=1).mean() # Calculates the average sales from the previous specified number of months without including the current month

            g[f'sales_roll_std_{w}'] = g[target_col].shift(1).rolling(window=w, min_periods=1).std().fillna(0.0) # Calculates the standard deviation of previous sales within the specified window and replaces missing values with zero

    # Price features

        g['price_lag_1'] = g[price_col].shift(1) # Creates a feature containing the product's price from the previous month

        g['price_pln_per_unit_pct_change'] = (g[price_col] - g['price_lag_1']) / g['price_lag_1'].replace(0, np.nan) # Calculates the percentage change in price from the previous month and replaces zero previous prices with missing values to prevent division by zero

        g['price_pln_per_unit_pct_change'] = g['price_pln_per_unit_pct_change'].fillna(0.0) # Replaces missing price percentage change values with zero

    # Inventory / demand ratios

        g['average_monthly_demand_safe'] = g['average_monthly_demand'].replace(0, np.nan).fillna(1.0) # Replaces zero average monthly demand values with one to prevent division by zero in later calculations

        g['inventory_to_avg_monthly_demand'] = g['inventory_level'] / g['average_monthly_demand_safe'] # Calculates the ratio of current inventory to average monthly demand

        g['excess_inventory'] = g.get('excess_inventory', np.nan) # Retrieves the existing excess inventory column or creates missing values if the column does not exist

        g['excess_inventory_pct_of_avg_demand'] = g['excess_inventory'] / g['average_monthly_demand_safe'] # Calculates excess inventory as a proportion of average monthly demand

     # lead time / safety stock

        if 'lead_time_demand' not in g.columns: # Checks whether the lead time demand column is missing from the DataFrame

            g['lead_time_demand'] = g['average_monthly_demand_safe'] * g.get('lead_time_months', 0) # Calculates expected demand during the product's lead time if the feature does not already exist
    
    # flag for short shelf life: shelf_life_months <= 6 (example threshold)

        g['short_shelf_life_flag'] = (g['shelf_life_months'] <= 6).astype(int) # Assigns 1 to products with a shelf life of six months or less and 0 to products with a longer shelf life

        
        feats.append(g.reset_index()) # Resets the date index and adds the current product's feature-engineered DataFrame to the results list

    df_feats = pd.concat(feats, ignore_index=True, sort=False) # Combines the feature-engineered DataFrames for all products into a single DataFrame

    return df_feats


def select_and_save(df: pd.DataFrame, output_path: str = OUTPUT_CSV):

    original_cols = [

        'productid', 'product_name', 'year_month', 'date',

        'inventory_level', 'sales_units', 'price_pln_per_unit',

        'storage_cost_pln_per_unit_month', 'shelf_life_months',

        'lead_time_months', 'safety_stock_months', 'monthly_revenue',

        'inventory_carrying_cost', 'annual_sales', 'average_inventory',

        'average_monthly_demand', 'excess_inventory', 'inventory_turnover',

        'lead_time_demand', 'reorder_point', 'expiration_risk', 'year'

    ] # Creates a list of original dataset columns to retain in the final output

    engineered_cols = [c for c in df.columns if c.startswith('sales_lag_') or

        c.startswith('sales_roll_') or

        c in (

            'price_lag_1',

            'price_pln_per_unit_pct_change',

            'inventory_to_avg_monthly_demand',

            'excess_inventory_pct_of_avg_demand',

            'average_monthly_demand_safe',

            'short_shelf_life_flag',

            'month', 'month_sin', 'month_cos'

        )] # Uses a list comprehension to identify engineered feature columns

    keep_cols = [c for c in original_cols if c in df.columns] + engineered_cols # Combines available original columns with all identified engineered feature columns

    # also ensure group/index columns are present

    if 'productid' not in keep_cols and 'productid' in df.columns: # Checks whether product ID exists in the DataFrame but is missing from the columns selected for output

        keep_cols.insert(0, 'productid') # Adds product ID to the beginning of the selected column list

    if 'date' not in keep_cols and 'date' in df.columns: # Checks whether date exists in the DataFrame but is missing from the columns selected for output

        keep_cols.insert(1, 'date') # Adds date as the second column in the selected column list

    out_df = df[keep_cols].copy() # Creates a new DataFrame containing only the selected original and engineered feature columns

    out_df = out_df.sort_values(['productid', 'date']).reset_index(drop=True) # Sorts the output by product and date and resets the row index

    # Save to CSV 

    out_df.to_csv(output_path, index=False) 

    print(f"Saved engineered features to: {output_path}") 

    return out_df # Returns the final feature-engineered DataFrame


def main(): # Defines the main function that runs the complete feature engineering process 

    print("Loading:", INPUT_CSV) # Prints the name of the input CSV file being loaded 

    df = pd.read_csv(INPUT_CSV, parse_dates=['date']) # Loads the input CSV into a DataFrame and parses the date column as datetime values 

    print("Initial rows:", len(df)) # Prints the number of rows in the input DataFrame 

 

    df = add_time_features(df) # Adds calendar-based time features to the input DataFrame 

    df_feats = add_lag_roll_features(df) # Adds lag, rolling, price, inventory, and product features to the DataFrame 

    out_df = select_and_save(df_feats, OUTPUT_CSV) # Selects the final columns and saves the feature-engineered dataset to the output CSV file 

    print("Feature engineering complete. Rows in output:", len(out_df)) # Prints a completion message and the number of rows in the final output dataset 


if __name__ == "__main__": # Checks whether the script is being run directly rather than imported as a module 

    main() # Runs the main function when the script is executed directly