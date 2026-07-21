# ----------------Imports----------------
import pandas as pd
import numpy as np

old_file = "final_dataset.csv" # INSERT FILE NAME HERE
dataset = pd.read_csv(old_file) # Read the dataset
print(dataset.head()) # Print a preview of the original for comparability
new_file = "dataset_with_derived_metrics.csv" # INSERT NEW FILE NAME HERE

# ---------Derived Metrics Function----------
def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame: # UPDATE COLUMN NAMES/COMMENT OUT ANY METRICS YOU DON'T WISH TO CALCULATE/ADD YOUR OWN AS NEEDED

  # Extract year from year_month column (format: YYYYMM)
  df['year'] = df['year_month'].astype(str).str[:4]
  
  # Convert year_month to numeric for sorting and lookback calculation
  df['year_month_numeric'] = df['year_month'].astype(int)
  
  # Sort by productid and year_month to enable proper lookback
  df = df.sort_values(['productid', 'year_month_numeric']).reset_index(drop=True)

  # Multiply two columns (monthly revenue)
  if 'sales_units' in df.columns and 'price_pln_per_unit' in df.columns:
    df['monthly_revenue'] = ((df['sales_units'] * df['price_pln_per_unit']).round(2))

  # Multiply two columns (inventory carrying cost)
  if 'inventory_level' in df.columns and 'storage_cost_pln_per_unit_month' in df.columns:
    df['inventory_carrying_cost'] = ((df['inventory_level'] * df['storage_cost_pln_per_unit_month']).round(2))

  # Calculate annual_sales: sum of sales_units from 12 months prior (no future leakage)
  if 'sales_units' in df.columns and 'productid' in df.columns:
    df['annual_sales'] = df.groupby('productid')['sales_units'].rolling(window=12, min_periods=12).sum().reset_index(level=0, drop=True).round(0)

  # Calculate average monthly inventory from 12 months prior (no future leakage)
  if 'inventory_level' in df.columns and 'productid' in df.columns:
    df['average_inventory'] = (df.groupby('productid')['inventory_level'].rolling(window=12, min_periods=12).sum().reset_index(level=0, drop=True) / 12).round(0)

  # Calculate average monthly demand
  if 'annual_sales' in df.columns:
    df['average_monthly_demand'] = (df['annual_sales'] / 12).round(0)

  # Calculate excess inventory
  if 'inventory_level' in df.columns and 'sales_units' in df.columns:
    df['excess_inventory'] = df['inventory_level'] - df['sales_units']

  # Calculate inventory turnover
  if 'annual_sales' in df.columns and 'average_inventory' in df.columns:
    df['inventory_turnover'] = df['annual_sales'] / df['average_inventory']

  # Calculate lead time demand
  if 'average_monthly_demand' in df.columns and 'lead_time_months' in df.columns:
    df['lead_time_demand'] = df['average_monthly_demand'] * df['lead_time_months']

  # Calculate reorder point
  if 'lead_time_demand' in df.columns and 'safety_stock_months' in df.columns:
    df['reorder_point'] = (df['lead_time_demand'] / df['safety_stock_months']).round(0)

  # Calculate expiration risk
  if 'excess_inventory' in df.columns and 'price_pln_per_unit' in df.columns:
    df['expiration_risk'] = (df['excess_inventory'] * df['price_pln_per_unit'])

  # Calculate percentage change between two columns
  #if 'forecast_value' in df.columns and 'actual_value' in df.columns:
      # df['forecast_error_pct'] = (
          # (df['actual_value'] - df['forecast_value']) / df['forecast_value'] * 100
      # ).round(2)
    
  # Calculate absolute error
  # if 'forecast_value' in df.columns and 'actual_value' in df.columns:
    # df['absolute_error'] = abs(df['actual_value'] - df['forecast_value'])
    
  # Calculate rolling average
  # if 'sales_value' in df.columns:
    # df['sales_7day_avg'] = df['sales_value'].rolling(window=7, min_periods=1).mean().round(2)
    
  # Conditional metric (flag/category)
  # if 'absolute_error' in df.columns:
    # df['error_category'] = pd.cut(
      # df['absolute_error'],
      # bins=[0, 10, 50, np.inf],
      # labels=['low', 'medium', 'high']
    # )
    
  # Aggregate metric (sum/count by group)
  # if 'product_id' in df.columns and 'quantity' in df.columns:
    # df['product_total_qty'] = df.groupby('product_id')['quantity'].transform('sum')
    
  # Date-based metrics (if you have date columns)
  # if 'date' in df.columns:
    # df['date'] = pd.to_datetime(df['date'])
    # df['month'] = df['date'].dt.month
    # df['quarter'] = df['date'].dt.quarter
     # df['year'] = df['date'].dt.year
    
  print("Derived Metrics Added")
  print("---------------------")
  print(f"New columns: {[col for col in df.columns if col not in df.columns[:-11]]}") # REPLACE INDEX DIGIT WITH NUMBER OF COLUMNS ADDED
  print(df.head(20))
  print("")
  return df

dataset = add_derived_metrics(dataset)
dataset = dataset.to_csv(new_file, index=False)