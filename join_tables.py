import pandas as pd

# Load all CSVs
sales_train = pd.read_csv('sales_train_2023_2024_clean.csv')
sales_test = pd.read_csv('sales_test_2025_clean.csv')
parameters = pd.read_csv('products_parameters_clean.csv')
financial = pd.read_csv('financial_plan_clean.csv')

# Combine the two sales datasets into one
sales = pd.concat([sales_train, sales_test], ignore_index=True)

# Unpivot the combined sales dataset 
sales_long = sales.melt( # Begins the unpivot
    id_vars=['productid', 'product_name', 'type'], # Specifies which columns will remain as identifiers
    var_name='year_month', # Names the column that will hold the "melted" columns' names
    value_name='value' # Names the column that will hold the values from the "melted" columns
)

# Pivot table back to columns
sales_pivot = sales_long.pivot_table( # Begins the pivot
    index=['productid', 'product_name', 'year_month'], # Specifies the columns that will identify each row
    columns='type', # Creates a new column for the "type" value (sales or inventory_level)
    values='value', # Populates the column with the values from the melted table
    aggfunc='first' # Handles duplicate columns (keeps only the first instance, in this case)
).reset_index()

sales_pivot.columns.name = None # Cleans up metadata leftover from the melt and pivot
sales_pivot.rename(columns={'Sales': 'sales_units', 'Inventory_level': 'inventory_level'}, inplace=True) # Renames columns for clarity

# Join all tables
project_data_merged = (sales_pivot
    .merge(parameters, on='productid', how='left', suffixes=('', '_params'))
    .merge(financial, on='productid', how='left', suffixes=('', '_plan'))
)

# Convert year_month to date
project_data_merged['date'] = pd.to_datetime(project_data_merged['year_month'] + '01', format='%Y%m%d')

# Save or use directly
project_data_merged.to_csv('project_data_merged.csv', index=False)
print(project_data_merged.head())
