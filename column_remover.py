# ----------------Imports----------------
import numpy as np
import pandas as pd

old_file = "project_data_merged.csv" # INSERT FILE NAME HERE
dataset = pd.read_csv(old_file) # Read the dataset
print(dataset.head()) # Print a preview of the original for comparability
new_file = "final_dataset.csv" # INSERT NEW FILE NAME HERE
column_1 = "product_name_params"
column_2 = "product_name_plan"
# column_3 = "" # UNCOMMENT IF NEEDED
# column_4 = "" # UNCOMMENT IF NEEDED

def remove_columns(df: pd.DataFrame) -> pd.DataFrame:
  return df.drop(columns=[column_1, column_2])  # ADD MORE COLUMNS HERE AS NEEDED

dataset = remove_columns(dataset)  # Assign the result back to dataset

dataset.to_csv(new_file, index=False) # Save the new file
print(dataset.head()) # Print a preview of the new file for comparability
