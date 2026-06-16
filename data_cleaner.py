# ----------------Imports----------------
import numpy as np
import pandas as pd

remove_outliers = True # TOGGLE TO FALSE IF CAPPING OUTLIERS INSTEAD
dataset = "sales_train_2023_2024.csv" # INSERT FILE NAME HERE
original_dataset = pd.read_csv(dataset) # Provide a way to access original dataset for comparability
new_file = "sales_train_2023_2024_clean.csv"
initial_rows = ""
new_rows = ""



# ---------Function Definitions----------
def load_and_inspect(file_path: str) -> pd.DataFrame: # Load a dataset and print basic structural health metrics
    global df
    df = pd.read_csv(file_path) # Read the data (ADJUST DELIMITER/ENCODING IF NEEDED)
    print("Initial Data Assessment")
    print("-----------------------")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n") # Check row and column counts
    print("Data Types & Missing Values:")
    print(df.info()) # Check data types and missing values
    print("\nDuplicate Rows:", df.duplicated().sum()) # Check for duplicates
    print("Dataset Preview")
    print("---------------")
    print(df.head())
    print("")
    print("")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame: # Standardize columns
    df.columns = (
        df.columns.str.strip() # Remove leading and trailing whitespace from column names
        .str.lower() # Converts column names to lowercase
        .str.replace(" ", "_", regex=False) # Replaces whitespace with underscores
        .str.replace(r"[^\w]", "", regex=True) # Removes any non-alphanumeric/underscore characters
    )
    print("Column Name Cleanup")
    print("-------------------")
    print(df.head())
    print("")
    print("")
    return df


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame: # Remove exact duplicate rows from the dataset
  initial_rows = len(df)
  df = df.drop_duplicates().reset_index(drop=True) # Remove duplicates and clean up index
  new_rows = len(df)
  rows_dropped = initial_rows - new_rows
  print("Exact Duplicates Cleanup")
  print("------------------------")
  print(df.head())
  print(f"Dropped {rows_dropped} exact duplicate row(s).")
  print("")
  print("")
  return df


def handle_missing_values(
  df: pd.DataFrame, num_strategy="median", cat_strategy="placeholder"
) -> pd.DataFrame: # Impute or drop missing data based on column data type (EDIT num_strategy, cat_strategy, AND fillna() BELOW AS NEEDED)
  for col in df.columns:
    if df[col].isnull().sum() == 0:
      continue 
# Process numerical columns 
    if pds := pd.api.types.is_numeric_dtype(df[col]):
      if num_strategy == "median":
        df[col] = df[col].fillna(df[col].median())
      elif num_strategy == "mean":
        df[col] = df[col].fillna(df[col].mean())
      elif num_strategy == "drop":
        initial_rows = len(df)
        df = df.dropna(subset=[col])
        new_rows = len(df)
        rows_dropped = initial_rows - new_rows
        print(f"Dropped {rows_dropped} row(s) with missing values.")
# Process categorical or object columns
    else:
      if cat_strategy == "placeholder": # UPDATE placeholder AS NEEDED
        df[col] = df[col].fillna("unknown") # UPDATE Unknown AS NEEDED
      elif cat_strategy == "mode":
        df[col] = df[col].fillna(df[col].mode()[0])
      elif cat_strategy == "drop":
        initial_rows = len(df)
        df = df.dropna(subset=[col])
        new_rows = len(df)
        rows_dropped = initial_rows - new_rows
        print(f"Dropped {rows_dropped} row(s) with missing values.")
    print("Missing Values Cleanup")
    print("----------------------")
    print(df.head())
    print("")
    print("")
  return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame: # Standardize text string formatting across object columns
  for col in df.select_dtypes(include=["object"]): # Strip trailing/leading whitespace and normalize to lowercase
    df[col] = df[col].astype(str).str.strip().str.lower() # Replace empty strings resulting from strip with NaN
    df[col] = df[col].replace("", np.nan)
  print("Text Columns Cleanup")
  print("--------------------")
  print(df.head())
  print("")
  print("")
  return df


# def handle_outliers_iqr(df: pd.DataFrame, factor=1.5) -> pd.DataFrame: # Cap numerical outliers using the interquartile range (IQR) method
  # initial_rows = len(df)
  # for col in df.select_dtypes(include=[np.number]).columns:
    # q1 = df[col].quantile(0.25)
    # q3 = df[col].quantile(0.75)
    # iqr = q3 - q1
    # lower_bound = q1 - factor * iqr
    # upper_bound = q3 + factor * iqr
    # if remove_outliers:
      # df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    # else:
      # values_capped = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
      # df[col] = np.clip(df[col], lower_bound, upper_bound) # Cap values outside the boundaries
  # new_rows = len(df)
  # rows_dropped = initial_rows - new_rows
  # print("Outliers Cleanup")
  # print("----------------")
  # print(df.head())
  # if remove_outliers:
    # print(f"Dropped {rows_dropped} row(s) with outliers.")
  # else:
    # print(f"Capped {values_capped} value(s) outside the IQR boundaries.")
  # print("")
  # print("")
  # return df


def save_clean_file(output_path: str): # Save cleaned file
  df.to_csv(output_path, index=False)
  print(f"\nPipeline complete! Cleaned data saved to: {output_path}")
  return df


def clean_pipeline(file_path: str, output_path: str) -> pd.DataFrame:
  df = load_and_inspect(dataset)
  df = clean_column_names(df)
  df = handle_duplicates(df)
  df = handle_missing_values(df)
  df = clean_text_columns(df)
  # df = handle_outliers_iqr(df)
  df = save_clean_file(new_file)



# ------------Function Call--------------
clean_pipeline(dataset, new_file)