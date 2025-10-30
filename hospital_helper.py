import pandas as pd

def clean_variable_names(df):
  df.columns = df.columns.str.lower()
  df.columns = df.columns.str.replace(' ', '_') # replace white space in columns names with underscores
  df.columns = df.columns.str.replace('*', '') # strips out * from columns
  df.columns = df.columns.str.replace('_of_', '')
  return df


def process_missing(df, verbose=True):
  """
  Processes a DataFrame to handle missing values by dropping, imputing, and reporting.

  Args:
      df (pd.DataFrame): The input DataFrame.
      verbose (bool): If True, prints detailed steps and summaries to the console.
                      Defaults to True.

  Returns:
      pd.DataFrame: The processed DataFrame with missing values handled.
  """
  if verbose:
    print("--- Starting Missing Value Analysis ---")
    # Group by hospital and calculate the sum of missing values for each column
    missing_counts_by_hospital = df.groupby(by='hospital')[df.columns].apply(lambda x: x.isnull().sum())
    average_missing_by_hospital = missing_counts_by_hospital.mean(axis=1)
    print("Top 5 hospitals with the most missing data points on average:")
    print(average_missing_by_hospital.sort_values(ascending=False).head())

    if 'apd' in df.columns:
        print("\nHospitals with missing 'apd' values:")
        # confirm that Saint Elizabeth - Peru is missing all apd counts
        print(df[df['apd'].isnull()]['hospital'].value_counts())

  # dropping Saint Elizabeth - Peru for now because apd is focal
  df = df[df['hospital'] != 'Saint Elizabeth - Peru']

  # Calculate missing percentage for each column
  missing_percent = df.isnull().sum() / len(df) * 100

  # Identify columns to drop (more than 40% missing)
  cols_to_drop = missing_percent[missing_percent > 40].index

  # Drop the identified columns
  df = df.drop(columns=cols_to_drop)

  # Identify columns to impute (less than or equal to 40% missing)
  cols_to_impute = missing_percent[(missing_percent <= 40) & (missing_percent > 0)].index

  # Impute with the median for each hospital
  for col in cols_to_impute:
      if pd.api.types.is_numeric_dtype(df[col]):
          df[col] = df.groupby('hospital')[col].transform(lambda x: x.fillna(x.median()))

  # For any remaining NaNs, fill with the global median of the column
  for col in cols_to_impute:
      if pd.api.types.is_numeric_dtype(df[col]):
          global_median = df[col].median()
          df[col] = df[col].fillna(global_median)

  if verbose:
    print("\n--- Imputation and Dropping Summary ---")
    print("\nMissing values remaining after two-step imputation (likely non-numeric):")
    # Only show columns that were targeted for imputation
    if not cols_to_impute.empty:
        print(df[cols_to_impute].isnull().sum().sort_values(ascending=False).head())
    else:
        print("No columns required imputation.")


    print("\nColumns dropped due to >40% missing values:")
    if not cols_to_drop.empty:
      for s in cols_to_drop:
        print(f"- {s}")
    else:
        print("No columns were dropped.")
    print("\n--- Processing Complete ---")

  return df

if __name__ == '__main__':
    # Code to execute when the file is run directly
    pass
