import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def identify_numeric_columns_with_object_type(df):
    """
    Identifies numeric columns (int or float) where any value contains quotes (single or double).
    Returns a list of column names that contain values with quotes in numeric columns.
    """
    columns_with_quotes = []

    categorical_columns = df.select_dtypes(include=[object, "category"]).columns

    # Iterate through the dataframe columns
    for column in categorical_columns:
        # Check if any value in the column contains quotes
        if df[column].apply(lambda x: isinstance(x, str) and ('"' in x or "'" in x)).any():
            columns_with_quotes.append(column)

    return columns_with_quotes


def correct_numeric_columns_with_object_type(value, lower_bound=1, upper_bound=5):
    """
    Tries to convert the value to an integer between lower_bound and upper_bound.
    If it's a valid float/int as a string, it converts it to an int.
    Otherwise, it keeps the original value.
    """
    try:
        value = value.strip('"')
        value = value.strip("'")
        # Attempt to parse the value to a float, then convert to an int
        value = float(value)
        # Ensure it's within the specified bounds
        if lower_bound <= value <= upper_bound:
            return value
    except (ValueError, TypeError, AttributeError):
        # If conversion fails, return the original value
        return value


def validate_columns_against_reference(df, reference_column, columns_to_validate):
    """
    Adds a boolean feature `sanity_feature_value_check` that indicates if all values in
    the specified columns are less than or equal to the reference column.

    :param df: The DataFrame.
    :param reference_column: The column to use as the reference for the check.
    :param columns_to_validate: A list of columns to compare against the reference column.
    :return: The DataFrame with the new sanity check feature.
    """
    # Initialize sanity_feature_value_check to True
    df["sanity_feature_value_check"] = True

    # Loop through the columns to check if any value is greater than the reference column
    for col in columns_to_validate:
        # Check if each value in the column is less than or equal to the reference column, then update the sanity check column
        df["sanity_feature_value_check"] = df["sanity_feature_value_check"] & (
            df[col] <= df[reference_column]
        )

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates a pipeline to fill missing values (NaN) in both numeric and categorical features.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - df_filled (pd.DataFrame): The dataframe with NaN values filled for both numeric and categorical columns.
    """

    # Identify numeric and categorical columns
    numeric_columns = df.select_dtypes(include=[float, int]).columns
    categorical_columns = df.select_dtypes(include=[object, "category"]).columns

    # Define imputers
    numeric_imputer = SimpleImputer(strategy="median")  # Use median to fill NaN for numeric columns
    categorical_imputer = SimpleImputer(
        strategy="most_frequent"
    )  # Use most frequent value for categorical columns

    # Create pipelines for numeric and categorical preprocessing
    numeric_pipeline = Pipeline(
        [
            ("imputer", numeric_imputer),
        ]
    )

    categorical_pipeline = Pipeline([("imputer", categorical_imputer)])

    # Combine numeric and categorical pipelines
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )

    # Fit and transform the data using the preprocessor pipeline
    df_filled = pd.DataFrame(
        preprocessor.fit_transform(df),
        columns=numeric_columns.tolist() + categorical_columns.tolist(),
    )

    # Ensure the categorical columns retain their original data types (converted to strings after transformation)
    for cat_col in categorical_columns:
        df_filled[cat_col] = df_filled[cat_col].astype(str)

    for cat_col in numeric_columns:
        df_filled[cat_col] = pd.to_numeric(df_filled[cat_col])

    # Assert to ensure we haven't removed any rows and columns
    assert df_filled.shape == df.shape

    return df_filled


def modified_z_score_anomaly_detection(
    df: pd.DataFrame, column: str, threshold: float = 3.5
) -> pd.DataFrame:
    """
    Detect anomalies in a specific column using the Modified Z-score method.

    :param df: DataFrame containing the data.
    :param column: The column to perform the Modified Z-score anomaly detection on.
    :param threshold: Modified Z-score threshold for detecting anomalies. Default is 3.5.
    :return: DataFrame with additional columns for Modified Z-scores and anomaly detection.
    """
    # Calculate the median of the column
    median = df[column].median()

    # Calculate the Median Absolute Deviation (MAD)
    mad = np.median(np.abs(df[column] - median))

    # Avoid division by zero
    if mad == 0:
        mad = 1e-10

    # Calculate Modified Z-scores
    df[f"modified_z_score_{column}"] = 0.6745 * (df[column] - median) / mad

    # Add a column to indicate anomalies (True if outside the range)
    df[f"anomaly_{column}"] = np.abs(df[f"modified_z_score_{column}"]) > threshold

    return df


def drop_columns_with_missing_data(df: pd.DataFrame, nan_data_threshold_perc: float) -> tuple:
    """
    Drops columns from the DataFrame where the percentage of missing data exceeds the threshold.

    :param df: The input DataFrame.
    :param nan_data_threshold_perc: The percentage (0.0 to 1.0) of missing data allowed for each column.
                      Columns with more than this percentage of missing data will be dropped.
    :return: A tuple containing the modified DataFrame and a list of dropped columns.
    """
    # Calculate the percentage of missing data for each column
    missing_percentage = df.isnull().mean()

    # Identify columns where the missing percentage is greater than the threshold
    columns_to_drop = missing_percentage[
        missing_percentage > nan_data_threshold_perc
    ].index.tolist()

    # Drop the identified columns from the DataFrame
    df_cleaned = df.drop(columns=columns_to_drop)

    return df_cleaned, columns_to_drop
