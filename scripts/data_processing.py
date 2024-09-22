#!/usr/bin/env python3
# data_preparation.py

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

# Add my parent directory to path variables
current_location = Path(os.path.abspath('')).resolve()
print(current_location)
sys.path.append(str(current_location))


from utils.utils import convert_target_str_to_int, setup_logging
from utils.constants import random_value
from utils.cleaning import (
    drop_columns_with_missing_data,
    identify_numeric_columns_with_object_type,
    correct_numeric_columns_with_object_type,
    fill_missing_values,
    modified_z_score_anomaly_detection,
    validate_columns_against_reference,
)

setup_logging()


def setup_directories(current_location):
    """
    Sets up the necessary directories for raw, transformed, and output data.
    """
    data_dir = os.path.join(current_location, "data")
    os.makedirs(data_dir, exist_ok=True)

    raw_data_dir = os.path.join(data_dir, "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    transformed_data_dir = os.path.join(data_dir, "transformed_data")
    os.makedirs(transformed_data_dir, exist_ok=True)
    
    dashboard_dir = os.path.join(data_dir, "dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    return raw_data_dir, transformed_data_dir, dashboard_dir


def load_data(file_in_name, raw_data_dir):
    """
    Loads the employee attrition dataset from an Excel file.
    """
    attrition_fname = os.path.join(raw_data_dir, file_in_name)
    try:
        df = pd.read_excel(attrition_fname)
        logging.info(f"Data loaded successfully from {attrition_fname}.")
    except FileNotFoundError:
        logging.error(f"File not found: {attrition_fname}")
        sys.exit(1)
    
    logging.info(f"Columns: {df.columns.tolist()}")
    logging.info(f"Shape: {df.shape}")
    logging.debug(f"First 5 rows:\n{df.head()}")
    
    # Drop 'EmployeeID' if unique
    if "EmployeeID" in df.columns and df["EmployeeID"].is_unique:
        df = df.drop(columns=["EmployeeID"])
        logging.info("Dropped 'EmployeeID' column as it contains unique values.")
    
    logging.debug(f"Data after dropping 'EmployeeID':\n{df.head(2)}")
    
    return df


def filter_anomaly_using_modified_z_score(df, threshold = 3.0):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    logging.info(f"numeric_columns: {numeric_columns}")

    for column in numeric_columns:
        # Calculate MAD stats
        df = modified_z_score_anomaly_detection(df, column, threshold)
    
    anomaly_columns = [col for col in df.columns if col.startswith("anomaly")]
    logging.info(f"Anomalous columns: {anomaly_columns}")
    
    # Use the any() function to filter rows where any of the anomaly columns are True
    filtered_df = df[~df[anomaly_columns].any(axis=1)]
    
    # Now filter out columns with anomaly and z-score
    # Clean up Cols
    z_score_col_names = [col for col in filtered_df.columns if col.startswith("modified_z_score_")]
    anomaly_column_names = [col for col in filtered_df.columns if col.startswith("anomaly")]
    cos_to_drop = z_score_col_names + anomaly_column_names

    filtered_df = filtered_df.drop(
        columns=cos_to_drop, 
    )
    logging.debug(f"Dropped all columns:\n", z_score_col_names)
    logging.debug(f"Dropped all columns:\n", anomaly_column_names)

    logging.info(f"Shape after filtering anomalies: {filtered_df.shape}")
    logging.debug(f"Data after filtering anomalies:\n{filtered_df.head()}")
    
    filtered_df = filtered_df.drop(columns=[anomaly_columns], errors="ignore")
    
    return filtered_df
    
def clean_data(df, dashboard_file_pre):
    """
    Cleans the dataset by handling missing values, correcting data types, and filtering anomalies.
    """
    # Drop columns with more than 50% missing data
    df, columns_to_drop = drop_columns_with_missing_data(
        df=df,
        nan_data_threshold_perc=0.5
    )
    logging.info(f"Dropped columns due to missing data: {columns_to_drop}")
    logging.info(f"Shape after dropping columns: {df.shape}")
    
    # Identify and correct numeric columns stored as objects
    columns_w_string_values = identify_numeric_columns_with_object_type(df)
    logging.info(f"Columns with string values that should be numeric: {columns_w_string_values}")
    
    for col_name in columns_w_string_values:
        df[col_name] = df[col_name].apply(correct_numeric_columns_with_object_type)
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        logging.info(f"Converted column '{col_name}' to numeric.")
    
    # Fill missing values
    df = fill_missing_values(df)
    logging.info("Filled missing values.")
    logging.debug(f"Missing values after filling:\n{df.isnull().sum()}")
    
    # Drop rows where 'YearsAtCompany' <= 0
    remove_years_at_company_zero_rows = True
    if remove_years_at_company_zero_rows:
        initial_shape = df.shape
        df = df[df['YearsAtCompany'] > 0]
        dropped_rows = initial_shape[0] - df.shape[0]
        logging.info(f"Dropped {dropped_rows} rows where 'YearsAtCompany' <= 0.")
        logging.info(f"Shape after dropping rows: {df.shape}")
     
    # Drop Rows where 'Age' < 18
    df = df[df["Age"] >= 18]
    
    # Convert specific columns to categorical
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    logging.info(f"categorical_columns: {categorical_columns}")

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
            logging.info(f"Converted column '{col}' to categorical.")
    logging.debug(f"Data types after conversion:\n{df.dtypes}")
    
    write_dashboard_data_flag = True   
    if write_dashboard_data_flag:
        write_dashboard_data(df, dashboard_file_pre, convert_target=True)
    
    # Filter anomalies
    df = filter_anomaly_using_modified_z_score(df, threshold=3.0)
    logging.info(f"Shape after filtering anomalies rows: {df.shape}")
    
    # Validate columns against reference
    df = validate_columns_against_reference(
        df=df,
        reference_column='YearsAtCompany',
        columns_to_validate=["YearsInCurrentRole", "YearsWithCurrManager", "YearsSinceLastPromotion"]
    )
    sanity_counts = df["sanity_feature_value_check"].value_counts()
    logging.info(f"Sanity check value counts:\n{sanity_counts}")
    
    # Keep only valid rows
    df = df[df["sanity_feature_value_check"] == True].drop(columns=["sanity_feature_value_check"])
    logging.info(f"Shape after sanity check: {df.shape}")
    logging.debug(f"Data after sanity check:\n{df.tail(2)}")
    
    return df

def add_dummy_features(df, seed_value=42):
    """
    Optionally adds dummy survey fields to the dataset.
    """
    from utils.feature_engineering import add_random_survey_fields
    
    add_dummy_data = True
    if add_dummy_data:
        df, fields_added = add_random_survey_fields(df, seed_value=seed_value)
        logging.info(f"Added {len(fields_added)} dummy fields: {fields_added}")
        logging.debug(f"Data after adding dummy fields:\n{df.head(2)}")
    else:
        logging.info("Dummy data not added.")
    
    return df

def feature_engineering_steps(df):
    """
    Applies feature engineering steps to the dataset.
    """
    from utils.feature_engineering import apply_feature_engineering
    
    df, additional_columns = apply_feature_engineering(
        df=df,
        add_dummy_data=True
    )
    logging.info(f"Added new feature columns: {additional_columns}")
    logging.debug(f"Data after feature engineering:\n{df.head()}")
    
    # Encode target variable
    target_class_str_mapper = {
        "No": 0,
        "Yes": 1,
    }
    target_class_str = "Attrition"
    target_class = "target"
    if target_class_str in df.columns:
        df[target_class] = df[target_class_str].apply(lambda x: target_class_str_mapper.get(x, 0))
        df[target_class] = df[target_class].astype(int)
        df = df.drop(columns=[target_class_str])
        logging.info(f"Encoded target variable '{target_class}' with value counts:\n{df[target_class].value_counts()}")
    else:
        logging.error(f"Target column '{target_class_str}' not found in DataFrame.")
        sys.exit(1)
    
    return df

def encoding_categorical_features(df):
    """
    Performs one-hot encoding on categorical features and saves the encoder.
    """
    
    one_hot_mappings = {}
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if "Attrition" in categorical_features:
        categorical_features.remove("Attrition")
    
    logging.info(f"Categorical features to encode: {categorical_features}")

    # Iterate through the specified columns
    for column in categorical_features:
        # Check if the column is boolean (True/False)
        if df[column].dtype == 'bool':
            # Convert boolean values to 1 and 0
            df[column] = df[column].astype(int)
            one_hot_mappings[column] = [f'{column}_0', f'{column}_1']
        else:
            # Perform one-hot encoding on non-boolean columns
            dummies = pd.get_dummies(df[column], prefix=column)
            # Save the mapping of original column to one-hot encoded columns
            one_hot_mappings[column] = dummies.columns.tolist()
            # Drop the original column and concatenate the one-hot encoded columns
            df = df.drop(column, axis=1)
            df = pd.concat([df, dummies], axis=1)
            
    logging.info(f"Shape after one-hot encoding: {df.shape}")
    logging.debug(f"Data after one-hot encoding:\n{df.head()}")
    
    return df


def write_cleaned_data(df, transformed_data_dir):
    """
    Splits the data into train, validation, and test sets and saves them as CSV files.
    """
    target_class = "target"
    
    # Split the data into train (70%), validation (15%), and test sets (15%)
    train_df, valid_test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=random_value,
        stratify=df[target_class]
    )
    val_df, test_df = train_test_split(
        valid_test_df,
        test_size=0.5,  # 0.3 x 0.5 = 0.15
        random_state=random_value,
        stratify=valid_test_df[target_class]
    )
    
    logging.info(f"Train shape: {train_df.shape}")
    logging.info(f"Validation shape: {val_df.shape}")
    logging.info(f"Test shape: {test_df.shape}")
    
    # Save the processed data
    train_path = os.path.join(transformed_data_dir, "train_script.csv")
    val_path = os.path.join(transformed_data_dir, "validation_script.csv")
    test_path = os.path.join(transformed_data_dir, "test_script.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logging.info(f"Saved train data to {train_path}")
    logging.info(f"Saved validation data to {val_path}")
    logging.info(f"Saved test data to {test_path}")


def write_dashboard_data(df, dashboard_file_post, convert_target=False):
    if convert_target:
        df = convert_target_str_to_int(df)
    
    dummy_cols = list(df.columns[df.columns.str.endswith("_dummy")]) + ["Attrition"]
    dash_data_post = df.drop(columns=dummy_cols, errors="ignore")
    dash_data_post.to_csv(dashboard_file_post, index=False)       
    
    logging.info(f"Wrote dashboard data to storage with shape: {dash_data_post.shape}")


def main():
    """
    Main function to orchestrate the data preparation pipeline.
    """
    logging.info("Starting data preparation pipeline.")

    # Set up directories
    raw_data_dir, transformed_data_dir, dashboard_dir = setup_directories(current_location)
    # Create locations for dashboard files
    dashboard_file_pre =  os.path.join(dashboard_dir, "dashboard_display_data_pre_anomaly.csv")
    dashboard_file_post =  os.path.join(dashboard_dir, "dashboard_display_data_post_anomaly.csv")
    
    # Load data
    input_file_name = "employee_attrition_data_final.xlsx"
    df = load_data(input_file_name, raw_data_dir)
    
    # Clean data
    df = clean_data(df, dashboard_file_pre)
    
    # Add dummy features
    df = add_dummy_features(df, seed_value=random_value)
    
    # Feature engineering
    df = feature_engineering_steps(df)
    
    write_dashboard_data_flag = True   
    if write_dashboard_data_flag:
        write_dashboard_data(df, dashboard_file_post, False)
        
    # Encode categorical features
    df = encoding_categorical_features(df)
    
    # Write cleaned and processed data
    write_cleaned_data(df, transformed_data_dir)
    
    logging.info("Data preparation pipeline completed successfully.")

if __name__ == "__main__":
    main()
