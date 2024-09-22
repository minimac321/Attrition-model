import time
import streamlit as st
from datetime import datetime
import os
from pathlib import Path
import joblib
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add my parent directory to path variables
current_location = Path(os.path.abspath('')).resolve()
print(current_location)
sys.path.append(str(current_location))

# Import necessary functions

# Directories
data_dir = os.path.join(current_location, "data")
transformed_data_dir = os.path.join(data_dir, "transformed_data")
processed_dir = os.path.join(data_dir, "processed_dir")
dashboard_dir = os.path.join(data_dir, "dashboard")

# Load the dataset
def load_pre_anomaly_cleaned_data():
    cleaned_fname = os.path.join(dashboard_dir, "dashboard_display_data_pre_anomaly.csv")
    modelling_data = pd.read_csv(cleaned_fname)
    if "EmployeeID" in modelling_data.columns:
        modelling_data = modelling_data.drop(columns=["EmployeeID"], errors="ignore")

    assert "target" in modelling_data.columns
    return modelling_data

# Load the dataset
def load_post_anomaly_cleaned_data():
    dashboard_data_fname = os.path.join(dashboard_dir, "dashboard_display_data_post_anomaly.csv")
    modelling_data = pd.read_csv(dashboard_data_fname)
    if "Attrition" in modelling_data.columns:
        modelling_data = modelling_data.drop(columns=["Attrition"], errors="ignore")
        
    assert "target" in modelling_data.columns
    return modelling_data


# Function to plot anomaly detection results
def plot_anomaly_detection_results(df: pd.DataFrame, column: str, threshold: float = 3.5, figsize=(16, 6)):
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    upper_bound = median + threshold * mad / 0.6745
    lower_bound = median - threshold * mad / 0.6745
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.histplot(df[column], kde=True, ax=axes[0])
    axes[0].axvline(upper_bound, color='r', linestyle='--', label=f'Upper bound ({upper_bound:.2f})')
    axes[0].axvline(lower_bound, color='b', linestyle='--', label=f'Lower bound ({lower_bound:.2f})')
    
    # Count anomalies and display it on the plot
    anomaly_count = ((df[column] > upper_bound) | (df[column] < lower_bound)).sum()
    data_mean = df[column].mean()
    std = df[column].std()
    title_t = f"""
        Modified Z-Score Detection: {column}.
        Anomaly Count: {anomaly_count}
        Mean: {data_mean:.2f}
        STD : {std: .2f}
    """
    axes[0].set_title(title_t, fontsize=14)
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')

    filtered_data = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    sns.histplot(filtered_data[column], kde=True, ax=axes[1])
    axes[1].set_title(f"Filtered Data for {column} - Anomalies removed")
    axes[1].set_xlabel(column)
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Function to plot numeric vs target boxplot
def plot_numeric_vs_target_boxplot(df, numeric_feature, target):
    # Clear the previous plot
    plt.clf()
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x=target, y=numeric_feature, data=df, 
        # palette={'Yes': 'lightblue', 'No': 'lightcoral'}
    )
    plt.title(f'Distribution of {numeric_feature} by {target}')
    plt.xlabel(target)
    plt.ylabel(numeric_feature)
    st.pyplot(plt, clear_figure=True)

# Function to plot the correlation matrix
def plot_correlation_matrix(df, target_column):
    # Clear the previous plot
    plt.clf()
    plt.close()
    
    numeric_columns = list(df.select_dtypes(include=['float64', 'int64']).columns)
    for col in numeric_columns:
        if col.strip().endswith("_dummy"):
            numeric_columns.remove(col)
        
    corr_matrix = df[numeric_columns].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar=True, square=True, linewidths=0.5, mask=mask)
    plt.title(f'Correlation Heatmap with {target_column}', fontsize=15)
    plt.tight_layout()
    st.pyplot(plt, clear_figure=True)

    
def plot_categorical_vs_target_grouped(df, feature, target):
    # Clear the previous plot
    plt.clf()
    plt.close()
    
    total_counts = df[feature].value_counts()
    attrition_yes_counts = df[df[target] == 1][feature].value_counts()
    attrition_no_counts = df[df[target] == 0][feature].value_counts()
    attrition_yes_percentage = (attrition_yes_counts / total_counts * 100).fillna(0)
    attrition_no_percentage = (attrition_no_counts / total_counts * 100).fillna(0)
    
    # Create a DataFrame for easier plotting
    percentages_df = pd.DataFrame({
        'Attrition = Yes': attrition_yes_percentage,
        'Attrition = No': attrition_no_percentage
    })
    
    # Plot side-by-side bars
    percentages_df.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Percentage of Attrition by {feature}')
    plt.ylabel('Attrition Percentage (%)')
    plt.xlabel(feature)
    plt.xticks(rotation=0)
    st.pyplot(plt, clear_figure=True)

def plot_categorical_columns(df: pd.DataFrame, columns_per_row: int = 4, width=16, height_per_row = 3):
    """
    Plots bar charts for all categorical columns in the dataframe, with a specific number of plots per row.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - columns_per_row (int): The number of plots to show in each row.
    """
    # Clear the previous plot
    plt.clf()
    plt.close()
    
    print(df.shape)
    # Get all object (categorical) columns
    object_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if object_columns == 0:
        return
    # Calculate the number of rows required for the plot layout
    num_columns = len(object_columns)
    num_rows = (num_columns + columns_per_row - 1) // columns_per_row  # To handle uneven number of columns

    # Create subplots
    fig, axes = plt.subplots(num_rows, columns_per_row, figsize=(width, num_rows * height_per_row))

    # Flatten axes in case of a single row
    axes = axes.flatten()

    # Loop through each categorical column and plot the value counts
    for i, col in enumerate(object_columns):
        # Get value counts for the column
        value_counts = df[col].value_counts()

        # Plot on the corresponding subplot axis
        axes[i].bar(value_counts.index, value_counts.values, color='skyblue')
        axes[i].set_title(col)
        axes[i].tick_params(axis='x', rotation=45, labelsize=8)  # Rotate x-axis labels and set size
        axes[i].set_ylabel('Count')
        axes[i].set_xlabel(col)

    # Remove unused subplots if there are any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Categorical Features\n", fontsize=20)
    plt.tight_layout()    
    st.pyplot(plt, clear_figure=True)


def plot_target_feature(df):
    # Clear the previous plot
    plt.clf()
    plt.close()
    
    if "Attrition" not in df.columns:
        sns.countplot(x='target', data=df, palette="Set2")
    else:
        df['Attrition_Label'] = df['Attrition'].map({1: 'Yes', 0: 'No'})
        sns.countplot(x='Attrition_Label', data=df, palette="Set2")
        
    plt.title('Attrition Distribution')  # Add title
    plt.ylabel('Count')  # Set y-label
    st.pyplot(plt, clear_figure=True)

    
# EDA page with plots
def eda_page():
    st.title("EDA Dashboard")

    # Sidebar option for dataset selection
    dataset_choice = st.sidebar.radio(
        "Select Dataset", 
        options=["Dashboard data with Anomalies", "Data with no Anomalies"],
        index=0
    )

    # Load data based on user's choice
    if dataset_choice == "Data with no Anomalies":
        df_dashboard = load_post_anomaly_cleaned_data()
    else:
        df_dashboard = load_pre_anomaly_cleaned_data()

    target_var = "target"
    df_dash_numeric_cols = df_dashboard.select_dtypes(include=['float64', 'int64']).drop(columns=[target_var], errors='ignore').columns        
    df_dash_cat_cols = df_dashboard.select_dtypes(include=['object', 'category']).columns
    
    # Show basic info
    st.subheader("Dataset Overview")
    st.write(f"Dataset Shape: {df_dashboard.shape}")

    st.subheader(f"Dataset Statistics:")
    st.write(df_dashboard.describe().T)
    st.subheader(f"Dataset Sampled:")
    st.dataframe(df_dashboard.head(3))
    
    st.subheader("Target Feature Distributions")
    plot_target_feature(
        df=df_dashboard,
    )

    # Plot categorical columns
    st.subheader("Categorical Feature Distributions")
    plot_categorical_columns(
        df=df_dashboard,
        height_per_row=6,
        width=16,
        columns_per_row=2
    )
    
    # Plot correlation matrix
    st.subheader("Correlation Matrix")
    plot_correlation_matrix(df_dashboard, "target")

    # Anomaly detection on a sample numeric column
    st.subheader("Anomaly Detection")
    numeric_column = st.selectbox("Select a numeric column for anomaly detection", df_dash_numeric_cols)
    plot_anomaly_detection_results(df_dashboard, numeric_column)

    # Numeric vs target boxplot
    st.subheader("Boxplot for Numeric vs Target")
    numeric_column = st.selectbox("Select a numeric column for boxplot", df_dash_numeric_cols)
    plot_numeric_vs_target_boxplot(
        df_dashboard, numeric_column, 
        target_var
    )
    
    # Example usage of the function
    st.subheader("Bar chart showing the percentage of a categorical feature's values grouped by the target variable")
    categoric_column = st.selectbox("Select a categoric column for bar chart", df_dash_cat_cols)
    plot_categorical_vs_target_grouped(
        df_dashboard,
        feature=categoric_column, 
        target=target_var
    )

# Main function for Streamlit
def main():
    st.sidebar.title("Navigation")
    # page = st.sidebar.selectbox("Choose a page", ["EDA"])
    eda_page()

if __name__ == "__main__":
    main()