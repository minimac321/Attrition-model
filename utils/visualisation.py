import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(pipeline, feature_names, top_n=20):
    # Extract feature importances
    importances = pipeline.named_steps['classifier'].feature_importances_
    feature_importance = pd.Series(importances, index=feature_names)
    top_features = feature_importance.sort_values(ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10,8))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title('Top Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
   
   
def plot_categorical_columns(df: pd.DataFrame, columns_per_row: int = 4, width=16, height_per_row = 3):
    """
    Plots bar charts for all categorical columns in the dataframe, with a specific number of plots per row.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - columns_per_row (int): The number of plots to show in each row.
    """
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
    

def plot_anomaly_detection_results(df: pd.DataFrame, column: str, threshold: float = 3.5, figsize=(16, 6)):
    """
    Plot the results of Modified Z-score anomaly detection.

    :param df: DataFrame containing the data with anomaly detection results.
    :param column: The column on which anomaly detection was performed.
    :param threshold: The threshold used for anomaly detection.
    """
    # Get the median and MAD for the column to calculate bounds
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    upper_bound = median + threshold * mad / 0.6745
    lower_bound = median - threshold * mad / 0.6745
    
    # Plot the histogram with the upper and lower bounds
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Full data plot (Left plot)
    sns.histplot(df[column], kde=True, ax=axes[0])
    axes[0].axvline(upper_bound, color='r', linestyle='--', label=f'Upper bound ({upper_bound:.2f})')
    axes[0].axvline(lower_bound, color='b', linestyle='--', label=f'Lower bound ({lower_bound:.2f})')
    
    # Count anomalies and display it on the plot
    anomaly_count = df[f'anomaly_{column}'].sum()
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
    axes[0].legend()

    # Plot only the data within bounds (Right plot)
    filtered_data = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    filtered_data_mean = filtered_data[column].mean()
    filtered_std = filtered_data[column].std()
    sns.histplot(filtered_data[column], kde=True, ax=axes[1])
    title = f"""
        {column} Data within Bounds
        Filtered Mean: {filtered_data_mean:.2f}
        Filtered STD : {filtered_std: .2f}
    """
    axes[1].set_title(title, fontsize=14)
    axes[1].set_xlabel(column)
    axes[1].set_ylabel('Frequency')

    # Adjust layout and show the plots
    plt.tight_layout()    
    return fig, axes


# def plot_categorical_vs_target_grouped(df, feature, target):
#     """
#     Plots a bar chart showing the percentage of a categorical feature's values grouped by the target variable.

#     Parameters:
#     - df (pd.DataFrame): The input dataframe.
#     - feature (str): The categorical feature to plot.
#     - target (str): The target variable to group by.

#     Returns:
#     - fig, axes: Matplotlib figure and axes objects.
#     """
#     total_counts = df[feature].value_counts()
    
#     # Handle both 0/1 and 'Yes'/'No' cases for the target
#     attrition_yes_counts = df[df[target] == 1][feature].value_counts()
#     attrition_no_counts = df[df[target] == 0][feature].value_counts()

#     # Calculate percentages
#     attrition_yes_percentage = (attrition_yes_counts / total_counts * 100).fillna(0)
#     attrition_no_percentage = (attrition_no_counts / total_counts * 100).fillna(0)
    
#     # Create a DataFrame for easier plotting
#     percentages_df = pd.DataFrame({
#         'Attrition = Yes': attrition_yes_percentage,
#         'Attrition = No': attrition_no_percentage
#     })

#     # Create the figure and axes objects
#     fig, axes = plt.subplots(figsize=(10, 6))

#     # Plot side-by-side bars
#     percentages_df.plot(kind='bar', ax=axes)
    
#     # Set plot title and labels
#     axes.set_title(f'Percentage of Attrition by {feature}')
#     axes.set_ylabel('Attrition Percentage (%)')
#     axes.set_xlabel(feature)
#     axes.set_xticklabels(axes.get_xticklabels(), rotation=0)
    
#     # Return the figure and axes for rendering outside the function
#     return fig, axes


def plot_categorical_vs_target_grouped(df, feature, target):
    total_counts = df[feature].value_counts()
    attrition_yes_counts = df[df[target] == 'Yes'][feature].value_counts()
    attrition_no_counts = df[df[target] == 'No'][feature].value_counts()
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
    
    
def plot_numeric_vs_target_boxplot(df, numeric_feature, target):
    """
    Plots a boxplot of the distribution of a numeric feature, grouped by the target variable.

    :param df: DataFrame containing the data.
    :param numeric_feature: The numeric feature to be plotted.
    :param target: The target variable to group by (e.g., 'Attrition').
    """
    plt.figure(figsize=(10, 6))
    
    # Create the boxplot grouped by the target
    sns.boxplot(x=target, y=numeric_feature, data=df, palette={'Yes': 'lightblue', 'No': 'lightcoral'})
    
    # Add title and labels
    plt.title(f'Distribution of {numeric_feature} by {target}')
    plt.xlabel(target)
    plt.ylabel(numeric_feature)
    
    # Show plot
    plt.show()    
    


def plot_correlation_matrix_with_target(df, target_column, figsize=(8, 8)):
    """
    Plots a correlation heatmap for all features including the target variable.
    A diagonal heatmap will be shown, where the correlation between each feature and the target variable is highlighted.
    
    :param df: The DataFrame containing the features and the target column.
    :param target_column: The name of the target variable (e.g., 'Attrition').
    :param figsize: Size of the figure for plotting.
    """
    # Select only numeric columns for the correlation matrix
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Calculate the correlation matrix for the numeric features
    corr_matrix = df[numeric_columns].corr()
    
    # Plot the heatmap with a diagonal line
    plt.figure(figsize=figsize)
    
    # Generate a mask for the upper triangle (keeping the diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        cbar=True, 
        square=True, 
        linewidths=0.5,
        linecolor='white', 
        mask=mask,
        annot_kws={"size": 9}
    )
    
    # Highlight the diagonal
    plt.title(f'Correlation Heatmap with {target_column}', fontsize=15)
    plt.show()