import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import mlflow
from sklearn.pipeline import Pipeline


# Add my parent directory to path variables
current_location = Path(os.path.abspath("")).resolve()
print(current_location)
sys.path.append(str(current_location))

from utils.constants import RANDOM_VALUE


def plot_confusion_matrix(model, X_test, y_test, save_path=None):
    """
    Plots and saves confusion matrix.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["False", "True"],
        yticklabels=["False", "True"],
        annot_kws={"size": 16},
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.close()

    # Log the plot to MLflow
    if mlflow.active_run():
        mlflow.log_artifact(save_path)


def plot_categorical_vs_target_grouped(df, feature, target):
    total_counts = df[feature].value_counts()
    attrition_yes_counts = df[df[target] == "Yes"][feature].value_counts()
    attrition_no_counts = df[df[target] == "No"][feature].value_counts()
    attrition_yes_percentage = (attrition_yes_counts / total_counts * 100).fillna(0)
    attrition_no_percentage = (attrition_no_counts / total_counts * 100).fillna(0)

    # Create a DataFrame for easier plotting
    percentages_df = pd.DataFrame(
        {"Attrition = Yes": attrition_yes_percentage, "Attrition = No": attrition_no_percentage}
    )

    # Plot side-by-side bars
    percentages_df.plot(kind="bar", figsize=(10, 6))
    plt.title(f"Percentage of Attrition by {feature}")
    plt.ylabel("Attrition Percentage (%)")
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
    sns.boxplot(
        x=target, y=numeric_feature, data=df, palette={"Yes": "lightblue", "No": "lightcoral"}
    )

    # Add title and labels
    plt.title(f"Distribution of {numeric_feature} by {target}")
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
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns

    # Calculate the correlation matrix for the numeric features
    corr_matrix = df[numeric_columns].corr()

    # Plot the heatmap with a diagonal line
    plt.figure(figsize=figsize)

    # Generate a mask for the upper triangle (keeping the diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
        mask=mask,
        annot_kws={"size": 9},
    )

    # Highlight the diagonal
    plt.title(f"Correlation Heatmap with {target_column}", fontsize=15)
    plt.show()


# ===========================
# Plotting functions for model training script
# ===========================


def plot_feature_importance(
    model: Pipeline,
    feature_names: list,
    save_path: str,
    top_n_features: int = 20,
    show_plot: bool = False,
) -> None:
    """
    Plots and saves feature importance.

    Args:
        model (Pipeline): The trained model pipeline.
        feature_names (list): List of feature names.
        save_path (str): Path where the plot will be saved.
        top_n_features (int, optional): Number of top features to display. Defaults to 20.
        show_plot (bool, optional): Whether to display the plot. Defaults to False.
    """
    importances = model.named_steps["classifier"].feature_importances_
    feature_importance = pd.Series(importances, index=feature_names)
    top_features = feature_importance.sort_values(ascending=False).head(top_n_features)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title(f"Top {top_n_features} Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if not show_plot:
        plt.close()
    else:
        plt.show()

    # Log the plot to MLflow
    if mlflow.active_run():
        mlflow.log_artifact(save_path)


def plot_permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list,
    save_path: str,
    top_n_features: int = 20,
    n_repeats: int = 20,
    scoring: str = "balanced_accuracy",
    show_plot: bool = False,
) -> None:
    """
    Plots and saves permutation importance.

    Args:
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test feature data.
        y_test (pd.Series): The test labels.
        feature_names (list): List of feature names.
        save_path (str): Path where the plot will be saved.
        top_n_features (int, optional): Number of top features to display. Defaults to 20.
        n_repeats (int, optional): Number of repetitions for permutation. Defaults to 20.
        scoring (str, optional): The scoring metric used. Defaults to 'balanced_accuracy'.
        show_plot (bool, optional): Whether to display the plot. Defaults to False.
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, scoring=scoring, random_state=RANDOM_VALUE
    )
    # Extract feature names and permutation importance values
    importances = (
        result.importances
    )  # This gives all the permutation importance values for each feature

    # Sort features by mean importance and select the top_n_features
    sorted_indices = np.argsort(result.importances_mean)[-top_n_features:]

    # Plot the box plot for the  top_n_features features
    plt.figure(figsize=(10, 6))
    plt.boxplot([importances[i] for i in sorted_indices], vert=False, patch_artist=True)
    plt.yticks(range(1, len(sorted_indices) + 1), [feature_names[i] for i in sorted_indices])
    plt.xlabel(f"Decrease in {scoring} After Permutation")
    plt.title(f"Top {top_n_features} Features - Permutation Importance (with Variability)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if not show_plot:
        plt.close()
    else:
        plt.show()

    # Log the plot to MLflow
    if mlflow.active_run():
        mlflow.log_artifact(save_path)


def plot_shap_values(
    model: Pipeline, X_test: pd.DataFrame, save_path: str, show_plot: bool = False
) -> None:
    """
    Calculates and plots SHAP values.

    Args:
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test feature data.
        save_path (str): Path where the plot will be saved.
        show_plot (bool, optional): Whether to display the plot. Defaults to False.
    """
    X_test_transformed = model.named_steps["preprocessor"].transform(X_test)
    transformed_data_df = pd.DataFrame(data=X_test_transformed, columns=X_test.columns)

    explainer = shap.Explainer(model.named_steps["classifier"])
    shap_values = explainer(transformed_data_df)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, transformed_data_df, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    if not show_plot:
        plt.close()
    else:
        plt.show()

    # Log the plot to MLflow
    if mlflow.active_run():
        mlflow.log_artifact(save_path)


def plot_roc_curve(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    save_path: str,
    show_plot: bool = False,
) -> None:
    """
    Plots the ROC curve for both training and validation datasets.

    Args:
        model (Pipeline): The trained model pipeline.
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training labels.
        X_valid (pd.DataFrame): Validation feature data.
        y_valid (pd.Series): Validation labels.
        save_path (str): Path where the plot will be saved.
        show_plot (bool, optional): Whether to display the plot. Defaults to False.
    """
    # Get the predicted probabilities for the ROC curve
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_valid_proba = model.predict_proba(X_valid)[:, 1]

    # Compute ROC curve and AUC for training and validation sets
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_valid, tpr_valid, _ = roc_curve(y_valid, y_valid_proba)
    auc_train = roc_auc_score(y_train, y_train_proba)
    auc_valid = roc_auc_score(y_valid, y_valid_proba)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {auc_train:.3f})")
    plt.plot(fpr_valid, tpr_valid, label=f"Validation ROC (AUC = {auc_valid:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")

    # Label the plot
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if not show_plot:
        plt.close()
    else:
        plt.show()

    # Log the plot to MLflow
    if mlflow.active_run():
        mlflow.log_artifact(save_path)
