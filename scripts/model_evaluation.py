import os
import sys
import logging
import mlflow
from pathlib import Path


# Add my parent directory to path variables
current_location = Path(os.path.abspath('')).resolve()
print(current_location)
sys.path.append(str(current_location))

data_dir = os.path.join(current_location, "data")
transformed_data_dir = os.path.join(data_dir, "transformed_data")
evaluation_dir = os.path.join(data_dir, "evaluation")

from scripts.model_training import plot_confusion_matrix
from scripts.model_inference import load_inference_data
from utils.classification_class import ClassificationMetrics
from utils.utils import setup_logging
from utils.evaluation import load_model

# ===========================
# Setup Logger
# ===========================
setup_logging(
    logger_level=logging.INFO
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.info("Starting model training pipeline.")

# ===========================
# Load Test Data
# ===========================
# def load_test_data(test_data_path: str, target_col: str, skip_dummy_data: bool = True) -> tuple:
#     """
#     Loads the test data from CSV file and splits into features and target.

#     Args:
#         test_data_path (str): Path to the test data CSV file.
#         target_col (str): The name of the target column.
#         skip_dummy_data (bool, optional): Whether to exclude dummy columns. Defaults to True.

#     Returns:
#         tuple: A tuple containing the features (X_test) and target (y_test) as pandas DataFrames.
#     """
#     logging.info(f"Loading test data from {test_data_path}")
#     test_df = pd.read_csv(test_data_path)
    
#     cols_to_drop = [target_col] + [col for col in test_df.columns if "_dummy" in str(col)]    
#     X_test = test_df.drop(columns=cols_to_drop)
#     y_test = test_df[target_col]
#     logging.info(f"Test data loaded with shape: {test_df.shape}")
#     return X_test, y_test


# ===========================
# Plot Functions
# ===========================
# def plot_feature_importance(model: object, save_path: str) -> None:
#     """
#     Plots and saves feature importance.

#     Args:
#         model (object): The trained model.
#         save_path (str): Path where the plot will be saved.
#     """
#     feature_importances = model.named_steps['classifier'].feature_importances_
#     feature_names = model.named_steps['preprocessor'].get_feature_names_out()
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=feature_importances, y=feature_names)
#     plt.title("Feature Importance")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     mlflow.log_artifact(save_path)

# def plot_permutation_importance(model: object, X_test: pd.DataFrame, y_test: pd.Series, save_path: str) -> None:
#     """
#     Plots and saves permutation importance.

#     Args:
#         model (object): The trained model.
#         X_test (pd.DataFrame): Test feature data.
#         y_test (pd.Series): Test target labels.
#         save_path (str): Path where the plot will be saved.
#     """
#     result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
#     sorted_idx = result.importances_mean.argsort()[-10:]  # Top 10 features
#     plt.figure(figsize=(10, 6))
#     plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
#     plt.title("Permutation Importance")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     mlflow.log_artifact(save_path)

# def plot_shap_values(model, X_test, save_path):
#     """
#     Plots and saves SHAP values.
#     """
#     explainer = shap.Explainer(model.named_steps['classifier'])
#     shap_values = explainer(model.named_steps['preprocessor'].transform(X_test))
#     plt.figure(figsize=(10, 6))
#     shap.summary_plot(shap_values, X_test, show=False)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     mlflow.log_artifact(save_path)


# ===========================
# Main Script
# ===========================
def main():
    # Define paths
    model_uri = 'runs:/6d34577c481f4040b70158e370bdffdb/model'
    test_data_path = os.path.join(transformed_data_dir, "test_script.csv")

    # Load test data
    X_test, y_test = load_inference_data(test_data_path, target_col="target", skip_dummy_data=True)

    # Load trained model from MLflow
    model = load_model(model_uri)
    logging.info(f"Loaded in model successfully from uri {model_uri}")
        
    # Create a ClassificationMetrics instance to compute the metrics
    classifier_metric_obj = ClassificationMetrics(model, X_test, y_test, cutoff_point=0.3232)
    classifier_metric_obj.run_evaluation()
    y_test_preds = classifier_metric_obj.get_predictions()
    y_test_prob_preds = classifier_metric_obj.get_probability_predictions()
    
    metrics_dict = classifier_metric_obj.get_metrics_dict()
    
    
    # Start an MLflow run to log new metrics
    with mlflow.start_run(run_id=model_uri.split('/')[1]) as run:
        logging.info(f"Logging metrics for run: {run.info.run_id}")
        
        # Log evaluation metrics to MLflow
        for metric_name, metric_value in metrics_dict.items():
            eval_metric_name = f"test_{metric_name}"
            logging.info(f"{metric_name}: {metric_value:.3f}")
            mlflow.log_metric(eval_metric_name, metric_value)
            

        # Plot Confusion Matrix
        conf_save_path = os.path.join(evaluation_dir, f"test_set_confusion_matrix.png")
        plot_confusion_matrix(model, X_test, y_test, save_path=conf_save_path)
        
        # Log the confusion matrix plot as an artifact in MLflow
        mlflow.log_artifact(conf_save_path)
    
    logging.info("Evaluation completed")
    
    
if __name__ == "__main__":
    main()