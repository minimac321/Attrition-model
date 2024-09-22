import os
import sys
import logging
import mlflow
from pathlib import Path


# Add my parent directory to path variables
current_location = Path(os.path.abspath("")).resolve()
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
setup_logging(logger_level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.info("Starting model training pipeline.")



# ===========================
# Main Script
# ===========================
def main(model_uri: str):
    # Define paths
    test_data_path = os.path.join(transformed_data_dir, "test_script.csv")

    # Load test data
    X_test, y_test = load_inference_data(test_data_path, target_col="target", skip_dummy_data=True)

    # Load trained model from MLflow
    model = load_model(model_uri)
    logging.info(f"Loaded in model successfully from uri {model_uri}")

    # Create a ClassificationMetrics instance to compute the metrics
    classifier_metric_obj = ClassificationMetrics(model, X_test, y_test, cutoff_point=0.3232)
    classifier_metric_obj.run_evaluation()
    metrics_dict = classifier_metric_obj.get_metrics_dict()

    # Start an MLflow run to log new metrics
    with mlflow.start_run(run_id=model_uri.split("/")[1]) as run:
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
    # Use sys.argv to fetch the model_uri from command-line arguments
    if len(sys.argv) != 2:
        logging.info("Usage: python script_name.py <mlflow-model-uri>")
        sys.exit(1)

    # model_uri will be the second argument in sys.argv
    model_uri = sys.argv[1]
    logging.info(f"model_uri: {model_uri}")
    
    # Pass the model_uri to main function
    main(model_uri=model_uri)