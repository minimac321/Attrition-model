
import logging
from pathlib import Path
import sys
import os

import mlflow

def setup_logging(logger_level=logging.INFO, log_file="data_preparation.log"):
    """
    Configures the logging settings.
    """
    logger = logging.basicConfig(
        level=logger_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    return logger

# ===========================
# Setup MLflow Experiment
# ===========================
def get_or_create_experiment(experiment_name: str, current_location=Path(os.path.abspath('')).resolve()) -> str:
    """
    Fetches the experiment with the given name if it exists. If it doesn't exist, creates a new one.
    
    Parameters:
    - experiment_name (str): The name of the MLFlow experiment.
    
    Returns:
    - experiment_id (str): The ID of the experiment to use.
    """
    mlruns_dir = os.path.join(current_location, "mlruns")
    mlflow.set_tracking_uri(mlruns_dir)

    # Fetch the experiment if it exists
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            logging.info(f"Experiment '{experiment_name}' exists with ID: {experiment.experiment_id}")
            return experiment.experiment_id
        else:
            # Create a new experiment if it does not exist
            experiment_id = mlflow.create_experiment(experiment_name)
            logging.info(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
            return experiment_id
    except mlflow.MlflowException as e:
        logging.info(f"Error: {e}")
        raise

def convert_target_str_to_int(df):
    assert "Attrition" in df.columns
    target_class_str_mapper = {
        "No": 0,
        "Yes": 1,
    }
    target_class_str = "Attrition"
    target_class = "target"
    df[target_class] = df[target_class_str].apply(lambda x: target_class_str_mapper[x])
    
    return df