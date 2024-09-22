import os
import sys
import logging
import mlflow
from pathlib import Path
from datetime import datetime

# Add my parent directory to path variables
current_location = Path(os.path.abspath("")).resolve()
print(current_location)
sys.path.append(str(current_location))

data_dir = os.path.join(current_location, "data")
transformed_data_dir = os.path.join(data_dir, "transformed_data")
evaluation_dir = os.path.join(data_dir, "evaluation")
output_dir = os.path.join(data_dir, "output_data")

from utils.utils import setup_logging
from utils.evaluation import load_inference_data, load_model

setup_logging(logger_level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.info("Starting model training pipeline.")


def main():
    # Define paths
    model_uri = "runs:/6d34577c481f4040b70158e370bdffdb/model"
    run_id = model_uri.split("/")[1]
    inference_data_fname = os.path.join(transformed_data_dir, "inference_sample.csv")

    # Load test data
    X_test, y_test = load_inference_data(
        data_path=inference_data_fname, target_col="target", skip_dummy_data=True
    )

    # Load trained model from MLflow
    model = load_model(model_uri)
    logging.info(f"Loaded in model successfully from uri {model_uri}")
    cutoff_point = 0.4

    # Use the cutoff point to create binary predictions
    y_prob_preds = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob_preds >= cutoff_point).astype(int)

    X_test["prediction_prob"] = y_prob_preds
    X_test["prediction"] = y_pred
    logging.info(X_test["prediction"].value_counts())

    output_df_path = os.path.join(output_dir, f"inference_df_{run_id}.csv")
    X_test.to_csv(output_df_path, index=False)

    logging.info(f"Inference completed - output df shape: {X_test.shape}")


if __name__ == "__main__":
    main()
