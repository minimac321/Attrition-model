import streamlit as st
import os
from pathlib import Path
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
import mlflow
import mlflow.pyfunc
import pandas as pd

# Add my parent directory to path variables
current_location = Path(os.path.abspath('')).resolve()
print(current_location)
sys.path.append(str(current_location))

from utils.constants import EXPERIMENT_NAME

# Directories
data_dir = os.path.join(current_location, "data")
transformed_data_dir = os.path.join(data_dir, "transformed_data")
processed_dir = os.path.join(data_dir, "processed_dir")
dashboard_dir = os.path.join(data_dir, "dashboard")


# Function to load model parameters from MLflow using a model_uri
def load_model_params_from_mlflow(model_uri):
    """
    Load model parameters from MLflow for the specified model_uri.
    """
    run_id = model_uri.split('/')[1]  # Extract the run ID from the model URI
    client = mlflow.tracking.MlflowClient()
    
    # Get the run details
    run = client.get_run(run_id)
    
    # Extract model parameters
    model_params = run.data.params
    return model_params



# Function to load model parameters from MLflow using a model_uri
def load_model_metrics_from_mlflow(model_uri):
    """
    Load model parameters from MLflow for the specified model_uri.
    """
    run_id = model_uri.split('/')[1]  # Extract the run ID from the model URI
    client = mlflow.tracking.MlflowClient()
    
    # Get the run details
    run = client.get_run(run_id)
    
    # Extract model parameters
    metrics = run.data.metrics
    return metrics

# Function to fetch all model URIs from a given experiment
def fetch_model_uris_from_experiment(experiment_name):
    """
    Fetch all model URIs for a given experiment name in MLflow.
    """
    # Get the experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        return None, []
    
    experiment_id = experiment.experiment_id
    
    # Fetch all runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    
    # Extract model URIs for each run
    model_uris = []
    for run_id in runs['run_id']:
        model_uris.append(f"runs:/{run_id}/model")
    
    return experiment_id, model_uris


def load_model_from_mlflow(model_uri):
    """
    Load the model from MLflow given the model_uri.
    """
    model = mlflow.pyfunc.load_model(model_uri)
    return model


# Function to fetch and return image from MLflow artifacts
def fetch_image_from_mlflow(run_id, image_name):
    """
    Fetch the specified image from MLflow artifacts.
    """
    client = mlflow.tracking.MlflowClient()
    
    # Try fetching the specific artifact (image) by its name
    artifacts = client.list_artifacts(run_id)
    
    for artifact in artifacts:
        if artifact.path.endswith(image_name):
            # Download the artifact
            local_path = client.download_artifacts(run_id, artifact.path)
            return Image.open(local_path)
    
    return None


def fetch_confusion_matrix_image(run_id):
    """
    Fetch the confusion matrix image from MLflow artifacts.
    """
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    
    # Find the confusion matrix file
    for artifact in artifacts:
        if artifact.path == 'confusion_matrix.png':
            # Download the artifact
            local_path = client.download_artifacts(run_id, artifact.path)
            return Image.open(local_path)
    
    return None


# Main Streamlit app
def main():
    st.title("Model Training and Evaluation")

    # Sidebar for experiment name input and model selection
    st.sidebar.title("MLFlow model identifiers")
    
    # Input for MLflow experiment name
    experiment_name = st.sidebar.text_input("Enter the MLflow Experiment Name", value=experiment_name)

    if experiment_name:
        # Fetch all model URIs for the selected experiment
        experiment_id, model_uris = fetch_model_uris_from_experiment(EXPERIMENT_NAME)
        
        if model_uris:
            # Let the user select a model URI
            selected_model_uri = st.sidebar.selectbox("Select Model URI", model_uris)
        else:
            st.sidebar.write(f"No models found for experiment: {experiment_name}")
            return
        
        # If a model is selected, display the evaluation
        if selected_model_uri:
            run_id = selected_model_uri.split('/')[1]

            # Main dashboard
            st.header(f"Model Evaluation for Run: {selected_model_uri}")
            
            model_params = load_model_params_from_mlflow(selected_model_uri)
            model_metrics = load_model_metrics_from_mlflow(selected_model_uri)
            model_type = model_params.get('model_type', 'N/A')

            # Toggle for Features
            with st.expander("Model Parameters", expanded=False):
                st.write("### Model Parameters")
                for param, value in model_params.items():
                    if value is not None and value.strip() != "None":
                        st.write(f"{param}: {value}")
                        
            # Toggle for Metrics
            with st.expander("Model Metrics", expanded=False):
                st.write("### Metrics")
                # Find the longest metric name for padding
                max_param_length = max(len(param) for param in model_metrics.keys())
                # Display metrics with alignment
                print("model_metrics", model_metrics)
                for param, value in model_metrics.items():
                    st.write(f"{param.ljust(max_param_length)}: {value:14.3f}")

            try:
                # Feature Importance
                st.subheader("Top N Feature Importance")
                feature_importance_img = fetch_image_from_mlflow(run_id, "feature_importance.png")
                if feature_importance_img:
                    st.image(feature_importance_img, caption="Feature Importance", use_column_width=True)
                else:
                    st.write("No feature importance image found in the artifacts.")
                
                # Permutation Importance
                st.subheader("Top N Permutation Importance")
                permutation_importance_img = fetch_image_from_mlflow(run_id, "permutation_importance.png")
                if permutation_importance_img:
                    st.image(permutation_importance_img, caption="Permutation Importance", use_column_width=True)
                else:
                    st.write("No permutation importance image found in the artifacts.")

                # SHAP Summary
                st.subheader("SHAP Summary")
                shap_summary_img = fetch_image_from_mlflow(run_id, "shap_summary.png")
                if shap_summary_img:
                    st.image(shap_summary_img, caption="SHAP Summary", use_column_width=True)
                else:
                    st.write("No SHAP summary image found in the artifacts.")
                
                # ROC Curve
                st.subheader("ROC Curve")
                roc_curve_img = fetch_image_from_mlflow(run_id, "roc_curve.png")
                if roc_curve_img:
                    st.image(roc_curve_img, caption="ROC Curve", use_column_width=True)
                else:
                    st.write("No ROC curve found in the artifacts.")
                    
                # Confusion Matrix
                st.subheader("Training Set Confusion Matrix")
                confusion_matrix_img = fetch_image_from_mlflow(run_id, "confusion_matrix.png")
                if confusion_matrix_img:
                    st.image(confusion_matrix_img, caption="Confusion Matrix", use_column_width=True)
                else:
                    st.write("No confusion matrix found in the artifacts.")
                
                # Test Set Confusion Matrix
                st.subheader("Test Set Confusion Matrix")
                test_confusion_matrix_img = fetch_image_from_mlflow(run_id, "test_set_confusion_matrix.png")
                if test_confusion_matrix_img:
                    st.image(test_confusion_matrix_img, caption="Test Set Confusion Matrix", use_column_width=True)
                else:
                    st.write("No test set confusion matrix found in the artifacts.")

            except Exception as e:
                st.error(f"Error loading model from MLflow: {str(e)}")

if __name__ == "__main__":
    main()