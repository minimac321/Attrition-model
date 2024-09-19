
import streamlit as st
from datetime import datetime
import os
from pathlib import Path
import joblib
import pandas as pd
import sys

# Add my parent directory to path variables
current_location = Path(os.path.abspath('')).resolve()
print(current_location)
sys.path.append(str(current_location))

# Import necessary functions
from utils.train import split_and_save_data, train_model
from utils.evaluation import evaluate_model

# Directories
transformed_data_dir = os.path.join(current_location, "transformed_data")
processed_dir = os.path.join(current_location, "processed_dir")
models_dir = os.path.join(current_location, "models")

random_state_var = 42

# Initialize session state variables if they don't exist
if 'training_started' not in st.session_state:
    st.session_state.training_started = False

if 'training_completed' not in st.session_state:
    st.session_state.training_completed = False

if 'h_opt_max_evals' not in st.session_state:
    st.session_state.h_opt_max_evals = 25  # Default value

if 'model_name' not in st.session_state:
    st.session_state.model_name = 'RandomForest'

if 'best_metric' not in st.session_state:
    st.session_state.best_metric = 'accuracy'  # Default to accuracy

def load_data():
    # Load the dataset
    cleaned_fname = os.path.join(transformed_data_dir, "cleaned_scaled_data_17_09_2024.csv")
    modelling_data = pd.read_csv(cleaned_fname)
    # if "Attrition" in modelling_data.columns:
    #     modelling_data = modelling_data.drop(columns=["Attrition"], errors="ignore")
    return modelling_data

def train_and_evaluate():
    st.title("Model Training and Evaluation")
    
    # Load data
    target_col = "target"
    modelling_data = load_data()
    data_splits = split_and_save_data(
        input_df=modelling_data,
        target_col=target_col,
        processed_dir=processed_dir,
        random_state=random_state_var,
    )
    X_train = data_splits['X_train']
    y_train = data_splits['y_train']
    X_valid = data_splits['X_valid']
    y_valid = data_splits['X_valid']
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']
    
    st.write(f"X_train.shape: {X_train.shape}")
    st.write(f"y_train.shape: {y_train.shape}")
    st.write(f"X_valid.shape: {X_valid.shape}")
    st.write(f"y_valid.shape: {y_valid.shape}")
    st.write(f"X_test.shape: {X_test.shape}")
    st.write(f"y_test.shape: {y_test.shape}")
    
    st.write(f"Training started with {st.session_state.h_opt_max_evals} hyperopt evaluations.")

    # Sidebar parameters for training
    model_name = st.sidebar.selectbox(
        "Select Model to Train", ["RandomForest", "XGBoost"]
    )
    st.session_state.model_name = model_name

    # Hyperopt evaluations
    h_opt_max_evals = st.sidebar.slider("Number of Hyperopt Evaluations", 2, 50, value=25)
    st.session_state.h_opt_max_evals = h_opt_max_evals

    # Metric to choose the best model
    st.session_state.best_metric = st.sidebar.selectbox(
        "Select Metric for Best Model Selection", ["accuracy", "balanced_accuracy", "f1_score"]
    )

    # Train model button
    if st.sidebar.button("Train Model"):
        st.session_state.training_started = True
        train_model_and_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test)


def train_model_and_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test):
    models_to_train = [st.session_state.model_name]
    best_model = None
    best_score = -float('inf')
    best_model_name = None

    st.write(f"Training started with {st.session_state.h_opt_max_evals} hyperopt evaluations.")
    st.write(f"Evaluating models based on {st.session_state.best_metric}")

    for model_name in models_to_train:
        st.write(f"Training {model_name}...")
        model = train_model(
            X_train, y_train, X_valid, y_valid,
            model_name=model_name,
            h_opt_max_evals=st.session_state.h_opt_max_evals  # Number of hyperopt evaluations
        )
        # Evaluate on validation set
        eval_metrics_valid = evaluate_model(model=model, X=X_valid, y=y_valid, model_name=model_name, dataset_name="valid")
        st.write(f"Validation {st.session_state.best_metric.capitalize()} ({model_name}): {eval_metrics_valid[st.session_state.best_metric]:.4f}")
        
        # Select the best model based on the chosen metric
        if eval_metrics_valid[st.session_state.best_metric] > best_score:
            best_score = eval_metrics_valid[st.session_state.best_metric]
            best_model = model
            best_model_name = model_name

    st.write(f"Best model on validation set: {best_model_name} with {st.session_state.best_metric}: {best_score:.4f}")
    
    # Save the best model with the current date (dd_mm_yy)
    current_date = datetime.now().strftime("%d_%m_%y")
    best_model_fname = os.path.join(models_dir, f"{best_model_name}_best_model_{current_date}.joblib")
    joblib.dump(best_model, best_model_fname)
    st.write(f"Best model saved to {best_model_fname}")

    # Evaluate the best model on the test set
    eval_metrics_test = evaluate_model(model=best_model, X=X_test, y=y_test, model_name=best_model_name, dataset_name="test")
    st.write(f"Test {st.session_state.best_metric.capitalize()}: {eval_metrics_test[st.session_state.best_metric]:.4f}")
    st.write("Classification Report (Test Data):")
    st.dataframe(eval_metrics_test["conf_matrix_df"])

    st.session_state.training_completed = True

# Main function
def main():
    train_and_evaluate()

if __name__ == "__main__":
    main()
