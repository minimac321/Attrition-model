import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
import logging
import mlflow

def evaluate_model(model, X, y, model_name="Model", dataset_name="validation"):
    # with mlflow.start_run(run_name=f"Evaluation_{model_name}_{dataset_name}", nested=True):
    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:,1]
    
    inference_df = X.copy()
    inference_df["predictions"] = y_pred
    inference_df["predictions"] = y_pred
    inference_df["true_class"] = y
    
    # Evaluation Metrics
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y, y_proba)
    conf_matrix = confusion_matrix(y, y_pred)
    
    # Log confusion matrix as an artifact
    conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
        
    # Save evaluation metrics
    eval_metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'classification_report': report,
        'roc_auc_score': roc_auc,
        'confusion_matrix': conf_matrix.tolist(),
        'conf_matrix_df': conf_matrix_df,
        'inference_df': inference_df,
    }
    return eval_metrics


def load_inference_data(data_path: str, target_col: str, skip_dummy_data: bool = True) -> tuple:
    """
    Loads the test data from CSV file and splits into features and target.

    Args:
        data_path (str): Path to the test data CSV file.
        target_col (str): The name of the target column.
        skip_dummy_data (bool, optional): Whether to exclude dummy columns. Defaults to True.

    Returns:
        tuple: A tuple containing the features (X_test) and target (y_test) as pandas DataFrames.
    """
    logging.info(f"Loading inference data from {data_path}")
    df = pd.read_csv(data_path)
    
    if skip_dummy_data:
        cols_to_drop = [target_col] + [col for col in df.columns if "_dummy" in str(col)]    
        X = df.drop(columns=cols_to_drop, errors="ignore")
    else:
        X = df.copy()
    
    try:
        y = df[target_col]
    except:
        logging.error(f"Dataset has NO target variables")
        y = None
    
    logging.info(f"Data loaded with shape: {df.shape}")
    return X, y


def load_model(model_uri: str) -> object:
    """
    Loads the trained model from the given MLflow model URI.

    Args:
        model_uri (str): The URI of the MLflow model.

    Returns:
        object: The loaded model.
    """
    logging.info(f"Loading model from {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    return model