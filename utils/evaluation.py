import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score


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
    
    # Log metrics
    # mlflow.log_metric("roc_auc", roc_auc)
    # mlflow.log_metrics({
    #     "precision": report['1']['precision'],
    #     "recall": report['1']['recall'],
    #     "f1_score": report['1']['f1-score']
    # })
    
    # Log confusion matrix as an artifact
    conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    # conf_matrix_df.to_csv('confusion_matrix.csv')
    # mlflow.log_artifact('confusion_matrix.csv')
        
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