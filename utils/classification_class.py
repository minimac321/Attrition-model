import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import logging 

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score
)

# Add my parent directory to path variables
current_location = Path(os.path.abspath('')).resolve()
print(current_location)
sys.path.append(str(current_location))

# ===========================
# Evaluation Metrics Class
# ===========================

# Helper function to find the optimal threshold based on balanced accuracy
def find_best_threshold(y_true, y_probs):
    """
    Finds the best threshold to classify probabilities as True (1) or False (0)
    based on maximizing balanced accuracy.
    
    Parameters:
    y_true (array-like): True class labels (0 or 1).
    y_probs (array-like): Predicted probabilities for class 1.
    
    Returns:
    float: The best threshold for classification.
    """
    best_threshold = 0.5
    best_balanced_acc = 0
    thresholds = np.linspace(0, 1, 100)  # Test thresholds between 0 and 1
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_threshold = threshold
    
    return best_threshold


# ===========================
# Evaluation Metrics Class
# ===========================

class ClassificationMetrics:
    """
    A class to compute and store classification metrics.
    """

    def __init__(self, model, X_test, y_test, find_optimal_threshold=False, cutoff_point=0.5):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.metrics = {}
        self.y_pred_prob = None
        self.find_optimal_threshold = find_optimal_threshold
        self.cutoff_point = cutoff_point

    def compute_metrics(self):
        """
        Computes various classification metrics, using either a default threshold, 
        an optimal threshold, or a provided cutoff point.
        """
        # Get the predicted probabilities for class 1 (positive class)
        y_prob = self.model.predict_proba(self.X_test)[:, 1] if hasattr(self.model, "predict_proba") else None
        self.y_pred_prob = y_prob
        
        # Determine the cutoff point
        if y_prob is not None:
            if self.find_optimal_threshold:
                # Find the optimal threshold based on balanced accuracy
                optimal_threshold = find_best_threshold(self.y_test, y_prob)
                self.cutoff_point = optimal_threshold
                logging.info(f"Optimal Threshold Found: {optimal_threshold:.4f}")
            
            # Use the cutoff point to create binary predictions
            y_pred = (y_prob >= self.cutoff_point).astype(int)
        else:
            # If no probabilities are available, use standard model predictions
            y_pred = self.model.predict(self.X_test)
        
        # Assign predictions after threshold used
        self.preds = y_pred
        self.pred_probs = y_prob
        
        # Calculate metrics based on the predictions
        self.metrics['f1_score'] = f1_score(self.y_test, y_pred, average='weighted')
        self.metrics['accuracy'] = accuracy_score(self.y_test, y_pred)
        self.metrics['balanced_accuracy'] = balanced_accuracy_score(self.y_test, y_pred)
        self.metrics['precision'] = precision_score(self.y_test, y_pred, average='weighted')
        self.metrics['recall'] = recall_score(self.y_test, y_pred, average='weighted')
        
        if y_prob is not None:
            self.metrics['roc_auc'] = roc_auc_score(self.y_test, y_prob, average='weighted')

    def run_evaluation(self, metrics_dir: str = None):
        """
        Runs all evaluation steps and saves outputs.
        """
        self.compute_metrics()
        
        current_date = datetime.now().strftime("%d_%m_%y")

        # Save metrics to a CSV
        metrics_df = pd.DataFrame([self.metrics])   
        
        if metrics_dir is not None:
            metrics_csv_path = os.path.join(metrics_dir, f"classification_metrics_{current_date}.csv")     
            metrics_df.to_csv(metrics_csv_path, index=False)
        
            if mlflow.active_run():
                mlflow.log_artifact(metrics_csv_path)


    def get_metrics_dict(self):
        """
        Returns the metrics as a dictionary.
        """
        return self.metrics

    def get_predictions(self):
        """
        Returns the predictions.
        """
        return self.preds

    def get_probability_predictions(self):
        """
        Returns the predictions.
        """
        return self.pred_probs