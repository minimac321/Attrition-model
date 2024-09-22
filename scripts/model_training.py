import logging
import os
from pathlib import Path
import sys
import pandas as pd

import os
import sys
import logging
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Add my parent directory to path variables
current_location = Path(os.path.abspath("")).resolve()
print(current_location)
sys.path.append(str(current_location))

from utils.train import (
    create_pipeline,
    define_hyperparam_space,
    load_train_valid_data,
    tune_hyperparameters,
)
from utils.visualisation import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_permutation_importance,
    plot_roc_curve,
    plot_shap_values,
)
from utils.classification_class import ClassificationMetrics
from utils.utils import get_or_create_experiment, setup_logging

setup_logging(logger_level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.info("Starting model training pipeline.")


def setup_directories(current_location: Path) -> tuple:
    """
    Sets up necessary directories for transformed data and outputs.

    Args:
        current_location (Path): The current directory where data and output folders will be created.

    Returns:
        tuple: Paths to transformed data directory, output data directory, evaluation directory, and metrics directory.
    """
    data_dir = os.path.join(current_location, "data")
    os.makedirs(data_dir, exist_ok=True)

    transformed_data_dir = os.path.join(data_dir, "transformed_data")
    os.makedirs(transformed_data_dir, exist_ok=True)

    output_data_dir = os.path.join(data_dir, "output_data")
    os.makedirs(output_data_dir, exist_ok=True)

    # Subdirectories for evaluation outputs
    evaluation_dir = os.path.join(data_dir, "evaluation")
    metrics_dir = os.path.join(evaluation_dir, "metrics")

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)

    return transformed_data_dir, output_data_dir, evaluation_dir, metrics_dir


def run_evaluation_with_plots(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics_dir: str,
    evaluation_dir: str,
    show_plots: bool = False,
) -> object:
    """
    Runs evaluation and generates plots.

    Args:
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): Test feature data.
        y_test (pd.Series): Test labels.
        metrics_dir (str): Directory to save metrics.
        evaluation_dir (str): Directory to save evaluation outputs.
        show_plots (bool, optional): Whether to display the plots. Defaults to False.

    Returns:
        object: A ClassificationMetrics instance containing the evaluation results.
    """
    selected_feature_names = list(X_test.columns)

    # Create a ClassificationMetrics instance to compute the metrics
    classifier_metrics = ClassificationMetrics(model, X_test, y_test, find_optimal_threshold=True)
    classifier_metrics.run_evaluation(metrics_dir)

    # Feature Importance Plot
    save_path_fe = None
    if evaluation_dir:
        save_path_fe = os.path.join(evaluation_dir, f"feature_importance.png")

    plot_feature_importance(
        model=model,
        feature_names=selected_feature_names,
        save_path=save_path_fe,
        show_plot=show_plots,
    )

    # Permutation Importance Plot
    save_path_pi = None
    if evaluation_dir:
        save_path_pi = os.path.join(evaluation_dir, f"permutation_importance.png")

    plot_permutation_importance(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=selected_feature_names,
        save_path=save_path_pi,
        show_plot=show_plots,
    )
    try:
        # SHAP Values Plot
        save_path_shap = None
        if evaluation_dir:
            save_path_shap = os.path.join(evaluation_dir, f"shap_summary.png")

        plot_shap_values(
            model=model,
            X_test=X_test,
            save_path=save_path_shap,
            show_plot=show_plots,
        )
    except:
        logging.error(f"SHAP values failed too plot - skipping")

    # Confusion Matrix Plot
    save_path_cm = None
    if evaluation_dir:
        save_path_cm = os.path.join(evaluation_dir, f"confusion_matrix.png")

    plot_confusion_matrix(model=model, X_test=X_test, y_test=y_test, save_path=save_path_cm)

    return classifier_metrics


def main():
    """
    Main function to orchestrate model training and evaluation.
    """
    # Setup directories
    transformed_data_dir, output_data_dir, evaluation_dir, metrics_dir = setup_directories(
        current_location
    )

    # Setup MLflow experiment
    experiment_id = get_or_create_experiment(
        experiment_name="EmployeeAttrition_Experiment", current_location=current_location
    )
    experiment = mlflow.set_experiment(experiment_id=experiment_id)

    # Load data
    X_train, y_train, X_valid, y_valid = load_train_valid_data(
        target="target",
        transformed_data_dir=transformed_data_dir,
    )
    # Create pipeline
    model_name = "XGBoost"  # Options: [XGBoost, DecisionTree, RandomForrest]
    pipeline = create_pipeline(X_train, model_name)

    # Define hyperparameter space
    space = define_hyperparam_space(model_name)
    logging.debug(f"Hyperparameter space:\n{space}")

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters space (not the actual parameters)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("hyperopt_space", space)

        # Hyperparameter tuning
        pipeline, best_params = tune_hyperparameters(
            pipeline=pipeline,
            space=space,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            n_trials=15,
        )

        # Fit the model on the combined train and validation set
        include_validation_data = False
        if include_validation_data:
            X_combined = pd.concat([X_train, X_valid], axis=0)
            y_combined = pd.concat([y_train, y_valid], axis=0)
            pipeline.fit(X_combined, y_combined)
            logging.info("Pipeline fitted on combined training and validation data.")
        else:
            pipeline.fit(X_train, y_train)
            logging.info("Pipeline fitted on only training data.")

        # Log the best hyperparameters
        all_params = pipeline.named_steps["classifier"].get_params()
        mlflow.log_params(all_params)

        # Call the ROC plotting function
        plot_roc_curve(
            model=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            save_path=os.path.join(evaluation_dir, f"roc_curve.png"),
        )

        # Evaluate performance on test set
        classifier_metrics = run_evaluation_with_plots(
            model=pipeline,
            X_test=X_valid,
            y_test=y_valid,
            metrics_dir=metrics_dir,
            evaluation_dir=evaluation_dir,
        )
        validation_metrics = classifier_metrics.get_metrics_dict()

        # Log metrics to MLflow
        mlflow.log_metrics(validation_metrics)
        logging.info(f"validation_metrics: {validation_metrics}")

        # Log the model with input example and signature
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=X_valid.head(5),  # Providing a small example input
            signature=infer_signature(X_valid, pipeline.predict(X_valid)),
        )
        mlflow.sklearn.log_model(pipeline, "model")
        logging.info("Model logged to MLflow.")

    logging.info(f"Finished with Model uri: {str(model_info.model_uri)}")
    logging.info("Model training pipeline completed successfully.")


if __name__ == "__main__":
    main()
