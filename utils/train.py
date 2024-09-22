import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import os
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import logging


# Add my parent directory to path variables
current_location = Path(os.path.abspath("")).resolve()
print(current_location)
sys.path.append(str(current_location))

from utils.constants import random_value


def load_train_valid_data(target, transformed_data_dir, skip_corr_columns: bool = True):
    """
    Reads in the train and validation data from CSV files.
    """
    train_path = os.path.join(transformed_data_dir, "train_script.csv")
    val_path = os.path.join(transformed_data_dir, "validation_script.csv")

    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        logging.info("Successfully loaded train and validation  data.")
        logging.info(f"Train df: {train_df.shape}")
        logging.info(f"Validation df: {val_df.shape}")

        # Skip Highly Correlated Columns:
        skip_corr_columns = True
        if skip_corr_columns:
            corr_cols_to_skip = [
                "YearsWithCurrManager",
                "YearsSinceLastPromotion",
                "YearsInCurrentRole",
            ]
            train_df = train_df.drop(columns=corr_cols_to_skip, errors="ignore")
            val_df = val_df.drop(columns=corr_cols_to_skip, errors="ignore")

        # Skip dummy columns
        skip_dummy_columns = True
        if skip_dummy_columns:
            dummy_cols_to_drop = [col for col in train_df.columns if "_dummy" in str(col)]
            train_df = train_df.drop(columns=dummy_cols_to_drop, errors="ignore")
            val_df = val_df.drop(columns=dummy_cols_to_drop, errors="ignore")

        keep_subset_cols = False
        if keep_subset_cols:
            cols_to_kep = [
                "target",
                "Age",
                "YearsAtCompany",
                "MonthlyIncome",
                "DistanceFromHome",
                "PerformanceRating",
            ]
            train_df = train_df[cols_to_kep]
            val_df = val_df[cols_to_kep]

        # Separate features and target
        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]
        X_valid = val_df.drop(columns=[target])
        y_valid = val_df[target]

    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}")
        sys.exit(1)

    return X_train, y_train, X_valid, y_valid


def split_and_save_data(input_df: pd.DataFrame, target_col: str, processed_dir: str, random_state):
    # Define features and target
    X = input_df.drop(target_col, axis=1)
    y = input_df[target_col]

    # Split into train (70%), validation (15%), and test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    # Save splits
    train_df = pd.concat([X_train, y_train], axis=1)
    valid_df = pd.concat([X_valid, y_valid], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(processed_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test,
    }


# Get model and hyperparameter search space
def get_model_and_params(model_name, random_state=42):
    if model_name == "RandomForest":
        model = RandomForestClassifier(random_state=random_state, class_weight="balanced")
        param_distributions = {
            "classifier__n_estimators": [100, 200, 300, 400, 500],
            "classifier__max_depth": [None, 10, 20, 30, 40, 50],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
        }
    elif model_name == "XGBoost":
        model = XGBClassifier(random_state=random_state, eval_metric="logloss")
        param_distributions = {
            "classifier__n_estimators": [100, 200, 300, 400, 500],
            "classifier__max_depth": [3, 5, 7, 9, 11],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        }
    else:
        raise ValueError("Unsupported model name")

    return model, param_distributions


# ===========================
# Create Pipeline
# ===========================
def create_pipeline(X_train: pd.DataFrame, model_name: str) -> Pipeline:
    """
    Creates the preprocessing and classification pipeline.

    Args:
        X_train (pd.DataFrame): The training dataset.
        model_name (str): The name of the model to be used in the pipeline.

    Returns:
        Pipeline: The complete preprocessing and classification pipeline.
    """
    # Identify feature types
    bool_features = X_train.select_dtypes(include=["bool"]).columns.tolist()
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    assert (
        len(bool_features) + len(numeric_features) + len(categorical_features) == X_train.shape[1]
    ), "Mismatch in feature categorization."

    logging.info(f"Boolean features: {bool_features}")
    logging.info(f"Numeric features: {numeric_features}")
    logging.info(f"Categorical features: {categorical_features}")

    # Define transformers
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    def convert_bool_to_int(x):
        return x.astype(int)

    boolean_transformer = Pipeline(
        steps=[("convert_bool", FunctionTransformer(convert_bool_to_int))]
    )

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bool", boolean_transformer, bool_features),
            ("cat", "passthrough", categorical_features),
        ]
    )

    # Define the classifier
    if model_name == "XGBoost":
        classifier = XGBClassifier(
            random_state=random_value,
            eval_metric="logloss",
            reg_lambda=3.0,  # L2 regularization (Default is 1.0)
            reg_alpha=1.5,
            gamma=2,
        )
    elif model_name == "DecisionTree":
        classifier = DecisionTreeClassifier(
            random_state=random_value,
        )
    elif model_name == "RandomForrest":
        classifier = RandomForestClassifier(
            random_state=random_value,
        )
    else:
        print("No Model selected")
        sys.exit()

    # Create the full pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    logging.info("Pipeline created successfully.")

    return pipeline


# ===========================
# Define Hyperparameter Space
# ===========================
def define_hyperparam_space(model_name: str) -> dict:
    """
    Defines the hyperparameter space for tuning based on the model selected.

    Args:
        model_name (str): The name of the model for which the hyperparameter space is defined.

    Returns:
        dict: A dictionary defining the hyperparameter space.
    """
    if model_name == "XGBoost":
        space = {
            "classifier__n_estimators": scope.int(hp.randint("classifier__n_estimators", 10, 301)),
            "classifier__max_depth": scope.int(hp.randint("classifier__max_depth", 1, 20)),
            "classifier__learning_rate": hp.uniform("classifier__learning_rate", 0.01, 0.3),
            "classifier__colsample_bytree": hp.uniform("classifier__colsample_bytree", 0.1, 1.0),
            "classifier__colsample_bylevel": hp.uniform("classifier__colsample_bylevel", 0.4, 1.0),
            "classifier__min_child_weight": scope.int(
                hp.randint("classifier__min_child_weight", 1, 100)
            ),
        }

    elif model_name == "DecisionTree":
        space = {
            "classifier__max_depth": scope.int(hp.randint("classifier__max_depth", 2, 20)),
            "classifier__min_samples_split": scope.int(
                hp.randint("classifier__min_samples_split", 2, 20)
            ),
            "classifier__min_samples_leaf": scope.int(
                hp.randint("classifier__min_samples_leaf", 1, 10)
            ),
        }
    elif model_name == "RandomForrest":
        space = {
            "classifier__n_estimators": scope.int(hp.randint("classifier__n_estimators", 10, 300)),
            "classifier__max_depth": scope.int(hp.randint("classifier__max_depth", 2, 20)),
            "classifier__min_samples_split": scope.int(
                hp.randint("classifier__min_samples_split", 2, 21)
            ),
            "classifier__min_samples_leaf": scope.int(
                hp.randint("classifier__min_samples_leaf", 1, 11)
            ),
        }

    else:
        print("No valid model selected")
        sys.exit()

    return space


# ===========================
# Hyperparameter Tuning
# ===========================
def tune_hyperparameters(
    pipeline: Pipeline,
    space: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    n_trials: int = 10,
) -> tuple:
    """
    Runs hyperparameter tuning using Hyperopt.

    Args:
        pipeline (Pipeline): The machine learning pipeline to be tuned.
        space (dict): The hyperparameter space to explore.
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training labels.
        X_valid (pd.DataFrame): Validation feature data.
        y_valid (pd.Series): Validation labels.
        n_trials (int, optional): The number of hyperparameter tuning trials. Defaults to 10.

    Returns:
        tuple: The updated pipeline and best hyperparameters.
    """
    from hyperopt import STATUS_OK

    def objective(params):
        """
        Objective function for Hyperopt.
        """
        pipeline.set_params(**params)

        add_cv_folds = False
        if add_cv_folds:
            # Perform cross-validation
            X_combined = pd.concat([X_train, X_valid], axis=0)
            y_combined = pd.concat([y_train, y_valid], axis=0)
            y_combined_pred = cross_val_predict(pipeline, X_combined, y_combined, cv=4)
            score = f1_score(y_combined, y_combined_pred, average="weighted")
        else:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_valid)
            score = f1_score(y_valid, y_pred, average="weighted")

        logging.info(f"F1 Score: {score} with params: {params}")
        return {"loss": -score, "status": STATUS_OK}

    logging.info("Starting hyperparameter tuning using Hyperopt.")
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=trials,
        rstate=np.random.default_rng(random_value),
    )
    logging.info(f"Best hyperparameters found: {best_params}")

    # Update pipeline with best parameters
    pipeline.set_params(**best_params)
    logging.info("Pipeline updated with best hyperparameters.")

    return pipeline, best_params
