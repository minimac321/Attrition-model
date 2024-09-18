

# src/data_preprocessing.py
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import RandomizedSearchCV
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def split_and_save_data(input_df: pd.DataFrame, target_col: str, processed_dir:str, random_state):
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
    
    train_df.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(processed_dir, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)
    

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_valid': X_valid,
        'y_valid': y_valid,
        'X_test': X_test,
        'y_test': y_test
    }


# Define a function for preprocessing pipeline
def create_preprocessing_pipeline(X_train):
    """
    Create a preprocessing pipeline to scale numeric features and one-hot encode categorical features.
    """
    # Identify numeric and categorical columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    # Create the preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine numeric and categorical pipelines
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor


# Train model function with hyperparameter tuning (RandomizedSearchCV or Hyperopt)
def train_model(
    X_train, y_train, X_valid, y_valid, 
    model_name: str, 
    use_hyperopt: bool =True, 
    h_opt_max_evals: int = 30,
    random_state: int=42
):
    """
    Train a model (RandomForest or XGBoost) with hyperparameter tuning using either RandomizedSearchCV or Hyperopt.

    :param X_train: Training features.
    :param y_train: Training labels.
    :param X_valid: Validation features.
    :param y_valid: Validation labels.
    :param model_name: Either 'RandomForest' or 'XGBoost'.
    :param use_hyperopt: If True, use Hyperopt for Bayesian optimization; otherwise, use RandomizedSearchCV.
    :param random_state: Random state for reproducibility.
    :return: The best model after hyperparameter tuning.
    """

    # Get model and hyperparameter distributions
    model, param_distributions = get_model_and_params(model_name, random_state)

    # Create a preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(X_train)

    # Combine preprocessing and model into one pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Perform hyperparameter tuning with either RandomizedSearchCV or Hyperopt
    if use_hyperopt:
        print("Using Hyperopt for Bayesian optimization")

        # Define the objective function for hyperopt
        def objective(params):
            # Update the parameters in the pipeline
            pipeline.set_params(**params)
            # Fit the model
            pipeline.fit(X_train, y_train)
            # Predict on the validation set
            y_pred = pipeline.predict(X_valid)
            # Calculate F1 score for class imbalance
            f1 = f1_score(y_valid, y_pred, average='weighted')
            return {'loss': -f1, 'status': STATUS_OK}

        # Define the search space for Hyperopt
        if model_name == 'RandomForest':
            space = {
                'classifier__n_estimators': scope.int(hp.randint('classifier__n_estimators', 100, 501)),
                'classifier__max_depth': scope.int(hp.randint('classifier__max_depth', 3, 41)),
                'classifier__min_samples_split': scope.int(hp.randint('classifier__min_samples_split', 2, 11)),
                'classifier__min_samples_leaf': scope.int(hp.randint('classifier__min_samples_leaf', 1, 5)),
                'classifier__max_features': hp.uniform('classifier__max_features', 0, 1.0)
            }
        elif model_name == 'XGBoost':
            space = {
                'classifier__n_estimators': scope.int(hp.randint('classifier__n_estimators', 100, 501)),
                'classifier__max_depth': scope.int(hp.randint('classifier__max_depth', 3, 21)),
                'classifier__learning_rate': hp.uniform('classifier__learning_rate', 0.01, 0.2),
                'classifier__subsample': hp.uniform('classifier__subsample', 0.6, 1.0),
                'classifier__colsample_bytree': hp.uniform('classifier__colsample_bytree', 0.6, 1.0)
            }

        # Define trials object to store the history of optimization
        trials = Trials()

        # Run hyperopt optimization
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=h_opt_max_evals,
            trials=trials
        )

        # Update the pipeline with the best parameters found by hyperopt
        best["classifier__max_depth"] = int(best["classifier__max_depth"])
        print(f"Best set of params found: {best}")

        pipeline.set_params(**best)
        
        # Fit the model on the entire training data
        pipeline.fit(X_train, y_train)

    else:
        print("Using RandomizedSearchCV for hyperparameter tuning")
        
        # Perform RandomizedSearchCV for hyperparameter tuning
        random_search = RandomizedSearchCV(
            pipeline, param_distributions=param_distributions,
            n_iter=10, 
            cv=5, 
            scoring='f1',  # Use F1 score for class imbalance
            random_state=random_state,
            verbose=1, 
            n_jobs=-1
        )
        
        # Fit the model on the training data
        random_search.fit(X_train, y_train)

        # Return the best estimator from RandomizedSearchCV
        pipeline = random_search.best_estimator_

    # Print best set of params
    print(f"Best set of params found: {best}")

    # Evaluate the model on the validation set
    y_valid_pred = pipeline.predict(X_valid)
    validation_accuracy = accuracy_score(y_valid, y_valid_pred)
    print(f"Validation Accuracy: {validation_accuracy:.4f}")
    print("Classification Report (Validation Data):")
    print(classification_report(y_valid, y_valid_pred))

    return pipeline


# Get model and hyperparameter search space
def get_model_and_params(model_name, random_state=42):
    if model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=random_state, class_weight='balanced')
        param_distributions = {
            'classifier__n_estimators': [100, 200, 300, 400, 500],
            'classifier__max_depth': [None, 10, 20, 30, 40, 50],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2']
        }
    elif model_name == 'XGBoost':
        model = XGBClassifier(random_state=random_state, eval_metric='logloss')
        param_distributions = {
            'classifier__n_estimators': [100, 200, 300, 400, 500],
            'classifier__max_depth': [3, 5, 7, 9, 11],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0]
        }
    else:
        raise ValueError("Unsupported model name")
    
    return model, param_distributions
