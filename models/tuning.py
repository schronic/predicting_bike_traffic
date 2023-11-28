import datetime
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump
import sys

import feature_engineering
import problem


def optimize_model(model, param_grid, save_path, n_iter, parallel_jobs: int = 1):
    """
    Optimizes a given estimator using grid search and saves the results.

    Parameters:
    - model: The machine learning model/estimator to be optimized.
    - param_grid: Parameter grid for grid search.
    - save_path: Directory path to save the results.
    - parallel_jobs: Number of jobs to run in parallel (default is 1).
    """
    # Generating a timestamp for file naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    
    
    # Feature categories
    date_features = ["hour_cos", "hour_sin", "weekday_cos", "weekday_sin", 
                 "day_cos", "day_sin", "week_cos", "week_sin", "month_cos", "month_sin"]

    numeric_features = ['temp', 'humidity', 'precip', 'cloudcover']
    # Removed: 'velib_mean', 'velib_std', 'velib_min', 'velib_25%', 'latitude', 'longitude', 'windspeed' 

    categorical_features = ["counter_name", "site_name"]
    # Removed:  "year"

    binary_features = ["precipprob", "in_Lockdown", "is_Bank_Holiday", 
                       "if_School_Holiday", "is_workday"]
    # Removed:  "is_Rush_Hour"

    # Function to transform and add additional features
    feature_transformer = FunctionTransformer(feature_engineering._additional_features)

    # Preprocessing pipeline
    data_preprocessor = ColumnTransformer([
        ("numeric_scaler", StandardScaler(), numeric_features),
        ("categorical_onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("binary_passthrough", "passthrough", binary_features),
        ("date_onehot", "passthrough", date_features)
    ])

    # Full pipeline including model
    full_pipeline = make_pipeline(feature_transformer, data_preprocessor, model)

    # Loading training data
    X_train, y_train = problem.get_train_data('..')
    

    # Time series cross-validation
    cv_split = problem.get_cv(X_train, y_train)

    # Grid search setup
    grid_search = RandomizedSearchCV(
        full_pipeline,
        param_grid,
        cv=cv_split,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
        n_jobs=parallel_jobs,
        verbose=10,
        error_score='raise'
    )
    grid_search.fit(X_train, y_train)

    # Saving cross-validation results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results_file = f"{timestamp}_results.csv"
    cv_results.to_csv(os.path.join(save_path, cv_results_file), index=False)