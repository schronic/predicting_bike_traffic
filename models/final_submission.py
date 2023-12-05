import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

np.random.seed(42)
    
def _read_data(path, f_name):
    data = pd.read_parquet(os.path.join(path, "data", f_name))
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array


def get_train_data(path=".", basic=True):
    if basic:
        f_name = "train.parquet"
    else:
        f_name = "train_dropped.parquet.gzip"
        
    return _read_data(path, f_name)


def get_test_data(path="."):
    f_name = "test.parquet"
    return _read_data(path, f_name)


# Load training and test data
X_train, y_train = get_train_data("..", basic=False)
X_test, y_test = get_test_data("..")

# Path to the tuning results
path_tuning = "tuning_results"

def init_pipe(cyclic=True):
    """
    Initialize the preprocessing pipeline.
    
    Args:
        cyclic (bool): If True, use cyclic features. Otherwise, use one-hot encoding.
        
    Returns:
        A tuple of (feature_transformer, data_preprocessor).
    """
    # Feature categories
    if cyclic == True:
        date_features = ["hour_cos", "hour_sin", "month_cos", "month_sin", "weekday_cos", "weekday_sin", "day_cos", "day_sin",
                         "week_cos", "week_sin"]
        cyclic_condition = ("date_passthrough", "passthrough", date_features)
    else:
        date_features = ["hour", "month"]
        cyclic_condition = ("date_onehot", OneHotEncoder(handle_unknown="ignore"), date_features)
        # Removed: "day", "week", "weekday",
        
    numeric_features = ['temp', 'precip', 'cloudcover']
    # Removed: 'velib_mean', 'velib_std', 'velib_min', 'velib_25%', 'latitude', 'longitude', 'windspeed', 'humidity'
    
    categorical_features = ["counter_name", "site_name"]
    # Removed:  "year"
    
    binary_features = ["precipprob", "is_Bank_Holiday", "if_School_Holiday", "is_workday"]
    # Removed:  "is_Rush_Hour", "in_Lockdown"
    
    # Function to transform and add additional features
    feature_transformer = FunctionTransformer(feature_engineering._additional_features)

    # Preprocessing pipeline
    data_preprocessor = ColumnTransformer([
        ("numeric_scaler", StandardScaler(), numeric_features),
        ("categorical_onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("binary_passthrough", "passthrough", binary_features),
        cyclic_condition
    ])
    
    return feature_transformer, data_preprocessor

# Load the best hyperparameters from tuning results
xgb_params = pd.read_csv(os.path.join(path_tuning, 'tuning_XGB/best_True_False_all_cyclic_202312042346_results.csv'))
xgb_max_params = xgb_params.loc[xgb_params['mean_test_score'] == xgb_params['mean_test_score'].max()]
best_params = ast.literal_eval(xgb_max_params.params.values[0])
params_new = {key.split('__')[1]: value for key, value in best_params.items()}

# Define and fit the XGBoost regression model pipeline
model = XGBRegressor(**params_new)
feature_transformer, data_preprocessor = init_pipe()
xgb_reg_pipe = make_pipeline(feature_transformer, data_preprocessor, model)
xgb_reg_pipe.fit(X_train, y_train)

# Prepare the test data and generate predictions
final_test = pd.read_parquet(os.path.join("..", "data", "test_final.parquet"))
final_test_pred = xgb_reg_pipe.predict(final_test)

# Create and save the submission file
submission = pd.DataFrame(final_test_pred, columns=['log_bike_count']).reset_index()
submission.rename(columns={'index': 'Id'}, inplace=True)
submission.to_csv("best_True_False_202312042258_results.csv", index=False)
