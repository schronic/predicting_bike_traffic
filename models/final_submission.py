import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

np.random.seed(42)


def calculate_cyclic_time(X, column, cycle_length):
    """
    Calculates the cosine and sine transformations for cyclic time features.

    Parameters:
    - X: DataFrame containing the time-related column.
    - column: Column name for the time-related feature.
    - cycle_length: The length of the time cycle (e.g., 24 for hours, 7 for weekdays).

    Returns:
    - cos_transform: Cosine transformation of the time feature.
    - sin_transform: Sine transformation of the time feature.
    """
    values = X[column].unique()
    cos_transform = np.cos(values / cycle_length * 2 * np.pi)
    sin_transform = np.sin(values / cycle_length * 2 * np.pi)

    return cos_transform, sin_transform

def encode_date_features(X):
    """
    Encodes date-related features in the DataFrame.

    Parameters:
    - X: DataFrame with a 'date' column.

    Returns:
    - DataFrame with additional encoded date-related features.
    """
    X = X.copy()  # Working with a copy to avoid modifying the original DataFrame
    X['date'] = pd.to_datetime(X['date'])

    # Extracting date components
    X['year'] = X['date'].dt.year
    X['week'] = X['date'].dt.isocalendar().week.astype(np.int64)
    X['month'] = X['date'].dt.month
    X['day'] = X['date'].dt.day
    X['weekday'] = X['date'].dt.weekday
    X['hour'] = X['date'].dt.hour

    # Additional binary features
    X['is_workday'] = (X['weekday'] < 5).astype(int)
    X['if_School_Holiday'] = check_school_holidays(X)
    X['is_Bank_Holiday'] = check_bank_holidays(X)
    X['is_Rush_Hour'] = ((X['hour'] >= 6) & (X['hour'] <= 8)) | ((X['hour'] >= 17) & (X['hour'] <= 20)).astype(int)
    X['in_Lockdown'] = check_lockdown_dates(X)

    # Apply cyclic encoding
    for col, cycle_length in [('hour', 24), ('weekday', 7), ('day', 31), ('week', 52), ('month', 12)]:
        cos_transform, sin_transform = calculate_cyclic_time(X, col, cycle_length)
        X[f'{col}_cos'] = X[col].replace(X[col].unique(), sin_transform)
        X[f'{col}_sin'] = X[col].replace(X[col].unique(), cos_transform)

    # Dropping the original date-related columns
    return X#.drop(columns=['date', 'hour', 'weekday', 'day', 'week', 'month'])

def check_school_holidays(X):
    """
    Checks if the dates in DataFrame X are school holidays.

    Parameters:
    - X: DataFrame with a 'date' column.

    Returns:
    - Series indicating if the dates are school holidays.
    """

    school_holidays = pd.read_csv('/kaggle/input/school-holidays/school_holidays.csv')
        
    
    school_holidays['date'] = pd.to_datetime(school_holidays['date'])
    relevant_holidays = school_holidays.loc[(school_holidays['date'].dt.date >= X['date'].dt.date.min()) & 
                                            (school_holidays['date'].dt.date <= X['date'].dt.date.max())]
    relevant_holidays = relevant_holidays[relevant_holidays['vacances_zone_c']].date.dt.date.values

    return X['date'].isin(relevant_holidays).astype(int)

def check_bank_holidays(X):
    """
    Checks if the dates in DataFrame X are bank holidays.

    Parameters:
    - X: DataFrame with a 'date' column.

    Returns:
    - Series indicating if the dates are bank holidays.
    """
    holidays = pd.read_csv('/kaggle/input/bank-holidays/bank_holidays.csv')['0']
    bank_holidays = pd.to_datetime(holidays, format="%m%d%Y").dt.date.values.tolist()

    return X['date'].isin(bank_holidays).astype(int)

def check_lockdown_dates(X):
    """
    Checks if the dates in DataFrame X fall within lockdown periods.

    Parameters:
    - X: DataFrame with a 'date' column.

    Returns:
    - Series indicating if the dates are within lockdown periods.
    """
    first_lockdown = pd.date_range(start="2020-10-30", end="2020-12-15")
    second_lockdown = pd.date_range(start="2021-03-20", end="2021-04-30")
    lockdown_dates = first_lockdown.union(second_lockdown)

    return X['date'].isin(lockdown_dates.date).astype(int)

def add_weather_and_velib_features(X):
    """
    Adds weather and velib statistics to the DataFrame.

    Parameters:
    - X: DataFrame with date information.
    - weather_data: DataFrame with weather data.
    - velib_stats: DataFrame with velib statistics.

    Returns:
    - DataFrame with merged weather and velib data.
    """
    X = X.copy()  # Working with a copy to avoid modifying the original DataFrame
    
    weather_data = pd.read_csv('/kaggle/input/scaledweatherdata/scaled_weather_data.csv')
    weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
    
    # Merging weather data
    # https://www.visualcrossing.com/weather-data
    weather_data = weather_data.loc[(weather_data['datetime'] >= X['date'].min()) & 
                                    (weather_data['datetime'] <= X['date'].max())]
    merged_data = pd.merge(X, weather_data, how='left', left_on='date', right_on='datetime')
    merged_data.drop(columns=['datetime'], inplace=True)

    return merged_data

def additional_features(X):
    """
    Applies additional feature transformations to the DataFrame.

    Parameters:
    - X: DataFrame with initial data.
    - weather_data: DataFrame containing weather data.
    - velib_stats: DataFrame containing velib statistics.

    Returns:
    - DataFrame with additional transformed features.
    """
    X = X.copy()  # Working with a copy to avoid modifying the original DataFrame
    
    merged_data = add_weather_and_velib_features(X)
    final_data = encode_date_features(merged_data)

    return final_data


_target_column_name = "log_bike_count"

# Load training and test data
data = pd.read_parquet('/kaggle/input/train-dropped-nan/train_dropped.parquet.gzip')
# Sort by date first, so that time based cross-validation would produce correct results
data = data.sort_values(["date", "counter_name"])
y_train = data[_target_column_name].values
X_train = data.drop([_target_column_name, "bike_count"], axis=1)

data = pd.read_parquet('/kaggle/input/testing-file/test.parquet')
# Sort by date first, so that time based cross-validation would produce correct results
data = data.sort_values(["date", "counter_name"])
y_test = data[_target_column_name].values
X_test = data.drop([_target_column_name, "bike_count"], axis=1)


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
    feature_transformer = FunctionTransformer(additional_features)

    # Preprocessing pipeline
    data_preprocessor = ColumnTransformer([
        ("numeric_scaler", StandardScaler(), numeric_features),
        ("categorical_onehot", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("binary_passthrough", "passthrough", binary_features),
        cyclic_condition
    ])
    
    return feature_transformer, data_preprocessor

params = {'colsample_bylevel': 0.5041435472696711,
          'colsample_bynode': 0.537175516017131,
          'colsample_bytree': 0.9467806263705418,
          'gamma': 4.59514979237692,
          'learning_rate': 0.10126221981149418,
          'max_depth': 32,
          'n_estimators': 238,
          'reg_alpha': 0.7716220829480913,
          'reg_lambda': 0.38660506655992255,
          'subsample': 0.777860931304462
         }
    
# Define and fit the XGBoost regression model pipeline
model = XGBRegressor(**params)

feature_transformer, data_preprocessor = init_pipe()
xgb_reg_pipe = make_pipeline(feature_transformer, data_preprocessor, model)
xgb_reg_pipe.fit(X_train, y_train)

# Prepare the test data and generate predictions
final_test = pd.read_parquet(os.path.join("/kaggle/input/mdsb-2023/final_test.parquet"))
final_test_pred = xgb_reg_pipe.predict(final_test)

# Create and save the submission file
submission = pd.DataFrame(final_test_pred, columns=['log_bike_count']).reset_index()
submission.rename(columns={'index': 'Id'}, inplace=True)
submission.to_csv("/kaggle/working/submission.csv", index=False)
