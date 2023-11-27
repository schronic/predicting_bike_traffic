#feature_engineering.add_additional_features

import pandas as pd
import numpy as np
from jours_feries_france.compute import JoursFeries

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
    return X.drop(columns=['date', 'hour', 'weekday', 'day', 'week', 'month'])

def check_school_holidays(X):
    """
    Checks if the dates in DataFrame X are school holidays.

    Parameters:
    - X: DataFrame with a 'date' column.

    Returns:
    - Series indicating if the dates are school holidays.
    """

    school_holidays = pd.read_csv('../data/school_holidays.csv')
        
    
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
    years = range(X['year'].min(), X['year'].max() + 1)
    bank_holidays = []
    for year in years:
        bank_holidays.extend(JoursFeries.for_year(year).values())

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
    
    weather_data = pd.read_csv('../data/scaled_weather_data.csv')
    weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
    
    # Merging weather data
    # https://www.visualcrossing.com/weather-data
    weather_data = weather_data.loc[(weather_data['datetime'] >= X['date'].min()) & 
                                    (weather_data['datetime'] <= X['date'].max())]
    merged_data = pd.merge(X, weather_data, how='left', left_on='date', right_on='datetime')
    merged_data.drop(columns=['datetime'], inplace=True)

    # Merging velib statistics
    # https://opendata.paris.fr/explore/dataset/velib-disponibilite-en-temps-reel/information/?disjunctive.name&disjunctive.is_installed&disjunctive.is_renting&disjunctive.is_returning&disjunctive.nom_arrondissement_communes
    velib_stats = pd.read_csv('../data/velib_processed.csv')
    merged_data = pd.merge(merged_data, velib_stats, how='left', on='site_id')

    return merged_data

def _additional_features(X):
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

