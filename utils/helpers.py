"""
Helper utilities for Sea Level Predictor.
Common functions used across the application.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import pickle
from utils.logger import get_logger

logger = get_logger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        file_path: Path to the data file
    
    Returns:
        Loaded DataFrame
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix == '.json':
        return pd.read_json(file_path)
    elif file_path.suffix == '.parquet':
        return pd.read_parquet(file_path)
    elif file_path.suffix == '.pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def save_data(data: pd.DataFrame, file_path: str, format: str = 'csv') -> None:
    """
    Save data to various file formats.
    
    Args:
        data: DataFrame to save
        file_path: Path to save the file
        format: File format ('csv', 'json', 'parquet', 'pickle')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        data.to_csv(file_path, index=False)
    elif format == 'json':
        data.to_json(file_path, orient='records', indent=2)
    elif format == 'parquet':
        data.to_parquet(file_path, index=False)
    elif format == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Data saved to {file_path}")

def create_date_range(start_date: str, end_date: str, freq: str = 'D') -> pd.DatetimeIndex:
    """
    Create a date range for data collection.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        freq: Frequency ('D' for daily, 'H' for hourly)
    
    Returns:
        DatetimeIndex
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)

def calculate_statistics(data: pd.DataFrame, column: str) -> Dict[str, float]:
    """
    Calculate basic statistics for a column.
    
    Args:
        data: DataFrame
        column: Column name
    
    Returns:
        Dictionary of statistics
    """
    stats = {
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'median': data[column].median(),
        'count': data[column].count(),
        'missing': data[column].isnull().sum()
    }
    return stats

def detect_anomalies(data: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """
    Detect anomalies in data using various methods.
    
    Args:
        data: DataFrame
        column: Column to analyze
        method: Detection method ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for anomaly detection
    
    Returns:
        Boolean series indicating anomalies
    """
    if method == 'iqr':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data[column] < lower_bound) | (data[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        return z_scores > threshold
    
    else:
        raise ValueError(f"Unsupported method: {method}")

def resample_data(data: pd.DataFrame, date_column: str, value_column: str, freq: str = 'D') -> pd.DataFrame:
    """
    Resample time series data to a different frequency.
    
    Args:
        data: DataFrame with time series data
        date_column: Name of the date column
        value_column: Name of the value column
        freq: Target frequency ('D', 'H', 'M', etc.)
    
    Returns:
        Resampled DataFrame
    """
    data_copy = data.copy()
    data_copy[date_column] = pd.to_datetime(data_copy[date_column])
    data_copy.set_index(date_column, inplace=True)
    
    resampled = data_copy[value_column].resample(freq).mean()
    return resampled.reset_index()

def validate_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame has required columns and data types.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a number with appropriate units and decimal places.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
    
    Returns:
        Formatted string
    """
    if abs(value) >= 1e6:
        return f"{value/1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"

def get_station_info(station_id: str) -> Dict[str, Any]:
    """
    Get information about a NOAA station.
    
    Args:
        station_id: NOAA station ID
    
    Returns:
        Dictionary with station information
    """
    # This would typically fetch from NOAA API
    # For now, return a basic structure
    station_info = {
        'id': station_id,
        'name': f'Station {station_id}',
        'location': 'Unknown',
        'latitude': 0.0,
        'longitude': 0.0,
        'timezone': 'UTC'
    }
    return station_info

def calculate_trend(data: pd.DataFrame, date_column: str, value_column: str) -> Dict[str, float]:
    """
    Calculate linear trend in time series data.
    
    Args:
        data: DataFrame with time series data
        date_column: Name of the date column
        value_column: Name of the value column
    
    Returns:
        Dictionary with trend information
    """
    data_copy = data.copy()
    data_copy[date_column] = pd.to_datetime(data_copy[date_column])
    data_copy = data_copy.sort_values(date_column)
    
    # Convert dates to numeric for regression
    data_copy['date_numeric'] = (data_copy[date_column] - data_copy[date_column].min()).dt.days
    
    # Linear regression
    slope, intercept = np.polyfit(data_copy['date_numeric'], data_copy[value_column], 1)
    
    # Calculate R-squared
    y_pred = slope * data_copy['date_numeric'] + intercept
    r_squared = 1 - np.sum((data_copy[value_column] - y_pred) ** 2) / np.sum((data_copy[value_column] - data_copy[value_column].mean()) ** 2)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'trend_per_year': slope * 365.25,  # Convert daily slope to yearly
        'trend_per_decade': slope * 365.25 * 10  # Convert to per decade
    }

def save_model(model: Any, file_path: str) -> None:
    """
    Save a machine learning model.
    
    Args:
        model: Model to save
        file_path: Path to save the model
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {file_path}")

def load_model(file_path: str) -> Any:
    """
    Load a machine learning model.
    
    Args:
        file_path: Path to the model file
    
    Returns:
        Loaded model
    """
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded from {file_path}")
    return model 