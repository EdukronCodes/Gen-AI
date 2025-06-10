import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        Loaded DataFrame
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(
    data: pd.DataFrame,
    date_col: str = 'date',
    target_col: str = 'sales',
    feature_cols: Optional[List[str]] = None,
    handle_missing: str = 'interpolate'
) -> pd.DataFrame:
    """
    Preprocess the data for modeling.

    Args:
        data: Input DataFrame
        date_col: Name of the date column
        target_col: Name of the target column
        feature_cols: List of feature columns
        handle_missing: Strategy for handling missing values

    Returns:
        Preprocessed DataFrame
    """
    try:
        # Convert date column to datetime
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Sort by date
        data = data.sort_values(date_col)
        
        # Handle missing values
        if handle_missing == 'interpolate':
            data = data.interpolate(method='time')
        elif handle_missing == 'ffill':
            data = data.fillna(method='ffill')
        elif handle_missing == 'bfill':
            data = data.fillna(method='bfill')
        else:
            data = data.dropna()
        
        # Add time-based features
        data['year'] = data[date_col].dt.year
        data['month'] = data[date_col].dt.month
        data['day'] = data[date_col].dt.day
        data['dayofweek'] = data[date_col].dt.dayofweek
        data['quarter'] = data[date_col].dt.quarter
        
        # Add lag features
        for lag in [1, 7, 14, 30]:
            data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
        
        # Add rolling mean features
        for window in [7, 14, 30]:
            data[f'{target_col}_rolling_mean_{window}'] = (
                data[target_col].rolling(window=window).mean()
            )
        
        # Drop rows with NaN values from lag features
        data = data.dropna()
        
        return data
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def split_data(
    data: pd.DataFrame,
    date_col: str = 'date',
    target_col: str = 'sales',
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.

    Args:
        data: Input DataFrame
        date_col: Name of the date column
        target_col: Name of the target column
        feature_cols: List of feature columns
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation

    Returns:
        Tuple of (train, validation, test) DataFrames
    """
    try:
        # Sort by date
        data = data.sort_values(date_col)
        
        # Calculate split indices
        n = len(data)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        # Split data
        train_data = data.iloc[:val_idx]
        val_data = data.iloc[val_idx:test_idx]
        test_data = data.iloc[test_idx:]
        
        return train_data, val_data, test_data
        
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        raise

def scale_features(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'sales'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler.

    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        feature_cols: List of feature columns
        target_col: Name of the target column

    Returns:
        Tuple of (scaled_train, scaled_val, scaled_test, scaler)
    """
    try:
        # Initialize scaler
        scaler = StandardScaler()
        
        # Fit scaler on training data
        scaler.fit(train_data[feature_cols])
        
        # Transform data
        train_scaled = train_data.copy()
        val_scaled = val_data.copy()
        test_scaled = test_data.copy()
        
        train_scaled[feature_cols] = scaler.transform(train_data[feature_cols])
        val_scaled[feature_cols] = scaler.transform(val_data[feature_cols])
        test_scaled[feature_cols] = scaler.transform(test_data[feature_cols])
        
        return train_scaled, val_scaled, test_scaled, scaler
        
    except Exception as e:
        logger.error(f"Error scaling features: {str(e)}")
        raise

def create_sequences(
    data: pd.DataFrame,
    sequence_length: int,
    target_col: str = 'sales',
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series data.

    Args:
        data: Input DataFrame
        sequence_length: Length of sequences
        target_col: Name of the target column
        feature_cols: List of feature columns

    Returns:
        Tuple of (X, y) arrays
    """
    try:
        if feature_cols is None:
            feature_cols = [target_col]
        
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[feature_cols].iloc[i:(i + sequence_length)].values)
            y.append(data[target_col].iloc[i + sequence_length])
        
        return np.array(X), np.array(y)
        
    except Exception as e:
        logger.error(f"Error creating sequences: {str(e)}")
        raise

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate performance metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    try:
        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise 