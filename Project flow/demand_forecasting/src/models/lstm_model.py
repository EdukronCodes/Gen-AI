import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential, load_model

logger = logging.getLogger(__name__)

class LSTMForecaster:
    """
    Demand forecasting using LSTM neural networks.
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        n_features: int = 1,
        n_units: int = 50,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize the LSTM forecaster.

        Args:
            sequence_length: Number of time steps to look back
            n_features: Number of input features
            n_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimization
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        self.metrics: Dict[str, float] = {}
    
    def _build_model(self) -> Sequential:
        """
        Build the LSTM model architecture.

        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(
                units=self.n_units,
                return_sequences=True,
                input_shape=(self.sequence_length, self.n_features)
            ),
            Dropout(self.dropout_rate),
            LSTM(units=self.n_units),
            Dropout(self.dropout_rate),
            Dense(units=1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_col: str = 'sales',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model.

        Args:
            data: Input DataFrame
            target_col: Name of the target column
            feature_cols: List of feature column names

        Returns:
            Tuple of (X, y) arrays
        """
        # Select features
        if feature_cols is None:
            feature_cols = [target_col]
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data[feature_cols])
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length, 0])
        
        return np.array(X), np.array(y)
    
    def fit(
        self,
        data: pd.DataFrame,
        target_col: str = 'sales',
        feature_cols: Optional[List[str]] = None,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ) -> None:
        """
        Fit the model to the data.

        Args:
            data: Training data
            target_col: Name of the target column
            feature_cols: List of feature column names
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping
        """
        try:
            # Prepare data
            X, y = self.prepare_data(
                data,
                target_col=target_col,
                feature_cols=feature_cols
            )
            
            # Add early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            )
            
            # Train model
            history = self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Calculate metrics
            self._calculate_metrics(history)
            
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise
    
    def predict(
        self,
        data: pd.DataFrame,
        target_col: str = 'sales',
        feature_cols: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Generate predictions.

        Args:
            data: Input data
            target_col: Name of the target column
            feature_cols: List of feature column names

        Returns:
            Array of predictions
        """
        try:
            # Prepare data
            if feature_cols is None:
                feature_cols = [target_col]
            
            scaled_data = self.scaler.transform(data[feature_cols])
            
            X = []
            for i in range(len(scaled_data) - self.sequence_length + 1):
                X.append(scaled_data[i:(i + self.sequence_length)])
            
            X = np.array(X)
            
            # Generate predictions
            scaled_predictions = self.model.predict(X)
            
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(
                np.concatenate([
                    scaled_predictions,
                    np.zeros((len(scaled_predictions), len(feature_cols) - 1))
                ], axis=1)
            )[:, 0]
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def _calculate_metrics(self, history: tf.keras.callbacks.History) -> None:
        """
        Calculate model performance metrics.

        Args:
            history: Training history
        """
        try:
            self.metrics = {
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'final_mae': history.history['mae'][-1],
                'final_val_mae': history.history['val_mae'][-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get model performance metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    def save_model(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save the model
        """
        try:
            self.model.save(path)
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, path: str) -> 'LSTMForecaster':
        """
        Load model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded LSTMForecaster instance
        """
        try:
            forecaster = cls()
            forecaster.model = load_model(path)
            return forecaster
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 