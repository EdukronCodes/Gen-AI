import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

logger = logging.getLogger(__name__)

class DemandForecaster:
    """
    Demand forecasting using Facebook Prophet.
    """
    
    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        seasonality_mode: str = 'multiplicative'
    ):
        """
        Initialize the forecaster.

        Args:
            changepoint_prior_scale: Flexibility of the trend
            seasonality_prior_scale: Strength of the seasonal components
            holidays_prior_scale: Strength of the holiday components
            seasonality_mode: 'additive' or 'multiplicative'
        """
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode
        )
        
        # Add default seasonalities
        self.model.add_seasonality(
            name='weekly',
            period=7,
            fourier_order=3
        )
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        self.model.add_seasonality(
            name='yearly',
            period=365.25,
            fourier_order=10
        )
        
        self.metrics: Dict[str, float] = {}
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        date_col: str = 'date',
        target_col: str = 'sales',
        regressor_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare data for Prophet.

        Args:
            data: Input DataFrame
            date_col: Name of the date column
            target_col: Name of the target column
            regressor_cols: List of regressor column names

        Returns:
            Prepared DataFrame
        """
        # Ensure date column is datetime
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Rename columns for Prophet
        df = data.rename(columns={
            date_col: 'ds',
            target_col: 'y'
        })
        
        # Add regressors if provided
        if regressor_cols:
            for col in regressor_cols:
                if col in data.columns:
                    self.model.add_regressor(col)
        
        return df
    
    def fit(
        self,
        data: pd.DataFrame,
        date_col: str = 'date',
        target_col: str = 'sales',
        regressor_cols: Optional[List[str]] = None
    ) -> None:
        """
        Fit the model to the data.

        Args:
            data: Training data
            date_col: Name of the date column
            target_col: Name of the target column
            regressor_cols: List of regressor column names
        """
        try:
            # Prepare data
            df = self.prepare_data(
                data,
                date_col=date_col,
                target_col=target_col,
                regressor_cols=regressor_cols
            )
            
            # Fit model
            self.model.fit(df)
            
            # Calculate metrics
            self._calculate_metrics(df)
            
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise
    
    def predict(
        self,
        periods: int,
        freq: str = 'D',
        regressor_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate forecasts.

        Args:
            periods: Number of periods to forecast
            freq: Frequency of the forecast ('D' for daily, 'W' for weekly, etc.)
            regressor_data: DataFrame with future regressor values

        Returns:
            DataFrame with forecasts
        """
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=periods,
                freq=freq
            )
            
            # Add regressor data if provided
            if regressor_data is not None:
                future = future.merge(
                    regressor_data,
                    on='ds',
                    how='left'
                )
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def _calculate_metrics(self, data: pd.DataFrame) -> None:
        """
        Calculate model performance metrics.

        Args:
            data: Training data
        """
        try:
            # Perform cross-validation
            df_cv = cross_validation(
                self.model,
                initial='365 days',
                period='30 days',
                horizon='90 days'
            )
            
            # Calculate metrics
            metrics = performance_metrics(df_cv)
            
            self.metrics = {
                'mape': metrics['mape'].mean(),
                'rmse': metrics['rmse'].mean(),
                'mae': metrics['mae'].mean()
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
    
    def plot_components(self) -> None:
        """Plot model components."""
        try:
            self.model.plot_components()
        except Exception as e:
            logger.error(f"Error plotting components: {str(e)}")
            raise
    
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
    def load_model(cls, path: str) -> 'DemandForecaster':
        """
        Load model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded DemandForecaster instance
        """
        try:
            forecaster = cls()
            forecaster.model = Prophet.load(path)
            return forecaster
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 