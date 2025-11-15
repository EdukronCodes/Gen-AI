"""
Predictive Agent - Performs forecasting and predictions
"""
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent
from app.models import AgentTask, TaskStatus
from config import settings


class PredictiveAgent(BaseAgent):
    """Agent responsible for predictive analytics"""
    
    def __init__(self):
        super().__init__(
            agent_id="predictive_agent",
            capabilities=["time_series_forecasting", "demand_forecasting", "trend_prediction"]
        )
    
    def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Execute prediction task"""
        try:
            self._update_task_status(task, TaskStatus.IN_PROGRESS)
            
            prediction_type = task.parameters.get("prediction_type", "time_series")
            data = task.parameters.get("data")
            
            if not data:
                raise ValueError("No data provided for prediction")
            
            df = pd.DataFrame(data)
            
            if prediction_type == "time_series":
                result = self._time_series_forecast(df, task.parameters)
            elif prediction_type == "linear_trend":
                result = self._linear_trend_forecast(df, task.parameters)
            else:
                raise ValueError(f"Unsupported prediction type: {prediction_type}")
            
            self._update_task_status(task, TaskStatus.COMPLETED, result)
            return result
        
        except Exception as e:
            self._update_task_status(task, TaskStatus.FAILED, error=str(e))
            raise
    
    def _time_series_forecast(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform time series forecasting"""
        date_column = parameters.get("date_column")
        value_column = parameters.get("value_column")
        horizon = parameters.get("horizon", settings.forecast_horizon)
        
        if not date_column or not value_column:
            # Try to infer columns
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                    date_column = col
                    break
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                value_column = numeric_cols[0]
        
        if not date_column or not value_column:
            return {"error": "Could not identify date and value columns"}
        
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
        
        # Simple moving average forecast
        window = min(7, len(df) // 4)  # Use 7-day window or 25% of data
        df['ma'] = df[value_column].rolling(window=window).mean()
        
        # Extend forecast
        last_date = df[date_column].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        # Simple forecast: use last moving average value
        last_ma = df['ma'].iloc[-1]
        forecast_values = [last_ma] * horizon
        
        # Calculate confidence intervals (simplified)
        std_dev = df[value_column].std()
        upper_bound = [v + 1.96 * std_dev for v in forecast_values]
        lower_bound = [v - 1.96 * std_dev for v in forecast_values]
        
        return {
            "method": "time_series_forecast",
            "forecast_horizon": horizon,
            "forecast": [
                {
                    "date": str(date),
                    "value": float(value),
                    "upper_bound": float(upper),
                    "lower_bound": float(lower)
                }
                for date, value, upper, lower in zip(
                    forecast_dates, forecast_values, upper_bound, lower_bound
                )
            ],
            "historical_data": df[[date_column, value_column]].to_dict(orient='records'),
            "confidence_interval": settings.confidence_interval
        }
    
    def _linear_trend_forecast(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform linear trend forecasting"""
        from scipy import stats
        
        value_column = parameters.get("value_column")
        horizon = parameters.get("horizon", settings.forecast_horizon)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not value_column and len(numeric_cols) > 0:
            value_column = numeric_cols[0]
        
        if not value_column:
            return {"error": "Could not identify value column"}
        
        # Fit linear trend
        x = np.arange(len(df))
        y = df[value_column].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Forecast
        future_x = np.arange(len(df), len(df) + horizon)
        forecast_values = slope * future_x + intercept
        
        # Confidence intervals
        std_dev = std_err
        upper_bound = forecast_values + 1.96 * std_dev
        lower_bound = forecast_values - 1.96 * std_dev
        
        return {
            "method": "linear_trend",
            "forecast_horizon": horizon,
            "trend": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value)
            },
            "forecast": [
                {
                    "period": int(i),
                    "value": float(value),
                    "upper_bound": float(upper),
                    "lower_bound": float(lower)
                }
                for i, (value, upper, lower) in enumerate(
                    zip(forecast_values, upper_bound, lower_bound)
                )
            ],
            "confidence_interval": settings.confidence_interval
        }

