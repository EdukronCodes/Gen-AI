import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.lstm_model import LSTMForecaster
from models.prophet_model import DemandForecaster
from utils.data_processing import (
    calculate_metrics,
    load_data,
    preprocess_data,
    scale_features,
    split_data,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Demand Forecasting API",
    description="API for demand forecasting using Prophet and LSTM models",
    version="1.0.0",
)

# Initialize models
prophet_model = None
lstm_model = None

class ForecastRequest(BaseModel):
    """Request model for forecasting."""
    model_type: str  # 'prophet' or 'lstm'
    forecast_horizon: int
    data_path: Optional[str] = None
    features: Optional[List[str]] = None

class ModelMetrics(BaseModel):
    """Response model for model metrics."""
    mae: float
    mse: float
    rmse: float
    mape: float

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global prophet_model, lstm_model
    try:
        # Initialize Prophet model
        prophet_model = DemandForecaster()
        
        # Initialize LSTM model
        lstm_model = LSTMForecaster(
            sequence_length=30,
            n_features=1,
            n_units=50,
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise

@app.post("/train")
async def train_model(
    model_type: str,
    data_path: str,
    target_col: str = "sales",
    feature_cols: Optional[List[str]] = None
) -> Dict:
    """
    Train the specified model.
    
    Args:
        model_type: Type of model to train ('prophet' or 'lstm')
        data_path: Path to training data
        target_col: Name of target column
        feature_cols: List of feature columns
        
    Returns:
        Dictionary with training results
    """
    try:
        # Load and preprocess data
        data = load_data(data_path)
        data = preprocess_data(data, target_col=target_col)
        
        if model_type == "prophet":
            # Train Prophet model
            prophet_model.fit(data)
            metrics = prophet_model.get_metrics()
            
            return {
                "status": "success",
                "model": "prophet",
                "metrics": metrics
            }
            
        elif model_type == "lstm":
            # Prepare data for LSTM
            train_data, val_data, test_data = split_data(data)
            
            # Scale features
            if feature_cols is None:
                feature_cols = [target_col]
            train_scaled, val_scaled, test_scaled, scaler = scale_features(
                train_data, val_data, test_data, feature_cols
            )
            
            # Create sequences
            X_train, y_train = lstm_model.prepare_data(train_scaled)
            X_val, y_val = lstm_model.prepare_data(val_scaled)
            
            # Train LSTM model
            lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val))
            metrics = lstm_model.get_metrics()
            
            return {
                "status": "success",
                "model": "lstm",
                "metrics": metrics
            }
            
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid model type. Use 'prophet' or 'lstm'"
            )
            
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error training model: {str(e)}"
        )

@app.post("/forecast")
async def generate_forecast(request: ForecastRequest) -> Dict:
    """
    Generate forecasts using the specified model.
    
    Args:
        request: ForecastRequest object containing model type and parameters
        
    Returns:
        Dictionary with forecast results
    """
    try:
        if request.model_type == "prophet":
            if prophet_model is None:
                raise HTTPException(
                    status_code=400,
                    detail="Prophet model not trained"
                )
            
            # Generate forecast
            forecast = prophet_model.predict(request.forecast_horizon)
            
            return {
                "status": "success",
                "model": "prophet",
                "forecast": forecast.to_dict(orient="records")
            }
            
        elif request.model_type == "lstm":
            if lstm_model is None:
                raise HTTPException(
                    status_code=400,
                    detail="LSTM model not trained"
                )
            
            if request.data_path is None:
                raise HTTPException(
                    status_code=400,
                    detail="Data path required for LSTM forecasting"
                )
            
            # Load and preprocess data
            data = load_data(request.data_path)
            data = preprocess_data(data)
            
            # Prepare data for prediction
            if request.features is None:
                request.features = ["sales"]
            
            # Generate forecast
            forecast = lstm_model.predict(data[request.features])
            
            return {
                "status": "success",
                "model": "lstm",
                "forecast": forecast.tolist()
            }
            
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid model type. Use 'prophet' or 'lstm'"
            )
            
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating forecast: {str(e)}"
        )

@app.get("/metrics/{model_type}")
async def get_metrics(model_type: str) -> ModelMetrics:
    """
    Get performance metrics for the specified model.
    
    Args:
        model_type: Type of model ('prophet' or 'lstm')
        
    Returns:
        ModelMetrics object with performance metrics
    """
    try:
        if model_type == "prophet":
            if prophet_model is None:
                raise HTTPException(
                    status_code=400,
                    detail="Prophet model not trained"
                )
            metrics = prophet_model.get_metrics()
            
        elif model_type == "lstm":
            if lstm_model is None:
                raise HTTPException(
                    status_code=400,
                    detail="LSTM model not trained"
                )
            metrics = lstm_model.get_metrics()
            
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid model type. Use 'prophet' or 'lstm'"
            )
        
        return ModelMetrics(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting metrics: {str(e)}"
        )

@app.post("/save/{model_type}")
async def save_model(model_type: str, path: str) -> Dict:
    """
    Save the specified model to disk.
    
    Args:
        model_type: Type of model to save ('prophet' or 'lstm')
        path: Path to save the model
        
    Returns:
        Dictionary with save status
    """
    try:
        if model_type == "prophet":
            if prophet_model is None:
                raise HTTPException(
                    status_code=400,
                    detail="Prophet model not trained"
                )
            prophet_model.save_model(path)
            
        elif model_type == "lstm":
            if lstm_model is None:
                raise HTTPException(
                    status_code=400,
                    detail="LSTM model not trained"
                )
            lstm_model.save_model(path)
            
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid model type. Use 'prophet' or 'lstm'"
            )
        
        return {
            "status": "success",
            "message": f"{model_type} model saved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error saving model: {str(e)}"
        )

@app.post("/load/{model_type}")
async def load_model(model_type: str, path: str) -> Dict:
    """
    Load the specified model from disk.
    
    Args:
        model_type: Type of model to load ('prophet' or 'lstm')
        path: Path to load the model from
        
    Returns:
        Dictionary with load status
    """
    try:
        global prophet_model, lstm_model
        
        if model_type == "prophet":
            prophet_model = DemandForecaster.load_model(path)
            
        elif model_type == "lstm":
            lstm_model = LSTMForecaster.load_model(path)
            
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid model type. Use 'prophet' or 'lstm'"
            )
        
        return {
            "status": "success",
            "message": f"{model_type} model loaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model: {str(e)}"
        ) 