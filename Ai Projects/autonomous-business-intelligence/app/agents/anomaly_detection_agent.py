"""
Anomaly Detection Agent - Detects anomalies and outliers in data
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from app.agents.base_agent import BaseAgent
from app.models import AgentTask, TaskStatus
from config import settings


class AnomalyDetectionAgent(BaseAgent):
    """Agent responsible for detecting anomalies"""
    
    def __init__(self):
        super().__init__(
            agent_id="anomaly_detection_agent",
            capabilities=["statistical_anomaly_detection", "isolation_forest", "pattern_deviation"]
        )
    
    def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Execute anomaly detection task"""
        try:
            self._update_task_status(task, TaskStatus.IN_PROGRESS)
            
            data = task.parameters.get("data")
            method = task.parameters.get("method", "isolation_forest")
            
            if not data:
                raise ValueError("No data provided for anomaly detection")
            
            df = pd.DataFrame(data)
            
            if method == "isolation_forest":
                result = self._isolation_forest_detection(df)
            elif method == "statistical":
                result = self._statistical_anomaly_detection(df)
            elif method == "z_score":
                result = self._z_score_detection(df)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            self._update_task_status(task, TaskStatus.COMPLETED, result)
            return result
        
        except Exception as e:
            self._update_task_status(task, TaskStatus.FAILED, error=str(e))
            raise
    
    def _isolation_forest_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return {"error": "No numeric columns found for anomaly detection"}
        
        # Fit Isolation Forest
        contamination = 1 - settings.anomaly_sensitivity
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso_forest.fit_predict(numeric_df)
        
        # Mark anomalies
        df['is_anomaly'] = anomalies == -1
        anomaly_indices = df[df['is_anomaly']].index.tolist()
        
        return {
            "method": "isolation_forest",
            "anomaly_count": int(df['is_anomaly'].sum()),
            "anomaly_percentage": float(df['is_anomaly'].mean() * 100),
            "anomaly_indices": anomaly_indices,
            "anomaly_data": df[df['is_anomaly']].to_dict(orient='records'),
            "sensitivity": settings.anomaly_sensitivity
        }
    
    def _statistical_anomaly_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using statistical methods"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return {"error": "No numeric columns found for anomaly detection"}
        
        anomalies = []
        for col in numeric_df.columns:
            mean = numeric_df[col].mean()
            std = numeric_df[col].std()
            threshold = settings.alert_threshold * std
            
            col_anomalies = df[
                (numeric_df[col] < mean - threshold) | 
                (numeric_df[col] > mean + threshold)
            ].index.tolist()
            
            anomalies.extend(col_anomalies)
        
        unique_anomalies = list(set(anomalies))
        df['is_anomaly'] = df.index.isin(unique_anomalies)
        
        return {
            "method": "statistical",
            "anomaly_count": len(unique_anomalies),
            "anomaly_percentage": float(len(unique_anomalies) / len(df) * 100),
            "anomaly_indices": unique_anomalies,
            "anomaly_data": df[df['is_anomaly']].to_dict(orient='records'),
            "threshold": settings.alert_threshold
        }
    
    def _z_score_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using Z-score"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return {"error": "No numeric columns found for anomaly detection"}
        
        from scipy import stats
        
        anomalies = []
        for col in numeric_df.columns:
            z_scores = np.abs(stats.zscore(numeric_df[col]))
            col_anomalies = df[z_scores > settings.alert_threshold].index.tolist()
            anomalies.extend(col_anomalies)
        
        unique_anomalies = list(set(anomalies))
        df['is_anomaly'] = df.index.isin(unique_anomalies)
        
        return {
            "method": "z_score",
            "anomaly_count": len(unique_anomalies),
            "anomaly_percentage": float(len(unique_anomalies) / len(df) * 100),
            "anomaly_indices": unique_anomalies,
            "anomaly_data": df[df['is_anomaly']].to_dict(orient='records'),
            "threshold": settings.alert_threshold
        }

