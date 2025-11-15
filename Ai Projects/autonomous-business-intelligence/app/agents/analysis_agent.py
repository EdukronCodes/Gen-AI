"""
Analysis Agent - Performs statistical analysis and pattern detection
"""
from typing import Dict, Any
import pandas as pd
import numpy as np
from scipy import stats
from app.agents.base_agent import BaseAgent
from app.models import AgentTask, TaskStatus
from config import settings


class AnalysisAgent(BaseAgent):
    """Agent responsible for data analysis"""
    
    def __init__(self):
        super().__init__(
            agent_id="analysis_agent",
            capabilities=["statistical_analysis", "correlation_analysis", "trend_analysis", "cohort_analysis"]
        )
    
    def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Execute analysis task"""
        try:
            self._update_task_status(task, TaskStatus.IN_PROGRESS)
            
            analysis_type = task.parameters.get("analysis_type", "statistical")
            data = task.parameters.get("data")
            
            if not data:
                raise ValueError("No data provided for analysis")
            
            df = pd.DataFrame(data)
            
            if analysis_type == "statistical":
                result = self._statistical_analysis(df)
            elif analysis_type == "correlation":
                result = self._correlation_analysis(df)
            elif analysis_type == "trend":
                result = self._trend_analysis(df)
            elif analysis_type == "cohort":
                result = self._cohort_analysis(df, task.parameters)
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            self._update_task_status(task, TaskStatus.COMPLETED, result)
            return result
        
        except Exception as e:
            self._update_task_status(task, TaskStatus.FAILED, error=str(e))
            raise
    
    def _statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats_result = {
            "summary_statistics": df.describe().to_dict(),
            "numeric_columns": list(numeric_cols),
            "categorical_columns": list(df.select_dtypes(include=['object']).columns),
            "missing_data": df.isnull().sum().to_dict(),
            "data_shape": {"rows": len(df), "columns": len(df.columns)}
        }
        
        # Additional statistics for numeric columns
        for col in numeric_cols:
            stats_result[f"{col}_stats"] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "skewness": float(df[col].skew()),
                "kurtosis": float(df[col].kurtosis())
            }
        
        return stats_result
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform correlation analysis"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis"}
        
        correlation_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= settings.correlation_threshold:
                    strong_correlations.append({
                        "column1": correlation_matrix.columns[i],
                        "column2": correlation_matrix.columns[j],
                        "correlation": float(corr_value)
                    })
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "strong_correlations": strong_correlations,
            "threshold": settings.correlation_threshold
        }
    
    def _trend_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform trend analysis"""
        # Assume first column is date/time if available
        date_col = None
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if not date_col:
            return {"error": "No date/time column found for trend analysis"}
        
        df_sorted = df.sort_values(date_col)
        numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns
        
        trends = {}
        for col in numeric_cols:
            # Simple linear trend
            x = np.arange(len(df_sorted))
            y = df_sorted[col].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            trends[col] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "significance": "significant" if p_value < settings.statistical_significance else "not_significant"
            }
        
        return {
            "date_column": date_col,
            "trends": trends,
            "time_period": {
                "start": str(df_sorted[date_col].min()),
                "end": str(df_sorted[date_col].max())
            }
        }
    
    def _cohort_analysis(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cohort analysis"""
        cohort_column = parameters.get("cohort_column")
        metric_column = parameters.get("metric_column")
        
        if not cohort_column or not metric_column:
            return {"error": "cohort_column and metric_column required for cohort analysis"}
        
        if cohort_column not in df.columns or metric_column not in df.columns:
            return {"error": "Specified columns not found in data"}
        
        cohort_summary = df.groupby(cohort_column)[metric_column].agg([
            'count', 'mean', 'sum', 'std', 'min', 'max'
        ]).to_dict(orient='index')
        
        return {
            "cohort_column": cohort_column,
            "metric_column": metric_column,
            "cohort_summary": {str(k): v for k, v in cohort_summary.items()},
            "total_cohorts": len(df[cohort_column].unique())
        }

