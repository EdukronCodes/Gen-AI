"""
Data Collection Agent - Collects data from various sources
"""
from typing import Dict, Any, List
import pandas as pd
from sqlalchemy import create_engine, text
from app.agents.base_agent import BaseAgent
from app.models import AgentTask, TaskStatus


class DataCollectionAgent(BaseAgent):
    """Agent responsible for collecting data from various sources"""
    
    def __init__(self):
        super().__init__(
            agent_id="data_collection_agent",
            capabilities=["database_query", "api_fetch", "file_reading", "data_validation"]
        )
    
    def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Execute data collection task"""
        try:
            self._update_task_status(task, TaskStatus.IN_PROGRESS)
            
            source_type = task.parameters.get("source_type", "database")
            source_config = task.parameters.get("source_config", {})
            
            if source_type == "database":
                data = self._collect_from_database(source_config)
            elif source_type == "api":
                data = self._collect_from_api(source_config)
            elif source_type == "file":
                data = self._collect_from_file(source_config)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            # Validate data
            validation_result = self._validate_data(data)
            
            result = {
                "data": data.to_dict(orient="records") if isinstance(data, pd.DataFrame) else data,
                "row_count": len(data) if isinstance(data, pd.DataFrame) else 0,
                "columns": list(data.columns) if isinstance(data, pd.DataFrame) else [],
                "validation": validation_result
            }
            
            self._update_task_status(task, TaskStatus.COMPLETED, result)
            return result
        
        except Exception as e:
            self._update_task_status(task, TaskStatus.FAILED, error=str(e))
            raise
    
    def _collect_from_database(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Collect data from database"""
        connection_string = config.get("connection_string")
        query = config.get("query")
        
        if not connection_string or not query:
            raise ValueError("Missing connection_string or query in config")
        
        engine = create_engine(connection_string)
        df = pd.read_sql(text(query), engine)
        return df
    
    def _collect_from_api(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Collect data from API"""
        import httpx
        
        url = config.get("url")
        method = config.get("method", "GET")
        headers = config.get("headers", {})
        params = config.get("params", {})
        
        if not url:
            raise ValueError("Missing url in config")
        
        with httpx.Client() as client:
            response = client.request(method, url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame if it's a list
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to find data array
                for key in ["data", "results", "items"]:
                    if key in data and isinstance(data[key], list):
                        return pd.DataFrame(data[key])
                return pd.DataFrame([data])
            else:
                raise ValueError("Unsupported API response format")
    
    def _collect_from_file(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Collect data from file"""
        file_path = config.get("file_path")
        file_type = config.get("file_type", "csv")
        
        if not file_path:
            raise ValueError("Missing file_path in config")
        
        if file_type == "csv":
            return pd.read_csv(file_path)
        elif file_type == "excel":
            return pd.read_excel(file_path)
        elif file_type == "json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate collected data"""
        validation = {
            "is_valid": True,
            "row_count": len(data),
            "column_count": len(data.columns),
            "missing_values": data.isnull().sum().to_dict(),
            "duplicate_rows": data.duplicated().sum(),
            "data_types": data.dtypes.astype(str).to_dict()
        }
        
        # Check for critical issues
        if len(data) == 0:
            validation["is_valid"] = False
            validation["errors"] = ["No data collected"]
        
        return validation

