"""
Orchestrator Agent - Coordinates and manages other agents
"""
from typing import Dict, Any, List, Optional
from openai import OpenAI
from app.agents.base_agent import BaseAgent
from app.agents.data_collection_agent import DataCollectionAgent
from app.agents.analysis_agent import AnalysisAgent
from app.agents.anomaly_detection_agent import AnomalyDetectionAgent
from app.agents.predictive_agent import PredictiveAgent
from app.models import AgentTask, TaskType, TaskStatus
from config import settings


class OrchestratorAgent(BaseAgent):
    """Master orchestrator that coordinates all agents"""
    
    def __init__(self):
        super().__init__(
            agent_id="orchestrator_agent",
            capabilities=["task_decomposition", "agent_selection", "coordination", "result_synthesis"]
        )
        self.client = OpenAI(api_key=settings.openai_api_key)
        
        # Initialize specialized agents
        self.agents = {
            "data_collection": DataCollectionAgent(),
            "analysis": AnalysisAgent(),
            "anomaly_detection": AnomalyDetectionAgent(),
            "predictive": PredictiveAgent()
        }
    
    def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Execute task by orchestrating appropriate agents"""
        try:
            self._update_task_status(task, TaskStatus.IN_PROGRESS)
            
            # Decompose task if needed
            subtasks = self._decompose_task(task)
            
            results = []
            for subtask in subtasks:
                # Select appropriate agent
                agent = self._select_agent(subtask)
                
                if agent:
                    # Execute subtask
                    result = agent.execute(subtask)
                    results.append(result)
                else:
                    results.append({"error": f"No suitable agent found for {subtask.task_type}"})
            
            # Synthesize results
            final_result = self._synthesize_results(task, results)
            
            self._update_task_status(task, TaskStatus.COMPLETED, final_result)
            return final_result
        
        except Exception as e:
            self._update_task_status(task, TaskStatus.FAILED, error=str(e))
            raise
    
    def _decompose_task(self, task: AgentTask) -> List[AgentTask]:
        """Decompose complex task into subtasks"""
        # For now, return single task
        # In production, use LLM to decompose complex queries
        return [task]
    
    def _select_agent(self, task: AgentTask) -> Optional[BaseAgent]:
        """Select appropriate agent for task"""
        task_type = task.task_type
        
        if task_type == TaskType.DATA_COLLECTION:
            return self.agents["data_collection"]
        elif task_type == TaskType.ANALYSIS:
            return self.agents["analysis"]
        elif task_type == TaskType.ANOMALY_DETECTION:
            return self.agents["anomaly_detection"]
        elif task_type == TaskType.PREDICTION:
            return self.agents["predictive"]
        else:
            return None
    
    def _synthesize_results(self, task: AgentTask, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize results from multiple agents"""
        if len(results) == 1:
            return results[0]
        
        return {
            "synthesized_result": True,
            "subtask_count": len(results),
            "results": results,
            "summary": self._generate_summary(results)
        }
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> str:
        """Generate summary of results using LLM"""
        if not settings.openai_api_key:
            return "Results synthesized from multiple agents"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a business intelligence analyst. Summarize the analysis results."},
                    {"role": "user", "content": f"Summarize these results: {str(results)}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Results synthesized: {len(results)} subtasks completed"
    
    def process_natural_language_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query and create appropriate tasks"""
        if not settings.openai_api_key:
            return {"error": "OpenAI API key required for natural language processing"}
        
        try:
            # Use LLM to understand query and create task
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": """You are a task planner for a business intelligence system. 
                    Analyze the user query and determine:
                    1. Task type (data_collection, analysis, prediction, anomaly_detection)
                    2. Required parameters
                    3. Priority
                    
                    Respond in JSON format with: task_type, parameters, priority"""},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            import json
            task_info = json.loads(response.choices[0].message.content.strip())
            
            task = AgentTask(
                agent_id=self.agent_id,
                task_type=TaskType(task_info.get("task_type", "analysis")),
                parameters=task_info.get("parameters", {}),
                priority=task_info.get("priority", "medium")
            )
            
            return self.execute(task)
        
        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}

