"""
Orchestrator Agent - Coordinates all agents in the framework
"""
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from agents.query_agent import QueryAgent
from agents.analysis_agent import AnalysisAgent
from agents.pdf_agent import PDFAgent
import asyncio

class OrchestratorAgent(BaseAgent):
    """Orchestrator that coordinates all agents"""
    
    def __init__(self, db_path: str = "retail_banking.db", gemini_client=None):
        super().__init__("OrchestratorAgent", "Multi-Agent Coordinator", gemini_client)
        
        # Initialize specialized agents
        self.query_agent = QueryAgent(db_path, gemini_client)
        self.analysis_agent = AnalysisAgent(gemini_client)
        self.pdf_agent = PDFAgent(gemini_client=gemini_client)
        
        self.agents = [
            self.query_agent,
            self.analysis_agent,
            self.pdf_agent
        ]
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Orchestrate the multi-agent workflow
        """
        self.log(f"Orchestrating task: {task}")
        
        context = context or {}
        context["original_question"] = task
        
        workflow_results = {
            "original_question": task,
            "workflow_steps": []
        }
        
        try:
            # Step 1: Query Agent - Get data from database
            self.log("Step 1: Querying database...")
            query_result = await self.query_agent.execute(task, context)
            workflow_results["workflow_steps"].append({
                "step": 1,
                "agent": "QueryAgent",
                "status": query_result.get("status"),
                "row_count": query_result.get("row_count", 0)
            })
            
            if query_result.get("status") != "success":
                return {
                    "status": "error",
                    "error": "Query agent failed",
                    "workflow_results": workflow_results
                }
            
            # Update context with query results
            context["query_results"] = query_result.get("results", [])
            context["sql_query"] = query_result.get("sql_query", "")
            
            # Step 2: Analysis Agent - Analyze results
            self.log("Step 2: Analyzing results...")
            analysis_result = await self.analysis_agent.execute(task, context)
            workflow_results["workflow_steps"].append({
                "step": 2,
                "agent": "AnalysisAgent",
                "status": analysis_result.get("status")
            })
            
            if analysis_result.get("status") != "success":
                return {
                    "status": "error",
                    "error": "Analysis agent failed",
                    "workflow_results": workflow_results
                }
            
            # Update context with analysis
            context["analysis"] = analysis_result.get("analysis", "")
            context["summary"] = analysis_result.get("summary", "")
            
            # Step 3: PDF Agent - Generate PDF
            self.log("Step 3: Generating PDF...")
            pdf_result = await self.pdf_agent.execute(task, context)
            workflow_results["workflow_steps"].append({
                "step": 3,
                "agent": "PDFAgent",
                "status": pdf_result.get("status"),
                "pdf_path": pdf_result.get("pdf_path")
            })
            
            if pdf_result.get("status") != "success":
                return {
                    "status": "error",
                    "error": "PDF generation failed",
                    "workflow_results": workflow_results
                }
            
            # Final result
            workflow_results["status"] = "success"
            workflow_results["pdf_path"] = pdf_result.get("pdf_path")
            workflow_results["summary"] = context.get("summary", "")
            workflow_results["analysis"] = context.get("analysis", "")
            workflow_results["query_results_count"] = len(context.get("query_results", []))
            
            self.log("Workflow completed successfully")
            return workflow_results
            
        except Exception as e:
            self.log(f"Workflow error: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "workflow_results": workflow_results
            }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            "orchestrator": self.name,
            "agents": [agent.name for agent in self.agents],
            "total_agents": len(self.agents)
        }

