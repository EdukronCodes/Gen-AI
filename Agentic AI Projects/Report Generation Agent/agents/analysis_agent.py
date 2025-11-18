"""
Analysis Agent - Specialized in analyzing query results and generating insights
"""
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
import json

class AnalysisAgent(BaseAgent):
    """Agent responsible for analyzing query results and generating insights"""
    
    def __init__(self, gemini_client=None):
        super().__init__("AnalysisAgent", "Data Analysis Specialist", gemini_client)
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze query results and generate insights
        """
        self.log("Analyzing query results")
        
        if not context or "query_results" not in context:
            return {
                "agent": self.name,
                "analysis": "No query results to analyze",
                "status": "error"
            }
        
        query_results = context["query_results"]
        original_question = context.get("original_question", task)
        
        # Generate analysis using Gemini
        analysis = await self._analyze_results(original_question, query_results)
        
        # Generate summary
        summary = await self._generate_summary(original_question, query_results, analysis)
        
        return {
            "agent": self.name,
            "analysis": analysis,
            "summary": summary,
            "row_count": len(query_results),
            "status": "success"
        }
    
    async def _analyze_results(self, question: str, results: List[Dict[str, Any]]) -> str:
        """Analyze the query results and provide insights"""
        
        # Limit results for analysis to avoid token limits
        results_sample = results[:50] if len(results) > 50 else results
        
        system_prompt = """You are a data analyst expert. Analyze the query results and provide meaningful insights.
        Focus on patterns, trends, and key findings. Be concise but comprehensive."""
        
        user_prompt = f"""Question asked: {question}

Query Results (showing {len(results_sample)} of {len(results)} rows):
{json.dumps(results_sample, indent=2, default=str)}

Provide a detailed analysis of these results. Include:
1. Key findings
2. Patterns or trends observed
3. Notable statistics
4. Business insights"""
        
        analysis = await self._call_gemini(user_prompt, system_prompt, temperature=0.7)
        return analysis
    
    async def _generate_summary(self, question: str, results: List[Dict[str, Any]], analysis: str) -> str:
        """Generate a concise summary"""
        
        system_prompt = """You are a business intelligence expert. Create a concise executive summary."""
        
        user_prompt = f"""Question: {question}

Analysis:
{analysis}

Number of results: {len(results)}

Create a concise executive summary (2-3 paragraphs) that answers the question and highlights key insights."""
        
        summary = await self._call_gemini(user_prompt, system_prompt, temperature=0.5)
        return summary

