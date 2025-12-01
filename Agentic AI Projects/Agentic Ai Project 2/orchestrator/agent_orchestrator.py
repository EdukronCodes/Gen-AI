"""
Multi-Agent Orchestrator - Routes queries to appropriate agents
"""
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

from agents import (
    FlightSearchAgent,
    FlightBookingAgent,
    CustomerServiceAgent,
    BaggageAgent,
    CheckInAgent,
    FlightStatusAgent,
    RewardsAgent
)

load_dotenv()


class AgentOrchestrator:
    """Orchestrates multiple agents to handle airline queries"""
    
    def __init__(self, db: Session):
        self.db = db
        self.llm = OpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize all agents
        self.agents = {
            "flight_search": FlightSearchAgent(db),
            "flight_booking": FlightBookingAgent(db),
            "customer_service": CustomerServiceAgent(db),
            "baggage": BaggageAgent(db),
            "check_in": CheckInAgent(db),
            "flight_status": FlightStatusAgent(db),
            "rewards": RewardsAgent(db),
        }
        
        self.agent_descriptions = {
            "flight_search": "Searches for available flights based on routes, dates, and preferences",
            "flight_booking": "Creates and manages flight reservations and bookings",
            "customer_service": "Handles general inquiries, booking modifications, and support",
            "baggage": "Provides information about baggage policies, allowances, and tracking",
            "check_in": "Processes flight check-ins and generates boarding passes",
            "flight_status": "Provides real-time flight status, delays, and gate information",
            "rewards": "Manages loyalty programs, points, and membership benefits",
        }
    
    def route_query(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Route query to appropriate agent using LLM"""
        context = context or {}
        
        # Create routing prompt
        routing_prompt = f"""
        You are an intelligent router for an airline multi-agent system. 
        Based on the user's query, determine which agent should handle it.
        
        Available agents:
        {chr(10).join([f"- {name}: {desc}" for name, desc in self.agent_descriptions.items()])}
        
        User Query: {user_input}
        
        Respond with ONLY the agent name (e.g., "flight_search", "customer_service", etc.)
        """
        
        response = self.llm.complete(routing_prompt)
        agent_name = str(response).strip().lower()
        
        # Fallback routing based on keywords
        if agent_name not in self.agents:
            agent_name = self._keyword_routing(user_input)
        
        return agent_name
    
    def _keyword_routing(self, user_input: str) -> str:
        """Fallback keyword-based routing"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["search", "find", "available", "flights", "routes"]):
            return "flight_search"
        elif any(word in user_lower for word in ["book", "reserve", "reservation", "ticket"]):
            return "flight_booking"
        elif any(word in user_lower for word in ["check-in", "checkin", "boarding pass"]):
            return "check_in"
        elif any(word in user_lower for word in ["status", "delay", "gate", "departure time"]):
            return "flight_status"
        elif any(word in user_lower for word in ["baggage", "luggage", "bag", "weight", "tracking"]):
            return "baggage"
        elif any(word in user_lower for word in ["points", "loyalty", "rewards", "membership", "tier"]):
            return "rewards"
        else:
            return "customer_service"
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user query through appropriate agent"""
        context = context or {}
        
        # Route to appropriate agent
        agent_name = self.route_query(user_input, context)
        agent = self.agents.get(agent_name, self.agents["customer_service"])
        
        # Process with selected agent
        result = agent.process(user_input, context)
        
        return {
            "orchestrator": "AgentOrchestrator",
            "routed_to": agent_name,
            "agent_name": agent.name,
            **result
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about all available agents"""
        return {
            "total_agents": len(self.agents),
            "agents": [
                {
                    "name": agent.name,
                    "description": self.agent_descriptions.get(key, ""),
                    "key": key
                }
                for key, agent in self.agents.items()
            ]
        }


