# Module 12: End-to-End Agentic AI System

**Course:** Generative AI & Agentic AI  
**Module Duration:** 1 week  
**Class:** 20

---

## Class 20: Agentic AI Systems – Advanced

### Topics Covered

- Agent types: Reactive, Planning, Multi-Agent Systems
- Memory management: Long-term memory, episodic recall
- Tool use and reasoning chains
- Building a simple multi-agent system using LangGraph or LangChain Agents

### Learning Objectives

By the end of this class, students will be able to:
- Understand different types of AI agents
- Implement memory systems for agents
- Enable tool use in agents
- Build multi-agent systems
- Deploy agentic AI applications

### Core Concepts

#### Agent Types - Complete Analysis

Agents are autonomous systems that can perceive their environment and take actions to achieve goals. This section provides a comprehensive analysis of different agent types, their architectures, and use cases.

**Agent Architecture Overview:**

```
┌─────────────────────────────────────────────────────────────┐
│              AGENT ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────┘

Environment/User Input
    │
    ▼
┌──────────────────┐
│ Perception       │
│ • Observe state  │
│ • Process input  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Reasoning        │
│ • Analyze        │
│ • Plan           │
│ • Decide         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Action           │
│ • Execute tool   │
│ • Generate output│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Memory           │
│ • Store state    │
│ • Learn          │
└──────────────────┘
```

**1. Reactive Agents - Complete Analysis:**

```
Reactive_Agent_Model:

Definition:
Agent that responds directly to current input without planning or memory

Mathematical Model:
Action(state) = Policy(state)

Where:
- state: Current environment state
- Policy: Direct mapping from state to action
- No planning or memory involved

Architecture:
Input → Perception → Immediate Response → Output

Characteristics:
1. No Memory:
   - Each response independent
   - No context retention
   - Stateless operation

2. Simple Decision-Making:
   - Direct input-output mapping
   - No complex reasoning
   - Fast responses

3. Fast Execution:
   - No planning overhead
   - Immediate responses
   - Low latency

Mathematical Formulation:
For input x at time t:
    a_t = f(x_t)
    
Where:
- x_t: Current input
- f: Agent function (policy)
- a_t: Action at time t

Example:
Chatbot:
    User: "Hello"
    Agent: "Hi there!"  # Direct response
    
    User: "What's the weather?"
    Agent: "I don't have access to weather data"  # No memory of previous context

Use Cases:
1. Simple Chatbots:
   - Customer support
   - FAQ systems
   - Rule-based responses

2. Stateless Services:
   - API endpoints
   - Microservices
   - Simple queries

3. Quick Responses:
   - Real-time systems
   - Low latency requirements
   - Simple tasks

Limitations:
- No context awareness
- Cannot learn from history
- Limited capability
- No complex reasoning
```

**2. Planning Agents - Complete Analysis:**

```
Planning_Agent_Model:

Definition:
Agent that creates plans before executing actions

Mathematical Model:
Plan = Planning_Algorithm(Goal, Current_State, Constraints)

Action = Execute(Plan, Current_State)

Where:
- Planning: Multi-step reasoning
- Plan: Sequence of actions
- Execute: Carry out plan

Architecture:
Goal → Planning → Plan → Execution → Feedback → Replan (if needed)

Planning Process:
1. Goal Analysis:
   Goal = Analyze_Goal(goal_description)
   
2. State Assessment:
   State = Assess_Current_State()
   
3. Plan Generation:
   Plan = Generate_Plan(Goal, State, Constraints)
   
   Where:
   Plan = [action_1, action_2, ..., action_n]
   
4. Plan Execution:
   For each action in Plan:
       result = Execute(action)
       Update_State(result)
       if Plan_Failed:
           Replan()

Mathematical Formulation:
For goal G and current state S:
    Plan = argmin_{P} Cost(P, G, S)
    
    Where:
    - P: Possible plan
    - Cost: Execution cost
    - Minimize total cost

Example:
Task: "Research AI and write a report"

Plan:
1. Research AI topics → Result: Information gathered
2. Analyze information → Result: Key points identified
3. Write report → Result: Report generated
4. Review report → Result: Report finalized

Characteristics:
1. Goal-Oriented:
   - Plans to achieve goals
   - Multi-step reasoning
   - Strategic thinking

2. Forward Planning:
   - Considers future states
   - Plans ahead
   - Optimizes sequences

3. Adaptive:
   - Can replan if needed
   - Handles failures
   - Adjusts to changes

Use Cases:
1. Complex Tasks:
   - Multi-step problem solving
   - Task decomposition
   - Strategic planning

2. Autonomous Systems:
   - Robotics
   - Autonomous agents
   - Goal-oriented AI

3. Research Agents:
   - Information gathering
   - Analysis workflows
   - Report generation
```

**3. Multi-Agent Systems - Complete Analysis:**

```
Multi_Agent_System_Model:

Definition:
System with multiple agents working together

Mathematical Model:
MAS = {Agent_1, Agent_2, ..., Agent_n}

Coordination:
For task T:
    Task_Decomposition(T) = {T_1, T_2, ..., T_n}
    
    For each agent_i:
        result_i = Agent_i(T_i)
    
    Final_Result = Aggregate({result_1, ..., result_n})

Where:
- Task_Decomposition: Split task among agents
- Agent_i: Specialized agent
- Aggregate: Combine results

Architecture:
Task
    │
    ▼
┌──────────────────┐
│ Orchestrator     │
│ • Task split     │
│ • Agent assign   │
└────────┬─────────┘
         │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Agent 1  │  │ Agent 2  │  │ Agent N │
│ (Role 1) │  │ (Role 2) │  │ (Role N)│
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │
     └──────────────┴──────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ Result Synthesis  │
          └──────────────────┘

Coordination Patterns:

1. Sequential:
   Result = Agent_N(...Agent_2(Agent_1(Input)))

2. Parallel:
   Results = [Agent_i(Input_i) for i in 1 to N]
   Final = Aggregate(Results)

3. Hierarchical:
   Manager_Agent → Worker_Agents
   Manager delegates to workers

4. Market-Based:
   Agents bid for tasks
   Task assigned to best agent
```

**Agent Type Comparison:**

```
┌──────────────────┬──────────────┬──────────────┬──────────────┐
│ Feature          │ Reactive    │ Planning     │ Multi-Agent  │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ Memory           │ No          │ Yes          │ Yes          │
│ Planning         │ No          │ Yes          │ Optional     │
│ Complexity       │ Low         │ Medium       │ High         │
│ Latency          │ Low         │ Medium       │ High         │
│ Capability       │ Limited     │ Moderate     │ High         │
│ Use Cases        │ Simple      │ Complex      │ Collaborative│
│ Coordination     │ N/A         │ N/A          │ Required     │
│ Best For         │ Fast resp.  │ Goal-oriented│ Team tasks   │
└──────────────────┴──────────────┴──────────────┴──────────────┘
```

#### Memory Management - Complete Analysis

Memory management is crucial for agents to maintain context, learn from experience, and make informed decisions. This section provides a comprehensive analysis of memory types, architectures, and implementations.

**Memory Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              AGENT MEMORY ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────┘

Agent Experience
    │
    ├──────────────────┬──────────────────┬──────────────────┐
    │                  │                  │                  │
    ▼                  ▼                  ▼                  ▼
┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
│Short-term│      │Long-term │      │Episodic  │      │Semantic  │
│Memory    │      │Memory    │      │Memory    │      │Memory    │
├──────────┤      ├──────────┤      ├──────────┤      ├──────────┤
│Current   │      │Persistent│      │Events    │      │Knowledge │
│Context   │      │Knowledge │      │Experiences│      │Concepts  │
│          │      │          │      │          │      │          │
│Fast      │      │Slow      │      │Indexed   │      │Embedded  │
│Access    │      │Access    │      │by time   │      │Vectors   │
│          │      │          │      │          │      │          │
│Limited   │      │Unlimited │      │Structured│      │Semantic  │
│Capacity  │      │Capacity  │      │Context   │      │Search    │
└──────────┘      └──────────┘      └──────────┘      └──────────┘
```

**1. Short-term Memory - Complete Analysis:**

```
Short_Term_Memory_Model:

Definition:
Memory for current conversation context and recent interactions

Mathematical Model:
STM = {interaction_i}_{i=t-k}^{t}

Where:
- t: Current time
- k: Window size (recent interactions)
- interaction_i: (input_i, output_i, context_i)

Memory Capacity:
|STM| ≤ Capacity

When capacity exceeded:
    STM = Summarize(STM)  # Compress old interactions

Memory Access:
For query at time t:
    Context = STM[t-k:t]  # Recent k interactions
    
    Response = Agent(query, Context)

Characteristics:
1. Limited Capacity:
   - Typically 5-10 recent interactions
   - Sliding window
   - Oldest forgotten first

2. Fast Access:
   - Direct memory access
   - No search needed
   - Low latency

3. Current Context:
   - Conversation flow
   - Recent references
   - Immediate context

Mathematical Formulation:
Memory State:
    M_t = {m_{t-k}, m_{t-k+1}, ..., m_t}
    
Where:
    m_i = (input_i, output_i, metadata_i)

Memory Update:
    M_{t+1} = Update(M_t, new_interaction)
    
If |M_{t+1}| > Capacity:
    M_{t+1} = Compress(M_{t+1})
```

**2. Long-term Memory - Complete Analysis:**

```
Long_Term_Memory_Model:

Definition:
Persistent memory for knowledge, preferences, and historical data

Mathematical Model:
LTM = {knowledge_i, preference_j, history_k}

Where:
- knowledge_i: Factual knowledge
- preference_j: User preferences
- history_k: Historical interactions

Storage:
LTM can be stored in:
1. Vector Database:
   - Embeddings of memories
   - Semantic search
   - Similarity retrieval
   
2. Relational Database:
   - Structured data
   - SQL queries
   - Relationships

3. Knowledge Graph:
   - Entity relationships
   - Graph traversal
   - Complex queries

Memory Retrieval:
For query q:
    Relevant_Memories = Retrieve(LTM, q)
    
Where:
Retrieve can be:
- Semantic search (vector similarity)
- Keyword search
- Graph traversal
- SQL query

Mathematical Formulation:
Memory Storage:
    LTM = LTM ∪ {new_memory}
    
Memory Retrieval:
    Relevant = {m ∈ LTM : Relevance(m, query) > threshold}
    
Where:
    Relevance(m, q) = Similarity(Embed(m), Embed(q))

Memory Update:
    LTM = Update(LTM, new_knowledge, old_knowledge)
```

**3. Episodic Memory - Complete Analysis:**

```
Episodic_Memory_Model:

Definition:
Memory for specific events and experiences with temporal context

Mathematical Model:
Episode = {
    event: str,
    context: dict,
    timestamp: datetime,
    outcome: str,
    metadata: dict
}

Episodic_Memory = {Episode_i}_{i=1}^N

Where:
- Each episode is a specific event
- Temporal ordering preserved
- Contextual information included

Memory Structure:
Episode_i = {
    "event": "user_query",
    "context": {
        "user_id": "user123",
        "session_id": "session456",
        "previous_topics": ["AI", "ML"]
    },
    "timestamp": "2024-01-01T12:00:00",
    "outcome": "successful",
    "metadata": {
        "tools_used": ["search", "calculator"],
        "duration": 2.5,
        "satisfaction": 0.9
    }
}

Memory Retrieval:
For context C:
    Relevant_Episodes = {
        Episode_i : Match(Episode_i.context, C)
    }

Where:
Match can be:
- Temporal proximity
- Context similarity
- Outcome relevance
- User similarity

Mathematical Formulation:
Episode Similarity:
    Similarity(Ep_i, Ep_j) = w_temporal × Temporal_Sim(Ep_i, Ep_j) +
                             w_context × Context_Sim(Ep_i, Ep_j) +
                             w_outcome × Outcome_Sim(Ep_i, Ep_j)

Retrieval:
    Relevant = {Ep ∈ Episodic_Memory : Similarity(Ep, query) > threshold}
```

**Complete Memory Implementation:**

```python
"""
Complete Agent Memory System
"""

from typing import List, Dict, Optional
from datetime import datetime
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import json

class AgentMemory:
    """
    Complete memory management system for agents.
    
    Mathematical Model:
        STM = Recent interactions (limited capacity)
        LTM = Persistent knowledge (vector store)
        Episodic = Event memories (temporal)
    """
    
    def __init__(self, 
                 stm_capacity: int = 10,
                 vector_store_path: str = "./memory_db"):
        """
        Initialize memory system.
        
        Args:
            stm_capacity: Short-term memory capacity
            vector_store_path: Path for long-term memory storage
        """
        # Short-term memory
        self.stm_capacity = stm_capacity
        self.stm = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000  # Approximate capacity
        )
        
        # Long-term memory (vector store)
        self.embeddings = OpenAIEmbeddings()
        self.ltm = Chroma(
            persist_directory=vector_store_path,
            embedding_function=self.embeddings,
            collection_name="long_term_memory"
        )
        
        # Episodic memory
        self.episodic_memory: List[Dict] = []
        
        print("[Memory] Initialized memory system")
        print(f"[Memory] STM capacity: {stm_capacity}")
        print(f"[Memory] LTM storage: {vector_store_path}")
    
    def add_to_stm(self, input_text: str, output_text: str):
        """
        Add interaction to short-term memory.
        
        Mathematical Model:
            STM = STM ∪ {new_interaction}
            If |STM| > capacity: Compress
        """
        self.stm.save_context(
            {"input": input_text},
            {"output": output_text}
        )
        
        # Check capacity and compress if needed
        if len(self.stm.chat_memory.messages) > self.stm_capacity * 2:
            self._compress_stm()
        
        print(f"[STM] Added interaction. Current size: {len(self.stm.chat_memory.messages)}")
    
    def _compress_stm(self):
        """Compress STM by summarizing old interactions."""
        print("[STM] Compressing memory...")
        # Implementation: Summarize old messages
        # For now, keep recent messages
        messages = self.stm.chat_memory.messages
        if len(messages) > self.stm_capacity * 2:
            # Keep most recent
            self.stm.chat_memory.messages = messages[-self.stm_capacity * 2:]
    
    def add_to_ltm(self, knowledge: str, metadata: Dict):
        """
        Add knowledge to long-term memory.
        
        Mathematical Model:
            LTM = LTM ∪ {Embed(knowledge), metadata}
        """
        self.ltm.add_texts(
            texts=[knowledge],
            metadatas=[{
                **metadata,
                "timestamp": datetime.now().isoformat()
            }]
        )
        print(f"[LTM] Added knowledge to long-term memory")
    
    def retrieve_from_ltm(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant knowledge from long-term memory.
        
        Mathematical Model:
            Relevant = Top_k({Similarity(Embed(query), Embed(m)) : m ∈ LTM})
        """
        docs = self.ltm.similarity_search(query, k=k)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": None  # Could add similarity score
            })
        
        print(f"[LTM] Retrieved {len(results)} relevant memories")
        return results
    
    def add_episode(self, event: str, context: Dict, outcome: str):
        """
        Add episode to episodic memory.
        
        Mathematical Model:
            Episode = {event, context, timestamp, outcome}
            Episodic_Memory = Episodic_Memory ∪ {Episode}
        """
        episode = {
            "event": event,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "outcome": outcome
        }
        
        self.episodic_memory.append(episode)
        
        # Limit episodic memory size
        if len(self.episodic_memory) > 1000:
            self.episodic_memory = self.episodic_memory[-1000:]
        
        print(f"[Episodic] Added episode. Total: {len(self.episodic_memory)}")
    
    def retrieve_episodes(self, 
                          context: Optional[Dict] = None,
                          time_range: Optional[tuple] = None) -> List[Dict]:
        """
        Retrieve relevant episodes from episodic memory.
        
        Mathematical Model:
            Relevant = {Ep : Match(Ep, context, time_range)}
        """
        relevant = []
        
        for episode in self.episodic_memory:
            # Context matching
            if context:
                if not self._context_match(episode["context"], context):
                    continue
            
            # Time range filtering
            if time_range:
                ep_time = datetime.fromisoformat(episode["timestamp"])
                if not (time_range[0] <= ep_time <= time_range[1]):
                    continue
            
            relevant.append(episode)
        
        print(f"[Episodic] Retrieved {len(relevant)} relevant episodes")
        return relevant
    
    def _context_match(self, ep_context: Dict, query_context: Dict) -> bool:
        """Check if episode context matches query context."""
        for key, value in query_context.items():
            if key in ep_context and ep_context[key] == value:
                return True
        return False
    
    def get_memory_summary(self) -> Dict:
        """Get summary of memory state."""
        return {
            "stm_size": len(self.stm.chat_memory.messages),
            "ltm_size": self.ltm._collection.count() if hasattr(self.ltm, '_collection') else 0,
            "episodic_size": len(self.episodic_memory)
        }


# Example Usage
if __name__ == "__main__":
    print("=" * 70)
    print("AGENT MEMORY SYSTEM - COMPLETE IMPLEMENTATION")
    print("=" * 70)
    
    memory = AgentMemory(stm_capacity=5)
    
    # Add to STM
    memory.add_to_stm("Hello", "Hi there!")
    memory.add_to_stm("What is AI?", "AI is artificial intelligence...")
    
    # Add to LTM
    memory.add_to_ltm(
        "User prefers detailed explanations",
        {"type": "preference", "user_id": "user123"}
    )
    
    # Add episode
    memory.add_episode(
        event="successful_query",
        context={"query_type": "definition", "user_id": "user123"},
        outcome="user_satisfied"
    )
    
    # Retrieve from LTM
    relevant = memory.retrieve_from_ltm("user preferences", k=3)
    print(f"\nRelevant LTM memories: {len(relevant)}")
    
    # Get summary
    summary = memory.get_memory_summary()
    print(f"\nMemory Summary: {summary}")
    
    print("\n" + "=" * 70)
    print("MEMORY SYSTEM DEMO COMPLETE")
    print("=" * 70)

"""
Expected Output:
======================================================================
AGENT MEMORY SYSTEM - COMPLETE IMPLEMENTATION
======================================================================
[Memory] Initialized memory system
[Memory] STM capacity: 5
[Memory] LTM storage: ./memory_db
[STM] Added interaction. Current size: 2
[STM] Added interaction. Current size: 4
[LTM] Added knowledge to long-term memory
[Episodic] Added episode. Total: 1
[LTM] Retrieved 1 relevant memories

Relevant LTM memories: 1

Memory Summary: {'stm_size': 4, 'ltm_size': 1, 'episodic_size': 1}

======================================================================
MEMORY SYSTEM DEMO COMPLETE
======================================================================
"""
```

#### Tool Use - Complete Analysis

Tool use enables agents to interact with external systems, extending their capabilities beyond language generation. This section provides a comprehensive analysis of tool architecture, reasoning, and implementation.

**Tool Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              AGENT TOOL ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────┘

Agent Query/Goal
    │
    ▼
┌──────────────────┐
│ Tool Selection   │
│ • Analyze query  │
│ • Match tools    │
│ • Choose tool    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Tool Execution    │
│ • Prepare input   │
│ • Call tool       │
│ • Get result      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Result Processing │
│ • Validate        │
│ • Format          │
│ • Integrate       │
└────────┬─────────┘
         │
         ▼
Agent Response
```

**Tool Use Mathematical Model:**

```
Tool_Use_Model:

Tool Definition:
Tool = {
    name: str,
    function: Callable,
    description: str,
    input_schema: Dict,
    output_schema: Dict
}

Tool Selection:
For query q:
    Tool_Score(tool, q) = Relevance(tool.description, q)
    
    Selected_Tool = argmax_{tool ∈ Tools} Tool_Score(tool, q)

Tool Execution:
    result = Selected_Tool.function(input)
    
    Where:
    - input: Extracted from query
    - result: Tool output

Tool Reasoning Chain:
    Query → Tool_Selection → Tool_Execution → Result_Integration → Response

Mathematical Formulation:
Agent_Tool_Use(query):
    1. Tool_Scores = {Score(tool, query) for tool in Tools}
    2. Best_Tool = argmax(Tool_Scores)
    3. Tool_Input = Extract_Input(query, Best_Tool)
    4. Tool_Output = Best_Tool.execute(Tool_Input)
    5. Response = Synthesize(query, Tool_Output)
```

**Tool Types - Complete Analysis:**

```
Tool_Categories:

1. Search Tools:
   - Web search (DuckDuckGo, Google)
   - Database queries
   - Vector search
   - Knowledge base search
   
   Mathematical Model:
   Search_Tool(query) = Retrieve(Knowledge_Base, query)
   
   Where:
   - query: Search query
   - Knowledge_Base: Searchable data source
   - Retrieve: Search/retrieval function

2. Computation Tools:
   - Calculator
   - Code execution
   - Mathematical operations
   - Data processing
   
   Mathematical Model:
   Compute_Tool(expression) = Evaluate(expression)
   
   Example:
   Calculator("2 + 2") = 4
   Code_Executor("print('Hello')") = "Hello"

3. API Tools:
   - External service calls
   - REST APIs
   - GraphQL queries
   - Webhooks
   
   Mathematical Model:
   API_Tool(request) = HTTP_Request(api_endpoint, request)
   
   Where:
   - request: API request parameters
   - api_endpoint: External service URL

4. Custom Tools:
   - Domain-specific functions
   - Business logic
   - Internal systems
   - Specialized operations
   
   Mathematical Model:
   Custom_Tool(input) = Domain_Specific_Logic(input)
```

**ReAct (Reasoning + Acting) - Complete Analysis:**

```
ReAct_Architecture:

ReAct combines reasoning and acting in an iterative loop:

Loop:
    Thought → Action → Observation → ... → Final Answer

Mathematical Model:
ReAct_Agent(query):
    context = [query]
    
    while not done:
        # Reasoning
        thought = LLM(Reason(context))
        
        # Acting
        if "Action:" in thought:
            action = Extract_Action(thought)
            tool = Select_Tool(action)
            observation = tool.execute(action)
            context.append(thought, observation)
        else:
            # Final answer
            answer = Extract_Answer(thought)
            return answer

Example:
Query: "What is the weather in Paris and what's 15 * 23?"

Step 1 (Thought):
"I need to get the weather in Paris and calculate 15 * 23.
Action: Weather(Paris)"

Step 2 (Observation):
"Weather in Paris: 20°C, sunny"

Step 3 (Thought):
"Good, I have the weather. Now I need to calculate 15 * 23.
Action: Calculator(15 * 23)"

Step 4 (Observation):
"345"

Step 5 (Thought):
"I have both pieces of information.
Answer: The weather in Paris is 20°C and sunny. 15 * 23 = 345"
```

**Complete Tool Implementation:**

```python
"""
Complete Agent Tool System with ReAct
"""

from typing import List, Dict, Optional, Callable
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

class ToolSystem:
    """
    Complete tool system for agents.
    
    Mathematical Model:
        Tool_Selection(query) = argmax(Relevance(tool, query))
        Result = Selected_Tool.execute(Extract_Input(query))
    """
    
    def __init__(self, llm):
        """
        Initialize tool system.
        
        Args:
            llm: Language model for agent
        """
        self.llm = llm
        self.tools: List[Tool] = []
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        print("[ToolSystem] Initialized")
    
    def register_tool(self, 
                     name: str, 
                     func: Callable, 
                     description: str):
        """
        Register a tool.
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description for agent
        """
        tool = Tool(
            name=name,
            func=func,
            description=description
        )
        self.tools.append(tool)
        print(f"[ToolSystem] Registered tool: {name}")
    
    def create_agent(self, agent_type: str = "zero-shot-react-description"):
        """
        Create agent with tools.
        
        Mathematical Model:
            Agent = Initialize_Agent(tools, llm, agent_type)
        """
        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
        print(f"[ToolSystem] Created agent with {len(self.tools)} tools")
        return agent
    
    def execute_query(self, agent, query: str) -> str:
        """
        Execute query with agent.
        
        Mathematical Model:
            Response = Agent(query, tools, memory)
        """
        print(f"[Agent] Processing query: '{query}'")
        response = agent.run(query)
        print(f"[Agent] Response generated: {len(response)} characters")
        return response


# Example Tool Implementations
class AgentTools:
    """Collection of tool implementations."""
    
    @staticmethod
    def calculator(expression: str) -> str:
        """
        Mathematical calculator tool.
        
        Mathematical Model:
            Result = Evaluate(expression)
        """
        try:
            # Safe evaluation (in production, use more secure method)
            result = eval(expression)
            print(f"[Calculator] {expression} = {result}")
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def web_search(query: str) -> str:
        """
        Web search tool (mock implementation).
        
        Mathematical Model:
            Results = Search(Web, query)
        """
        # In production, use actual search API
        print(f"[WebSearch] Searching: '{query}'")
        return f"Search results for: {query}\n1. Result 1\n2. Result 2\n3. Result 3"
    
    @staticmethod
    def database_query(query: str) -> str:
        """
        Database query tool (mock implementation).
        
        Mathematical Model:
            Results = Query(Database, query)
        """
        print(f"[Database] Querying: '{query}'")
        return f"Database results: Found 5 records matching '{query}'"
    
    @staticmethod
    def weather_api(location: str) -> str:
        """
        Weather API tool (mock implementation).
        
        Mathematical Model:
            Weather = API_Call(Weather_Service, location)
        """
        print(f"[Weather] Getting weather for: '{location}'")
        return f"Weather in {location}: 20°C, sunny, light breeze"


# Example Usage
if __name__ == "__main__":
    print("=" * 70)
    print("AGENT TOOL SYSTEM - COMPLETE IMPLEMENTATION")
    print("=" * 70)
    
    # Initialize
    llm = OpenAI(temperature=0)
    tool_system = ToolSystem(llm)
    
    # Register tools
    tool_system.register_tool(
        name="Calculator",
        func=AgentTools.calculator,
        description="Perform mathematical calculations. Input should be a valid mathematical expression."
    )
    
    tool_system.register_tool(
        name="WebSearch",
        func=AgentTools.web_search,
        description="Search the web for current information. Input should be a search query."
    )
    
    tool_system.register_tool(
        name="Database",
        func=AgentTools.database_query,
        description="Query internal database. Input should be a database query."
    )
    
    tool_system.register_tool(
        name="Weather",
        func=AgentTools.weather_api,
        description="Get weather information for a location. Input should be a location name."
    )
    
    # Create agent
    agent = tool_system.create_agent()
    
    # Execute query
    query = "What's the weather in Paris? Also calculate 15 * 23."
    result = tool_system.execute_query(agent, query)
    
    print(f"\nFinal Answer:\n{result}")
    
    print("\n" + "=" * 70)
    print("TOOL SYSTEM DEMO COMPLETE")
    print("=" * 70)

"""
Expected Output:
======================================================================
AGENT TOOL SYSTEM - COMPLETE IMPLEMENTATION
======================================================================
[ToolSystem] Initialized
[ToolSystem] Registered tool: Calculator
[ToolSystem] Registered tool: WebSearch
[ToolSystem] Registered tool: Database
[ToolSystem] Registered tool: Weather
[ToolSystem] Created agent with 4 tools
[Agent] Processing query: 'What's the weather in Paris? Also calculate 15 * 23.'
[Agent] Thinking about tools to use...
[Weather] Getting weather for: 'Paris'
[Calculator] 15 * 23 = 345
[Agent] Response generated: 234 characters

Final Answer:
The weather in Paris is 20°C, sunny, light breeze. 15 * 23 = 345.

======================================================================
TOOL SYSTEM DEMO COMPLETE
======================================================================
"""
```

#### Building Multi-Agent Systems - Complete Analysis

Multi-agent systems enable collaborative problem-solving by coordinating multiple specialized agents. This section provides a comprehensive analysis of multi-agent architectures, coordination patterns, and complete implementations.

**Multi-Agent System Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              MULTI-AGENT SYSTEM ARCHITECTURE                  │
└─────────────────────────────────────────────────────────────┘

User Query/Task
    │
    ▼
┌──────────────────┐
│ Orchestrator      │
│ • Task Analysis   │
│ • Decomposition   │
│ • Agent Selection │
└────────┬──────────┘
         │
    ├──────────────┬──────────────┬──────────────┐
    │              │              │              │
    ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Research  │  │Analysis  │  │Writing   │  │Review    │
│Agent     │  │Agent     │  │Agent     │  │Agent     │
│          │  │          │  │          │  │          │
│Tool:     │  │Tool:     │  │Tool:     │  │Tool:     │
│Search    │  │Analyze   │  │LLM       │  │Evaluate  │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
    │              │              │              │
    └──────────────┴──────────────┴──────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ Result Synthesis  │
          │ • Aggregate       │
          │ • Combine         │
          │ • Format          │
          └──────────────────┘
                    │
                    ▼
            Final Response
```

**Coordination Patterns - Complete Analysis:**

```
Coordination_Patterns:

1. Sequential Pattern:
   Mathematical Model:
   Result = Agent_N(...Agent_2(Agent_1(Input)))
   
   Architecture:
   Input → Agent1 → Agent2 → ... → AgentN → Output
   
   Use Case:
   - Pipeline workflows
   - Step-by-step processing
   - Dependent tasks

2. Parallel Pattern:
   Mathematical Model:
   Results = [Agent_i(Input_i) for i in 1 to N]
   Final = Aggregate(Results)
   
   Architecture:
   Input
     ├─→ Agent1 ──┐
     ├─→ Agent2 ──┤
     └─→ Agent3 ──┘
              │
              ▼
        Aggregate
              │
              ▼
           Output
   
   Use Case:
   - Independent tasks
   - Speed optimization
   - Parallel processing

3. Hierarchical Pattern:
   Mathematical Model:
   Manager_Agent → {Worker_Agent_1, ..., Worker_Agent_N}
   
   Architecture:
   Task
     │
     ▼
   Manager Agent
     ├─→ Worker1 (Task1)
     ├─→ Worker2 (Task2)
     └─→ Worker3 (Task3)
     │
     ▼
   Aggregate Results
   
   Use Case:
   - Complex workflows
   - Manager-worker pattern
   - Task delegation

4. Market-Based Pattern:
   Mathematical Model:
   For task T:
       Bids = {Agent_i.bid(T) for i in 1 to N}
       Winner = argmax(Bids)  # Best bid
       Result = Winner.execute(T)
   
   Use Case:
   - Resource allocation
   - Dynamic task assignment
   - Competitive selection
```

**Complete Multi-Agent System with LangGraph:**

```python
"""
Complete Multi-Agent System using LangGraph
"""

from typing import Dict, List, TypedDict, Annotated
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langgraph.graph import StateGraph, END
import operator

# Define state structure
class MultiAgentState(TypedDict):
    """State shared across agents."""
    query: str
    research_results: Annotated[List[str], operator.add]
    analysis_results: Annotated[List[str], operator.add]
    writing_results: Annotated[List[str], operator.add]
    final_response: str
    current_step: str


class MultiAgentSystem:
    """
    Complete multi-agent system with LangGraph.
    
    Mathematical Model:
        Task → Decompose → {Agent_1(T_1), ..., Agent_N(T_N)} → Aggregate
    """
    
    def __init__(self):
        """Initialize multi-agent system."""
        self.llm = OpenAI(temperature=0)
        
        # Create specialized agents
        self.research_agent = self._create_research_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.writing_agent = self._create_writing_agent()
        self.review_agent = self._create_review_agent()
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        print("[MultiAgent] System initialized")
        print("[MultiAgent] Agents: Research, Analysis, Writing, Review")
    
    def _create_research_agent(self):
        """Create research agent."""
        tools = [
            Tool(
                name="WebSearch",
                func=self._search,
                description="Search the web for information"
            ),
            Tool(
                name="Database",
                func=self._database_query,
                description="Query internal database"
            )
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            verbose=False
        )
        return agent
    
    def _create_analysis_agent(self):
        """Create analysis agent."""
        tools = [
            Tool(
                name="Analyze",
                func=self._analyze_data,
                description="Analyze data and extract insights"
            )
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            verbose=False
        )
        return agent
    
    def _create_writing_agent(self):
        """Create writing agent."""
        # Writing agent uses LLM directly
        return self.llm
    
    def _create_review_agent(self):
        """Create review agent."""
        # Review agent uses LLM directly
        return self.llm
    
    def _search(self, query: str) -> str:
        """Search tool (mock)."""
        print(f"[Research] Searching: {query}")
        return f"Search results for '{query}': Found relevant information about the topic."
    
    def _database_query(self, query: str) -> str:
        """Database query tool (mock)."""
        print(f"[Research] Database query: {query}")
        return f"Database results: Found 3 records matching '{query}'"
    
    def _analyze_data(self, data: str) -> str:
        """Analysis tool (mock)."""
        print(f"[Analysis] Analyzing data...")
        return f"Analysis: Key insights extracted from the data. Main points identified."
    
    def _build_workflow(self) -> StateGraph:
        """
        Build LangGraph workflow.
        
        Mathematical Model:
            Workflow = Graph(Nodes, Edges)
            Nodes = {Research, Analysis, Writing, Review}
            Edges = Sequential flow
        """
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes (agents)
        workflow.add_node("research", self._research_node)
        workflow.add_node("analysis", self._analysis_node)
        workflow.add_node("writing", self._writing_node)
        workflow.add_node("review", self._review_node)
        
        # Set entry point
        workflow.set_entry_point("research")
        
        # Define edges
        workflow.add_edge("research", "analysis")
        workflow.add_edge("analysis", "writing")
        workflow.add_edge("writing", "review")
        workflow.add_edge("review", END)
        
        # Compile workflow
        app = workflow.compile()
        return app
    
    def _research_node(self, state: MultiAgentState) -> Dict:
        """
        Research agent node.
        
        Mathematical Model:
            Research_Results = Research_Agent(query)
        """
        print(f"[Research Agent] Starting research for: {state['query']}")
        
        research_query = f"Research information about: {state['query']}"
        result = self.research_agent.run(research_query)
        
        return {
            "research_results": [result],
            "current_step": "research_complete"
        }
    
    def _analysis_node(self, state: MultiAgentState) -> Dict:
        """
        Analysis agent node.
        
        Mathematical Model:
            Analysis_Results = Analysis_Agent(Research_Results)
        """
        print(f"[Analysis Agent] Analyzing research results...")
        
        research_context = "\n".join(state.get("research_results", []))
        analysis_query = f"Analyze the following research and extract key insights:\n{research_context}"
        
        result = self.analysis_agent.run(analysis_query)
        
        return {
            "analysis_results": [result],
            "current_step": "analysis_complete"
        }
    
    def _writing_node(self, state: MultiAgentState) -> Dict:
        """
        Writing agent node.
        
        Mathematical Model:
            Writing_Results = Writing_Agent(Analysis_Results)
        """
        print(f"[Writing Agent] Writing response...")
        
        analysis_context = "\n".join(state.get("analysis_results", []))
        writing_prompt = f"""Based on the following analysis, write a comprehensive response to: {state['query']}

Analysis:
{analysis_context}

Write a clear, well-structured response:"""
        
        result = self.writing_agent(writing_prompt)
        
        return {
            "writing_results": [result],
            "current_step": "writing_complete"
        }
    
    def _review_node(self, state: MultiAgentState) -> Dict:
        """
        Review agent node.
        
        Mathematical Model:
            Final_Response = Review_Agent(Writing_Results)
        """
        print(f"[Review Agent] Reviewing and finalizing response...")
        
        writing_content = "\n".join(state.get("writing_results", []))
        review_prompt = f"""Review and improve the following response to ensure it:
1. Answers the original question: {state['query']}
2. Is clear and well-structured
3. Incorporates all relevant information

Response to review:
{writing_content}

Provide the final, improved response:"""
        
        final_response = self.review_agent(review_prompt)
        
        return {
            "final_response": final_response,
            "current_step": "complete"
        }
    
    def execute(self, query: str) -> Dict:
        """
        Execute multi-agent workflow.
        
        Mathematical Model:
            Result = Workflow(query)
        """
        print("=" * 70)
        print(f"EXECUTING MULTI-AGENT WORKFLOW")
        print(f"Query: {query}")
        print("=" * 70)
        
        # Initial state
        initial_state: MultiAgentState = {
            "query": query,
            "research_results": [],
            "analysis_results": [],
            "writing_results": [],
            "final_response": "",
            "current_step": "started"
        }
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        
        print("\n" + "=" * 70)
        print("WORKFLOW COMPLETE")
        print("=" * 70)
        
        return final_state


# Example Usage
if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-AGENT SYSTEM - COMPLETE IMPLEMENTATION")
    print("=" * 70)
    
    # Initialize system
    system = MultiAgentSystem()
    
    # Execute query
    query = "Explain artificial intelligence and its applications"
    result = system.execute(query)
    
    print(f"\nFinal Response:\n{result['final_response']}")
    
    print("\n" + "=" * 70)
    print("MULTI-AGENT DEMO COMPLETE")
    print("=" * 70)

"""
Expected Output:
======================================================================
MULTI-AGENT SYSTEM - COMPLETE IMPLEMENTATION
======================================================================
[MultiAgent] System initialized
[MultiAgent] Agents: Research, Analysis, Writing, Review
======================================================================
EXECUTING MULTI-AGENT WORKFLOW
Query: Explain artificial intelligence and its applications
======================================================================
[Research Agent] Starting research for: Explain artificial intelligence and its applications
[Research] Searching: artificial intelligence applications
[Analysis Agent] Analyzing research results...
[Analysis] Analyzing data...
[Writing Agent] Writing response...
[Review Agent] Reviewing and finalizing response...

======================================================================
WORKFLOW COMPLETE
======================================================================

Final Response:
[Comprehensive response about AI and its applications based on research and analysis]

======================================================================
MULTI-AGENT DEMO COMPLETE
======================================================================
"""
```

#### Agent Capabilities

**Reasoning:**
- Chain-of-thought
- Multi-step thinking
- Problem decomposition

**Planning:**
- Goal setting
- Action sequences
- Resource allocation

**Tool Use:**
- Function calling
- API integration
- External resources

**Memory:**
- Context retention
- Learning from experience
- User preferences

**Communication:**
- Inter-agent messaging
- User interaction
- Status reporting

### Advanced Topics

#### Agent Orchestration

**Patterns:**
- **Sequential:** Agents work in sequence
- **Parallel:** Agents work simultaneously
- **Hierarchical:** Agents organized in hierarchy
- **Market-based:** Agents bid for tasks

#### Agent Evaluation

**Metrics:**
- Task completion rate
- Response quality
- Tool usage efficiency
- User satisfaction

**Evaluation Framework:**
- Define success criteria
- Test scenarios
- Measure performance
- Iterate improvements

### Readings

- Recent papers on agentic AI:
  - "ReAct: Synergizing Reasoning and Acting" (Yao et al., 2022)
  - "AutoGPT: Autonomous Agents" (research)
  - "LangGraph: Multi-Agent Workflows" (documentation)

- LangGraph documentation:
  - [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

- Multi-agent system research:
  - Survey papers on multi-agent systems
  - Recent agentic AI papers

 

### Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- [Agentic AI Research](https://arxiv.org/search/?query=agentic+AI)

### Practical Code Examples

#### Complete Agentic System with LangGraph

```python
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, END

class AgenticSystem:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Define tools
        self.tools = [
            Tool(
                name="Calculator",
                func=self._calculate,
                description="Perform mathematical calculations"
            ),
            Tool(
                name="Search",
                func=self._search,
                description="Search for information"
            )
        ]
        
        # Initialize agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="zero-shot-react-description",
            memory=self.memory,
            verbose=True
        )
    
    def _calculate(self, expression: str) -> str:
        """Simple calculator"""
        try:
            result = eval(expression)
            return str(result)
        except:
            return "Error: Invalid expression"
    
    def _search(self, query: str) -> str:
        """Search function"""
        # Implement search logic
        return f"Search results for: {query}"
    
    def run(self, task: str) -> str:
        """Execute agentic task"""
        return self.agent.run(task)

# Usage
system = AgenticSystem()
result = system.run("Calculate 15 * 23, then search for information about AI")
print(result)
```

**Pro Tip:** Use LangGraph for complex multi-agent workflows. It provides better control and visualization of agent interactions.

**Common Pitfall:** Agents without proper constraints can get stuck in loops or make poor decisions. Always implement timeouts, token limits, and validation.

#### Multi-Agent System

```python
from typing import Dict, List
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool

class MultiAgentSystem:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        
        # Define specialized agents
        self.researcher = self._create_agent("researcher")
        self.writer = self._create_agent("writer")
        self.reviewer = self._create_agent("reviewer")
    
    def _create_agent(self, role: str):
        """Create specialized agent"""
        tools = self._get_tools_for_role(role)
        return initialize_agent(
            tools,
            self.llm,
            agent="zero-shot-react-description",
            verbose=True
        )
    
    def _get_tools_for_role(self, role: str) -> List[Tool]:
        """Get tools for specific role"""
        # Define role-specific tools
        if role == "researcher":
            return [Tool(name="Search", func=self._search, description="Search")]
        elif role == "writer":
            return [Tool(name="Write", func=self._write, description="Write")]
        else:
            return []
    
    def _search(self, query: str) -> str:
        return f"Research: {query}"
    
    def _write(self, content: str) -> str:
        return f"Written: {content}"
    
    def collaborative_task(self, task: str) -> str:
        """Execute collaborative multi-agent task"""
        # Research phase
        research = self.researcher.run(f"Research: {task}")
        
        # Writing phase
        writing = self.writer.run(f"Write about: {research}")
        
        # Review phase
        review = self.reviewer.run(f"Review: {writing}")
        
        return review

# Usage
system = MultiAgentSystem()
result = system.collaborative_task("Create article about AI")
print(result)
```

### Troubleshooting Guide

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Agent loops** | Repeating actions | Add max iterations, implement stopping conditions |
| **Poor decisions** | Wrong tool choices | Improve tool descriptions, add examples, refine prompts |
| **Memory issues** | Context overflow | Implement summarization, limit conversation history |
| **Slow execution** | High latency | Optimize tools, use caching, parallelize where possible |
| **Tool errors** | Tool failures | Add error handling, validate inputs, implement fallbacks |

### Quick Reference Guide

#### Agent Architecture Patterns

| Pattern | Use Case | Example |
|---------|----------|---------|
| Sequential | Step-by-step tasks | Research → Write → Review |
| Parallel | Independent tasks | Multiple searches simultaneously |
| Hierarchical | Complex workflows | Manager → Worker agents |
| Market-based | Resource allocation | Agents bid for tasks |

#### Agent Evaluation Metrics

| Metric | Purpose | Measurement |
|--------|---------|-------------|
| Task completion | Success rate | % of tasks completed |
| Tool efficiency | Resource usage | Tools per task |
| Response quality | Output quality | Human evaluation |
| Latency | Performance | Time to complete |

### Case Studies

#### Case Study: Autonomous Research Agent

**Challenge:** Automate research and report generation.

**Solution:** Multi-agent system with:
- Research agent for information gathering
- Analysis agent for data processing
- Writing agent for report generation

**Results:**
- 80% time reduction
- Consistent quality
- Handles complex queries

**Lessons Learned:**
- Clear agent roles critical
- Proper orchestration essential
- Evaluation framework necessary

### Key Takeaways

1. Agentic AI enables autonomous, goal-oriented behavior
2. Different agent types suit different tasks
3. Memory management is crucial for context-aware agents
4. Tool use extends agent capabilities beyond language
5. Multi-agent systems enable complex collaborative tasks
6. Proper orchestration ensures effective agent coordination
7. Evaluation frameworks help improve agent performance
8. Agentic AI represents the future of autonomous AI systems
9. Constraints and validation prevent agent failures
10. Monitoring and evaluation are essential for agent systems

---

**Previous Module:** [Module 11: Frameworks, Libraries & Platforms Overview](../module_11.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)

