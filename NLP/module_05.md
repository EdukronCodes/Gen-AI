# Module 5: Frameworks for Building GenAI Applications

**Course:** Generative AI & Agentic AI  
**Module Duration:** 1 week  
**Classes:** 8-9

---

## Class 8: LangChain – The Core Framework

### Topics Covered

- Core components: Chains, Agents, Tools, Memory
- Building a simple RAG pipeline
- LangChain Expression Language (LCEL)
- Integration with OpenAI and ChromaDB

### Learning Objectives

By the end of this class, students will be able to:
- Understand LangChain's architecture and core concepts
- Build chains for sequential processing
- Create agents with tool use capabilities
- Implement memory in conversational applications
- Build a complete RAG pipeline using LangChain

### Core Concepts

#### LangChain Overview - Complete Architecture Analysis

LangChain is a comprehensive framework designed to simplify the development of LLM-powered applications. It provides abstractions, modular components, and integrations that enable developers to build complex AI applications rapidly while maintaining flexibility and production readiness.

**LangChain Architecture - Component-Based Design:**

```
┌─────────────────────────────────────────────────────────────┐
│              LANGCHAIN ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────┘

Application Layer
    │
    ├──────────────────┬──────────────────┬──────────────────┐
    │                  │                  │                  │
    ▼                  ▼                  ▼                  ▼
┌──────────┐    ┌──────────┐      ┌──────────┐      ┌──────────┐
│ Chains   │    │ Agents   │      │ Memory   │      │ Prompts  │
│ • Sequential│  │ • ReAct  │      │ • Buffer │      │ • Templates│
│ • Router │  │ • Plan   │      │ • Summary│      │ • Few-shot│
│ • LLMChain│  │ • Custom │      │ • Entity │      │ • Version │
└──────────┘    └──────────┘      └──────────┘      └──────────┘
    │                  │                  │                  │
    └──────────────────┴──────────────────┴──────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Core Abstractions│
              │ • LLMs           │
              │ • Embeddings     │
              │ • Vector Stores  │
              │ • Retrievers     │
              └────────┬─────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│ LLM APIs │    │ Tools    │    │ Databases│
│ • OpenAI │    │ • Search │    │ • Chroma │
│ • Claude│    │ • Calc   │    │ • Pinecone│
│ • Local │    │ • Custom │    │ • FAISS   │
└──────────┘    └──────────┘    └──────────┘
```

**What is LangChain? - Detailed Definition:**

LangChain is a framework that provides a standardized interface for building LLM applications by abstracting common patterns and providing reusable components.

**Core Philosophy:**

```
Abstraction_Layers:

1. LLM Abstraction:
   - Uniform interface across different LLMs
   - Model-agnostic code
   - Easy switching between providers
   
   Example:
   llm = OpenAI()  # or ChatAnthropic() or ChatOllama()
   response = llm.invoke("Hello")

2. Component Composition:
   - Modular, reusable components
   - Chain components together
   - Easy testing and debugging
   
   Example:
   chain = loader | splitter | embeddings | vectorstore | retriever | llm

3. Integration Standardization:
   - Consistent interfaces for tools
   - Unified error handling
   - Standardized data formats
```

**Mathematical Model of LangChain Chains:**

```
Chain_Execution_Model:

For a chain C = [c₁, c₂, ..., cₙ]:

Execution:
output = C(input) = cₙ(cₙ₋₁(...c₂(c₁(input))...))

Where:
- cᵢ: Component i in chain
- Each component transforms input → output
- Output of cᵢ becomes input of cᵢ₊₁

Example:
Chain = [DocumentLoader, TextSplitter, Embeddings, VectorStore, Retriever, LLM]

Execution Flow:
input: "document.pdf"
  → DocumentLoader: PDF → Documents
  → TextSplitter: Documents → Chunks
  → Embeddings: Chunks → Vectors
  → VectorStore: Vectors → Index
  → Retriever: Query → Relevant Chunks
  → LLM: Chunks + Query → Answer

output: "Answer to query"
```

**Key Benefits - Detailed Analysis:**

```
1. Rapid Prototyping:
   Benefit: Pre-built components reduce development time
   Impact: 5-10x faster initial development
   
   Example:
   - Without LangChain: 2-3 days for basic RAG
   - With LangChain: 2-3 hours for basic RAG

2. Modular Architecture:
   Benefit: Components can be swapped independently
   Impact: Easy experimentation and optimization
   
   Example:
   - Switch embedding model: Change one line
   - Switch vector store: Change one line
   - Switch LLM: Change one line

3. Extensive Integrations:
   Benefit: 200+ integrations out of the box
   Impact: No need to build custom connectors
   
   Integrations:
   - LLMs: OpenAI, Anthropic, Google, Cohere, etc.
   - Vector Stores: Chroma, Pinecone, Weaviate, FAISS, etc.
   - Tools: Search, Calculator, Database, APIs, etc.

4. Production Features:
   Benefits:
   - Error handling and retries
   - Streaming support
   - Observability and monitoring
   - Caching and optimization
   
   Impact: Production-ready out of the box

5. Active Community:
   - 70K+ GitHub stars
   - Active development
   - Extensive documentation
   - Community contributions
```

**LangChain Component Architecture:**

```
Component_Hierarchy:

┌─────────────────────────────────────────────────────────────┐
│                    LANGCHAIN COMPONENTS                      │
└─────────────────────────────────────────────────────────────┘

Level 1: Core Abstractions
├── LLMs (Language Models)
│   ├── OpenAI
│   ├── Anthropic
│   ├── Local Models
│   └── Custom LLMs
│
├── Embeddings
│   ├── OpenAI Embeddings
│   ├── Sentence Transformers
│   └── Custom Embeddings
│
└── Vector Stores
    ├── ChromaDB
    ├── Pinecone
    ├── FAISS
    └── Custom Stores

Level 2: Processing Components
├── Document Loaders
│   ├── PDF
│   ├── CSV
│   ├── Web
│   └── Database
│
├── Text Splitters
│   ├── Recursive Character
│   ├── Token-based
│   └── Semantic
│
└── Retrievers
    ├── Vector Store
    ├── BM25
    └── Hybrid

Level 3: Orchestration
├── Chains
│   ├── LLMChain
│   ├── SequentialChain
│   ├── RouterChain
│   └── Custom Chains
│
├── Agents
│   ├── ReAct Agent
│   ├── Plan-and-Execute
│   └── Custom Agents
│
└── Memory
    ├── Conversation Buffer
    ├── Conversation Summary
    └── Entity Memory

Level 4: Application Layer
├── RAG Pipelines
├── Conversational AI
├── Agentic Systems
└── Custom Applications
```

**LangChain Data Flow - Complete Pipeline:**

```
┌─────────────────────────────────────────────────────────────┐
│              LANGCHAIN DATA FLOW                            │
└─────────────────────────────────────────────────────────────┘

User Input: "What is machine learning?"
    │
    ▼
┌──────────────────┐
│ Input Validation │
│ • Sanitization   │
│ • Type checking  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Memory Retrieval │
│ • Conversation   │
│   history        │
│ • Context        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Query Processing │
│ • Rewriting      │
│ • Expansion      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Retrieval        │
│ • Vector Search  │
│ • BM25 Search    │
│ • Hybrid         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Prompt Assembly  │
│ • Template       │
│ • Context        │
│ • History        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ LLM Generation   │
│ • Streaming      │
│ • Token by token │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Output Parsing   │
│ • Validation     │
│ • Structuring    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Memory Update     │
│ • Store history  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Final Response   │
│ • Answer         │
│ • Sources        │
│ • Metadata       │
└──────────────────┘
```

**LangChain Mathematical Model:**

```
LangChain_Execution_Model:

For application A with components [C₁, C₂, ..., Cₙ]:

A(input) = Cₙ ∘ Cₙ₋₁ ∘ ... ∘ C₂ ∘ C₁(input)

Where:
- ∘: Function composition
- Cᵢ: Component function
- Input: User query or data
- Output: Final response

Component Function:
Cᵢ: Inputᵢ → Outputᵢ

Data Transformation:
input → C₁ → intermediate₁ → C₂ → intermediate₂ → ... → Cₙ → output

Type Safety:
For each component:
  Input_Type → Output_Type
  
Example:
  DocumentLoader: FilePath → List[Document]
  TextSplitter: List[Document] → List[Chunk]
  Embeddings: List[Chunk] → List[Vector]
  VectorStore: List[Vector] → Index
  Retriever: Query → List[Chunk]
  LLM: Prompt → Response
```

**LangChain vs. Direct API Usage:**

```
Comparison:

Direct API Usage:
┌─────────────────────────────────────────────────────────────┐
│              DIRECT API APPROACH                             │
└─────────────────────────────────────────────────────────────┘

User Query
    │
    ▼
┌──────────────────┐
│ Manual:          │
│ • Load docs      │
│ • Chunk text     │
│ • Generate       │
│   embeddings     │
│ • Create index   │
│ • Search         │
│ • Build prompt   │
│ • Call LLM API   │
│ • Parse response │
└──────────────────┘

Time: 2-3 days development
Complexity: High
Maintainability: Low

LangChain Approach:
┌─────────────────────────────────────────────────────────────┐
│              LANGCHAIN APPROACH                              │
└─────────────────────────────────────────────────────────────┘

User Query
    │
    ▼
┌──────────────────┐
│ LangChain:       │
│ • Pre-built      │
│   components     │
│ • Chain          │
│   composition    │
│ • Standardized   │
│   interfaces     │
└──────────────────┘

Time: 2-3 hours development
Complexity: Low
Maintainability: High

Benefits Quantification:
- Development Time: 10x faster
- Code Lines: 5x fewer
- Error Rate: 50% reduction
- Maintainability: 3x improvement
```

#### Core Components - Comprehensive Analysis

**1. Chains - Orchestration Architecture**

Chains are the fundamental orchestration mechanism in LangChain, allowing you to compose multiple components into a single executable pipeline. They represent sequences of operations that transform input through multiple stages to produce a final output.

**Chain Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    CHAIN ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────┘

Input: {"query": "What is AI?"}
    │
    ▼
┌──────────────────┐
│ Component 1      │ → Intermediate Result 1
│ (DocumentLoader) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Component 2      │ → Intermediate Result 2
│ (TextSplitter)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Component 3      │ → Intermediate Result 3
│ (Embeddings)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Component N      │ → Final Output
│ (LLM)            │
└──────────────────┘

Output: {"answer": "AI is...", "sources": [...]}
```

**Mathematical Model of Chains:**

```
Chain_Execution:

For chain C with components [c₁, c₂, ..., cₙ]:

C(input) = cₙ ∘ cₙ₋₁ ∘ ... ∘ c₂ ∘ c₁(input)

Where:
- ∘: Function composition
- cᵢ: Component function
- Input: Dictionary or structured input
- Output: Final result

Type Flow:
Input_Type → c₁ → Type₁ → c₂ → Type₂ → ... → cₙ → Output_Type

Example - RAG Chain:
Query: str
  → Retriever: str → List[Document]
  → PromptBuilder: List[Document] + Query → str
  → LLM: str → str
  → Parser: str → Structured Output
```

**Chain Types:**

```
1. LLMChain:
   - Simplest chain type
   - Single LLM call with prompt
   - Formula: LLMChain(prompt, llm) = llm(prompt.format(input))
   
   Example:
   chain = LLMChain(
       prompt=PromptTemplate(...),
       llm=ChatOpenAI()
   )
   result = chain.run("What is AI?")

2. SequentialChain:
   - Multiple chains executed in sequence
   - Output of one chain becomes input of next
   - Formula: SequentialChain([C₁, C₂, ..., Cₙ]) = Cₙ(...C₂(C₁(input)))
   
   Example:
   chain = SequentialChain(
       chains=[chain1, chain2, chain3],
       input_variables=["query"],
       output_variables=["answer"]
   )

3. RouterChain:
   - Routes input to different chains based on condition
   - Decision function: route(input) → chain_index
   - Formula: RouterChain(input) = Selected_Chain(input)
   
   Example:
   router = RouterChain(
       chains=[chain_a, chain_b],
       router_chain=decision_chain
   )
   result = router.route("query")

4. TransformChain:
   - Custom transformation function
   - Format: TransformChain(transform_fn)
   - Formula: TransformChain(input) = transform_fn(input)
   
   Example:
   chain = TransformChain(
       transform=lambda x: x.upper()
   )
```

**Chain Execution Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│              CHAIN EXECUTION FLOW                            │
└─────────────────────────────────────────────────────────────┘

Input Received
    │
    ▼
┌──────────────────┐
│ Input Validation │
│ • Type checking  │
│ • Required vars  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Component 1      │
│ • Execute        │
│ • Error handling │
│ • Retry logic    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Intermediate     │
│ • Store result   │
│ • Validate       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Component 2      │
│ • Execute        │
│ • Use previous   │
│   result         │
└────────┬─────────┘
         │
         ▼
         ...
         │
         ▼
┌──────────────────┐
│ Final Component  │
│ • Generate       │
│   output         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Output Parsing   │
│ • Format        │
│ • Validate       │
└────────┬─────────┘
         │
         ▼
    Final Output
```

**2. Agents - Autonomous Decision Making**

Agents are autonomous systems that can use tools, make decisions, and take actions based on observations. They represent the most advanced form of LLM orchestration, enabling complex multi-step reasoning and task completion.

**Agent Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────┘

User Query: "What is the weather in Paris and convert 100 EUR to USD?"
    │
    ▼
┌──────────────────┐
│ Agent Brain      │
│ (LLM)            │
│ • Reasoning      │
│ • Planning       │
│ • Decision       │
└────────┬─────────┘
         │
         ├──────────────────┬──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Tool 1       │    │ Tool 2       │    │ Tool 3       │
│ Weather API  │    │ Currency API │    │ Search API   │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                    │                    │
       └────────────────────┴────────────────────┘
                           │
                           ▼
              ┌──────────────────────┐
              │ Tool Results         │
              │ • Weather: 20°C      │
              │ • Currency: 108.5 USD│
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Agent Synthesis       │
              │ • Combine results     │
              │ • Generate response   │
              └──────────┬───────────┘
                         │
                         ▼
              Final Answer: "Weather is 20°C, 100 EUR = 108.5 USD"
```

**Agent Decision Loop:**

```
┌─────────────────────────────────────────────────────────────┐
│              AGENT DECISION LOOP                              │
└─────────────────────────────────────────────────────────────┘

Start
    │
    ▼
┌──────────────────┐
│ Observation      │
│ • Current state  │
│ • Query          │
│ • History        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Thought          │
│ • Analyze        │
│ • Plan           │
│ • Decide action  │
└────────┬─────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌──────────────┐    ┌──────────────┐
│ Use Tool?    │    │ Final Answer?│
│ Yes          │    │ Yes          │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   │
┌──────────────┐           │
│ Execute Tool │           │
│ • Get result │           │
└──────┬───────┘           │
       │                   │
       └───────────────────┘
              │
              ▼
    ┌──────────────────┐
    │ Update State     │
    │ • Add observation│
    │ • Update history │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Max Iterations?  │
    │ No → Continue    │
    │ Yes → Stop       │
    └──────────────────┘
```

**Agent Types:**

```
1. ReAct Agent (Reasoning + Acting):
   - Combines reasoning and acting
   - Pattern: Thought → Action → Observation → Thought → ...
   - Formula: ReAct(query) = Reason(Observe(Act(Reason(...))))
   
   Example:
   agent = initialize_agent(
       tools=[weather_tool, calculator_tool],
       llm=ChatOpenAI(),
       agent="react-docstore"
   )

2. Plan-and-Execute Agent:
   - First plans, then executes
   - Two-phase: Planning → Execution
   - Formula: Plan_Execute(query) = Execute(Plan(query))
   
   Example:
   agent = PlanAndExecuteAgent(
       planner_llm=ChatOpenAI(),
       executor_llm=ChatOpenAI(),
       tools=[...]
   )

3. Self-Ask-with-Search Agent:
   - Asks follow-up questions
   - Uses search to find answers
   - Formula: SelfAsk(query) = Search(Ask(query))
   
   Example:
   agent = initialize_agent(
       tools=[search_tool],
       llm=ChatOpenAI(),
       agent="self-ask-with-search"
   )
```

**Agent Mathematical Model:**

```
Agent_Execution_Model:

For agent A with tools T = [t₁, t₂, ..., tₙ] and LLM L:

A(query) = Execute_Plan(L, query, T)

Where:
- L: Language model for reasoning
- query: User query
- T: Set of available tools
- Execute_Plan: Agent execution algorithm

Agent State:
State = {
    "query": str,
    "history": List[Action],
    "observations": List[Observation],
    "iteration": int
}

Action Selection:
action = L(State) → Action

Where Action ∈ {Use_Tool(tool, args), Final_Answer(answer)}

Tool Execution:
observation = tool(action.args)

State Update:
State' = Update(State, action, observation)

Termination:
Terminate if action == Final_Answer or iteration >= max_iterations
```

**3. Tools - Extending Agent Capabilities**

Tools are external functions that agents can use to interact with the world, access information, or perform computations. They enable agents to go beyond text generation and perform actual actions.

**Tool Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    TOOL ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────┘

Agent
    │
    ▼
┌──────────────────┐
│ Tool Selection   │
│ • Which tool?    │
│ • What args?     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Tool Registry    │
│ • Search tool    │
│ • Calculator     │
│ • Database       │
│ • Custom tools   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Tool Execution   │
│ • Validate args  │
│ • Execute        │
│ • Error handling │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Result Return    │
│ • Format output  │
│ • Return to agent│
└──────────────────┘
```

**Tool Types:**

```
1. Search Tools:
   - Web search: Search(query) → Results
   - Example: DuckDuckGoSearchTool
   
2. Calculation Tools:
   - Calculator: Calculate(expression) → Number
   - Example: PythonREPLTool
   
3. Database Tools:
   - Query: QueryDB(query) → Results
   - Example: SQLDatabaseTool
   
4. API Tools:
   - API call: CallAPI(endpoint, params) → Response
   - Example: Custom API tool
   
5. File Tools:
   - Read file: ReadFile(path) → Content
   - Example: FileReadTool
```

**Tool Implementation Pattern:**

```
Tool_Definition:

class Tool:
    name: str
    description: str
    func: Callable
    
    def run(self, input: str) -> str:
        """
        Execute tool with input
        Returns: Tool output as string
        """
        return self.func(input)

Tool_Registration:

tools = [
    Tool(
        name="search",
        description="Search the web for information",
        func=search_web
    ),
    Tool(
        name="calculator",
        description="Perform mathematical calculations",
        func=calculate
    )
]

Agent_Usage:

agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI()
)
```

**4. Memory - Context Management**

Memory enables agents and chains to maintain conversation history and context across multiple interactions. It's essential for building conversational applications that can reference previous exchanges.

**Memory Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────┘

User Input: "What is AI?"
    │
    ▼
┌──────────────────┐
│ Memory Retrieval │
│ • Load history   │
│ • Get context    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ LLM Processing   │
│ • With context   │
│ • Generate       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Memory Storage   │
│ • Save input     │
│ • Save output    │
│ • Update state   │
└──────────────────┘
```

**Memory Types:**

```
1. ConversationBufferMemory:
   - Stores all messages
   - Simple, complete history
   - Memory Size: O(n) where n = number of messages
   
   Example:
   memory = ConversationBufferMemory()
   memory.save_context(
       {"input": "Hi"},
       {"output": "Hello!"}
   )

2. ConversationSummaryMemory:
   - Summarizes old messages
   - Keeps recent messages
   - Memory Size: O(1) constant
   - Formula: Summary = LLM(Old_Messages)
   
   Example:
   memory = ConversationSummaryMemory(llm=ChatOpenAI())
   # Automatically summarizes old messages

3. ConversationBufferWindowMemory:
   - Keeps last k messages
   - Sliding window approach
   - Memory Size: O(k) constant
   
   Example:
   memory = ConversationBufferWindowMemory(k=5)

4. EntityMemory:
   - Extracts and stores entities
   - Entity-based context
   - Memory Size: O(e) where e = number of entities
   
   Example:
   memory = ConversationEntityMemory(llm=ChatOpenAI())
```

**Memory Mathematical Model:**

```
Memory_State_Model:

For memory M with history H = [m₁, m₂, ..., mₙ]:

Memory State:
M = {
    "messages": H,
    "summary": S,
    "entities": E
}

Memory Retrieval:
context = Retrieve(M, query_type)

Where:
- If BufferMemory: context = H
- If SummaryMemory: context = [S, H_recent]
- If WindowMemory: context = H[-k:]
- If EntityMemory: context = Extract_Entities(H)

Memory Update:
M' = Update(M, input, output)

Where:
- BufferMemory: M'.messages = M.messages + [input, output]
- SummaryMemory: M'.summary = Summarize(M.messages[:-k])
- WindowMemory: M'.messages = M.messages[-k+2:] + [input, output]
- EntityMemory: M'.entities = Update_Entities(M.entities, input, output)
```

**5. Prompts - Template Management**

Prompts in LangChain provide a structured way to manage prompt templates, enabling reusability, versioning, and dynamic composition of prompts for different use cases.

**Prompt Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    PROMPT ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────┘

Input Variables: {"topic": "AI", "context": "..."}
    │
    ▼
┌──────────────────┐
│ Prompt Template  │
│ • Template text  │
│ • Variables      │
│ • Formatting     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Variable Binding │
│ • Replace vars   │
│ • Validate       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Final Prompt     │
│ • Formatted text │
│ • Ready for LLM  │
└──────────────────┘
```

**Prompt Types:**

```
1. PromptTemplate:
   - Basic string template
   - Variable substitution
   - Formula: PromptTemplate.format(**vars) → str
   
   Example:
   template = PromptTemplate(
       template="Tell me about {topic}",
       input_variables=["topic"]
   )
   prompt = template.format(topic="AI")

2. ChatPromptTemplate:
   - Multi-message prompts
   - System/user/assistant messages
   - Formula: ChatPromptTemplate.format_messages(**vars) → List[Message]
   
   Example:
   template = ChatPromptTemplate.from_messages([
       ("system", "You are a helpful assistant"),
       ("user", "{question}")
   ])

3. FewShotPromptTemplate:
   - Includes examples
   - Few-shot learning
   - Formula: FewShotPromptTemplate(examples) → Prompt
   
   Example:
   template = FewShotPromptTemplate(
       examples=examples,
       example_prompt=example_prompt,
       prefix="...",
       suffix="...",
       input_variables=["input"]
   )
```

**6. Document Loaders - Data Ingestion**

Document loaders provide a standardized interface for loading documents from various sources, enabling seamless integration with different data formats and storage systems.

**Document Loader Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              DOCUMENT LOADER ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────┘

Data Source
    │
    ├──────────────┬──────────────┬──────────────┬──────────────┐
    │              │              │              │              │
    ▼              ▼              ▼              ▼              ▼
┌─────────┐ ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ PDF       │ │ CSV     │  │ Web     │  │ Database│  │ File    │
│ Loader    │ │ Loader  │  │ Loader  │  │ Loader  │  │ Loader  │
└─────┬─────┘ └─────┬───┘  └─────┬───┘  └─────┬───┘  └─────┬───┘
      │              │             │             │             │
      └──────────────┴─────────────┴─────────────┴─────────────┘
                           │
                           ▼
              ┌──────────────────────┐
              │ Document Object       │
              │ • content: str        │
              │ • metadata: dict      │
              │ • source: str         │
              └──────────────────────┘
```

**Document Loader Types:**

```
1. PDF Loader:
   - PyPDFLoader: Loads PDF files
   - PDFMinerLoader: Advanced PDF parsing
   
   Example:
   loader = PyPDFLoader("document.pdf")
   documents = loader.load()

2. CSV Loader:
   - CSVLoader: Loads CSV files
   - Customizable column mapping
   
   Example:
   loader = CSVLoader("data.csv")
   documents = loader.load()

3. Web Loader:
   - WebBaseLoader: Loads web pages
   - BeautifulSoup integration
   
   Example:
   loader = WebBaseLoader("https://example.com")
   documents = loader.load()

4. Database Loader:
   - SQLDatabaseLoader: Loads from SQL databases
   - Custom query support
   
   Example:
   loader = SQLDatabaseLoader(
       connection_string="...",
       query="SELECT * FROM documents"
   )
```

**7. Text Splitters - Chunking Strategy**

Text splitters divide documents into smaller chunks that fit within token limits while preserving semantic meaning and context. They're crucial for RAG systems and long-context processing.

**Text Splitter Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              TEXT SPLITTER ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────┘

Document: "Long text document..."
    │
    ▼
┌──────────────────┐
│ Text Splitter    │
│ • Chunk size     │
│ • Overlap        │
│ • Separators     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Chunking Logic   │
│ • Split by       │
│   separators     │
│ • Respect size   │
│ • Add overlap    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Chunks           │
│ • Chunk 1        │
│ • Chunk 2        │
│ • Chunk 3        │
│ • ...            │
└──────────────────┘
```

**Text Splitter Types:**

```
1. RecursiveCharacterTextSplitter:
   - Recursively splits by separators
   - Tries separators in order: ["\n\n", "\n", " ", ""]
   - Formula: Split_Recursive(text, size, overlap)
   
   Example:
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=1000,
       chunk_overlap=200
   )
   chunks = splitter.split_documents(documents)

2. TokenTextSplitter:
   - Splits by token count
   - Uses tokenizer to count
   - Formula: Split_Tokens(text, max_tokens, overlap_tokens)
   
   Example:
   splitter = TokenTextSplitter(
       chunk_size=1000,
       chunk_overlap=200
   )

3. CharacterTextSplitter:
   - Simple character-based splitting
   - Fixed chunk size
   - Formula: Split_Chars(text, size, overlap)
   
   Example:
   splitter = CharacterTextSplitter(
       chunk_size=1000,
       chunk_overlap=0
   )
```

**Chunking Mathematical Model:**

```
Chunking_Algorithm:

For document D with text T and chunk_size C, overlap O:

1. Split by separators:
   separators = ["\n\n", "\n", " ", ""]
   segments = Split_By_Separators(T, separators)

2. Create chunks:
   chunks = []
   current_chunk = ""
   
   for segment in segments:
       if len(current_chunk) + len(segment) <= C:
           current_chunk += segment
       else:
           chunks.append(current_chunk)
           # Add overlap
           overlap_text = current_chunk[-O:]
           current_chunk = overlap_text + segment
   
   if current_chunk:
       chunks.append(current_chunk)

3. Result:
   Chunks = [chunk₁, chunk₂, ..., chunkₙ]
   
   Where:
   - len(chunkᵢ) ≤ C for all i
   - Overlap between consecutive chunks = O
```

**8. Vector Stores - Embedding Storage**

Vector stores provide persistent storage and retrieval for document embeddings, enabling efficient similarity search in RAG systems.

**Vector Store Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              VECTOR STORE ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────┘

Documents
    │
    ▼
┌──────────────────┐
│ Embedding Model  │
│ • Generate       │
│   vectors        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Vector Store     │
│ • Index vectors  │
│ • Store metadata │
│ • Enable search  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Query Vector     │
│ • Generate       │
│   embedding      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Similarity Search│
│ • Find similar   │
│   documents      │
└──────────────────┘
```

**Vector Store Operations:**

```
1. Add Documents:
   vectorstore.add_documents(documents)
   - Generate embeddings
   - Store vectors
   - Store metadata
   
2. Similarity Search:
   results = vectorstore.similarity_search(query, k=5)
   - Embed query
   - Find k nearest neighbors
   - Return documents

3. Similarity Search with Score:
   results = vectorstore.similarity_search_with_score(query, k=5)
   - Returns documents + similarity scores

4. As Retriever:
   retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
   - Creates LangChain retriever interface
```

**9. Retrievers - Document Retrieval**

Retrievers provide a unified interface for document retrieval, abstracting different retrieval strategies (vector search, keyword search, hybrid).

**Retriever Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              RETRIEVER ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────┘

Query: "What is machine learning?"
    │
    ▼
┌──────────────────┐
│ Retriever        │
│ • Strategy       │
│ • Parameters     │
└────────┬─────────┘
         │
         ├──────────────────┬──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Vector       │    │ BM25         │    │ Hybrid       │
│ Retriever    │    │ Retriever    │    │ Retriever    │
│ • Embed query│    │ • TF-IDF     │    │ • Combine    │
│ • Similarity │    │ • Ranking    │    │   both       │
│   search     │    │              │    │              │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                    │
       └───────────────────┴────────────────────┘
                           │
                           ▼
              ┌──────────────────────┐
              │ Retrieved Documents   │
              │ • Ranked by relevance│
              │ • With metadata      │
              └──────────────────────┘
```

**Retriever Types:**

```
1. Vector Store Retriever:
   - Uses vector similarity
   - Embedding-based search
   - Formula: Retrieve_Vector(query) = TopK(Similarity(query_embedding, doc_embeddings))
   
   Example:
   retriever = vectorstore.as_retriever(
       search_type="similarity",
       search_kwargs={"k": 5}
   )

2. BM25 Retriever:
   - Keyword-based retrieval
   - TF-IDF scoring
   - Formula: Retrieve_BM25(query) = TopK(BM25_Score(query, docs))
   
   Example:
   retriever = BM25Retriever.from_documents(documents)

3. Hybrid Retriever:
   - Combines vector + BM25
   - Reciprocal Rank Fusion
   - Formula: Retrieve_Hybrid(query) = RRF(Retrieve_Vector(query), Retrieve_BM25(query))
   
   Example:
   retriever = EnsembleRetriever(
       retrievers=[vector_retriever, bm25_retriever]
   )
```

**10. Output Parsers - Structured Outputs**

Output parsers structure and validate LLM outputs, ensuring they conform to expected formats and enabling type-safe integration with downstream components.

**Output Parser Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT PARSER ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────┘

LLM Output: "Raw text response..."
    │
    ▼
┌──────────────────┐
│ Output Parser    │
│ • Parse text     │
│ • Validate       │
│ • Structure      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Structured Output│
│ • Validated      │
│ • Type-safe      │
│ • Ready for use  │
└──────────────────┘
```

**Output Parser Types:**

```
1. StrOutputParser:
   - Simple string output
   - No parsing needed
   
   Example:
   parser = StrOutputParser()
   result = parser.parse(llm_output)

2. PydanticOutputParser:
   - Structured data parsing
   - Pydantic model validation
   - Formula: Parse_Pydantic(text) → Pydantic_Model
   
   Example:
   class Answer(BaseModel):
       answer: str
       confidence: float
   
   parser = PydanticOutputParser(pydantic_object=Answer)
   result = parser.parse(llm_output)

3. JSONOutputParser:
   - JSON parsing
   - JSON schema validation
   - Formula: Parse_JSON(text) → Dict
   
   Example:
   parser = JSONOutputParser()
   result = parser.parse(llm_output)

4. CommaSeparatedListOutputParser:
   - List parsing
   - Comma-separated values
   - Formula: Parse_List(text) → List[str]
   
   Example:
   parser = CommaSeparatedListOutputParser()
   result = parser.parse(llm_output)
```

#### Building a Simple RAG Pipeline - Complete Implementation

A RAG (Retrieval-Augmented Generation) pipeline combines document retrieval with LLM generation to answer questions based on a knowledge base. This section provides a complete, production-ready implementation with detailed explanations.

**RAG Pipeline Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              RAG PIPELINE ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────┘

Phase 1: Indexing (One-time setup)
┌─────────────────────────────────────────────────────────────┐
│ Document.pdf                                                  │
│    │                                                          │
│    ▼                                                          │
│ ┌──────────────────┐                                         │
│ │ Document Loader  │ → Load raw text from PDF                │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Text Splitter    │ → Split into chunks (1000 chars)        │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Embeddings      │ → Generate vector embeddings            │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Vector Store    │ → Store vectors + metadata              │
│ └──────────────────┘                                         │
└─────────────────────────────────────────────────────────────┘

Phase 2: Querying (Runtime)
┌─────────────────────────────────────────────────────────────┐
│ User Query: "What is machine learning?"                      │
│    │                                                          │
│    ▼                                                          │
│ ┌──────────────────┐                                         │
│ │ Query Embedding │ → Generate query vector                  │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Vector Search   │ → Find top-k similar documents          │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Context Assembly │ → Combine retrieved chunks              │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Prompt Template  │ → Build prompt with context            │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ LLM Generation   │ → Generate answer                       │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ Final Answer: "Machine learning is..."                        │
└─────────────────────────────────────────────────────────────┘
```

**Complete RAG Implementation with Detailed Comments:**

```python
"""
Complete RAG Pipeline Implementation with LangChain

This implementation demonstrates:
1. Document loading and preprocessing
2. Chunking strategy
3. Embedding generation
4. Vector store creation
5. Retrieval configuration
6. Prompt engineering
7. Chain composition
8. Query execution
"""

import os
from typing import List, Dict, Any
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

# ============================================================================
# STEP 1: Document Loading
# ============================================================================
# Load documents from various sources (PDF, web, database, etc.)
# Document loaders extract text and metadata from source files

def load_documents(file_path: str) -> List[Document]:
    """
    Load documents from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of Document objects containing text and metadata
        
    Example:
        documents = load_documents("knowledge_base.pdf")
        # Output: [Document(page_content="...", metadata={"source": "...", "page": 1})]
    """
    print(f"[Step 1] Loading documents from: {file_path}")
    
    # Initialize PDF loader
    # PyPDFLoader extracts text from PDF pages
    loader = PyPDFLoader(file_path)
    
    # Load all documents
    # Each page becomes a separate Document object
    documents = loader.load()
    
    print(f"[Step 1] Loaded {len(documents)} documents")
    print(f"[Step 1] Total pages: {sum(doc.metadata.get('page', 0) for doc in documents)}")
    
    return documents

# ============================================================================
# STEP 2: Text Splitting (Chunking)
# ============================================================================
# Split documents into smaller chunks that fit within token limits
# Overlap ensures context is preserved across chunk boundaries

def split_documents(documents: List[Document], 
                   chunk_size: int = 1000, 
                   chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into chunks with specified size and overlap.
    
    Args:
        documents: List of Document objects
        chunk_size: Maximum characters per chunk (default: 1000)
        chunk_overlap: Characters to overlap between chunks (default: 200)
        
    Returns:
        List of Document chunks
        
    Mathematical Model:
        For document D with text T:
        - Split T into segments of size ≤ chunk_size
        - Overlap consecutive chunks by chunk_overlap characters
        - Result: Chunks = [chunk₁, chunk₂, ..., chunkₙ]
        
    Example:
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
        # Output: [Document(chunk 1), Document(chunk 2), ...]
    """
    print(f"[Step 2] Splitting documents into chunks...")
    print(f"[Step 2] Chunk size: {chunk_size} characters")
    print(f"[Step 2] Chunk overlap: {chunk_overlap} characters")
    
    # RecursiveCharacterTextSplitter tries separators in order:
    # ["\n\n", "\n", " ", ""]
    # This preserves semantic boundaries (paragraphs, sentences, words)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # Maximum chunk size
        chunk_overlap=chunk_overlap, # Overlap between chunks
        length_function=len,         # Function to measure text length
        separators=["\n\n", "\n", " ", ""]  # Splitting order
    )
    
    # Split documents into chunks
    # Each chunk maintains metadata from original document
    chunks = text_splitter.split_documents(documents)
    
    print(f"[Step 2] Created {len(chunks)} chunks")
    print(f"[Step 2] Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} characters")
    
    return chunks

# ============================================================================
# STEP 3: Embedding Generation and Vector Store Creation
# ============================================================================
# Generate embeddings for chunks and store in vector database
# Embeddings enable semantic similarity search

def create_vector_store(chunks: List[Document], 
                       persist_directory: str = "./chroma_db") -> Chroma:
    """
    Create vector store from document chunks.
    
    Args:
        chunks: List of Document chunks
        persist_directory: Directory to persist vector store
        
    Returns:
        Chroma vector store instance
        
    Process:
        1. Initialize embedding model (OpenAIEmbeddings)
        2. Generate embeddings for all chunks
        3. Store vectors + metadata in ChromaDB
        4. Create index for fast similarity search
        
    Mathematical Model:
        For each chunk chunkᵢ:
            embeddingᵢ = EmbeddingModel(chunkᵢ.text)
            VectorStore.add(embeddingᵢ, chunkᵢ.metadata)
        
        Index = Build_Index([embedding₁, embedding₂, ..., embeddingₙ])
        
    Example:
        vectorstore = create_vector_store(chunks)
        # Output: ChromaDB instance with indexed vectors
    """
    print(f"[Step 3] Creating vector store...")
    print(f"[Step 3] Embedding model: OpenAI text-embedding-ada-002")
    print(f"[Step 3] Vector dimension: 1536")
    
    # Initialize embedding model
    # OpenAIEmbeddings uses OpenAI's embedding API
    # Each text is converted to a 1536-dimensional vector
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",  # OpenAI embedding model
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create vector store from documents
    # ChromaDB automatically:
    # 1. Generates embeddings for all chunks
    # 2. Creates an index for fast similarity search
    # 3. Stores metadata alongside vectors
    vectorstore = Chroma.from_documents(
        documents=chunks,              # Document chunks
        embedding=embeddings,         # Embedding model
        persist_directory=persist_directory  # Persistence directory
    )
    
    print(f"[Step 3] Vector store created successfully")
    print(f"[Step 3] Persisted to: {persist_directory}")
    print(f"[Step 3] Total vectors: {len(chunks)}")
    
    return vectorstore

# ============================================================================
# STEP 4: Retriever Configuration
# ============================================================================
# Configure retriever to fetch relevant documents for queries

def create_retriever(vectorstore: Chroma, k: int = 4) -> Any:
    """
    Create retriever from vector store.
    
    Args:
        vectorstore: Chroma vector store instance
        k: Number of documents to retrieve (default: 4)
        
    Returns:
        Retriever instance
        
    Mathematical Model:
        Retrieve(query) = TopK(Similarity(query_embedding, doc_embeddings))
        
        Where:
        - Similarity: Cosine similarity
        - TopK: Returns k most similar documents
        - query_embedding: Embedding of user query
        
    Example:
        retriever = create_retriever(vectorstore, k=4)
        # Output: Retriever that returns top 4 similar documents
    """
    print(f"[Step 4] Creating retriever...")
    print(f"[Step 4] Retrieval strategy: Similarity search")
    print(f"[Step 4] Number of documents to retrieve (k): {k}")
    
    # Create retriever from vector store
    # Similarity search finds k most similar documents
    retriever = vectorstore.as_retriever(
        search_type="similarity",      # Use similarity search
        search_kwargs={"k": k}          # Retrieve top k documents
    )
    
    print(f"[Step 4] Retriever created successfully")
    
    return retriever

# ============================================================================
# STEP 5: LLM Configuration
# ============================================================================
# Initialize language model for answer generation

def create_llm(temperature: float = 0, model_name: str = "gpt-3.5-turbo") -> Any:
    """
    Create LLM instance.
    
    Args:
        temperature: Sampling temperature (0 = deterministic, 1 = creative)
        model_name: Model to use
        
    Returns:
        LLM instance
        
    Mathematical Model:
        LLM(prompt) = Generate(prompt, temperature)
        
        Where:
        - Generate: Autoregressive text generation
        - Temperature: Controls randomness in sampling
        
    Example:
        llm = create_llm(temperature=0, model_name="gpt-3.5-turbo")
        # Output: ChatOpenAI instance configured for deterministic responses
    """
    print(f"[Step 5] Creating LLM...")
    print(f"[Step 5] Model: {model_name}")
    print(f"[Step 5] Temperature: {temperature}")
    
    # Initialize chat model
    # ChatOpenAI provides chat-based interface
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,      # 0 = deterministic, higher = creative
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print(f"[Step 5] LLM created successfully")
    
    return llm

# ============================================================================
# STEP 6: Prompt Template Creation
# ============================================================================
# Create prompt template for RAG queries

def create_prompt_template() -> PromptTemplate:
    """
    Create prompt template for RAG.
    
    Returns:
        PromptTemplate instance
        
    Template Structure:
        - System instructions
        - Context from retrieved documents
        - User question
        - Answer format
        
    Example:
        template = create_prompt_template()
        # Output: PromptTemplate with RAG-specific prompt structure
    """
    print(f"[Step 6] Creating prompt template...")
    
    # RAG prompt template
    # Includes context from retrieved documents
    template = """Use the following pieces of context to answer the question.
    
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]  # Variables to fill
    )
    
    print(f"[Step 6] Prompt template created successfully")
    
    return prompt

# ============================================================================
# STEP 7: Chain Composition
# ============================================================================
# Compose retrieval and generation into a single chain

def create_rag_chain(llm: Any, retriever: Any, 
                    return_source_documents: bool = True) -> RetrievalQA:
    """
    Create RAG chain.
    
    Args:
        llm: LLM instance
        retriever: Retriever instance
        return_source_documents: Whether to return source documents
        
    Returns:
        RetrievalQA chain instance
        
    Chain Flow:
        Query → Retriever → Context → Prompt → LLM → Answer
        
    Mathematical Model:
        RAG_Chain(query) = LLM(Prompt(Retrieve(query), query))
        
        Where:
        - Retrieve: Returns top-k documents
        - Prompt: Assembles prompt with context
        - LLM: Generates answer
        
    Example:
        chain = create_rag_chain(llm, retriever)
        # Output: RetrievalQA chain ready for queries
    """
    print(f"[Step 7] Creating RAG chain...")
    print(f"[Step 7] Chain type: Stuff (all context in single prompt)")
    print(f"[Step 7] Return source documents: {return_source_documents}")
    
    # Create RAG chain
    # RetrievalQA handles:
    # 1. Retrieving relevant documents
    # 2. Assembling prompt with context
    # 3. Generating answer with LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,                          # Language model
        chain_type="stuff",               # Stuff all context into prompt
        retriever=retriever,              # Document retriever
        return_source_documents=return_source_documents  # Include sources
    )
    
    print(f"[Step 7] RAG chain created successfully")
    
    return qa_chain

# ============================================================================
# STEP 8: Query Execution
# ============================================================================
# Execute queries against the RAG system

def query_rag(chain: RetrievalQA, query: str) -> Dict[str, Any]:
    """
    Execute query against RAG system.
    
    Args:
        chain: RAG chain instance
        query: User query
        
    Returns:
        Dictionary with answer and source documents
        
    Process:
        1. Embed query
        2. Find similar documents
        3. Assemble prompt
        4. Generate answer
        5. Return result with sources
        
    Example:
        result = query_rag(chain, "What is machine learning?")
        # Output: {
        #   "query": "What is machine learning?",
        #   "result": "Machine learning is...",
        #   "source_documents": [...]
        # }
    """
    print(f"[Step 8] Executing query...")
    print(f"[Step 8] Query: {query}")
    
    # Execute query
    # Chain handles entire pipeline automatically
    response = chain({"query": query})
    
    print(f"[Step 8] Query executed successfully")
    print(f"[Step 8] Answer length: {len(response['result'])} characters")
    print(f"[Step 8] Source documents: {len(response.get('source_documents', []))}")
    
    return response

# ============================================================================
# MAIN FUNCTION: Complete RAG Pipeline
# ============================================================================

def main():
    """
    Complete RAG pipeline demonstration.
    
    This function demonstrates the entire RAG pipeline from document
    loading to query execution.
    """
    print("=" * 70)
    print("RAG PIPELINE - COMPLETE IMPLEMENTATION")
    print("=" * 70)
    
    # Step 1: Load documents
    documents = load_documents("knowledge_base.pdf")
    
    # Step 2: Split documents
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
    
    # Step 3: Create vector store
    vectorstore = create_vector_store(chunks)
    
    # Step 4: Create retriever
    retriever = create_retriever(vectorstore, k=4)
    
    # Step 5: Create LLM
    llm = create_llm(temperature=0, model_name="gpt-3.5-turbo")
    
    # Step 6: Create prompt template
    prompt = create_prompt_template()
    
    # Step 7: Create RAG chain
    chain = create_rag_chain(llm, retriever, return_source_documents=True)
    
    # Step 8: Execute queries
    queries = [
        "What is the main topic of this document?",
        "Summarize the key concepts",
        "What are the practical applications?"
    ]
    
    for query in queries:
        print("\n" + "=" * 70)
        result = query_rag(chain, query)
        
        print(f"\nQuery: {query}")
        print(f"Answer: {result['result']}")
        print(f"Sources: {len(result.get('source_documents', []))} documents")
    
    print("\n" + "=" * 70)
    print("RAG PIPELINE COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

"""
Expected Output:

======================================================================
RAG PIPELINE - COMPLETE IMPLEMENTATION
======================================================================
[Step 1] Loading documents from: knowledge_base.pdf
[Step 1] Loaded 10 documents
[Step 1] Total pages: 10
[Step 2] Splitting documents into chunks...
[Step 2] Chunk size: 1000 characters
[Step 2] Chunk overlap: 200 characters
[Step 2] Created 25 chunks
[Step 2] Average chunk size: 950 characters
[Step 3] Creating vector store...
[Step 3] Embedding model: OpenAI text-embedding-ada-002
[Step 3] Vector dimension: 1536
[Step 3] Vector store created successfully
[Step 3] Persisted to: ./chroma_db
[Step 3] Total vectors: 25
[Step 4] Creating retriever...
[Step 4] Retrieval strategy: Similarity search
[Step 4] Number of documents to retrieve (k): 4
[Step 4] Retriever created successfully
[Step 5] Creating LLM...
[Step 5] Model: gpt-3.5-turbo
[Step 5] Temperature: 0
[Step 5] LLM created successfully
[Step 6] Creating prompt template...
[Step 6] Prompt template created successfully
[Step 7] Creating RAG chain...
[Step 7] Chain type: Stuff (all context in single prompt)
[Step 7] Return source documents: True
[Step 7] RAG chain created successfully
[Step 8] Executing query...
[Step 8] Query: What is the main topic of this document?
[Step 8] Query executed successfully
[Step 8] Answer length: 245 characters
[Step 8] Source documents: 4

Query: What is the main topic of this document?
Answer: The main topic of this document is machine learning and its applications...
Sources: 4 documents
======================================================================
"""
```

#### LangChain Expression Language (LCEL) - Complete Guide

LangChain Expression Language (LCEL) is a declarative way to compose chains using Python's pipe operator (`|`). It provides a clean, type-safe interface for building complex LLM applications with better streaming, debugging, and composability.

**LCEL Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              LCEL ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────┘

Components (Runnable)
    │
    ├──────────────────┬──────────────────┬──────────────────┐
    │                  │                  │                  │
    ▼                  ▼                  ▼                  ▼
┌──────────┐    ┌──────────┐      ┌──────────┐      ┌──────────┐
│ Prompt   │    │ LLM      │      │ Parser   │      │ Custom   │
│ Template │    │ Model    │      │          │      │ Runnable │
└─────┬────┘    └─────┬────┘      └─────┬────┘      └─────┬────┘
      │               │                  │                  │
      └───────────────┴──────────────────┴──────────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │ Pipe Operator (|) │
            │ Chain Composition │
            └─────────┬─────────┘
                      │
                      ▼
            ┌──────────────────┐
            │ LCEL Chain        │
            │ • Type-safe       │
            │ • Streamable      │
            │ • Composable      │
            └──────────────────┘
```

**What is LCEL? - Detailed Explanation:**

LCEL is a domain-specific language for composing LangChain components using Python's pipe operator. It provides a unified interface for all runnable components, enabling seamless composition and type safety.

**Core Concepts:**

```
1. Runnable Interface:
   - All LCEL components implement Runnable
   - Runnable.invoke(input) → output
   - Runnable.stream(input) → AsyncIterator[output]
   
2. Pipe Operator Composition:
   - chain = component1 | component2 | component3
   - Equivalent to: chain(input) = component3(component2(component1(input)))
   
3. Type Safety:
   - Input/Output types are enforced
   - Type checking at composition time
   - Better IDE support
   
4. Streaming:
   - Built-in streaming support
   - Components can stream intermediate results
   - Enables real-time responses
```

**Mathematical Model of LCEL:**

```
LCEL_Chain_Model:

For chain C = c₁ | c₂ | ... | cₙ:

C.invoke(input) = cₙ.invoke(...c₂.invoke(c₁.invoke(input))...)

Where:
- cᵢ: Runnable component
- invoke: Synchronous execution
- stream: Asynchronous streaming execution

Type Flow:
Input_Type → c₁ → Type₁ → c₂ → Type₂ → ... → cₙ → Output_Type

Streaming:
C.stream(input) = AsyncIterator[output]
  = cₙ.stream(...c₂.stream(c₁.stream(input))...)
```

**LCEL Components:**

```
1. Prompt Templates:
   - ChatPromptTemplate: Multi-message prompts
   - PromptTemplate: String templates
   - Runnable interface: .invoke() and .stream()
   
2. LLMs:
   - ChatOpenAI: OpenAI chat models
   - ChatAnthropic: Anthropic models
   - Local models: Ollama, etc.
   
3. Output Parsers:
   - StrOutputParser: String output
   - PydanticOutputParser: Structured output
   - JSONOutputParser: JSON output
   
4. Custom Components:
   - Any function can be wrapped as Runnable
   - Lambda functions
   - Custom classes implementing Runnable
```

**Complete LCEL Example with Detailed Comments:**

```python
"""
LangChain Expression Language (LCEL) - Complete Example

This example demonstrates:
1. LCEL chain composition
2. Type safety
3. Streaming support
4. Error handling
5. Parallel execution
"""

from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

# ============================================================================
# Basic LCEL Chain
# ============================================================================

def basic_lcel_chain():
    """
    Basic LCEL chain demonstration.
    
    Chain Flow:
        Input: {"topic": "AI"}
          → Prompt: Format template with topic
          → LLM: Generate response
          → Parser: Extract string output
          → Output: "AI joke..."
    
    Mathematical Model:
        chain = prompt | model | parser
        output = parser(model(prompt(input)))
    """
    print("=" * 70)
    print("BASIC LCEL CHAIN")
    print("=" * 70)
    
    # Step 1: Create prompt template
    # ChatPromptTemplate supports multi-message prompts
    prompt = ChatPromptTemplate.from_template(
        "Tell me a joke about {topic}"
    )
    
    # Step 2: Initialize LLM
    # ChatOpenAI provides chat-based interface
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7  # Creative responses
    )
    
    # Step 3: Create output parser
    # StrOutputParser extracts string from LLM response
    output_parser = StrOutputParser()
    
    # Step 4: Compose chain using pipe operator
    # This is equivalent to: output_parser(model(prompt(input)))
    chain = prompt | model | output_parser
    
    # Step 5: Invoke chain
    # invoke() executes chain synchronously
    result = chain.invoke({"topic": "AI"})
    
    print(f"Input: {{'topic': 'AI'}}")
    print(f"Output: {result}")
    print()
    
    return chain

# ============================================================================
# LCEL Chain with Streaming
# ============================================================================

def streaming_lcel_chain():
    """
    LCEL chain with streaming support.
    
    Streaming Flow:
        Input → Prompt → LLM (streaming) → Parser (streaming) → Output (streaming)
    
    Benefits:
        - Real-time response generation
        - Better user experience
        - Lower perceived latency
    """
    print("=" * 70)
    print("LCEL CHAIN WITH STREAMING")
    print("=" * 70)
    
    # Create chain
    prompt = ChatPromptTemplate.from_template(
        "Write a short story about {topic} in 3 sentences."
    )
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    parser = StrOutputParser()
    
    chain = prompt | model | parser
    
    # Stream results
    # stream() returns AsyncIterator for real-time streaming
    print("Streaming response:")
    print("-" * 70)
    
    for chunk in chain.stream({"topic": "space exploration"}):
        print(chunk, end="", flush=True)  # Print without newline
    
    print("\n" + "-" * 70)
    print()

# ============================================================================
# LCEL Chain with Custom Functions
# ============================================================================

def custom_function_chain():
    """
    LCEL chain with custom functions.
    
    Custom functions can be wrapped as RunnableLambda to integrate
    with LCEL chains.
    """
    print("=" * 70)
    print("LCEL CHAIN WITH CUSTOM FUNCTIONS")
    print("=" * 70)
    
    # Custom function: Uppercase transformation
    def uppercase(text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()
    
    # Custom function: Word count
    def word_count(text: str) -> Dict[str, Any]:
        """Count words in text."""
        words = text.split()
        return {
            "text": text,
            "word_count": len(words),
            "char_count": len(text)
        }
    
    # Wrap functions as RunnableLambda
    uppercase_runnable = RunnableLambda(uppercase)
    word_count_runnable = RunnableLambda(word_count)
    
    # Create chain with custom functions
    prompt = ChatPromptTemplate.from_template(
        "Describe {topic} in one sentence."
    )
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    parser = StrOutputParser()
    
    # Chain: prompt → model → parser → uppercase → word_count
    chain = prompt | model | parser | uppercase_runnable | word_count_runnable
    
    # Execute chain
    result = chain.invoke({"topic": "artificial intelligence"})
    
    print(f"Input: {{'topic': 'artificial intelligence'}}")
    print(f"Output: {result}")
    print(f"  - Text: {result['text'][:50]}...")
    print(f"  - Word Count: {result['word_count']}")
    print(f"  - Character Count: {result['char_count']}")
    print()

# ============================================================================
# LCEL Chain with Parallel Execution
# ============================================================================

def parallel_execution_chain():
    """
    LCEL chain with parallel execution.
    
    RunnableParallel allows executing multiple chains in parallel,
    then combining their results.
    
    Mathematical Model:
        parallel = RunnableParallel({
            "chain1": c₁,
            "chain2": c₂,
            "chain3": c₃
        })
        
        result = {
            "chain1": c₁(input),
            "chain2": c₂(input),
            "chain3": c₃(input)
        }
        
        All chains execute in parallel for efficiency.
    """
    print("=" * 70)
    print("LCEL CHAIN WITH PARALLEL EXECUTION")
    print("=" * 70)
    
    # Create multiple chains
    prompt1 = ChatPromptTemplate.from_template("Summarize {topic} in one sentence.")
    prompt2 = ChatPromptTemplate.from_template("List 3 key points about {topic}.")
    prompt3 = ChatPromptTemplate.from_template("Explain {topic} to a beginner.")
    
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    parser = StrOutputParser()
    
    chain1 = prompt1 | model | parser
    chain2 = prompt2 | model | parser
    chain3 = prompt3 | model | parser
    
    # Execute chains in parallel
    # RunnableParallel runs all chains concurrently
    parallel_chain = RunnableParallel({
        "summary": chain1,
        "key_points": chain2,
        "explanation": chain3
    })
    
    # Execute parallel chain
    result = parallel_chain.invoke({"topic": "machine learning"})
    
    print(f"Input: {{'topic': 'machine learning'}}")
    print(f"Parallel Execution Results:")
    print(f"  - Summary: {result['summary']}")
    print(f"  - Key Points: {result['key_points']}")
    print(f"  - Explanation: {result['explanation'][:100]}...")
    print()

# ============================================================================
# LCEL Chain with Conditional Logic
# ============================================================================

def conditional_chain():
    """
    LCEL chain with conditional routing.
    
    RunnableBranch allows routing to different chains based on conditions.
    """
    print("=" * 70)
    print("LCEL CHAIN WITH CONDITIONAL LOGIC")
    print("=" * 70)
    
    from langchain.schema.runnable import RunnableBranch
    
    # Define condition function
    def route_condition(input_dict: Dict[str, Any]) -> str:
        """Route based on query length."""
        query = input_dict.get("query", "")
        if len(query) < 10:
            return "short"
        elif len(query) < 50:
            return "medium"
        else:
            return "long"
    
    # Create different chains for different conditions
    short_prompt = ChatPromptTemplate.from_template(
        "Give a brief answer: {query}"
    )
    medium_prompt = ChatPromptTemplate.from_template(
        "Give a detailed answer: {query}"
    )
    long_prompt = ChatPromptTemplate.from_template(
        "Give a comprehensive answer: {query}"
    )
    
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    parser = StrOutputParser()
    
    short_chain = short_prompt | model | parser
    medium_chain = medium_prompt | model | parser
    long_chain = long_prompt | model | parser
    
    # Create conditional chain
    branch = RunnableBranch(
        (lambda x: len(x.get("query", "")) < 10, short_chain),
        (lambda x: len(x.get("query", "")) < 50, medium_chain),
        long_chain
    )
    
    # Test with different query lengths
    queries = [
        {"query": "AI?"},  # Short
        {"query": "What is machine learning and how does it work?"},  # Medium
        {"query": "Explain in detail the complete history and development of artificial intelligence, including key milestones, major breakthroughs, and current state of the field."}  # Long
    ]
    
    for query_dict in queries:
        result = branch.invoke(query_dict)
        print(f"Query ({len(query_dict['query'])} chars): {query_dict['query'][:50]}...")
        print(f"Response: {result[:100]}...")
        print()

# ============================================================================
# MAIN FUNCTION: Complete LCEL Demonstration
# ============================================================================

def main():
    """Complete LCEL demonstration."""
    print("\n" + "=" * 70)
    print("LANGCHAIN EXPRESSION LANGUAGE (LCEL) - COMPLETE GUIDE")
    print("=" * 70 + "\n")
    
    # Basic chain
    basic_lcel_chain()
    
    # Streaming chain
    streaming_lcel_chain()
    
    # Custom function chain
    custom_function_chain()
    
    # Parallel execution
    parallel_execution_chain()
    
    # Conditional logic
    conditional_chain()
    
    print("=" * 70)
    print("LCEL DEMONSTRATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

"""
Expected Output:

======================================================================
LANGCHAIN EXPRESSION LANGUAGE (LCEL) - COMPLETE GUIDE
======================================================================

======================================================================
BASIC LCEL CHAIN
======================================================================
Input: {'topic': 'AI'}
Output: Why did the AI go to therapy? Because it had too many neural networks to process!

======================================================================
LCEL CHAIN WITH STREAMING
======================================================================
Streaming response:
----------------------------------------------------------------------
Once upon a time, in the vast expanse of space, a brave astronaut discovered a new planet...
----------------------------------------------------------------------

======================================================================
LCEL CHAIN WITH CUSTOM FUNCTIONS
======================================================================
Input: {'topic': 'artificial intelligence'}
Output: {'text': 'ARTIFICIAL INTELLIGENCE IS A FIELD OF COMPUTER SCIENCE...', 'word_count': 12, 'char_count': 85}
  - Text: ARTIFICIAL INTELLIGENCE IS A FIELD OF COMPUTER SCIENCE...
  - Word Count: 12
  - Character Count: 85

======================================================================
LCEL CHAIN WITH PARALLEL EXECUTION
======================================================================
Input: {'topic': 'machine learning'}
Parallel Execution Results:
  - Summary: Machine learning is a subset of AI that enables computers to learn from data...
  - Key Points: 1. Data-driven learning, 2. Pattern recognition, 3. Predictive modeling
  - Explanation: Machine learning is like teaching a computer to recognize patterns...

======================================================================
LCEL CHAIN WITH CONDITIONAL LOGIC
======================================================================
Query (3 chars): AI?...
Response: AI stands for Artificial Intelligence...

Query (47 chars): What is machine learning and how does it work?...
Response: Machine learning is a subset of artificial intelligence that enables systems...

Query (165 chars): Explain in detail the complete history and development of artificial intelligence...
Response: The history of artificial intelligence spans several decades, beginning with...
"""
```

**LCEL Benefits - Detailed Analysis:**

```
1. Cleaner Syntax:
   Traditional:
   result = output_parser(model(prompt.format(input)))
   
   LCEL:
   chain = prompt | model | output_parser
   result = chain.invoke(input)
   
   Benefit: More readable and maintainable

2. Better Composability:
   - Components can be easily composed
   - Reusable chain building blocks
   - Easy to modify and extend
   
   Example:
   base_chain = prompt | model | parser
   enhanced_chain = base_chain | custom_function | validator

3. Built-in Streaming:
   - All LCEL chains support streaming out of the box
   - stream() method for async iteration
   - Real-time response generation
   
   Example:
   for chunk in chain.stream(input):
       print(chunk, end="")

4. Parallel Execution:
   - RunnableParallel for concurrent execution
   - Faster processing for independent operations
   - Better resource utilization
   
   Example:
   parallel = RunnableParallel({
       "result1": chain1,
       "result2": chain2
   })

5. Type Safety:
   - Input/Output types are enforced
   - Better IDE support and autocomplete
   - Catch errors at composition time
   
   Example:
   chain: Runnable[Dict[str, str], str]
   # Type checker ensures correct input/output types

6. Easier Debugging:
   - Clear component boundaries
   - Easy to add logging and monitoring
   - Better error messages
   
   Example:
   chain = prompt | model | parser
   # Can add logging between any step
```

#### Integration Examples

**OpenAI Integration:**
```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Completion model
llm = OpenAI(model_name="text-davinci-003")

# Chat model
chat = ChatOpenAI(model_name="gpt-3.5-turbo")
```

**ChromaDB Integration:**
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

---

## Class 9: LlamaIndex & Other Frameworks

### Topics Covered

- Overview of LlamaIndex, differences from LangChain
- Index types: Summary, Vector, List
- Integration with LLM APIs and databases
- When to use LlamaIndex vs LangChain

### Learning Objectives

By the end of this class, students will be able to:
- Understand LlamaIndex architecture and philosophy
- Create different types of indexes
- Compare LlamaIndex with LangChain
- Choose appropriate framework for use case
- Build applications using LlamaIndex

### Core Concepts

#### LlamaIndex Overview - Complete Architecture Analysis

LlamaIndex is a data framework specifically designed for LLM applications, focusing on data ingestion, indexing, and querying. It provides a powerful interface for building RAG systems and knowledge base applications with advanced indexing strategies and query optimization.

**LlamaIndex Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              LLAMAINDEX ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────┘

Data Sources
    │
    ├──────────────┬──────────────┬──────────────┬──────────────┐
    │              │              │              │              │
    ▼              ▼              ▼              ▼              ▼
┌─────────┐ ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ PDF     │ │ CSV     │  │ Web     │  │ Database│  │ Files   │
│ Documents│ │ Files   │  │ Pages   │  │         │  │         │
└─────┬────┘ └─────┬───┘  └─────┬───┘  └─────┬───┘  └─────┬───┘
      │            │             │             │             │
      └────────────┴─────────────┴─────────────┴─────────────┘
                           │
                           ▼
              ┌──────────────────────┐
              │ Document Loaders      │
              │ • SimpleDirectoryReader│
              │ • PDFReader           │
              │ • WebReader           │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Document Processing   │
              │ • Chunking            │
              │ • Metadata Extraction │
              │ • Text Cleaning       │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Index Types            │
              │ • Vector Store Index   │
              │ • Summary Index        │
              │ • List Index           │
              │ • Tree Index           │
              │ • Keyword Table Index  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Query Engine          │
              │ • Query Interface     │
              │ • Retrieval Strategy  │
              │ • Response Synthesis  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ LLM Integration      │
              │ • OpenAI               │
              │ • Anthropic            │
              │ • Local Models         │
              └──────────────────────┘
```

**What is LlamaIndex? - Detailed Definition:**

LlamaIndex is a data framework that provides a structured approach to building LLM applications by focusing on data ingestion, indexing, and querying. It abstracts away the complexities of document processing, embedding generation, and retrieval, allowing developers to focus on application logic.

**Core Philosophy:**

```
1. Data-First Approach:
   - Data is the primary concern
   - Optimized for document-heavy workloads
   - Efficient data ingestion pipelines
   - Flexible data processing

2. Flexible Indexing Strategies:
   - Multiple index types for different use cases
   - Hybrid indexing approaches
   - Customizable indexing algorithms
   - Optimized for specific query patterns

3. Query Optimization:
   - Intelligent query routing
   - Multi-index querying
   - Query transformation and refinement
   - Efficient retrieval strategies

4. Production-Ready Data Pipelines:
   - Scalable data processing
   - Persistent storage
   - Incremental updates
   - Error handling and recovery
```

**LlamaIndex Data Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│              LLAMAINDEX DATA FLOW                            │
└─────────────────────────────────────────────────────────────┘

Phase 1: Data Ingestion
┌─────────────────────────────────────────────────────────────┐
│ Raw Documents                                                 │
│    │                                                          │
│    ▼                                                          │
│ ┌──────────────────┐                                         │
│ │ Document Loader  │ → Load documents from sources            │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Document Nodes   │ → Convert to structured nodes           │
│ │ • Text chunks    │                                         │
│ │ • Metadata      │                                         │
│ │ • Relationships  │                                         │
│ └────────┬─────────┘                                         │
└──────────┼──────────────────────────────────────────────────┘
           │
           ▼
Phase 2: Indexing
┌─────────────────────────────────────────────────────────────┐
│ ┌──────────────────┐                                         │
│ │ Index Builder    │ → Create index structure                │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ├──────────────────┬──────────────────┐            │
│          │                  │                  │            │
│          ▼                  ▼                  ▼            │
│ ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│ │ Vector Store │    │ Summary     │    │ Tree Index   │   │
│ │ Index        │    │ Index       │    │              │   │
│ │ • Embeddings │    │ • Summaries │    │ • Hierarchy  │   │
│ │ • Metadata   │    │ • Keywords  │    │ • Nodes      │   │
│ └──────────────┘    └──────────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
Phase 3: Querying
┌─────────────────────────────────────────────────────────────┐
│ User Query: "What is machine learning?"                      │
│    │                                                          │
│    ▼                                                          │
│ ┌──────────────────┐                                         │
│ │ Query Engine     │ → Process query                         │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Retrieval        │ → Retrieve relevant nodes               │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Response Synthesis│ → Generate answer with LLM             │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ Final Answer: "Machine learning is..."                        │
└─────────────────────────────────────────────────────────────┘
```

**Mathematical Model of LlamaIndex:**

```
LlamaIndex_Model:

For document set D = {d₁, d₂, ..., dₙ}:

1. Document Processing:
   Nodes = Process(D)
   Where each node nᵢ contains:
   - text: Text content
   - metadata: Document metadata
   - relationships: Connections to other nodes

2. Index Construction:
   Index = Build_Index(Nodes, Index_Type)
   
   Where Index_Type ∈ {
       Vector_Store_Index,
       Summary_Index,
       List_Index,
       Tree_Index,
       Keyword_Table_Index
   }

3. Query Processing:
   Query(q) = Synthesize(Retrieve(Index, q))
   
   Where:
   - Retrieve: Find relevant nodes
   - Synthesize: Generate answer using LLM

4. Retrieval Strategy:
   Relevant_Nodes = Retrieve(Index, q, k)
   = TopK(Similarity(q, Index.Nodes))
   
   Where:
   - Similarity: Embedding similarity or keyword matching
   - k: Number of nodes to retrieve
```

**Key Features:**

```
1. Multiple Index Types:
   - Vector Store Index: Embedding-based semantic search
   - Summary Index: Hierarchical document summaries
   - List Index: Sequential document retrieval
   - Tree Index: Hierarchical tree structure
   - Keyword Table Index: Keyword-based lookup

2. Advanced Querying:
   - Multi-index queries
   - Query routing
   - Response synthesis
   - Streaming responses

3. Data Management:
   - Persistent storage
   - Incremental updates
   - Document versioning
   - Metadata filtering

4. LLM Integration:
   - OpenAI integration
   - Anthropic integration
   - Local model support
   - Custom LLM providers
```

#### LlamaIndex vs LangChain

**LlamaIndex:**
- Focus: Data ingestion and indexing
- Strength: RAG, knowledge bases, data queries
- Best for: Document-heavy applications
- Query interface: Rich query capabilities

**LangChain:**
- Focus: Application orchestration
- Strength: Chains, agents, tool use
- Best for: Complex workflows, agentic systems
- Flexibility: More modular components

**When to Use LlamaIndex:**
- Document-heavy applications
- Knowledge base queries
- Need for advanced indexing strategies
- Data-centric workflows

**When to Use LangChain:**
- Complex agent workflows
- Multiple tool integrations
- Chain compositions
- Application orchestration

**Can Use Both:**
- LlamaIndex for data layer
- LangChain for application layer
- Complementary strengths

#### Index Types

**1. Vector Store Index**
- Most common type
- Stores embeddings in vector database
- Semantic search capability
- Good for general RAG

**2. Summary Index**
- Creates summaries of documents
- Hierarchical summaries
- Good for overview queries
- Efficient for large documents

**3. List Index**
- Sequential document list
- Simple retrieval
- Good for small datasets
- Deterministic ordering

**4. Tree Index**
- Hierarchical tree structure
- Top-down querying
- Good for structured documents
- Efficient for specific queries

**5. Keyword Table Index**
- Keyword-based indexing
- Fast keyword lookup
- Good for exact matches
- Complementary to vector index

#### Building with LlamaIndex

**Basic RAG Example:**
```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine()

# Query
response = query_engine.query("What is the main topic?")
print(response)
```

**Advanced Example with Multiple Indexes:**
```python
from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage
)

# Create vector index
vector_index = VectorStoreIndex.from_documents(documents)

# Create summary index
summary_index = SummaryIndex.from_documents(documents)

# Query both
vector_response = vector_index.as_query_engine().query(question)
summary_response = summary_index.as_query_engine().query(question)
```

#### Integration with LLM APIs

**OpenAI:**
```python
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
```

**Anthropic:**
```python
from llama_index.llms import Anthropic

llm = Anthropic(model="claude-3-sonnet-20240229")
```

**Local Models:**
```python
from llama_index.llms import Ollama

llm = Ollama(model="llama2")
```

#### Integration with Databases

**ChromaDB:**
```python
from llama_index.vector_stores import ChromaVectorStore
import chromadb

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
```

**Pinecone:**
```python
from llama_index.vector_stores import PineconeVectorStore
import pinecone

pinecone.init(api_key="your-key")
index = pinecone.Index("my-index")
vector_store = PineconeVectorStore(pinecone_index=index)
```

#### Other Frameworks

**Haystack (by deepset):**
- Focus: Production-ready NLP pipelines
- Strength: Document processing, QA systems
- Good for: Enterprise applications

**Semantic Kernel (Microsoft):**
- Focus: AI orchestration for .NET
- Strength: Enterprise integration
- Good for: Microsoft ecosystem

**AutoGPT / LangGraph:**
- Focus: Agentic workflows
- Strength: Multi-step reasoning
- Good for: Complex autonomous agents

### Readings

- LangChain documentation: https://python.langchain.com/
- LlamaIndex documentation: https://docs.llamaindex.ai/
- Framework comparison articles

 

### Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
- [Framework Comparison Guide](https://www.pinecone.io/learn/langchain-vs-llamaindex/)

### Practical Code Examples

#### Complete LangChain RAG Pipeline

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class LangChainRAG:
    def __init__(self, documents_path, persist_directory="./chroma_db"):
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)
        self.persist_directory = persist_directory
        self._setup_vectorstore(documents_path)
    
    def _setup_vectorstore(self, documents_path):
        """Load documents and create vector store"""
        loader = PyPDFLoader(documents_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
    
    def create_qa_chain(self):
        """Create QA chain with custom prompt"""
        prompt_template = """Use the following pieces of context to answer the question.
        If you don't know the answer, just say that you don't know.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def query(self, question):
        """Query the RAG system"""
        qa_chain = self.create_qa_chain()
        result = qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": result.get("source_documents", [])
        }

# Usage
rag = LangChainRAG("./documents.pdf")
response = rag.query("What is the main topic?")
print(response["answer"])
```

**Pro Tip:** Use LCEL (LangChain Expression Language) for cleaner, more maintainable chain definitions. It provides better debugging and streaming capabilities.

**Common Pitfall:** Not properly managing document chunking can lead to information loss. Always review chunk boundaries and adjust overlap based on document structure.

### Troubleshooting Guide

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Import errors** | ModuleNotFoundError | Install missing packages: `pip install langchain openai chromadb` |
| **Vector store errors** | Persistence failures | Check directory permissions, ensure path exists |
| **Empty retrieval** | No documents returned | Verify vector store has data, check embedding model, adjust similarity threshold |
| **Slow performance** | High latency | Use caching, optimize chunk size, batch operations |
| **Memory issues** | Out of memory | Reduce batch size, use streaming, optimize vector store |

### Quick Reference Guide

#### Framework Selection Matrix

| Use Case | Recommended Framework | Why |
|----------|---------------------|-----|
| Application orchestration | LangChain | Better agent workflows, tool integration |
| Data ingestion | LlamaIndex | Optimized loaders, indexing |
| RAG systems | Both | LangChain for chains, LlamaIndex for indexing |
| Enterprise apps | LangChain | Better integration, production features |
| Knowledge bases | LlamaIndex | Specialized query engines |

### Case Studies

#### Case Study: Building Production RAG with LangChain

**Challenge:** A company needed a production-ready RAG system for customer support.

**Solution:** Built with LangChain using:
- ChromaDB for vector storage
- GPT-4 for generation
- Custom prompt templates
- Error handling and retries

**Results:**
- 85% accuracy
- 2-second response time
- 70% cost reduction

**Lessons Learned:**
- LangChain's modularity enabled rapid iteration
- Proper error handling critical for production
- Monitoring and observability essential

### Key Takeaways

1. LangChain excels at application orchestration and agent workflows
2. LlamaIndex is optimized for data ingestion and knowledge base queries
3. Both frameworks can be used together for complementary strengths
4. LCEL provides a clean, declarative way to compose chains
5. Framework choice depends on use case and requirements
6. Integration with various LLMs and databases is straightforward in both
7. Production considerations (scaling, monitoring) should guide framework selection
8. Error handling and retries are essential for production systems
9. Proper chunking and retrieval configuration significantly impact performance
10. Testing and validation should be integrated into development workflow

---

**Previous Module:** [Module 4: Search Algorithms & Retrieval Techniques](../module_04.md)  
**Next Module:** [Module 6: RAG & Transformer Architecture](../module_06.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)

