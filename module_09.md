# Module 9: LLM Inference & Prompt Engineering

**Course:** Generative AI & Agentic AI  
**Module Duration:** 1 week  
**Class:** 16

---

## Class 16: Inference, Prompt Engineering & Context Windows

### Topics Covered

- Prompt templates (Zero-shot, Few-shot, Chain-of-thought)
- Context window optimization & token budgeting
- Prompt caching and reusability
- Hands-on: Design optimal prompts for a task

### Learning Objectives

By the end of this class, students will be able to:
- Understand different prompting strategies
- Design effective prompts for various tasks
- Optimize context window usage
- Implement prompt caching
- Create reusable prompt templates

### Core Concepts

#### Prompt Engineering Fundamentals - Complete Analysis

Prompt engineering is the art and science of crafting inputs for language models to maximize performance without fine-tuning. This section provides a comprehensive analysis of prompt engineering principles, strategies, and best practices.

**Prompt Engineering Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              PROMPT ENGINEERING ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────┘

User Intent
    │
    ▼
┌──────────────────┐
│ Prompt Design    │
│ • Structure      │
│ • Instructions   │
│ • Examples       │
│ • Format         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ LLM Processing   │
│ • Tokenization   │
│ • Inference     │
│ • Generation     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Output           │
│ • Result         │
│ • Quality        │
│ • Evaluation     │
└──────────────────┘
```

**What is Prompt Engineering? - Detailed Explanation:**

```
Prompt_Engineering_Definition:

Process: Crafting input text to guide LLM behavior

Mathematical Model:
For LLM f(θ) with parameters θ:

Standard Input:
P = User prompt

Prompt Engineering:
P* = Optimize(P) such that:
    Quality(f(P*, θ)) > Quality(f(P, θ))

Where:
- P: Original prompt
- P*: Optimized prompt
- f(P, θ): LLM output given prompt P
- Quality: Task-specific quality metric

Components of Effective Prompt:
P* = [System_Context, Instructions, Examples, Format, Constraints]

Where:
- System_Context: Role, domain, expertise
- Instructions: Task description, steps
- Examples: Demonstrations (few-shot)
- Format: Output structure
- Constraints: Limits, requirements
```

**Key Principles - Detailed Analysis:**

```
1. Clarity - Mathematical Formulation:

Clear Prompt:
P_clear = {instruction_i : unambiguous(instruction_i) for all i}

Where:
- unambiguous: Instruction has single interpretation
- Reduces entropy in output distribution

Example:
Bad: "Make it better"  # Ambiguous
Good: "Improve the grammar and fix spelling errors"

Clarity Metric:
Clarity(P) = 1 / H(P(output | P))

Where:
- H: Entropy of output distribution
- Lower entropy = higher clarity

2. Specificity - Mathematical Formulation:

Specific Prompt:
P_specific = P ∪ {constraint_i : specific(constraint_i)}

Where:
- constraint_i: Specific requirement
- Reduces output space

Example:
Bad: "Write a story"
Good: "Write a 200-word science fiction story about AI"

Specificity Benefit:
Reduces output space from all stories → specific subset

3. Context - Mathematical Formulation:

Contextual Prompt:
P_context = P ∪ {context_i : relevant(context_i)}

Where:
- context_i: Relevant information
- Improves conditional probability

Example:
Without context: "Translate: Hello"
With context: "Translate to French (formal): Hello"

Context Impact:
P(output | P_context) > P(output | P)  # Better conditional probability

4. Examples - Mathematical Formulation:

Few-Shot Prompt:
P_fewshot = P ∪ {example_i : (input_i, output_i)}

Where:
- example_i: Demonstration pair
- Enables in-context learning

In-Context Learning:
P(output | P_fewshot) ≈ P(output | P_finetuned)

Where:
- Few-shot can approximate fine-tuning
- No parameter updates needed

5. Format - Mathematical Formulation:

Formatted Prompt:
P_format = P ∪ {format_spec : structure(output)}

Where:
- format_spec: Output structure specification
- Ensures parseable output

Example:
Format: JSON → Ensures valid JSON output
Format: Markdown → Ensures markdown structure
```

**Prompt Engineering Benefits - Detailed Analysis:**

```
1. Cost-Effectiveness:
   - No training required
   - No GPU resources needed
   - Fast iteration
   - Low cost per experiment

2. Flexibility:
   - Easy to modify
   - Quick A/B testing
   - Version control
   - Dynamic adaptation

3. Performance:
   - Can match fine-tuning
   - Works with any LLM
   - No model changes
   - Immediate results

4. Accessibility:
   - No ML expertise needed
   - Quick to learn
   - Iterative improvement
   - Democratizes AI
```

#### Prompt Templates

**1. Zero-Shot Prompting**

**Definition:**
- No examples provided
- Model relies on pretraining knowledge
- Simple and direct

**Example:**
```
Translate the following text to French:
"Hello, how are you?"
```

**Use Cases:**
- Simple tasks
- Well-defined problems
- When model has strong prior knowledge

**Advantages:**
- Simple to use
- Fast
- Low token usage

**Limitations:**
- May not follow format
- Inconsistent results
- Limited control

**2. Few-Shot Prompting**

**Definition:**
- Provides examples in prompt
- Demonstrates desired behavior
- In-context learning

**Example:**
```
Translate the following text to French:

English: "Hello"
French: "Bonjour"

English: "Good morning"
French: "Bonjour"

English: "How are you?"
French:
```

**Use Cases:**
- Format specification
- Task-specific behavior
- Complex reasoning

**Advantages:**
- Better format control
- More consistent outputs
- No training needed

**Limitations:**
- Uses more tokens
- Examples consume context
- May overfit to examples

**3. Chain-of-Thought (CoT) Prompting - Complete Analysis**

Chain-of-Thought prompting encourages models to generate intermediate reasoning steps before arriving at the final answer. This significantly improves performance on complex reasoning tasks.

**CoT Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              CHAIN-OF-THOUGHT ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────┘

Problem Statement
    │
    ▼
┌──────────────────┐
│ Step 1:          │
│ Initial Analysis │
│ • Understand     │
│ • Identify parts │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Step 2:           │
│ Intermediate     │
│ Reasoning        │
│ • Calculate      │
│ • Deduce         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Step 3:          │
│ Final Answer     │
│ • Conclude       │
│ • Verify         │
└──────────────────┘
```

**CoT Mathematical Model:**

```
Chain_of_Thought_Model:

Standard Prompting:
P(output | problem) = Direct generation

CoT Prompting:
P(output | problem) = P(step_1 | problem) × 
                      P(step_2 | problem, step_1) × 
                      ... × 
                      P(output | problem, step_1, ..., step_n)

Where:
- step_i: Intermediate reasoning step
- Sequential generation of reasoning chain

Probability Decomposition:
P(output | problem) = Π_{i=1}^n P(step_i | problem, step_{<i})

CoT Example:
Problem: "Roger has 5 balls. He buys 2 cans of 3 balls each. How many total?"

CoT Reasoning:
Step 1: "Roger started with 5 balls"
Step 2: "2 cans × 3 balls = 6 balls"
Step 3: "5 + 6 = 11 balls"
Output: "11"

Standard Prompting:
Direct: "11" (may be wrong)

CoT Benefit:
Decomposes complex reasoning into simpler steps
Each step is easier to compute correctly
Final answer more accurate
```

**CoT Training Signal:**

```
CoT_Training_Signal:

Few-Shot CoT:
P_fewshot_cot = P ∪ {
    example_1: (problem_1, reasoning_1, answer_1),
    example_2: (problem_2, reasoning_2, answer_2),
    ...
}

Where:
- reasoning_i: Step-by-step reasoning process
- Demonstrates desired reasoning pattern

Zero-Shot CoT:
P_zeroshot_cot = P ∪ {"Let's think step by step"}

Trigger Phrase:
- "Let's think step by step"
- "Step by step"
- "Show your reasoning"
- Model learns to generate reasoning

CoT Effect:
Encourages model to:
1. Break down problem
2. Generate intermediate steps
3. Use reasoning to derive answer
4. Verify answer makes sense
```

**CoT Benefits - Detailed Analysis:**

```
1. Improved Accuracy:
   - Math problems: 60% → 90%+
   - Logical reasoning: Significant improvement
   - Multi-step tasks: Better decomposition
   
   Empirical Evidence:
   - GSM8K: 17.9% → 58.1% (GPT-3)
   - Commonsense QA: 56.4% → 60.5%
   - StrategyQA: Significant improvements

2. Transparency:
   - See reasoning process
   - Debug errors
   - Understand model thinking
   - Verify correctness

3. Better Generalization:
   - Learns reasoning patterns
   - Applies to new problems
   - More robust
   - Fewer errors

4. Error Identification:
   - Can spot reasoning errors
   - Easier to correct
   - Better debugging
   - Improved reliability
```

**CoT Limitations - Detailed Analysis:**

```
1. Token Usage:
   - Longer outputs
   - More tokens per request
   - Higher costs
   - Slower generation

2. Prompt Length:
   - Examples are long
   - Context window usage
   - May hit limits
   - Need careful management

3. Inconsistency:
   - May generate wrong reasoning
   - Can still produce errors
   - Not always correct
   - Needs verification

4. Complexity:
   - Harder to evaluate
   - More complex outputs
   - Parsing challenges
   - Verification needed
```

**Complete CoT Example:**

```
Few-Shot CoT Prompt:

Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 tennis balls.
   2 cans of tennis balls × 3 tennis balls per can = 6 tennis balls.
   5 tennis balls + 6 tennis balls = 11 tennis balls.
   The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The cafeteria started with 23 apples.
   They used 20 apples, so they had 23 - 20 = 3 apples left.
   They bought 6 more apples.
   3 apples + 6 apples = 9 apples.
   The answer is 9.

Q: {new_problem}

A:
```

**4. Zero-Shot CoT - Complete Analysis**

Zero-Shot Chain-of-Thought is a simple but powerful technique that adds a trigger phrase to encourage reasoning without providing examples.

**Zero-Shot CoT Mathematical Model:**

```
Zero_Shot_CoT_Model:

Trigger Phrase:
T = "Let's think step by step" or "Step by step"

Prompt Structure:
P_zeroshot_cot = P_problem ∪ T

Where:
- P_problem: Problem statement
- T: Trigger phrase

Generation Process:
1. Model sees problem + trigger
2. Generates reasoning steps
3. Produces final answer

Mathematical Effect:
P(reasoning | problem, trigger) > P(reasoning | problem)

The trigger phrase encourages reasoning generation

Example:
Problem: "Roger has 5 balls. He buys 2 cans of 3 balls each. How many?"
Trigger: "Let's think step by step"

Output:
"Roger started with 5 balls.
2 cans × 3 balls = 6 balls.
5 + 6 = 11 balls.
The answer is 11."
```

**Zero-Shot CoT vs Few-Shot CoT:**

```
┌──────────────────┬──────────────┬──────────────┐
│ Feature          │ Zero-Shot CoT│ Few-Shot CoT│
├──────────────────┼──────────────┼──────────────┤
│ Examples Required│ ❌ No        │ ✅ Yes       │
│ Prompt Length    │ Short        │ Long         │
│ Token Usage      │ Low          │ High         │
│ Performance      │ Good         │ Better       │
│ Flexibility      │ ✅ High      │ ⚠️ Medium    │
│ Setup Complexity │ Low          │ Medium       │
└──────────────────┴──────────────┴──────────────┘

When to Use:
- Zero-Shot: Quick iteration, simple problems
- Few-Shot: Complex problems, need examples
```

**Trigger Phrases - Effective Variants:**

```
Common Trigger Phrases:
1. "Let's think step by step"
2. "Step by step"
3. "Show your reasoning"
4. "Break this down"
5. "Think through this carefully"
6. "Explain your reasoning"

Effectiveness Ranking:
1. "Let's think step by step" - Most effective
2. "Step by step" - Very effective
3. "Show your reasoning" - Effective
4. Others - Variable effectiveness

Best Practice:
- Use "Let's think step by step" for best results
- Can be combined with other instructions
- Works across different models
```

**5. Self-Consistency - Complete Analysis**

Self-Consistency is an advanced prompting technique that generates multiple reasoning paths and selects the most consistent answer through majority voting.

**Self-Consistency Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              SELF-CONSISTENCY ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────┘

Problem Statement
    │
    ├──────────────┬──────────────┬──────────────┐
    │              │              │              │
    ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ CoT Path │  │ CoT Path │  │ CoT Path │  │ CoT Path │
│ 1        │  │ 2        │  │ 3        │  │ N        │
│ Reasoning│  │ Reasoning│  │ Reasoning│  │ Reasoning│
│ Answer: A│  │ Answer: A│  │ Answer: B│  │ Answer: A│
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │              │
     └──────────────┴──────────────┴──────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ Majority Vote     │
          │ Answer: A (3/4)   │
          └──────────────────┘
```

**Self-Consistency Mathematical Model:**

```
Self_Consistency_Model:

Single CoT Generation:
answer_1 = Generate_CoT(problem, temperature=T)

Multiple Generations:
answers = [Generate_CoT(problem, temperature=T) for i in range(N)]

Where:
- N: Number of generations (typically 5-40)
- T: Temperature (typically 0.7-1.0)
- Higher temperature = more diversity

Answer Extraction:
final_answers = [Extract_Answer(path_i) for path_i in reasoning_paths]

Majority Voting:
final_answer = Mode(final_answers)

Where:
- Mode: Most frequent answer
- Ties: Can use additional heuristics

Probability Model:
P(correct_answer) = Σ_{i=1}^N P(answer_i = correct) / N

With independence assumption:
P(all_wrong) = Π_{i=1}^N P(answer_i ≠ correct)

If each path has accuracy p:
P(all_wrong) = (1-p)^N
P(at_least_one_correct) = 1 - (1-p)^N

Example:
If p = 0.6 (60% accuracy per path):
- 1 path: 60% accuracy
- 5 paths: 92.2% accuracy (majority)
- 10 paths: 99.4% accuracy (majority)
```

**Self-Consistency Benefits:**

```
1. Improved Accuracy:
   - Single CoT: 60-70% accuracy
   - Self-Consistency: 80-90%+ accuracy
   - Significant improvement
   
   Empirical Evidence:
   - GSM8K: 58.1% → 74.4% (GPT-3)
   - Commonsense QA: 60.5% → 72.3%
   - StrategyQA: Significant improvements

2. Robustness:
   - Reduces impact of errors
   - Multiple paths compensate
   - More reliable results
   - Better generalization

3. Confidence Estimation:
   - Agreement rate = confidence
   - High agreement = high confidence
   - Low agreement = uncertain
   - Can identify hard problems
```

**Self-Consistency Limitations:**

```
1. Computational Cost:
   - N times more inference
   - N times more tokens
   - N times higher cost
   - Slower generation

2. Token Usage:
   - Multiple full generations
   - High token consumption
   - Expensive for long outputs
   - Context window limits

3. Diminishing Returns:
   - Accuracy improvement plateaus
   - N=5-10 often sufficient
   - More paths = less benefit
   - Cost-benefit trade-off
```

**6. Tree of Thoughts (ToT) - Complete Analysis:**

Tree of Thoughts is an advanced reasoning framework that explores multiple reasoning paths in a tree structure, evaluating and pruning paths to find the best solution.

**ToT Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              TREE OF THOUGHTS ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────┘

Problem
    │
    ▼
┌──────────────────┐
│ Level 1:         │
│ Initial Thoughts │
│ • Thought 1      │
│ • Thought 2      │
│ • Thought 3      │
└────────┬─────────┘
         │
         ├──────────────┬──────────────┐
         ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Level 2:     │ │ Level 2:     │ │ Level 2:     │
│ Expand       │ │ Expand       │ │ Expand       │
│ Thought 1    │ │ Thought 2    │ │ Thought 3    │
│ • Sub 1      │ │ • Sub 1      │ │ • Sub 1      │
│ • Sub 2      │ │ • Sub 2      │ │ • Sub 2      │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Evaluate &   │ │ Evaluate &   │ │ Evaluate &   │
│ Prune        │ │ Prune        │ │ Prune        │
│ Keep best    │ │ Keep best    │ │ Keep best    │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                 │                 │
       └─────────────────┴─────────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ Best Solution     │
          └──────────────────┘
```

**ToT Mathematical Model:**

```
Tree_of_Thoughts_Model:

Problem: P
Thought: t ∈ T (set of possible thoughts)

Tree Structure:
T = {t_1, t_2, ..., t_n}
Children(t_i) = {t_{i1}, t_{i2}, ..., t_{im}}

Evaluation Function:
score(t) = Evaluate(t, problem)

Where:
- score: Quality of thought
- Higher score = better thought

Search Process:
1. Generate initial thoughts: T_0 = {t_1, t_2, ..., t_k}
2. For each level:
   a. Expand: Generate children for each thought
   b. Evaluate: Score each thought
   c. Prune: Keep top-k thoughts
   d. Continue until solution found

Pruning Strategy:
Keep top-k thoughts:
T_next = {t : rank(score(t)) ≤ k}

Where:
- k: Number of thoughts to keep (typically 3-5)
- rank: Ranking by score

Best-First Search:
Select thought with highest score:
t_best = argmax_{t ∈ T} score(t)
```

**ToT Benefits:**

```
1. Better Reasoning:
   - Explores multiple paths
   - Evaluates alternatives
   - Finds optimal solution
   - Better than greedy

2. Handles Complexity:
   - Multi-step problems
   - Requires backtracking
   - Complex reasoning
   - Better decomposition

3. Quality Control:
   - Evaluates each step
   - Prunes bad paths
   - Focuses on good paths
   - Improves accuracy
```

**ToT Limitations:**

```
1. Computational Cost:
   - Multiple generations
   - Evaluation overhead
   - Very expensive
   - Slow generation

2. Complexity:
   - Complex implementation
   - Requires evaluation function
   - Hard to tune
   - Debugging challenges

3. Scalability:
   - Tree grows exponentially
   - Need effective pruning
   - Limited depth
   - Memory intensive
```


#### Advanced Prompting Techniques

**1. Role Prompting:**
- Assign role to model
- Example: "You are an expert Python programmer"
- Improves domain-specific responses

**2. Format Specification:**
- Explicitly specify output format
- JSON, XML, markdown, etc.
- Ensures parseable output

**3. Constraint Prompting:**
- Specify constraints
- Length limits, style requirements
- Content restrictions

**4. Multi-Turn Prompting:**
- Break complex tasks into steps
- Iterative refinement
- Better for complex tasks

**5. Prompt Chaining:**
- Chain multiple prompts
- Output of one becomes input of next
- Complex workflows

#### Context Window Optimization - Complete Analysis

Context window optimization is crucial for managing limited token budgets in LLM applications. This section provides comprehensive strategies for maximizing context window utilization.

**Context Window Optimization Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              CONTEXT WINDOW OPTIMIZATION                      │
└─────────────────────────────────────────────────────────────┘

Total Context Window
    │
    ├─────────────────────────────────────────────────┐
    │                                                 │
    ▼                                                 ▼
┌──────────────┐                            ┌──────────────┐
│ Fixed        │                            │ Optimizable │
│ Components   │                            │ Components   │
│ • System     │                            │ • Context    │
│ • Structure  │                            │ • Examples   │
└──────────────┘                            └──────────────┘
    │                                                 │
    ▼                                                 ▼
┌──────────────┐                            ┌──────────────┐
│ Optimization │                            │ Optimization │
│ Strategies   │                            │ Techniques   │
│ • Budgeting  │                            │ • Summarize  │
│ • Prioritize │                            │ • Chunk      │
│ • Compress   │                            │ • Filter     │
└──────────────┘                            └──────────────┘
```

**1. Token Budgeting - Mathematical Model:**

```
Token_Budgeting_Model:

Total Context Window:
T_max = Context window size

Components:
1. System Prompt: T_sys
2. User Prompt: T_user
3. Context: T_context
4. Response Buffer: T_reserve
5. Examples: T_examples (if few-shot)

Constraint:
T_sys + T_user + T_context + T_reserve + T_examples ≤ T_max

Available for Context:
T_context_available = T_max - T_sys - T_user - T_reserve - T_examples

Optimization Goal:
Maximize: T_context_available
Subject to: Quality requirements

Strategies:
1. Minimize T_sys: Concise system prompts
2. Minimize T_user: Efficient user prompts
3. Optimize T_examples: Select best examples
4. Minimize T_reserve: Appropriate buffer
```

**2. Prioritization - Complete Analysis:**

```
Prioritization_Strategy:

Mathematical Model:
For context chunks C = {c₁, c₂, ..., cₙ}:
    Priority(c_i) = Importance(c_i) × Recency(c_i) × Relevance(c_i)

Where:
- Importance: How critical is the information
- Recency: How recent is the information
- Relevance: How relevant to current query

Selection:
Select top-k chunks:
C_selected = {c_i : rank(Priority(c_i)) ≤ k}

Where:
- k: Number of chunks that fit in budget
- rank: Ranking by priority

Prioritization Heuristics:
1. Recency: Recent information first
2. Relevance: Most relevant to query
3. Importance: Critical information
4. Frequency: Frequently accessed
5. User preference: User-specified priority

Example:
For RAG system:
- Query: "What is AI?"
- Chunks: [c₁: recent AI news, c₂: AI textbook, c₃: AI history]
- Priority: c₁ > c₂ > c₃ (relevance + recency)
- Select: c₁, c₂ (if budget allows)
```

**3. Summarization - Mathematical Model:**

```
Summarization_Strategy:

Original Context:
C = {c₁, c₂, ..., cₙ}
T_original = Σ tokens(c_i)

Summarized Context:
C_summary = Summarize(C)
T_summary = tokens(C_summary)

Compression Ratio:
R = T_original / T_summary

Goal:
Maximize: Information preservation
Minimize: T_summary
Subject to: Quality threshold

Summarization Methods:
1. Extractive: Select key sentences
2. Abstractive: Generate summaries
3. Hierarchical: Multi-level summaries
4. Query-focused: Summarize relevant parts

Quality Metric:
Quality = Information_retained / Information_original

Trade-off:
Higher compression → Lower quality
Lower compression → Higher quality
```

**4. Chunking Strategies - Complete Analysis:**

```
Chunking_Strategies:

1. Sliding Window:
   For context C with length L:
       window_size = W
       step_size = S
       
       chunks = [C[i:i+W] for i in range(0, L-W, S)]
       
       Overlap = W - S
       N_chunks = ceil((L - W) / S) + 1

2. Hierarchical Summarization:
   Level 1: Document summary
   Level 2: Section summaries
   Level 3: Detailed chunks
   
   Total = T_summary + Σ T_section + Σ T_detail

3. Selective Retrieval:
   Retrieve only relevant chunks:
       chunks = Retrieve(query, C, k)
       
   Where:
   - query: User query
   - C: Full context
   - k: Number of chunks to retrieve
   
   Retrieval methods:
   - Semantic search
   - Keyword search
   - Hybrid search
```

**5. Compression Techniques:**

```
Compression_Techniques:

1. Token Removal:
   - Remove stop words
   - Remove redundancy
   - Remove filler words
   
2. Token Substitution:
   - Replace phrases with abbreviations
   - Use shorter synonyms
   - Compress whitespace
   
3. Information Encoding:
   - Use structured formats
   - Efficient encoding
   - Remove metadata
   
4. Semantic Compression:
   - Preserve meaning
   - Remove redundant information
   - Keep essential content
```

#### Prompt Caching

**Concept:**
- Cache prompt prefixes
- Reuse across requests
- Reduces computation

**Benefits:**
- Faster inference
- Lower costs
- Better throughput

**Use Cases:**
- System prompts
- Few-shot examples
- Common prefixes

**Implementation:**
- API support (OpenAI, Anthropic)
- Framework support (vLLM, etc.)
- Cache management

#### Prompt Optimization Process

**1. Define Objective:**
- Clear success criteria
- Measurable metrics
- Task requirements

**2. Create Baseline:**
- Start with simple prompt
- Test and measure
- Establish baseline

**3. Iterate:**
- Add examples
- Refine instructions
- Test variations

**4. Evaluate:**
- Measure performance
- Compare versions
- A/B testing

**5. Refine:**
- Optimize based on results
- Remove unnecessary parts
- Finalize prompt

#### Prompt Templates Library

**Common Patterns:**

**Classification:**
```
Classify the following text: "{text}"

Categories: {categories}

Answer:
```

**Extraction:**
```
Extract the following information from the text:

Text: "{text}"
Information to extract: {fields}

Format as JSON:
```

**Summarization:**
```
Summarize the following text in {n} sentences:

Text: "{text}"

Summary:
```

**Question Answering:**
```
Answer the question based on the context:

Context: "{context}"
Question: "{question}"

Answer:
```

#### Prompt Engineering Best Practices

**1. Be Specific:**
- Clear instructions
- Specific requirements
- Avoid ambiguity

**2. Provide Examples:**
- Show desired format
- Demonstrate behavior
- Include edge cases

**3. Use Formatting:**
- Clear structure
- Sections and headers
- Visual organization

**4. Test Thoroughly:**
- Multiple test cases
- Edge cases
- Error handling

**5. Iterate and Refine:**
- Continuous improvement
- Measure results
- Optimize based on feedback

**6. Document:**
- Keep prompt versions
- Document changes
- Share learnings

### Readings

- Prompt engineering guides:
  - "Language Models are Few-Shot Learners" (Brown et al., 2020)
  - "Chain-of-Thought Prompting Elicits Reasoning" (Wei et al., 2022)
  - "Tree of Thoughts: Deliberate Problem Solving" (Yao et al., 2023)

- Recent papers on in-context learning:
  - "What Makes In-Context Learning Work?" (Min et al., 2022)
  - "Large Language Models are Zero-Shot Reasoners" (Kojima et al., 2022)

 

### Additional Resources

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/)

### Practical Code Examples

#### Advanced Prompt Engineering Patterns

```python
from typing import List, Dict
from openai import OpenAI

class PromptEngine:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def zero_shot(self, task: str, input_text: str) -> str:
        """Zero-shot prompting"""
        prompt = f"""Task: {task}
        
Input: {input_text}
        
Output:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def few_shot(self, task: str, examples: List[Dict], input_text: str) -> str:
        """Few-shot prompting with examples"""
        examples_text = "\n\n".join([
            f"Input: {ex['input']}\nOutput: {ex['output']}"
            for ex in examples
        ])
        
        prompt = f"""Task: {task}

Examples:
{examples_text}

Input: {input_text}
Output:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def chain_of_thought(self, problem: str) -> str:
        """Chain-of-thought prompting"""
        prompt = f"""Solve the following problem step by step.

Problem: {problem}

Let's think step by step:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def optimize_prompt(self, base_prompt: str, examples: List[Dict], 
                       iterations: int = 3) -> str:
        """Iteratively optimize prompt"""
        best_prompt = base_prompt
        best_score = 0
        
        for i in range(iterations):
            # Test prompt variant
            # Score based on results
            # Refine prompt
            
            # Simplified: In practice, use evaluation metrics
            pass
        
        return best_prompt

# Usage
import os
engine = PromptEngine(api_key=os.getenv("OPENAI_API_KEY"))

# Zero-shot
result = engine.zero_shot("Summarize", "Long text here...")

# Few-shot
examples = [
    {"input": "text1", "output": "summary1"},
    {"input": "text2", "output": "summary2"}
]
result = engine.few_shot("Summarize", examples, "text3")

# Chain-of-thought
result = engine.chain_of_thought("Complex problem here...")
```

**Pro Tip:** Start with zero-shot, add examples if needed, then use chain-of-thought for complex reasoning. Always test multiple prompt variations.

**Common Pitfall:** Over-engineering prompts can reduce effectiveness. Start simple and only add complexity when needed.

#### Token Budget Management

```python
import tiktoken

class TokenBudgetManager:
    def __init__(self, model="gpt-4"):
        self.encoder = tiktoken.encoding_for_model(model)
        self.context_windows = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))
    
    def budget_context(self, system_prompt: str, user_prompt: str,
                      model: str, response_buffer: int = 500) -> int:
        """Calculate available tokens for context"""
        context_window = self.context_windows.get(model, 4096)
        
        system_tokens = self.count_tokens(system_prompt)
        user_tokens = self.count_tokens(user_prompt)
        
        used = system_tokens + user_tokens + response_buffer
        available = context_window - used
        
        return max(0, available)
    
    def truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token budget"""
        tokens = self.encoder.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        truncated = tokens[:max_tokens]
        return self.encoder.decode(truncated)

# Usage
manager = TokenBudgetManager()
available = manager.budget_context(
    system_prompt="You are a helpful assistant.",
    user_prompt="What is AI?",
    model="gpt-4",
    response_buffer=500
)
print(f"Available for context: {available} tokens")
```

### Troubleshooting Guide

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Poor prompt performance** | Low quality outputs | Add examples, refine instructions, use chain-of-thought |
| **Context overflow** | Token limit errors | Implement token budgeting, use summarization, chunk documents |
| **Inconsistent results** | Different outputs for same input | Lower temperature, use seed, refine prompt |
| **High costs** | Expensive API calls | Optimize prompts, use caching, implement token limits |
| **Slow responses** | High latency | Use faster models, optimize prompts, implement caching |

### Quick Reference Guide

#### Prompting Strategies

| Strategy | Use When | Example |
|----------|----------|---------|
| Zero-shot | Simple tasks, clear instructions | "Summarize this text" |
| Few-shot | Complex tasks, need examples | Provide 2-5 examples |
| Chain-of-thought | Reasoning tasks | "Let's think step by step" |
| Self-consistency | Important decisions | Multiple reasoning paths |

### Key Takeaways

1. Prompt engineering is crucial for maximizing LLM performance
2. Few-shot and chain-of-thought prompting significantly improve results
3. Context window management is essential for long documents
4. Prompt caching improves efficiency and reduces costs
5. Iterative refinement leads to better prompts
6. Different prompting strategies suit different tasks
7. Well-designed prompts can match fine-tuning performance for many tasks
8. Token budgeting prevents context overflow and manages costs
9. Prompt optimization should be data-driven with proper evaluation
10. Caching and optimization reduce costs and improve latency

---

**Previous Module:** [Module 8: LLM Training & Fine-tuning](../module_08.md)  
**Next Module:** [Module 10: Database, Frameworks & Deployment](../module_10.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)

