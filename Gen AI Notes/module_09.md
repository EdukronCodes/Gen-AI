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

#### Prompt Engineering Fundamentals

**What is Prompt Engineering?**
- Art and science of crafting inputs for LLMs
- Maximizes model performance without fine-tuning
- Critical for getting desired outputs
- Cost-effective alternative to fine-tuning

**Key Principles:**
- **Clarity:** Clear, unambiguous instructions
- **Specificity:** Specific task requirements
- **Context:** Provide relevant context
- **Examples:** Include examples when helpful
- **Format:** Specify desired output format

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

**3. Chain-of-Thought (CoT) Prompting**

**Definition:**
- Encourages step-by-step reasoning
- Shows reasoning process in examples
- Improves complex reasoning tasks

**Example:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 2 × 3 = 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
A:
```

**Use Cases:**
- Math problems
- Logical reasoning
- Multi-step tasks
- Complex problem-solving

**Advantages:**
- Better reasoning
- More accurate results
- Transparent process

**Limitations:**
- Longer prompts
- More tokens
- May be slower

**4. Zero-Shot CoT**

**Definition:**
- Add "Let's think step by step" to prompt
- Model generates reasoning without examples
- Simpler than few-shot CoT

**Example:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now? Let's think step by step.
```

**5. Self-Consistency**

**Definition:**
- Generate multiple reasoning paths
- Take majority vote
- Improves accuracy

**Process:**
1. Generate multiple CoT responses
2. Extract answers
3. Select most common answer

**6. Tree of Thoughts**

**Definition:**
- Explores multiple reasoning paths
- Evaluates and prunes paths
- More sophisticated reasoning

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

#### Context Window Optimization

**Context Window Management:**

**1. Token Budgeting:**
```
Available tokens = Context window - System prompt - User prompt - Response buffer
```

**2. Prioritization:**
- Most important information first
- Recent information prioritized
- Remove redundant content

**3. Summarization:**
- Summarize long contexts
- Preserve key information
- Reduce token count

**4. Chunking Strategies:**
- Sliding window
- Hierarchical summarization
- Selective retrieval

**5. Compression:**
- Remove unnecessary tokens
- Compress similar information
- Efficient encoding

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

