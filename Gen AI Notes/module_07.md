# Module 7: Tokenization & Embeddings in LLMs

**Course:** Generative AI & Agentic AI  
**Module Duration:** 1 week  
**Class:** 12

---

## Class 12: Tokenization, Embeddings, Positional Encoding

### Topics Covered

- Byte-Pair Encoding (BPE)
- Token limits, context window management
- Positional encodings and their mathematical intuition
- Tokenization strategies and trade-offs

### Learning Objectives

By the end of this class, students will be able to:
- Understand different tokenization algorithms
- Implement BPE tokenization
- Manage context windows and token limits
- Understand positional encoding mechanisms
- Choose appropriate tokenization for use case

### Core Concepts

#### Tokenization Fundamentals

**What is Tokenization?**
- Breaking text into smaller units (tokens)
- Required for LLM input processing
- Affects model performance and efficiency
- Different strategies for different languages

**Tokenization Levels:**
- **Character-level:** Each character is a token
- **Word-level:** Each word is a token
- **Subword-level:** Words split into subwords (common)
- **Sentence-level:** Entire sentences (rare)

**Challenges:**
- Out-of-vocabulary (OOV) words
- Multilingual support
- Special tokens handling
- Token limit management

#### Byte-Pair Encoding (BPE)

**History:**
- Originally for data compression
- Adapted for NLP by Sennrich et al. (2016)
- Used in GPT, GPT-2, GPT-3, GPT-4
- Foundation for many modern tokenizers

**Algorithm:**

**Training Phase:**
```
1. Initialize vocabulary with all characters
2. While vocabulary size < desired size:
   a. Count all pairs of consecutive tokens
   b. Find most frequent pair
   c. Merge pair into new token
   d. Add to vocabulary
3. Return vocabulary and merge rules
```

**Example:**
```
Initial: "low", "lower", "newest", "widest"
Vocabulary: {l, o, w, e, r, n, s, t, i, d}

Step 1: Most frequent pair: "lo" (2 times)
Merge: "lo" → "lo"
Vocabulary: {l, o, w, e, r, n, s, t, i, d, lo}

Step 2: Most frequent pair: "low" (2 times)
Merge: "low" → "low"
Vocabulary: {..., low}

Continue until desired vocabulary size
```

**Encoding Phase:**
```
1. Split text into characters
2. Apply merge rules in order
3. Return token IDs
```

**Benefits:**
- Handles OOV words (subword decomposition)
- Balances vocabulary size and token count
- Language-agnostic (works with any text)
- Efficient encoding and decoding

**Limitations:**
- Order-dependent merge rules
- May split words unintuitively
- Fixed vocabulary size

#### WordPiece Tokenization

**Difference from BPE:**
- Uses likelihood-based merging
- Selects pairs that maximize language model likelihood
- Used in BERT
- More principled approach

**Algorithm:**
```
1. Initialize vocabulary with characters
2. While vocabulary size < desired:
   a. For each possible pair, compute likelihood increase
   b. Merge pair with highest likelihood increase
   c. Add to vocabulary
```

#### SentencePiece Tokenization

**Features:**
- Treats text as raw Unicode
- Language-agnostic
- Reversible (no information loss)
- Used in T5, LLaMA, many multilingual models

**Benefits:**
- Handles multiple languages well
- No preprocessing needed
- Preserves all information

#### Token Limits & Context Window Management

**Context Window:**
- Maximum number of tokens model can process
- Fixed for each model architecture
- Examples:
  - GPT-3.5-turbo: 4K, 16K tokens
  - GPT-4: 8K, 32K, 128K tokens
  - Claude 3: 200K tokens
  - Gemini 1.5: 1M tokens (experimental)

**Token Limit Strategies:**

**1. Truncation:**
- Cut text at token limit
- Simple but loses information
- Can use head, tail, or middle

**2. Chunking:**
- Split into multiple chunks
- Process each chunk separately
- Combine results
- Common in RAG systems

**3. Summarization:**
- Summarize long text
- Reduce token count
- Preserves key information

**4. Sliding Window:**
- Process overlapping windows
- Maintains context
- More computation

**5. Hierarchical Processing:**
- Process at multiple levels
- Summary + details
- Efficient for long documents

**Token Budgeting:**
```
Total tokens = System prompt + User prompt + Context + Response

For RAG:
Tokens available = Context window - System prompt - User prompt - Response buffer
Context chunks = Available tokens / Average chunk size
```

#### Positional Encoding

**Why Needed?**
- Attention is permutation-invariant
- Need to encode sequence position
- Enables model to understand order

**Sinusoidal Positional Encoding:**

**Mathematical Formulation:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
- pos = position in sequence
- i = dimension index
- d_model = model dimension
```

**Properties:**
- Fixed, not learned
- Can extrapolate to longer sequences
- Relative positions encoded
- Different frequencies for different dimensions

**Intuition:**
- Lower dimensions: Lower frequency (longer patterns)
- Higher dimensions: Higher frequency (shorter patterns)
- Each position has unique encoding
- Similar positions have similar encodings

**Learned Positional Embeddings:**
- Learned parameters (like word embeddings)
- Better for training data lengths
- Cannot extrapolate beyond training
- Used in BERT, GPT-2

**Relative Position Encoding:**
- Encodes relative positions
- Better for variable-length sequences
- Used in some transformer variants

#### Comparison of Tokenization Methods

| Method | Used By | Pros | Cons |
|--------|---------|------|------|
| BPE | GPT family | Simple, effective | Order-dependent |
| WordPiece | BERT | Likelihood-based | More complex |
| SentencePiece | T5, LLaMA | Language-agnostic | Larger vocabulary |
| Unigram | Some models | Probabilistic | Slower training |

#### Tokenization Best Practices

**1. Choose Appropriate Tokenizer:**
- Match tokenizer to model
- Consider language support
- Evaluate vocabulary size

**2. Handle Special Tokens:**
- Padding tokens
- Unknown tokens
- Separator tokens
- Task-specific tokens

**3. Manage Context Windows:**
- Monitor token counts
- Implement chunking strategies
- Use summarization when needed

**4. Optimize Token Usage:**
- Remove unnecessary whitespace
- Use efficient prompts
- Compress context when possible

**5. Multilingual Considerations:**
- Use multilingual tokenizers
- Handle different scripts
- Consider tokenization efficiency per language

### Readings

- BPE and tokenization papers:
  - "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)
  - "Google's Neural Machine Translation System" (Wu et al., 2016)
  - "SentencePiece: A simple and language independent subword tokenizer" (Kudo & Richardson, 2018)

- Positional encoding research:
  - "Attention Is All You Need" (Vaswani et al., 2017) - Sinusoidal encoding
  - "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018) - Learned embeddings
  - "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" (Dai et al., 2019) - Relative encoding

 

### Additional Resources

- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)
- [SentencePiece GitHub](https://github.com/google/sentencepiece)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/) - Shows tokenization
- [Tokenization Guide](https://huggingface.co/learn/nlp-course/chapter6/1)

### Practical Code Examples

#### Tokenization and Token Counting

```python
from transformers import AutoTokenizer
from typing import List
import tiktoken

class TokenManager:
    def __init__(self, model_name="gpt-3.5-turbo"):
        # OpenAI tokenizer
        self.openai_encoder = tiktoken.encoding_for_model(model_name)
        
        # Hugging Face tokenizer
        self.hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    def count_tokens_openai(self, text: str) -> int:
        """Count tokens using OpenAI tokenizer"""
        return len(self.openai_encoder.encode(text))
    
    def count_tokens_hf(self, text: str) -> int:
        """Count tokens using Hugging Face tokenizer"""
        return len(self.hf_tokenizer.encode(text))
    
    def budget_tokens(self, text: str, max_tokens: int, 
                     reserved: int = 500) -> str:
        """Truncate text to fit token budget"""
        tokens = self.openai_encoder.encode(text)
        
        if len(tokens) <= max_tokens - reserved:
            return text
        
        # Truncate to fit budget
        allowed_tokens = max_tokens - reserved
        truncated_tokens = tokens[:allowed_tokens]
        return self.openai_encoder.decode(truncated_tokens)
    
    def chunk_by_tokens(self, text: str, chunk_size: int, 
                       overlap: int = 100) -> List[str]:
        """Chunk text by token count"""
        tokens = self.openai_encoder.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.openai_encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks

# Usage
token_manager = TokenManager()
text = "Your long text here..."
token_count = token_manager.count_tokens_openai(text)
print(f"Token count: {token_count}")

chunks = token_manager.chunk_by_tokens(text, chunk_size=1000, overlap=200)
```

**Pro Tip:** Always use the tokenizer that matches your model. Different tokenizers can produce different token counts for the same text.

**Common Pitfall:** Using character or word count instead of token count can lead to context overflow errors. Always count tokens accurately.

### Troubleshooting Guide

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Context overflow** | Token limit exceeded | Count tokens, implement chunking, use summarization |
| **Tokenizer mismatch** | Different token counts | Use matching tokenizer for your model |
| **Slow tokenization** | High latency | Cache tokenized results, use faster tokenizers |
| **Encoding errors** | Unicode errors | Ensure UTF-8 encoding, handle special characters |
| **Token budget issues** | Incomplete responses | Reserve tokens for response, monitor usage |

### Quick Reference Guide

#### Tokenization Methods Comparison

| Method | Used By | Pros | Cons |
|--------|---------|------|------|
| BPE | GPT | Simple, effective | Order-dependent |
| WordPiece | BERT | Likelihood-based | More complex |
| SentencePiece | T5, LLaMA | Language-agnostic | Larger vocabulary |

#### Token Budget Calculator

```python
def calculate_token_budget(context_window: int, system_prompt: str, 
                          user_prompt: str, response_buffer: int = 500) -> int:
    """Calculate available tokens for context"""
    token_manager = TokenManager()
    
    system_tokens = token_manager.count_tokens_openai(system_prompt)
    user_tokens = token_manager.count_tokens_openai(user_prompt)
    
    used = system_tokens + user_tokens + response_buffer
    available = context_window - used
    
    return max(0, available)

# Usage
available = calculate_token_budget(
    context_window=4096,
    system_prompt="You are a helpful assistant.",
    user_prompt="What is AI?",
    response_buffer=500
)
print(f"Available tokens for context: {available}")
```

### Key Takeaways

1. Tokenization is crucial for LLM input processing
2. BPE provides good balance between vocabulary size and token efficiency
3. Different tokenization methods suit different models and languages
4. Context window management is essential for long documents
5. Positional encoding enables models to understand sequence order
6. Token budgeting helps optimize RAG and other systems
7. Choice of tokenizer affects model performance and efficiency
8. Always count tokens accurately to avoid context overflow
9. Chunking strategies must account for token boundaries
10. Token-aware preprocessing improves system efficiency

---

**Previous Module:** [Module 6: RAG & Transformer Architecture](../module_06.md)  
**Next Module:** [Module 8: LLM Training & Fine-tuning](../module_08.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)

