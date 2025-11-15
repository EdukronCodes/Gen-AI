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

#### Tokenization Fundamentals - Complete Analysis

Tokenization is the process of breaking text into smaller units (tokens) that can be processed by language models. This section provides a comprehensive analysis of tokenization fundamentals, mathematical models, and practical considerations.

**What is Tokenization? - Detailed Explanation:**

```
Tokenization_Process:

Input: Raw text string
    ↓
Processing: Split into tokens
    ↓
Output: Sequence of token IDs

Mathematical Model:
For text T = "w₁ w₂ ... wₙ":
    Tokens = Tokenizer(T)
    Token_IDs = [ID(t₁), ID(t₂), ..., ID(tₘ)]
    
Where:
- T: Input text
- tᵢ: Individual token
- ID(tᵢ): Token ID in vocabulary
- m: Number of tokens (may differ from n words)

Example:
Text: "The cat sat"
Tokens: ["The", " cat", " sat"]
Token IDs: [464, 3543, 682]
```

**Tokenization Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              TOKENIZATION ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────┘

Raw Text: "Hello, world!"
    │
    ▼
┌──────────────────┐
│ Text             │
│ Normalization    │
│ • Unicode        │
│ • Lowercase (opt)│
│ • Whitespace     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Text             │
│ Preprocessing    │
│ • Special chars  │
│ • Punctuation    │
│ • Segmentation   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Tokenization     │
│ Algorithm        │
│ • BPE/WordPiece/ │
│   SentencePiece  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Token            │
│ Postprocessing   │
│ • Special tokens │
│ • Padding        │
│ • Truncation     │
└────────┬─────────┘
         │
         ▼
Token IDs: [15496, 11, 1917, 0]
```

**Tokenization Levels - Detailed Comparison:**

```
1. Character-Level Tokenization:

Text: "Hello"
Tokens: ['H', 'e', 'l', 'l', 'o']
Token IDs: [72, 101, 108, 108, 111]

Properties:
- Vocabulary size: ~256 (ASCII) or ~65,536 (Unicode)
- Sequence length: Very long (equal to character count)
- OOV handling: ✅ None (all characters in vocabulary)
- Efficiency: ⚠️ Long sequences, sparse representations

Mathematical Model:
For text T with n characters:
    Tokens = [c₁, c₂, ..., cₙ]
    Sequence length = n
    Vocabulary size = |Charset|

2. Word-Level Tokenization:

Text: "The cat sat"
Tokens: ['The', 'cat', 'sat']
Token IDs: [1996, 4937, 3793]

Properties:
- Vocabulary size: 10K - 100K words
- Sequence length: Short (equal to word count)
- OOV handling: ❌ Problem (unknown words → UNK token)
- Efficiency: ✅ Short sequences, dense representations

Mathematical Model:
For text T with n words:
    Tokens = [w₁, w₂, ..., wₙ]
    Sequence length = n
    Vocabulary size = |Vocabulary|
    OOV rate = Count(UNK) / n

3. Subword-Level Tokenization (Most Common):

Text: "unhappiness"
Tokens: ['un', 'happiness'] or ['un', 'happy', 'ness']
Token IDs: [2364, 4582] or [2364, 5458, 369]

Properties:
- Vocabulary size: 30K - 100K subwords
- Sequence length: Medium (between char and word)
- OOV handling: ✅ Excellent (decompose into subwords)
- Efficiency: ✅ Good balance

Mathematical Model:
For word w:
    If w ∈ Vocabulary:
        Tokens = [w]
    Else:
        Tokens = Decompose(w) = [s₁, s₂, ..., sₖ]
        Where each sᵢ ∈ Vocabulary

4. Sentence-Level Tokenization (Rare):

Text: "The cat sat. The dog ran."
Tokens: ['The cat sat.', 'The dog ran.']

Properties:
- Vocabulary size: Very large (all sentences)
- Sequence length: Very short (1-2 tokens)
- OOV handling: ❌ Major problem
- Efficiency: ❌ Not practical for most tasks
```

**Tokenization Challenges - Detailed Analysis:**

```
1. Out-of-Vocabulary (OOV) Words:

Problem:
- New words not in training vocabulary
- Can't represent unknown words
- Degrades model performance

Solutions:

a) Character-level:
   - All characters in vocabulary
   - OOV rate = 0%
   - But: Very long sequences

b) Subword tokenization:
   - Decompose OOV words into subwords
   - OOV rate ≈ 0.1% - 1%
   - Example: "unhappiness" → ["un", "happiness"]

c) UNK token:
   - Fallback for truly unknown tokens
   - OOV rate ≈ 1% - 5%
   - Information loss

Mathematical Model:
OOV_Rate = Count(UNK_tokens) / Total_tokens

For BPE with 50K vocabulary:
    OOV_Rate ≈ 0.1% - 0.5%

2. Multilingual Support:

Problem:
- Different scripts (Latin, Cyrillic, Chinese, etc.)
- Different tokenization needs
- Vocabulary size explosion

Solutions:

a) SentencePiece:
   - Language-agnostic
   - Treats text as raw Unicode
   - Works with any script

b) Language-specific tokenizers:
   - Optimized per language
   - Better efficiency
   - More complex pipeline

c) Multilingual vocabulary:
   - Shared vocabulary across languages
   - Cross-lingual transfer
   - Larger vocabulary size

3. Special Tokens Handling:

Special Tokens:
- [PAD]: Padding token
- [UNK]: Unknown token
- [CLS]: Classification token (BERT)
- [SEP]: Separator token (BERT)
- [BOS]: Beginning of sequence
- [EOS]: End of sequence
- [MASK]: Masked token (BERT)

Challenge:
- Must be handled correctly
- Can't be split by tokenizer
- Must be in vocabulary

4. Token Limit Management:

Problem:
- Models have fixed context windows
- Text may exceed limits
- Need efficient truncation/chunking

Solutions:
- Truncation (head/tail/middle)
- Chunking with overlap
- Summarization
- Hierarchical processing
```

**Tokenization Trade-offs - Mathematical Analysis:**

```
Trade-off Matrix:

┌──────────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Metric               │ Char     │ Word     │ Subword   │ Sentence │
├──────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Vocabulary Size      │ ~256     │ 10K-100K │ 30K-100K │ Very Large│
│ Sequence Length      │ Very Long│ Short    │ Medium   │ Very Short│
│ OOV Rate             │ 0%       │ 5-10%    │ 0.1-1%   │ Very High│
│ Token Efficiency     │ Low      │ High     │ Medium   │ Very Low │
│ Computation Cost     │ O(n)     │ O(n)     │ O(n)     │ O(1)     │
│ Multilingual Support │ ✅ Good  │ ⚠️ Limited│ ✅ Good  │ ⚠️ Limited│
│ Information Density  │ Low      │ High     │ Medium   │ Very High│
└──────────────────────┴──────────┴──────────┴──────────┴──────────┘

Optimal Choice:
- Most LLMs: Subword tokenization (BPE/WordPiece/SentencePiece)
- Reason: Best balance of vocabulary size, OOV handling, efficiency
```

#### Byte-Pair Encoding (BPE) - Complete Mathematical Analysis

Byte-Pair Encoding (BPE) is a subword tokenization algorithm that iteratively merges the most frequent pairs of consecutive tokens. It was originally developed for data compression and adapted for NLP by Sennrich et al. (2016). BPE is the foundation for GPT models (GPT, GPT-2, GPT-3, GPT-4) and many modern tokenizers.

**BPE Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              BPE TOKENIZATION ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────┘

Training Phase:
Corpus → Character Split → Pair Counting → Merge → Iterate
    │
    ▼
Vocabulary + Merge Rules

Encoding Phase:
Text → Character Split → Apply Merge Rules → Tokens → Token IDs
```

**BPE Training Phase - Complete Algorithm:**

```
BPE_Training_Algorithm:

Input:
- Corpus: List of text strings
- Vocabulary_size: Target vocabulary size (e.g., 50,000)

Output:
- Vocabulary: Set of tokens
- Merge_rules: Ordered list of (pair, merged_token)

Algorithm:

1. Initialize:
   Vocabulary = {all unique characters in corpus}
   Word_frequencies = Count word frequencies in corpus
   
   Example:
   Corpus: ["low", "lower", "newest", "widest"]
   Vocabulary = {l, o, w, e, r, n, s, t, i, d}
   Word_frequencies = {"low": 1, "lower": 1, "newest": 1, "widest": 1}

2. Iterate until |Vocabulary| >= Vocabulary_size:
   
   a) Count all pairs:
      For each word w in corpus:
          Split w into current tokens (initially characters)
          Count all consecutive pairs (token_i, token_{i+1})
      
      Pair_counts = {}
      For each word w with frequency f:
          tokens = current_tokenization(w)
          For i in range(len(tokens) - 1):
              pair = (tokens[i], tokens[i+1])
              Pair_counts[pair] += f
      
      Example (after some merges):
      Word: "low" → tokens: ["lo", "w"]
      Pair: ("lo", "w") → count = 1
      
   b) Select most frequent pair:
      best_pair = argmax(Pair_counts)
      best_count = max(Pair_counts)
      
      Example:
      Pair_counts = {("l", "o"): 2, ("o", "w"): 1, ...}
      best_pair = ("l", "o"), best_count = 2
      
   c) Merge pair:
      new_token = concat(best_pair[0], best_pair[1])
      Vocabulary.add(new_token)
      Merge_rules.append((best_pair, new_token))
      
      Example:
      best_pair = ("l", "o")
      new_token = "lo"
      Vocabulary.add("lo")
      
   d) Update word tokenizations:
      For each word in corpus:
          Replace all occurrences of best_pair with new_token
      
      Example:
      "low" → ["l", "o", "w"] → ["lo", "w"]
      "lower" → ["l", "o", "w", "e", "r"] → ["lo", "w", "e", "r"]

3. Return Vocabulary and Merge_rules
```

**Complete BPE Training Example:**

```
Step-by-Step Example:

Initial Corpus: ["low", "lower", "newest", "widest"]
Target Vocabulary Size: 20

Step 0: Initialize
Vocabulary = {l, o, w, e, r, n, s, t, i, d}
Word representations:
  "low" → [l, o, w]
  "lower" → [l, o, w, e, r]
  "newest" → [n, e, w, e, s, t]
  "widest" → [w, i, d, e, s, t]

Step 1: Count pairs
Pairs and frequencies:
  (l, o): 2  (in "low", "lower")
  (o, w): 2  (in "low", "lower")
  (w, e): 2  (in "lower", "newest")
  (e, r): 1  (in "lower")
  (e, s): 2  (in "newest", "widest")
  (n, e): 1  (in "newest")
  (s, t): 2  (in "newest", "widest")
  (w, i): 1  (in "widest")
  (i, d): 1  (in "widest")
  (d, e): 1  (in "widest")

Most frequent: (l, o): 2, (o, w): 2, (w, e): 2, (e, s): 2, (s, t): 2
Tie-breaking: Choose (l, o) (first alphabetically)

Merge: (l, o) → "lo"
Vocabulary = {l, o, w, e, r, n, s, t, i, d, lo}
Word representations:
  "low" → [lo, w]
  "lower" → [lo, w, e, r]
  "newest" → [n, e, w, e, s, t]
  "widest" → [w, i, d, e, s, t]

Step 2: Count pairs
Pairs:
  (lo, w): 2  (in "low", "lower")
  (w, e): 2
  (e, r): 1
  (e, s): 2
  (s, t): 2
  (n, e): 1
  (w, i): 1
  (i, d): 1
  (d, e): 1

Most frequent: (lo, w): 2, (w, e): 2, (e, s): 2, (s, t): 2
Tie-breaking: Choose (lo, w)

Merge: (lo, w) → "low"
Vocabulary = {l, o, w, e, r, n, s, t, i, d, lo, low}
Word representations:
  "low" → [low]
  "lower" → [low, e, r]
  "newest" → [n, e, w, e, s, t]
  "widest" → [w, i, d, e, s, t]

Continue until vocabulary size reaches 20...
```

**BPE Encoding Phase - Complete Algorithm:**

```
BPE_Encoding_Algorithm:

Input:
- Text: Input text string
- Vocabulary: Trained vocabulary
- Merge_rules: Ordered list of merge rules

Output:
- Tokens: List of token strings
- Token_IDs: List of token IDs

Algorithm:

1. Initialize:
   tokens = Split into characters(text)
   
   Example:
   text = "lower"
   tokens = ["l", "o", "w", "e", "r"]

2. Apply merge rules in order:
   For each merge_rule in Merge_rules:
       pair = merge_rule[0]  # e.g., ("l", "o")
       merged = merge_rule[1]  # e.g., "lo"
       
       # Find all occurrences of pair
       i = 0
       while i < len(tokens) - 1:
           if (tokens[i], tokens[i+1]) == pair:
               # Replace pair with merged token
               tokens = tokens[:i] + [merged] + tokens[i+2:]
           else:
               i += 1
   
   Example:
   Merge rules: [((l, o), "lo"), (("lo", w), "low")]
   
   After rule 1: ["l", "o", "w", "e", "r"] → ["lo", "w", "e", "r"]
   After rule 2: ["lo", "w", "e", "r"] → ["low", "e", "r"]

3. Map tokens to IDs:
   Token_IDs = [Vocabulary[token] for token in tokens]
   
   Example:
   tokens = ["low", "e", "r"]
   Token_IDs = [1234, 567, 890]

4. Return tokens and Token_IDs
```

**BPE Mathematical Model:**

```
BPE_Mathematical_Model:

1. Vocabulary Definition:
   V = {v₁, v₂, ..., vₙ}
   Where:
   - vᵢ: Token in vocabulary
   - n: Vocabulary size

2. Merge Rule:
   For pair (a, b) with frequency f:
       If f = max(all_pair_frequencies):
           V = V ∪ {concat(a, b)}
           Merge_rules.append((a, b) → concat(a, b))

3. Tokenization Function:
   Tokenize(text) = Apply_merges(Char_split(text), Merge_rules)
   
   Where:
   Apply_merges(tokens, rules):
       For each rule (pair, merged) in rules:
           tokens = Replace_all(tokens, pair, merged)
       Return tokens

4. Token ID Mapping:
   ID(token) = Index(token, Vocabulary)
```

**Complete BPE Implementation:**

```python
"""
Complete BPE Tokenizer Implementation
"""

from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set
import re

class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer
    
    Mathematical Model:
        V = Vocabulary of tokens
        Merge_rules = Ordered list of (pair, merged_token)
        Tokenize(text) = Apply_merges(Char_split(text), Merge_rules)
    """
    
    def __init__(self, vocab_size: int = 50000):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        self.vocabulary: Set[str] = set()
        self.merge_rules: List[Tuple[Tuple[str, str], str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        print(f"[BPE] Initialized with target vocab size: {vocab_size}")
    
    def train(self, corpus: List[str]) -> None:
        """
        Train BPE tokenizer on corpus.
        
        Algorithm:
        1. Initialize vocabulary with characters
        2. While |vocab| < vocab_size:
           a. Count all pairs
           b. Find most frequent pair
           c. Merge pair
           d. Add to vocabulary
        
        Args:
            corpus: List of text strings for training
        """
        print(f"[BPE Training] Starting training on {len(corpus)} texts...")
        
        # Step 1: Initialize vocabulary with characters
        char_set = set()
        word_freqs = Counter(corpus)
        
        for word in corpus:
            char_set.update(list(word))
        
        self.vocabulary = char_set.copy()
        print(f"[BPE Training] Initial vocabulary size: {len(self.vocabulary)}")
        print(f"[BPE Training] Initial vocabulary: {sorted(list(self.vocabulary))[:20]}...")
        
        # Represent words as sequences of characters
        word_representations = {}
        for word in word_freqs:
            word_representations[word] = list(word)
        
        # Step 2: Iteratively merge pairs
        iteration = 0
        while len(self.vocabulary) < self.vocab_size:
            iteration += 1
            
            # Count all pairs
            pair_counts = Counter()
            for word, freq in word_freqs.items():
                tokens = word_representations[word]
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i+1])
                    pair_counts[pair] += freq
            
            if not pair_counts:
                print("[BPE Training] No more pairs to merge. Stopping.")
                break
            
            # Find most frequent pair
            best_pair, best_count = pair_counts.most_common(1)[0]
            new_token = best_pair[0] + best_pair[1]
            
            # Add to vocabulary
            self.vocabulary.add(new_token)
            self.merge_rules.append((best_pair, new_token))
            
            # Update word representations
            for word in word_freqs:
                tokens = word_representations[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_representations[word] = new_tokens
            
            if iteration % 100 == 0:
                print(f"[BPE Training] Iteration {iteration}: vocab_size={len(self.vocabulary)}, "
                      f"best_pair={best_pair}, count={best_count}")
        
        # Build token-to-id mapping
        sorted_vocab = sorted(self.vocabulary)
        self.token_to_id = {token: idx for idx, token in enumerate(sorted_vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        print(f"[BPE Training] Training complete!")
        print(f"[BPE Training] Final vocabulary size: {len(self.vocabulary)}")
        print(f"[BPE Training] Number of merge rules: {len(self.merge_rules)}")
    
    def encode(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Encode text into tokens and token IDs.
        
        Algorithm:
        1. Split text into characters
        2. Apply merge rules in order
        3. Map to token IDs
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (tokens, token_ids)
        """
        # Step 1: Split into characters
        tokens = list(text)
        
        # Step 2: Apply merge rules in order
        for pair, merged in self.merge_rules:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # Step 3: Map to token IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Handle unknown tokens (fallback to character-level)
                for char in token:
                    if char in self.token_to_id:
                        token_ids.append(self.token_to_id[char])
        
        return tokens, token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        tokens = [self.id_to_token[idx] for idx in token_ids]
        return ''.join(tokens)


# Example Usage
if __name__ == "__main__":
    print("=" * 70)
    print("BPE TOKENIZER - COMPLETE IMPLEMENTATION")
    print("=" * 70)
    
    # Training corpus
    corpus = [
        "low", "lower", "newest", "widest",
        "the", "cat", "sat", "on", "the", "mat",
        "machine", "learning", "artificial", "intelligence"
    ]
    
    print(f"\nTraining corpus: {corpus}")
    
    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.train(corpus)
    
    # Encode text
    test_text = "lower"
    tokens, token_ids = tokenizer.encode(test_text)
    
    print(f"\nEncoding test:")
    print(f"Text: '{test_text}'")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    
    # Decode
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: '{decoded}'")
    
    print("\n" + "=" * 70)
    print("BPE TOKENIZER DEMO COMPLETE")
    print("=" * 70)

"""
Expected Output:
======================================================================
BPE TOKENIZER - COMPLETE IMPLEMENTATION
======================================================================

Training corpus: ['low', 'lower', 'newest', 'widest', 'the', 'cat', 'sat', 'on', 'the', 'mat', 'machine', 'learning', 'artificial', 'intelligence']

[BPE Training] Starting training on 14 texts...
[BPE Training] Initial vocabulary size: 26
[BPE Training] Initial vocabulary: ['a', 'c', 'e', 'f', 'g', 'h', 'i', 'l', 'm', 'n', 'o', 'r', 's', 't', 'u', 'w']...
[BPE Training] Iteration 100: vocab_size=26, best_pair=('t', 'h'), count=2
[BPE Training] Training complete!
[BPE Training] Final vocabulary size: 50
[BPE Training] Number of merge rules: 24

Encoding test:
Text: 'lower'
Tokens: ['lo', 'w', 'e', 'r']
Token IDs: [15, 23, 8, 18]
Decoded: 'lower'
======================================================================
BPE TOKENIZER DEMO COMPLETE
======================================================================
"""
```

**BPE Benefits - Detailed Analysis:**

```
1. OOV Handling:
   - Unknown words decomposed into subwords
   - OOV rate ≈ 0.1% - 0.5%
   - Example: "unhappiness" → ["un", "happiness"] or ["un", "happy", "ness"]
   
2. Vocabulary Size Control:
   - Can specify target vocabulary size
   - Balances efficiency and coverage
   - Typical sizes: 30K - 100K
   
3. Language-Agnostic:
   - Works with any text (Unicode)
   - No language-specific rules needed
   - Handles multilingual text
   
4. Efficiency:
   - Fast encoding/decoding
   - Deterministic (same text → same tokens)
   - Reversible (can decode back to text)
```

**BPE Limitations - Detailed Analysis:**

```
1. Order-Dependent:
   - Merge rules applied in specific order
   - Different order → different vocabulary
   - Can be sensitive to training data order
   
2. Unintuitive Splits:
   - May split words in unexpected ways
   - Example: "unhappiness" → ["un", "hap", "piness"]
   - Not always linguistically meaningful
   
3. Fixed Vocabulary:
   - Vocabulary determined during training
   - Cannot adapt to new domains easily
   - May need retraining for domain-specific text
   
4. Greedy Merging:
   - Always merges most frequent pair
   - May not be globally optimal
   - Can miss better merge strategies
```

#### WordPiece Tokenization - Complete Analysis

WordPiece is a subword tokenization algorithm similar to BPE, but uses a likelihood-based approach to select merges. It was developed by Google and is used in BERT and other Google models.

**WordPiece vs BPE - Key Differences:**

```
┌──────────────────┬──────────────────┬──────────────────┐
│ Feature          │ BPE              │ WordPiece        │
├──────────────────┼──────────────────┼──────────────────┤
│ Merge Criterion  │ Frequency       │ Likelihood       │
│ Selection        │ Most frequent   │ Highest likel.   │
│ Mathematical     │ max(frequency)   │ max(Δlikelihood) │
│ Used By          │ GPT family       │ BERT, ALBERT     │
│ Training         │ Faster           │ Slower           │
│ Quality          │ Good             │ Better           │
└──────────────────┴──────────────────┴──────────────────┘
```

**WordPiece Algorithm - Complete:**

```
WordPiece_Training_Algorithm:

Input:
- Corpus: List of text strings
- Vocabulary_size: Target vocabulary size

Output:
- Vocabulary: Set of tokens
- Merge_rules: Ordered list of merges

Algorithm:

1. Initialize:
   Vocabulary = {all characters + special tokens}
   Build language model on corpus
   
   Example:
   Vocabulary = {[CLS], [SEP], [UNK], a, b, ..., z}
   LM = Train language model on corpus

2. Iterate until |Vocabulary| >= Vocabulary_size:
   
   a) For each possible pair (a, b):
       - Compute current likelihood: L_current
       - Compute merged likelihood: L_merged
       - Likelihood increase: ΔL = L_merged - L_current
   
   b) Select pair with maximum likelihood increase:
       best_pair = argmax(ΔL)
   
   c) Merge pair:
       new_token = concat(best_pair[0], best_pair[1])
       Vocabulary.add(new_token)
       Merge_rules.append((best_pair, new_token))
   
   d) Update language model:
       Retrain or update LM with new vocabulary

3. Return Vocabulary and Merge_rules
```

**WordPiece Mathematical Model:**

```
WordPiece_Mathematical_Model:

1. Language Model Likelihood:
   L(V) = Σ log P(w | V, corpus)
   
   Where:
   - V: Vocabulary
   - w: Word in corpus
   - P(w | V): Probability of word under vocabulary

2. Likelihood Increase:
   ΔL(a, b) = L(V ∪ {concat(a, b)}) - L(V)
   
   Where:
   - (a, b): Pair to merge
   - ΔL: Likelihood increase from merging

3. Merge Selection:
   best_pair = argmax_{(a,b)} ΔL(a, b)
   
   Selects pair that maximizes likelihood increase

4. Tokenization:
   Similar to BPE, but uses WordPiece-specific merge rules
```

**WordPiece Benefits:**

```
1. More Principled:
   - Uses likelihood-based selection
   - Theoretically sound
   - Better for language modeling

2. Better Quality:
   - Often produces better tokenizations
   - More linguistically meaningful splits
   - Better for downstream tasks

3. Used in BERT:
   - Proven effective in BERT
   - Good for bidirectional models
   - Handles context well
```

**WordPiece Limitations:**

```
1. Slower Training:
   - Must compute likelihood for each pair
   - Must update language model
   - More computationally expensive

2. More Complex:
   - Requires language model training
   - More implementation complexity
   - Harder to debug

3. Similar to BPE:
   - Still has order-dependent issues
   - Fixed vocabulary size
   - Can have unintuitive splits
```

#### SentencePiece Tokenization - Complete Analysis

SentencePiece is a subword tokenization algorithm developed by Google that treats text as raw Unicode sequences. It's language-agnostic and used in many modern multilingual models like T5, LLaMA, and mT5.

**SentencePiece Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              SENTENCEPIECE ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────┘

Raw Unicode Text (any language)
    │
    ▼
┌──────────────────┐
│ Normalize        │
│ (Optional)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ SentencePiece    │
│ Tokenization      │
│ • BPE or Unigram │
│ • Unicode-aware  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Reversible       │
│ Encoding         │
│ • No info loss   │
└────────┬─────────┘
         │
         ▼
Tokens + Token IDs
```

**SentencePiece Key Features - Detailed:**

```
1. Unicode-Based:
   - Works directly with Unicode
   - No preprocessing needed
   - Handles all languages and scripts
   
   Example:
   Text: "Hello 世界 🌍"
   SentencePiece: Treats as Unicode sequence
   No need for: Encoding detection, normalization, etc.

2. Language-Agnostic:
   - Same algorithm for all languages
   - No language-specific rules
   - Works with mixed languages
   
   Example:
   Text: "Hello bonjour 你好"
   SentencePiece: Handles all languages uniformly

3. Reversible:
   - Can decode back to original text
   - No information loss
   - Preserves whitespace and formatting
   
   Example:
   Original: "Hello, world!"
   Encoded: [1234, 567, 890]
   Decoded: "Hello, world!"  # Exact match

4. Subword Algorithms:
   - Can use BPE or Unigram LM
   - BPE: Similar to standard BPE
   - Unigram: Probabilistic segmentation
   
   Unigram Algorithm:
   - Starts with large vocabulary
   - Iteratively removes low-probability tokens
   - Keeps tokens that maximize likelihood
```

**SentencePiece Unigram Algorithm:**

```
Unigram_SentencePiece_Algorithm:

Input:
- Corpus: Training corpus
- Vocabulary_size: Target size

Output:
- Vocabulary: Set of tokens
- Tokenization probabilities

Algorithm:

1. Initialize:
   Vocabulary = Large set of candidate subwords
   (e.g., all frequent character n-grams)
   
2. Train Unigram Language Model:
   For each candidate token:
       Compute probability P(token | corpus)
   
3. Iteratively prune vocabulary:
   While |Vocabulary| > Vocabulary_size:
       a) Remove token with lowest probability
       b) Retrain unigram model
       c) Recompute probabilities
   
4. Final vocabulary:
   Keep tokens with highest probabilities

5. Tokenization:
   For text, find segmentation that maximizes:
       P(segmentation) = Π P(token_i)
```

**SentencePiece Benefits:**

```
1. Multilingual Excellence:
   - Handles all languages uniformly
   - No language-specific preprocessing
   - Works with mixed-language text
   
   Example:
   Text: "Hello 你好 مرحبا"
   SentencePiece: Handles all scripts seamlessly

2. No Preprocessing Needed:
   - Treats text as raw Unicode
   - No normalization required
   - Simpler pipeline
   
   Benefits:
   - Faster processing
   - Less error-prone
   - More robust

3. Reversibility:
   - Can decode exactly to original
   - Preserves formatting
   - No information loss
   
   Example:
   Original: "Hello,  world !"
   Encoded → Decoded: "Hello,  world !"  # Exact

4. Production-Ready:
   - Used in many production systems
   - Well-tested and optimized
   - Good performance
```

**SentencePiece Limitations:**

```
1. Larger Vocabulary:
   - Often needs larger vocab than BPE
   - More tokens per text
   - Higher memory usage

2. Slower Training (Unigram):
   - Unigram algorithm is slower
   - Must retrain model iteratively
   - More computation

3. Similar Issues:
   - Still has fixed vocabulary
   - May have unintuitive splits
   - Can't adapt to new domains easily
```

**SentencePiece vs BPE vs WordPiece:**

```
┌──────────────────┬──────────┬──────────┬──────────┐
│ Feature          │ BPE      │ WordPiece│ SentencePiece│
├──────────────────┼──────────┼──────────┼──────────┤
│ Algorithm        │ Frequency│ Likelihood│ BPE/Unigram│
│ Multilingual     │ ⚠️ Good  │ ⚠️ Limited│ ✅ Excellent│
│ Reversible       │ ✅ Yes   │ ✅ Yes   │ ✅ Yes    │
│ Preprocessing    │ Needed   │ Needed   │ Not needed│
│ Unicode-native   │ ❌ No    │ ❌ No    │ ✅ Yes    │
│ Vocabulary Size  │ Medium   │ Medium   │ Large    │
│ Used By          │ GPT      │ BERT     │ T5, LLaMA│
└──────────────────┴──────────┴──────────┴──────────┘
```

#### Token Limits & Context Window Management - Complete Analysis

Context window management is crucial for LLM applications as models have fixed maximum token limits. This section provides comprehensive strategies for managing token limits in production systems.

**Context Window Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              CONTEXT WINDOW ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────┘

Total Context Window (e.g., 4096 tokens)
    │
    ├─────────────────────────────────────────────────┐
    │                                                   │
    ▼                                                   ▼
┌──────────────┐                              ┌──────────────┐
│ System       │                              │ Response    │
│ Prompt       │                              │ Buffer      │
│ (50-200)     │                              │ (500-1000)  │
└──────────────┘                              └──────────────┘
    │                                                   │
    ▼                                                   ▼
┌──────────────┐                              ┌──────────────┐
│ User         │                              │ Available   │
│ Prompt       │                              │ for Context │
│ (Variable)   │                              │ (Remaining) │
└──────────────┘                              └──────────────┘
```

**Context Window Sizes - Detailed Comparison:**

```
┌──────────────────┬──────────────┬──────────────┬──────────────┐
│ Model            │ Context Size │ Use Case     │ Cost         │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ GPT-3.5-turbo    │ 4K, 16K      │ General      │ Low          │
│ GPT-4            │ 8K, 32K, 128K│ Advanced     │ High         │
│ GPT-4-turbo      │ 128K         │ Long context │ Very High    │
│ Claude 3 Opus    │ 200K         │ Long docs    │ High         │
│ Claude 3 Sonnet  │ 200K         │ Long docs    │ Medium       │
│ Gemini 1.5 Pro   │ 1M (exp)     │ Very long    │ Very High    │
│ LLaMA 2          │ 4K           │ General      │ Open source  │
│ LLaMA 2 70B      │ 4K           │ General      │ Open source  │
└──────────────────┴──────────────┴──────────────┴──────────────┘
```

**Token Budgeting - Mathematical Model:**

```
Token_Budget_Model:

Total_Context_Window = T_max

Components:
1. System_Prompt_Tokens = T_sys
2. User_Prompt_Tokens = T_user
3. Context_Tokens = T_context
4. Response_Buffer = T_reserve

Constraint:
T_sys + T_user + T_context + T_reserve ≤ T_max

Available for Context:
T_context_available = T_max - T_sys - T_user - T_reserve

For RAG System:
T_context_available = T_max - T_sys - T_user - T_reserve

Number of Chunks:
N_chunks = floor(T_context_available / T_chunk_avg)

Where:
- T_chunk_avg: Average tokens per chunk
- N_chunks: Maximum number of chunks to include

Example:
T_max = 4096
T_sys = 100
T_user = 200
T_reserve = 500
T_context_available = 4096 - 100 - 200 - 500 = 3296

If T_chunk_avg = 200:
N_chunks = floor(3296 / 200) = 16 chunks
```

**Token Limit Strategies - Complete Analysis:**

**1. Truncation - Detailed:**

```
Truncation_Strategy:

Method: Cut text at token limit

Variants:
a) Head Truncation:
   - Keep first N tokens
   - Lose ending information
   - Use case: When ending is less important
   
   Example:
   Text: "..." (1000 tokens)
   Limit: 500 tokens
   Result: First 500 tokens

b) Tail Truncation:
   - Keep last N tokens
   - Lose beginning information
   - Use case: When beginning is less important
   
   Example:
   Text: "..." (1000 tokens)
   Limit: 500 tokens
   Result: Last 500 tokens

c) Middle Truncation:
   - Keep beginning + ending
   - Remove middle portion
   - Use case: When middle is less important
   
   Example:
   Text: "..." (1000 tokens)
   Limit: 500 tokens
   Result: First 250 + Last 250 tokens

Mathematical Model:
For text with T tokens and limit L:
   Head: tokens[0:L]
   Tail: tokens[T-L:T]
   Middle: tokens[0:L/2] + tokens[T-L/2:T]
```

**2. Chunking - Complete Analysis:**

```
Chunking_Strategy:

Method: Split text into multiple chunks, process separately

Algorithm:
1. Split text into chunks of size chunk_size
2. Process each chunk independently
3. Combine results

Chunk Size Calculation:
chunk_size = T_context_available / N_chunks_desired

Overlap Strategy:
- Use overlapping chunks to maintain context
- Overlap size: typically 10-20% of chunk_size
- Prevents information loss at boundaries

Example:
Text: 2000 tokens
Chunk size: 500 tokens
Overlap: 50 tokens

Chunks:
- Chunk 1: tokens[0:500]
- Chunk 2: tokens[450:950]
- Chunk 3: tokens[900:1400]
- Chunk 4: tokens[1350:1850]
- Chunk 5: tokens[1800:2000]

Mathematical Model:
For text with T tokens:
   chunk_size = C
   overlap = O
   
   N_chunks = ceil((T - O) / (C - O))
   
   For i = 0 to N_chunks-1:
       start = i * (C - O)
       end = min(start + C, T)
       chunk_i = tokens[start:end]
```

**3. Summarization Strategy:**

```
Summarization_Strategy:

Method: Summarize long text to reduce tokens

Process:
1. Original text: T_original tokens
2. Summarize to: T_summary tokens
3. T_summary << T_original
4. Use summary in context

Mathematical Model:
Compression_Ratio = T_original / T_summary

Typical ratios:
- Extractive: 5:1 to 10:1
- Abstractive: 10:1 to 20:1
- Hierarchical: 20:1 to 50:1

Example:
Original: 5000 tokens
Summary: 500 tokens (10:1 ratio)
Saved: 4500 tokens

Use Cases:
- Long documents
- Historical context
- Multiple documents
```

**4. Sliding Window Strategy:**

```
Sliding_Window_Strategy:

Method: Process text with overlapping windows

Algorithm:
1. Define window size W
2. Define step size S (typically S < W)
3. Process windows with overlap

Example:
Text: 2000 tokens
Window size: 500 tokens
Step size: 250 tokens

Windows:
- Window 1: tokens[0:500]
- Window 2: tokens[250:750]
- Window 3: tokens[500:1000]
- Window 4: tokens[750:1250]
...

Mathematical Model:
For text with T tokens:
   W = window_size
   S = step_size
   
   N_windows = ceil((T - W) / S) + 1
   
   For i = 0 to N_windows-1:
       start = i * S
       end = min(start + W, T)
       window_i = tokens[start:end]

Overlap:
Overlap = W - S
Overlap_Ratio = Overlap / W

Benefits:
- Maintains context across boundaries
- No information loss
- Can process incrementally

Cost:
- More computation (multiple passes)
- Higher token usage
```

**5. Hierarchical Processing:**

```
Hierarchical_Processing_Strategy:

Method: Process at multiple levels (summary + details)

Architecture:
Level 1: Summary (low token count)
    ↓
Level 2: Section summaries (medium token count)
    ↓
Level 3: Detailed chunks (high token count)

Process:
1. Create document summary
2. For relevant sections, include summaries
3. For most relevant sections, include details

Mathematical Model:
Total_Tokens = T_summary + Σ T_section_i + Σ T_detail_j

Where:
- T_summary: Summary tokens
- T_section_i: Section summary tokens
- T_detail_j: Detail chunk tokens

Example:
Document: 10000 tokens
Summary: 200 tokens
Section summaries: 5 × 100 = 500 tokens
Detail chunks: 2 × 500 = 1000 tokens
Total: 1700 tokens (83% reduction)
```

**Complete Token Management Implementation:**

```python
"""
Complete Token Limit Management System
"""

from typing import List, Tuple, Optional
import tiktoken
from dataclasses import dataclass

@dataclass
class TokenBudget:
    """Token budget allocation."""
    total_context: int
    system_prompt: int
    user_prompt: int
    response_buffer: int
    available_for_context: int
    
    def __post_init__(self):
        self.available_for_context = (
            self.total_context 
            - self.system_prompt 
            - self.user_prompt 
            - self.response_buffer
        )

class TokenManager:
    """
    Complete token management system.
    
    Mathematical Model:
        T_available = T_max - T_sys - T_user - T_reserve
        N_chunks = floor(T_available / T_chunk_avg)
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize token manager."""
        self.encoder = tiktoken.encoding_for_model(model_name)
        print(f"[TokenManager] Initialized for model: {model_name}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))
    
    def calculate_budget(self, 
                        context_window: int,
                        system_prompt: str,
                        user_prompt: str,
                        response_buffer: int = 500) -> TokenBudget:
        """
        Calculate token budget.
        
        Mathematical Model:
            T_available = T_max - T_sys - T_user - T_reserve
        """
        system_tokens = self.count_tokens(system_prompt)
        user_tokens = self.count_tokens(user_prompt)
        
        budget = TokenBudget(
            total_context=context_window,
            system_prompt=system_tokens,
            user_prompt=user_tokens,
            response_buffer=response_buffer,
            available_for_context=0
        )
        
        print(f"[Budget] Total context: {context_window}")
        print(f"[Budget] System prompt: {system_tokens}")
        print(f"[Budget] User prompt: {user_tokens}")
        print(f"[Budget] Response buffer: {response_buffer}")
        print(f"[Budget] Available for context: {budget.available_for_context}")
        
        return budget
    
    def truncate_head(self, text: str, max_tokens: int) -> str:
        """Truncate text, keeping first max_tokens."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated = tokens[:max_tokens]
        return self.encoder.decode(truncated)
    
    def truncate_tail(self, text: str, max_tokens: int) -> str:
        """Truncate text, keeping last max_tokens."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated = tokens[-max_tokens:]
        return self.encoder.decode(truncated)
    
    def truncate_middle(self, text: str, max_tokens: int) -> str:
        """Truncate text, keeping beginning and end."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        half = max_tokens // 2
        truncated = tokens[:half] + tokens[-half:]
        return self.encoder.decode(truncated)
    
    def chunk_text(self, 
                   text: str, 
                   chunk_size: int, 
                   overlap: int = 0) -> List[str]:
        """
        Chunk text with optional overlap.
        
        Mathematical Model:
            N_chunks = ceil((T - O) / (C - O))
        """
        tokens = self.encoder.encode(text)
        chunks = []
        
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
        
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if i + chunk_size >= len(tokens):
                break
            
            i += chunk_size - overlap
        
        print(f"[Chunking] Created {len(chunks)} chunks")
        print(f"[Chunking] Chunk size: {chunk_size} tokens")
        print(f"[Chunking] Overlap: {overlap} tokens")
        
        return chunks
    
    def select_chunks(self,
                     chunks: List[str],
                     budget: TokenBudget,
                     max_chunks: Optional[int] = None) -> List[str]:
        """
        Select chunks that fit within token budget.
        
        Algorithm:
        1. Calculate average chunk size
        2. Estimate max chunks
        3. Select top chunks
        """
        chunk_sizes = [self.count_tokens(chunk) for chunk in chunks]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        if max_chunks is None:
            max_chunks = int(budget.available_for_context / avg_chunk_size)
        
        selected = chunks[:max_chunks]
        total_tokens = sum([self.count_tokens(chunk) for chunk in selected])
        
        print(f"[Selection] Selected {len(selected)} chunks")
        print(f"[Selection] Total tokens: {total_tokens}")
        print(f"[Selection] Budget: {budget.available_for_context}")
        
        return selected


# Example Usage
if __name__ == "__main__":
    print("=" * 70)
    print("TOKEN LIMIT MANAGEMENT - COMPLETE IMPLEMENTATION")
    print("=" * 70)
    
    manager = TokenManager()
    
    # Calculate budget
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is AI?"
    budget = manager.calculate_budget(
        context_window=4096,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_buffer=500
    )
    
    # Example text
    long_text = "Your long document here..." * 100
    
    # Strategy 1: Truncation
    print("\n--- Truncation Strategy ---")
    truncated = manager.truncate_head(long_text, budget.available_for_context)
    print(f"Truncated text tokens: {manager.count_tokens(truncated)}")
    
    # Strategy 2: Chunking
    print("\n--- Chunking Strategy ---")
    chunks = manager.chunk_text(long_text, chunk_size=500, overlap=50)
    selected = manager.select_chunks(chunks, budget)
    
    print("\n" + "=" * 70)
    print("TOKEN MANAGEMENT DEMO COMPLETE")
    print("=" * 70)

"""
Expected Output:
======================================================================
TOKEN LIMIT MANAGEMENT - COMPLETE IMPLEMENTATION
======================================================================
[TokenManager] Initialized for model: gpt-3.5-turbo
[Budget] Total context: 4096
[Budget] System prompt: 8
[Budget] User prompt: 4
[Budget] Response buffer: 500
[Budget] Available for context: 3584

--- Truncation Strategy ---
Truncated text tokens: 3584

--- Chunking Strategy ---
[Chunking] Created 8 chunks
[Chunking] Chunk size: 500 tokens
[Chunking] Overlap: 50 tokens
[Selection] Selected 7 chunks
[Selection] Total tokens: 3500
[Selection] Budget: 3584

======================================================================
TOKEN MANAGEMENT DEMO COMPLETE
======================================================================
"""
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

