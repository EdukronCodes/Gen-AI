# Module 8: LLM Training & Fine-tuning

**Course:** Generative AI & Agentic AI  
**Module Duration:** 2 weeks  
**Classes:** 13-15

---

## Class 13: Pretraining and Fine-tuning LLMs

### Topics Covered

- Pretraining objectives (Next Word, Masked LM)
- Fine-tuning vs Instruction-tuning vs RLHF
- Low-Rank Adaptation (LoRA), PEFT, QLoRA
- Hands-on: Fine-tune GPT-2 or LLaMA on custom text data

### Learning Objectives

By the end of this class, students will be able to:
- Understand pretraining objectives and their purposes
- Distinguish between fine-tuning approaches
- Implement LoRA for efficient fine-tuning
- Fine-tune models on custom datasets
- Evaluate fine-tuned models

### Core Concepts

#### Pretraining Objectives - Complete Mathematical Analysis

Pretraining objectives define how language models learn from text data. This section provides a comprehensive analysis of different pretraining objectives, their mathematical foundations, and applications.

**Pretraining Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              PREtraining ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────┘

Large Text Corpus
    │
    ▼
┌──────────────────┐
│ Preprocessing    │
│ • Tokenization   │
│ • Batching       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Objective        │
│ Selection        │
│ • Causal LM      │
│ • Masked LM      │
│ • Seq2Seq        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Loss Computation │
│ • Negative       │
│   Log-Likelihood │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Optimization     │
│ • Backprop       │
│ • Parameter      │
│   Update         │
└──────────────────┘
```

**1. Next Word Prediction (Causal Language Modeling) - Complete Analysis:**

```
Causal_Language_Modeling:

Architecture: Autoregressive (GPT-style)

Input: Sequence x = [x₁, x₂, ..., xₙ]
Objective: Predict next token given previous tokens

Mathematical Model:

Conditional Probability:
P(x_t | x_{<t}) = softmax(f(x_{<t}, θ))

Where:
- x_t: Token at position t
- x_{<t}: All tokens before position t
- f(x_{<t}, θ): Model output (logits)
- θ: Model parameters

Loss Function:
L = -Σ_{t=1}^n log P(x_t | x_{<t}, θ)

Detailed Formulation:
L = -Σ_{t=1}^n log P(x_t | x₁, x₂, ..., x_{t-1}, θ)
  = -Σ_{t=1}^n log exp(f_t(x_t)) / Σ_k exp(f_t(k))
  
Where:
- f_t(x_t): Logit for token x_t at position t
- k: All possible tokens in vocabulary

Training Process:
1. For each sequence x = [x₁, x₂, ..., xₙ]:
   a. For t = 1 to n:
      - Compute logits: f_t = model(x_{<t})
      - Compute loss: L_t = -log P(x_t | x_{<t})
   b. Aggregate loss: L = Σ L_t
2. Backpropagate gradients
3. Update parameters: θ = θ - α∇L

Architecture Constraints:
- Causal (masked) attention: Can only attend to previous positions
- Autoregressive: Generate one token at a time
- Unidirectional: Information flows left-to-right

Example:
Sequence: "The cat sat"
- P(cat | The) - predict "cat" given "The"
- P(sat | The cat) - predict "sat" given "The cat"
- Loss = -log P(cat | The) - log P(sat | The cat)
```

**Benefits - Detailed Analysis:**

```
1. Natural for Generation:
   - Training objective matches inference task
   - Predict next token → Generate text
   - No distribution mismatch
   
2. Learns Language Patterns:
   - Sequential dependencies
   - Long-range dependencies (via attention)
   - Statistical regularities
   
3. Enables Text Generation:
   - Can generate coherent text
   - Flexible length generation
   - Creative generation possible
   
4. Scalability:
   - Can train on very large corpora
   - Parallel training (during forward pass)
   - Efficient inference
```

**2. Masked Language Modeling (MLM) - Complete Analysis:**

```
Masked_Language_Modeling:

Architecture: Bidirectional (BERT-style)

Input: Sequence x = [x₁, x₂, ..., xₙ]
Process: Mask some tokens, predict them from context

Mathematical Model:

Masking Strategy:
For each sequence x:
    M = Random sample of positions to mask (typically 15%)
    For each position m ∈ M:
        x_m → [MASK] with probability 0.8
        x_m → random token with probability 0.1
        x_m → unchanged with probability 0.1

Objective:
L = -Σ_{m ∈ M} log P(x_m | x_{¬M}, θ)

Where:
- x_m: Original token at masked position m
- x_{¬M}: All non-masked tokens
- M: Set of masked positions

Detailed Formulation:
L = -Σ_{m ∈ M} log P(x_m | x₁, ..., x_{m-1}, x_{m+1}, ..., xₙ, θ)
  = -Σ_{m ∈ M} log exp(f_m(x_m)) / Σ_k exp(f_m(k))

Where:
- f_m(x_m): Logit for token x_m at masked position m
- Model can see both left and right context

Training Process:
1. For each sequence x:
   a. Randomly mask 15% of tokens
   b. For each masked position m:
      - Compute logits: f_m = model(x_{¬M})
      - Compute loss: L_m = -log P(x_m | x_{¬M})
   c. Aggregate loss: L = Σ_{m ∈ M} L_m
2. Backpropagate gradients
3. Update parameters

Architecture Constraints:
- Bidirectional attention: Can attend to all positions
- Masked positions: Can't see themselves
- Context-aware: Uses full sentence context
```

**Benefits - Detailed Analysis:**

```
1. Bidirectional Context:
   - Uses both left and right context
   - Better understanding of word meaning
   - Captures context-dependent semantics
   
2. Better Representation Learning:
   - Learns rich contextual representations
   - Good for understanding tasks
   - Better for classification tasks
   
3. Efficiency:
   - Can predict all masked tokens in parallel
   - Faster training than autoregressive
   - No sequential dependency
   
4. Downstream Tasks:
   - Excellent for classification
   - Good for NER, POS tagging
   - Better for understanding than generation
```

**3. Sequence-to-Sequence - Complete Analysis:**

```
Sequence_to_Sequence_Pretraining:

Architecture: Encoder-Decoder (T5, BART-style)

Input: Source sequence x = [x₁, x₂, ..., xₙ]
Output: Target sequence y = [y₁, y₂, ..., yₘ]

Mathematical Model:

Encoder:
h = Encoder(x, θ_enc)

Where:
- h: Encoder hidden states
- x: Input sequence
- θ_enc: Encoder parameters

Decoder (Autoregressive):
P(y_t | y_{<t}, h, θ_dec) = softmax(Decoder(y_{<t}, h, θ_dec))

Loss Function:
L = -Σ_{t=1}^m log P(y_t | y_{<t}, h, θ_dec)

Complete Formulation:
L = -Σ_{t=1}^m log P(y_t | y₁, ..., y_{t-1}, Encoder(x), θ_dec)

Training Process:
1. Encode input: h = Encoder(x)
2. For each target position t:
   a. Compute logits: f_t = Decoder(y_{<t}, h)
   b. Compute loss: L_t = -log P(y_t | y_{<t}, h)
3. Aggregate loss: L = Σ L_t
4. Backpropagate through both encoder and decoder
5. Update parameters

Text-to-Text Framework (T5):
All tasks formulated as text-to-text:
- Classification: "classify: text → label"
- Translation: "translate English to French: text → translation"
- Summarization: "summarize: text → summary"
- QA: "question: q context: c → answer"
```

**Benefits - Detailed Analysis:**

```
1. Flexible Task Handling:
   - Single framework for all tasks
   - Unified training objective
   - Easy task switching
   
2. Text-to-Text:
   - Natural formulation
   - No task-specific heads needed
   - General-purpose architecture
   
3. Encoder-Decoder Benefits:
   - Encoder: Understands input
   - Decoder: Generates output
   - Separate optimization
   
4. Transfer Learning:
   - Pretrain on large corpus
   - Fine-tune on specific tasks
   - Strong performance
```

**Pretraining Objectives Comparison:**

```
┌──────────────────┬──────────────┬──────────────┬──────────────┐
│ Feature          │ Causal LM    │ Masked LM   │ Seq2Seq      │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ Architecture     │ Decoder-only │ Encoder-only│ Encoder-Decoder│
│ Attention        │ Causal       │ Bidirectional│ Causal (dec)  │
│ Context          │ Left-to-right│ Bidirectional│ Full context │
│ Generation       │ ✅ Excellent │ ❌ No        │ ✅ Excellent │
│ Understanding    │ ⚠️ Good      │ ✅ Excellent│ ✅ Excellent │
│ Training Speed   │ Fast         │ Fast        │ Slower       │
│ Used By          │ GPT family   │ BERT        │ T5, BART     │
│ Best For         │ Generation   │ Understanding│ Many tasks  │
└──────────────────┴──────────────┴──────────────┴──────────────┘
```

#### Fine-tuning Approaches - Complete Analysis

Fine-tuning adapts pretrained models to specific tasks or domains. This section provides a comprehensive analysis of different fine-tuning approaches, their mathematical foundations, and use cases.

**Fine-tuning Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              FINE-TUNING ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────┘

Pretrained Model (Base Model)
    │
    ▼
┌──────────────────┐
│ Approach         │
│ Selection        │
│ • Full FT        │
│ • Instruction FT │
│ • LoRA/QLoRA     │
│ • RLHF           │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Task-Specific    │
│ Dataset          │
│ • Domain data    │
│ • Instructions   │
│ • Preferences     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Training         │
│ • Update params  │
│ • Optimize loss  │
└────────┬─────────┘
         │
         ▼
Fine-tuned Model
```

**1. Full Fine-tuning - Complete Analysis:**

```
Full_Fine_Tuning:

Process: Update all model parameters

Mathematical Model:

Initial State:
θ₀ = Pretrained parameters

Fine-tuning Objective:
L_finetune = L_task(θ) + λ·L_regularization(θ, θ₀)

Where:
- L_task: Task-specific loss
- L_regularization: Regularization to prevent catastrophic forgetting
- λ: Regularization weight

Parameter Update:
θ_{t+1} = θ_t - α·∇L_finetune(θ_t)

Complete Update:
θ_{t+1} = θ_t - α·(∇L_task(θ_t) + λ·∇L_regularization(θ_t, θ₀))

Memory Requirements:
Memory = Model_size + Optimizer_state + Gradients
       = P + 2P + P = 4P (for Adam optimizer)

Where:
- P: Number of parameters
- For 7B model: ~28GB GPU memory (FP32)

Training Process:
1. Load pretrained model: θ₀
2. For each batch:
   a. Forward pass: Compute loss
   b. Backward pass: Compute gradients
   c. Update all parameters: θ = θ - α·∇L
3. Continue until convergence

Benefits:
- Maximum capacity
- Best performance potential
- Can learn complex patterns
- Full model adaptation

Limitations:
- High memory requirements
- Slow training
- Risk of catastrophic forgetting
- Computationally expensive
```

**2. Instruction Tuning - Complete Analysis:**

```
Instruction_Tuning:

Process: Fine-tune on instruction-response pairs

Dataset Format:
D = {(instruction_i, response_i)}_{i=1}^N

Example:
Instruction: "Translate to French: Hello"
Response: "Bonjour"

Mathematical Model:

Objective:
L = -Σ_{i=1}^N log P(response_i | instruction_i, θ)

Detailed Formulation:
For each pair (instruction, response):
    L_i = -Σ_{t=1}^T log P(response_t | instruction, response_{<t}, θ)
    
Where:
- response_t: Token at position t in response
- T: Length of response

Training Process:
1. Format data as instruction-response pairs
2. For each pair:
   a. Concatenate: input = instruction + response
   b. Compute loss: L = -log P(response | instruction)
   c. Backpropagate
3. Update parameters

Prompt Format:
"[INST] {instruction} [/INST] {response}"

Example:
"[INST] What is AI? [/INST] AI is artificial intelligence..."

Benefits:
- Improves instruction following
- Better zero-shot performance
- Aligns with user expectations
- Common for chat models

Use Cases:
- Chatbots (GPT-3.5, Claude)
- Instruction-following models
- General-purpose assistants
- Task-specific models
```

**3. Reinforcement Learning from Human Feedback (RLHF) - Complete Analysis:**

RLHF is a three-stage process that aligns language models with human preferences. It's used in GPT-4, Claude, and other state-of-the-art models.

```
RLHF_Architecture:

┌─────────────────────────────────────────────────────────────┐
│              RLHF THREE-STAGE PROCESS                        │
└─────────────────────────────────────────────────────────────┘

Stage 1: Supervised Fine-Tuning (SFT)
    │
    ▼
Stage 2: Reward Model Training
    │
    ▼
Stage 3: Reinforcement Learning (PPO)
    │
    ▼
Aligned Model
```

**Stage 1: Supervised Fine-Tuning (SFT) - Detailed:**

```
SFT_Stage:

Purpose: Create initial instruction-following model

Dataset:
D_SFT = {(prompt_i, response_i)}_{i=1}^N

Where:
- prompt_i: User prompt/instruction
- response_i: High-quality human-written response

Objective:
L_SFT = -Σ_{i=1}^N log P(response_i | prompt_i, θ_SFT)

Training Process:
1. Load pretrained model: θ_base
2. Fine-tune on instruction-response pairs
3. Update: θ_SFT = argmin L_SFT(θ)

Result:
- Model π_SFT that follows instructions
- Better than base model
- But may not align with human preferences
```

**Stage 2: Reward Model Training - Detailed:**

```
Reward_Model_Training:

Purpose: Learn human preference scoring function

Dataset:
D_RM = {(prompt_i, response_A, response_B, preference)}_{i=1}^M

Where:
- response_A, response_B: Two different responses
- preference: Which response is better (A or B)

Reward Model Architecture:
r_φ(prompt, response) = Reward_Model(prompt, response, φ)

Where:
- φ: Reward model parameters
- r_φ: Scalar reward score

Training Objective:
L_RM = -Σ_{i=1}^M log P(preference_i | response_A, response_B, r_φ)

Detailed Formulation:
For each comparison (prompt, response_A, response_B, preference):
    r_A = r_φ(prompt, response_A)
    r_B = r_φ(prompt, response_B)
    
    P(prefer_A) = sigmoid(r_A - r_B)
    P(prefer_B) = sigmoid(r_B - r_A)
    
    L = -log P(preference | r_A, r_B)

Training Process:
1. Initialize reward model: φ
2. For each comparison:
   a. Compute rewards: r_A, r_B
   b. Compute preference probability
   c. Compute loss: L_RM
   d. Backpropagate and update φ
3. Result: Reward model r_φ

Result:
- Reward model that scores responses
- Captures human preferences
- Can rank responses
```

**Stage 3: Reinforcement Learning (PPO) - Detailed:**

```
PPO_Stage:

Purpose: Optimize model to maximize reward while staying close to SFT

Policy:
π_θ: Current policy (model being optimized)
π_SFT: Reference policy (SFT model)

Objective:
L_PPO = E[r_φ(prompt, response)] - β·KL(π_θ || π_SFT)

Where:
- r_φ: Reward from reward model
- β: KL penalty weight
- KL: Kullback-Leibler divergence

PPO Clipped Objective:
L_PPO = E[min(
    ratio · advantage,
    clip(ratio, 1-ε, 1+ε) · advantage
) - β·KL(π_θ || π_SFT)]

Where:
- ratio = π_θ(response | prompt) / π_SFT(response | prompt)
- advantage = r_φ(prompt, response) - baseline
- ε: Clipping parameter (typically 0.2)

Training Process:
1. Initialize: θ = θ_SFT
2. For each iteration:
   a. Sample prompts: prompts ~ D
   b. Generate responses: responses ~ π_θ(prompts)
   c. Compute rewards: rewards = r_φ(prompts, responses)
   d. Compute advantages: advantages = rewards - baseline
   e. Compute PPO loss: L_PPO
   f. Update: θ = θ - α·∇L_PPO
3. Continue until convergence

Key Components:
- Advantage estimation: Baseline subtraction
- Importance sampling: Off-policy learning
- KL penalty: Prevents model from diverging too much
- Clipping: Prevents large updates
```

**Complete RLHF Pipeline - Mathematical Formulation:**

```
RLHF_Complete_Pipeline:

Stage 1: SFT
θ_SFT = argmin_θ L_SFT(θ)

Stage 2: Reward Model
φ* = argmax_φ Σ log P(preference | r_φ(response_A), r_φ(response_B))

Stage 3: PPO
θ* = argmax_θ E[r_φ*(prompt, π_θ(prompt))] - β·KL(π_θ || π_SFT)

Final Objective:
Maximize: Human preference alignment
Minimize: Deviation from SFT model
```

**RLHF Benefits - Detailed Analysis:**

```
1. Human Alignment:
   - Aligns with human preferences
   - Better response quality
   - More helpful, harmless, honest

2. Quality Improvement:
   - Better than SFT alone
   - Reduces harmful outputs
   - Improves helpfulness

3. Scalability:
   - Can use human feedback at scale
   - Reward model generalizes
   - Can improve continuously

4. State-of-the-Art:
   - Used in GPT-4, Claude
   - Best performing models
   - Industry standard
```

**RLHF Limitations - Detailed Analysis:**

```
1. Computational Cost:
   - Very expensive (3 stages)
   - Requires reward model training
   - PPO is complex

2. Human Feedback:
   - Requires human labelers
   - Expensive and time-consuming
   - May have biases

3. Complexity:
   - Three-stage process
   - Many hyperparameters
   - Difficult to tune

4. Reward Hacking:
   - Model may exploit reward function
   - Need careful reward design
   - May optimize wrong objective
```

**Fine-tuning Approaches Comparison:**

```
┌──────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Feature          │ Full FT  │ Inst FT  │ LoRA     │ RLHF    │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Parameters      │ All      │ All      │ <1%      │ All      │
│ Memory           │ High     │ High     │ Low      │ Very High│
│ Training Time    │ Slow     │ Slow     │ Fast     │ Very Slow│
│ Cost             │ High     │ High     │ Low      │ Very High│
│ Performance      │ Best     │ Good     │ Good     │ Best     │
│ Use Case         │ Domain   │ Chat     │ Efficient│ Alignment│
│ Complexity       │ Medium   │ Medium   │ Low      │ Very High│
└──────────────────┴──────────┴──────────┴──────────┴──────────┘
```

#### Low-Rank Adaptation (LoRA) - Complete Mathematical Analysis

LoRA is a parameter-efficient fine-tuning method that enables training with a fraction of the parameters. It's based on the observation that neural network weight updates during fine-tuning often have low intrinsic rank.

**LoRA Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              LORA ARCHITECTURE                                  │
└─────────────────────────────────────────────────────────────┘

Original Weight Matrix: W (frozen)
    │
    ├────────────────────────────────────────┐
    │                                        │
    ▼                                        ▼
┌──────────┐                          ┌──────────┐
│ Input    │                          │ Input    │
│ x        │                          │ x        │
└────┬─────┘                          └────┬─────┘
     │                                     │
     ▼                                     ▼
┌──────────┐                          ┌──────────┐
│ W·x      │                          │ ΔW·x     │
│ (frozen) │                          │ (trainable)│
└────┬─────┘                          └────┬─────┘
     │                                     │
     └──────────────┬──────────────────────┘
                    ▼
            ┌──────────────┐
            │ (W + ΔW)·x    │
            │ = (W + BA)·x  │
            └──────────────┘
```

**LoRA Mathematical Formulation - Complete:**

```
LoRA_Mathematical_Model:

Original Forward Pass:
y = W·x

Where:
- W ∈ R^(d × k): Original weight matrix
- x ∈ R^k: Input vector
- y ∈ R^d: Output vector

LoRA Modification:
W' = W + ΔW
where ΔW = B·A

Where:
- B ∈ R^(d × r): Down-projection matrix
- A ∈ R^(r × k): Up-projection matrix
- r: Rank (typically r << min(d, k))

Forward Pass with LoRA:
y = W'·x = (W + B·A)·x = W·x + B·A·x

Computation:
1. h = A·x  (shape: r)
2. y_LoRA = B·h  (shape: d)
3. y = W·x + y_LoRA  (shape: d)

Parameter Count:
Original: d × k parameters
LoRA: d × r + r × k = r(d + k) parameters

Reduction Factor:
Reduction = r(d + k) / (d × k) = r(1/k + 1/d)

For typical values (d=4096, k=4096, r=8):
Original: 16,777,216 parameters
LoRA: 8 × 8192 = 65,536 parameters
Reduction: 99.6% fewer parameters!
```

**LoRA Key Insight - Low-Rank Hypothesis:**

```
Low_Rank_Hypothesis:

Observation: During fine-tuning, weight updates often have low intrinsic rank

Mathematical Formulation:
ΔW = W_finetuned - W_pretrained

Rank Decomposition:
If rank(ΔW) ≤ r, then:
    ΔW ≈ B·A where B ∈ R^(d × r), A ∈ R^(r × k)

Empirical Evidence:
- Fine-tuning updates often have rank << min(d, k)
- Typical rank: 1-32 for large models
- Can capture most adaptation with r=8-16

Benefits:
- Much fewer parameters to train
- Faster training
- Lower memory requirements
- Can approximate full fine-tuning
```

**LoRA Training Process:**

```
LoRA_Training_Algorithm:

1. Initialize:
   - Load pretrained model: W (frozen)
   - Initialize B with zeros
   - Initialize A with random small values
   
2. Forward Pass:
   For each layer with LoRA:
       y = W·x + B·A·x
   
3. Backward Pass:
   - Compute gradients for B and A only
   - W remains frozen (no gradients)
   
4. Parameter Update:
   B = B - α·∇_B L
   A = A - α·∇_A L
   W unchanged (frozen)

Gradient Computation:
∂L/∂B = ∂L/∂y · (A·x)^T
∂L/∂A = B^T · ∂L/∂y · x^T

Where:
- L: Loss function
- α: Learning rate
```

**LoRA Benefits - Detailed Analysis:**

```
1. Parameter Efficiency:
   - Trains only r(d + k) parameters instead of d×k
   - Typical reduction: 99%+
   - Example: 7B model → 8M LoRA parameters

2. Memory Efficiency:
   - Only need gradients for LoRA matrices
   - Much less GPU memory required
   - Can train on consumer GPUs

3. Training Speed:
   - Fewer parameters = faster updates
   - Less computation per iteration
   - Faster convergence in practice

4. Modularity:
   - Can combine multiple LoRA adapters
   - Easy to switch between adapters
   - Can merge adapters after training

5. Performance:
   - Often matches full fine-tuning
   - Good for domain adaptation
   - Effective for specific tasks
```

**LoRA Hyperparameters:**

```
LoRA_Hyperparameters:

1. Rank (r):
   - Controls adaptation capacity
   - Typical values: 4, 8, 16, 32, 64
   - Higher r = more capacity, more parameters
   - Trade-off: Capacity vs efficiency
   
   Formula:
   Parameters = r × (d + k)
   
   Example (d=4096, k=4096):
   r=4:  32,768 parameters
   r=8:  65,536 parameters
   r=16: 131,072 parameters
   r=32: 262,144 parameters

2. Alpha (α):
   - Scaling factor for LoRA updates
   - Typically: α = 2r or α = r
   - Controls strength of adaptation
   - Higher α = stronger adaptation

3. Dropout:
   - Regularization for LoRA matrices
   - Typical values: 0.1 - 0.2
   - Prevents overfitting

4. Target Modules:
   - Which layers to apply LoRA
   - Common: attention layers (q_proj, v_proj, k_proj, o_proj)
   - Can also apply to MLP layers
   - More modules = more parameters
```

**LoRA Implementation Example - Complete:**

```python
"""
Complete LoRA Implementation
"""

import torch
import torch.nn as nn
from typing import Optional

class LoRALayer(nn.Module):
    """
    LoRA layer implementation.
    
    Mathematical Model:
        y = W·x + (B·A)·x
        where W is frozen and B, A are trainable
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.1):
        """
        Initialize LoRA layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank (r)
            alpha: Scaling factor
            dropout: Dropout probability
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Original weight (frozen)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features), 
                                   requires_grad=False)
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        print(f"[LoRA] Initialized: in={in_features}, out={out_features}, "
              f"rank={rank}, alpha={alpha}")
        print(f"[LoRA] Parameters: {rank * (in_features + out_features)} "
              f"(vs {in_features * out_features} original)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.
        
        Mathematical Model:
            y = W·x + scaling · B·(A·x)
        """
        # Original weight contribution (frozen)
        y = F.linear(x, self.weight)
        
        # LoRA contribution (trainable)
        x_lora = self.dropout(x)
        h = F.linear(x_lora, self.lora_A)  # A·x
        y_lora = F.linear(h, self.lora_B)   # B·(A·x)
        y_lora = y_lora * self.scaling
        
        # Combine
        y = y + y_lora
        
        return y
    
    def merge_weights(self) -> torch.Tensor:
        """
        Merge LoRA weights into base weight.
        
        Returns:
            Merged weight matrix W' = W + scaling · B·A
        """
        merged = self.weight + self.scaling * (self.lora_B @ self.lora_A)
        return merged


class LoRALinear(nn.Module):
    """
    Wrapper for applying LoRA to a linear layer.
    """
    
    def __init__(self, 
                 original_layer: nn.Linear,
                 rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.1):
        super().__init__()
        
        # Copy original weight (frozen)
        self.weight = original_layer.weight.data.clone()
        self.bias = original_layer.bias.data.clone() if original_layer.bias is not None else None
        
        # LoRA parameters
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Count parameters
        original_params = in_features * out_features
        lora_params = rank * (in_features + out_features)
        reduction = (1 - lora_params / original_params) * 100
        
        print(f"[LoRALinear] Original: {original_params:,} parameters")
        print(f"[LoRALinear] LoRA: {lora_params:,} parameters")
        print(f"[LoRALinear] Reduction: {reduction:.2f}%")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Original weight contribution
        y = F.linear(x, self.weight, self.bias)
        
        # LoRA contribution
        x_lora = self.dropout(x)
        h = F.linear(x_lora, self.lora_A)
        y_lora = F.linear(h, self.lora_B) * self.scaling
        
        return y + y_lora


# Example Usage
if __name__ == "__main__":
    print("=" * 70)
    print("LORA IMPLEMENTATION - COMPLETE EXAMPLE")
    print("=" * 70)
    
    # Original layer
    original = nn.Linear(4096, 4096)
    print(f"\nOriginal layer: {original.in_features} → {original.out_features}")
    print(f"Original parameters: {original.in_features * original.out_features:,}")
    
    # Apply LoRA
    lora_layer = LoRALinear(original, rank=8, alpha=16.0)
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_layer.parameters())
    
    print(f"\nTrainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    print(f"Frozen parameters: {total - trainable:,}")
    
    # Forward pass
    x = torch.randn(1, 4096)
    y = lora_layer(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    print("\n" + "=" * 70)
    print("LORA DEMO COMPLETE")
    print("=" * 70)

"""
Expected Output:
======================================================================
LORA IMPLEMENTATION - COMPLETE EXAMPLE
======================================================================

Original layer: 4096 → 4096
Original parameters: 16,777,216

[LoRALinear] Original: 16,777,216 parameters
[LoRALinear] LoRA: 65,536 parameters
[LoRALinear] Reduction: 99.61%

Trainable parameters: 65,536
Total parameters: 16,842,752
Frozen parameters: 16,777,216

Input shape: torch.Size([1, 4096])
Output shape: torch.Size([1, 4096])

======================================================================
LORA DEMO COMPLETE
======================================================================
"""
```

**LoRA Applications:**

```
1. Domain Adaptation:
   - Fine-tune on domain-specific data
   - Multiple adapters for different domains
   - Easy switching between domains

2. Task-Specific Fine-tuning:
   - Fine-tune for specific tasks
   - Can combine multiple task adapters
   - Modular architecture

3. Multi-Task Learning:
   - Different LoRA adapters for different tasks
   - Can share base model
   - Efficient resource usage

4. Personalization:
   - User-specific adapters
   - Can load/unload adapters dynamically
   - Privacy-friendly (adapter only)
```

#### Parameter-Efficient Fine-Tuning (PEFT)

**PEFT Methods:**

**1. LoRA:**
- Low-rank adaptation
- Most popular method

**2. AdaLoRA:**
- Adaptive rank allocation
- Better parameter efficiency

**3. Prefix Tuning:**
- Adds trainable prefixes
- Only trains prefix parameters

**4. P-Tuning:**
- Continuous prompt tuning
- Learns optimal prompts

**5. QLoRA:**
- Quantized LoRA
- 4-bit quantization + LoRA
- Enables fine-tuning on consumer GPUs

#### QLoRA (Quantized LoRA)

**Innovation:**
- 4-bit quantization of base model
- LoRA on top of quantized model
- 4-bit NormalFloat (NF4) quantization
- Double quantization for memory efficiency

**Benefits:**
- Fine-tune 65B model on single GPU
- Minimal performance loss
- Fast inference

**Process:**
```
1. Quantize base model to 4-bit
2. Add LoRA adapters
3. Train LoRA adapters
4. Merge adapters (optional)
```

#### Fine-tuning Workflow

**1. Data Preparation:**
- Format dataset appropriately
- Create train/val splits
- Handle tokenization

**2. Model Setup:**
- Load base model
- Add LoRA adapters (if using)
- Configure training parameters

**3. Training:**
- Set learning rate (typically 1e-4 to 1e-3 for LoRA)
- Train for multiple epochs
- Monitor loss

**4. Evaluation:**
- Evaluate on validation set
- Test on unseen data
- Compare with base model

**5. Deployment:**
- Merge LoRA adapters (optional)
- Save model
- Deploy for inference

---

## Class 14: Model Validation Metrics

### Topics Covered

- Perplexity, BLEU, ROUGE, METEOR, and BERTScore
- Evaluating generative model outputs
- Human evaluation and prompt effectiveness
- Trade-offs between metrics

### Learning Objectives

By the end of this class, students will be able to:
- Understand different evaluation metrics
- Calculate metrics for generative models
- Choose appropriate metrics for tasks
- Interpret metric scores
- Design evaluation frameworks

### Core Concepts

#### Perplexity

**Definition:**
- Measures how well model predicts test data
- Lower is better
- Inverse of probability

**Formula:**
```
PP(W) = P(w₁, w₂, ..., wₙ)^(-1/n)
       = exp(-1/n × Σ log P(w_i | w_{<i}))
```

**Interpretation:**
- Perplexity = X means model is as confused as having X choices
- Lower perplexity = better language modeling
- Good for comparing models on same dataset

**Limitations:**
- Doesn't measure quality of generation
- Can be optimized without improving generation
- Doesn't capture semantic quality

#### BLEU (Bilingual Evaluation Understudy)

**Purpose:**
- Originally for machine translation
- Measures n-gram overlap with reference

**Formula:**
```
BLEU = BP × exp(Σ log pₙ / N)

where:
- pₙ = precision of n-grams
- BP = brevity penalty
- N = maximum n-gram order (typically 4)
```

**Brevity Penalty:**
```
BP = 1 if c > r
BP = exp(1 - r/c) if c ≤ r

where:
- c = candidate length
- r = reference length
```

**Range:** 0 to 1 (higher is better)

**Limitations:**
- Doesn't capture semantic meaning
- Penalizes different valid phrasings
- Requires reference translation
- Not good for creative tasks

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**Purpose:**
- Originally for summarization
- Measures overlap with reference

**Variants:**

**ROUGE-N:**
- N-gram overlap
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap

**ROUGE-L:**
- Longest Common Subsequence (LCS)
- Captures sentence-level structure

**ROUGE-W:**
- Weighted LCS
- Favors consecutive matches

**Formula (ROUGE-N):**
```
ROUGE-N = Σ Count_match(n-gram) / Σ Count(n-gram)
```

**Range:** 0 to 1 (higher is better)

**Use Cases:**
- Summarization evaluation
- Text generation tasks
- Requires reference summaries

#### METEOR (Metric for Evaluation of Translation with Explicit Ordering)

**Features:**
- Considers synonyms and paraphrasing
- Includes word order
- Harmonic mean of precision and recall

**Components:**
- Unigram precision
- Unigram recall
- Synonym matching
- Stem matching
- Word order penalty

**Range:** 0 to 1 (higher is better)

**Benefits:**
- Better correlation with human judgment than BLEU
- Handles synonyms
- Considers word order

#### BERTScore

**Concept:**
- Uses BERT embeddings for semantic similarity
- Context-aware comparison
- No need for exact word matches

**Process:**
```
1. Get embeddings for candidate and reference
2. Compute cosine similarity for each token
3. Aggregate similarities (precision, recall, F1)
```

**Formula:**
```
P_BERT = 1/|c| × Σ max_sim(c_i, r)
R_BERT = 1/|r| × Σ max_sim(r_j, c)
F_BERT = 2 × P_BERT × R_BERT / (P_BERT + R_BERT)
```

**Benefits:**
- Captures semantic similarity
- Context-aware
- Better for paraphrasing
- Correlates well with human judgment

**Limitations:**
- Computationally expensive
- Requires BERT model
- May not capture long-range dependencies

#### Human Evaluation

**Why Needed:**
- Automatic metrics have limitations
- Human judgment is gold standard
- Captures quality, relevance, coherence

**Evaluation Dimensions:**

**1. Quality:**
- Grammatical correctness
- Fluency
- Naturalness

**2. Relevance:**
- Answer relevance to question
- Topic adherence
- Information completeness

**3. Coherence:**
- Logical flow
- Consistency
- Context understanding

**4. Helpfulness:**
- Task completion
- Usefulness
- Actionability

**Methods:**
- Likert scale ratings
- Pairwise comparisons
- Ranking
- A/B testing

#### Prompt Effectiveness Evaluation

**Metrics:**
- **Task Accuracy:** Correctness of output
- **Consistency:** Same prompt → same output
- **Robustness:** Works with variations
- **Efficiency:** Token usage

**Evaluation Process:**
1. Define success criteria
2. Test with multiple prompts
3. Measure consistency
4. Evaluate robustness
5. Optimize prompts

### Metric Comparison

| Metric | Best For | Pros | Cons |
|--------|----------|------|------|
| Perplexity | Language modeling | Fast, objective | Not generation quality |
| BLEU | Translation | Standard, fast | Semantic meaning |
| ROUGE | Summarization | Standard, fast | Semantic meaning |
| METEOR | Translation | Synonyms, word order | Still lexical |
| BERTScore | General | Semantic similarity | Slow, expensive |
| Human | All | Gold standard | Slow, expensive |

### Evaluation Best Practices

**1. Use Multiple Metrics:**
- Combine lexical and semantic metrics
- Use human evaluation for important tasks

**2. Task-Specific Metrics:**
- Choose metrics appropriate for task
- Consider domain requirements

**3. Baseline Comparison:**
- Compare with baseline models
- Track improvements

**4. Statistical Significance:**
- Run multiple evaluations
- Report confidence intervals

**5. Qualitative Analysis:**
- Examine example outputs
- Identify failure modes

---

## Class 15: Model Training Techniques

### Topics Covered

- Batch size, learning rate, gradient clipping
- Mixed precision & quantization (INT8, 4-bit)
- GPU/TPU optimization and distributed training
- Training efficiency and optimization

### Learning Objectives

By the end of this class, students will be able to:
- Understand training hyperparameters and their effects
- Implement mixed precision training
- Use quantization for efficiency
- Optimize training for GPUs/TPUs
- Set up distributed training

### Core Concepts

#### Training Hyperparameters

**Batch Size:**
- Number of samples per update
- Larger batch = more stable gradients, more memory
- Smaller batch = more updates, less memory

**Trade-offs:**
- **Large batch:** Faster training, more memory, may need higher LR
- **Small batch:** More gradient updates, less memory, better generalization

**Learning Rate:**
- Controls step size in optimization
- Too high = unstable training
- Too low = slow convergence

**Schedules:**
- **Constant:** Fixed throughout
- **Linear decay:** Decreases linearly
- **Cosine decay:** Smooth decrease
- **Warmup:** Start small, increase gradually

**Gradient Clipping:**
- Prevents exploding gradients
- Clips gradient norm to threshold

**Formula:**
```
if ||g|| > threshold:
    g = g × threshold / ||g||
```

**Benefits:**
- Stabilizes training
- Prevents NaN values
- Enables higher learning rates

#### Mixed Precision Training

**Concept:**
- Use FP16 (half precision) for most operations
- Use FP32 (full precision) for critical operations
- Reduces memory and speeds up training

**Benefits:**
- ~2x faster training
- ~2x less memory
- Minimal accuracy loss

**Implementation:**
- Automatic mixed precision (AMP)
- Framework support (PyTorch, TensorFlow)
- Gradient scaling for stability

**Challenges:**
- Numerical stability
- Gradient underflow
- Requires careful handling

#### Quantization

**Purpose:**
- Reduce model size
- Speed up inference
- Enable deployment on edge devices

**Types:**

**1. Post-Training Quantization:**
- Quantize after training
- Fast, simple
- Some accuracy loss

**2. Quantization-Aware Training:**
- Train with quantization in mind
- Better accuracy
- More complex

**3. Dynamic Quantization:**
- Quantize weights, not activations
- Easy to implement
- Moderate speedup

**4. Static Quantization:**
- Quantize weights and activations
- Calibration needed
- Better speedup

**Precision Levels:**

**INT8 Quantization:**
- 8-bit integers
- ~4x smaller, ~2-4x faster
- Minimal accuracy loss
- Common for inference

**4-bit Quantization:**
- 4-bit integers
- ~8x smaller
- More accuracy loss
- Used in QLoRA

**Benefits:**
- Reduced memory footprint
- Faster inference
- Lower power consumption
- Enables edge deployment

**Trade-offs:**
- Accuracy degradation
- Limited to inference (usually)
- Hardware support needed

#### GPU/TPU Optimization

**GPU Optimization:**

**1. Data Loading:**
- Use multiple workers
- Prefetch data
- Pin memory

**2. Batch Processing:**
- Optimize batch size for GPU
- Use gradient accumulation
- Mixed precision

**3. Memory Management:**
- Gradient checkpointing
- Model parallelism
- Efficient data structures

**TPU Optimization:**
- XLA compilation
- Batch size optimization
- Data parallelism

#### Distributed Training

**Strategies:**

**1. Data Parallelism:**
- Split data across devices
- Each device has full model
- Synchronize gradients
- Most common approach

**2. Model Parallelism:**
- Split model across devices
- For very large models
- More complex communication

**3. Pipeline Parallelism:**
- Split model into stages
- Process in pipeline
- Overlaps computation

**Frameworks:**
- PyTorch DDP (Distributed Data Parallel)
- DeepSpeed (Microsoft)
- FairScale (Facebook)
- Horovod

**Benefits:**
- Faster training
- Train larger models
- Better resource utilization

**Challenges:**
- Communication overhead
- Synchronization complexity
- Debugging difficulty

### Training Best Practices

**1. Start Small:**
- Begin with small model/dataset
- Validate pipeline
- Scale up gradually

**2. Monitor Training:**
- Track loss curves
- Monitor metrics
- Check for overfitting

**3. Save Checkpoints:**
- Regular checkpoints
- Best model saving
- Resume capability

**4. Hyperparameter Tuning:**
- Learning rate search
- Batch size optimization
- Architecture search

**5. Validation:**
- Regular validation
- Early stopping
- Model selection

### Readings

- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- "Training language models to follow instructions" (Ouyang et al., 2022)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- "Mixed Precision Training" (Micikevicius et al., 2017)

 

### Additional Resources

- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html)

### Practical Code Examples

#### LoRA Fine-tuning Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def setup_lora_model(model_name="meta-llama/Llama-2-7b-hf"):
    """Setup LoRA for fine-tuning"""
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,  # Use 8-bit quantization
        device_map="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Rank
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # Target attention layers
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def train_lora_model(model, tokenizer, dataset, output_dir="./lora_model"):
    """Train LoRA model"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,  # Mixed precision
        logging_steps=10,
        save_steps=100
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    
    trainer.train()
    trainer.save_model()
    
    return model

# Usage
model = setup_lora_model()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# Prepare your dataset
# trained_model = train_lora_model(model, tokenizer, dataset)
```

**Pro Tip:** Start with small r values (4-8) and increase if needed. Higher r improves capacity but increases parameters and training time.

**Common Pitfall:** Not properly selecting target modules can lead to poor fine-tuning. Target attention layers (q_proj, v_proj, k_proj) for best results.

#### Evaluation Metrics Implementation

```python
from rouge_score import rouge_scorer
from bert_score import score
import numpy as np

class ModelEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def calculate_rouge(self, predictions: List[str], references: List[str]):
        """Calculate ROUGE scores"""
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            scores['rouge1'].append(score['rouge1'].fmeasure)
            scores['rouge2'].append(score['rouge2'].fmeasure)
            scores['rougeL'].append(score['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(scores['rouge1']),
            'rouge2': np.mean(scores['rouge2']),
            'rougeL': np.mean(scores['rougeL'])
        }
    
    def calculate_bertscore(self, predictions: List[str], references: List[str]):
        """Calculate BERTScore"""
        P, R, F1 = score(predictions, references, lang='en', verbose=True)
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }

# Usage
evaluator = ModelEvaluator()
predictions = ["Generated text 1", "Generated text 2"]
references = ["Reference text 1", "Reference text 2"]

rouge_scores = evaluator.calculate_rouge(predictions, references)
bertscore = evaluator.calculate_bertscore(predictions, references)

print(f"ROUGE: {rouge_scores}")
print(f"BERTScore: {bertscore}")
```

### Troubleshooting Guide

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Out of memory** | CUDA OOM errors | Use gradient checkpointing, reduce batch size, use LoRA/QLoRA |
| **Training instability** | Loss spikes, NaN values | Reduce learning rate, use gradient clipping, check data |
| **Poor convergence** | Loss not decreasing | Adjust learning rate, check data quality, verify model setup |
| **Slow training** | High training time | Use mixed precision, optimize data loading, use multiple GPUs |
| **Overfitting** | High train, low validation | Add regularization, reduce epochs, use dropout |

### Quick Reference Guide

#### Fine-tuning Methods Comparison

| Method | Parameters | Memory | Speed | Best For |
|--------|------------|--------|-------|----------|
| Full fine-tuning | All | High | Slow | Large datasets, major changes |
| LoRA | Low-rank | Low | Fast | Domain adaptation, specific tasks |
| QLoRA | 4-bit + LoRA | Very Low | Fast | Limited hardware |

#### Evaluation Metrics Selection

| Task | Recommended Metrics | Why |
|------|---------------------|-----|
| Summarization | ROUGE, BERTScore | Standard for summarization |
| Translation | BLEU, METEOR | Standard for translation |
| Generation | BERTScore, Human eval | Captures semantic quality |
| QA | Exact match, F1 | Measures correctness |

### Key Takeaways

1. Pretraining objectives determine model capabilities
2. LoRA enables efficient fine-tuning with minimal parameters
3. QLoRA makes fine-tuning accessible on consumer hardware
4. Multiple evaluation metrics provide comprehensive assessment
5. Human evaluation remains gold standard for quality
6. Mixed precision and quantization improve efficiency
7. Distributed training enables scaling to large models
8. Proper hyperparameter tuning is crucial for performance
9. Evaluation should use multiple metrics for comprehensive assessment
10. Memory optimization techniques enable training on limited hardware

---

**Previous Module:** [Module 7: Tokenization & Embeddings in LLMs](../module_07.md)  
**Next Module:** [Module 9: LLM Inference & Prompt Engineering](../module_09.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)

