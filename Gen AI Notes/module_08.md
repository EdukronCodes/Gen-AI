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

#### Pretraining Objectives

**1. Next Word Prediction (Causal Language Modeling):**
- Predict next token in sequence
- Used in GPT models
- Autoregressive generation
- Unidirectional attention

**Objective:**
```
L = -Σ log P(x_t | x_{<t})
```

**Benefits:**
- Natural for generation tasks
- Learns language patterns
- Enables text generation

**2. Masked Language Modeling (MLM):**
- Predict masked tokens
- Used in BERT
- Bidirectional attention
- Better for understanding

**Objective:**
```
L = -Σ log P(x_m | x_{¬m})
where x_m are masked tokens
```

**Benefits:**
- Uses bidirectional context
- Better representation learning
- Good for classification tasks

**3. Sequence-to-Sequence:**
- Encoder-decoder architecture
- Used in T5, BART
- Text-to-text framework
- Flexible task handling

#### Fine-tuning Approaches

**1. Full Fine-tuning:**
- Update all model parameters
- Requires full model in memory
- Best performance
- Computationally expensive

**2. Instruction Tuning:**
- Fine-tune on instruction-response pairs
- Improves instruction following
- Better zero-shot performance
- Common for chat models

**3. Reinforcement Learning from Human Feedback (RLHF):**
- Three stages:
  1. Supervised fine-tuning (SFT)
  2. Reward model training
  3. RL optimization (PPO)
- Aligns model with human preferences
- Used in GPT-4, Claude

**Comparison:**

| Approach | Parameters Updated | Use Case | Cost |
|----------|-------------------|----------|------|
| Full Fine-tuning | All | Domain adaptation | High |
| Instruction Tuning | All | Instruction following | High |
| LoRA | <1% | Efficient adaptation | Low |
| RLHF | All | Alignment | Very High |

#### Low-Rank Adaptation (LoRA)

**Motivation:**
- Full fine-tuning is expensive
- Most updates are low-rank
- Can approximate with smaller matrices

**Key Idea:**
- Freeze original weights
- Add low-rank adaptation matrices
- Only train adaptation matrices

**Mathematical Formulation:**
```
W' = W + ΔW
where ΔW = BA

W: original weight matrix (d × k)
B: adaptation matrix (d × r), r << min(d, k)
A: adaptation matrix (r × k)
```

**Benefits:**
- Much fewer parameters (r << d, k)
- Faster training
- Lower memory requirements
- Can combine multiple LoRA adapters

**Parameters:**
- **r (rank):** Controls adaptation capacity
- Typical values: 4, 8, 16, 32
- Higher r = more capacity, more parameters

**Example:**
```
Original weight: 4096 × 4096 = 16.7M parameters
LoRA with r=8: 4096 × 8 + 8 × 4096 = 65K parameters
Reduction: 99.6% fewer parameters!
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

