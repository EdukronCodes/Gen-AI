# 100 LLM Interview Questions and Answers

## Table of Contents
1. [Fundamentals & Architecture](#fundamentals--architecture)
2. [Training & Fine-tuning](#training--fine-tuning)
3. [Tokenization & Embeddings](#tokenization--embeddings)
4. [Attention Mechanisms](#attention-mechanisms)
5. [Prompt Engineering](#prompt-engineering)
6. [Evaluation & Metrics](#evaluation--metrics)
7. [Optimization & Deployment](#optimization--deployment)
8. [Ethics & Safety](#ethics--safety)
9. [Applications & Use Cases](#applications--use-cases)
10. [Advanced Topics](#advanced-topics)

---

## Fundamentals & Architecture

### Q1: What is a Large Language Model (LLM)?
**Answer:** A Large Language Model is a type of artificial intelligence model trained on vast amounts of text data to understand and generate human-like text. LLMs use deep learning architectures (typically transformers) to learn patterns, relationships, and context from text, enabling them to perform tasks like text generation, translation, summarization, and question answering.

### Q2: Explain the transformer architecture.
**Answer:** The transformer architecture, introduced in "Attention Is All You Need" (2017), consists of:
- **Encoder-Decoder structure**: Encoder processes input, decoder generates output
- **Self-attention mechanism**: Allows the model to weigh the importance of different words in a sequence
- **Multi-head attention**: Multiple attention mechanisms run in parallel
- **Positional encoding**: Adds information about word positions
- **Feed-forward networks**: Applied to each position independently
- **Layer normalization and residual connections**: For stable training

### Q3: What is the difference between encoder-only, decoder-only, and encoder-decoder models?
**Answer:**
- **Encoder-only** (e.g., BERT): Bidirectional understanding, good for classification, NER, Q&A
- **Decoder-only** (e.g., GPT): Autoregressive generation, good for text generation, completion
- **Encoder-decoder** (e.g., T5, BART): Both understanding and generation, good for translation, summarization

### Q4: What is autoregressive generation?
**Answer:** Autoregressive generation is a method where the model generates text one token at a time, using previously generated tokens as context. Each new token is predicted based on all previous tokens in the sequence, creating a left-to-right generation process.

### Q5: Explain the concept of context window.
**Answer:** The context window is the maximum number of tokens (words/subwords) that an LLM can process in a single input. It limits how much text the model can consider at once. Modern LLMs have context windows ranging from 2K to 200K+ tokens.

### Q6: What is the difference between parameters and tokens?
**Answer:**
- **Parameters**: The weights and biases learned during training (e.g., GPT-3 has 175B parameters)
- **Tokens**: The units of text the model processes (words or subwords). Training involves processing billions of tokens.

### Q7: What is few-shot learning in LLMs?
**Answer:** Few-shot learning is the ability of LLMs to perform new tasks with just a few examples provided in the prompt, without additional training. The model learns the pattern from the examples and applies it to new inputs.

### Q8: Explain zero-shot, one-shot, and few-shot prompting.
**Answer:**
- **Zero-shot**: No examples provided, model relies on pre-training knowledge
- **One-shot**: Single example provided in the prompt
- **Few-shot**: Multiple examples (typically 2-10) provided to demonstrate the task

### Q9: What is in-context learning?
**Answer:** In-context learning is the ability of LLMs to learn and adapt to new tasks based solely on the examples provided in the prompt, without updating model weights. The model uses its pre-trained knowledge to understand the pattern from examples.

### Q10: What are the main components of a transformer block?
**Answer:**
1. Multi-head self-attention layer
2. Add & Norm (residual connection + layer normalization)
3. Feed-forward network
4. Another Add & Norm layer

---

## Training & Fine-tuning

### Q11: What is pre-training?
**Answer:** Pre-training is the initial phase where an LLM learns general language patterns from a large corpus of unlabeled text. The model learns to predict the next token (for decoder models) or masked tokens (for encoder models), building a broad understanding of language.

### Q12: What is fine-tuning?
**Answer:** Fine-tuning is the process of further training a pre-trained model on a specific task or domain. The model's weights are updated on a smaller, task-specific dataset, adapting the general knowledge to specialized use cases.

### Q13: Explain transfer learning in the context of LLMs.
**Answer:** Transfer learning involves using knowledge gained from pre-training on large datasets and applying it to downstream tasks. LLMs transfer their general language understanding to specific tasks through fine-tuning or prompting, avoiding the need to train from scratch.

### Q14: What is instruction tuning?
**Answer:** Instruction tuning is fine-tuning an LLM on a dataset of instruction-following examples. The model learns to follow human instructions better, improving its ability to understand and execute tasks based on natural language commands.

### Q15: What is Reinforcement Learning from Human Feedback (RLHF)?
**Answer:** RLHF is a training method that uses human feedback to align LLM outputs with human preferences. It involves:
1. Collecting human comparisons of model outputs
2. Training a reward model to predict human preferences
3. Using reinforcement learning (e.g., PPO) to optimize the LLM against the reward model

### Q16: What is supervised fine-tuning (SFT)?
**Answer:** SFT is training a pre-trained model on labeled examples for a specific task. The model learns from input-output pairs, adapting its behavior to the target task while retaining general language knowledge.

### Q17: What is the difference between full fine-tuning and parameter-efficient fine-tuning?
**Answer:**
- **Full fine-tuning**: Updates all model parameters, requires more compute and memory
- **Parameter-efficient fine-tuning** (PEFT): Updates only a small subset of parameters (e.g., LoRA, adapters), reducing compute and memory requirements while maintaining performance

### Q18: Explain LoRA (Low-Rank Adaptation).
**Answer:** LoRA is a parameter-efficient fine-tuning technique that adds trainable low-rank matrices to the model's attention layers. Instead of updating all weights, only these small matrices are trained, dramatically reducing memory and compute requirements.

### Q19: What is prompt tuning?
**Answer:** Prompt tuning involves learning soft prompts (continuous embeddings) that are prepended to inputs. Only these prompt embeddings are trained while the model weights remain frozen, making it very parameter-efficient.

### Q20: What is the difference between training data and inference?
**Answer:**
- **Training**: Process of learning from data, updating model weights, requires backpropagation
- **Inference**: Using the trained model to make predictions on new data, only forward pass, no weight updates

---

## Tokenization & Embeddings

### Q21: What is tokenization?
**Answer:** Tokenization is the process of breaking text into smaller units (tokens) that the model can process. Tokens can be words, subwords, or characters, depending on the tokenizer used.

### Q22: Explain BPE (Byte Pair Encoding).
**Answer:** BPE is a subword tokenization algorithm that:
1. Starts with character-level vocabulary
2. Iteratively merges the most frequent pairs of tokens
3. Creates a vocabulary of subword units that balance between word-level and character-level representations

### Q23: What is the difference between word-level, subword-level, and character-level tokenization?
**Answer:**
- **Word-level**: Each word is a token (large vocabulary, OOV problem)
- **Subword-level**: Words split into subword units (balanced, handles OOV)
- **Character-level**: Each character is a token (small vocabulary, long sequences)

### Q24: What are embeddings?
**Answer:** Embeddings are dense vector representations of tokens that capture semantic meaning. They map discrete tokens to continuous vectors in a high-dimensional space where similar meanings are close together.

### Q25: What is positional encoding?
**Answer:** Positional encoding adds information about token positions in a sequence to the embeddings. Since transformers don't inherently understand order, positional encodings (learned or fixed sinusoidal) are added to help the model understand sequence position.

### Q26: Explain the difference between learned and fixed positional encodings.
**Answer:**
- **Fixed positional encodings**: Pre-defined sinusoidal functions that encode position information
- **Learned positional encodings**: Embeddings learned during training, allowing the model to learn optimal position representations

### Q27: What is the vocabulary size in typical LLMs?
**Answer:** Most modern LLMs use vocabularies ranging from 30K to 100K+ tokens. GPT models typically use ~50K tokens, while BERT uses ~30K tokens.

### Q28: What is the embedding dimension?
**Answer:** The embedding dimension is the size of the vector space where tokens are represented. Common sizes range from 512 to 4096 dimensions, with larger models typically using larger embedding dimensions.

---

## Attention Mechanisms

### Q29: What is self-attention?
**Answer:** Self-attention is a mechanism where each token in a sequence attends to all other tokens (including itself) to compute a weighted representation. It allows the model to capture relationships between all positions in the sequence simultaneously.

### Q30: Explain the attention mechanism formula.
**Answer:** Attention(Q, K, V) = softmax(QK^T / √d_k) V
- Q (Query): What we're looking for
- K (Key): What we're matching against
- V (Value): The actual information we retrieve
- d_k: Dimension scaling factor to prevent large dot products

### Q31: What is multi-head attention?
**Answer:** Multi-head attention runs multiple attention mechanisms in parallel, each with different learned linear transformations. This allows the model to attend to different types of relationships simultaneously (e.g., syntactic, semantic, positional).

### Q32: What is the difference between self-attention and cross-attention?
**Answer:**
- **Self-attention**: Queries, keys, and values come from the same sequence
- **Cross-attention**: Queries come from one sequence, keys and values from another (used in encoder-decoder models)

### Q33: What is scaled dot-product attention?
**Answer:** Scaled dot-product attention is the attention mechanism used in transformers. The dot product of queries and keys is scaled by √d_k to prevent the softmax from having extremely small gradients when the dot products are large.

### Q34: What is the computational complexity of self-attention?
**Answer:** Self-attention has O(n²) complexity where n is the sequence length, because each token attends to all other tokens. This quadratic complexity is a limitation for very long sequences.

### Q35: What are some alternatives to standard attention for long sequences?
**Answer:**
- Sparse attention (only attend to subset of tokens)
- Linear attention (approximate attention with linear complexity)
- Flash Attention (memory-efficient implementation)
- Longformer, BigBird (sparse attention patterns)

---

## Prompt Engineering

### Q36: What is prompt engineering?
**Answer:** Prompt engineering is the practice of designing effective input prompts to get desired outputs from LLMs. It involves carefully crafting instructions, examples, and context to guide the model's behavior.

### Q37: What is chain-of-thought prompting?
**Answer:** Chain-of-thought (CoT) prompting encourages the model to show its reasoning process step-by-step before providing the final answer. This improves performance on complex reasoning tasks.

### Q38: What is few-shot chain-of-thought?
**Answer:** Few-shot CoT combines few-shot learning with chain-of-thought reasoning. Examples in the prompt demonstrate the step-by-step reasoning process, teaching the model to reason similarly.

### Q39: What is zero-shot chain-of-thought?
**Answer:** Zero-shot CoT adds "Let's think step by step" or similar phrases to prompts, encouraging the model to show reasoning without providing examples.

### Q40: What is prompt injection?
**Answer:** Prompt injection is a security vulnerability where malicious input manipulates the model's behavior by overriding the intended prompt instructions, potentially causing the model to ignore safety measures or leak information.

### Q41: What is the difference between system prompts and user prompts?
**Answer:**
- **System prompt**: Sets the model's behavior, role, and constraints (e.g., "You are a helpful assistant")
- **User prompt**: The actual user query or instruction

### Q42: What is temperature in LLM generation?
**Answer:** Temperature controls the randomness of outputs. Lower temperature (0-0.5) produces more deterministic, focused outputs. Higher temperature (0.7-1.5) produces more creative, diverse outputs.

### Q43: What is top-p (nucleus) sampling?
**Answer:** Top-p sampling considers only tokens whose cumulative probability mass exceeds threshold p. It dynamically adjusts the number of tokens considered, providing a balance between diversity and quality.

### Q44: What is top-k sampling?
**Answer:** Top-k sampling restricts sampling to the k most likely tokens at each step, filtering out low-probability tokens to improve output quality.

### Q45: What is beam search?
**Answer:** Beam search maintains multiple candidate sequences during generation, keeping the top-k most promising paths. It's more deterministic than sampling but can be repetitive.

### Q46: What is the difference between greedy decoding and sampling?
**Answer:**
- **Greedy decoding**: Always selects the most likely token (deterministic, can be repetitive)
- **Sampling**: Randomly selects from the probability distribution (more diverse, less predictable)

---

## Evaluation & Metrics

### Q47: What metrics are used to evaluate LLMs?
**Answer:** Common metrics include:
- **Perplexity**: Measures how well the model predicts the next token
- **BLEU**: For translation tasks
- **ROUGE**: For summarization tasks
- **F1 score**: For classification tasks
- **Human evaluation**: Subjective quality assessment

### Q48: What is perplexity?
**Answer:** Perplexity measures how surprised the model is by the test data. Lower perplexity indicates better language modeling. It's the exponentiated average negative log-likelihood.

### Q49: What is the difference between intrinsic and extrinsic evaluation?
**Answer:**
- **Intrinsic evaluation**: Measures how well the model captures language (e.g., perplexity)
- **Extrinsic evaluation**: Measures performance on downstream tasks (e.g., accuracy on classification)

### Q50: What is the HELM benchmark?
**Answer:** HELM (Holistic Evaluation of Language Models) is a comprehensive evaluation framework that tests LLMs across multiple scenarios, metrics, and tasks to provide standardized comparisons.

### Q51: What is the difference between accuracy and perplexity?
**Answer:**
- **Accuracy**: Percentage of correct predictions (for classification tasks)
- **Perplexity**: Measure of prediction uncertainty (for language modeling), lower is better

---

## Optimization & Deployment

### Q52: What is model quantization?
**Answer:** Quantization reduces the precision of model weights (e.g., from 32-bit to 8-bit or 4-bit), reducing memory and compute requirements with minimal accuracy loss.

### Q53: What is model pruning?
**Answer:** Pruning removes less important weights or neurons from the model, creating a sparser, smaller model that requires less memory and computation.

### Q54: What is knowledge distillation?
**Answer:** Knowledge distillation trains a smaller "student" model to mimic a larger "teacher" model, transferring knowledge while reducing model size and inference cost.

### Q55: What is batch inference?
**Answer:** Batch inference processes multiple inputs together, improving GPU utilization and throughput compared to processing inputs one at a time.

### Q56: What is KV caching?
**Answer:** KV (Key-Value) caching stores computed key and value vectors from previous tokens during generation, avoiding recomputation and significantly speeding up autoregressive generation.

### Q57: What is speculative decoding?
**Answer:** Speculative decoding uses a smaller model to draft tokens and a larger model to verify them, potentially speeding up generation while maintaining quality.

### Q58: What is Flash Attention?
**Answer:** Flash Attention is a memory-efficient attention algorithm that computes attention in blocks, reducing memory usage from O(n²) to O(n) and enabling longer sequences.

### Q59: What is the difference between training and inference optimization?
**Answer:**
- **Training optimization**: Focuses on faster convergence, better gradients (e.g., mixed precision, gradient accumulation)
- **Inference optimization**: Focuses on faster generation, lower latency (e.g., quantization, KV caching, batching)

### Q60: What is model parallelism?
**Answer:** Model parallelism splits a large model across multiple GPUs, with each GPU holding a portion of the model. This enables running models too large for a single GPU.

---

## Ethics & Safety

### Q61: What is AI alignment?
**Answer:** AI alignment is ensuring that AI systems pursue intended goals and behave according to human values. For LLMs, this means generating helpful, harmless, and honest outputs.

### Q62: What is hallucination in LLMs?
**Answer:** Hallucination occurs when LLMs generate plausible-sounding but factually incorrect or nonsensical information. It's a major challenge because models don't have a ground truth understanding of facts.

### Q63: What is bias in LLMs?
**Answer:** Bias refers to unfair or prejudiced outputs reflecting stereotypes or discrimination present in training data or model design. This can manifest as gender, racial, cultural, or other forms of bias.

### Q64: What is jailbreaking?
**Answer:** Jailbreaking is attempting to bypass safety measures and content filters in LLMs to make them produce harmful, unethical, or restricted content.

### Q65: What is data poisoning?
**Answer:** Data poisoning involves injecting malicious data into training datasets to manipulate model behavior, potentially causing the model to fail on specific inputs or produce harmful outputs.

### Q66: What is membership inference?
**Answer:** Membership inference attacks attempt to determine whether a specific data point was in the model's training set, raising privacy concerns about training data exposure.

### Q67: What is the difference between safety and alignment?
**Answer:**
- **Safety**: Preventing immediate harms (e.g., generating harmful content)
- **Alignment**: Ensuring long-term beneficial behavior aligned with human values

### Q68: What are red teaming and adversarial testing?
**Answer:** Red teaming involves systematically testing LLMs with adversarial prompts to find vulnerabilities, biases, and safety failures before deployment.

### Q69: What is interpretability in LLMs?
**Answer:** Interpretability is understanding how and why LLMs make decisions. It's challenging due to the complexity of transformer architectures and billions of parameters.

### Q70: What is fairness in AI?
**Answer:** Fairness ensures that AI systems treat all individuals and groups equitably, without discrimination based on protected characteristics like race, gender, or religion.

---

## Applications & Use Cases

### Q71: What are some common applications of LLMs?
**Answer:** 
- Text generation and completion
- Question answering
- Summarization
- Translation
- Code generation
- Chatbots and virtual assistants
- Content creation
- Information extraction

### Q72: What is Retrieval-Augmented Generation (RAG)?
**Answer:** RAG combines LLMs with external knowledge retrieval. The system retrieves relevant documents and includes them in the prompt, allowing the model to answer questions using up-to-date information not in its training data.

### Q73: What is fine-tuning vs. RAG?
**Answer:**
- **Fine-tuning**: Updates model weights for domain-specific knowledge
- **RAG**: Uses external retrieval without changing model weights, better for frequently updating information

### Q74: What is function calling in LLMs?
**Answer:** Function calling (tool use) allows LLMs to interact with external tools and APIs. The model identifies when to call functions and with what parameters, enabling actions beyond text generation.

### Q75: What is code generation with LLMs?
**Answer:** Code generation uses LLMs (like Codex, GitHub Copilot) to generate, complete, or debug code based on natural language descriptions or partial code.

### Q76: What is few-shot learning for code?
**Answer:** Providing code examples in prompts to teach the model coding patterns, style, or specific APIs without fine-tuning.

### Q77: What is the difference between text-to-text and text-to-code models?
**Answer:**
- **Text-to-text**: Generates natural language outputs
- **Text-to-code**: Generates programming code, often trained on code datasets

### Q78: What is conversational AI?
**Answer:** Conversational AI uses LLMs to create chatbots and virtual assistants that can engage in natural, multi-turn conversations with users.

### Q79: What is the difference between open-domain and closed-domain QA?
**Answer:**
- **Open-domain**: Can answer questions about any topic
- **Closed-domain**: Specialized for a specific domain (e.g., medical, legal)

### Q80: What is text summarization?
**Answer:** Text summarization generates concise summaries of longer texts. Can be extractive (selecting important sentences) or abstractive (generating new summary text).

---

## Advanced Topics

### Q81: What is Mixture of Experts (MoE)?
**Answer:** MoE architectures use multiple expert networks, with a routing mechanism selecting which experts process each input. This allows scaling model capacity without proportional compute increase.

### Q82: What is sparse expert models?
**Answer:** Sparse expert models (like Switch Transformers) activate only a subset of experts for each token, reducing computation while maintaining large model capacity.

### Q83: What is continual learning?
**Answer:** Continual learning is the ability to learn new tasks or information without forgetting previously learned knowledge, a challenge for LLMs that typically require full retraining.

### Q84: What is catastrophic forgetting?
**Answer:** Catastrophic forgetting occurs when a model loses previously learned information after training on new data, a major challenge in continual learning.

### Q85: What is in-context learning vs. fine-tuning?
**Answer:**
- **In-context learning**: Adapts behavior through prompts, no weight updates
- **Fine-tuning**: Updates model weights on task-specific data

### Q86: What is the scaling law?
**Answer:** Scaling laws describe how model performance improves predictably with increases in model size, data, and compute. They help predict performance of larger models.

### Q87: What is emergent behavior in LLMs?
**Answer:** Emergent behaviors are capabilities that appear suddenly as models scale, such as reasoning, few-shot learning, or tool use, which weren't explicitly trained.

### Q88: What is the difference between GPT and BERT?
**Answer:**
- **GPT**: Decoder-only, autoregressive, good for generation
- **BERT**: Encoder-only, bidirectional, good for understanding tasks

### Q89: What is the transformer's computational bottleneck?
**Answer:** The quadratic attention complexity (O(n²)) is the main bottleneck, limiting the context window size and making long sequences computationally expensive.

### Q90: What is the difference between causal and bidirectional attention?
**Answer:**
- **Causal attention**: Tokens can only attend to previous tokens (used in decoders)
- **Bidirectional attention**: Tokens can attend to all tokens (used in encoders)

### Q91: What is the role of layer normalization?
**Answer:** Layer normalization stabilizes training by normalizing activations within each layer, reducing internal covariate shift and enabling deeper networks.

### Q92: What is gradient checkpointing?
**Answer:** Gradient checkpointing trades compute for memory by recomputing activations during backpropagation instead of storing them, enabling training of larger models with limited memory.

### Q93: What is mixed precision training?
**Answer:** Mixed precision training uses both 16-bit and 32-bit floating point operations, reducing memory usage and speeding up training on modern GPUs with minimal accuracy loss.

### Q94: What is the difference between pre-training and instruction tuning?
**Answer:**
- **Pre-training**: Learning general language patterns from raw text
- **Instruction tuning**: Learning to follow instructions and be helpful, typically after pre-training

### Q95: What is the Chinchilla scaling law?
**Answer:** Chinchilla scaling suggests that for optimal performance, model size and training data should scale together, with more data being as important as larger models.

### Q96: What is the difference between supervised and unsupervised learning in LLMs?
**Answer:**
- **Unsupervised**: Pre-training on raw text without labels (next token prediction)
- **Supervised**: Fine-tuning on labeled examples (input-output pairs)

### Q97: What is the role of the softmax function in attention?
**Answer:** Softmax converts attention scores into a probability distribution, ensuring that attention weights sum to 1 and represent relative importance of each token.

### Q98: What is the difference between encoder and decoder in transformers?
**Answer:**
- **Encoder**: Processes input sequences bidirectionally, creating rich representations
- **Decoder**: Generates output sequences autoregressively, using encoder outputs and previous tokens

### Q99: What is the maximum sequence length limitation?
**Answer:** Maximum sequence length is limited by:
- Context window size
- Quadratic attention complexity
- Memory constraints
- Training data characteristics

### Q100: What are the future directions for LLM research?
**Answer:** Key directions include:
- Longer context windows
- More efficient architectures
- Better reasoning capabilities
- Improved safety and alignment
- Multimodal capabilities
- Reduced computational requirements
- Better interpretability
- Continual learning

---

## Summary

This collection covers fundamental concepts, architectures, training methods, optimization techniques, applications, and advanced topics in Large Language Models. These questions span from basic understanding to cutting-edge research areas, providing comprehensive coverage for LLM interviews.

