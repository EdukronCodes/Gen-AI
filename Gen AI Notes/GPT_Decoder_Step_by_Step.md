# GPT-Style Decoder: Step-by-Step Calculation

A walkthrough of how a **decoder-only** (GPT-style) transformer computes the next token from an input sequence. Each step is shown with small, explicit numbers so the math is traceable.

---

## Table of Contents

1. [Overview: Decoder-Only Flow](#1-overview-decoder-only-flow)
2. [Step 1: Tokenization](#2-step-1-tokenization)
3. [Step 2: Embedding + Positional Encoding](#3-step-2-embedding--positional-encoding)
4. [Step 3: Weight Matrices (Learned Parameters)](#4-step-3-weight-matrices-learned-parameters)
5. [Step 4: Compute Q, K, V](#5-step-4-compute-q-k-v)
6. [Step 5: Scaled Dot-Product Attention](#6-step-5-scaled-dot-product-attention)
7. [Step 6: Causal Mask (Decoder Rule)](#7-step-6-causal-mask-decoder-rule)
8. [Step 7: Softmax](#8-step-7-softmax)
9. [Step 8: Weighted Sum of V](#9-step-8-weighted-sum-of-v)
10. [Step 9: Add & Layer Normalization](#10-step-9-add--layer-normalization)
11. [Step 10: Feed Forward Network](#11-step-10-feed-forward-network)
12. [Step 11: Final Decoder Output](#12-step-11-final-decoder-output)
13. [Step 12: Output Projection (Vocabulary)](#13-step-12-output-projection-vocabulary)
14. [Complete Decoder Summary](#14-complete-decoder-summary)

---

## 1. Overview: Decoder-Only Flow

In a **decoder-only** model (e.g. GPT):

- Input: a sequence of tokens (e.g. `"i like data science"`).
- Output: a **distribution over the next token** (e.g. predict what comes after `"science"`).

High-level pipeline:

```text
Tokens → Embedding + Position → Q, K, V → Masked Self-Attention
       → Softmax Weights → Weighted Sum of V → Add & Norm
       → FFN → Linear → Softmax → Next Token
```

We walk through each step with a tiny example so you can follow the numbers.

---

## 2. Step 1: Tokenization

**Goal:** Turn raw text into a list of **token IDs** the model can process.

**Example sentence:** `"i like data science"`

**Tokenized sequence:**

| Position | Token   | (Example) Token ID |
|----------|---------|---------------------|
| 1        | i       | 1                   |
| 2        | like    | 2                   |
| 3        | data    | 3                   |
| 4        | science | 4                   |

**Result:**

```text
Token list:  ["i", "like", "data", "science"]
Token IDs:   [1, 2, 3, 4]
```

**Formula (conceptual):**
\[
\text{token\_ids} = [\text{vocab}(w_1), \ldots, \text{vocab}(w_n)], \quad \text{where } w_i \text{ is the } i\text{-th token}
\]

In real models, tokenization uses a subword vocabulary (e.g. BPE); here we use simple integer IDs for clarity.

---

## 3. Step 2: Embedding + Positional Encoding

**Goal:** Turn each token ID into a **vector** and add **position** information so the model knows order.

**Formulas:**

- **Token embedding:** \(\mathbf{e}_i = E[\text{id}_i] \in \mathbb{R}^{d}\) where \(E\) is the embedding matrix and \(d\) is the embedding dimension.
- **Positional encoding:** \(\mathbf{p}_i = \text{PE}(i) \in \mathbb{R}^{d}\) (learned or sinusoidal).
- **Input at position \(i\):**
\[
\mathbf{x}_i = \mathbf{e}_i + \mathbf{p}_i \quad \Rightarrow \quad X = E + \text{PE}
\]
- **Sinusoidal (original Transformer):** for dimension \(2k\) and \(2k+1\),
\[
\text{PE}(i, 2k) = \sin\left(\frac{i}{10000^{2k/d}}\right), \qquad \text{PE}(i, 2k+1) = \cos\left(\frac{i}{10000^{2k/d}}\right)
\]

We use **embedding dimension = 2** so all matrices stay small and easy to follow.

### 3.1 Token Embeddings

Each token ID is mapped to a learned vector:

| Token   | Embedding vector |
|---------|-------------------|
| i       | [1.0, 0.0]        |
| like    | [0.0, 1.0]        |
| data    | [1.0, 1.0]        |
| science | [2.0, 1.0]        |

### 3.2 Positional Embeddings

Each position gets a vector (learned or sinusoidal):

| Position | Positional embedding |
|----------|----------------------|
| 1        | [0.1, 0.1]           |
| 2        | [0.2, 0.2]           |
| 3        | [0.3, 0.3]           |
| 4        | [0.4, 0.4]           |

### 3.3 Final Input Vectors (X)

For each position we **add** token embedding + positional embedding:

```text
x1 = embed(i)      + pos(1)  = [1.0, 0.0] + [0.1, 0.1] = [1.1, 0.1]
x2 = embed(like)   + pos(2)  = [0.0, 1.0] + [0.2, 0.2] = [0.2, 1.2]
x3 = embed(data)   + pos(3)  = [1.0, 1.0] + [0.3, 0.3] = [1.3, 1.3]
x4 = embed(science)+ pos(4)  = [2.0, 1.0] + [0.4, 0.4] = [2.4, 1.4]
```

**Input matrix X (one row per token):**

```text
     dim 0   dim 1
x1 [ 1.1     0.1  ]
x2 [ 0.2     1.2  ]
x3 [ 1.3     1.3  ]
x4 [ 2.4     1.4  ]
```

So **X** has shape **(sequence length × embedding dimension)** = (4 × 2). Shape: \(X \in \mathbb{R}^{n \times d}\) with \(n = 4\), \(d = 2\).

---

## 4. Step 3: Weight Matrices (Learned Parameters)

**Goal:** Define the **Query**, **Key**, and **Value** weight matrices. In real transformers these are learned; here we pick simple values so the math is clear.

**Formulas:**

\[
Q = X W_q, \qquad K = X W_k, \qquad V = X W_v
\]

where \(W_q, W_k, W_v \in \mathbb{R}^{d \times d_k}\) (often \(d_k = d\)). So \(Q, K, V \in \mathbb{R}^{n \times d_k}\).

We use:

- **Wq** (Query):  maps input X to queries
- **Wk** (Key):    maps input X to keys
- **Wv** (Value):  maps input X to values

**Chosen matrices (2×2):**

```text
Wq = [ 1  0 ]     Wk = [ 1  1 ]     Wv = [ 1  0 ]
     [ 0  1 ]          [ 0  1 ]          [ 1  1 ]
```

So:

- **Q = X · Wq**
- **K = X · Wk**
- **V = X · Wv**

Each of Q, K, V has shape **(4 × 2)** (same as X in this setup).

---

## 5. Step 4: Compute Q, K, V

**Goal:** From the input matrix **X**, compute the **Query**, **Key**, and **Value** matrices used in attention.

**Formulas (matrix form):**
\[
Q = X W_q, \quad K = X W_k, \quad V = X W_v
\]
**Row form (per token \(i\)):**
\[
\mathbf{q}_i = \mathbf{x}_i W_q, \quad \mathbf{k}_i = \mathbf{x}_i W_k, \quad \mathbf{v}_i = \mathbf{x}_i W_v
\]

### 5.1 Query: Q = X · Wq

With our choice **Wq = I** (identity), **Q equals X**:

```text
Q = X · Wq = X

Q = [ 1.1  0.1 ]
    [ 0.2  1.2 ]
    [ 1.3  1.3 ]
    [ 2.4  1.4 ]
```

So each row **q_i** is the query vector for the i-th token.

### 5.2 Key: K = X · Wk

Compute each row of **K** as **x_i · Wk**:

```text
k1 = x1 · Wk = [1.1, 0.1] · [1 1]  = [1.1×1 + 0.1×0,  1.1×1 + 0.1×1] = [1.1, 1.2]
                           [0 1]
k2 = x2 · Wk = [0.2, 1.2] · Wk     = [0.2, 1.4]
k3 = x3 · Wk = [1.3, 1.3] · Wk     = [1.3, 2.6]
k4 = x4 · Wk = [2.4, 1.4] · Wk     = [2.4, 3.8]
```

**Key matrix K:**

```text
K = [ 1.1  1.2 ]
    [ 0.2  1.4 ]
    [ 1.3  2.6 ]
    [ 2.4  3.8 ]
```

### 5.3 Value: V = X · Wv

Compute each row of **V** as **x_i · Wv**:

```text
v1 = x1 · Wv = [1.1×1 + 0.1×1,  1.1×0 + 0.1×1] = [1.2, 0.1]
v2 = x2 · Wv = [0.2×1 + 1.2×1,  0.2×0 + 1.2×1] = [1.4, 1.2]
v3 = x3 · Wv = [1.3×1 + 1.3×1,  1.3×0 + 1.3×1] = [2.6, 1.3]
v4 = x4 · Wv = [2.4×1 + 1.4×1,  2.4×0 + 1.4×1] = [3.8, 1.4]
```

**Value matrix V:**

```text
V = [ 1.2  0.1 ]
    [ 1.4  1.2 ]
    [ 2.6  1.3 ]
    [ 3.8  1.4 ]
```

**Summary:** From **X** we have **Q**, **K**, **V**; attention will use them to compute how much each token attends to the others (subject to the causal mask).

---

## 6. Step 5: Scaled Dot-Product Attention

**Goal:** For each token, compute **scores** against all keys (later we will mask and softmax these to get attention weights).

**Formulas:**

- **Attention scores (before scaling):** \(\mathbf{q}_i^\top \mathbf{k}_j\) for query \(i\) and key \(j\).
- **Scaled scores (full matrix):**
\[
\text{Scores} = \frac{Q K^\top}{\sqrt{d_k}} \quad \in \mathbb{R}^{n \times n}
\]
- **Per query (row \(i\) of Scores):**
\[
\text{score}_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}}
\]
The \(\sqrt{d_k}\) scaling keeps the dot products from growing too large when \(d_k\) is large, so softmax does not become too peaked.

We focus on the **last token** ("science", position 4) as the **query** and score it against all keys.

**Query vector for "science":**

```text
q4 = [2.4, 1.4]
```

**Attention scores** = dot product of **q4** with each **k_i**, then scaled by **√d_k** (here d_k = 2, so √2 ≈ 1.414):

```text
q4 · k1 = 2.4×1.1 + 1.4×1.2 = 2.64 + 1.68 = 4.32
q4 · k2 = 2.4×0.2 + 1.4×1.4 = 0.48 + 1.96 = 2.44
q4 · k3 = 2.4×1.3 + 1.4×2.6 = 3.12 + 3.64 = 6.76
q4 · k4 = 2.4×2.4 + 1.4×3.8 = 5.76 + 5.32 = 11.08
```

**Scaled scores** (divide by √2 ≈ 1.414):

```text
4.32 / 1.414 ≈ 3.06
2.44 / 1.414 ≈ 1.73
6.76 / 1.414 ≈ 4.78
11.08 / 1.414 ≈ 7.84
```

**Scaled score vector (for q4):**

```text
[3.06, 1.73, 4.78, 7.84]
```

Scaling by **√d_k** keeps the magnitudes stable so softmax does not become too peaked when dimension grows.

---

## 7. Step 6: Causal Mask (Decoder Rule)

**Goal:** Enforce **autoregressive** behavior: each token may only attend to **past and current** tokens, not future ones.

**Formulas:**

- **Causal mask matrix** \(M \in \mathbb{R}^{n \times n}\):
\[
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \text{ (can attend)} \\ -\infty & \text{if } j > i \text{ (masked)} \end{cases}
\]
- **Masked scores** (before softmax):
\[
\text{Scores}_{\text{masked}} = \frac{Q K^\top}{\sqrt{d_k}} + M
\]
So position \(i\) never attends to positions \(j > i\). After softmax, masked positions get weight 0.

For the **last** token ("science", position 4), it is allowed to see positions 1, 2, 3, and 4. So for this token we do **not** zero out any of the four scores:

```text
Mask for position 4: [✔ ✔ ✔ ✔]  (can see i, like, data, science)
```

So the **scores we use for softmax** are unchanged:

```text
[3.06, 1.73, 4.78, 7.84]
```

For an **earlier** token (e.g. position 2), we would mask out positions 3 and 4 (set those scores to −∞ before softmax). Here we only need the last position’s attention, so the causal mask leaves these four scores as-is.

---

## 8. Step 7: Softmax

**Goal:** Turn the (masked) attention **scores** into **probabilities** that sum to 1.

**Input (scores for q4):** `[3.06, 1.73, 4.78, 7.84]`

**Compute exponentials:**

```text
exp(3.06) ≈ 21.3
exp(1.73) ≈ 5.65
exp(4.78) ≈ 119
exp(7.84) ≈ 2550
```

**Sum:**

```text
21.3 + 5.65 + 119 + 2550 = 2695
```

**Attention weights (α_i = exp(score_i) / sum):**

```text
α1 = 21.3  / 2695 ≈ 0.008
α2 = 5.65  / 2695 ≈ 0.002
α3 = 119   / 2695 ≈ 0.044
α4 = 2550  / 2695 ≈ 0.946
```

**Interpretation:** The last token ("science") attends **strongly to itself** (0.946) and a bit to "data" (0.044); almost no weight on "i" and "like". So the model is focusing on the most relevant (and local) tokens for predicting what comes next.

---

## 9. Step 8: Weighted Sum of V

**Goal:** Use the attention weights to form a **single vector** as the output of the attention layer: a weighted sum of the **value** vectors.

**Formula:**

```text
attention_output = α1·v1 + α2·v2 + α3·v3 + α4·v4
```

**Values (from Step 4):**

```text
v1 = [1.2, 0.1]
v2 = [1.4, 1.2]
v3 = [2.6, 1.3]
v4 = [3.8, 1.4]
```

**Weights:** α1 ≈ 0.008, α2 ≈ 0.002, α3 ≈ 0.044, α4 ≈ 0.946

**First dimension:**

```text
0.008×1.2 + 0.002×1.4 + 0.044×2.6 + 0.946×3.8
= 0.0096 + 0.0028 + 0.1144 + 3.5948 ≈ 3.73
```

**Second dimension:**

```text
0.008×0.1 + 0.002×1.2 + 0.044×1.3 + 0.946×1.4
= 0.0008 + 0.0024 + 0.0572 + 1.3244 ≈ 1.39
```

**Attention output (for the last position):**

```text
attention_output = [3.73, 1.39]
```

This vector is the **output of the masked self-attention** for the token "science".

---

## 10. Step 9: Add & Layer Normalization

**Goal:** Add a **residual** connection (attention output + original input) and then **layer norm** for stability. Here we only show the residual step; we treat the normalized result as unchanged for simplicity.

**Residual (add attention output to the input at that position):**

For the last token, input was **x4 = [2.4, 1.4]** and attention output is **[3.73, 1.39]**:

```text
x4 + attention_output = [2.4, 1.4] + [3.73, 1.39] = [6.13, 2.79]
```

**After (simplified) layer normalization:** we keep the same vector for the next step:

```text
after_add_norm = [6.13, 2.79]
```

In a real model, layer norm would rescale and recenter this vector; the idea is that the residual path helps gradient flow and layer norm stabilizes training.

---

## 11. Step 10: Feed Forward Network (FFN)

**Goal:** Apply a small **two-layer MLP** (linear → activation → linear) to each position’s vector. This adds non-linearity and capacity.

**Typical form:** `FFN(x) = ReLU(x · W1 + b1) · W2 + b2`  
We use **W1 = W2 = I** (identity) and no bias so the effect is easy to follow:

```text
W1 = [ 1  0 ]    W2 = [ 1  0 ]
     [ 0  1 ]         [ 0  1 ]
```

**Input:** `[6.13, 2.79]` (from add & norm).

**Step 1:** `x · W1 = [6.13, 2.79]` (no change with identity).

**Step 2:** `ReLU([6.13, 2.79]) = [6.13, 2.79]` (both positive).

**Step 3:** Multiply by W2 (identity) → unchanged.

**FFN output:**

```text
FFN([6.13, 2.79]) = [6.13, 2.79]
```

So in this toy setup the FFN does not change the vector; in real transformers, W1 and W2 are learned and have larger hidden dimension.

---

## 12. Step 11: Final Decoder Output

**Goal:** The **hidden state** at the last position after attention + add/norm + FFN is the representation we use to predict the next token.

For the last token ("science") we have:

```text
h_science = [6.13, 2.79]
```

This vector is the **decoder output** for that position and is passed to the **output (vocabulary) projection** to get logits over tokens.

---

## 13. Step 12: Output Projection (Vocabulary)

**Goal:** Map the hidden state **h_science** to **logits** (one per vocabulary token), then apply softmax to get a **probability distribution** over the next token.

**Simplified vocabulary (size 3):** `["projects", "field", "is"]`

**Output weight matrix Wo** (hidden_size × vocab_size = 2×3):

```text
Wo = [ 1  0  1 ]
     [ 0  1  1 ]
```

Columns correspond to "projects", "field", "is".

**Logits = h_science · Wo:**

```text
logit(projects) = 6.13×1 + 2.79×0 = 6.13
logit(field)    = 6.13×0 + 2.79×1 = 2.79
logit(is)       = 6.13×1 + 2.79×1 = 8.92
```

**Logits:** `[6.13, 2.79, 8.92]`

**Softmax** would convert these to probabilities; the **argmax** is the third token:

**Predicted next token:** **"is"**

So the full chain (embedding → attention → FFN → output projection) produces a distribution over the vocabulary; we take the most likely token as the model’s prediction.

---

## 14. Complete Decoder Summary

End-to-end flow in one place:

| Step | Name                         | What happens |
|------|------------------------------|--------------|
| 1    | **Tokenization**             | Text → list of token IDs, e.g. [1,2,3,4]. |
| 2    | **Embedding + Position**      | IDs → vectors X; add positional embeddings. |
| 3    | **Weight matrices**          | Define Wq, Wk, Wv (learned in real models). |
| 4    | **Q, K, V**                  | Q = X·Wq, K = X·Wk, V = X·Wv. |
| 5    | **Scaled dot-product**       | Scores = (Q·K^T) / √d_k. |
| 6    | **Causal mask**              | Mask out future positions (set to −∞). |
| 7    | **Softmax**                  | Scores → attention weights (sum to 1). |
| 8    | **Weighted sum of V**        | Attention output = sum of α_i · v_i. |
| 9    | **Add & norm**               | Residual + layer normalization. |
| 10   | **FFN**                      | Linear → ReLU → linear. |
| 11   | **Decoder output**           | Hidden state h at last position. |
| 12   | **Output projection**        | h · Wo → logits → softmax → next token. |

**Pipeline diagram:**

```text
Tokens
  → Embedding + Position  →  X
  → Q = X·Wq,  K = X·Wk,  V = X·Wv
  → Scaled dot-product (Q, K)  →  scores
  → Causal mask
  → Softmax  →  attention weights
  → Weighted sum of V  →  attention output
  → Add & Norm
  → FFN
  → Final hidden state h
  → Linear (h · Wo)  →  logits
  → Softmax  →  next-token distribution  →  next token
```

This is the core of a single **decoder block**. Real GPT-style models stack many such blocks and use larger dimensions and vocabularies, but the steps above are the same in spirit.

---

*End of GPT-Style Decoder Step-by-Step Notes.*
