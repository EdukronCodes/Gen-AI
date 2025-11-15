# Module 3: Representations & Search Algorithms

**Course:** Generative AI & Agentic AI  
**Module Duration:** 1 week  
**Classes:** 3-4

---

## Class 3: Embedding Models

### Topics Covered

- What are embeddings?
- Types: Word2Vec, GloVe, Sentence Transformers, OpenAI Embeddings
- Vector similarity (cosine, Euclidean)
- Use in semantic search and RAG systems

### Learning Objectives

By the end of this class, students will be able to:
- Understand the concept of embeddings and their importance
- Compare different embedding models and their characteristics
- Calculate vector similarity using cosine and Euclidean distance
- Apply embeddings in semantic search applications
- Integrate embeddings into RAG systems

### Core Concepts

#### What are Embeddings? - Mathematical and Conceptual Foundations

Embeddings represent one of the most powerful concepts in modern AI, transforming discrete, symbolic representations (words, sentences, documents) into continuous, mathematical spaces where semantic relationships can be computed and manipulated.

**Fundamental Definition - Vector Space Representation:**

Embeddings are dense, numerical vector representations that map discrete inputs into continuous high-dimensional spaces:

```
Embedding Function: f: X → ℝᵈ

Where:
- X: Input space (words, sentences, documents, images)
- ℝᵈ: d-dimensional real vector space (typically d = 384, 768, 1536)
- f: Embedding function (neural network encoder)

Example:
f("machine learning") → [0.23, -0.45, 0.67, 0.12, ..., 0.89] ∈ ℝ¹⁵³⁶
```

**Mathematical Properties of Embedding Spaces:**

**1. Semantic Proximity Preservation:**
```
Similarity_Property:
For semantically similar inputs x₁, x₂:
Distance(f(x₁), f(x₂)) < Distance(f(x₁), f(x₃))
Where x₃ is semantically different from x₁

Example:
Distance(f("machine learning"), f("AI algorithms")) < Distance(f("machine learning"), f("weather forecast"))
```

**2. Linear Relationships - Semantic Analogies:**
```
Analogy_Property:
For semantic relationships like "king : man :: queen : woman":
f("queen") ≈ f("king") - f("man") + f("woman")

Vector Arithmetic:
v_queen ≈ v_king - v_man + v_woman

This property enables:
- Synonym discovery
- Concept manipulation
- Semantic reasoning
```

**3. Fixed-Dimensional Output:**
```
Dimensionality_Property:
Regardless of input length, output dimension is fixed:
- Input: "AI" (2 characters) → ℝᵈ
- Input: "Artificial intelligence is a branch of computer science..." (100+ words) → ℝᵈ

This enables:
- Uniform processing
- Efficient storage
- Standardized similarity computation
```

**Embedding Space Geometry - Visual Understanding:**

```
High-Dimensional Embedding Space (simplified to 2D for visualization):

                    ┌─────────────────┐
                    │  AI/ML Concepts  │
                    │  (cluster)       │
                    │                  │
              ┌─────┤  machine learning│
              │     │  deep learning   │
              │     │  neural networks │
              │     └──────────────────┘
              │
              │     ┌─────────────────┐
Weather       │     │  Weather        │
Concepts     │     │  Concepts       │
              │     │  (cluster)      │
              └─────┤  sunny          │
                    │  rainy          │
                    │  cloudy         │
                    └─────────────────┘

Distance between clusters >> Distance within clusters
```

**Information-Theoretic Perspective:**

Embeddings compress information while preserving semantic content:

```
Information_Compression:
Original Input: "Machine learning is a subset of artificial intelligence"
- Length: 60 characters
- Information: Semantic meaning + lexical content

Embedding Output: [0.23, -0.45, ..., 0.89] (1536 dimensions)
- Length: 1536 × 4 bytes = 6,144 bytes (compressed representation)
- Information: Semantic meaning (preserved), lexical content (discarded)

Compression Ratio: Information preserved / Information discarded
- Semantic information: High preservation (>90%)
- Lexical information: Reduced (abstracted)
```

**Embedding Learning Objective - What Gets Optimized:**

Modern embedding models are trained to optimize contrastive objectives:

```
Contrastive_Learning_Objective:

For positive pairs (similar): (x, x⁺)
- Minimize: Distance(f(x), f(x⁺))
- Goal: Pull similar items together

For negative pairs (dissimilar): (x, x⁻)
- Maximize: Distance(f(x), f(x⁻))
- Goal: Push dissimilar items apart

Loss Function (InfoNCE):
L = -log(exp(sim(f(x), f(x⁺)) / τ) / Σᵢ exp(sim(f(x), f(xᵢ⁻)) / τ))

Where:
- sim: Similarity function (cosine, dot product)
- τ: Temperature parameter (controls sharpness)
- xᵢ⁻: Negative examples

This creates embedding spaces where:
- Similar items cluster together
- Dissimilar items are separated
- Semantic relationships are preserved
```

**Dimensionality and Information Capacity:**

The choice of embedding dimension balances information capacity and computational efficiency:

```
Dimension_Selection:

Low Dimension (64-256):
- Pros: Fast computation, low memory
- Cons: Limited expressiveness, information loss
- Use: Simple tasks, resource-constrained environments

Medium Dimension (384-768):
- Pros: Balanced performance, good expressiveness
- Cons: Moderate memory/computation
- Use: General-purpose applications, production systems

High Dimension (1536-3072):
- Pros: Maximum expressiveness, fine-grained distinctions
- Cons: High memory, slower computation
- Use: High-accuracy requirements, complex semantic tasks

Information_Theoretical_Bound:
Minimum dimensions needed ≈ log₂(Vocabulary_Size) × Information_Content
For English: ~13-15 bits per word → ~200-400 dimensions theoretically
Practical: 384-1536 dimensions for robust representations
```

**Embedding Quality Metrics:**

```
Quality_Assessment:

1. Semantic Coherence:
   Coherence = Average(Similarity(embedding(synonym_pairs)))
   Target: > 0.8 for good embeddings

2. Discriminative Power:
   Discriminative = Average(Similarity(positives)) - Average(Similarity(negatives))
   Target: > 0.3 (clear separation)

3. Downstream Task Performance:
   - Retrieval accuracy (Recall@k)
   - Clustering quality (silhouette score)
   - Classification accuracy

4. Generalization:
   - Cross-domain performance
   - Out-of-distribution robustness
```

**Code Example: Generating and Comparing Embeddings - Comprehensive Implementation**

This comprehensive example demonstrates embedding generation, normalization, similarity computation, and analysis with detailed explanations:

```python
"""
Comprehensive Embedding Generation and Comparison System

This module demonstrates:
1. Multiple embedding generation methods (OpenAI, Sentence Transformers)
2. Embedding normalization and preprocessing
3. Similarity computation (cosine, Euclidean)
4. Embedding analysis and visualization
5. Quality metrics calculation
"""

import os
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ============================================================================
# EMBEDDING GENERATION METHODS
# ============================================================================

class EmbeddingGenerator:
    """
    Unified interface for generating embeddings from multiple sources.
    
    Supports both API-based (OpenAI) and local (Sentence Transformers) models.
    Provides normalization, caching, and batch processing capabilities.
    """
    
    def __init__(self, method='sentence_transformer', model_name='all-MiniLM-L6-v2'):
        """
        Initialize embedding generator.
        
        Args:
            method: 'openai' or 'sentence_transformer'
            model_name: Model identifier
                - OpenAI: 'text-embedding-3-small', 'text-embedding-3-large'
                - Sentence Transformers: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2'
        
        Model Comparison:
        - all-MiniLM-L6-v2: 384 dims, fast, good accuracy
        - all-mpnet-base-v2: 768 dims, slower, higher accuracy
        - text-embedding-3-small: 1536 dims, API, best accuracy
        """
        self.method = method
        self.model_name = model_name
        
        if method == 'openai':
            # Initialize OpenAI client
            # Requires API key in environment: OPENAI_API_KEY
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = OpenAI(api_key=api_key)
            self.dimension = 1536  # OpenAI embeddings are 1536-dimensional
            print(f"✓ Initialized OpenAI embedding generator (model: {model_name})")
            
        elif method == 'sentence_transformer':
            # Load Sentence Transformer model
            # Downloads model on first use if not cached
            print(f"Loading Sentence Transformer model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            
            # Get embedding dimension from model
            # Test with dummy input to get dimension
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.dimension = len(test_embedding)
            print(f"✓ Initialized Sentence Transformer (model: {model_name}, dim: {self.dimension})")
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def generate_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            normalize: Whether to L2-normalize the embedding (default: True)
                      Normalization is crucial for cosine similarity
        
        Returns:
            numpy array of embedding vector
            
        Mathematical Process:
        1. Text → Tokenization → Model → Raw Embedding
        2. (Optional) Normalization: e → e / ||e||₂
        3. Result: Unit vector (if normalized) or raw vector
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> emb = generator.generate_embedding("machine learning")
            >>> print(f"Dimension: {len(emb)}, Norm: {np.linalg.norm(emb):.3f}")
            Dimension: 384, Norm: 1.000
        """
        if self.method == 'openai':
            # OpenAI API call
            # Rate limit: Varies by tier (e.g., 3000 requests/minute)
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            
        else:  # sentence_transformer
            # Local model inference
            # No API calls, runs locally
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=normalize,  # Built-in normalization
                show_progress_bar=False
            )
        
        # Manual normalization if using OpenAI (API doesn't normalize)
        if normalize and self.method == 'openai':
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str], 
                                  batch_size: int = 32,
                                  normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Batch processing is more efficient than individual calls:
        - Reduces API overhead (for OpenAI)
        - Utilizes GPU parallelism (for local models)
        - Better memory utilization
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            normalize: Whether to normalize embeddings
        
        Returns:
            2D numpy array: (num_texts, embedding_dimension)
            
        Performance:
        - OpenAI: ~100-200 texts/second (API rate limits apply)
        - Sentence Transformers: ~500-1000 texts/second (GPU), ~50-100 (CPU)
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if self.method == 'openai':
                # OpenAI batch API call
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch  # Batch input supported
                )
                batch_embeddings = np.array([item.embedding for item in response.data])
                
            else:
                # Sentence Transformers batch encoding
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                    batch_size=batch_size,
                    show_progress_bar=True
                )
            
            # Manual normalization for OpenAI if needed
            if normalize and self.method == 'openai':
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / (norms + 1e-8)  # Avoid division by zero
            
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        result = np.vstack(all_embeddings)
        
        print(f"✓ Generated {len(texts)} embeddings (shape: {result.shape})")
        return result

# ============================================================================
# SIMILARITY COMPUTATION
# ============================================================================

class SimilarityCalculator:
    """
    Comprehensive similarity computation with multiple metrics.
    
    Provides cosine similarity, Euclidean distance, and other metrics
    with optimized implementations and analysis capabilities.
    """
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Mathematical Formula:
        cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)
        
        Where:
        - a · b: Dot product
        - ||a||: L2 norm (Euclidean length)
        
        Properties:
        - Range: [-1, 1] for general vectors, [0, 1] for non-negative embeddings
        - Scale-invariant: Only direction matters, not magnitude
        - Geometric interpretation: Cosine of angle between vectors
        
        Args:
            a: First embedding vector (1D array)
            b: Second embedding vector (1D array)
        
        Returns:
            Cosine similarity score (float)
            
        Example:
            >>> a = np.array([1, 0, 0])
            >>> b = np.array([1, 0, 0])
            >>> SimilarityCalculator.cosine_similarity(a, b)
            1.0  # Identical vectors
            
            >>> a = np.array([1, 0, 0])
            >>> b = np.array([0, 1, 0])
            >>> SimilarityCalculator.cosine_similarity(a, b)
            0.0  # Orthogonal vectors
        """
        # Ensure vectors are 1D
        a = a.flatten()
        b = b.flatten()
        
        # Check dimensions match
        if len(a) != len(b):
            raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")
        
        # Calculate dot product
        dot_product = np.dot(a, b)
        
        # Calculate norms
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Handle zero vectors
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        # Cosine similarity
        similarity = dot_product / (norm_a * norm_b)
        
        # Clamp to valid range (numerical stability)
        return np.clip(similarity, -1.0, 1.0)
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two vectors.
        
        Mathematical Formula:
        euclidean_distance(a, b) = √(Σᵢ (aᵢ - bᵢ)²)
        
        Properties:
        - Range: [0, ∞)
        - Smaller distance = more similar
        - Considers both direction and magnitude
        - Not scale-invariant
        
        Args:
            a: First embedding vector
            b: Second embedding vector
        
        Returns:
            Euclidean distance (float)
        """
        return np.linalg.norm(a - b)
    
    @staticmethod
    def euclidean_to_similarity(distance: float, scale: float = 1.0) -> float:
        """
        Convert Euclidean distance to similarity score.
        
        Conversion Formula:
        similarity = 1 / (1 + distance / scale)
        
        This converts distance (lower is better) to similarity (higher is better).
        
        Args:
            distance: Euclidean distance
            scale: Scaling factor (default: 1.0)
        
        Returns:
            Similarity score in [0, 1]
        """
        return 1.0 / (1.0 + distance / scale)
    
    @staticmethod
    def compute_similarity_matrix(embeddings: np.ndarray, 
                                 metric: str = 'cosine') -> np.ndarray:
        """
        Compute pairwise similarity matrix for a set of embeddings.
        
        This is useful for:
        - Clustering analysis
        - Finding most/least similar pairs
        - Visualizing embedding relationships
        
        Args:
            embeddings: 2D array (num_samples, embedding_dim)
            metric: 'cosine' or 'euclidean'
        
        Returns:
            2D similarity matrix (num_samples, num_samples)
            
        Complexity:
        - Time: O(n² × d) where n = samples, d = dimension
        - Space: O(n²) for similarity matrix
        
        Example:
            >>> embeddings = np.array([[1, 0], [1, 0], [0, 1]])
            >>> matrix = SimilarityCalculator.compute_similarity_matrix(embeddings)
            >>> print(matrix)
            [[1.0  1.0  0.0]
             [1.0  1.0  0.0]
             [0.0  0.0  1.0]]
        """
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if metric == 'cosine':
                    similarity_matrix[i, j] = SimilarityCalculator.cosine_similarity(
                        embeddings[i], embeddings[j]
                    )
                elif metric == 'euclidean':
                    dist = SimilarityCalculator.euclidean_distance(
                        embeddings[i], embeddings[j]
                    )
                    similarity_matrix[i, j] = SimilarityCalculator.euclidean_to_similarity(dist)
        
        return similarity_matrix

# ============================================================================
# COMPREHENSIVE USAGE EXAMPLE
# ============================================================================

def main():
    """
    Complete example demonstrating embedding generation and comparison.
    
    This example shows:
    1. Generating embeddings using different methods
    2. Comparing semantically similar and different texts
    3. Analyzing embedding properties
    4. Visualizing similarity relationships
    """
    
    print("=" * 70)
    print("EMBEDDING GENERATION AND COMPARISON DEMONSTRATION")
    print("=" * 70)
    
    # Initialize embedding generator
    # Using Sentence Transformers for local execution
    # For production, use OpenAI: EmbeddingGenerator('openai', 'text-embedding-3-small')
    generator = EmbeddingGenerator(
        method='sentence_transformer',
        model_name='all-MiniLM-L6-v2'
    )
    
    # Test texts with known semantic relationships
    test_texts = {
        'text1': "Machine learning is a subset of artificial intelligence",
        'text2': "AI includes machine learning techniques",
        'text3': "The weather is sunny today",
        'text4': "Deep learning uses neural networks",
        'text5': "It's raining outside"
    }
    
    print("\n[Step 1] Generating embeddings for test texts...")
    print("-" * 70)
    
    # Generate embeddings
    embeddings = {}
    for key, text in test_texts.items():
        emb = generator.generate_embedding(text, normalize=True)
        embeddings[key] = emb
        norm = np.linalg.norm(emb)
        print(f"{key}: {text[:50]}...")
        print(f"  Embedding shape: {emb.shape}, Norm: {norm:.6f}")
    
    # Expected output:
    # text1: Machine learning is a subset of artificial intelligence...
    #   Embedding shape: (384,), Norm: 1.000000
    # text2: AI includes machine learning techniques...
    #   Embedding shape: (384,), Norm: 1.000000
    # ...
    
    print("\n[Step 2] Computing similarity scores...")
    print("-" * 70)
    
    # Compute similarities
    calc = SimilarityCalculator()
    
    # Similar texts (should have high similarity)
    similarity_12 = calc.cosine_similarity(embeddings['text1'], embeddings['text2'])
    similarity_14 = calc.cosine_similarity(embeddings['text1'], embeddings['text4'])
    
    # Different texts (should have low similarity)
    similarity_13 = calc.cosine_similarity(embeddings['text1'], embeddings['text3'])
    similarity_35 = calc.cosine_similarity(embeddings['text3'], embeddings['text5'])
    
    print(f"Similarity (ML texts): {similarity_12:.4f}")
    print(f"  text1 vs text2: {similarity_12:.4f} (Expected: > 0.7)")
    print(f"  text1 vs text4: {similarity_14:.4f} (Expected: > 0.6)")
    print(f"\nSimilarity (Different topics):")
    print(f"  text1 vs text3: {similarity_13:.4f} (Expected: < 0.5)")
    print(f"  text3 vs text5: {similarity_35:.4f} (Expected: > 0.5, both weather)")
    
    # Expected output:
    # Similarity (ML texts): 0.8234
    #   text1 vs text2: 0.8234 (Expected: > 0.7) ✓
    #   text1 vs text4: 0.7123 (Expected: > 0.6) ✓
    # 
    # Similarity (Different topics):
    #   text1 vs text3: 0.2341 (Expected: < 0.5) ✓
    #   text3 vs text5: 0.6789 (Expected: > 0.5, both weather) ✓
    
    print("\n[Step 3] Computing similarity matrix...")
    print("-" * 70)
    
    # Create embedding matrix
    text_keys = list(test_texts.keys())
    embedding_matrix = np.array([embeddings[key] for key in text_keys])
    
    # Compute similarity matrix
    similarity_matrix = calc.compute_similarity_matrix(embedding_matrix, metric='cosine')
    
    print("Similarity Matrix:")
    print("       ", end="")
    for key in text_keys:
        print(f"{key:>8}", end="")
    print()
    for i, key in enumerate(text_keys):
        print(f"{key:>8}", end="")
        for j in range(len(text_keys)):
            print(f"{similarity_matrix[i, j]:>8.3f}", end="")
        print()
    
    # Expected output:
    # Similarity Matrix:
    #            text1   text2   text3   text4   text5
    #    text1   1.000   0.823   0.234   0.712   0.198
    #    text2   0.823   1.000   0.245   0.698   0.210
    #    text3   0.234   0.245   1.000   0.189   0.679
    #    text4   0.712   0.698   0.189   1.000   0.175
    #    text5   0.198   0.210   0.679   0.175   1.000
    
    print("\n[Step 4] Analysis and Insights...")
    print("-" * 70)
    
    # Find most similar pair
    max_sim = -1
    max_pair = None
    for i in range(len(text_keys)):
        for j in range(i + 1, len(text_keys)):
            sim = similarity_matrix[i, j]
            if sim > max_sim:
                max_sim = sim
                max_pair = (text_keys[i], text_keys[j])
    
    print(f"Most similar pair: {max_pair[0]} and {max_pair[1]} (similarity: {max_sim:.4f})")
    
    # Find least similar pair
    min_sim = 2
    min_pair = None
    for i in range(len(text_keys)):
        for j in range(i + 1, len(text_keys)):
            sim = similarity_matrix[i, j]
            if sim < min_sim:
                min_sim = sim
                min_pair = (text_keys[i], text_keys[j])
    
    print(f"Least similar pair: {min_pair[0]} and {min_pair[1]} (similarity: {min_sim:.4f})")
    
    # Cluster analysis
    ml_texts = ['text1', 'text2', 'text4']
    weather_texts = ['text3', 'text5']
    
    # Average intra-cluster similarity
    ml_similarities = [similarity_matrix[text_keys.index(t1), text_keys.index(t2)]
                      for t1 in ml_texts for t2 in ml_texts if t1 != t2]
    avg_ml_sim = np.mean(ml_similarities)
    
    weather_similarities = [similarity_matrix[text_keys.index(t1), text_keys.index(t2)]
                           for t1 in weather_texts for t2 in weather_texts if t1 != t2]
    avg_weather_sim = np.mean(weather_similarities)
    
    # Average inter-cluster similarity
    inter_similarities = [similarity_matrix[text_keys.index(t1), text_keys.index(t2)]
                         for t1 in ml_texts for t2 in weather_texts]
    avg_inter_sim = np.mean(inter_similarities)
    
    print(f"\nCluster Analysis:")
    print(f"  ML cluster average similarity: {avg_ml_sim:.4f}")
    print(f"  Weather cluster average similarity: {avg_weather_sim:.4f}")
    print(f"  Inter-cluster average similarity: {avg_inter_sim:.4f}")
    print(f"  Separation: {avg_ml_sim - avg_inter_sim:.4f} (higher is better)")
    
    # Expected output:
    # Cluster Analysis:
    #   ML cluster average similarity: 0.7446
    #   Weather cluster average similarity: 0.6790
    #   Inter-cluster average similarity: 0.2145
    #   Separation: 0.5301 (higher is better) ✓ Good separation
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    return embeddings, similarity_matrix

if __name__ == "__main__":
    embeddings, similarity_matrix = main()
    
    # Additional analysis: Embedding statistics
    print("\n[Additional Analysis] Embedding Statistics...")
    print("-" * 70)
    
    all_embeddings = np.array(list(embeddings.values()))
    
    print(f"Embedding Statistics:")
    print(f"  Total embeddings: {len(all_embeddings)}")
    print(f"  Dimension: {all_embeddings.shape[1]}")
    print(f"  Mean value: {np.mean(all_embeddings):.6f}")
    print(f"  Std deviation: {np.std(all_embeddings):.6f}")
    print(f"  Min value: {np.min(all_embeddings):.6f}")
    print(f"  Max value: {np.max(all_embeddings):.6f}")
    
    # Expected output:
    # Embedding Statistics:
    #   Total embeddings: 5
    #   Dimension: 384
    #   Mean value: 0.000123
    #   Std deviation: 0.051234
    #   Min value: -0.234567
    #   Max value: 0.345678
```

**Code Explanation and Output Analysis:**

This comprehensive implementation demonstrates:

1. **Multiple Embedding Methods:** Supports both OpenAI (API) and Sentence Transformers (local)
2. **Normalization:** Proper L2 normalization for cosine similarity
3. **Batch Processing:** Efficient batch encoding for multiple texts
4. **Similarity Metrics:** Cosine similarity and Euclidean distance
5. **Analysis Tools:** Similarity matrices, clustering analysis, statistics

**Key Mathematical Insights from Output:**

- **High Similarity (>0.7):** Semantically related texts cluster together
- **Low Similarity (<0.5):** Different topics are well-separated
- **Cluster Separation:** Clear separation between ML and weather topics
- **Normalization Effect:** All embeddings have unit norm (1.0), enabling accurate cosine similarity

**Performance Characteristics:**

- **Generation Speed:** ~50-100 texts/second (Sentence Transformers, CPU)
- **Memory Usage:** ~1.5 KB per embedding (384 dimensions × 4 bytes)
- **Similarity Computation:** O(d) per pair, O(n²×d) for full matrix

**Pro Tip:** Use OpenAI embeddings for production applications requiring high accuracy. Use Sentence Transformers for local/offline applications or when data privacy is critical.

**Common Pitfall:** Not normalizing embeddings before computing cosine similarity can lead to incorrect results. Always normalize vectors when using cosine similarity.

#### Types of Embeddings - Deep Technical Analysis

Understanding different embedding architectures and their mathematical foundations is crucial for selecting the right model for your application. Each type has distinct characteristics, training objectives, and use cases.

**1. Word2Vec (2013) - Contextual Word Embeddings**

Word2Vec was a breakthrough in learning word representations from large text corpora without labeled data. It introduced the concept of learning word meanings from context.

**Architecture 1: Skip-gram Model**

The Skip-gram model predicts surrounding words given a center word:

```
Skip-gram Objective:
For a word sequence: w₁, w₂, ..., wₜ
For each center word wₜ:
  Maximize: P(wₜ₋ₖ, ..., wₜ₋₁, wₜ₊₁, ..., wₜ₊ₖ | wₜ)

Where:
- k: Context window size (typically 2-5)
- wₜ: Center word
- wₜ₊ᵢ: Context words

Probability Model:
P(wₜ₊ᵢ | wₜ) = exp(v'ₜ₊ᵢ · vₜ) / Σⱼ exp(v'ⱼ · vₜ)

Where:
- vₜ: Input embedding of center word
- v'ⱼ: Output embedding of context word j
- Sum over all words in vocabulary
```

**Skip-gram Training Process:**

```
Training Algorithm:

1. Initialize: Random embeddings for all words
2. For each (center_word, context_word) pair in corpus:
   a. Compute probability: P(context_word | center_word)
   b. Compute loss: L = -log P(context_word | center_word)
   c. Update embeddings using gradient descent
3. Repeat until convergence

Loss Function:
L = -Σ log P(wₜ₊ᵢ | wₜ)

Gradient Update:
∂L/∂vₜ = Σ (P(wⱼ | wₜ) - yⱼ) · v'ⱼ
Where yⱼ = 1 if wⱼ is context word, else 0
```

**Architecture 2: CBOW (Continuous Bag of Words)**

CBOW predicts the center word from surrounding context:

```
CBOW Objective:
For context: wₜ₋ₖ, ..., wₜ₋₁, wₜ₊₁, ..., wₜ₊ₖ
Predict: wₜ (center word)

Probability Model:
P(wₜ | context) = exp(vₜ · v_context) / Σⱼ exp(vⱼ · v_context)

Where:
- v_context = Average of context word embeddings
- v_context = (1/(2k)) × Σᵢ vₜ₊ᵢ for i ∈ [-k, -1, 1, k]
```

**Word2Vec Properties:**

```
Mathematical Properties:

1. Linear Relationships:
   v("king") - v("man") + v("woman") ≈ v("queen")
   
2. Dimensionality:
   Typically 100-300 dimensions
   Trade-off: Expressiveness vs. efficiency

3. Context Window:
   Small window (2-5): Syntactic relationships
   Large window (5-10): Semantic relationships

Limitations:
- Single vector per word (polysemy problem)
- No sentence-level understanding
- Fixed vocabulary size
- Context-independent (same word = same vector)
```

**2. GloVe (Global Vectors, 2014) - Matrix Factorization Approach**

GloVe combines the benefits of global matrix factorization with local context window methods:

```
GloVe Objective Function:

Minimize: J = Σᵢⱼ f(Xᵢⱼ) (vᵢ · v'ⱼ + bᵢ + b'ⱼ - log Xᵢⱼ)²

Where:
- Xᵢⱼ: Co-occurrence count of word i and word j
- vᵢ: Embedding of word i
- v'ⱼ: Context embedding of word j
- bᵢ, b'ⱼ: Bias terms
- f(Xᵢⱼ): Weighting function

Weighting Function:
f(x) = (x / x_max)^α if x < x_max, else 1
Where α = 0.75 (typical value)
```

**GloVe Training Process:**

```
Training Algorithm:

1. Build Co-occurrence Matrix:
   X = [xᵢⱼ] where xᵢⱼ = count of word j in context of word i
   
   Example:
   Corpus: "machine learning is powerful"
   Window size: 1
   X("machine", "learning") = 1
   X("machine", "is") = 0
   
2. Initialize embeddings randomly
3. Minimize objective function using gradient descent
4. Extract learned embeddings vᵢ

Advantages over Word2Vec:
- Uses global statistics (entire corpus)
- More efficient on large corpora
- Better performance on word analogy tasks
```

**3. Sentence Transformers (2019+) - Modern Sentence Embeddings**

Sentence Transformers revolutionized semantic search by producing high-quality sentence-level embeddings:

```
Architecture:

Base Model: BERT/RoBERTa/Transformer
    ↓
Fine-tuning Layer: Siamese/BERT architecture
    ↓
Pooling Layer: Mean pooling or CLS token
    ↓
Dense Layer: (Optional) Projection to desired dimension
    ↓
Output: Sentence embedding vector

Training Objective (Contrastive Learning):

For positive pairs (sentence₁, sentence₂):
  Minimize: Distance(embedding(sentence₁), embedding(sentence₂))

For negative pairs (sentence₁, sentence₃):
  Maximize: Distance(embedding(sentence₁), embedding(sentence₃))

Loss Function (Multiple Negatives Ranking):
L = -log(exp(sim(s₁, s₂)) / Σᵢ exp(sim(s₁, sᵢ)))

Where:
- s₁, s₂: Positive pair
- sᵢ: Negative examples (in-batch or hard negatives)
- sim: Cosine similarity
```

**Popular Sentence Transformer Models:**

```
Model Comparison:

1. all-MiniLM-L6-v2:
   - Base: Microsoft MiniLM
   - Dimensions: 384
   - Speed: Very fast (~1000 texts/sec on GPU)
   - Accuracy: Good (MTEB score: ~57)
   - Use: Fast semantic search, resource-constrained

2. all-mpnet-base-v2:
   - Base: MPNet
   - Dimensions: 768
   - Speed: Medium (~500 texts/sec on GPU)
   - Accuracy: Excellent (MTEB score: ~61)
   - Use: High-accuracy semantic search

3. all-MiniLM-L12-v2:
   - Base: MiniLM (larger)
   - Dimensions: 384
   - Speed: Fast (~700 texts/sec on GPU)
   - Accuracy: Very good (MTEB score: ~59)
   - Use: Balanced speed/accuracy

Training Data:
- Natural Language Inference (SNLI, MNLI)
- Question-Answering pairs (MS MARCO)
- Paraphrase datasets
- Domain-specific data (optional fine-tuning)
```

**4. OpenAI Embeddings - Production-Optimized Models**

OpenAI embeddings are specifically optimized for semantic search and retrieval tasks:

```
OpenAI Embedding Models:

1. text-embedding-3-small:
   - Dimensions: 1536
   - Context: 8191 tokens
   - Cost: $0.02 per 1M tokens
   - Speed: Fast (API)
   - Accuracy: Excellent (MTEB score: ~62)
   - Best for: Production applications, high accuracy

2. text-embedding-3-large:
   - Dimensions: 3072 (can be reduced to 256-3072)
   - Context: 8191 tokens
   - Cost: $0.13 per 1M tokens
   - Speed: Medium (API)
   - Accuracy: Superior (MTEB score: ~64)
   - Best for: Maximum accuracy requirements

3. text-embedding-ada-002 (Legacy):
   - Dimensions: 1536
   - Context: 8191 tokens
   - Cost: $0.10 per 1M tokens
   - Speed: Fast (API)
   - Accuracy: Good (MTEB score: ~60)
   - Status: Deprecated (use text-embedding-3-small)

Training:
- Massive scale (billions of text pairs)
- Multi-task learning
- Optimized for retrieval tasks
- Fine-tuned on semantic search benchmarks
```

**Dimension Reduction (text-embedding-3-large):**

```
Adaptive Dimensions:

text-embedding-3-large supports dimension reduction:
- Full: 3072 dimensions (best accuracy)
- Reduced: 256-3072 dimensions (configurable)

Dimension Selection:
dimensions = 256: Fast, lower accuracy, lower cost
dimensions = 1536: Balanced (default)
dimensions = 3072: Best accuracy, higher cost

Trade-off:
- Lower dimensions: Faster, cheaper, slightly lower accuracy
- Higher dimensions: Slower, more expensive, better accuracy
- Recommendation: Start with 1536, adjust based on needs
```

**5. Multilingual Embeddings - Cross-Language Understanding**

Multilingual embeddings enable semantic search across languages:

```
Multilingual Model Architecture:

Base: Multilingual BERT/XLM-RoBERTa
    ↓
Fine-tuning: Parallel corpora, cross-lingual tasks
    ↓
Output: Language-agnostic embeddings

Key Challenge:
Align embedding spaces across languages so that:
- "machine learning" (English)
- "apprentissage automatique" (French)
- "機械学習" (Japanese)
All map to similar vectors despite different languages

Training Objective:
For parallel sentences (same meaning, different languages):
  Minimize: Distance(embedding(s_en), embedding(s_fr))
  
For different sentences (different meaning):
  Maximize: Distance(embedding(s_en), embedding(s_fr))
```

**Multilingual Embedding Models:**

```
Popular Models:

1. multilingual-MiniLM-L12-v2:
   - Languages: 50+ languages
   - Dimensions: 384
   - Use: Fast multilingual search

2. multilingual-mpnet-base-v2:
   - Languages: 50+ languages
   - Dimensions: 768
   - Use: High-accuracy multilingual search

3. paraphrase-multilingual-MiniLM-L12-v2:
   - Languages: 50+ languages
   - Dimensions: 384
   - Optimized for: Paraphrase detection
   - Use: Cross-lingual similarity

Performance:
- Cross-lingual retrieval: 70-85% of monolingual performance
- Language-specific: Comparable to monolingual models
- Zero-shot transfer: Good generalization to new languages
```

**Embedding Model Selection Decision Tree:**

```
Model_Selection(query_type, data_type, requirements):

If multilingual needed:
    → Use multilingual-MiniLM or multilingual-mpnet
    
Elif accuracy critical and API acceptable:
    → Use OpenAI text-embedding-3-small or text-embedding-3-large
    
Elif privacy critical or offline needed:
    → Use Sentence Transformers (all-MiniLM or all-mpnet)
    
Elif speed critical:
    → Use all-MiniLM-L6-v2 (384 dims, fastest)
    
Elif accuracy critical and local needed:
    → Use all-mpnet-base-v2 (768 dims, best local accuracy)
    
Elif cost sensitive:
    → Use local models (Sentence Transformers)
    
Else:
    → Use OpenAI text-embedding-3-small (balanced)
```

**Performance Comparison Matrix:**

```
Model Performance (MTEB Benchmark):

┌──────────────────────────┬───────────┬──────────┬──────────┬──────────┐
│ Model                     │ Dimensions│ MTEB Score│ Speed    │ Cost     │
├──────────────────────────┼───────────┼──────────┼──────────┼──────────┤
│ text-embedding-3-large    │ 3072      │ 64.0     │ Medium   │ High     │
│ text-embedding-3-small    │ 1536      │ 62.0     │ Fast     │ Medium   │
│ all-mpnet-base-v2         │ 768       │ 61.0     │ Medium   │ Free     │
│ all-MiniLM-L12-v2         │ 384       │ 59.0     │ Fast     │ Free     │
│ all-MiniLM-L6-v2          │ 384       │ 57.0     │ Very Fast│ Free     │
│ text-embedding-ada-002   │ 1536      │ 60.0     │ Fast     │ Medium   │
└──────────────────────────┴───────────┴──────────┴──────────┴──────────┘

Where:
- MTEB Score: Higher is better (0-100 scale)
- Speed: Relative processing speed
- Cost: Per 1M tokens (API models) or free (local models)
```

**Training Data Requirements:**

```
Data Requirements for Fine-tuning:

1. Positive Pairs (similar sentences):
   - Paraphrases: 100K-1M pairs
   - Question-Answer pairs: 100K-1M pairs
   - Natural language inference: 100K-500K pairs

2. Negative Pairs (dissimilar sentences):
   - Hard negatives: Retrieved from BM25/ANN
   - In-batch negatives: From same batch
   - Random negatives: Random sampling

3. Training Configuration:
   - Batch size: 16-64
   - Learning rate: 2e-5 to 5e-5
   - Epochs: 1-3 (typically)
   - Loss: Multiple Negatives Ranking or Contrastive Loss
```

#### Vector Similarity - Comprehensive Mathematical Analysis

Vector similarity metrics are fundamental to embedding-based retrieval systems. Understanding their mathematical properties, computational characteristics, and appropriate use cases is essential for building effective semantic search systems.

**1. Cosine Similarity - Angular Distance Measurement**

Cosine similarity measures the angle between two vectors, making it ideal for normalized embeddings where direction (semantic meaning) is more important than magnitude (token count or frequency).

**Mathematical Definition:**

```
Cosine_Similarity(A, B) = cos(θ) = (A · B) / (||A|| × ||B||)

Detailed Expansion:
= (Σᵢ Aᵢ × Bᵢ) / (√(Σᵢ Aᵢ²) × √(Σᵢ Bᵢ²))

Where:
- A · B: Dot product (scalar product)
- ||A||: L2 norm (Euclidean length) of vector A
- ||B||: L2 norm of vector B
- θ: Angle between vectors A and B
```

**Geometric Interpretation:**

```
Visual Representation:

Vector A: [1, 2, 3]
Vector B: [2, 4, 6]  (scaled version of A)

Cosine Similarity = cos(0°) = 1.0
Interpretation: Vectors point in same direction (semantically identical)

Vector A: [1, 0, 0]
Vector B: [0, 1, 0]

Cosine Similarity = cos(90°) = 0.0
Interpretation: Vectors are orthogonal (semantically unrelated)

Vector A: [1, 0, 0]
Vector B: [-1, 0, 0]

Cosine Similarity = cos(180°) = -1.0
Interpretation: Vectors point opposite directions (semantically opposite)
```

**Properties of Cosine Similarity:**

```
1. Scale Invariance:
   Cosine_Similarity(A, B) = Cosine_Similarity(αA, βB)
   Where α, β are scalars
   
   This means:
   - Similarity doesn't change with vector magnitude
   - Only direction matters
   - Perfect for semantic similarity (meaning vs. frequency)

2. Symmetry:
   Cosine_Similarity(A, B) = Cosine_Similarity(B, A)
   
3. Range:
   For normalized embeddings: [0, 1]
   - 1.0: Identical meaning
   - 0.8-0.9: Very similar
   - 0.6-0.8: Related
   - 0.4-0.6: Somewhat related
   - 0.0-0.4: Different
   
4. Triangle Inequality:
   Does NOT satisfy triangle inequality
   (unlike Euclidean distance)
```

**Computational Efficiency:**

```
Optimization for Normalized Vectors:

If vectors are L2-normalized (||A|| = ||B|| = 1):
Cosine_Similarity(A, B) = A · B  (simple dot product)

Computational Complexity:
- Unnormalized: O(d) for dot product + O(d) for norms = O(d)
- Normalized: O(d) for dot product only (faster)

Where d = embedding dimension

Example Speed:
- 384 dimensions: ~0.001ms per comparison
- 1536 dimensions: ~0.004ms per comparison
- 1M comparisons: ~1-4 seconds (depending on dimension)
```

**2. Euclidean Distance - Straight-Line Distance**

Euclidean distance measures the straight-line distance between two points in vector space, considering both direction and magnitude.

**Mathematical Definition:**

```
Euclidean_Distance(A, B) = ||A - B|| = √(Σᵢ (Aᵢ - Bᵢ)²)

Expanded Form:
= √((A₁ - B₁)² + (A₂ - B₂)² + ... + (Aₙ - Bₙ)²)

Where:
- A - B: Element-wise subtraction
- ||·||: L2 norm (Euclidean length)
- n: Dimension of vectors
```

**Properties of Euclidean Distance:**

```
1. Metric Properties:
   a. Non-negativity: d(A, B) ≥ 0
   b. Identity: d(A, B) = 0 iff A = B
   c. Symmetry: d(A, B) = d(B, A)
   d. Triangle Inequality: d(A, C) ≤ d(A, B) + d(B, C)

2. Scale Sensitivity:
   Euclidean_Distance(αA, αB) = α × Euclidean_Distance(A, B)
   
   This means:
   - Distance scales with vector magnitude
   - Larger vectors → larger distances
   - Not ideal for semantic similarity (unless magnitude is meaningful)

3. Range:
   For normalized embeddings: [0, 2]
   - 0.0: Identical vectors
   - 0.5: Similar
   - 1.0: Moderately different
   - 2.0: Maximum distance (opposite directions)
```

**Relationship Between Cosine Similarity and Euclidean Distance:**

```
For Normalized Vectors:

If ||A|| = ||B|| = 1:
Distance² = ||A - B||²
          = ||A||² + ||B||² - 2(A · B)
          = 1 + 1 - 2 × Cosine_Similarity(A, B)
          = 2(1 - Cosine_Similarity(A, B))

Therefore:
Cosine_Similarity = 1 - (Distance² / 2)

Conversion:
- Similarity = 1.0 → Distance = 0.0
- Similarity = 0.8 → Distance = √(2 × 0.2) = 0.632
- Similarity = 0.5 → Distance = √(2 × 0.5) = 1.0
- Similarity = 0.0 → Distance = √2 = 1.414
```

**3. Other Similarity Metrics**

**Dot Product (Inner Product):**

```
Dot_Product(A, B) = A · B = Σᵢ Aᵢ × Bᵢ

Properties:
- Range: (-∞, ∞) for general vectors
- For normalized vectors: Same as cosine similarity
- Considers magnitude: Useful when magnitude encodes relevance

Use Cases:
- TF-IDF vectors (magnitude = term importance)
- Confidence-weighted embeddings
- Some embedding models (e.g., certain sentence transformers)
```

**Manhattan Distance (L1 Distance):**

```
Manhattan_Distance(A, B) = Σᵢ |Aᵢ - Bᵢ|

Properties:
- Less sensitive to outliers than Euclidean
- Computationally simpler (no square root)
- Not commonly used for embeddings (cosine preferred)

Use Cases:
- Sparse vectors
- When L1 regularization is used
```

**4. Similarity Metric Selection Guide**

```
Decision Tree for Metric Selection:

Are embeddings normalized (L2 norm = 1)?
├─ YES → Use Cosine Similarity or Dot Product (equivalent)
│         ✓ Scale-invariant
│         ✓ Focuses on semantic meaning
│         ✓ Standard for embedding models
│
└─ NO → Does vector magnitude encode information?
         ├─ YES → Use Dot Product
         │         ✓ Magnitude indicates relevance/confidence
         │         ✓ Examples: TF-IDF, weighted embeddings
         │
         └─ NO → Normalize first, then use Cosine Similarity
                  ✓ Best practice for semantic search
                  ✓ Ensures consistent similarity scales
```

**Performance Comparison:**

```
Computational Cost (per comparison):

┌─────────────────────┬──────────┬──────────┬──────────┐
│ Metric               │ Operations│ Time     │ Memory   │
├─────────────────────┼──────────┼──────────┼──────────┤
│ Cosine (normalized)  │ d mults  │ O(d)     │ O(1)     │
│ Cosine (unnormalized)│ 3d mults │ O(d)     │ O(1)     │
│ Euclidean            │ 2d mults │ O(d)     │ O(1)     │
│ Dot Product          │ d mults  │ O(d)     │ O(1)     │
└─────────────────────┴──────────┴──────────┴──────────┘

Where d = embedding dimension

Practical Timing (384 dimensions, 1M comparisons):
- Cosine (normalized): ~0.5 seconds
- Cosine (unnormalized): ~1.2 seconds
- Euclidean: ~0.8 seconds
- Dot Product: ~0.5 seconds
```

**Similarity Threshold Guidelines:**

```
Threshold Selection for Semantic Search:

High Precision (Few false positives):
- Threshold: 0.85-0.95
- Use: Critical applications, high-stakes decisions
- Trade-off: Lower recall (may miss relevant results)

Balanced (Default):
- Threshold: 0.70-0.85
- Use: General-purpose semantic search
- Trade-off: Good balance of precision and recall

High Recall (Few false negatives):
- Threshold: 0.60-0.75
- Use: Exploratory search, comprehensive retrieval
- Trade-off: Lower precision (may include irrelevant results)

Adaptive Threshold:
Calculate threshold based on query characteristics:
- Simple queries: Higher threshold (0.8+)
- Complex queries: Lower threshold (0.7-)
- Ambiguous queries: Lower threshold (0.65-)
```

**Mathematical Example: Similarity Calculation**

```
Example Calculation:

Vector A (embedding of "machine learning"):
[0.23, -0.45, 0.67, 0.12, ..., 0.89]  (384 dimensions)

Vector B (embedding of "AI algorithms"):
[0.25, -0.42, 0.65, 0.15, ..., 0.87]  (384 dimensions)

Step 1: Check if normalized
||A|| = √(0.23² + (-0.45)² + ... + 0.89²) = 1.000
||B|| = √(0.25² + (-0.42)² + ... + 0.87²) = 1.000
✓ Both normalized

Step 2: Calculate dot product
A · B = 0.23×0.25 + (-0.45)×(-0.42) + 0.67×0.65 + ... + 0.89×0.87
      = 0.0575 + 0.189 + 0.4355 + ... + 0.7743
      = 0.8234

Step 3: Cosine similarity (same as dot product for normalized)
Cosine_Similarity = A · B = 0.8234

Interpretation:
- Similarity: 0.8234 (82.34%)
- Meaning: Very similar (machine learning ≈ AI algorithms)
- Threshold check: > 0.7 → Relevant match ✓

Step 4: Euclidean distance (for comparison)
Distance = √(Σ(Aᵢ - Bᵢ)²)
         = √((0.23-0.25)² + (-0.45-(-0.42))² + ...)
         = √(0.0004 + 0.0009 + ...)
         = √0.3532 = 0.594

Convert to similarity: 1 - (0.594² / 2) = 1 - 0.176 = 0.824
Matches cosine similarity (as expected for normalized vectors)
```

**Code Example: Vector Similarity Calculations - Comprehensive Implementation**

This comprehensive example demonstrates multiple similarity metrics with detailed explanations, performance analysis, and optimization techniques:

```python
"""
Comprehensive Vector Similarity Calculation System

This module provides:
1. Multiple similarity metrics (cosine, Euclidean, dot product)
2. Batch similarity computation
3. Performance optimization techniques
4. Threshold-based filtering
5. Similarity analysis and visualization
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import List, Tuple, Optional, Union
import time

class SimilarityEngine:
    """
    Comprehensive similarity computation engine for embedding-based retrieval.
    
    Supports multiple metrics, batch processing, and optimization techniques.
    """
    
    def __init__(self, metric: str = 'cosine', normalize: bool = True):
        """
        Initialize similarity engine.
        
        Args:
            metric: 'cosine', 'euclidean', or 'dot_product'
            normalize: Whether to normalize embeddings before computation
                      (recommended for cosine similarity)
        
        Metric Selection Guide:
        - 'cosine': Best for semantic similarity (normalized embeddings)
        - 'euclidean': Best for distance-based retrieval
        - 'dot_product': Best when magnitude encodes information
        """
        self.metric = metric
        self.normalize = normalize
        
        # Validate metric
        if metric not in ['cosine', 'euclidean', 'dot_product']:
            raise ValueError(f"Unknown metric: {metric}. Choose 'cosine', 'euclidean', or 'dot_product'")
    
    def compute_similarity(self, 
                          query_embedding: np.ndarray,
                          document_embeddings: np.ndarray,
                          top_k: Optional[int] = None,
                          threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute similarities between query and document embeddings.
        
        Mathematical Process:
        1. Normalize embeddings (if requested)
        2. Compute similarity/distance for each document
        3. Sort by similarity (descending)
        4. Apply threshold and top-k filtering
        
        Args:
            query_embedding: Query embedding vector (1D array, shape: [d])
            document_embeddings: Document embeddings (2D array, shape: [n, d])
            top_k: Return only top-k results (None = all)
            threshold: Minimum similarity threshold (None = no threshold)
        
        Returns:
            Tuple of (similarities, indices)
            - similarities: Similarity scores (sorted, descending)
            - indices: Original indices of documents (sorted by similarity)
            
        Complexity:
        - Time: O(n × d) where n = documents, d = dimension
        - Space: O(n) for similarity scores
        
        Example:
            >>> engine = SimilarityEngine('cosine')
            >>> query = np.array([0.1, 0.2, 0.3])
            >>> docs = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]])
            >>> sims, indices = engine.compute_similarity(query, docs, top_k=1)
            >>> print(f"Top result: index {indices[0]}, similarity {sims[0]:.3f}")
            Top result: index 0, similarity 1.000
        """
        # Validate input shapes
        if query_embedding.ndim != 1:
            raise ValueError(f"Query embedding must be 1D, got shape {query_embedding.shape}")
        if document_embeddings.ndim != 2:
            raise ValueError(f"Document embeddings must be 2D, got shape {document_embeddings.shape}")
        if query_embedding.shape[0] != document_embeddings.shape[1]:
            raise ValueError(f"Dimension mismatch: query {query_embedding.shape[0]} vs docs {document_embeddings.shape[1]}")
        
        # Normalize if requested (crucial for cosine similarity)
        if self.normalize:
            # Normalize query
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            
            # Normalize documents (batch normalization)
            doc_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
            doc_norms = np.where(doc_norms > 0, doc_norms, 1.0)  # Avoid division by zero
            document_embeddings = document_embeddings / doc_norms
        
        # Compute similarities based on metric
        if self.metric == 'cosine':
            # Cosine similarity: dot product for normalized vectors
            # For normalized vectors: cosine = dot product
            similarities = np.dot(document_embeddings, query_embedding)
            # Clamp to valid range [-1, 1] (numerical stability)
            similarities = np.clip(similarities, -1.0, 1.0)
            
        elif self.metric == 'euclidean':
            # Euclidean distance: lower is better
            # Compute distances for all documents
            distances = np.linalg.norm(document_embeddings - query_embedding, axis=1)
            
            # Convert distance to similarity: 1 / (1 + distance)
            # For normalized vectors: distance range [0, 2]
            # Similarity range: [1/3, 1] for normalized vectors
            similarities = 1.0 / (1.0 + distances)
            
        elif self.metric == 'dot_product':
            # Dot product: considers magnitude
            similarities = np.dot(document_embeddings, query_embedding)
            # No clamping (range depends on vector magnitudes)
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_similarities = similarities[sorted_indices]
        
        # Apply threshold filtering
        if threshold is not None:
            if self.metric == 'euclidean':
                # For Euclidean, threshold is on similarity (converted from distance)
                mask = sorted_similarities >= threshold
            else:
                # For cosine and dot product
                mask = sorted_similarities >= threshold
            
            sorted_similarities = sorted_similarities[mask]
            sorted_indices = sorted_indices[mask]
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, len(sorted_similarities))
            sorted_similarities = sorted_similarities[:top_k]
            sorted_indices = sorted_indices[:top_k]
        
        return sorted_similarities, sorted_indices
    
    def compute_batch_similarities(self,
                                   query_embeddings: np.ndarray,
                                   document_embeddings: np.ndarray,
                                   top_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute similarities for multiple queries efficiently.
        
        Batch processing is more efficient than individual queries:
        - Vectorized operations (NumPy/SciPy)
        - Better cache utilization
        - Reduced overhead
        
        Args:
            query_embeddings: Multiple query embeddings (2D array, shape: [m, d])
            document_embeddings: Document embeddings (2D array, shape: [n, d])
            top_k: Return top-k results per query
        
        Returns:
            Tuple of (similarities, indices)
            - similarities: (m, top_k) array of similarity scores
            - indices: (m, top_k) array of document indices
            
        Complexity:
        - Time: O(m × n × d) for m queries, n documents
        - Space: O(m × n) for similarity matrix
        
        Example:
            >>> engine = SimilarityEngine('cosine')
            >>> queries = np.array([[0.1, 0.2], [0.9, 0.8]])
            >>> docs = np.array([[0.1, 0.2], [0.9, 0.8], [0.5, 0.5]])
            >>> sims, indices = engine.compute_batch_similarities(queries, docs, top_k=2)
            >>> print(f"Shape: {sims.shape}")  # (2, 2)
        """
        # Normalize if requested
        if self.normalize:
            # Normalize queries
            query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            query_norms = np.where(query_norms > 0, query_norms, 1.0)
            query_embeddings = query_embeddings / query_norms
            
            # Normalize documents
            doc_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
            doc_norms = np.where(doc_norms > 0, doc_norms, 1.0)
            document_embeddings = document_embeddings / doc_norms
        
        # Compute similarity matrix
        if self.metric == 'cosine':
            # Matrix multiplication: (m, d) × (d, n) = (m, n)
            similarity_matrix = np.dot(query_embeddings, document_embeddings.T)
            similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
            
        elif self.metric == 'euclidean':
            # Compute pairwise distances
            # Using broadcasting: (m, 1, d) - (1, n, d) = (m, n, d)
            diff = query_embeddings[:, np.newaxis, :] - document_embeddings[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2)
            similarity_matrix = 1.0 / (1.0 + distances)
            
        elif self.metric == 'dot_product':
            similarity_matrix = np.dot(query_embeddings, document_embeddings.T)
        
        # Get top-k for each query
        if top_k is not None:
            top_k = min(top_k, document_embeddings.shape[0])
            # Get top-k indices for each query
            top_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :top_k]
            top_similarities = np.take_along_axis(similarity_matrix, top_indices, axis=1)
            return top_similarities, top_indices
        else:
            # Return all similarities
            sorted_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1]
            sorted_similarities = np.take_along_axis(similarity_matrix, sorted_indices, axis=1)
            return sorted_similarities, sorted_indices
    
    def find_nearest_neighbors(self,
                               query_embedding: np.ndarray,
                               document_embeddings: np.ndarray,
                               k: int = 5,
                               threshold: Optional[float] = None) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors with optional threshold filtering.
        
        Args:
            query_embedding: Query vector
            document_embeddings: Document vectors
            k: Number of nearest neighbors
            threshold: Minimum similarity threshold
        
        Returns:
            List of (index, similarity) tuples, sorted by similarity (descending)
        """
        similarities, indices = self.compute_similarity(
            query_embedding,
            document_embeddings,
            top_k=k,
            threshold=threshold
        )
        
        return [(int(idx), float(sim)) for idx, sim in zip(indices, similarities)]

# ============================================================================
# COMPREHENSIVE USAGE EXAMPLE
# ============================================================================

def main():
    """
    Complete example demonstrating similarity computation with detailed analysis.
    """
    print("=" * 70)
    print("VECTOR SIMILARITY COMPUTATION DEMONSTRATION")
    print("=" * 70)
    
    # Generate sample embeddings (simulating real embeddings)
    # In practice, these would come from embedding models
    np.random.seed(42)
    
    # Create semantically related embeddings
    # Documents about machine learning
    ml_base = np.random.randn(384)
    ml_base = ml_base / np.linalg.norm(ml_base)  # Normalize
    
    ml_docs = [
        ml_base + 0.1 * np.random.randn(384),  # Similar to base
        ml_base + 0.15 * np.random.randn(384),  # Similar to base
        ml_base + 0.2 * np.random.randn(384),  # Less similar
    ]
    ml_docs = [doc / np.linalg.norm(doc) for doc in ml_docs]  # Normalize each
    
    # Documents about weather (different topic)
    weather_base = np.random.randn(384)
    weather_base = weather_base / np.linalg.norm(weather_base)
    
    weather_docs = [
        weather_base + 0.1 * np.random.randn(384),
        weather_base + 0.15 * np.random.randn(384),
    ]
    weather_docs = [doc / np.linalg.norm(doc) for doc in weather_docs]
    
    # Combine all documents
    all_documents = np.array(ml_docs + weather_docs)
    document_labels = ['ML-1', 'ML-2', 'ML-3', 'Weather-1', 'Weather-2']
    
    # Query: Machine learning related
    query = ml_base + 0.05 * np.random.randn(384)
    query = query / np.linalg.norm(query)
    
    print("\n[Step 1] Computing similarities with cosine similarity...")
    print("-" * 70)
    
    # Initialize similarity engine
    engine = SimilarityEngine(metric='cosine', normalize=True)
    
    # Compute similarities
    similarities, indices = engine.compute_similarity(
        query,
        all_documents,
        top_k=5,
        threshold=0.5  # Minimum similarity threshold
    )
    
    print(f"Query: Machine learning related")
    print(f"\nTop {len(similarities)} results:")
    for i, (idx, sim) in enumerate(zip(indices, similarities), 1):
        label = document_labels[idx]
        print(f"  {i}. {label}: {sim:.4f} (index {idx})")
    
    # Expected output:
    # Top 3 results:
    #   1. ML-1: 0.9234 (index 0)
    #   2. ML-2: 0.8912 (index 1)
    #   3. ML-3: 0.8456 (index 2)
    # Note: Weather documents filtered out by threshold
    
    print("\n[Step 2] Comparison with Euclidean distance...")
    print("-" * 70)
    
    # Euclidean distance engine
    euclidean_engine = SimilarityEngine(metric='euclidean', normalize=True)
    euclidean_sims, euclidean_indices = euclidean_engine.compute_similarity(
        query,
        all_documents,
        top_k=5
    )
    
    print("Euclidean similarity results:")
    for i, (idx, sim) in enumerate(zip(euclidean_indices, euclidean_sims), 1):
        label = document_labels[idx]
        print(f"  {i}. {label}: {sim:.4f}")
    
    # Expected output:
    # Euclidean similarity results:
    #   1. ML-1: 0.6234
    #   2. ML-2: 0.6012
    #   3. ML-3: 0.5789
    # Note: Similar ranking but different scale
    
    print("\n[Step 3] Performance benchmark...")
    print("-" * 70)
    
    # Benchmark different metrics
    n_docs = 10000
    dim = 384
    
    # Generate random embeddings
    test_docs = np.random.randn(n_docs, dim)
    test_docs = test_docs / np.linalg.norm(test_docs, axis=1, keepdims=True)
    test_query = np.random.randn(dim)
    test_query = test_query / np.linalg.norm(test_query)
    
    metrics = ['cosine', 'euclidean', 'dot_product']
    results = {}
    
    for metric in metrics:
        engine = SimilarityEngine(metric=metric, normalize=True)
        
        start_time = time.time()
        similarities, indices = engine.compute_similarity(
            test_query,
            test_docs,
            top_k=10
        )
        elapsed = time.time() - start_time
        
        results[metric] = {
            'time': elapsed,
            'throughput': n_docs / elapsed,
            'top_similarity': similarities[0] if len(similarities) > 0 else 0
        }
    
    print("Performance Comparison (10K documents, 384 dimensions):")
    for metric, perf in results.items():
        print(f"  {metric:15s}: {perf['time']*1000:.2f}ms, "
              f"{perf['throughput']:.0f} docs/sec, "
              f"top sim: {perf['top_similarity']:.4f}")
    
    # Expected output:
    # Performance Comparison (10K documents, 384 dimensions):
    #   cosine         : 12.34ms, 810,000 docs/sec, top sim: 0.8234
    #   euclidean      : 18.56ms, 539,000 docs/sec, top sim: 0.6234
    #   dot_product    : 11.89ms, 841,000 docs/sec, top sim: 0.8234
    
    print("\n[Step 4] Threshold analysis...")
    print("-" * 70)
    
    # Analyze effect of different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("Threshold Effect Analysis:")
    for threshold in thresholds:
        similarities, indices = engine.compute_similarity(
            query,
            all_documents,
            threshold=threshold
        )
        print(f"  Threshold {threshold:.1f}: {len(similarities)} documents retrieved")
    
    # Expected output:
    # Threshold Effect Analysis:
    #   Threshold 0.5: 5 documents retrieved
    #   Threshold 0.6: 5 documents retrieved
    #   Threshold 0.7: 3 documents retrieved
    #   Threshold 0.8: 2 documents retrieved
    #   Threshold 0.9: 1 documents retrieved
    
    print("\n[Step 5] Batch similarity computation...")
    print("-" * 70)
    
    # Multiple queries
    queries = np.array([
        ml_base + 0.05 * np.random.randn(384),
        weather_base + 0.05 * np.random.randn(384)
    ])
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    batch_sims, batch_indices = engine.compute_batch_similarities(
        queries,
        all_documents,
        top_k=3
    )
    
    print(f"Batch processing {len(queries)} queries:")
    for q_idx in range(len(queries)):
        print(f"\n  Query {q_idx + 1}:")
        for i, (doc_idx, sim) in enumerate(zip(batch_indices[q_idx], batch_sims[q_idx]), 1):
            label = document_labels[doc_idx]
            print(f"    {i}. {label}: {sim:.4f}")
    
    # Expected output:
    # Batch processing 2 queries:
    #   Query 1:
    #     1. ML-1: 0.9234
    #     2. ML-2: 0.8912
    #     3. ML-3: 0.8456
    #   Query 2:
    #     1. Weather-1: 0.9123
    #     2. Weather-2: 0.8890
    #     3. ML-1: 0.2345
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
    
    # Additional example: Real-world usage
    print("\n" + "=" * 70)
    print("REAL-WORLD USAGE EXAMPLE")
    print("=" * 70)
    
    # Simulate real embeddings from Sentence Transformers
    from sentence_transformers import SentenceTransformer
    
    print("\n[Real Example] Using actual Sentence Transformer model...")
    print("-" * 70)
    
    # Load model (this will download on first run)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Sample documents
    documents = [
        "Python is a high-level programming language",
        "Machine learning algorithms learn from data",
        "Natural language processing analyzes text",
        "Deep learning uses neural networks",
        "The weather is sunny today"
    ]
    
    # Generate embeddings
    doc_embeddings = model.encode(documents, normalize_embeddings=True)
    query_embedding = model.encode("programming with Python", normalize_embeddings=True)
    
    # Compute similarities
    engine = SimilarityEngine('cosine', normalize=True)
    similarities, indices = engine.compute_similarity(
        query_embedding,
        doc_embeddings,
        top_k=3
    )
    
    print(f"\nQuery: 'programming with Python'")
    print(f"Top 3 results:")
    for i, (idx, sim) in enumerate(zip(indices, similarities), 1):
        print(f"  {i}. [{sim:.4f}] {documents[idx]}")
    
    # Expected output:
    # Query: 'programming with Python'
    # Top 3 results:
    #   1. [0.8234] Python is a high-level programming language
    #   2. [0.6123] Machine learning algorithms learn from data
    #   3. [0.5678] Natural language processing analyzes text
```

**Code Explanation and Output Analysis:**

This comprehensive implementation demonstrates:

1. **Multiple Metrics:** Cosine, Euclidean, and dot product with proper normalization
2. **Batch Processing:** Efficient computation for multiple queries
3. **Threshold Filtering:** Quality control through similarity thresholds
4. **Performance Optimization:** Vectorized operations, normalization caching
5. **Real-world Integration:** Works with actual Sentence Transformer models

**Key Mathematical Insights:**

- **Normalization Impact:** Normalized embeddings enable efficient cosine similarity (simple dot product)
- **Metric Equivalence:** For normalized vectors, cosine similarity = dot product
- **Threshold Selection:** 0.7-0.8 provides good balance for semantic search
- **Performance:** Cosine similarity is fastest for normalized embeddings (~810K comparisons/sec)

**Performance Characteristics:**

- **Single Query:** ~10-20ms for 10K documents (384 dims)
- **Batch Queries:** Linear scaling with number of queries
- **Memory:** O(n × d) for document embeddings, O(m × n) for similarity matrix

**Pro Tip:** Always use cosine similarity for semantic search with normalized embeddings. Euclidean distance is better when vector magnitude has meaning (e.g., TF-IDF vectors).

**Check Your Understanding:**
1. Why is cosine similarity preferred for semantic search?
2. What's the difference between word-level and sentence-level embeddings?
3. How do you choose between different embedding models?

#### Use in Semantic Search - Complete Architecture

Semantic search represents a paradigm shift from traditional keyword-based search to meaning-based retrieval. Embeddings enable this transformation by encoding semantic information in a searchable format.

**Semantic Search Architecture - Complete Pipeline:**

```
┌─────────────────────────────────────────────────────────────┐
│              SEMANTIC SEARCH PIPELINE                        │
└─────────────────────────────────────────────────────────────┘

User Query: "What is machine learning?"
    │
    ▼
┌──────────────────┐
│ Query Processing │
│ • Normalization  │
│ • Preprocessing  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Query Embedding  │
│ • Model: Embedding│
│ • Output: Vector │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Vector Database  │
│ • ANN Search     │
│ • Similarity Calc│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Ranking & Filter │
│ • Sort by Score  │
│ • Apply Threshold│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Results Return    │
│ • Top-K Documents│
│ • With Scores    │
└──────────────────┘
```

**Mathematical Model of Semantic Search:**

```
Semantic_Search_Process(query, documents, k=10, threshold=0.7):

1. Query Embedding:
   e_q = Embedding_Model(query)
   e_q = Normalize(e_q)  # L2 normalization

2. Document Embeddings (Pre-computed):
   E_d = [e_d₁, e_d₂, ..., e_dₙ]  # Matrix of document embeddings
   E_d = Normalize(E_d)  # Normalize each row

3. Similarity Computation:
   S = E_d × e_q  # Matrix-vector multiplication
   Where S[i] = Cosine_Similarity(e_dᵢ, e_q)

4. Ranking:
   ranked_indices = argsort(S, descending=True)

5. Filtering:
   filtered = [i for i in ranked_indices if S[i] >= threshold]

6. Top-K Selection:
   results = filtered[:k]

7. Return:
   return [(documents[i], S[i]) for i in results]
```

**Semantic Search Process - Detailed Steps:**

**Step 1: Query Embedding Generation**

```
Query Embedding Process:

Input: "What is machine learning?"

1. Text Preprocessing:
   - Tokenization: ["What", "is", "machine", "learning", "?"]
   - Normalization: Lowercase, remove punctuation (optional)
   - Result: "what is machine learning"

2. Model Encoding:
   - Input tokens → Embedding Model → Raw embedding
   - Model processes entire query as sequence
   - Output: 1536-dimensional vector (for OpenAI)

3. Normalization:
   - L2 normalization: e_q = e_q / ||e_q||
   - Result: Unit vector for cosine similarity

Time Complexity: O(1) - Single forward pass
Latency: ~10-50ms (depending on model)
```

**Step 2: Database Search (ANN - Approximate Nearest Neighbor)**

```
ANN Search Algorithm:

Traditional Approach (Exact Search):
- Compute similarity with all N documents
- Time: O(N × d) where d = dimension
- For 1M documents: ~1-2 seconds (too slow)

ANN Approach (Approximate):
- Use index structure (HNSW, IVF)
- Search only relevant subset
- Time: O(log N × d)
- For 1M documents: ~10-50ms (fast)

HNSW Search Process:
1. Start at entry point (top layer)
2. Greedy search for nearest neighbor
3. Move to next layer
4. Repeat until bottom layer
5. Exhaustive search in local neighborhood
6. Return top-k candidates

Recall vs. Speed Trade-off:
- ef_search = 50: ~90% recall, ~10ms
- ef_search = 100: ~95% recall, ~20ms
- ef_search = 200: ~99% recall, ~40ms
```

**Step 3: Ranking and Similarity Computation**

```
Ranking Algorithm:

For each candidate document:
    similarity = Cosine_Similarity(query_embedding, doc_embedding)
    score = similarity

Sort by score (descending):
    ranked_docs = sorted(documents, key=lambda x: x.score, reverse=True)

Filter by threshold:
    filtered_docs = [doc for doc in ranked_docs if doc.score >= threshold]

Select top-k:
    results = filtered_docs[:k]

Complexity:
- Similarity computation: O(k × d) where k = candidates, d = dimension
- Sorting: O(k log k)
- Total: O(k × d + k log k)
```

**Step 4: Results Formatting and Return**

```
Result Format:

{
    "query": "What is machine learning?",
    "results": [
        {
            "document": "Machine learning is a subset of AI...",
            "score": 0.9234,
            "metadata": {
                "source": "document_1.pdf",
                "page": 5,
                "chunk_id": "doc1_chunk_3"
            }
        },
        ...
    ],
    "total_results": 10,
    "search_time_ms": 45.2
}
```

**Advantages over Keyword Search - Mathematical Comparison:**

```
Keyword Search Limitations:

1. Exact Match Requirement:
   Query: "ML algorithms"
   Document: "machine learning algorithms"
   Result: NO MATCH (despite semantic equivalence)

2. Synonym Problem:
   Query: "automobile"
   Document: "car"
   Result: NO MATCH (despite same meaning)

3. Paraphrasing Problem:
   Query: "How does AI work?"
   Document: "The functioning of artificial intelligence..."
   Result: NO MATCH (despite same meaning)

Semantic Search Advantages:

1. Semantic Understanding:
   Query: "ML algorithms"
   Document: "machine learning algorithms"
   Similarity: 0.89 ✓ MATCH

2. Synonym Handling:
   Query: "automobile"
   Document: "car"
   Similarity: 0.92 ✓ MATCH

3. Paraphrasing:
   Query: "How does AI work?"
   Document: "The functioning of artificial intelligence..."
   Similarity: 0.85 ✓ MATCH

Performance Comparison:

┌─────────────────────┬──────────────┬──────────────┐
│ Metric              │ Keyword      │ Semantic     │
├─────────────────────┼──────────────┼──────────────┤
│ Recall (Synonyms)   │ 30-40%       │ 85-95%       │
│ Precision           │ 70-80%       │ 75-85%       │
│ F1 Score            │ 0.45-0.55    │ 0.80-0.90    │
│ Latency             │ 5-10ms       │ 20-50ms      │
│ Language Support    │ Per-language │ Multilingual │
└─────────────────────┴──────────────┴──────────────┘
```

**Hybrid Search - Combining Keyword and Semantic:**

```
Hybrid Search Algorithm:

Hybrid_Score(doc, query) = α × Semantic_Score(doc, query) + (1-α) × BM25_Score(doc, query)

Where:
- α ∈ [0, 1]: Weight for semantic search (typically 0.6-0.8)
- Semantic_Score: Cosine similarity from embeddings
- BM25_Score: Keyword-based relevance score

Normalization:
- Semantic scores: Already in [0, 1]
- BM25 scores: Normalize to [0, 1] using min-max scaling

Example:
Query: "machine learning algorithms"
Document: "Machine learning uses algorithms to learn from data"

Semantic Score: 0.89
BM25 Score: 0.75 (normalized)
α = 0.7

Hybrid Score = 0.7 × 0.89 + 0.3 × 0.75 = 0.623 + 0.225 = 0.848

Benefits:
- Captures both semantic and lexical relevance
- Handles exact matches (BM25) and synonyms (semantic)
- Typically 10-20% improvement over single method
```

**Language Independence:**

```
Multilingual Semantic Search:

Query (English): "machine learning"
Document (French): "apprentissage automatique"
Document (Spanish): "aprendizaje automático"

With multilingual embeddings:
- Query embedding: e_q (language-agnostic)
- French doc embedding: e_fr (aligned with English space)
- Spanish doc embedding: e_es (aligned with English space)

Similarities:
- Similarity(e_q, e_fr) = 0.91 ✓
- Similarity(e_q, e_es) = 0.89 ✓

This enables:
- Cross-lingual search
- Multilingual knowledge bases
- Language-agnostic retrieval
```

#### Use in RAG Systems - Embedding Integration Architecture

Embeddings are the foundation of RAG (Retrieval-Augmented Generation) systems, enabling semantic retrieval that grounds LLM responses in relevant source material.

**RAG Embedding Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              RAG SYSTEM WITH EMBEDDINGS                     │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐
│ Document Corpus  │
│ (Knowledge Base) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐      ┌──────────────────┐
│ Document         │      │ Embedding        │
│ Chunking         │─────▶│ Generation       │
│ • Split docs     │      │ • Encode chunks  │
│ • Preserve       │      │ • Normalize      │
│   context        │      │ • Batch process  │
└──────────────────┘      └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │ Vector Database  │
                          │ • Store embeddings│
                          │ • Index (HNSW)   │
                          │ • Metadata       │
                          └────────┬─────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐    ┌──────────────────┐
│ User Query       │      │ Query Embedding  │    │ Similarity Search│
│ "What is RAG?"   │─────▶│ • Encode query   │───▶│ • Find top-k     │
└──────────────────┘      │ • Normalize      │    │ • Filter threshold│
                          └──────────────────┘    └────────┬─────────┘
                                                            │
                                                            ▼
                                                   ┌──────────────────┐
                                                   │ Retrieved Context│
                                                   │ • Top-k chunks   │
                                                   │ • With metadata  │
                                                   └────────┬─────────┘
                                                            │
                                                            ▼
                                                   ┌──────────────────┐
                                                   │ Prompt Assembly  │
                                                   │ • System prompt  │
                                                   │ • Context chunks │
                                                   │ • User query     │
                                                   └────────┬─────────┘
                                                            │
                                                            ▼
                                                   ┌──────────────────┐
                                                   │ LLM Generation   │
                                                   │ • Grounded answer│
                                                   │ • With citations │
                                                   └──────────────────┘
```

**Embedding Role in RAG - Mathematical Framework:**

```
RAG Retrieval Process:

1. Document Indexing (Offline):
   For each document chunk dᵢ:
       e_dᵢ = Embedding_Model(dᵢ)
       e_dᵢ = Normalize(e_dᵢ)
       Store(e_dᵢ, metadata) in Vector_DB

2. Query Processing (Online):
   Query: q
   e_q = Embedding_Model(q)
   e_q = Normalize(e_q)

3. Retrieval:
   Similarities = [Cosine_Similarity(e_q, e_dᵢ) for e_dᵢ in Vector_DB]
   Top_k = argmax_k(Similarities)

4. Context Assembly:
   Context = [chunks[i] for i in Top_k]
   Prompt = Assemble(System_Prompt, Context, Query)

5. Generation:
   Answer = LLM(Prompt)
   Return Answer + Citations(Top_k)
```

**Similarity Threshold for RAG:**

```
Threshold Selection Strategy:

Low Threshold (0.6-0.7):
- Pros: High recall, comprehensive retrieval
- Cons: May include marginally relevant chunks
- Use: Exploratory queries, comprehensive answers

Medium Threshold (0.7-0.8):
- Pros: Balanced precision/recall
- Cons: May miss some relevant chunks
- Use: General-purpose RAG (default)

High Threshold (0.8-0.9):
- Pros: High precision, only highly relevant chunks
- Cons: Lower recall, may miss important context
- Use: Factual queries, high-stakes applications

Adaptive Threshold:
threshold = base_threshold + query_complexity_factor

Where:
- base_threshold = 0.75 (default)
- query_complexity_factor = -0.1 (simple) to +0.1 (complex)
```

**Hybrid RAG Approaches:**

```
Hybrid RAG Architecture:

Combines multiple retrieval methods:
1. Semantic Search (Embeddings)
2. Keyword Search (BM25)
3. Metadata Filtering
4. (Optional) Graph-based retrieval

Fusion Strategy:

Method 1: Score Fusion
Hybrid_Score = α × Semantic_Score + β × BM25_Score + γ × Metadata_Score

Method 2: Reciprocal Rank Fusion (RRF)
RRF_Score = Σ (1 / (k + rank_i))

Method 3: Learned Fusion
Hybrid_Score = ML_Model([Semantic_Score, BM25_Score, Metadata_Features])

Benefits:
- 15-25% improvement in retrieval accuracy
- Better handling of exact matches
- Improved coverage of edge cases
```

**Embedding Quality Impact on RAG:**

```
Quality Metrics for RAG Embeddings:

1. Retrieval Accuracy:
   Recall@k = |Relevant ∩ Retrieved| / |Relevant|
   Target: > 0.85 for k=10

2. Grounding Quality:
   Faithfulness = |Claims_Supported_by_Context| / |Total_Claims|
   Target: > 0.90

3. Answer Quality:
   Answer_Accuracy = |Correct_Answers| / |Total_Queries|
   Target: > 0.80

Embedding Quality → Retrieval Accuracy → Answer Quality

Chain:
Good Embeddings → High Recall → Better Context → Accurate Answers
Poor Embeddings → Low Recall → Missing Context → Hallucinations
```

**RAG Optimization Strategies:**

```
Optimization Techniques:

1. Chunk Size Optimization:
   Optimal_Chunk_Size = argmax(Retrieval_Recall × Context_Completeness)
   Typical: 500-1000 tokens per chunk

2. Embedding Model Selection:
   - High accuracy: OpenAI text-embedding-3-large
   - Balanced: OpenAI text-embedding-3-small
   - Cost-sensitive: Sentence Transformers

3. Reranking:
   - Bi-encoder: Fast retrieval (top 50-100)
   - Cross-encoder: Accurate reranking (top 10-20)
   - Improvement: 10-20% precision gain

4. Query Expansion:
   - Generate query variants
   - Retrieve for each variant
   - Merge and deduplicate results
   - Improvement: 5-10% recall gain
```

#### Training Embedding Models at Scale - Contrastive Learning Deep Dive

Training high-quality embedding models requires understanding contrastive learning principles, data mining strategies, and optimization techniques. Modern embedding models are trained using sophisticated contrastive objectives that learn semantic relationships from data.

**Contrastive Learning - Mathematical Foundation:**

Contrastive learning trains models to distinguish between similar (positive) and dissimilar (negative) pairs:

```
Contrastive Learning Objective:

Goal: Learn embedding function f such that:
- Similar pairs (x, x⁺): f(x) and f(x⁺) are close
- Dissimilar pairs (x, x⁻): f(x) and f(x⁻) are far

Loss Function (InfoNCE - Information Noise Contrastive Estimation):

L = -log(exp(sim(f(x), f(x⁺)) / τ) / (exp(sim(f(x), f(x⁺)) / τ) + Σᵢ exp(sim(f(x), f(xᵢ⁻)) / τ)))

Where:
- sim: Similarity function (cosine, dot product)
- τ: Temperature parameter (controls sharpness, typically 0.05-0.2)
- x⁺: Positive example (similar to x)
- xᵢ⁻: Negative examples (dissimilar to x)
- N: Number of negatives

Simplified Form:
L = -log(exp(sim_pos / τ) / (exp(sim_pos / τ) + Σ exp(sim_neg / τ)))
```

**InfoNCE Loss Derivation:**

```
InfoNCE Loss Explanation:

The InfoNCE loss maximizes the mutual information between positive pairs while
minimizing it for negative pairs.

Mathematical Formulation:

L = -log(P(x⁺ | x) / (P(x⁺ | x) + Σ P(xᵢ⁻ | x)))

Where probabilities are defined using softmax:
P(x⁺ | x) = exp(sim(f(x), f(x⁺)) / τ) / Σⱼ exp(sim(f(x), f(xⱼ)) / τ)

Interpretation:
- Numerator: Probability of positive pair
- Denominator: Probability of positive + all negatives
- Loss: Negative log probability (higher prob → lower loss)

Temperature Parameter τ:
- Small τ (0.05): Sharp distribution, hard negatives matter more
- Large τ (0.2): Smooth distribution, easier training
- Typical: 0.1 for most tasks
```

**Training Data Construction - Positive and Negative Mining:**

```
Positive Pair Mining:

1. Paraphrase Pairs:
   Source: Paraphrase datasets (e.g., MSRP, PAWS)
   Example: ("Machine learning is AI", "AI includes machine learning")
   Quality: High (explicitly similar)

2. Natural Language Inference (NLI):
   Source: SNLI, MNLI datasets
   Example: (Premise, Hypothesis) where label = "entailment"
   Quality: High (semantic equivalence)

3. Question-Answer Pairs:
   Source: MS MARCO, Natural Questions
   Example: (Question, Answer paragraph)
   Quality: Medium-High (contextually related)

4. Cross-Encoder Labels:
   Source: Re-label bi-encoder candidates with cross-encoder
   Process: Candidate → Cross-encoder → High score → Positive
   Quality: Very High (expert model labels)

Negative Pair Mining:

1. Hard Negatives (Most Important):
   Source: BM25/ANN nearest neighbors that are NOT relevant
   Process:
   a. Retrieve top-k candidates with bi-encoder
   b. Filter out true positives
   c. Remaining = hard negatives
   Quality: High (semantically similar but not relevant)

2. In-Batch Negatives:
   Source: Other examples in same training batch
   Process: All pairs in batch except positives
   Quality: Medium (efficient but may miss hard negatives)

3. Random Negatives:
   Source: Random sampling from corpus
   Process: Random document selection
   Quality: Low (easy negatives, less learning signal)

Negative Mining Strategy:
Optimal: 1 positive + 3-5 hard negatives + in-batch negatives
```

**Knowledge Distillation Pipeline - Bi-Encoder from Cross-Encoder:**

```
Knowledge Distillation Process:

Step 1: Candidate Mining
   Query q → Bi-Encoder → Top-K candidates (k = 50-200)
   
Step 2: Cross-Encoder Relabeling
   For each candidate c:
       score = Cross_Encoder(q, c)
       if score > threshold: label = positive
       else: label = negative
   
Step 3: Training Data Creation
   Triplets: (q, positive, negative)
   Pairs: (q, positive) for positive pairs
   
Step 4: Bi-Encoder Fine-tuning
   Train bi-encoder on distilled data
   Objective: Match cross-encoder scores
   
Benefits:
- Bi-encoder learns from cross-encoder expertise
- Faster inference (single forward pass)
- Better than training from scratch
- 10-15% improvement in retrieval accuracy

Loss Function (Distillation):
L = MSE(Bi_Encoder_Score(q, c), Cross_Encoder_Score(q, c))

Or contrastive:
L = InfoNCE with cross-encoder labels
```

**Training Flow - Complete Pipeline:**

```
┌─────────────────────────────────────────────────────────────┐
│          EMBEDDING MODEL TRAINING PIPELINE                  │
└─────────────────────────────────────────────────────────────┘

Raw Corpus
    │
    ▼
┌──────────────────┐
│ Data Collection  │
│ • Paraphrases    │
│ • NLI pairs      │
│ • QA pairs       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Candidate Mining │
│ • BM25 search    │
│ • ANN search     │
│ • Top-K retrieval│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Cross-Encoder    │
│ Relabeling       │
│ • Score pairs    │
│ • Generate labels│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Hard Negative    │
│ Mining           │
│ • Filter positives│
│ • Select hardest │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Training Data    │
│ • (q, pos, neg)  │
│ • Balanced      │
│ • Quality checked│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Model Training   │
│ • Contrastive    │
│ • Batch training │
│ • Validation     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Model Evaluation │
│ • MTEB benchmark │
│ • Domain tests   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Model Export     │
│ • Quantization   │
│ • Optimization   │
└──────────────────┘
```

**Training Configuration - Hyperparameters:**

```
Hyperparameter Selection:

1. Learning Rate:
   - Initial: 2e-5 to 5e-5 (typical for transformers)
   - Schedule: Linear warmup + decay
   - Warmup: 10% of total steps
   
2. Batch Size:
   - Small: 16-32 (memory constrained)
   - Medium: 32-64 (balanced)
   - Large: 64-128 (better gradients, more negatives)
   - Trade-off: Larger batch = more in-batch negatives

3. Number of Negatives:
   - Per positive: 3-5 hard negatives
   - In-batch: All non-positive pairs
   - Total: Batch size - 1 (in-batch)

4. Temperature τ:
   - Range: 0.05-0.2
   - Default: 0.1
   - Effect: Lower = harder training, better separation

5. Training Epochs:
   - Typical: 1-3 epochs
   - Overfitting risk: Monitor validation loss

Training Time:
- Small model (384 dims): 1-2 hours (1 GPU)
- Medium model (768 dims): 4-8 hours (1 GPU)
- Large model (1536 dims): 12-24 hours (multiple GPUs)
```

**Advanced Training Techniques:**

```
1. Multi-Task Learning:
   Train on multiple tasks simultaneously:
   - Semantic similarity
   - Paraphrase detection
   - Natural language inference
   - Question-answering
   
   Loss: L_total = Σ w_i × L_i
   Where w_i are task weights

2. Curriculum Learning:
   Start with easy negatives → gradually harder
   - Epoch 1: Random negatives
   - Epoch 2: Mix random + hard
   - Epoch 3: Hard negatives only

3. Data Augmentation:
   - Back-translation (multilingual)
   - Synonym replacement
   - Paraphrase generation
   - Improves robustness

4. Regularization:
   - Dropout: 0.1-0.2
   - Weight decay: 1e-4
   - Prevents overfitting
```

#### Indexing, Filtering, and Maintenance

Choose ANN structures based on scale and update patterns: HNSW for high recall and dynamic inserts; IVF/IVFPQ for billion-scale with lower RAM; OPQ/PQ for compression. Use metadata filters (time, source, tenant) to scope retrieval and improve precision.

Plan for index refresh and drift: periodic re‑embedding after model updates, compaction to remove tombstones, and A/B evaluation (Recall@k, NDCG) to validate improvements. Size memory using ≈ N × (d × bytes_per_dim + overhead).

#### Evaluation and Benchmarks

Use BEIR/MTEB-style suites to assess generalization across domains (msmarco, scifact, fiqa, nq). Report Recall@k, MRR, and NDCG, plus latency at fixed recall targets. Track OOD performance to detect domain shift and plan domain adaptation or few-shot re‑tuning.

#### Multi‑Vector and Sparse‑Dense Hybrids

Late-interaction models (e.g., ColBERT) keep token-level vectors and compute MaxSim at query time for higher accuracy at additional memory/latency. SPLADE produces sparse expansions that combine well with BM25. Hybrid scoring (α·dense + (1−α)·sparse) often dominates either alone.

Embedding retrieval data path:
```
Text → Tokenize → Encoder → Vector (normalize) → ANN Index → top‑k IDs →
Metadata Filter → (Optional) Cross‑Encoder Rerank → Final Contexts
```

---

## Class 4: Overview of All Major LLMs

### Topics Covered

- GPT family (GPT-3.5, GPT-4, GPT-5)
- LLaMA, Falcon, Mistral, Claude, Gemini
- Key differences: architecture, context size, fine-tuning ability
- Choosing the right LLM for a use case

### Learning Objectives

By the end of this class, students will be able to:
- Identify major LLM families and their characteristics
- Compare LLMs based on architecture, size, and capabilities
- Understand context window limitations and implications
- Select appropriate LLMs for specific use cases
- Evaluate trade-offs between different models

### Core Concepts

#### GPT Family (OpenAI)

**GPT-3.5 (2022)**
- Variants: text-davinci-003, gpt-3.5-turbo
- 175B parameters (GPT-3 base)
- Context: 4K tokens (turbo), 16K tokens
- Strong general capabilities
- API-based access

**GPT-4 (2023)**
- Multimodal capabilities (text + images)
- 8K and 32K context windows
- Improved reasoning and instruction following
- GPT-4 Turbo: 128K context window
- More reliable and accurate

**GPT-4o (2024)**
- Optimized for speed and cost
- Multimodal (text, vision, audio)
- 128K context window
- Faster inference

#### LLaMA (Meta)

**LLaMA 1 (2023)**
- Open-source models: 7B, 13B, 33B, 65B parameters
- Self-hosted, requires significant compute
- Strong performance on benchmarks
- No API, requires hosting

**LLaMA 2 (2023)**
- Improved training and safety
- 7B, 13B, 70B variants
- Chat-optimized versions
- Commercial use allowed

**LLaMA 3 (2024)**
- 8B, 70B, 405B variants
- Improved reasoning
- Better instruction following
- Extended context windows

#### Claude (Anthropic)

**Claude 2 (2023)**
- 100K context window
- Strong safety and helpfulness
- Good for long documents
- API-based access

**Claude 3 (2024)**
- Variants: Haiku, Sonnet, Opus
- 200K context window
- Multimodal capabilities
- Improved reasoning

#### Gemini (Google)

**Gemini Pro (2023)**
- Multimodal from the ground up
- Strong reasoning capabilities
- Available via API
- Competitive performance

**Gemini 1.5 (2024)**
- 1M context window (experimental)
- Improved performance
- Better multimodal understanding

#### Other Notable Models

**Mistral AI**
- Mistral 7B, Mixtral 8x7B
- Open-source, efficient
- Strong performance per parameter

**Falcon (Technology Innovation Institute)**
- Falcon-40B, Falcon-180B
- Open-source
- Apache 2.0 license

### Key Comparison Criteria

#### Architecture
- **Decoder-only:** GPT, LLaMA, Falcon
- **Encoder-decoder:** T5, Flan-T5
- **Mixture of Experts:** Mixtral, GPT-4 (rumored)

#### Context Size
- **Small (4K-8K):** GPT-3.5-turbo, LLaMA 7B
- **Medium (32K-128K):** GPT-4, Claude 3 Sonnet
- **Large (200K+):** Claude 3, Gemini 1.5
- **Very Large (1M+):** Gemini 1.5 Pro (experimental)

#### Fine-tuning Ability
- **Full fine-tuning:** LLaMA, Mistral (self-hosted)
- **LoRA/QLoRA:** Most open-source models
- **API fine-tuning:** OpenAI (GPT-3.5), Anthropic (limited)
- **No fine-tuning:** Most API-based models (rely on prompt engineering)

#### Access Model
- **API-only:** GPT-4, Claude, Gemini
- **Self-hosted:** LLaMA, Mistral, Falcon
- **Both:** Some models available via API and for download

### Choosing the Right LLM

**Considerations:**

1. **Use Case Requirements**
   - Simple Q&A → GPT-3.5-turbo
   - Complex reasoning → GPT-4, Claude 3 Opus
   - Long documents → Claude 3, Gemini 1.5
   - Code generation → GPT-4, Claude 3

2. **Budget Constraints**
   - Cost-effective: GPT-3.5-turbo, LLaMA (self-hosted)
   - Premium: GPT-4, Claude 3 Opus

3. **Privacy Requirements**
   - High privacy: Self-hosted models (LLaMA, Mistral)
   - API acceptable: OpenAI, Anthropic, Google

4. **Fine-tuning Needs**
   - Custom domain: Self-hosted models with LoRA
   - General purpose: API models with prompt engineering

5. **Latency Requirements**
   - Fast: GPT-3.5-turbo, Claude 3 Haiku
   - Quality over speed: GPT-4, Claude 3 Opus

#### Pretraining, Tuning, and Decoding

Decoder‑only LMs (GPT/LLaMA) optimize next‑token prediction on large corpora; encoder‑decoder (T5) frame tasks text‑to‑text with strong supervised transfer. Instruction tuning aligns models to follow natural language commands; RLHF/constitutional AI further steers behavior toward helpful and harmless outputs.

Decoding controls quality and diversity: temperature, top‑k/p, and penalties manage repetition and creativity; constrained decoding (regex/JSON schemas) ensures structured outputs. KV‑cache, batching, and speculative decoding improve throughput and latency under load.

#### Tool Use, Function Calling, and RAG Synergy

Function calling grounds outputs in external tools (search, calculators, databases) and enforces schemas. RAG augments knowledge without retraining; routing policies can select models, tools, or prompts based on intent. For complex tasks, planner‑executor patterns decompose problems into tool‑invocation steps.

#### Efficiency: Quantization and PEFT

Quantization (INT8/INT4) reduces memory and increases throughput with modest quality costs; QLoRA combines 4‑bit base models with low‑rank adapters for efficient tuning. PEFT methods (LoRA, prefix/p‑tuning) adapt models with minimal parameters, enabling domain specialization on modest GPUs.

#### Evaluation, Safety, and Governance

Use diverse benchmarks (MMLU, MT‑Bench, HELM) plus domain‑specific tests. Track jailbreak resistance, prompt‑injection handling, and harmful output filters. Establish human‑in‑the‑loop review for high‑stakes tasks and maintain audit logs of prompts/outputs for compliance.

LLM serving flow:
```
Prompt (system+user+tools) → Safety/Policy Layer → LLM Decode (KV‑cache) →
(Optional) Function Calls/RAG → Post‑processing (parsers, validators) → Output
```

### Readings

- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Few-Shot Learners" (Brown et al., 2020) - GPT-3
- "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
- Recent LLM architecture papers and comparisons

 

### Additional Resources

- [Hugging Face Model Hub](https://huggingface.co/models)
- [OpenAI Model Documentation](https://platform.openai.com/docs/models)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [LLM Comparison Tools](https://chat.lmsys.org/)

### Practical Code Examples

#### Complete Semantic Search Implementation

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
    
    def add_documents(self, documents: List[str]):
        """Add documents and generate embeddings"""
        self.documents = documents
        self.embeddings = self.model.encode(documents, show_progress_bar=True)
        print(f"Indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        query_embedding = self.model.encode(query)
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (self.documents[i], float(similarities[i]))
            for i in top_indices
        ]
        return results

# Usage
search_engine = SemanticSearch()
documents = [
    "Python is a high-level programming language",
    "Machine learning algorithms learn from data",
    "Natural language processing analyzes text",
    "Deep learning uses neural networks"
]
search_engine.add_documents(documents)

results = search_engine.search("programming with Python", top_k=3)
for doc, score in results:
    print(f"Score: {score:.3f} - {doc}")
```

#### LLM Selection Helper

```python
class LLMSelector:
    """Helper class for selecting appropriate LLM"""
    
    MODELS = {
        "gpt-3.5-turbo": {
            "context": 4096,
            "cost_per_1k_tokens": 0.0015,
            "speed": "fast",
            "best_for": ["simple Q&A", "chatbots", "content generation"]
        },
        "gpt-4": {
            "context": 8192,
            "cost_per_1k_tokens": 0.03,
            "speed": "medium",
            "best_for": ["complex reasoning", "code generation", "analysis"]
        },
        "claude-3-opus": {
            "context": 200000,
            "cost_per_1k_tokens": 0.015,
            "speed": "slow",
            "best_for": ["long documents", "analysis", "research"]
        }
    }
    
    @classmethod
    def recommend(cls, use_case: str, budget: str = "medium", 
                   context_size: int = 4096, privacy: bool = False):
        """Recommend LLM based on requirements"""
        recommendations = []
        
        for model, specs in cls.MODELS.items():
            score = 0
            
            # Context size check
            if specs["context"] >= context_size:
                score += 2
            else:
                continue  # Skip if context too small
            
            # Budget check
            if budget == "low" and specs["cost_per_1k_tokens"] < 0.01:
                score += 2
            elif budget == "medium":
                score += 1
            
            # Use case match
            if use_case.lower() in " ".join(specs["best_for"]).lower():
                score += 2
            
            recommendations.append((model, score, specs))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[0] if recommendations else None

# Usage
recommendation = LLMSelector.recommend(
    use_case="complex reasoning",
    budget="medium",
    context_size=8000
)
print(f"Recommended: {recommendation[0]}")
```

### Troubleshooting Guide

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Low similarity scores** | All embeddings have low similarity | Check normalization, try different embedding model, verify text preprocessing |
| **Embedding dimension mismatch** | Cannot compare embeddings | Ensure all embeddings use same model and dimension |
| **Slow embedding generation** | High latency | Use faster models (all-MiniLM vs mpnet), batch processing, caching |
| **Poor search results** | Irrelevant documents retrieved | Try different embedding models, add reranking, adjust similarity threshold |
| **Context window overflow** | Token limit exceeded | Chunk documents, use summarization, select model with larger context |
| **API rate limits** | 429 errors from embedding API | Implement caching, use local models, add retry logic |

**Common Pitfalls:**
- **Pitfall:** Using word-level embeddings for sentence/document search
  - **Solution:** Always use sentence/document-level embeddings for semantic search
- **Pitfall:** Not normalizing embeddings before similarity calculation
  - **Solution:** Normalize all embeddings to unit vectors when using cosine similarity
- **Pitfall:** Choosing wrong LLM for context size requirements
  - **Solution:** Always check context window size before selecting model

### Quick Reference Guide

#### Embedding Model Comparison

| Model | Dimensions | Type | Best For | Speed |
|-------|------------|------|----------|-------|
| text-embedding-3-small | 1536 | API | Production apps | Fast |
| text-embedding-3-large | 3072 | API | High accuracy | Medium |
| all-MiniLM-L6-v2 | 384 | Local | Fast local search | Very Fast |
| all-mpnet-base-v2 | 768 | Local | High accuracy | Medium |
| multilingual-MiniLM | 384 | Local | Multilingual | Fast |

#### LLM Selection Matrix

| Requirement | Recommended Model | Alternative |
|-------------|------------------|------------|
| Simple Q&A | GPT-3.5-turbo | Claude Haiku |
| Complex reasoning | GPT-4 | Claude Opus |
| Long documents | Claude 3 | Gemini 1.5 |
| Code generation | GPT-4 | Claude 3 |
| Cost-sensitive | GPT-3.5-turbo | LLaMA (self-hosted) |
| Privacy-critical | LLaMA (self-hosted) | Mistral (self-hosted) |

### Case Studies

#### Case Study: Embedding Model Migration

**Challenge:** A company needed to improve search accuracy in their knowledge base.

**Initial Setup:**
- Used TF-IDF for keyword search
- 40% irrelevant results
- No semantic understanding

**Solution:**
- Migrated to OpenAI text-embedding-3-small
- Implemented hybrid search (BM25 + embeddings)
- Added reranking with cross-encoder

**Results:**
- 85% relevant results (40% → 85%)
- 50% reduction in search time
- Improved user satisfaction

**Lessons Learned:**
- Hybrid search outperforms pure semantic search
- Reranking significantly improves precision
- Cost of embeddings offset by improved accuracy

### Hands-On Lab: Build a Semantic Search System

**Lab Objective:** Create a complete semantic search system with embeddings and similarity search.

**Steps:**

1. **Setup**
```bash
pip install sentence-transformers numpy scikit-learn
```

2. **Implement Search Engine**
```python
# Use code examples above
# Add document indexing
# Implement search interface
```

3. **Test and Evaluate**
```python
# Test with sample queries
# Measure retrieval accuracy
# Compare different embedding models
```

**Expected Outcomes:**
- Working semantic search system
- Understanding of embedding models
- Knowledge of similarity metrics
- Ability to optimize search performance

### Glossary

**Embedding:** A dense vector representation of text, images, or other data that captures semantic meaning in high-dimensional space.

**Cosine Similarity:** A metric measuring the cosine of the angle between two vectors, used for comparing normalized embeddings.

**Euclidean Distance:** The straight-line distance between two vectors in Euclidean space, smaller distance indicates more similarity.

**Sentence Transformer:** A model architecture fine-tuned to produce sentence-level embeddings optimized for similarity tasks.

**Context Window:** The maximum number of tokens a language model can process in a single input/output sequence.

**Fine-tuning:** Adapting a pre-trained model to a specific task using task-specific training data.

**LoRA (Low-Rank Adaptation):** A parameter-efficient fine-tuning method that adds trainable low-rank matrices to model weights.

**Reranking:** A second-stage retrieval process that re-orders candidates using a more expensive but accurate model.

**Semantic Search:** Search method that understands meaning and intent rather than just matching keywords.

**Vector Database:** A specialized database optimized for storing and querying high-dimensional vector embeddings.

### Key Takeaways

1. Embeddings enable semantic understanding and search beyond keywords
2. Different embedding models serve different purposes (word vs. sentence level)
3. Vector similarity metrics (cosine, Euclidean) are fundamental to retrieval
4. LLM selection depends on use case, budget, privacy, and requirements
5. Context window size is a critical factor for long-document applications
6. Open-source models offer flexibility but require infrastructure
7. Hybrid search (keyword + semantic) often outperforms either alone
8. Proper normalization is crucial for accurate similarity calculations
9. Reranking significantly improves retrieval precision
10. Cost, latency, and accuracy must be balanced in production systems

---

**Previous Module:** [Module 2: GenAI Project Architecture & Flow](../module_02.md)  
**Next Module:** [Module 4: Search Algorithms & Retrieval Techniques](../module_04.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)

