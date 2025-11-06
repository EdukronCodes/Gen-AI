# Generative AI & Agentic AI: Comprehensive Course Syllabus

**Course Code:** CS 4500  
**Credit Hours:** 3  
**Prerequisites:** Machine Learning, Deep Learning, Python Programming  
**Instructor:** [Instructor Name]  
**Office Hours:** [Schedule]  
**Email:** [Email Address]

---

## Course Description

This course provides a comprehensive introduction to Generative AI and Agentic AI systems, covering foundational concepts, architectures, frameworks, and practical applications. Students will explore Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), transformer architectures, fine-tuning techniques, and advanced agentic systems. The course emphasizes hands-on implementation using modern frameworks like LangChain, LlamaIndex, and vector databases.

---

## Learning Objectives

Upon completion of this course, students will be able to:

1. Understand the evolution from traditional AI to Generative AI and Agentic AI
2. Design and implement RAG (Retrieval-Augmented Generation) systems
3. Work with embedding models and vector databases for semantic search
4. Understand and implement transformer architectures
5. Fine-tune LLMs using techniques like LoRA, PEFT, and QLoRA
6. Build agentic AI systems with reasoning, planning, and tool use capabilities
7. Deploy generative AI applications using modern frameworks and APIs
8. Evaluate and optimize generative AI systems using appropriate metrics

---

## Course Structure

The course is divided into **12 modules** covering **20 classes** plus a capstone project.

---

## Course Outline

### Module 1: Foundations of Generative & Agentic AI

**Class 1: Introduction to Generative AI & Agentic AI**

- What is Generative AI?
- Evolution from Traditional AI → Generative AI → Agentic AI
- Core components: Foundation Models, Embeddings, Context Windows
- Agentic AI concepts: Reasoning, Planning, Tool Use, Memory
- Real-world use cases and architecture overview

**Readings:**
- Recent survey papers on Generative AI
- Foundation model papers

**Assignments:**
- Setup development environment
- Literature review on Generative AI applications

---

### Module 2: GenAI Project Architecture & Flow

**Class 2: Generative AI Project Flow**

- Problem framing & data preparation
- Text generation, summarization, and chatbots
- System components: LLM, Vector DB, Retriever, Frontend
- Example: End-to-End RAG (Retrieval-Augmented Generation) pipeline overview

**Readings:**
- RAG system design papers
- Industry case studies

**Assignments:**
- Design a RAG system architecture for a specific use case

---

### Module 3: Representations & Search Algorithms

**Class 3: Embedding Models**

- What are embeddings?
- Types: Word2Vec, GloVe, Sentence Transformers, OpenAI Embeddings
- Vector similarity (cosine, Euclidean)
- Use in semantic search and RAG systems

**Class 4: Overview of All Major LLMs**

- GPT family (GPT-3.5, GPT-4, GPT-5)
- LLaMA, Falcon, Mistral, Claude, Gemini
- Key differences: architecture, context size, fine-tuning ability
- Choosing the right LLM for a use case

**Readings:**
- "Attention Is All You Need" (Vaswani et al., 2017)
- Recent LLM architecture papers

**Assignments:**
- Implement embedding-based semantic search
- Compare different LLM APIs and their capabilities

---

### Module 4: Search Algorithms & Retrieval Techniques

**Class 5: Search Algorithms — Fundamentals**

- Keyword search vs Semantic search
- TF-IDF & BM25 introduction
- Dense vs Sparse retrieval

**Class 6: BM25 Algorithm Deep Dive**

- Formula and parameters (k1, b)
- Implementation in Python
- Comparing BM25 vs Embedding search

**Class 7: HNSW (Hierarchical Navigable Small World)**

- Introduction to approximate nearest neighbor search
- Graph-based retrieval
- Use cases in FAISS, Milvus, Weaviate, ChromaDB

**Readings:**
- BM25 algorithm papers
- HNSW and approximate nearest neighbor search papers

**Assignments:**
- Implement BM25 search algorithm
- Compare keyword search, BM25, and embedding search performance

---

### Module 5: Frameworks for Building GenAI Applications

**Class 8: LangChain – The Core Framework**

- Core components: Chains, Agents, Tools, Memory
- Building a simple RAG pipeline
- LangChain Expression Language (LCEL)
- Integration with OpenAI and ChromaDB

**Class 9: LlamaIndex & Other Frameworks**

- Overview of LlamaIndex, differences from LangChain
- Index types: Summary, Vector, List
- Integration with LLM APIs and databases

**Readings:**
- LangChain documentation
- LlamaIndex documentation

**Assignments:**
- Build a RAG system using LangChain
- Build a RAG system using LlamaIndex
- Compare and contrast both approaches

---

### Module 6: RAG & Transformer Architecture

**Class 10: RAG (Retrieval-Augmented Generation)**

- Architecture & Workflow
- Retriever + Generator pipeline
- Evaluation of RAG systems
- Detailed notes on hybrid retrieval (BM25 + Embedding)

**Class 11: Transformer Architecture Deep Dive**

- Attention mechanism (Self, Cross, Multi-head)
- Positional encoding, residual connections
- Encoder-Decoder models: BERT, GPT, T5
- Visual walkthrough & code snippets

**Readings:**
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al.)
- "Attention Is All You Need" (Vaswani et al., 2017)

**Assignments:**
- Implement a transformer component from scratch
- Build a complete RAG system with hybrid retrieval

---

### Module 7: Tokenization & Embeddings in LLMs

**Class 12: Tokenization, Embeddings, Positional Encoding**

- Byte-Pair Encoding (BPE)
- Token limits, context window management
- Positional encodings and their mathematical intuition

**Readings:**
- BPE and tokenization papers
- Positional encoding research

**Assignments:**
- Implement BPE tokenization
- Experiment with different positional encoding methods

---

### Module 8: LLM Training & Fine-tuning

**Class 13: Pretraining and Fine-tuning LLMs**

- Pretraining objectives (Next Word, Masked LM)
- Fine-tuning vs Instruction-tuning vs RLHF
- Low-Rank Adaptation (LoRA), PEFT, QLoRA
- Hands-on: Fine-tune GPT-2 or LLaMA on custom text data

**Class 14: Model Validation Metrics**

- Perplexity, BLEU, ROUGE, METEOR, and BERTScore
- Evaluating generative model outputs
- Human evaluation and prompt effectiveness

**Class 15: Model Training Techniques**

- Batch size, learning rate, gradient clipping
- Mixed precision & quantization (INT8, 4-bit)
- GPU/TPU optimization and distributed training

**Readings:**
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al.)
- "Training language models to follow instructions" (Ouyang et al.)

**Assignments:**
- Fine-tune an LLM using LoRA on a custom dataset
- Evaluate model performance using multiple metrics

---

### Module 9: LLM Inference & Prompt Engineering

**Class 16: Inference, Prompt Engineering & Context Windows**

- Prompt templates (Zero-shot, Few-shot, Chain-of-thought)
- Context window optimization & token budgeting
- Prompt caching and reusability
- Hands-on: Design optimal prompts for a task

**Readings:**
- Prompt engineering guides
- Recent papers on in-context learning

**Assignments:**
- Prompt engineering project: optimize prompts for specific tasks
- Compare zero-shot, few-shot, and chain-of-thought prompting

---

### Module 10: Database, Frameworks & Deployment

**Class 17: ChromaDB (Vector Database) – Full Notes**

- Architecture, ingestion, querying, filtering
- Integration with LangChain / LlamaIndex
- Comparison with Pinecone, Weaviate, Milvus

**Class 18: Model Deployment with Flask / FastAPI**

- Exposing LLM or RAG as API endpoints
- Building interactive UIs with Gradio / Streamlit
- Integration with MLflow for version control

**Readings:**
- Vector database documentation
- API deployment best practices

**Assignments:**
- Set up and use a vector database for RAG
- Deploy a generative AI application as a web service

---

### Module 11: Frameworks, Libraries & Platforms Overview

**Class 19: Tools, Frameworks & Platforms**

- **Frameworks:** PyTorch, TensorFlow, Hugging Face Transformers
- **Libraries:** LangChain, LlamaIndex, Gradio, Streamlit, MLflow
- **Platforms:** Hugging Face Hub, Azure OpenAI, Vertex AI, Ollama
- **Databases:** Pinecone, ChromaDB, Weaviate, Milvus
- **Utilities:** OpenAI API, Anthropic API, Replicate, Modal

**Readings:**
- Documentation for major frameworks and platforms
- Platform comparison guides

**Assignments:**
- Experiment with different platforms and compare capabilities
- Build a project using multiple frameworks

---

### Module 12: End-to-End Agentic AI System

**Class 20: Agentic AI Systems – Advanced**

- Agent types: Reactive, Planning, Multi-Agent Systems
- Memory management: Long-term memory, episodic recall
- Tool use and reasoning chains
- Building a simple multi-agent system using LangGraph or LangChain Agents

**Readings:**
- Recent papers on agentic AI
- LangGraph documentation
- Multi-agent system research

**Assignments:**
- Build a simple agentic AI system
- Implement tool use and memory in an agent

---

## Capstone Project

Students will build a complete end-to-end Generative AI or Agentic AI application. Projects can include:

- Advanced RAG system with multiple retrieval strategies
- Multi-agent system with planning and tool use
- Custom fine-tuned LLM application
- Creative application (text-to-image, code generation, etc.)
- Production-ready deployment with monitoring

**Project Timeline:**
- Proposal: Week 10
- Mid-project update: Week 15
- Final presentation: Week 16

---

## Required Textbook

1. **Primary Text:**  
   Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

2. **Additional Resources:**
   - Research papers (provided throughout the course)
   - LangChain documentation: https://python.langchain.com/
   - LlamaIndex documentation: https://www.llamaindex.ai/
   - Hugging Face course: https://huggingface.co/learn

---

## Software and Tools

**Required Software:**
- Python 3.8+
- PyTorch or TensorFlow
- Jupyter Notebooks
- Git and GitHub

**Required Libraries:**
- LangChain
- LlamaIndex
- ChromaDB (or alternative vector database)
- Hugging Face Transformers
- OpenAI Python SDK (or Anthropic SDK)

**Recommended Tools:**
- Gradio or Streamlit (for UI)
- Weights & Biases (for experiment tracking)
- MLflow (for model versioning)

**Cloud Resources:**
- Google Colab Pro (recommended)
- AWS/Azure credits (for final projects)
- Access to LLM APIs (OpenAI, Anthropic)

---

## Grading Policy

| Component | Weight |
|-----------|--------|
| Assignments (10 assignments) | 35% |
| Midterm Exam | 20% |
| Final Project | 30% |
| Participation & Quizzes | 10% |
| Research Paper Review | 5% |

**Grading Scale:**
- A: 90-100%
- B: 80-89%
- C: 70-79%
- D: 60-69%
- F: Below 60%

---

## Assignment Details

### Weekly Assignments (35%)
- 10 programming assignments throughout the semester
- Mix of theoretical problems and practical implementations
- Due dates: Typically one week after assignment release
- Late policy: 10% deduction per day, maximum 3 days late

### Midterm Exam (20%)
- Covers Modules 1-6
- Combination of theoretical questions and coding problems
- Date: Week 10

### Final Project (30%)
- Students work individually or in teams (max 2 members)
- Build a complete Generative AI or Agentic AI application
- Topics must be approved by instructor
- Deliverables:
  - Project proposal (Week 10)
  - Mid-project update (Week 15)
  - Final report and code (Week 16)
  - Presentation (Week 16)

### Participation & Quizzes (10%)
- Weekly reading quizzes
- In-class participation
- Discussion board engagement

### Research Paper Review (5%)
- Select and review 1 recent paper on Generative AI or Agentic AI
- Write detailed analysis and critique
- Present findings to class

---

## Course Schedule

| Week | Module | Topics | Assignment Due |
|------|--------|--------|----------------|
| 1 | Module 1 | Introduction to Generative & Agentic AI | - |
| 2 | Module 2 | GenAI Project Architecture & Flow | Assignment 1 |
| 3 | Module 3 | Representations & Search Algorithms | Assignment 2 |
| 4 | Module 4 | Search Algorithms & Retrieval Techniques | Assignment 3 |
| 5 | Module 4 (cont.) | BM25, HNSW | Assignment 4 |
| 6 | Module 5 | Frameworks: LangChain & LlamaIndex | Assignment 5 |
| 7 | Module 6 | RAG & Transformer Architecture | Assignment 6 |
| 8 | Module 7 | Tokenization & Embeddings | Assignment 7 |
| 9 | Module 8 | LLM Training & Fine-tuning I | Assignment 8 |
| 10 | Module 8 (cont.) | LLM Training & Fine-tuning II | **Midterm Exam** |
| 11 | Module 9 | LLM Inference & Prompt Engineering | Assignment 9 |
| 12 | Module 10 | Databases & Deployment | Assignment 10 |
| 13 | Module 11 | Frameworks & Platforms Overview | Project Proposal |
| 14 | Module 12 | Agentic AI Systems | Mid-project Update |
| 15 | Review | Advanced Topics & Review | Research Review |
| 16 | Finals | Final Project Presentations | Final Project |

---

## Academic Integrity

All students are expected to adhere to the university's academic integrity policy. This includes:

- **Collaboration:** Discussion of concepts is encouraged, but assignments must be completed individually unless specified as group work
- **Code:** You may use online resources for reference, but you must understand and be able to explain all code you submit
- **AI Tools:** Use of AI coding assistants (GitHub Copilot, ChatGPT for code) must be disclosed and is allowed for learning, but not for direct assignment submission
- **Plagiarism:** Any form of plagiarism will result in a failing grade

---

## Accommodations

Students with disabilities who need accommodations should contact the Office of Disability Services and provide documentation to the instructor within the first two weeks of class.

---

## Communication

- **Email:** Use email for private matters. Include course code in subject line.
- **Office Hours:** For detailed discussions and questions
- **Discussion Forum:** For general questions that benefit the class
- **Response Time:** Instructor will respond within 48 hours

---

## Resources

### Online Learning Platforms
- Hugging Face Course on Transformers
- LangChain Documentation and Tutorials
- Fast.ai Deep Learning Course
- Stanford CS224N: NLP with Deep Learning

### Communities
- Hugging Face Discord
- LangChain Discord
- r/MachineLearning
- Papers with Code

### Research Venues
- NeurIPS, ICML, ICLR
- ACL, EMNLP (for NLP-focused work)
- AAAI (for AI systems work)

---

## Prerequisites Knowledge

Students should be comfortable with:
- Linear algebra (vectors, matrices, eigenvalues)
- Calculus (derivatives, gradients, optimization)
- Probability and statistics
- Python programming (intermediate level)
- Deep learning fundamentals (neural networks, backpropagation)
- Experience with PyTorch or TensorFlow

---

## Course Policies

1. **Attendance:** Regular attendance is expected. More than 3 unexcused absences may result in grade reduction.

2. **Late Work:** See assignment section for late policy. Exceptions only for documented emergencies.

3. **Regrading:** Requests must be made within one week of grade release.

4. **Technology:** Laptops allowed for note-taking and coding. Please be respectful during lectures.

5. **Recording:** No recording of lectures without explicit permission.

---

*Last Updated: [Date]*  
*This syllabus is subject to change. Students will be notified of any modifications.*
