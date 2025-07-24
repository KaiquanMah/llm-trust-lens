# llm-trust-lens
Open Intent Classification [Before Merging into Final Branch]

# LLM-TrustLens  

**Comprehensive Evaluation Framework for LLM Trustworthiness**  

## Overview  
LLM-TrustLens is an open-source Python package for evaluating the **trustworthiness, reliability, and performance** of Large Language Models (LLMs). It provides a modular framework to assess **embedding quality, fairness, retrieval effectiveness, robustness, and bias** using a variety of industry-standard and novel evaluation techniques.  

## Features  

### 🏛 **Trust & Reliability Evaluation**  
- **Groundedness & Faithfulness Testing** – Ensure generated responses align with retrieved evidence.  
- **Hallucination Detection** – Identify content drift and factual inconsistencies in model outputs.  
- **Robustness Analysis** – Evaluate how well LLMs handle adversarial prompts and distribution shifts.  

### 🎭 **Bias & Fairness Assessment**  
- **Embedding Bias Detection** – Analyze cosine similarity, clustering, and bias projection across demographic attributes.  
- **Benchmarking Against Standard Datasets** – Compare fairness across models using **WinoBias, StereoSet, WEAT, SEAT**.  
- **Visualization Toolkit** – Generate **heatmaps, PCA, and t-SNE** plots for fairness interpretation.  

### 🔍 **Retrieval & Embedding Quality**  
- **Retrieval Effectiveness Benchmarking** – Measure performance with **Dense Passage Retrieval (DPR) and BLINK**.  
- **Sensitivity & Variability Analysis** – Compare embeddings across architectures (BERT, GPT, Word2Vec).  
- **Dimensionality Reduction Tools** – Apply **PCA, t-SNE, UMAP** for interpretability.  

### 📊 **Comprehensive Evaluation Suite**  
- **Query Classification Analysis** – Assess LLM intent understanding using **CLINC150, SQuAD-based intent classification**.  
- **Style & Verbosity Bias Testing** – Detect preferences for verbosity, emotional tone, and authority bias.  
- **Response Scoring Metrics** – Measure LLM response quality based on coherence, fluency, and informativeness.  

### 🏗 **Modular Python Package**  
- **Pluggable Evaluation Pipelines** – Customize tests for different LLM use cases.  
- **Automated Benchmarks** – Run predefined evaluations on embedding fairness, retrieval robustness, and response quality.  
- **API Endpoints & Notebooks** – Easily integrate with production pipelines and research workflows.  

## Installation  
```bash
pip install llm-trust-lens
```

## Usage  
```python
from llm_trust_lens import TrustEvaluator
from llm_trust_lens.embedding_model_interface import EmbeddingModelInterface

class MyEmbeddingModel(EmbeddingModelInterface):
    def get_embedding(self, text: str):
        return some_embedding_function(text)  # User's implementation

    def retrieve_similar(self, query: str, top_k: int = 5):
        return some_retrieval_function(query, top_k)  # User's implementation

my_model = MyEmbeddingModel()
evaluator = TrustEvaluator(model=my_model)

report = evaluator.evaluate(trust_metrics=["bias", "retrieval", "hallucination"])
evaluator.plot_results()
```

## Roadmap  
- ✅ Bias & fairness evaluation  
- ✅ Retrieval benchmarking  
- ✅ Robustness & hallucination detection  
- 🔄 Explainability & interpretability methods  
- 🔄 Expanded benchmarking for domain-specific applications  

## Contributing  
Contributions are welcome! See our [CONTRIBUTING.md](CONTRIBUTING.md) for details.  

## License  
MIT License.  



