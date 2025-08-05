# LLM Trust Lens - Open Intent Classification

## Overview

**LLM Trust Lens - Open Intent Classification** is a pipeline to evaluate the performance of various methods (such as LLMs) on various datasets, focusing on the topic of "Open Intent Classification".

**What is "Open Intent Classification"**

There are 2 ways to evaluate open intent classification:
1. Binary classification of open-intent/oos/unknown class vs 1 known class (grouped from all known classes)
2. Multi-class Classification of open-intent/oos/unknown class vs individual known classes


## Key Features

- **Multi-Model Support**: Evaluate both local models (via Ollama) and API-based models (Nebius, Google Gemini)
- **Flexible Prompt Scenarios**: Support for both zero-shot and few-shot prompt scenarios
- **Multiple Datasets**: Built-in support for Banking77, StackOverflow, and CLINC150OOS TSV datasets (Source: [2021 Adaptive Decision Boundary Clustering GitHub repo](https://github.com/thuiar/Adaptive-Decision-Boundary/tree/main/data)). For new datasets, bring them into the pipeline!
- **Configurable Experiments**: YAML-based configuration system for easy experiment setup
- **Traceable Results**: Generate LLM predictions, classification metrics and confusion matrix files for evaluation


## Usage

```python
!python3 /workspaces/llm-trust-lens/main.py
```


## License  
MIT License.  



