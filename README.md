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


## Folder Structure
```bash
.
├── LICENSE
├── README.md
├── .env                               # File containing API keys. Follow the "Setup" section to create your own .env file
├── config
│   ├── dataset                        # Dataset config. Naming convention: <dataset_name>.yaml
│   │   ├── banking77.yaml
│   │   ├── clinc150oos.yaml
│   │   └── stackoverflow.yaml
│   └── experiment                     # Experiment config. Naming convention: <ollama/api>_<modelprovider>_<modelname>_<method>_<dataset>.yaml
│       ├── api_google_gemini-2.5-flash-preview-05-20_fewshot_banking77.yaml
│       ├── api_nebius_qwen3-30b-a3b_fewshot_banking77.yaml
│       ├── ollama_deepseek-r1_7b_zeroshot_banking77.yaml
│       ├── ollama_gemma3_4b-it-qat_zeroshot_banking77.yaml
│       ├── ollama_llama3_2_3b_fewshot_banking77.yaml
│       ├── ollama_llama3_2_3b_fewshot_banking77_thresholdtest.yaml
│       ├── ollama_llama3_2_3b_fewshot_clinc150oos.yaml
│       ├── ollama_llama3_2_3b_fewshot_stackoverflow.yaml
│       ├── ollama_llama3_2_3b_zeroshot_banking77.yaml
│       ├── ollama_mistral_7b_zeroshot_banking77.yaml
│       ├── ollama_qwen3_8b_zeroshot_banking77.yaml
│       └── ollama_tulu3_8b_zeroshot_banking77.yaml
├── data                              # Place your datasets here - 1 folder per dataset
│   ├── banking                       # Train, dev, test TSV files, including class index to label mapping (idx2label.csv). Visit "analyse-results-zeroshot-fewshot, create-idx2label.ipynb" to understand how to create idx2label.csv for your dataset
│   │   ├── banking77_idx2label.csv
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   └── train.tsv
│   ├── oos
│   └── stackoverflow
├── debugger                          # Optional debugger directory to veryify ollama has been setup correctly
│   └── debug_ollama.py
├── examples
│   ├── example_usage.py
│   ├── few_shot_examples             # Fewshot prompt notebook examples
│   │   ├── 01i1-openintent-ollama-llama3-2-3b-banking77-fewshot_5hardcoded-previouslymisclassifiedexamples.ipynb
│   │   ├── 01j1-openintent-ollama-llama3-2-3b-banking77_10001_13082.ipynb
│   │   ├── 01j2-openintent-ollama-llama3-2-3b-stackoverflow_10001_19999.ipynb
│   │   ├── 01j3-openintent-ollama-llama3-2-3b-oos_22001_23699.ipynb
│   │   ├── 01l1-openintent-gemini-2.5-flash-banking77_0_19.ipynb
│   │   ├── 01l5-openintent-nebiusqwen-banking77-individualAPIcall.ipynb
│   │   ├── 01l6-openintent-nebiusqwen-banking77-batchof10.ipynb                          # Nebius qwen batch API - run for batch of 10 examples
│   │   ├── 01l7a-openintent-nebiusqwen-banking77-batchfull-n-downloadresults.ipynb       # Nebius qwen batch API - run for full banking77 dataset, then download results
│   │   ├── 01l7b-openintent-nebiusqwen-bk77-stitchresults.ipynb                          # Nebius qwen batch API - stitch results together with original dataframe for further analysis
│   │   ├── 01l8a-openintent-nebiusqwen-stackoverflow-batchfull-n-downloadresults.ipynb
│   │   ├── 01l8b-openintent-nebiusqwen-stkoflw-stitchresults.ipynb
│   │   ├── 01l9a-openintent-nebiusqwen-clincoos-batchfull.ipynb
│   │   ├── 01l9b-openintent-nebiusqwen-clincoos-downloadresults.ipynb
│   │   └── 01l9c-openintent-nebiusqwen-c150oos-stitchresults.ipynb
│   ├── thresholdtest                # Fewshot threshold test notebook examples
│   │   ├── 01k1-openintent-ollama-llama3-2-3b-banking77-1notoos.ipynb
│   │   ├── 01k1-openintent-ollama-llama3-2-3b-banking77-4notoos.ipynb
│   │   ├── 01k2-openintent-ollama-llama3-2-3b-stackoverflow-5notoos.ipynb
│   │   └── 01k3-openintent-ollama-llama3-2-3b-clinc150oos-14notoos.ipynb
│   └── zero_shot_examples             # Zeroshot prompt notebook examples
│       ├── 01e-kaggle-ollama-llama3-2-3b-banking77-w-force-oos-no-pydantic-enums.ipynb
│       ├── 01f-kaggle-ollama-llama3-2-3b-banking77-w-pydantic-schema.ipynb
│       ├── 01g1-kaggle-ollama-llama3-2-3b-banking77-no-oos-in-intentlist-keep-oos-in-enums.ipynb
│       ├── 01g2-kaggle-ollama-llama3-2-stackoverflow-no-oos-in-intentlist-keep-oos-in-enums.ipynb
│       ├── 01g3-kaggle-ollama-llama3-2-clinc150oos-no-oos-in-intentlist-keep-oos-in-enums.ipynb
│       ├── 01g4-openintent-ollama-deepseek-r1-7b-banking77-test-reasoningmodel.ipynb
│       ├── 01g5-openintent-ollama-gemma3-4b-it-qat-banking77-test-generalquantisedmodel.ipynb
│       ├── 01g6-openintent-ollama-qwen3-8b-banking77-test-mixtureofexpertmodel.ipynb
│       ├── 01g7-openintent-ollama-mistral-7b-banking77-test-generalmodel.ipynb
│       ├── 01g8-openintent-ollama-tulu3-8b-banking77-test-instructiontunedmodel.ipynb
│       └── 01h1-openintent-ollama-llama3-2-3b-banking77-group4similarclassesinoos-zeroshot.ipynb
├── prompts
│   ├── archive_zeroshot_fewshot        # Archived zeroshot, fewshot prompts
│   │   ├── fewshot_prompt_with_5hardcoded-previouslymisclassifiedexamples.txt
│   │   ├── zeroshot_prompt_with_oos_in_intentlist.txt
│   │   └── zeroshot_prompt_with_oos_in_intentlist_w_anchor_confidence.txt
│   ├── few_shot_examples               # Create file containing fewshot prompt examples for each dataset. Visit "analyse-results-zeroshot-fewshot, create-idx2label.ipynb" to understand how to create these files
│   │   ├── banking77
│   │   │   ├── banking_25perc_oos.txt
│   │   │   ├── banking_only1notoos.txt
            ...
│   │   │   └── banking_only70notoos.txt
│   │   ├── oos
│   │   │   ├── oos_25perc_oos.txt
│   │   │   ├── oos_1notoos.txt
            ...
│   │   │   ├── oos_100notoos.txt
│   │   └── stackoverflow
│   │       ├── stackoverflow_25perc_oos.txt
│   │       ├── stackoverflow_only1notoos.txt
            ...
│   │       ├── stackoverflow_only18notoos.txt
│   ├── fewshot_prompt.txt                                       # Last used fewshot prompt
│   └── zeroshot_prompt_without_oos_in_intentlist.txt            # Last used zeroshot prompt
├── requirements.txt                                             # Python libraries to install
├── results                                                      # Folder containing experiment results
│   ├── analysis                                                 # Folder containing analysis of zeroshot, fewshot, threshold-test
│   │   ├── analyse-results-fewshot-threshold-test.ipynb
│   │   └── analyse-results-zeroshot-fewshot, create-idx2label.ipynb
│   ├── banking77_fewshot_google_gemini-2.5-flash-preview-05-20
│   ├── banking77_fewshot_llama3.2_3b                                    # In each experiment folder
│   │   ├── classification_report_llama3.2_3b_banking.txt                # Multi-class classification report (OOS vs individual known classes)
│   │   ├── classification_report_llama3.2_3b_banking_open_vs_known.txt  # Binary classification report (OOS/Open vs known class)
│   │   ├── cm_llama3.2_3b_banking.csv                                   # Multi-class classification's confusion matrix (OOS vs individual known classes) in CSV
│   │   ├── cm_llama3.2_3b_banking.png                                   # Multi-class classification's confusion matrix (OOS vs individual known classes) in PNG
│   │   ├── metrics_llama3.2_3b_banking.txt                              # Multi-class classification metrics (OOS vs individual known classes)
│   │   ├── metrics_llama3.2_3b_banking_open_vs_known.txt                # Binary classification metrics (OOS/Open vs known class)
│   │   └── results_llama3.2_3b_banking_0_7.json                         # Results JSON: list of dictionaries, with 1 dictionary per classified example
│   ├── banking77_fewshot_nebiusqwen3-30b-a3b
│   ├── banking77_fewshot_thresholdtest
│   ├── banking77_fewshot_thresholdtest_only1notoos
│   ├── banking77_fewshot_thresholdtest_only4notoos
│   ├── banking77_zeroshot_deepseek-r1_7b
│   ├── banking77_zeroshot_gemma3_4b-it-qat
│   ├── banking77_zeroshot_llama_3.2_3b
│   ├── banking77_zeroshot_mistral_7b
│   ├── banking77_zeroshot_qwen3_8b
│   ├── banking77_zeroshot_tulu3_8b
│   ├── oos_fewshot_llama3.2_3b
│   ├── round1                                                           # round1 to round9: Results from experiments conducted in Jupyter notebooks
│   ├── round2-force_oos
│   ├── round3-pydantic
│   ├── round4-oos-out-of-prompt
│   ├── round5-other-models
│   ├── round6-groupsimilarclasses-n-fewshot
│   ├── round7-fewshot-5examplesperknownintent
│   ├── round8-fewshot-1exampleeach-k-knownintent-restoos-100oossentences
│   ├── round9-fewshot-nebiusqwen
│   └── stackoverflow_fewshot_llama3.2_3b
└── src                                     # Folder containing files to run for each experiment
    ├── data_utils.py                       # Preprocess dataset
    ├── ollama_utils.py                     # Ollama model setup
    ├── nebius_utils.py                     # Nebius API setup
    ├── google_utils.py                     # Google API setup
    ├── experiment_common.py                # Common experiment file used by the Ollama and API pipeline
    ├── experiment_ollama.py                # Ollma experiment file
    └── experiment_api.py                   # API experiment file

```

## Usage

```python
!python3 /workspaces/llm-trust-lens/main.py
```


## License  
MIT License.  



