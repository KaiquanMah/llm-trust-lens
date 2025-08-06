# LLM Trust Lens - Open Intent Classification

## 1. Overview

**LLM Trust Lens - Open Intent Classification** is a pipeline to evaluate the performance of various methods (such as LLMs) on various datasets, focusing on the topic of "Open Intent Classification".

**What is "Open Intent Classification"?**

There are 2 ways to evaluate open intent classification:
1. Binary classification of open-intent/oos/unknown class vs 1 known class (grouped from all known classes)
2. Multi-class Classification of open-intent/oos/unknown class vs individual known classes

**Project Team Members**
* Members: [Kaiquan Mah](https://www.linkedin.com/in/kaiquan-mah), [Michael Bernovskiy](https://www.linkedin.com/in/bernovskiy), [Ruslan Shuvalov](https://www.linkedin.com/in/rsshuvalov)
* Mentor: [Liat Friedman Antwarg](https://www.linkedin.com/in/liat-antwarg-friedman-a2367b6)
* CitrusX Representatives: [Shlomit Finegold](https://www.linkedin.com/in/shlomit-finegold), [Dagan Eshar](https://www.linkedin.com/in/dagan), [Ran Emuna (left around mid-July 2025)](https://www.linkedin.com/in/ran-emuna-ba902579)


## 2. Key Features

- **Multi-Model Support**: Evaluate both local models (via Ollama) and API-based models (Nebius, Google Gemini)
- **Flexible Prompt Scenarios**: Support for both zero-shot and few-shot prompt scenarios
- **Multiple Datasets**: Built-in support for Banking77, StackOverflow, and CLINC150OOS TSV datasets (Source: [2021 Adaptive Decision Boundary Clustering GitHub repo](https://github.com/thuiar/Adaptive-Decision-Boundary/tree/main/data)). For new datasets, bring them into the pipeline!
- **Configurable Experiments**: YAML-based configuration system for easy experiment setup
- **Traceable Results**: Generate LLM predictions, classification metrics and confusion matrix files for evaluation


## 3. Setup
1. Clone the Repository
```bash
# If you have not done so, update your Ubuntu packages and install git
sudo apt update && sudo apt install git -y

git clone https://github.com/KaiquanMah/llm-trust-lens.git
cd llm-trust-lens
```

2. Create a Virtual Environment (Recommended)
```bash
# If you have not done so, install python, pip and venv on your Ubuntu machine
sudo apt install -y python3 python3-pip python3-venv

python -m venv venv
source venv/bin/activate    # On Windows use `venv\Scripts\activate`
```

3. Install Dependencies. Install Ollama, then install the required Python packages using the requirements.txt file
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Upgrade pip
pip install --upgrade pip
# Install python dependencies to run the pipeline
pip install -r requirements.txt
```

4. Test Ollama has been installed successfully
```bash
# Check Ollama version
ollama --version
# As at August 2025: ollama version is 0.9.6

ps aux | grep ollama
# codespa+    1425  0.0  0.0   7080  2048 pts/0    S+   14:21   0:00 grep --color=auto ollama
```


## 4. Environment Configuration
To use API-based models from providers like Nebius or Google, you must configure your API keys in an environment file.

Create an **.env file** and add your API Keys.
```
NEBIUS_API_KEY = "your_nebius_api_key_here"    # These variables will be loaded by the pipeline to authenticate with the respective API services.
GOOGLE_API_KEY = "your_google_api_key_here"
```


## 5. Usage

* Non-embedding methods: Note that the experiment_*.py pipeline files currently work for non-embedding methods (zero-shot prompt and few-shot prompt)
* Embedding methods: For embedding methods (finetune BERT, then run Adaptive Decision Boundary Clustering or Variational Autoencoder), the team was still exploring these methods. Please visit the workings in the *.ipynb files we will share below.

### 5.1. Non-Embedding Methods (Zero-Shot Prompt and Few-Shot Prompt)

* The Terminal commands shown below run experiments from the root directory of the project
* You can execute different experiments by using the appropriate **experiment_*.py file** and **experiment configuration file**
* If you wish to check out the terminal workings and printouts during each pipeline run, please visit the [terminal_workings folder](https://github.com/KaiquanMah/llm-trust-lens/tree/main/examples/terminal_workings)

#### 5.1.1. Common Steps for Ollama/Local Model and API Model Experiments

1. Navigate to the **llm-trust-lens folder**
2. Activate your venv virtual environment containing the required python libraries to run the pipeline
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Dataset**: Use an existing TSV dataset or bring in new datasets into the [data folder](https://github.com/KaiquanMah/llm-trust-lens/tree/main/data)
4. **idx2label**: Use an existing idx2label.csv (mapping class indexes to labels) or create a new idx2label.csv in the [respective data folder]([https://github.com/KaiquanMah/llm-trust-lens/tree/main/data](https://github.com/KaiquanMah/llm-trust-lens/tree/main/data/banking)
   * To understand how to create a new idx2label.csv, please visit [analyse-results-zeroshot-fewshot, create-idx2label.ipynb](https://github.com/KaiquanMah/llm-trust-lens/blob/main/results/analysis/analyse-results-zeroshot-fewshot%2C%20create-idx2label.ipynb), then search for the sections near the end of the workbook using the **"idx2label_to_nonoos_listlabels" function**
5. **Dataset yaml**: Use an existing dataset yaml file or create a new dataset yaml file in the [dataset yaml folder](https://github.com/KaiquanMah/llm-trust-lens/tree/main/config/dataset)
6. **Experiment yaml**: Use an existing experiment yaml file or create a new experiment yaml file in the [experiment yaml folder](https://github.com/KaiquanMah/llm-trust-lens/tree/main/config/experiment)
   * We recommend creating separate experiment yaml files to trace back to each experiment's configuration (eg ollama vs api, the model you use, zeroshot vs fewshot, thresholdtest or not)
7. **Prompt**: Use an existing zero-shot or few-shot prompt, or create a new prompt.txt in the [prompts folder](https://github.com/KaiquanMah/llm-trust-lens/tree/main/prompts)
   * Remember to move old prompts to the [archive_zeroshot_fewshot folder](https://github.com/KaiquanMah/llm-trust-lens/tree/main/prompts/archive_zeroshot_fewshot)
8. **Few-shot Prompt Examples**: If you wish to use the few-shot prompt method, please use an existing few-shot examples file, or create a new few-shot examples txt file in the [few_shot_examples folder](https://github.com/KaiquanMah/llm-trust-lens/tree/main/prompts/few_shot_examples)

#### 5.1.2. Run Ollama/Local Model Experiments

* For Ollama models, we ran the pipeline successfully for 6 models
  ```
  llama3.2:3b
  qwen3:8b (Mixture-of-Experts LLM)
  gemma3:4b-it-qat (instruction-Following and Quantised LLM)
  mistral:7b (General-Purpose LLM)
  tulu3:8b (Instruction-Following LLM)
  deepseek-r1:7b (Reasoning LLM)
  ```
  * We expect the pipeline to be able to support other models that will be published onto Ollama
  * To explore Ollama models you can use, please visit [Ollama's model directory](https://ollama.com/search)

**5.1.2.1. Zero-Shot llama3-2-3b** on Banking77 dataset
```bash
python src/experiment_ollama.py --config config/experiment/ollama_llama3_2_3b_zeroshot_banking77.yaml
```

**5.1.2.2. Zero-Shot gemma-3-4b-it-qat** on Banking77 dataset
```bash
python src/experiment_ollama.py --config config/experiment/ollama_gemma3_4b-it-qat_zeroshot_banking77.yaml
```

**5.1.2.3. Few-Shot** llama3-2-3b on **Banking77 dataset**
```bash
python src/experiment_ollama.py --config config/experiment/ollama_llama3_2_3b_fewshot_banking77.yaml
```


**5.1.2.4. Few-Shot** llama3-2-3b on **Stackoverflow dataset**
```bash
python src/experiment_ollama.py --config config/experiment/ollama_llama3_2_3b_fewshot_stackoverflow.yaml
```

**5.1.2.5. Few-Shot** llama3-2-3b on **CLINIC150OOS dataset**
```bash
python src/experiment_ollama.py --config config/experiment/ollama_llama3_2_3b_fewshot_clinc150oos.yaml
```

#### 5.1.3. Run API Model Experiments

* For API models, our pipeline currently supports individual calls to 2 API providers (where we had some credits)
  ```
  Nebius
  Google
  ```
* To understand how to use the Nebius batch API, please visit [notebooks 01l6 to 01l9*](https://github.com/KaiquanMah/llm-trust-lens/tree/main/examples/few_shot_examples) where we prepared inputs for the batch API, called it, downloaded results, then stitched with the original dataset to get the output we expect for further analysis
* To integrate with other model providers, you will need to
  * Create a <model_provider>_utils.py file using [nebius_utils.py](https://github.com/KaiquanMah/llm-trust-lens/blob/main/src/nebius_utils.py) as a template. This file covers initialising your API client, retry config, and how to work with messages
  * Add a <ModelProviderApiClient> class to [experiment_api.py](https://github.com/KaiquanMah/llm-trust-lens/blob/main/src/experiment_api.py). The class should have 2 basic functions: `initialize, predict`
* For all API models, please remember to specify your model_provider, model_name and configuration in the [experiment yaml file which we shared in section 5.1.1. Common Steps - Step 6](https://github.com/KaiquanMah/llm-trust-lens/blob/main/config/experiment/api_nebius_qwen3-30b-a3b_fewshot_banking77.yaml)


**5.1.3.1. Few-Shot Nebius Qwen API** on Banking77 dataset
```bash
python src/experiment_api.py --config config/experiment/api_nebius_qwen3-30b-a3b_fewshot_banking77.yaml
```

**5.1.3.2. Few-Shot Google Gemini API** on Banking77 dataset
```bash
python src/experiment_api.py --config config/experiment/api_google_gemini-2.5-flash-preview-05-20_fewshot_banking77.yaml
```



### 5.2. Embedding-based Methods (Adaptive Decision Boundary Clustering and Variational Autoencoder)
TBC - to add to README after cleaning up, rerunning and checking notebooks



## 6. Folder Structure
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
│   ├── thresholdtest                 # Fewshot threshold test notebook examples
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
│   └── terminal_workings             # Terminal workings on how to run the Ollama or API Model pipeline
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
│   │   │   └── oos_100notoos.txt
│   │   └── stackoverflow
│   │       ├── stackoverflow_25perc_oos.txt
│   │       ├── stackoverflow_only1notoos.txt
            ...
│   │       └── stackoverflow_only18notoos.txt
│   ├── fewshot_prompt.txt                                       # Last used fewshot prompt
│   └── zeroshot_prompt_without_oos_in_intentlist.txt            # Last used zeroshot prompt
├── requirements.txt                                             # Python libraries to install
├── results                                                      # Folder containing experiment results
│   ├── analysis                                                 # Folder containing analysis of zeroshot, fewshot, threshold-test
│   │   ├── analyse-results-fewshot-threshold-test.ipynb
│   │   ├── analyse-results-zeroshot-fewshot, create-idx2label.ipynb
│   │   └── EDA_THUIAR_Banking_n_StackOverflow_n_OOS_Query_Classification_Datasets.ipynb
│   ├── banking77_fewshot_google_gemini-2.5-flash-preview-05-20
│   ├── banking77_fewshot_llama3.2_3b                                    # In each experiment folder
│   │   ├── classification_report_llama3.2_3b_banking.txt                # Multi-class classification report (OOS vs individual known classes)
│   │   ├── classification_report_llama3.2_3b_banking_open_vs_known.txt  # Binary classification report (OOS/Open vs known class)
│   │   ├── cm_llama3.2_3b_banking.csv                                   # Multi-class classification's confusion matrix (OOS vs individual known classes) in CSV
│   │   ├── cm_llama3.2_3b_banking.png                                   # Multi-class classification's confusion matrix (OOS vs individual known classes) in PNG
│   │   ├── metrics_llama3.2_3b_banking.txt                              # Multi-class classification metrics (OOS vs individual known classes)
│   │   ├── metrics_llama3.2_3b_banking_open_vs_known.txt                # Binary classification metrics (OOS/Open vs known class)
│   │   └── results_llama3.2_3b_banking_0_7.json                         # Results JSON: list of dictionaries, with 1 dictionary per classified example
│   ├── banking77_fewshot_nebiusqwen3-30b-a3b                            # Experiment folders from banking77, stackoverflow, oos pipeline test runs (after refactoring from Jupyter notebooks to GitHub repo)
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
│   ├── round1                                                           # round1 to round9 folders: Results from experiments conducted in Jupyter notebooks
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



## 7. Results Summary

Please note that for the results section below, we will show only 
* experiments using 25% of OOS classes, to compare to the THUIAR paper
* zero-shot and few-shot experiments using pydantic enums to enforce allowed list of classes for prediction

For experiments with other percentage of OOS classes or where we initially explored not enforcing allowed list of classes, you can still access the results in the [results folder](https://github.com/KaiquanMah/llm-trust-lens/tree/main/results).


### 7.1 Overall Accuracy & Macro F1-score - 25% OOS Class

* From experiments where we converted 25% of classes to 'OOS'/Open and ran the pipeline, below are the Overall Accuracy & Macro F1-scores.
* Note that **overall refers to all questions/examples across the entire dataset**

<table>
  <!--2-row header: Dataset, Metric-->
  <thead>
    <tr>
      <th rowspan="2" style="text-align:left">Methods</th>
      <th colspan="2" style="text-align:center">Banking77</th>
      <th colspan="2" style="text-align:center">StackOverflow</th>
      <th colspan="2" style="text-align:center">CLINC150OOS</th>
    </tr>
    <tr>
      <th style="text-align:center">Overall Accuracy</th>
      <th style="text-align:center">Overall Macro F1-score</th>
      <th style="text-align:center">Overall Accuracy</th>
      <th style="text-align:center">Overall Macro F1-score</th>
      <th style="text-align:center">Overall Accuracy</th>
      <th style="text-align:center">Overall Macro F1-score</th>
    </tr>
  </thead>
  <tbody>
    <!--2021 THUIAR ADB Paper's Metrics-->
    <tr>
      <td style="text-align:left">ADB (2021 THUIAR Paper)</td>
      <td style="text-align:center">78.85</td>
      <td style="text-align:center">71.62</td>
      <td style="text-align:center">86.72</td>
      <td style="text-align:center">80.83</td>
      <td style="text-align:center">87.59</td>
      <td style="text-align:center">77.19</td>
    </tr>
    <!--Our Metrics: base model on top, then sort from highest to lowest zeroshot model-->
    <tr>
      <td style="text-align:left">llama3.2:3b (Our Base Ollama/Local LLM) Zero-Shot with Pydantic Enums</td>
      <td style="text-align:center">43.74</td>
      <td style="text-align:center">53</td>
      <td style="text-align:center">66.62</td>
      <td style="text-align:center">73.10</td>
      <td style="text-align:center">45.76</td>
      <td style="text-align:center">55.79</td>
    </tr>
    <tr>
      <td style="text-align:left">qwen3:8b (Mixture-of-Experts LLM) Zero-Shot with Pydantic Enums</td>
      <td style="text-align:center">53.86</td>
      <td style="text-align:center">63.97</td>
      <td style="text-align:center">-</td>
      <td style="text-align:center">-</td>
      <td style="text-align:center">-</td>
      <td style="text-align:center">-</td>
    </tr>
    <tr>
        <td style="text-align:left">gemma3:4b-it-qa (Instruction-Following & Quantised LLM) Zero-Shot with Pydantic Enums</td>
        <td style="text-align:center">48.17</td>
        <td style="text-align:center">57.48</td>
        <td style="text-align:center">-</td>
        <td style="text-align:center">-</td>
        <td style="text-align:center">-</td>
        <td style="text-align:center">-</td>
      </tr>
      <tr>
        <td style="text-align:left">mistral:7b (General-Purpose LLM) Zero-Shot with Pydantic Enums</td>
        <td style="text-align:center">46.62</td>
        <td style="text-align:center">54.99</td>
        <td style="text-align:center">-</td>
        <td style="text-align:center">-</td>
        <td style="text-align:center">-</td>
        <td style="text-align:center">-</td>
      </tr>
      <tr>
        <td style="text-align:left">tulu3:8b (Instruction-Following LLM) Zero-Shot with Pydantic Enums</td>
        <td style="text-align:center">44.58</td>
        <td style="text-align:center">52.69</td>
        <td style="text-align:center">-</td>
        <td style="text-align:center">-</td>
        <td style="text-align:center">-</td>
        <td style="text-align:center">-</td>
      </tr>
      <tr>
        <td style="text-align:left">deepseek-r1:7b (Reasoning LLM) Zero-Shot with Pydantic Enums</td>
        <td style="text-align:center">32.14</td>
        <td style="text-align:center">36.70</td>
        <td style="text-align:center">-</td>
        <td style="text-align:center">-</td>
        <td style="text-align:center">-</td>
        <td style="text-align:center">-</td>
      </tr>
      <!--Our Metrics: fewshot models-->
      <tr>
         <td style="text-align:left">llama3.2:3b (Our Base Ollama/Local LLM) Few-Shot with 5 examples per known class, with Pydantic Enums</td>
         <td style="text-align:center">14.09</td>
         <td style="text-align:center">16.30</td>
         <td style="text-align:center">69.67</td>
         <td style="text-align:center">75.72</td>
         <td style="text-align:center">23.23</td>
         <td style="text-align:center">28.09</td>
       </tr>
       <tr>
         <td style="text-align:left">QWEN3-30B-A3B (Mixture-of-Experts API LLM) Few-Shot with 5 examples per known class, with Pydantic Enums</td>
         <td style="text-align:center">70.28</td>
         <td style="text-align:center">76.48</td>
         <td style="text-align:center">85.72</td>
         <td style="text-align:center">87.87</td>
         <td style="text-align:center">80.35</td>
         <td style="text-align:center">85.39</td>
       </tr>
  </tbody>
</table>


### 7.2 OOS/Open vs Known Macro F1-score - 25% OOS Class

* From experiments where we converted 25% of classes to 'OOS'/Open and ran the pipeline, below are the OOS/Open vs Known Macro F1-scores.
* Note that
  * **Non-embedding methods (zero-shot prompt, few-shot prompt) perform multi-class classification. So to get the 'known' class, we grouped all non-oos classes under 'known'**. Therefore for such experiments, we have
    * For multi-class classification
      * 1 classification_report.txt
      * 1 metrics.txt (Overall accuracy, Overall Weighted F1, Overall Macro F1)
      * 1 confusion_matrix.csv
      * 1 confusion_matrix.png
      * 1 results.json - containing the individual multi-class classification predictions
    * For open vs known (after grouping)
      * **1 classification_report.txt - We use the F1 scores for open vs known in this report for the table**
      * 1 metrics.txt (Accuracy, Weighted F1, Macro F1)
  * **Embedding methods (Adaptive Decision Boundary Clustering and Variational Autoencoder) currently perform only binary classification: open vs known**



<table>
  <!--2-row header: Dataset, Open vs Known-->
  <thead>
    <tr>
      <th rowspan="2" style="text-align:left">Methods</th>
      <th colspan="2" style="text-align:center">Banking77</th>
      <th colspan="2" style="text-align:center">StackOverflow</th>
      <th colspan="2" style="text-align:center">CLINC150OOS</th>
    </tr>
    <tr>
      <th style="text-align:center">Open</th>
      <th style="text-align:center">Known</th>
      <th style="text-align:center">Open</th>
      <th style="text-align:center">Known</th>
      <th style="text-align:center">Open</th>
      <th style="text-align:center">Known</th>
    </tr>
  </thead>
  <!--2021 THUIAR ADB Paper's F1-->
  <tbody>
    <tr>
      <td style="text-align:left">ADB (2021 THUIAR Paper)</td>
      <td style="text-align:center">84.56</td>
      <td style="text-align:center">70.94</td>
      <td style="text-align:center">90.88</td>
      <td style="text-align:center">78.82</td>
      <td style="text-align:center">91.84</td>
      <td style="text-align:center">76.80</td>
    </tr>
  </tbody>
  <!--Our F1-->
</table>






## 8. License  
This project is licensed under the MIT License - see the LICENSE file for details.

