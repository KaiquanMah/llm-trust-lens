# import libraries
import argparse
import yaml
import os
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pydantic import BaseModel, Field
from typing import Literal
import re

# Import our custom utility functions
from data_utils import load_dataset_and_labels
from ollama_utils import initialize_ollama




# --- Helper Functions ---

def load_config(config_path: str) -> dict:
    """Loads a YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_prompt_template(template_path: str) -> str:
    """Loads a prompt template from a file."""
    with open(template_path, 'r') as f:
        return f.read()

def format_zero_shot_prompt(template: str, text_input: str, labels: list) -> str:
    """Injects data into a zero-shot prompt template."""
    category_list_str = ", ".join(f'"{label}"' for label in labels)
    
    return template.format(categories=category_list_str, 
                           question=text_input)
    

def load_few_shot_examples(file_path: str) -> list:
    """
    Loads the entire content of the few-shot examples file as a single string.
    """
    print(f"Loading few-shot examples content directly from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: Few-shot examples file not found at {file_path}")
        return "" # Return an empty string to prevent crashes


def format_few_shot_prompt(template: str, text_input: str, labels: list, few_shot_examples_str: str) -> str:
    """
    Injects data and formatted few-shot examples into a prompt template by
    mapping the code's variable names to the prompt file's placeholder names.
    """
    # 1. Prepare the data strings, same as before.
    category_list_str = ", ".join(f'"{label}"' for label in labels)
    
    # 2. Use the correct keys ('categories', 'question') that match the .txt file.
    return template.format(
        categories=category_list_str,      # Maps to {categories}
        fewshot_examples=few_shot_examples_str,      # Maps to {fewshot_examples}
        question=text_input                # Maps to {question}
    )
    

def save_evaluation_results(df_results: pd.DataFrame, labels: list, output_dir: Path):
    """Calculates metrics and saves all evaluation artifacts."""
    valid_results = df_results[df_results['predicted_label'].isin(labels)]
    if len(valid_results) == 0:
        print("\nWarning: No valid predictions found to evaluate. Skipping metrics generation.")
        return

    true_labels = valid_results['true_label']
    predicted_labels = valid_results['predicted_label']

    report = classification_report(true_labels, predicted_labels, labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_path = output_dir / "classification_report.csv"
    report_df.to_csv(report_path)
    print(f"\nClassification Report saved to {report_path}")

    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_csv_path = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"Confusion matrix saved to {cm_csv_path}")

    # larger figure because CLINC150OOS dataset has 150 non-oos classes + 1 oos class
    # which needs a larger figure to avoid having overlapping numbers
    # plt.figure(figsize=(80, 80))
    # Make the plot larger for datasets with many labels
    figsize = max(20, len(labels) // 2 + 10)
    sns.heatmap(cm_df, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_png_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_png_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to {cm_png_path}")













# --- Main Orchestrator ---
def main():
    # This main function includes all the fixes from the previous step
    # (reading the full config, processing data, slicing, and JSON parsing)
    # and now works correctly with the modified format_few_shot_prompt function.
    
    parser = argparse.ArgumentParser(description="Run a prompting-based intent classification experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to the main experiment config file.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    
    # 1. Load All Configurations
    exp_config = load_config(project_root / args.config)
    dataset_config = load_config(project_root / exp_config['dataset_config'])
    
    print("--- Starting Experiment ---")
    print(f"Experiment Config: {args.config}")
    print(f"Dataset: {dataset_config['name']}, Model: {exp_config['model_name']}, Technique: {exp_config['technique']}")
    
    # 2. Setup Output Directory
    output_dir = project_root / exp_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # 3. Initialize Ollama Server and Client
    client = initialize_ollama(exp_config)

    # 4. Load Data, Labels, and Prompt Template
    df, labels = load_dataset_and_labels(dataset_config)
    
    print("\n--- Applying Data Processing from Experiment Config ---")
    if exp_config.get('force_oos', False):
        oos_indices = exp_config.get('list_oos_idx', [])
        if oos_indices:
            oos_labels_to_replace = [labels[i] for i in oos_indices if i < len(labels)]
            print(f"Found {len(oos_labels_to_replace)} labels to convert to 'oos'.")
            
            df[dataset_config['label_column']] = df[dataset_config['label_column']].replace(oos_labels_to_replace, 'oos')
            # Recalculate the list of unique labels after transformation
            labels = sorted(list(df[dataset_config['label_column']].unique()))
            print(f"Labels re-mapped. New number of unique labels: {len(labels)}")


    # filter dataframe - for threshold test only
    threshold_config = exp_config.get('threshold', {})
    if threshold_config.get('filter_oos_qns_only', False):
        n_oos = threshold_config.get('n_oos_qns', 100)
        df = df[df[dataset_config['label_column']] == 'oos'].head(n_oos)
        print(f"Dataset filtered to {len(df)} 'oos' questions.")

    # create Pydantic schema
    print("\n--- Preparing Model and Prompts ---")
    class IntentSchema(BaseModel):
        category: Literal[*labels]
        confidence: float = Field(..., ge=0.0, le=1.0)

    prompt_template = load_prompt_template(project_root / exp_config['prompt_template_path'])

    
    # 5. Prepare for the main loop based on technique
    few_shot_examples_string = ""
    if exp_config['technique'] == 'fewshot':
        examples_path = project_root / exp_config['few_shot']['examples_path']
        few_shot_examples_string = load_few_shot_examples(examples_path)

    # 6. Main Inference Loop
    print("\n--- Running Inference ---")
    results = []
    
    start_index = exp_config.get('start_index', 0)
    end_index = exp_config.get('end_index', len(df))
    if end_index is None:
        end_index = len(df)
    
    run_df = df.iloc[start_index:end_index]
    print(f"Processing {len(run_df)} records from index {start_index} to {end_index}.")

    start_time = time.time()
    total_rows = len(run_df)

    
    # SAMPLE 10 RECORDS
    # sample_df = df.head(10) # Using a small sample. Remove .head(10) for a full run.
    # for index, row in tqdm(sample_df.iterrows(), total=sample_df.shape[0], desc="Classifying Intents"):
    # vs
    # FULL RUN
    
    for i, (index, row) in enumerate(run_df.iterrows()):
        text_input = row[dataset_config['text_column']]
        true_label = row[dataset_config['label_column']]
        
        # A. Format the prompt based on the chosen technique
        prompt = format_few_shot_prompt(prompt_template, text_input, labels, few_shot_examples_string)

        if i == 0:
            print("\n--- Example Prompt (for first item) ---")
            print(prompt)
            print("--------------------------------------\n")

        try:
            # B. Get the prediction from the LLM (The core "predict_class" action)
            response = client.chat(
                model=exp_config['model_name'],
                messages=[{'role': 'user', 'content': prompt}],
                # format='json', # Tell Ollama to enforce JSON output
                format = IntentSchema.model_json_schema(),
                options={'temperature': 0.0}
            )           
            msg = response['message']['content']
            # Validate the response with our dynamic Pydantic model
            parsed_data = json.loads(msg)
            prediction = parsed_data.category
            confidence = parsed_data.confidence
        except (json.JSONDecodeError, Exception) as e:
            prediction = "ERROR"
            confidence = 0.0
            # Provide a more helpful error message
            print(f"\nError processing row {index}: {e}")
            if 'msg' in locals():
                print(f"LLM Response that caused error: {msg}")

        # C. Store the result
        results.append({
            'index': index,
            'text': text_input,
            'true_label': true_label,
            'predicted_label': prediction,
            'confidence': confidence
        })

        # --- FEATURE: Re-add advanced ETA logging ---
        log_every = exp_config.get('run_control', {}).get('log_every_n_examples', 100)
        if (i + 1) % log_every == 0 and i > 0:
            elapsed_time = time.time() - start_time
            avg_time_per_row = elapsed_time / (i + 1)
            remaining_rows = total_rows - (i + 1)
            eta = avg_time_per_row * remaining_rows
            print(f"  Processed {i + 1}/{total_rows} | Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} | ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}")

    # 7. Process and Save Raw Predictions
    results_df = pd.DataFrame(results)
    predictions_path = output_dir / "predictions.json"
    results_df.to_json(predictions_path, orient='records', indent=4)
    print(f"\nRaw predictions saved to {predictions_path}")

    # 8. Evaluate and Save Metrics
    print("\n--- Generating Evaluation Metrics ---")
    save_evaluation_results(results_df, labels, output_dir)

    print("\n--- Experiment Finished ---")


if __name__ == "__main__":
    main()
