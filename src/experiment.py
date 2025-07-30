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
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    f1_score
)
from pydantic import BaseModel, Field, create_model
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
    # Filter out 'oos' from the displayed categories list, but keep it as a valid output option
    display_labels = [label for label in labels if label != 'oos']
    # Convert to bullet-point format
    category_list_str = "\n".join(f"- {label}" for label in display_labels)
    
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
    # 1. Filter out 'oos' from the displayed categories list, but keep it as a valid output option
    display_labels = [label for label in labels if label != 'oos']
    # Convert to bullet-point format
    category_list_str = "\n".join(f"- {label}" for label in display_labels)
    
    # 2. Use the correct keys ('categories', 'question') that match the .txt file.
    return template.format(
        categories=category_list_str,      # Maps to {categories}
        fewshot_examples=few_shot_examples_str,      # Maps to {fewshot_examples}
        question=text_input                # Maps to {question}
    )
    

def get_base_filename(exp_config: dict, dataset_config: dict, start_index: int, end_index: int) -> str:
    """Creates a standardized base filename using model and dataset information."""
    model_name = exp_config['model_name'].replace('/', '_').replace(':', '_').lower()  # sanitize model name
    dataset_name = dataset_config['name'].lower()
    return f"{model_name}_{dataset_name}_{start_index}_{end_index-1}"

def add_open_vs_known_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add open vs known labels to the dataframe."""
    df = df.copy()
    # Add open vs known columns
    df.loc[(df["label"] != "oos"), "label_open_vs_known"] = "known"
    df.loc[(df["label"] == "oos"), "label_open_vs_known"] = "open"
    df.loc[(df["predicted"] != "oos"), "predicted_open_vs_known"] = "known"
    df.loc[(df["predicted"] == "oos"), "predicted_open_vs_known"] = "open"
    return df

def generate_metrics_summary(df_results: pd.DataFrame, model_name: str, dataset_name: str, start_index: int, end_index: int, is_open_vs_known: bool = False) -> str:
    """Generate a summary of metrics including accuracy and F1 scores."""
    # Calculate metrics
    if is_open_vs_known:
        true_labels = df_results['label_open_vs_known']
        predicted_labels = df_results['predicted_open_vs_known']
    else:
        true_labels = df_results['label']
        predicted_labels = df_results['predicted']
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
    
    # Format the summary
    summary = f"""
{model_name}
{dataset_name}
{start_index} to {end_index if end_index is not None else 'end'}
Overall Accuracy: {accuracy:.2%}
Overall Weighted F1: {weighted_f1:.2%}
Overall F1: {macro_f1:.2%}  # macro F1
"""
    return summary

def save_evaluation_results(df_results: pd.DataFrame, labels: list, output_dir: Path, base_filename: str, exp_config: dict, dataset_config: dict):
    """Calculates metrics and saves all evaluation artifacts."""
    valid_results = df_results[df_results['predicted'].isin(labels)]
    if len(valid_results) == 0:
        print("\nWarning: No valid predictions found to evaluate. Skipping metrics generation.")
        return

    # Add open vs known labels
    valid_results = add_open_vs_known_labels(valid_results)

    # Get model and dataset name without indices for cm and classification report
    # Replace both '/' and ':' with '_' in model name for safe filenames
    model_name = exp_config['model_name'].replace('/', '_').replace(':', '_').lower()
    dataset_name = dataset_config['name'].lower()
    filename_without_indices = f"{model_name}_{dataset_name}"
    
    # Get basic labels and predictions
    true_labels = valid_results['label']
    predicted_labels = valid_results['predicted']

    # Get open vs known labels and predictions
    true_labels_open_vs_known = valid_results['label_open_vs_known']
    predicted_labels_open_vs_known = valid_results['predicted_open_vs_known']
    
    # Generate and save regular metrics summary
    start_index = exp_config.get('start_index', 0)
    end_index = exp_config.get('end_index', None)
    
    # Regular metrics
    metrics_summary = generate_metrics_summary(valid_results, model_name, dataset_name, start_index, end_index, is_open_vs_known=False)
    metrics_path = output_dir / f"metrics_{filename_without_indices}.txt"
    with open(metrics_path, 'w') as f:
        f.write(metrics_summary)
    print(f"\nMetrics summary saved to {metrics_path}")
    
    # Open vs Known metrics
    metrics_summary_open_vs_known = generate_metrics_summary(valid_results, model_name, dataset_name, start_index, end_index, is_open_vs_known=True)
    metrics_path_open_vs_known = output_dir / f"metrics_{filename_without_indices}_open_vs_known.txt"
    with open(metrics_path_open_vs_known, 'w') as f:
        f.write(metrics_summary_open_vs_known)
    print(f"Open vs Known metrics summary saved to {metrics_path_open_vs_known}")
    
    # Save regular classification report
    report_txt_path = output_dir / f"classification_report_{filename_without_indices}.txt"
    with open(report_txt_path, 'w') as f:
        f.write(classification_report(true_labels, predicted_labels, labels=labels, zero_division=0))
    print(f"\nClassification Report saved to {report_txt_path}")
    
    # Save open vs known classification report
    report_txt_path_open_vs_known = output_dir / f"classification_report_{filename_without_indices}_open_vs_known.txt"
    with open(report_txt_path_open_vs_known, 'w') as f:
        f.write(classification_report(true_labels_open_vs_known, predicted_labels_open_vs_known, labels=['known', 'open'], zero_division=0))
    print(f"Open vs Known Classification Report saved to {report_txt_path_open_vs_known}")

    # Save confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_csv_path = output_dir / f"cm_{filename_without_indices}.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"Confusion matrix saved to {cm_csv_path}")

    # Create confusion matrix plot
    figsize = max(20, len(labels) // 2 + 10)
    plt.figure(figsize=(figsize, figsize))
    sns.heatmap(cm_df, 
                annot=True,           # Show numbers in cells
                fmt='d',              # Use integer format
                cmap='Blues',
                xticklabels=labels,   # Use our sorted labels
                yticklabels=labels)   # Use same labels for both axes
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_png_path = output_dir / f"cm_{filename_without_indices}.png"
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
    
    # Print initial dataset statistics
    sorted_intent = sorted(df[dataset_config['label_column']].unique())
    print("="*80)
    print(f"Original dataset intents: {sorted_intent}")
    print(f"Number of original intents: {len(sorted_intent)}\n")

    if exp_config.get('force_oos', False):
        oos_indices = exp_config.get('list_oos_idx', [])
        if oos_indices:
            # Get original labels to convert to OOS
            oos_labels_to_replace = [labels[i] for i in oos_indices if i < len(labels)]
            print("="*80)
            print("Original intents to convert to OOS class:")
            for i, label in enumerate(oos_labels_to_replace):
                print(f"{i:4d}  {label}")
            
            # Calculate percentage against original intent count
            perc_to_oos = len(oos_labels_to_replace)/len(sorted_intent)
            print(f"Percentage of original intents to convert to OOS class: {perc_to_oos}\n")
            
            # Convert labels to OOS
            df[dataset_config['label_column']] = df[dataset_config['label_column']].replace(oos_labels_to_replace, 'oos')
            
            # Verify the conversion was complete
            unique_labels_after_conversion = sorted(df[dataset_config['label_column']].unique())
            if len(unique_labels_after_conversion) != 2 or 'oos' not in unique_labels_after_conversion:
                print(f"WARNING: After conversion, found unexpected labels: {unique_labels_after_conversion}")
            
            # Find the last index that wasn't converted to OOS (should be only one)
            remaining_label = labels[-1]  # Get the last label from original ordered list
            if remaining_label in oos_labels_to_replace:
                print("ERROR: Last label was also converted to OOS. Check list_oos_idx configuration.")
                remaining_label = [label for label in labels if label not in oos_labels_to_replace][0]
            
            # Create final label list: 'oos' first, then the one remaining class
            labels = ['oos', remaining_label]  # Exactly two classes, in this specific order
            
            # Verify we have exactly what we expect
            if len(labels) != 2:
                print(f"ERROR: Expected exactly 2 labels, but got {len(labels)}: {labels}")
            if 'oos' not in labels:
                print("ERROR: 'oos' label is missing from final labels")
            
            print("="*80)
            print("Unique intents after converting some to OOS class:")
            print(labels)
            print(f"Number of unique intents after converting some to OOS class: {len(labels)}\n")
            
            # Sanity check
            int_oos_in_orig_dataset = int('oos' in sorted_intent)
            adjust_if_oos_not_in_orig_dataset = 0 if int_oos_in_orig_dataset else 1
            
            print("="*80)
            print("sanity check")
            # Original counts
            print(f"Number of original intents: {len(sorted_intent)}")
            print(f"Number of original intents + 1 OOS class (if doesnt exist in original dataset): {len(sorted_intent) + adjust_if_oos_not_in_orig_dataset}")
            print(f"Number of original intents to convert to OOS class: {len(oos_labels_to_replace)}")
            print(f"Percentage of original intents to convert to OOS class: {len(oos_labels_to_replace)/len(sorted_intent)}")  # Using original count as denominator
            
            # After conversion checks
            expected_final_count = 2  # We should have exactly 2 classes: one non-OOS and 'oos'
            print(f"Number of unique intents after converting some to OOS class: {len(labels)}")
            print(f"Number of original intents + 1 OOS class (if doesnt exist in original dataset) - converted classes: {len(sorted_intent) + adjust_if_oos_not_in_orig_dataset - len(oos_labels_to_replace)}")
            print(f"Numbers match: {len(labels) == expected_final_count}")
            
            if len(labels) != expected_final_count:
                print(f"WARNING: Expected {expected_final_count} classes (1 non-OOS + 'oos'), but found {len(labels)}: {labels}")
            print("Prepared unique intents")


    # filter dataframe - for threshold test only
    threshold_config = exp_config.get('threshold', {})
    if threshold_config.get('filter_oos_qns_only', False):
        dataset_name = dataset_config['name']
        n_oos = threshold_config.get('n_oos_qns', 100)
        
        # Get the appropriate first class based on dataset
        if dataset_name == 'banking':
            original_class = threshold_config.get('first_class_banking')
        elif dataset_name == 'stackoverflow':
            original_class = threshold_config.get('first_class_stackoverflow')
        else:  # oos dataset
            original_class = threshold_config.get('first_class_oos')
            
        # For threshold test, we want examples that were originally the specified class
        # but have been converted to 'oos'
        filtered_df = df[df[dataset_config['label_column']] == 'oos'].copy()
        
        if len(filtered_df) == 0:
            raise ValueError("No 'oos' examples found. Check if classes were properly converted to 'oos'.")
            
        # Sample n examples or all available if less than n_oos
        n_available = len(filtered_df)
        n_to_sample = min(n_oos, n_available)
        df = filtered_df.sample(n=n_to_sample, random_state=38)
        print(f"Dataset filtered to {len(df)} examples that were originally '{original_class}' and converted to 'oos' (requested {n_oos}, available {n_available}).")

    # create Pydantic schema
    print("\n--- Preparing Model and Prompts ---")
    
    # Create a dynamic Pydantic model with an enum for the categories
    IntentSchema = create_model(
        'IntentSchema',
        category=(str, Field(..., description="The predicted intent category", enum=labels)),
        # Allow confidence values from 0.0-100.0
        confidence=(float, Field(..., ge=0.0, le=100.0, description="Confidence score between 0.0 and 100.0"))
    )

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
            schema = IntentSchema.model_json_schema()
            # # Remove unnecessary schema elements to make it clearer for the LLM
            # schema.pop('title', None)
            # schema.pop('description', None)
            
            response = client.chat(
                model=exp_config['model_name'],
                messages=[{'role': 'user', 'content': prompt}],
                format=schema,  # Pass the schema to enforce valid categories and structure
                options={'temperature': 0.0}
            )           
            msg = response['message']['content']
            # Parse the JSON response
            parsed_json = json.loads(msg)
            # Validate the response with our dynamic Pydantic model
            parsed_data = IntentSchema(**parsed_json)
            predicted = parsed_data.category
            confidence = parsed_data.confidence
        except (json.JSONDecodeError, Exception) as e:
            predicted = 'error'
            confidence = 0.0
            # Provide a more helpful error message
            print(f"\nError processing row {index}: {e}")
            if 'msg' in locals():
                print(f"LLM Response that caused error: {msg}")

        # C. Store the result
        results.append({
            "Index": index,
            "text": text_input,
            "label": true_label,
            "dataset": dataset_config['name'],  # eg banking, oos, stackoverflow
            "split": row['split'] if 'split' in row else None,  # eg train, dev, test
            "predicted": predicted,
            "confidence": confidence
        })

        # --- FEATURE: Re-add advanced ETA logging ---
        log_every = exp_config.get('run_control', {}).get('log_every_n_examples', 100)
        if (i + 1) % log_every == 0 and i > 0:
            elapsed_time = time.time() - start_time
            avg_time_per_row = elapsed_time / (i + 1)
            remaining_rows = total_rows - (i + 1)
            eta = avg_time_per_row * remaining_rows
            print(f"  Processed {i + 1}/{total_rows} | Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} | ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}")

    # 7. Create base filename for all outputs
    base_filename = get_base_filename(exp_config, dataset_config, start_index, end_index)
    
    # 8. Process and Save Raw Predictions
    results_df = pd.DataFrame(results)
    predictions_path = output_dir / f"results_{base_filename}.json"
    results_df.to_json(predictions_path, orient='records', indent=4)
    print(f"\nRaw predictions saved to {predictions_path}")

    # 9. Evaluate and Save Metrics
    print("\n--- Generating Evaluation Metrics ---")
    save_evaluation_results(results_df, labels, output_dir, base_filename, exp_config, dataset_config)

    print("\n--- Experiment Finished ---")


if __name__ == "__main__":
    main()
