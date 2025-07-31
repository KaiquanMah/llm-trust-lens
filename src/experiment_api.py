# import libraries
import argparse
import os
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json
from pydantic import BaseModel, Field, create_model

# Import common utilities
from experiment_common import *
from nebius_utils import initialize_nebius_client, predict_with_nebius

# --- API-specific Functions ---
# Reuse common functions from experiment.py
def load_config(config_path: str) -> dict:
    """Loads a YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_prompt_template(template_path: str) -> str:
    """Loads a prompt template from a file."""
    with open(template_path, 'r') as f:
        return f.read()

def format_few_shot_prompt(template: str, text_input: str, labels: list, few_shot_examples_str: str) -> str:
    """
    Injects data and formatted few-shot examples into a prompt template.
    """
    display_labels = [label for label in labels if label != 'oos']
    category_list_str = "\n".join(f"- {label}" for label in display_labels)
    
    return template.format(
        categories=category_list_str,
        fewshot_examples=few_shot_examples_str,
        question=text_input
    )

def load_few_shot_examples(file_path: str) -> str:
    """Loads few-shot examples content."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: Few-shot examples file not found at {file_path}")
        return ""

# --- API-specific Functions ---
def create_intent_schema(labels: list) -> BaseModel:
    """
    Creates a Pydantic model for intent classification with the given labels.
    """
    # Create the category enum from labels
    CategoryEnum = Literal[tuple(labels)]  # type: ignore
    
    class IntentSchema(BaseModel):
        category: CategoryEnum = Field(..., description="The predicted intent category")
        confidence: float = Field(..., description="Confidence score between 0 and 1")
    
    return IntentSchema

def run_api_experiment(config_path: str):
    """
    Runs an experiment using an API-based model (e.g., Nebius Qwen).
    
    Args:
        config_path (str): Path to the experiment configuration file
    """
    # Load configurations
    exp_config = load_config(config_path)
    dataset_config = load_config(exp_config['dataset_config'])
    
    # Initialize API client
    client = initialize_nebius_client(exp_config)
    
    # Load dataset and labels
    dataset, labels = load_dataset_and_labels(
        dataset_config,
        exp_config.get('force_oos', False),
        exp_config.get('list_oos_idx', [])
    )
    
    # Create Pydantic schema for response validation
    IntentSchema = create_intent_schema(labels)
    
    # Load prompt template and few-shot examples
    prompt_template = load_prompt_template(exp_config['prompt_template_path'])
    few_shot_examples = load_few_shot_examples(exp_config['few_shot']['examples_path'])
    
    # Process data range
    start_index = exp_config.get('start_index', 0)
    end_index = exp_config.get('end_index')
    if end_index is None:
        end_index = len(dataset)
    
    # Prepare results storage
    results = []
    
    # Process examples
    for idx, row in tqdm(dataset.iloc[start_index:end_index].iterrows(), 
                        desc="Processing examples",
                        total=end_index-start_index):
        
        # Format prompt
        prompt = format_few_shot_prompt(
            prompt_template,
            row['text'],
            labels,
            few_shot_examples
        )
        
        try:
            # Get prediction from API
            response = predict_with_nebius(
                client=client,
                prompt=prompt,
                model_name=exp_config['model_name'],
                response_schema=IntentSchema,
                config=exp_config
            )
            
            # Store results
            result = {
                'text': row['text'],
                'label': row['label'],
                'predicted': response['category'],
                'confidence': response['confidence'],
                'dataset': dataset_config['name'],
                'split': row['split']
            }
            results.append(result)
            
            # Log progress
            if (idx + 1) % exp_config['run_control']['log_every_n_examples'] == 0:
                print(f"\nProcessed {idx + 1} examples")
                
        except Exception as e:
            print(f"Error processing example {idx}: {str(e)}")
            continue
    
    # Save results
    print("\nSaving experiment results...")
    results_df = pd.DataFrame(results)
    output_path = Path(exp_config['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add open vs known labels
    print("Adding open vs known classification labels...")
    results_df.loc[(results_df["label"] != "oos"), "label_open_vs_known"] = "known"
    results_df.loc[(results_df["label"] == "oos"), "label_open_vs_known"] = "open"
    results_df.loc[(results_df["predicted"] != "oos"), "predicted_open_vs_known"] = "known"
    results_df.loc[(results_df["predicted"] == "oos"), "predicted_open_vs_known"] = "open"
    
    # Save raw results
    base_filename = f"{exp_config['model_name'].replace('/', '_')}_{dataset_config['name']}_{start_index}_{end_index}"
    pickle_path = output_path / f"{base_filename}.pkl"
    results_df.to_pickle(pickle_path)
    print(f"Raw results saved to {pickle_path}")
    
    # Save detailed evaluation results
    print("\nGenerating evaluation metrics and visualizations...")
    save_evaluation_results(
        results_df,
        labels,
        output_path,
        base_filename,
        exp_config,
        dataset_config
    )

def save_evaluation_results(df_results: pd.DataFrame, 
                         labels: list, 
                         output_dir: Path, 
                         base_filename: str,
                         exp_config: dict,
                         dataset_config: dict):
    """
    Calculates metrics and saves all evaluation artifacts including:
    - Metrics summaries (both regular and open-vs-known)
    - Classification reports
    - Confusion matrices (CSV and visualization)
    
    Args:
        df_results: DataFrame with predictions
        labels: List of valid labels
        output_dir: Directory to save results
        base_filename: Base name for output files
        exp_config: Experiment configuration
        dataset_config: Dataset configuration
    """
    print("\nValidating results...")
    valid_results = df_results[df_results['predicted'].isin(labels)]
    if len(valid_results) == 0:
        print("\nWARNING: No valid predictions found to evaluate. Skipping metrics generation.")
        return
    print(f"Found {len(valid_results)} valid predictions out of {len(df_results)} total")

    # Get model and dataset name without indices for cm and classification report
    model_name = exp_config['model_name'].replace('/', '_').replace(':', '_').lower()
    dataset_name = dataset_config['name'].lower()
    filename_without_indices = f"{model_name}_{dataset_name}"
    
    # Get labels and predictions
    true_labels = valid_results['label']
    predicted_labels = valid_results['predicted']
    true_labels_open_vs_known = valid_results['label_open_vs_known']
    predicted_labels_open_vs_known = valid_results['predicted_open_vs_known']
    
    # Generate and save regular metrics summary
    print("\nGenerating regular metrics summary...")
    start_index = exp_config.get('start_index', 0)
    end_index = exp_config.get('end_index', None)
    
    metrics_summary = generate_metrics_summary(
        valid_results, model_name, dataset_name, 
        start_index, end_index, is_open_vs_known=False
    )
    metrics_path = output_dir / f"metrics_{filename_without_indices}.txt"
    with open(metrics_path, 'w') as f:
        f.write(metrics_summary)
    print(f"Metrics summary saved to {metrics_path}")
    
    # Open vs Known metrics
    print("\nGenerating open vs known metrics summary...")
    metrics_summary_open_vs_known = generate_metrics_summary(
        valid_results, model_name, dataset_name,
        start_index, end_index, is_open_vs_known=True
    )
    metrics_path_open_vs_known = output_dir / f"metrics_{filename_without_indices}_open_vs_known.txt"
    with open(metrics_path_open_vs_known, 'w') as f:
        f.write(metrics_summary_open_vs_known)
    print(f"Open vs Known metrics saved to {metrics_path_open_vs_known}")
    
    # Save confusion matrices
    print("\nGenerating confusion matrices...")
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Save CSV
    cm_csv_path = output_dir / f"cm_{filename_without_indices}.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"Confusion matrix saved to {cm_csv_path}")
    
    # Create and save visualization
    print("Generating confusion matrix visualization...")
    figsize = max(20, len(labels) // 2 + 10)
    plt.figure(figsize=(figsize, figsize))
    sns.heatmap(cm_df, 
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_png_path = output_dir / f"cm_{filename_without_indices}.png"
    plt.savefig(cm_png_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix visualization saved to {cm_png_path}")

def generate_metrics_summary(df_results: pd.DataFrame, 
                          model_name: str, 
                          dataset_name: str, 
                          start_index: int, 
                          end_index: int,
                          is_open_vs_known: bool = False) -> str:
    """
    Generate a summary of metrics including accuracy and F1 scores.
    
    Args:
        df_results: DataFrame with results
        model_name: Name of the model used
        dataset_name: Name of the dataset
        start_index: Starting index of processed examples
        end_index: Ending index of processed examples
        is_open_vs_known: Whether to generate metrics for open vs known classification
    """
    # Calculate metrics
    if is_open_vs_known:
        print("\nCalculating open vs known metrics...")
        true_labels = df_results['label_open_vs_known']
        predicted_labels = df_results['predicted_open_vs_known']
    else:
        print("\nCalculating intent classification metrics...")
        true_labels = df_results['label']
        predicted_labels = df_results['predicted']
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
    
    # Format the summary
    summary = f"""
=== Metrics Summary ===
Model: {model_name}
Dataset: {dataset_name}
Examples processed: {start_index} to {end_index if end_index is not None else 'end'}
Overall Accuracy: {accuracy:.2%}
Overall Weighted F1: {weighted_f1:.2%}
Overall Macro F1: {macro_f1:.2%}

Classification Report:
{classification_report(true_labels, predicted_labels)}
"""
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an API-based intent classification experiment")
    parser.add_argument("config", help="Path to experiment configuration YAML file")
    args = parser.parse_args()
    
    run_api_experiment(args.config)
