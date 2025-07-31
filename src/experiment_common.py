# import libraries
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

# --- Common Helper Functions ---

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

def load_few_shot_examples(file_path: str) -> str:
    """
    Loads the entire content of the few-shot examples file as a single string.
    """
    print(f"Loading few-shot examples content directly from: {file_path}")
    with open(file_path, 'r') as f:
        return f.read()


def create_intent_schema(labels: list) -> BaseModel:
    """
    Creates a Pydantic model for intent classification with the given labels.
    """
    # Create a dynamic Pydantic model with an enum for the categories
    return create_model(
        'IntentSchema',
        category=(str, Field(..., description="The predicted intent category", enum=labels)),
        confidence=(float, Field(..., ge=0.0, le=100.0, description="Confidence score between 0.0 and 100.0"))
    )


def format_few_shot_prompt(template: str, text_input: str, labels: list, few_shot_examples_str: str) -> str:
    """
    Injects data and formatted few-shot examples into a prompt template by
    mapping the code's variable names to the prompt file's placeholder names.
    """
    # Filter out 'oos' from the displayed categories list, but keep it as a valid output option
    display_labels = [label for label in labels if label != 'oos']
    # Convert to bullet-point format
    category_list_str = "\n".join(f"- {label}" for label in display_labels)
    
    # Use the correct keys ('categories', 'question') that match the .txt file.
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
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: Few-shot examples file not found at {file_path}")
        return "" # Return an empty string to prevent crashes

def format_few_shot_prompt(template: str, text_input: str, labels: list, few_shot_examples_str: str) -> str:
    """
    Injects data and formatted few-shot examples into a prompt template.
    """
    # Filter out 'oos' from the displayed categories list, but keep it as a valid output option
    display_labels = [label for label in labels if label != 'oos']
    # Convert to bullet-point format
    category_list_str = "\n".join(f"- {label}" for label in display_labels)
    
    return template.format(
        categories=category_list_str,
        fewshot_examples=few_shot_examples_str,
        question=text_input
    )

def get_base_filename(exp_config: dict, dataset_config: dict, start_index: int, end_index: int) -> str:
    """Creates a standardized base filename using model and dataset information."""
    model_name = exp_config['model_name'].replace('/', '_').replace(':', '_').lower()
    dataset_name = dataset_config['name'].lower()
    return f"{model_name}_{dataset_name}_{start_index}_{end_index-1}"

def add_open_vs_known_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add open vs known labels to the dataframe."""
    df = df.copy()
    df.loc[(df["label"] != "oos"), "label_open_vs_known"] = "known"
    df.loc[(df["label"] == "oos"), "label_open_vs_known"] = "open"
    df.loc[(df["predicted"] != "oos"), "predicted_open_vs_known"] = "known"
    df.loc[(df["predicted"] == "oos"), "predicted_open_vs_known"] = "open"
    return df

def generate_metrics_summary(df_results: pd.DataFrame, 
                          model_name: str, 
                          dataset_name: str, 
                          start_index: int, 
                          end_index: int,
                          is_open_vs_known: bool = False) -> str:
    """Generate a summary of metrics including accuracy and F1 scores."""
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
    model_name = exp_config['model_name'].replace('/', '_').replace(':', '_').lower()
    dataset_name = dataset_config['name'].lower()
    filename_without_indices = f"{model_name}_{dataset_name}"
    
    # Get labels and predictions
    true_labels = valid_results['label']
    predicted_labels = valid_results['predicted']
    true_labels_open_vs_known = valid_results['label_open_vs_known']
    predicted_labels_open_vs_known = valid_results['predicted_open_vs_known']
    
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
    print(f"Confusion matrix plot saved to {cm_png_path}")

def process_config_and_data(exp_config: dict, dataset_config: dict):
    """Process configuration and prepare dataset with OOS handling."""
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
            
            perc_to_oos = len(oos_labels_to_replace)/len(sorted_intent)
            print(f"Percentage of original intents to convert to OOS class: {perc_to_oos}\n")
            
            nonoos_labels = [label for label in labels if label not in oos_labels_to_replace]
            print(f"Non-OOS labels to preserve: {nonoos_labels}")
            
            total_unique_classes = len(sorted_intent)
            num_classes_to_convert = len(oos_indices)
            num_labels_to_preserve = total_unique_classes - num_classes_to_convert
            
            if num_labels_to_preserve <= 0:
                raise ValueError(f"Invalid configuration: list_oos_idx length ({num_classes_to_convert}) must be less than total unique classes ({total_unique_classes})")
            
            labels_to_preserve = set(nonoos_labels[-num_labels_to_preserve:])
            print(f"\nPreserving {num_labels_to_preserve} non-OOS labels")
            print(f"Labels to preserve: {sorted(labels_to_preserve)}")
            
            df[dataset_config['label_column']] = df[dataset_config['label_column']].apply(
                lambda x: next((l for l in labels_to_preserve if l.lower() == x.lower()), 'oos')
            )
            
            final_labels = sorted(set(df[dataset_config['label_column']].unique()))
            expected_labels = {'oos'} | labels_to_preserve
            if set(final_labels) != expected_labels:
                print(f"WARNING: Found unexpected labels. Expected {sorted(expected_labels)}, got {final_labels}")
            
            labels = ['oos'] + sorted(label for label in final_labels if label != 'oos')
            
            if 'oos' not in labels:
                print("ERROR: 'oos' label is missing from final labels")
            
            print(f"Final label count: {len(labels)} labels")
            print(f"Labels: {labels}")

    # Handle threshold test configuration
    threshold_config = exp_config.get('threshold', {})
    if threshold_config.get('filter_oos_qns_only', False):
        dataset_name = dataset_config['name']
        n_oos = threshold_config.get('n_oos_qns', 100)
        
        if dataset_name == 'banking':
            original_class = threshold_config.get('first_class_banking')
        elif dataset_name == 'stackoverflow':
            original_class = threshold_config.get('first_class_stackoverflow')
        else:
            original_class = threshold_config.get('first_class_oos')
            
        filtered_df = df[df[dataset_config['label_column']] == 'oos'].copy()
        
        if len(filtered_df) == 0:
            raise ValueError("No 'oos' examples found. Check if classes were properly converted to 'oos'.")
            
        n_available = len(filtered_df)
        n_to_sample = min(n_oos, n_available)
        df = filtered_df.sample(n=n_to_sample, random_state=38)
        print(f"Dataset filtered to {len(df)} examples that were originally '{original_class}' and converted to 'oos'")

    return df, labels
