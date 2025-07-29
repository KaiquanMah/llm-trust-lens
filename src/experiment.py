import argparse
import yaml
import os
import ollama
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

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
    label_list_str = ", ".join(f'"{label}"' for label in labels)
    return template.format(labels=label_list_str, text_input=text_input)

def save_evaluation_results(df_results: pd.DataFrame, labels: list, output_dir: Path):
    """Calculates metrics and saves all evaluation artifacts."""
    true_labels = df_results['true_label']
    predicted_labels = df_results['predicted_label']

    # 1. Classification Report
    report = classification_report(true_labels, predicted_labels, labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_path = output_dir / "classification_report.csv"
    report_df.to_csv(report_path)
    print(f"\nClassification Report:\n{pd.DataFrame(report).transpose()}")
    print(f"Classification report saved to {report_path}")

    # 2. Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Save CM to CSV
    cm_csv_path = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"Confusion matrix saved to {cm_csv_path}")

    # Save CM to PNG
    # larger confusion matrix for non-overlapping figures, especially for large CLINC150-oos dataset which has 151 classes (150 non-oos classes + 1 oos class)
    plt.figure(figsize=(80, 80))
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
    parser = argparse.ArgumentParser(description="Run a prompting-based intent classification experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to the main experiment config file.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    
    # 1. Load All Configurations
    exp_config_path = project_root / args.config
    exp_config = load_config(exp_config_path)
    dataset_config = load_config(project_root / exp_config['dataset_config'])
    
    print("--- Starting Experiment ---")
    print(f"Experiment Config: {args.config}")
    print(f"Dataset: {dataset_config['name']}")
    print(f"Model: {exp_config['model_name']}")
    print(f"Technique: {exp_config['technique']}")
    print("---------------------------")

    # 2. Setup Output Directory
    output_dir = project_root / exp_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # 3. Load Data and Prompt Template
    # Note: We need to import data_utils here to avoid circular dependencies if it ever needed to import from experiment
    from data_utils import load_dataset_and_labels
    df, labels = load_dataset_and_labels(dataset_config)
    prompt_template = load_prompt_template(project_root / exp_config['prompt_template_path'])

    # 4. Initialize Model Client
    try:
        client = ollama.Client(host=exp_config['ollama_host'])
    except Exception as e:
        print(f"ERROR: Could not connect to Ollama host at {exp_config['ollama_host']}.")
        print("Please ensure Ollama is running and accessible.")
        print(f"Details: {e}")
        return

    # 5. Run Inference Loop
    results = []
    # Using a small sample for demonstration. Remove .head(10) to run on the full dataset.
    sample_df = df.head(10) 
    for index, row in tqdm(sample_df.iterrows(), total=sample_df.shape[0], desc="Classifying Intents"):
        text_input = row[dataset_config['text_column']]
        true_label = row[dataset_config['label_column']]
        
        prompt = ""
        if exp_config['technique'] == 'zero-shot':
            prompt = format_zero_shot_prompt(prompt_template, text_input, labels)
        # Add elif for 'few-shot' here in the future
        else:
            raise ValueError(f"Technique '{exp_config['technique']}' not supported.")

        try:
            response = client.chat(
                model=exp_config['model_name'],
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0} # Hardcoding for reproducibility
            )
            prediction = response['message']['content'].strip().replace('"', '')
        except Exception as e:
            print(f"\nERROR during API call for row {index}: {e}")
            prediction = "API_ERROR"

        results.append({
            'index': index,
            'text': text_input,
            'true_label': true_label,
            'predicted_label': prediction
        })

    # 6. Process and Save Results
    results_df = pd.DataFrame(results)
    predictions_path = output_dir / "predictions.json"
    results_df.to_json(predictions_path, orient='records', indent=4)
    print(f"\nRaw predictions saved to {predictions_path}")

    # 7. Evaluate and Save Metrics
    save_evaluation_results(results_df, labels, output_dir)

    print("\n--- Experiment Finished ---")


if __name__ == "__main__":
    main()
