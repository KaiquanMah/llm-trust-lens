# import libraries
import argparse
import os
import time
import pandas as pd
import json
from pathlib import Path

# Import common utilities
from experiment_common import (
    load_config,
    load_prompt_template,
    load_few_shot_examples,
    create_intent_schema,
    format_few_shot_prompt,
    get_base_filename,
    save_evaluation_results,
    process_config_and_data
)
from ollama_utils import initialize_ollama

# --- Main Orchestrator ---
def run_ollama_experiment(config_path: str):
    """
    Runs an experiment using a local Ollama model.
    
    Args:
        config_path (str): Path to the experiment configuration file
    """
    project_root = Path(__file__).parent.parent
    
    # 1. Load All Configurations
    exp_config = load_config(project_root / config_path)
    dataset_config = load_config(project_root / exp_config['dataset_config'])
    
    print("--- Starting Experiment ---")
    print(f"Experiment Config: {config_path}")
    print(f"Dataset: {dataset_config['name']}, Model: {exp_config['model_name']}, Technique: {exp_config['technique']}")
    
    # 2. Setup Output Directory
    output_dir = project_root / exp_config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # 3. Initialize Ollama Server and Client
    client = initialize_ollama(exp_config)

    # 4. Load and Process Data
    df, labels = process_config_and_data(exp_config, dataset_config)
    
    # 5. Create Pydantic schema
    print("\n--- Preparing Model and Prompts ---")
    IntentSchema = create_intent_schema(labels)
    
    # 6. Load prompt template and examples
    prompt_template = load_prompt_template(project_root / exp_config['prompt_template_path'])
    few_shot_examples = ""
    if exp_config['technique'] == 'fewshot':
        examples_path = project_root / exp_config['few_shot']['examples_path']
        few_shot_examples = load_few_shot_examples(examples_path)

    # 7. Main Inference Loop
    print("\n--- Running Inference ---")
    start_index = exp_config.get('start_index', 0)
    end_index = exp_config.get('end_index', len(df))
    if end_index is None:
        end_index = len(df)
    
    run_df = df.iloc[start_index:end_index]
    print(f"Processing {len(run_df)} records from index {start_index} to {end_index}.")

    start_time = time.time()
    results = []
    
    for i, (index, row) in enumerate(run_df.iterrows()):
        text_input = row[dataset_config['text_column']]
        true_label = row[dataset_config['label_column']]
        
        # Format prompt
        prompt = format_few_shot_prompt(prompt_template, text_input, labels, few_shot_examples)

        if i == 0:
            print("\n--- Example Prompt (first item) ---")
            print(prompt)
            print("-" * 40 + "\n")

        try:
            # Get prediction from Ollama
            schema = IntentSchema.model_json_schema()
            response = client.chat(
                model=exp_config['model_name'],
                messages=[{'role': 'user', 'content': prompt}],
                format=schema,
                options={'temperature': 0.0}
            )           
            msg = response['message']['content']
            parsed_json = json.loads(msg)
            parsed_data = IntentSchema(**parsed_json)
            predicted = parsed_data.category
            confidence = parsed_data.confidence
            
        except Exception as e:
            predicted = 'error'
            confidence = 0.0
            print(f"\nError processing row {index}: {str(e)}")
            if 'msg' in locals():
                print(f"Response that caused error: {msg}")

        # Store result
        results.append({
            'text': text_input,
            'label': true_label,
            'predicted': predicted,
            'confidence': confidence,
            'dataset': dataset_config['name'],
            'split': row.get('split')
        })

        # Log progress with ETA
        log_every = exp_config.get('run_control', {}).get('log_every_n_examples', 100)
        if (i + 1) % log_every == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_row = elapsed_time / (i + 1)
            remaining_rows = len(run_df) - (i + 1)
            eta = avg_time_per_row * remaining_rows
            print(f"Processed {i + 1}/{len(run_df)} | Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} | ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}")

    # 8. Save Results and Generate Metrics
    results_df = pd.DataFrame(results)
    base_filename = get_base_filename(exp_config, dataset_config, start_index, end_index)
    
    # Save raw predictions
    predictions_path = output_dir / f"results_{base_filename}.json"
    results_df.to_json(predictions_path, orient='records', indent=4)
    print(f"\nRaw predictions saved to {predictions_path}")

    # Generate metrics and visualizations
    print("\n--- Generating Evaluation Metrics ---")
    save_evaluation_results(results_df, labels, output_dir, base_filename, exp_config, dataset_config)

    print("\n--- Experiment Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a prompting-based intent classification experiment with Ollama.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment configuration file")
    args = parser.parse_args()
    
    run_ollama_experiment(args.config)
