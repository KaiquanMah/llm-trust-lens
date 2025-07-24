from src.config import Config
import pandas as pd
import os
import ollama
import json
import pickle
import time
from pydantic import BaseModel
from typing import Literal
# from enum import Enum
from huggingface_hub import snapshot_download



# Config.target_dir
# Config.cloned_data_dir'
# Config.dataset_name
# Config.model_name
# Config.start_index
# Config.end_index
# Config.log_every_n_examples


#######################
# load data
#######################
def load_data(data_dir):
    """Loads train, dev, and test datasets from a specified directory."""

    main_df = pd.DataFrame()
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(data_dir, f'{split}.tsv')
        if os.path.exists(file_path):
          try:
            df = pd.read_csv(file_path, sep='\t')
            df['dataset'] = os.path.basename(data_dir)
            df['split'] = split
            main_df = pd.concat([main_df, df], ignore_index=True)
          except pd.errors.ParserError as e:
            print(f"Error parsing {file_path}: {e}")
            # Handle the error appropriately, e.g., skip the file, log the error, etc.
        else:
            print(f"Warning: {split}.tsv not found in {data_dir}")
    return main_df


def filter100examples_oos(dataset_name, df):
    # dont input 'only oos qns to model'
    if Config.filter_oos_qns_only == False:
        filtered_df = df
    # vs
    # input 'only oos qns to model'
    else:
        if dataset_name == 'banking':
            first_class = Config.first_class_banking
        elif dataset_name == 'stackoverflow':
            first_class = Config.first_class_stackoverflow
        else:
            first_class = Config.first_class_oos
    
        filtered_df = df.copy()
        filtered_df = filtered_df.loc[filtered_df["label"] == first_class]
        filtered_df = filtered_df.sample(n=Config.n_oos_qns, random_state=38)
    return filtered_df


df = pd.DataFrame()

data_dir = os.path.join(Config.cloned_data_dir, Config.dataset_name)
if os.path.exists(data_dir):
  df = load_data(data_dir)
  print(f"Loaded dataset into dataframe: {Config.dataset_name}")
  print(f"Dimensions: {df.shape}")
  print(f"Col names: {df.columns}")
else:
  print(f"Warning: Directory {data_dir} not found.")
#######################



#######################
# unique intents
#######################
sorted_intent = list(sorted(df.label.unique()))
print("="*80)
print(f"Original dataset intents: {sorted_intent}")
print(f"Number of original intents: {len(sorted_intent)}\n")


# 2025.06.03
# New OOS approach - get 25/50/75% of class indexes for each dataset within the team (for reproducibility and comparable results)
# Change their class labels to 'oos'
snapshot_download(repo_id="KaiquanMah/open-intent-query-classification", repo_type="space", allow_patterns="*_idx2label.csv", local_dir=Config.idx2label_target_dir)
idx2label_filepath = Config.idx2label_target_dir + '/dataset_idx2label/' + Config.idx2label_filename_hf
idx2label = pd.read_csv(idx2label_filepath)
idx2label_oos = idx2label[idx2label.index.isin(Config.list_oos_idx)]
idx2label_oos.reset_index(drop=True, inplace=True)

# 2025.06.17 keep track of non-oos labels, to use in IntentSchema
nonoos_labels = idx2label[~idx2label.label.isin(Config.list_oos_idx)]['label'].values
print("="*80)
print("Original intents to convert to OOS class")
print(idx2label_oos)
print(f"Percentage of original intents to convert to OOS class: {len(idx2label_oos)/len(idx2label)}\n")

oos_labels = idx2label_oos['label'].values
list_sorted_intent_aft_conversion = ['oos' if intent.lower() in oos_labels else intent for intent in sorted_intent]
list_sorted_intent_aft_conversion_deduped = sorted(set(list_sorted_intent_aft_conversion))
print("="*80)
print("Unique intents after converting some to OOS class")
print(list_sorted_intent_aft_conversion_deduped)
print(f"Number of unique intents after converting some to OOS class: {len(list_sorted_intent_aft_conversion_deduped)}\n")



# unique intents - from set to bullet points (to use in prompts)
# bulletpts_intent = "\n".join(f"- {category}" for category in set_intent)
# 2025.06.03: do not show 'oos' in the prompt (to avoid leakage of 'oos' class)
bulletpts_intent = "\n".join(f"- {category}" for category in list_sorted_intent_aft_conversion_deduped if category and (category!='oos'))

# 2025.06.04: fix adjustment if 'oos' is already in the original dataset
int_oos_in_orig_dataset = int('oos' in idx2label.label.values)
adjust_if_oos_not_in_orig_dataset = [0 if int_oos_in_orig_dataset == 1 else 1][0]

print("="*80)
print("sanity check")
print(f"Number of original intents: {len(sorted_intent)}")
print(f"Number of original intents + 1 OOS class (if doesnt exist in original dataset): {len(sorted_intent) + adjust_if_oos_not_in_orig_dataset}")
print(f"Number of original intents to convert to OOS class: {len(idx2label_oos)}")
print(f"Percentage of original intents to convert to OOS class: {len(idx2label_oos)/len(idx2label)}")
print(f"Number of unique intents after converting some to OOS class: {len(list_sorted_intent_aft_conversion_deduped)}")
print(f"Number of original intents + 1 OOS class (if doesnt exist in original dataset) - converted classes: {len(sorted_intent) + adjust_if_oos_not_in_orig_dataset - len(idx2label_oos)}")
print(f"Numbers match: {(len(sorted_intent) + adjust_if_oos_not_in_orig_dataset - len(idx2label_oos)) == len(list_sorted_intent_aft_conversion_deduped)}")
print("Prepared unique intents")
#######################




#######################
# Enforce schema on the model (e.g. allowed list of predicted categories)
#######################

class IntentSchema(BaseModel):
    # dynamically unpack list of categories for different dataset(s)
    category: Literal[*list_sorted_intent_aft_conversion_deduped]
    confidence: float
    
#######################




#######################
# filter after preparing intents
#######################
df = filter100examples_oos(Config.dataset_name, df)
print("Filtered dataset")
print(f"Dimensions: {df.shape}")
print(f"Col names: {df.columns}")
#######################



#######################
# Prompt
#######################
# prompt 2 with less information/compute, improve efficiency
# 2025.06.10 prompt 3 with 5 few shot examples only - notebook O1H1, O1i1
# 2025.06.16 prompt 4 with 5 examples per each known intent (ie non-oos intent) - notebook 01J1
snapshot_download(repo_id="KaiquanMah/open-intent-query-classification", repo_type="space", allow_patterns="*.txt", local_dir=Config.fewshot_examples_dir)
with open(Config.fewshot_examples_dir + Config.fewshot_subdir + Config.fewshot_examples_filename, 'r') as file:
    fewshot_examples = file.read()

def get_prompt(dataset_name, split, question, categories, fewshot_examples):
    
    prompt = f'''
You are an expert in understanding and identifying what users are asking you.

Your task is to analyze an input query from a user and assign the most appropriate category from the following list:
{categories}

Only classify as "oos" (out of scope category) if none of the other categories apply.

Below are several examples to guide your classification:

---
{fewshot_examples}
---

===============================

New Question: {question}

===============================

Provide your final classification in **valid JSON format** with the following structure:
{{
  "category": "your_chosen_category_name",
  "confidence": confidence_level_rounded_to_the_nearest_2_decimal_places
}}


Ensure the JSON has:
- Opening and closing curly braces
- Double quotes around keys and string values
- Confidence as a number (not a string), with maximum 2 decimal places

Do not include any explanations or extra text.
            '''
    return prompt



#######################


#######################
# Model on 1 Dataset
#######################
# Save a list of dictionaries 
# containing a dictionary for each record's
# - predicted category
# - confidence level and
# - original dataframe values

def predict_intent(model_name, df, categories, start_index=0, end_index=None, log_every_n_examples=100):
    start_time = time.time()
    results = []  # Store processed results
    
    # Slice DataFrame based on start/end indices
    if end_index is None:
        subset_df = df.iloc[start_index:]
    else:
        subset_df = df.iloc[start_index:end_index+1]
    
    total_rows = len(subset_df)
    subset_row_count = 0
    
    
    for row in subset_df.itertuples():
        subset_row_count+=1
        prompt = get_prompt(row.dataset, row.split, row.text, categories, fewshot_examples)
        if subset_row_count == 1:
            print("Example of how prompt looks, for the 1st example in this subset of data")
            print(prompt)

            print("Example of how IntentSchema looks")
            print(IntentSchema.model_json_schema())
        
        
        try:
            response = ollama.chat(model=model_name, 
                                   messages=[
                                                {'role': 'user', 'content': prompt}
                                            ],
                                   format = IntentSchema.model_json_schema(),
                                   options = {'temperature': 0},  # Set temperature to 0 for a more deterministic output
                                  )
            msg = response['message']['content']
            parsed = json.loads(msg)

            
            # Safely extract keys with defaults - resolve parsing error
            # maybe LLM did not output a particular key-value pair
            category = parsed.get('category', 'error')
            confidence = parsed.get('confidence', 0.0)
            parsed = {'category': category, 'confidence': confidence}
        except (json.JSONDecodeError, KeyError, Exception) as e:
            parsed = {'category': 'error', 'confidence': 0.0}
        
        # Combine original row data with predictions
        results.append({
            "Index": row.Index,
            "text": row.text,
            "label": row.label,
            "dataset": row.dataset,
            "split": row.split,
            "predicted": parsed['category'],
            "confidence": parsed['confidence']
        })

        
        # Log progress
        if subset_row_count % log_every_n_examples == 0:
            elapsed_time = time.time() - start_time
            
            avg_time_per_row = elapsed_time / subset_row_count
            remaining_rows = total_rows - subset_row_count
            eta = avg_time_per_row * remaining_rows
            
            print(f"Processed original df idx {row.Index} (subset row {subset_row_count}) | "
                  f"Elapsed: {elapsed_time:.2f}s | ETA: {eta:.2f}s")
    
    return results  # Return list of dictionaries
    

print(f"Starting intent classification using {Config.model_name}")
subset_results = predict_intent(Config.model_name, 
                                df, 
                                bulletpts_intent, 
                                start_index = Config.start_index, 
                                end_index = Config.end_index,
                                log_every_n_examples = Config.log_every_n_examples)



# update end_index for filename (if None is used for the end of the df)
# Get the last index of the DataFrame
last_index = df.index[-1] 
# Use last index if Config.end_index is None
end_index = Config.end_index if Config.end_index is not None else last_index



# 2025.05.23 changed from JSON to PKL
# because we are saving list of dictionaries
# Save to PKL
# 2025.06.04 explore changing back to JSON
# with open(f'results_{Config.model_name}_{Config.dataset_name}_{Config.start_index}_{end_index}.pkl', 'wb') as f:
#     pickle.dump(subset_results, f)
with open(f'results_{Config.model_name}_{Config.dataset_name}_{Config.start_index}_{end_index}.json', 'w') as f:
    json.dump(subset_results, f, indent=2)

print("Completed intent classification")


#######################
