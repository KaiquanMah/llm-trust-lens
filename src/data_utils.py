import pandas as pd
from pathlib import Path

def load_dataset_and_labels(dataset_config: dict):
    """
    Loads the dataset and its labels based on the provided configuration.

    Args:
        dataset_config: A dictionary loaded from a dataset YAML file.

    Returns:
        A tuple containing (pandas.DataFrame, list_of_labels).
    """
  
    project_root = Path(__file__).parent.parent
    
    # Get file paths from the config and resolve them relative to the project root
    data_path = project_root / dataset_config['path']
    label_map_path = project_root / dataset_config['label_map_path']
    
    print(f"Loading data from: {data_path}")
    print(f"Loading label map from: {label_map_path}")

    # Load the main data and the label map
    try:
        df = pd.read_csv(data_path)
        labels_df = pd.read_csv(label_map_path)
    except FileNotFoundError as e:
        print(f"ERROR: Could not find a file. Make sure your paths in the config are correct.")
        print(f"Details: {e}")
        exit() # Exit the script if data is not found

    # Extract the list of labels from the 'label' column
    labels = labels_df['label'].tolist()
    
    print(f"Successfully loaded dataset '{dataset_config['name']}' with {len(df)} records and {len(labels)} labels.")
    
    return df, labels

