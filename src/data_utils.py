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
    data_dir = project_root / dataset_config['path']
    label_map_path = project_root / dataset_config['label_map_path']

    
    print(f"Loading all splits from directory: {data_dir}")
    main_df = pd.DataFrame()
    for split in ['train', 'dev', 'test']:
        file_path = data_dir / f'{split}.tsv'
        if file_path.exists():
            try:
                # --- THIS IS THE CORRECTED LINE ---
                # It correctly reads a tab-separated file that has a header row.
                df = pd.read_csv(file_path, sep='\t')
                
                df['split'] = split
                main_df = pd.concat([main_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
        else:
            print(f"Warning: {split}.tsv not found in {data_dir}")

    if main_df.empty:
        print(f"ERROR: No data loaded from {data_dir}. Please check your paths.")
        exit()

    
    print(f"Loading label map from: {label_map_path}")
    try:
        labels_df = pd.read_csv(label_map_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find label map at {label_map_path}")
        exit()

    # Extract the list of labels from the 'label' column
    labels = labels_df['label'].tolist()
    
    print(f"Successfully loaded dataset '{dataset_config['name']}' with {len(df)} records and {len(labels)} labels.")
    
    return main_df, labels

