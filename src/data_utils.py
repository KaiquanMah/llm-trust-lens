import pandas as pd
from pathlib import Path

def load_dataset_and_labels(dataset_config: dict):
    """
    Loads the dataset and its labels based on the provided configuration.
    This function automatically detects and handles .tsv, .csv, and .json (ie list of dictionaries).

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

    # Define splits and supported file extensions
    splits = ['train', 'dev', 'test']
    supported_extensions = ['.tsv', '.csv', '.json']

    for split in splits:
        found_file_path = None
        # Search for a file with a supported extension for the current split
        for ext in supported_extensions:
            file_path = data_dir / f'{split}{ext}'
            if file_path.exists():
                found_file_path = file_path
                break # Found the file, move on to loading it
        
        if found_file_path:
            print(f"Found '{split}' split file: {found_file_path.name}")
            try:
                # Load the file based on its extension
                if found_file_path.suffix == '.tsv':
                    df = pd.read_csv(found_file_path, sep='\t')
                elif found_file_path.suffix == '.csv':
                    df = pd.read_csv(found_file_path)
                elif found_file_path.suffix == '.json':
                    # It reads a JSON file containing a list of dictionary objects.
                    # e.g., [ {"text": "...", "label": "..."}, ... ]
                    df = pd.read_json(found_file_path)
                
                df['split'] = split
                main_df = pd.concat([main_df, df], ignore_index=True)
                
            except Exception as e:
                print(f"Error parsing {found_file_path}: {e}")
        else:
            # Only print a warning if a file for a split is not found
            print(f"Warning: No data file found for split '{split}' in {data_dir} with extensions {supported_extensions}")

    if main_df.empty:
        print(f"ERROR: No data loaded from {data_dir}. Please check your paths and file names.")
        exit()

    
    print(f"Loading label map from: {label_map_path}")
    try:
        labels_df = pd.read_csv(label_map_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find label map at {label_map_path}")
        exit()

    # Extract the list of labels from the 'label' column
    labels = labels_df['label'].tolist()
    
    # Uses main_df to get the total count of records across all splits
    print(f"Successfully loaded dataset '{dataset_config['name']}' with {len(main_df)} records and {len(labels)} labels.")
    
    return main_df, labels

