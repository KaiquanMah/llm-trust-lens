class Config:
    target_dir = '/data' # data directory to clone into
    cloned_data_dir = target_dir + '/data'
    prediction_dir = target_dir + '/prediction'
    dataset_name = 'banking' # UPDATE options: 'banking', 'stackoverflow', 'oos'
    idx2label_target_dir = 'idx2label'
    idx2label_filename_hf = 'banking77_idx2label.csv' # UPDATE options: banking77_idx2label.csv, stackoverflow_idx2label.csv, clinc150_oos_idx2label.csv
    fewshot_examples_dir = '/fewshot'
    fewshot_subdir = '/fewshot-1example-per-nonoos/'
    fewshot_examples_filename = 'banking_only4notoos.txt' # UPDATE options: banking_25perc_oos.txt, stackoverflow_25perc_oos.txt, oos_25perc_oos.txt
    list_oos_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72] # UPDATE gathered from within the team - for reproducible, comparable results with other open intent classification approaches
    model_name = 'llama3.2:3b'
    start_index=0 # eg: 0, 10001, 11851
    end_index=None # eg: 10, 10000, 11850 or None (use end_index=None to process the full dataset)
    log_every_n_examples=10 # 2
    force_oos = True  # NEW: Add flag to force dataset to contain 'oos' class for the last class value (sorted alphabetically), if 'oos' class does not exist in the original dataset
    filter_oos_qns_only = True
    n_oos_qns = 100
    first_class_banking = 'activate_my_card' # following idx2label
    first_class_stackoverflow = 'wordpress' # following idx2label
    first_class_oos = 'oos'