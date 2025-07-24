from src.config import Config
import os
import subprocess
target_dir = Config.target_dir # data directory to clone into
cloned_data_dir = Config.cloned_data_dir

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# do not clone dataset repo if cloned data folder exists
if os.path.exists(cloned_data_dir):
    print("Dataset has already been downloaded. If this is incorrect, please delete the Adaptive-Decision-Boundary 'data' folder.")
else:
    # Clone the repository
    subprocess.run(["git",
                    "clone",
                    "https://github.com/thuiar/Adaptive-Decision-Boundary.git",
                    target_dir
                   ])