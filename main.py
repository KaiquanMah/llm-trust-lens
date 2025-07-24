import subprocess
import sys


# 1. Install libraries from requirements.txt
print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "/workspaces/llm-trust-lens/requirements.txt"], check=True)

# 2. Run setup_ollama.py
print("Starting Ollama setup...")
subprocess.run(
    ["python3", "-m", "src.setup_ollama"],  # Run as a module
    cwd="/workspaces/llm-trust-lens",  # Set working directory to parent of 'src'
    check=True
)

# 3. Run download_dataset.py
print("Downloading dataset...")
subprocess.run(["python3", "/workspaces/llm-trust-lens/download_dataset.py"], check=True)

# 4. Run predict_class.py
print("Running prediction script...")
subprocess.run(["python3", "/workspaces/llm-trust-lens/predict_class.py"], check=True)