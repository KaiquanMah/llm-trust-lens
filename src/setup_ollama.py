import os
import subprocess
import time
from src.config import Config # absolute import

# 1. Install Ollama (if not already installed)
try:
    # Check if Ollama is already installed
    subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    print("Ollama is already installed.")
except FileNotFoundError:
    print("Installing Ollama...")
    subprocess.run("curl -fsSL https://ollama.com/install.sh  | sh", shell=True, check=True)

# 2. Start Ollama server in the background
print("Starting Ollama server...")
process = subprocess.Popen("ollama serve", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Wait for the server to initialize
time.sleep(5)


# 3. Pull the model
model_name = Config.model_name
print(f"Pulling {model_name} model...")
subprocess.run(["ollama", "pull", model_name], check=True)

# 4. Install Python client
subprocess.run(["pip", "install", "ollama"], check=True)

print("Ollama setup complete!")