import os
import sys
import subprocess
import time
# from src.config import Config # absolute import


def initialize_ollama(model_config: dict):
    """
    Ensures the Ollama server is running and the required model is available.

    1. Checks if the Ollama server is responsive.
    2. If not, starts the server as a background process.
    3. Waits for the server to initialize.
    4. Checks if the specified model is available locally.
    5. If not, pulls the model from the Ollama hub.
    6. Returns an initialized Ollama client.

    Args:
        model_config (dict): A dictionary containing model configuration,
                             expected to have 'model_name' and 'ollama_host'.

    Returns:
        ollama.Client: An initialized Ollama client instance.
    """
    model_name = model_config['model_name']
    host = model_config['ollama_host']
    
    print("--- Initializing Ollama ---")

    
    # Step 1: Check if the Ollama server is already running
    try:
        ollama.Client(host=host).list()
        print("Ollama server is already running.")
    except Exception:
        # Step 2: If not running, start it as a background process
        print("Ollama server not found. Starting it in the background...")
        try:
            # Using Popen for a non-blocking background process
            subprocess.Popen("ollama serve", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Step 3: Wait for the server to initialize
            print("Waiting for server to start... (5 seconds)")
            time.sleep(5)
            # Verify it started
            ollama.Client(host=host).list()
            print("Ollama server started successfully.")
        except Exception as e:
            print("\n--- Ollama Setup Error ---")
            print("Failed to start the Ollama server. Please ensure Ollama is installed correctly.")
            print(f"You can install it by running: curl -fsSL https://ollama.com/install.sh | sh")
            print(f"Error details: {e}")
            print("----------------------------")
            sys.exit(1)

    
    # Step 4: Initialize Ollama client and check for the model
    try:
        client = ollama.Client(host=host)
        # local_models = [m['name'].split(':') for m in client.list()['models']]
        local_models = [m['name'].split(':')[0] for m in client.list()['models']]

        # pull model from Ollama if it has not been downloaded yet
        if model_name not in local_models:
            print(f"Model '{model_name}' not found locally. Pulling from Ollama hub...")
            # Step 5: Pull the model with simple streaming output
            for progress in ollama.pull(model_name, stream=True):
                print(f"\r{progress.get('status', '')}", end='', flush=True)
            print("\nSuccessfully pulled model.")
        else:
            print(f"Model '{model_name}' is already available locally.")
        
        print("--- Ollama Initialization Complete ---")
        return client

    except Exception as e:
        print(f"\n--- Ollama Connection Error ---")
        print(f"Failed to connect to Ollama or pull the model even after starting the server.")
        print(f"Error details: {e}")
        print("---------------------------------")
        sys.exit(1)


