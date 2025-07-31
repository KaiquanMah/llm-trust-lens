"""
Utilities for interacting with the Nebius API for LLM experiments.
Similar to ollama_utils.py but specialized for API-based models.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed
import json
from typing import Dict, Any, Optional
from experiment_common import create_intent_schema

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    print("Warning: .env file not found in project root. Please create one from .env.example")


def initialize_nebius_client(model_config: dict) -> OpenAI:
    """
    Initialize the Nebius API client with proper configuration.

    Args:
        model_config (dict): Configuration dictionary containing API settings
                           Expected keys:
                           - api_config.base_url: Nebius API endpoint
                           - api_config.api_key_env: Name of env var containing API key

    Returns:
        OpenAI: Configured OpenAI client for Nebius API

    Raises:
        ValueError: If API key environment variable is not set
        ConnectionError: If unable to connect to Nebius API
    """
    api_key_env = model_config['api_config']['api_key_env']
    base_url = model_config['api_config']['base_url']
    
    # Check for API key
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"API key environment variable {api_key_env} not set")

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        return client
    except Exception as e:
        raise ConnectionError(f"Failed to initialize Nebius API client: {str(e)}")


def create_retry_decorator(config: dict):
    """
    Creates a retry decorator based on configuration.
    
    Args:
        config (dict): Configuration containing retry settings
                      Expected keys:
                      - retry.max_attempts
                      - retry.wait_seconds
    
    Returns:
        decorator: Retry decorator with configured settings
    """
    max_attempts = config['retry']['max_attempts']
    wait_seconds = config['retry']['wait_seconds']
    
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_fixed(wait_seconds),
        reraise=True
    )


def format_messages(prompt: str) -> list:
    """
    Format a prompt into the message structure expected by the Nebius API.
    
    Args:
        prompt (str): The full prompt text
        
    Returns:
        list: List of message dictionaries in the format expected by the API
    """
    return [{"role": "user", "content": prompt}]


@retry(stop=stop_after_attempt(3), wait=wait_fixed(30))
def predict_with_nebius(
    client: OpenAI,
    prompt: str,
    model_name: str,
    response_schema: Any,
    config: dict
) -> Dict[str, Any]:
    """
    Make a prediction using the Nebius API with retry logic.
    
    Args:
        client: Initialized OpenAI client for Nebius
        prompt: The input prompt
        model_name: Name of the model to use
        response_schema: Pydantic schema for response validation
        config: Configuration dictionary with model parameters
        
    Returns:
        dict: Parsed API response
        
    Raises:
        Exception: If API call fails after all retries
    """
    try:
        response = client.beta.chat.completions.parse(
            model=f"Qwen/{model_name}",
            messages=format_messages(prompt),
            response_format=response_schema,
            seed=config['api_config']['seed'],
            temperature=config['api_config']['temperature']
        )
        
        # Extract and validate the response
        content = response.choices[0].message.content
        
        # Parse the response content
        if isinstance(content, str):
            parsed_content = json.loads(content)
        else:
            parsed_content = content
            
        return parsed_content
        
    except Exception as e:
        print(f"API Error: {str(e)}")
        if hasattr(e, 'code'):
            print(f"Status Code: {e.code}")
        if hasattr(e, 'details'):
            print(f"Details: {e.details}")
        raise
