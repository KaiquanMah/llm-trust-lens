"""
Utility functions for interacting with Google's Gemini API.
"""
import os
from typing import Dict, Any
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_fixed

class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self, api_config: Dict[str, Any]):
        """
        Initialize the Gemini client.
        
        Args:
            api_config: Dictionary containing API configuration
        """
        api_key = os.getenv(api_config['api_key_env'])
        if not api_key:
            raise ValueError(f"Environment variable {api_config['api_key_env']} not set")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = api_config['model_name']
        self.temperature = api_config.get('temperature', 0)
        self.retry_config = api_config.get('retry', {
            'max_attempts': 3,
            'wait_seconds': 30
        })

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(30))
    def generate_content(self, prompt: str, response_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate content using the Gemini API with retry logic.
        
        Args:
            prompt: The input prompt
            response_schema: JSON schema for response validation
            
        Returns:
            Dict containing the parsed response
        """
        config = {
            "temperature": self.temperature,
            "response_mime_type": "application/json",
            "response_schema": response_schema
        }
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )

        # Return the parsed response
        if hasattr(response, 'parsed'):
            parsed = response.parsed
            # Safely extract keys with defaults
            category = parsed.get('category', 'error')
            confidence = parsed.get('confidence', 0.0)
            return {'category': category, 'confidence': confidence}
        else:
            return {'category': 'error', 'confidence': 0.0}
