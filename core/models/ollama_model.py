"""
Ollama Model Adapter for AI Co-Scientist

This module implements the model adapter for Ollama, allowing the AI Co-Scientist
system to use locally hosted LLMs via the Ollama API.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional, Union

import aiohttp

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class OllamaModel(BaseModel):
    """
    Model adapter for Ollama.
    
    This adapter allows the AI Co-Scientist system to use LLMs hosted locally
    through Ollama (https://ollama.ai/).
    """
    
    def __init__(self, 
                model_name: str = "llama3",
                api_base: str = "http://localhost:11434",
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                timeout: int = 120,
                **kwargs):
        """
        Initialize the Ollama model adapter.
        
        Args:
            model_name: Name of the Ollama model to use
            api_base: Base URL for the Ollama API
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            timeout: Request timeout in seconds
        """
        super().__init__(kwargs)
        self.model_name = model_name
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        logger.info(f"Initialized Ollama model adapter with model: {model_name}")
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: The prompt to generate a response for
            system_prompt: System prompt to guide the model
            temperature: Sampling temperature (overrides default if provided)
            max_tokens: Maximum number of tokens to generate (overrides default if provided)
            stop_sequences: Sequences that stop generation when encountered
            
        Returns:
            The generated response
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        # Prepare the payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temp,
            "stream": False
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
            
        # Add max tokens if provided
        if max_tok:
            payload["max_tokens"] = max_tok
            
        # Add stop sequences if provided
        if stop_sequences:
            payload["stop"] = stop_sequences
            
        logger.debug(f"Generating text with Ollama model {self.model_name}")
        
        try:
            # Make the API request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/api/generate",
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {error_text}")
                        raise Exception(f"Ollama API error: {response.status} - {error_text}")
                    
                    result = await response.json()
                    return result.get("response", "")
                    
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          json_schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate a JSON response to the given prompt.
        
        Args:
            prompt: The prompt to generate a response for
            json_schema: JSON schema for the response
            system_prompt: System prompt to guide the model
            temperature: Sampling temperature (overrides default if provided)
            
        Returns:
            The generated JSON response
        """
        # Enhance the system prompt to ensure JSON output
        json_system_prompt = "You are a helpful assistant that always responds in valid JSON format. "
        if system_prompt:
            json_system_prompt += system_prompt
        
        # Format the schema as part of the prompt
        schema_prompt = f"""
You must respond with valid JSON that matches this schema:
```json
{json.dumps(json_schema, indent=2)}
```

Your response must be valid JSON only, without any explanations, markdown formatting, or text before or after the JSON.

{prompt}
"""
        
        # Generate the response
        response_text = await self.generate(
            prompt=schema_prompt,
            system_prompt=json_system_prompt,
            temperature=temperature,
            stop_sequences=["```"],
            **kwargs
        )
        
        # Extract and parse the JSON
        try:
            # Clean up the response to handle potential formatting issues
            cleaned_response = response_text.strip()
            
            # If the response starts with a markdown code block identifier, remove it
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
                
            # If the response ends with a markdown code block identifier, remove it
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
                
            cleaned_response = cleaned_response.strip()
            
            # Parse the JSON
            return json.loads(cleaned_response)
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            logger.error(f"Raw response: {response_text}")
            
            # Best effort to return something useful
            return {"error": "Failed to parse JSON response", "raw_response": response_text}
    
    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text.
        
        Args:
            text: Text to generate embeddings for (string or list of strings)
            
        Returns:
            List of embeddings (or list of lists for multiple texts)
            
        Note:
            This implementation uses Ollama's embeddings endpoint.
        """
        # Check if we're embedding a single string or multiple
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        
        embeddings = []
        for single_text in texts:
            payload = {
                "model": self.model_name,
                "prompt": single_text
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.api_base}/api/embeddings",
                        json=payload,
                        timeout=self.timeout
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Ollama embeddings API error: {error_text}")
                            raise Exception(f"Ollama API error: {response.status} - {error_text}")
                        
                        result = await response.json()
                        embedding = result.get("embedding", [])
                        embeddings.append(embedding)
                        
            except Exception as e:
                logger.error(f"Error calling Ollama embeddings API: {str(e)}")
                # Return zeros as a fallback
                embeddings.append([0.0] * 384)  # Common embedding dimension, adjust as needed
        
        # Return single embedding or list depending on input
        return embeddings if is_batch else embeddings[0]
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OllamaModel':
        """
        Create an OllamaModel instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured OllamaModel instance
        """
        # Extract Ollama-specific configuration
        ollama_config = config.get("ollama", {})
        
        return cls(
            model_name=ollama_config.get("model_name", "llama3"),
            api_base=ollama_config.get("api_base", "http://localhost:11434"),
            temperature=ollama_config.get("temperature", 0.7),
            max_tokens=ollama_config.get("max_tokens"),
            timeout=ollama_config.get("timeout", 120)
        ) 