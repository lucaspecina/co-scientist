"""
Ollama Model Adapter for AI Co-Scientist

This module provides integration with the Ollama API for running LLMs locally.
Ollama can run various models like Llama, Mistral, and others locally.
Visit https://ollama.ai for more information.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union

import aiohttp

from .base_model import BaseModel, ModelError, ServiceUnavailableError

logger = logging.getLogger(__name__)

class OllamaModel(BaseModel):
    """
    Model adapter for Ollama.
    
    This adapter allows the AI Co-Scientist to use locally hosted LLMs via Ollama.
    It implements all required methods from the BaseModel interface.
    """
    
    def __init__(self, 
                model_name: str = "llama3",
                api_base: str = "http://localhost:11434",
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                timeout: int = 120,
                context_window: int = 8192,  # Default context window size
                api_timeout: int = 60,      # HTTP timeout
                retry_count: int = 3,       # Number of retries
                **kwargs):
        """
        Initialize the Ollama model adapter.
        
        Args:
            model_name: Name of the model to use (e.g., "llama3", "mistral", "vicuna")
            api_base: Base URL for the Ollama API
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            timeout: Timeout for generation in seconds
            context_window: Size of the context window for the model
            api_timeout: Timeout for API calls in seconds
            retry_count: Number of times to retry failed requests
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_base = api_base.rstrip('/')  # Remove trailing slash if present
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.context_window = context_window
        self.api_timeout = api_timeout
        self.retry_count = retry_count
        
        # Initialize metrics
        self.last_response_time = 0.0
        self.connection_errors = 0
        
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text from the Ollama model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stop_sequences: List of sequences to stop generation at
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
            
        Raises:
            ModelError: If the API call fails
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        # Prepare payload
        payload = {
            "model": self.model_name,
            "prompt": json.dumps(prompt) if isinstance(prompt, dict) else prompt,
            "temperature": temp,
            "options": {}
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
            
        # Add max tokens if provided
        if max_tok:
            payload["options"]["num_predict"] = max_tok
            
        # Add stop sequences if provided
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences
        
        # Use the timed_generate helper for error handling and metrics
        return await self._timed_generate(self._generate_internal, payload)
        
    async def _generate_internal(self, payload: Dict[str, Any]) -> str:
        """
        Internal method to handle the API call to Ollama.
        
        Args:
            payload: Request payload
            
        Returns:
            Generated text
            
        Raises:
            ModelError: If the API call fails
        """
        url = f"{self.api_base}/api/generate"
        response_text = ""
        
        for attempt in range(self.retry_count):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=self.api_timeout
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_text = result.get("response", "")
                            return response_text
                        else:
                            error_text = await response.text()
                            logger.error(f"Ollama API error: {response.status} - {error_text}")
                            if attempt < self.retry_count - 1:
                                wait_time = (attempt + 1) * 2  # Exponential backoff
                                logger.info(f"Retrying in {wait_time} seconds (attempt {attempt + 1}/{self.retry_count})")
                                await asyncio.sleep(wait_time)
                            else:
                                raise ModelError(f"Ollama API error: {response.status} - {error_text}")
            except aiohttp.ClientError as e:
                self.connection_errors += 1
                logger.error(f"Connection error: {str(e)}")
                if attempt < self.retry_count - 1:
                    wait_time = (attempt + 1) * 2
                    logger.info(f"Retrying in {wait_time} seconds (attempt {attempt + 1}/{self.retry_count})")
                    await asyncio.sleep(wait_time)
                else:
                    raise ServiceUnavailableError(f"Failed to connect to Ollama API: {str(e)}")
            except asyncio.TimeoutError:
                logger.error("Request timed out")
                if attempt < self.retry_count - 1:
                    wait_time = (attempt + 1) * 2
                    logger.info(f"Retrying in {wait_time} seconds (attempt {attempt + 1}/{self.retry_count})")
                    await asyncio.sleep(wait_time)
                else:
                    raise ServiceUnavailableError("Ollama API request timed out")
                
        return response_text  # Return empty string if all retries fail
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the Ollama model.
        
        Args:
            prompt: User prompt
            schema: JSON schema that the output should conform to
            system_prompt: Optional system instructions
            temperature: Override default temperature
            default: Default JSON to return if generation fails
            **kwargs: Additional parameters
            
        Returns:
            Generated JSON as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        # Create a system prompt that includes the schema and formatting instructions
        schema_json = json.dumps(schema, indent=2)
        enhanced_system_prompt = (
            f"{system_prompt}\n\n" if system_prompt else ""
        ) + (
            f"You must respond with valid JSON that conforms to this schema:\n{schema_json}\n\n"
            "Do not include any text before or after the JSON. "
            "The response should be parseable by json.loads()."
        )
        
        # Try to generate valid JSON
        for attempt in range(3):  # Make 3 attempts to get valid JSON
            try:
                # Generate the response
                response = await self.generate(
                    prompt=prompt,
                    system_prompt=enhanced_system_prompt,
                    temperature=temperature,
                    **kwargs
                )
                
                # Try to extract and parse JSON from the response
                json_str = self._extract_json(response)
                if json_str:
                    result = json.loads(json_str)
                    # Validate against the schema (basic validation)
                    if self._validate_json_structure(result, schema):
                        return result
                    else:
                        logger.warning("Generated JSON doesn't match the schema, retrying...")
                else:
                    logger.warning("Failed to extract JSON from response, retrying...")
                
                # If we got here, the JSON was invalid or didn't match the schema
                if attempt < 2:  # Don't sleep on the last attempt
                    await asyncio.sleep(1)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {str(e)}, retrying...")
                if attempt < 2:
                    await asyncio.sleep(1)
        
        # If we got here, all attempts failed
        if default is not None:
            logger.warning("All attempts to generate valid JSON failed, using default")
            return default
        else:
            raise ModelError("Failed to generate valid JSON after multiple attempts")
    
    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the input text using Ollama.
        
        Args:
            text: Text or list of texts to embed
            
        Returns:
            Embedding vector(s)
            
        Raises:
            ModelError: If embedding generation fails
        """
        url = f"{self.api_base}/api/embeddings"
        
        if isinstance(text, list):
            # Handle batch embedding - will return a list of embeddings
            results = []
            for single_text in text:
                embedding = await self._embed_single(url, single_text)
                results.append(embedding)
            return results
        else:
            # Handle single text embedding
            return await self._embed_single(url, text)
    
    async def _embed_single(self, url: str, text: str) -> List[float]:
        """
        Generate embeddings for a single text.
        
        Args:
            url: API endpoint
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            ModelError: If embedding generation fails
        """
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        for attempt in range(self.retry_count):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=self.api_timeout
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result.get("embedding", [])
                        else:
                            error_text = await response.text()
                            logger.error(f"Ollama embeddings API error: {response.status} - {error_text}")
                            if attempt < self.retry_count - 1:
                                await asyncio.sleep((attempt + 1) * 2)
                            else:
                                raise ModelError(f"Ollama embeddings API error: {response.status} - {error_text}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.retry_count - 1:
                    await asyncio.sleep((attempt + 1) * 2)
                else:
                    raise ServiceUnavailableError(f"Failed to connect to Ollama embeddings API: {str(e)}")
        
        # If we get here, all retries failed
        raise ModelError("All attempts to generate embeddings failed")
    
    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extract JSON from text that might contain other content.
        
        Args:
            text: Text that might contain JSON
            
        Returns:
            Extracted JSON string or empty string if no JSON found
        """
        # Try to find JSON delimited by triple backticks
        import re
        json_pattern = r"```(?:json)?\n([\s\S]*?)\n```"
        match = re.search(json_pattern, text)
        if match:
            return match.group(1).strip()
        
        # If that fails, look for opening/closing braces
        brace_pattern = r"\{[\s\S]*\}"
        match = re.search(brace_pattern, text)
        if match:
            return match.group(0).strip()
        
        # If still no match, return the entire text as a last resort
        # (it might be valid JSON without any formatting)
        return text.strip()
    
    @staticmethod
    def _validate_json_structure(json_obj: Any, schema: Dict[str, Any]) -> bool:
        """
        Simple JSON structure validation against a schema.
        
        Args:
            json_obj: JSON object to validate
            schema: JSON schema
            
        Returns:
            True if the structure matches the basic requirements
        """
        # This is a very basic validator - for production, consider using jsonschema
        try:
            # Check if we have an object when the schema expects one
            if schema.get("type") == "object" and not isinstance(json_obj, dict):
                return False
                
            # Check if we have an array when the schema expects one
            if schema.get("type") == "array" and not isinstance(json_obj, list):
                return False
                
            # Check required properties if it's an object
            if isinstance(json_obj, dict) and "properties" in schema:
                required = schema.get("required", [])
                for prop in required:
                    if prop not in json_obj:
                        return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating JSON: {str(e)}")
            return False
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OllamaModel':
        """
        Create an OllamaModel instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured OllamaModel instance
        """
        # Get the ollama-specific configuration
        ollama_config = config.get("ollama", {})
        
        return cls(
            model_name=ollama_config.get("model_name", "llama3"),
            api_base=ollama_config.get("api_base", "http://localhost:11434"),
            temperature=ollama_config.get("temperature", 0.7),
            max_tokens=ollama_config.get("max_tokens"),
            timeout=ollama_config.get("timeout", 120),
            context_window=ollama_config.get("context_window", 8192),
            api_timeout=ollama_config.get("api_timeout", 60),
            retry_count=ollama_config.get("retry_count", 3)
        ) 