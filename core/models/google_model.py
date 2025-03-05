"""
Google Gemini Model Adapter for AI Co-Scientist

Implements the BaseModel interface for Google's Gemini models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union

import google.generativeai as genai

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class GoogleModel(BaseModel):
    """Google Gemini implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                model_name: str = "gemini-pro", 
                max_tokens: int = 8192,
                temperature: float = 0.7):
        """
        Initialize the Google Gemini model adapter.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY environment variable)
            model_name: Model name to use (e.g., "gemini-pro")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("Google API key not provided. Please set GOOGLE_API_KEY environment variable.")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Configure the Google API
        genai.configure(api_key=self.api_key)
        
        # Get the generative model
        self.model = genai.GenerativeModel(self.model_name)
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using Google Gemini API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional Gemini-specific parameters
            
        Returns:
            Generated text response from the model
        """
        # Set parameters, using instance defaults if not specified
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "stop_sequences": stop_sequences or [],
            **kwargs
        }
        
        try:
            # Create the content with the system prompt if provided
            if system_prompt:
                content = [
                    {"role": "system", "parts": [{"text": system_prompt}]},
                    {"role": "user", "parts": [{"text": prompt}]}
                ]
                response = await self.model.generate_content_async(
                    content, 
                    generation_config=generation_config
                )
            else:
                response = await self.model.generate_content_async(
                    prompt, 
                    generation_config=generation_config
                )
            
            return response.text
        except Exception as e:
            logger.error(f"Error generating response from Google Gemini: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional Gemini-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """
        # Create a prompt that includes the JSON schema requirement
        json_prompt = f"{prompt}\n\nRespond with valid JSON that matches this schema: {json.dumps(json_schema)}"
        
        # Create a system prompt that emphasizes JSON output if provided
        enhanced_system_prompt = system_prompt
        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nYou must respond with valid JSON that strictly follows the provided schema."
        else:
            enhanced_system_prompt = "You must respond with valid JSON that strictly follows the provided schema."
        
        try:
            # Generate the response
            response_text = await self.generate(
                prompt=json_prompt,
                system_prompt=enhanced_system_prompt,
                temperature=temperature,
                **kwargs
            )
            
            # Extract JSON from the response
            # Sometimes the model might wrap JSON in markdown code blocks, so we need to handle that
            json_text = response_text
            
            # Check if response is wrapped in markdown code block
            if "```json" in json_text:
                json_text = json_text.split("```json")[1]
                if "```" in json_text:
                    json_text = json_text.split("```")[0]
            elif "```" in json_text:
                json_text = json_text.split("```")[1]
                if "```" in json_text:
                    json_text = json_text.split("```")[0]
            
            # Parse the JSON
            return json.loads(json_text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}, Response: {response_text}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Google Gemini: {e}")
            raise
    
    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s) using Google's embedding model.
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            # Ensure text is a list
            text_list = [text] if isinstance(text, str) else text
            
            # Get the embedding model - currently using text-embedding-gecko for Gemini
            embedding_model = genai.get_model("embedding-001")
            
            # Get embeddings for each text
            embeddings = []
            for t in text_list:
                result = embedding_model.embed_content(t)
                embeddings.append(result["embedding"])
            
            # Return a single embedding if input was a string
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings from Google Gemini: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GoogleModel':
        """
        Create a Google model instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured GoogleModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            model_name=config.get("model_name", "gemini-pro"),
            max_tokens=config.get("max_tokens", 8192),
            temperature=config.get("temperature", 0.7)
        ) 