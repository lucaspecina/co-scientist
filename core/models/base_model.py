"""
Base Model Interface for AI Co-Scientist

This module defines the abstract base class that all model adapters must implement.
This enables the system to work with different LLM providers in a model-agnostic way.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union


class BaseModel(ABC):
    """Abstract base class for all LLM adapters."""
    
    @abstractmethod
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response from the model
        """
        pass
    
    @abstractmethod
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
            **kwargs: Additional provider-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """
        pass
    
    @abstractmethod
    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseModel':
        """
        Create a model instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured model instance
        """
        pass 