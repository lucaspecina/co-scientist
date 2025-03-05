"""
Azure OpenAI Model Adapter for AI Co-Scientist

Implements the BaseModel interface for Azure OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
import re

import openai
from openai import AsyncAzureOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AzureOpenAIModel(BaseModel):
    """Azure OpenAI model for generating text responses."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_version: Optional[str] = None,
                endpoint: Optional[str] = None,
                deployment_id: Optional[str] = None,
                embedding_deployment: Optional[str] = None,
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60,
                debug: bool = False):
        """Initialize the Azure OpenAI model.
        
        Args:
            api_key: Azure OpenAI API key
            api_version: Azure OpenAI API version
            endpoint: Azure OpenAI endpoint
            deployment_id: Azure OpenAI deployment ID (model name)
            embedding_deployment: Azure OpenAI embedding deployment ID
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            debug: Whether to enable debug logging
        """
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
        
        # Get endpoint and clean it if necessary
        raw_endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if raw_endpoint:
            # Extract just the base URL if it contains more information
            # e.g. "https://example.openai.azure.com/openai/deployments/..." -> "https://example.openai.azure.com"
            base_url_match = re.match(r'(https://[^/]+)', raw_endpoint)
            if base_url_match:
                self.endpoint = base_url_match.group(1)
            else:
                self.endpoint = raw_endpoint.rstrip('/')
        else:
            self.endpoint = None
        
        # Get deployment_id from environment if not provided
        self.deployment_id = deployment_id or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        
        self.embedding_deployment = embedding_deployment or os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", self.deployment_id)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.debug = debug
        
        # Set up logging
        self.logger = logging.getLogger("azure_openai_model")
        if debug:
            self.logger.setLevel(logging.DEBUG)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                ))
                self.logger.addHandler(handler)
        
        if not all([api_key, api_version, endpoint]):
            raise ValueError("Azure OpenAI API key, API version, and endpoint must be provided")
        
        try:
            if self.debug:
                self.logger.debug(f"Initializing AsyncAzureOpenAI client with:")
                self.logger.debug(f"  Endpoint: {endpoint}")
                self.logger.debug(f"  API Version: {api_version}")
                self.logger.debug(f"  Deployment ID: {deployment_id}")
            
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
                timeout=timeout
            )
            
        except ImportError:
            raise ImportError("Please install openai package: pip install openai>=1.0.0")
        except Exception as e:
            raise ValueError(f"Failed to initialize Azure OpenAI client: {str(e)}")

    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """Generate a response to the given prompt.
        
        Args:
            prompt: The prompt to generate a response for
            system_prompt: Optional system prompt for setting context
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum number of tokens to generate (overrides default)
            stop_sequences: Sequences that stop generation
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            The generated response
        """
        system_prompt = system_prompt or "You are a helpful AI assistant."
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Construct arguments dictionary
            args = {
                "messages": messages,
                "model": self.deployment_id,  # In Azure, deployment_id is passed as model
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Add stop sequences if provided
            if stop_sequences:
                args["stop"] = stop_sequences
                
            # Add any additional kwargs
            args.update(kwargs)
            
            if self.debug:
                self.logger.debug(f"Sending request to Azure OpenAI API:")
                self.logger.debug(f"  Deployment: {self.deployment_id}")
                self.logger.debug(f"  Temperature: {temperature}")
                self.logger.debug(f"  Max Tokens: {max_tokens}")
                request_url = f"{self.endpoint}/openai/deployments/{self.deployment_id}/chat/completions"
                self.logger.debug(f"  Expected URL: {request_url}?api-version={self.api_version}")
            
            response = await self.client.chat.completions.create(**args)
            
            # Extract the content from the response
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            return ""
            
        except Exception as e:
            error_msg = str(e)
            
            # Improve error reporting for common issues
            if "404" in error_msg:
                error_msg = (
                    f"Deployment '{self.deployment_id}' not found. Make sure this deployment exists "
                    f"in your Azure OpenAI resource and you're using the correct API version ({self.api_version})."
                )
            elif "401" in error_msg:
                error_msg = "Authentication failed. Check your Azure OpenAI API key."
            elif "429" in error_msg:
                error_msg = "Rate limit exceeded. Your Azure OpenAI resource might be hitting its quota limits."
                
            if self.debug:
                self.logger.error(f"Azure OpenAI API Error: {error_msg}")
                
            raise Exception(f"Azure OpenAI API Error: {error_msg}")

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
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """
        # Create a system prompt that includes the JSON schema if not provided
        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"
            
        # Use the response_format parameter to enforce JSON
        try:
            api_args = {
                "model": self.deployment_id,
                "messages": [
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature if temperature is not None else self.temperature,
                "response_format": {"type": "json_object"},
            }
            
            # Add any additional kwargs
            api_args.update(kwargs)
            
            response = await self.client.chat.completions.create(**api_args)
            
            result_text = response.choices[0].message.content
            return json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from Azure OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise
    
    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for the given text.
        
        Args:
            text: Text to embed (string or list of strings)
            
        Returns:
            Embeddings as a list of floats (single text) or list of list of floats (multiple texts)
        """
        try:
            # Handle both single string and list of strings
            is_single = isinstance(text, str)
            texts = [text] if is_single else text
            
            if self.debug:
                self.logger.debug(f"Generating embeddings for {len(texts)} text(s)")
                self.logger.debug(f"Using embedding deployment: {self.embedding_deployment}")
                
            # Create embeddings using the embedding deployment
            response = await self.client.embeddings.create(
                model=self.embedding_deployment,  # Use the embedding deployment
                input=texts
            )
            
            if hasattr(response, 'data') and response.data:
                embeddings = [item.embedding for item in response.data]
                
                # Return a single list for a single input, otherwise return the list of lists
                return embeddings[0] if is_single else embeddings
            
            raise ValueError("Failed to generate embeddings: Empty response data")
            
        except Exception as e:
            error_msg = str(e)
            
            # Improve error reporting for embedding-specific issues
            if "404" in error_msg:
                error_msg = (
                    f"Embedding deployment '{self.embedding_deployment}' not found. Make sure this deployment exists "
                    f"in your Azure OpenAI resource and you're using the correct API version ({self.api_version})."
                )
                
            if self.debug:
                self.logger.error(f"Azure OpenAI Embedding Error: {error_msg}")
                
            raise Exception(f"Azure OpenAI Embedding Error: {error_msg}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AzureOpenAIModel':
        """Create an AzureOpenAIModel from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            AzureOpenAIModel instance
        """
        required_keys = ['api_key', 'api_version', 'endpoint']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")
        
        # Extract optional parameters with defaults
        deployment_id = config.get('deployment_id', 'gpt-4')
        embedding_deployment = config.get('embedding_deployment', deployment_id)
        max_tokens = config.get('max_tokens', 4096)
        temperature = config.get('temperature', 0.7)
        timeout = config.get('timeout', 60)
        debug = config.get('debug', False)
        
        return cls(
            api_key=config['api_key'],
            api_version=config['api_version'],
            endpoint=config['endpoint'],
            deployment_id=deployment_id,
            embedding_deployment=embedding_deployment,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            debug=debug
        ) 