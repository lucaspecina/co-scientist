"""
Model Factory for AI Co-Scientist

Provides a factory pattern implementation for creating model instances
based on configuration settings. This enables easy switching between
different LLM providers.
"""

import importlib
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple, Callable

from .base_model import BaseModel

logger = logging.getLogger(__name__)

# Map provider names to their module paths
MODEL_PROVIDER_MAP = {
    "openai": "core.models.openai_model.OpenAIModel",
    "azure_openai": "core.models.azure_openai_model.AzureOpenAIModel",
    "anthropic": "core.models.anthropic_model.AnthropicModel",
    "google": "core.models.google_model.GoogleModel",
    "huggingface": "core.models.huggingface_model.HuggingFaceModel",
    "local": "core.models.local_model.LocalModel",
    "ollama": "core.models.ollama_model.OllamaModel",
}


class ModelFactory:
    """Factory for creating LLM instances based on configuration."""
    
    # Dictionary to store registered model classes
    registered_models = {}
    
    # Dictionary to cache model instances for better performance
    _model_cache = {}
    
    # Statistics for monitoring model performance
    _model_stats = {}
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on the provided configuration.
        
        Args:
            config: Configuration dictionary containing model settings
                Should have at least 'provider' key specifying which model to use
                
        Returns:
            Configured BaseModel instance
            
        Raises:
            ValueError: If provider is not supported or module cannot be loaded
        """
        # Get the provider name, defaulting to the default_provider if specific provider not given
        provider = config.get("provider")
        if provider == "default" or provider is None:
            provider = config.get("default_provider", "azure_openai")
        
        # Special handling for Azure OpenAI
        if provider == "azure_openai":
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

            azure_config = {
                "api_key": api_key,
                "api_version": api_version,
                "endpoint": endpoint,
                "deployment_id": deployment_name
            }

            if "models" in config and "azure_openai" in config["models"]:
                azure_model_config = config["models"]["azure_openai"]
                for key, value in azure_model_config.items():
                    if key not in azure_config:
                        azure_config[key] = value

            config_to_use = azure_config
        else:
            # For other providers, use the config as is
            config_to_use = config
        
        # Create a cache key based on the provider and relevant configuration
        cache_key = ModelFactory._create_cache_key(provider, config_to_use)
        
        # Check if we have a cached instance
        if cache_key in ModelFactory._model_cache:
            logger.debug(f"Using cached model instance for provider: {provider}")
            return ModelFactory._model_cache[cache_key]
            
        if provider not in MODEL_PROVIDER_MAP and provider not in ModelFactory.registered_models:
            raise ValueError(f"Unsupported model provider: {provider}")
        
        # Check if we have a registered model class
        if provider in ModelFactory.registered_models:
            model_class = ModelFactory.registered_models[provider]
        else:
            # Import the module and get the class
            try:
                module_path, class_name = MODEL_PROVIDER_MAP[provider].rsplit(".", 1)
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"Error loading model module for provider {provider}: {e}")
                raise ValueError(f"Could not load model implementation for provider: {provider}")
        
        # Create and return the model instance
        try:
            model = model_class.from_config(config_to_use)
            # Cache the model instance
            ModelFactory._model_cache[cache_key] = model
            # Initialize stats for this model
            ModelFactory._model_stats[cache_key] = {
                "created_at": time.time(),
                "success_count": 0,
                "failure_count": 0,
                "total_tokens": 0,
                "total_latency": 0.0
            }
            return model
        except Exception as e:
            logger.error(f"Error creating model instance for provider {provider}: {str(e)}")
            raise
    
    @staticmethod
    def create_model_for_agent(agent_type: str, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance for a specific agent type based on configuration.
        
        Args:
            agent_type: Type of agent (e.g., "generation", "reflection")
            config: Configuration dictionary containing model settings
                
        Returns:
            Configured BaseModel instance for the agent
        """
        # Get the model provider from the agent config
        model_provider = config.get("model_provider", "default")
        
        # Access the global config if available, otherwise use the config as is
        global_config = config.get("_global_config", {})
        
        # Create a new model config
        model_config = {}
        
        # Get provider-specific config
        if model_provider != "default" and model_provider in global_config.get("models", {}):
            provider_config = global_config.get("models", {}).get(model_provider, {})
            model_config.update(provider_config)
        
        # If model_provider is "default", use the default provider from global config
        if model_provider == "default":
            model_provider = global_config.get("models", {}).get("default_provider", "azure_openai")
        
        # Set the provider in the config
        model_config["provider"] = model_provider
        
        # Add global defaults
        model_config["default_provider"] = global_config.get("models", {}).get("default_provider", "azure_openai")
        
        # Override with agent-specific settings
        for key in ["temperature", "max_tokens"]:
            if key in config:
                model_config[key] = config[key]
        
        # Add any additional settings passed
        for k, v in config.items():
            if k not in ["_global_config", "count", "prompt_template"] and k not in model_config:
                model_config[k] = v
                
        return ModelFactory.create_model(model_config)
    
    @classmethod
    def register_model(cls, provider_name: str, model_class) -> None:
        """
        Register a custom model class for a provider.
        
        Args:
            provider_name: Name of the provider
            model_class: The model class to register
        """
        cls.registered_models[provider_name] = model_class
        logger.info(f"Registered custom model class for provider: {provider_name}")
    
    @classmethod
    def create_with_fallbacks(cls, 
                             config: Dict[str, Any], 
                             fallback_providers: List[str] = None,
                             max_retries: int = 3) -> Tuple[BaseModel, str]:
        """
        Create a model with fallback options in case the primary provider fails.
        
        Args:
            config: Configuration dictionary containing model settings
            fallback_providers: List of provider names to try if the primary fails
            max_retries: Maximum number of retries per provider
            
        Returns:
            Tuple of (model instance, provider name used)
            
        Raises:
            ValueError: If all providers fail
        """
        primary_provider = config.get("provider", config.get("default_provider", "azure_openai"))
        
        # If no fallbacks specified, use the default fallback order
        if fallback_providers is None:
            # Start with all available providers, remove the primary, and prioritize based on reliability
            fallback_providers = [p for p in MODEL_PROVIDER_MAP.keys() if p != primary_provider]
            # Prioritize OpenAI as first fallback after Azure OpenAI
            if "openai" in fallback_providers:
                fallback_providers.remove("openai")
                fallback_providers.insert(0, "openai")
            if "ollama" in fallback_providers:
                # Prioritize Ollama after OpenAI since it's local and reliable
                fallback_providers.remove("ollama")
                fallback_providers.insert(1, "ollama")
        
        # Try the primary provider first
        providers_to_try = [primary_provider] + fallback_providers
        last_error = None
        
        for provider in providers_to_try:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Create a copy of the config with the current provider
                    provider_config = dict(config)
                    provider_config["provider"] = provider
                    
                    model = cls.create_model(provider_config)
                    logger.info(f"Successfully created model with provider: {provider}")
                    return model, provider
                except Exception as e:
                    retry_count += 1
                    last_error = e
                    logger.warning(f"Attempt {retry_count} failed for provider {provider}: {e}")
                    # Add a small delay before retrying
                    time.sleep(1)
        
        # If we've exhausted all options, raise the last error
        raise ValueError(f"Failed to create model with any provider. Last error: {last_error}")
    
    @classmethod
    def update_model_stats(cls, cache_key: str, success: bool, tokens: int = 0, latency: float = 0.0) -> None:
        """
        Update statistics for a model instance.
        
        Args:
            cache_key: The cache key for the model
            success: Whether the model call was successful
            tokens: Number of tokens processed
            latency: Time taken for the model call in seconds
        """
        if cache_key not in cls._model_stats:
            return
            
        stats = cls._model_stats[cache_key]
        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
            
        stats["total_tokens"] += tokens
        stats["total_latency"] += latency
    
    @classmethod
    def get_model_stats(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all model instances.
        
        Returns:
            Dictionary of model statistics
        """
        return cls._model_stats
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """
        Get available model providers.
        
        Returns:
            Dictionary mapping provider names to their module paths
        """
        # Combine built-in providers with registered custom providers
        available_models = {provider: path for provider, path in MODEL_PROVIDER_MAP.items()}
        
        # Add registered custom models
        for provider, model_class in cls.registered_models.items():
            available_models[provider] = model_class.__name__
            
        return available_models
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the model cache."""
        cls._model_cache.clear()
        cls._model_stats.clear()
        logger.info("Model cache cleared")
    
    @staticmethod
    def _create_cache_key(provider: str, config: Dict[str, Any]) -> str:
        """
        Create a cache key for a model instance.
        
        Args:
            provider: The provider name
            config: The configuration dictionary
            
        Returns:
            A string key for caching
        """
        # Extract provider-specific config
        provider_config = config.get(provider, {})
        
        # For most providers, the model name is the key differentiator
        model_name = provider_config.get("model_name", "default")
        
        # For local models, also consider the model path
        if provider == "local":
            model_path = provider_config.get("model_path", "")
            return f"{provider}:{model_name}:{model_path}"
        
        # For Ollama, also consider the API base
        if provider == "ollama":
            api_base = provider_config.get("api_base", "http://localhost:11434")
            return f"{provider}:{model_name}:{api_base}"
            
        return f"{provider}:{model_name}"