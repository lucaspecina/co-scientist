"""
Model Factory for AI Co-Scientist

Provides a factory pattern implementation for creating model instances
based on configuration settings. This enables easy switching between
different LLM providers.
"""

import importlib
import logging
from typing import Dict, Any, Optional

from .base_model import BaseModel

logger = logging.getLogger(__name__)

# Map provider names to their module paths
MODEL_PROVIDER_MAP = {
    "openai": "core.models.openai_model.OpenAIModel",
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
            provider = config.get("default_provider", "openai")
            
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
            return model_class.from_config(config)
        except Exception as e:
            logger.error(f"Error creating model instance for provider {provider}: {e}")
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
        # Get the provider specified for this agent type, falling back to defaults
        agent_config = config.get("agents", {}).get(agent_type, {})
        provider = agent_config.get("model_provider", "default")
        
        # Copy the config to avoid modifying the original
        model_config = dict(config)
        
        # Set the provider in the config
        model_config["provider"] = provider
        
        # Override model settings with agent-specific settings if provided
        for key in ["temperature", "max_tokens"]:
            if key in agent_config:
                model_config[key] = agent_config[key]
        
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