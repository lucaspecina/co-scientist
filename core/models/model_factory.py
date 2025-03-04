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
}


class ModelFactory:
    """Factory for creating LLM instances based on configuration."""
    
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
            
        if provider not in MODEL_PROVIDER_MAP:
            supported = ", ".join(MODEL_PROVIDER_MAP.keys())
            raise ValueError(f"Unsupported model provider: {provider}. Supported providers: {supported}")
        
        # Get the model class path and load it dynamically
        model_class_path = MODEL_PROVIDER_MAP[provider]
        module_path, class_name = model_class_path.rsplit(".", 1)
        
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load model adapter for provider {provider}: {e}")
            raise ValueError(f"Failed to load model adapter: {e}")
        
        # Get provider-specific configuration
        provider_config = config.get(provider, {})
        
        # Create and return model instance
        try:
            return model_class.from_config(provider_config)
        except Exception as e:
            logger.error(f"Failed to instantiate {provider} model: {e}")
            raise ValueError(f"Failed to instantiate model: {e}")
    
    @staticmethod
    def create_model_for_agent(agent_type: str, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance for a specific agent type.
        
        Args:
            agent_type: The type of agent (e.g., 'generation', 'reflection')
            config: Complete configuration dictionary
                
        Returns:
            Configured BaseModel instance for the agent
        """
        # Get global model configuration
        models_config = config.get("models", {})
        default_provider = models_config.get("default_provider", "openai")
        
        # Get agent-specific configuration
        agent_config = config.get("agents", {}).get(agent_type, {})
        provider = agent_config.get("model_provider", default_provider)
        
        # If provider is specified as 'default', use the default provider
        if provider == "default":
            provider = default_provider
            
        # Create model config by combining default provider config with agent-specific overrides
        model_config = {
            "provider": provider,
            "default_provider": default_provider
        }
        
        # Add provider-specific configuration
        if provider in models_config:
            model_config[provider] = models_config[provider].copy()
            
            # Override with agent-specific settings if provided
            for key, value in agent_config.items():
                if key in model_config[provider]:
                    model_config[provider][key] = value
        
        return ModelFactory.create_model(model_config) 