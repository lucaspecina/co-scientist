#!/usr/bin/env python
"""
Fixed model factory with proper handling of Azure OpenAI configurations.
This is a temporary patch - ideally, this would be integrated into the main codebase.
"""

import importlib
import json
import logging
import os
import sys
import time
from typing import Dict, Any, Tuple, List, Optional, Type

# This is needed to import from the core package
core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, core_path)

from core.models.model_factory import ModelFactory, MODEL_PROVIDER_MAP
from core.models.base_model import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

class FixedModelFactory(ModelFactory):
    """
    Fixed version of the ModelFactory that properly handles Azure OpenAI configuration.
    """
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured BaseModel instance
            
        Raises:
            ValueError: If provider is not supported or module cannot be loaded
        """
        # Get the provider name, defaulting to the default_provider if specific provider not given
        provider = config.get("provider")
        if provider == "default" or provider is None:
            provider = config.get("default_provider", "azure_openai")
        
        # PATCH: Special handling for Azure OpenAI
        if provider == "azure_openai":
            # Get environment variables
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
            
            # Create a properly structured config for Azure OpenAI
            azure_config = {
                "api_key": api_key,
                "api_version": api_version,
                "endpoint": endpoint,
                "deployment_id": deployment_name
            }
            
            # Get additional configuration from models.azure_openai if available
            if "models" in config and "azure_openai" in config["models"]:
                azure_model_config = config["models"]["azure_openai"]
                # Use values from config if not set by environment variables
                if not api_key and "api_key" in azure_model_config:
                    azure_config["api_key"] = azure_model_config["api_key"]
                if not api_version and "api_version" in azure_model_config:
                    azure_config["api_version"] = azure_model_config["api_version"]
                if not endpoint and "endpoint" in azure_model_config:
                    azure_config["endpoint"] = azure_model_config["endpoint"]
                if not deployment_name and "deployment_id" in azure_model_config:
                    azure_config["deployment_id"] = azure_model_config["deployment_id"]
                
                # Copy over other configuration parameters
                for key, value in azure_model_config.items():
                    if key not in azure_config:
                        azure_config[key] = value
            
            # Use the Azure config
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

# Apply the monkey patch
def apply_patch():
    """Apply the monkey patch to the ModelFactory class."""
    from core.models.model_factory import ModelFactory
    ModelFactory.create_model = FixedModelFactory.create_model
    print("✅ Applied the FixedModelFactory patch for proper Azure OpenAI support")

if __name__ == "__main__":
    apply_patch()
    # Here we could add code to run the application 