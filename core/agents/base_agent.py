"""
Base Agent Interface for AI Co-Scientist

This module defines the abstract base class that all specialized agents must implement.
"""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union

from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all specialized agents."""
    
    def __init__(self, 
                model: BaseModel,
                prompt_template_path: str,
                agent_type: str,
                **kwargs):
        """
        Initialize the base agent.
        
        Args:
            model: The LLM model instance to use
            prompt_template_path: Path to the prompt template file
            agent_type: Type of agent (e.g., 'generation', 'reflection')
            **kwargs: Additional agent-specific settings
        """
        self.model = model
        self.agent_type = agent_type
        self.prompt_template_path = prompt_template_path
        self.prompt_template = self._load_prompt_template()
        
        # Additional metrics and settings
        self.last_execution_time = 0.0
        self.total_calls = 0
        
    def _load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        try:
            if not os.path.exists(self.prompt_template_path):
                logger.warning(f"Prompt template file not found: {self.prompt_template_path}")
                return ""
                
            with open(self.prompt_template_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            return ""
    
    def _format_prompt(self, **kwargs) -> str:
        """Format the prompt template with the provided values."""
        try:
            return self.prompt_template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing key in prompt format: {e}")
            raise ValueError(f"Missing key in prompt format: {e}")
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            raise ValueError(f"Error formatting prompt: {e}")
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task.
        
        Args:
            input_data: Input data for the agent task
            
        Returns:
            Dictionary containing the task output
        """
        pass
    
    async def run_with_timing(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task and record execution time.
        
        Args:
            input_data: Input data for the agent task
            
        Returns:
            Dictionary containing the task output
        """
        start_time = time.time()
        try:
            result = await self.execute(input_data)
            
            # Log execution time
            self.last_execution_time = time.time() - start_time
            self.total_calls += 1
            
            # Add execution metadata to result
            result["metadata"] = {
                "execution_time": self.last_execution_time,
                "agent_type": self.agent_type,
                "timestamp": time.time()
            }
            
            return result
        except Exception as e:
            logger.error(f"Error executing {self.agent_type} agent: {e}")
            return {
                "error": str(e),
                "metadata": {
                    "execution_time": time.time() - start_time,
                    "agent_type": self.agent_type,
                    "timestamp": time.time(),
                    "success": False
                }
            }
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any], model: BaseModel) -> 'BaseAgent':
        """
        Create an agent instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with agent settings
            model: Model instance to use
            
        Returns:
            Configured agent instance
        """
        pass 