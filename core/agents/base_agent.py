"""
Base Agent Interface for AI Co-Scientist

This module defines the abstract base class for all agents in the system.
"""

import abc
import logging
import os
import time
from typing import Dict, Any, Optional, List, Union, Callable

from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)


class AgentExecutionError(Exception):
    """Exception raised when an agent execution fails."""
    pass


class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents in the AI Co-Scientist system.
    
    An agent is responsible for a specific task in the workflow, such as
    hypothesis generation, criticism, evolution, etc. Each agent type
    implements custom logic for its specific role.
    """
    
    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        """
        Initialize an agent.
        
        Args:
            model: The language model used by this agent
            config: Configuration dictionary with agent settings
        """
        self.model = model
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.description = config.get("description", "")
        self.system_prompt = config.get("system_prompt", "")
        self.max_retries = config.get("max_retries", 2)
        
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
    
    @abc.abstractmethod
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task.
        
        Args:
            context: Context information for the task
            params: Parameters for the task
            
        Returns:
            Result of the task execution
            
        Raises:
            AgentExecutionError: If execution fails
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
    def from_config(cls, config: Dict[str, Any], model: 'BaseModel') -> 'BaseAgent':
        """
        Create an agent instance from configuration.
        
        Args:
            config: Configuration dictionary
            model: Language model to use
            
        Returns:
            Configured agent instance
        """
        return cls(model, config)
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "description": self.description
        }
    
    async def _call_model(self, 
                        prompt: str, 
                        system_prompt: Optional[str] = None,
                        schema: Optional[Dict[str, Any]] = None) -> Union[str, Dict[str, Any]]:
        """
        Call the language model with error handling and retries.
        
        Args:
            prompt: Main prompt for the model
            system_prompt: System prompt to prepend (optional)
            schema: JSON schema for structured output (optional)
            
        Returns:
            Model response as string or dictionary
            
        Raises:
            AgentExecutionError: On persistent model failures
        """
        system_prompt = system_prompt or self.system_prompt
        remaining_retries = self.max_retries
        
        while True:
            try:
                if schema:
                    return await self.model.generate_json(
                        prompt=prompt,
                        schema=schema,
                        system_prompt=system_prompt
                    )
                else:
                    return await self.model.generate(
                        prompt=prompt,
                        system_prompt=system_prompt
                    )
                    
            except Exception as e:
                remaining_retries -= 1
                logger.warning(f"Agent {self.name} model call failed: {str(e)}. Retries left: {remaining_retries}")
                
                if remaining_retries <= 0:
                    raise AgentExecutionError(f"Agent {self.name} failed after max retries: {str(e)}")
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format the context dictionary as a human-readable string.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted context string
        """
        # Default implementation - can be overridden by subclasses for custom formatting
        context_str = "Context:\n"
        
        for key, value in context.items():
            if isinstance(value, dict):
                context_str += f"\n{key.upper()}:\n"
                for k, v in value.items():
                    context_str += f"  {k}: {v}\n"
            elif isinstance(value, list):
                context_str += f"\n{key.upper()}:\n"
                for item in value:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            context_str += f"  {k}: {v}\n"
                        context_str += "\n"
                    else:
                        context_str += f"  - {item}\n"
            else:
                context_str += f"\n{key.upper()}: {value}\n"
        
        return context_str 