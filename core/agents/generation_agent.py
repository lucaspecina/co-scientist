"""
Generation Agent for AI Co-Scientist

This agent is responsible for generating initial scientific hypotheses
based on research goals and domain knowledge.
"""

import json
import logging
from typing import Dict, List, Optional, Any

from ..models.base_model import BaseModel
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class GenerationAgent(BaseAgent):
    """
    Generation Agent generates novel scientific hypotheses.
    
    It uses an LLM to create initial hypotheses based on research goals,
    domain constraints, and scientific knowledge.
    """
    
    def __init__(self, 
                model: BaseModel,
                prompt_template_path: str,
                num_hypotheses: int = 5,
                creativity: float = 0.7,
                **kwargs):
        """
        Initialize the Generation Agent.
        
        Args:
            model: The LLM model instance to use
            prompt_template_path: Path to the prompt template file
            num_hypotheses: Number of hypotheses to generate
            creativity: Value controlling creativity level (0.0-1.0)
            **kwargs: Additional settings
        """
        super().__init__(
            model=model,
            prompt_template_path=prompt_template_path,
            agent_type="generation",
            **kwargs
        )
        self.num_hypotheses = num_hypotheses
        self.creativity = creativity
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate initial scientific hypotheses based on research goals.
        
        Args:
            input_data: Dictionary containing:
                - research_goal: The scientific question or goal
                - domain: The scientific domain (e.g., "biomedicine")
                - constraints: Optional constraints or requirements
                - prior_knowledge: Optional relevant prior knowledge
                
        Returns:
            Dictionary containing:
                - hypotheses: List of generated hypotheses with details
        """
        # Extract input parameters
        research_goal = input_data.get("research_goal")
        domain = input_data.get("domain", "general science")
        constraints = input_data.get("constraints", "")
        prior_knowledge = input_data.get("prior_knowledge", "")
        
        if not research_goal:
            raise ValueError("Research goal is required for hypothesis generation")
        
        # Format the prompt
        prompt = self._format_prompt(
            research_goal=research_goal,
            domain=domain,
            constraints=constraints,
            prior_knowledge=prior_knowledge,
            num_hypotheses=self.num_hypotheses
        )
        
        # Define the expected JSON schema for structured output
        json_schema = {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "mechanism": {"type": "string"},
                            "testability": {"type": "string"},
                            "novelty_justification": {"type": "string"},
                            "potential_impact": {"type": "string"}
                        },
                        "required": ["id", "title", "description", "mechanism"]
                    }
                }
            },
            "required": ["hypotheses"]
        }
        
        # Set model parameters based on creativity setting
        temperature = 0.4 + (self.creativity * 0.6)  # Maps 0.0-1.0 to 0.4-1.0
        
        try:
            # Use the model to generate hypotheses with JSON output
            response = await self.model.generate_with_json_output(
                prompt=prompt,
                json_schema=json_schema,
                system_prompt="You are a creative scientific hypothesis generator. Your task is to generate novel, plausible, and testable scientific hypotheses based on the research goal. Focus on creating hypotheses that are grounded in scientific principles but explore new possibilities and connections.",
                temperature=temperature
            )
            
            # Validate response structure
            if "hypotheses" not in response:
                logger.warning("Model response missing 'hypotheses' key")
                response = {"hypotheses": []}
                
            # Add generation metadata to each hypothesis
            for hypothesis in response["hypotheses"]:
                hypothesis["source"] = "generation_agent"
                hypothesis["generation_parameters"] = {
                    "creativity": self.creativity,
                    "temperature": temperature
                }
            
            return response
        except Exception as e:
            logger.error(f"Error generating hypotheses: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], model: BaseModel) -> 'GenerationAgent':
        """
        Create a Generation Agent instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with agent settings
            model: Model instance to use
            
        Returns:
            Configured GenerationAgent instance
        """
        return cls(
            model=model,
            prompt_template_path=config.get("prompt_template", "config/templates/generation_agent.txt"),
            num_hypotheses=config.get("num_hypotheses", 5),
            creativity=config.get("creativity", 0.7)
        ) 