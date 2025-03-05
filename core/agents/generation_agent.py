"""
Generation Agent for AI Co-Scientist

This module implements the Generation Agent, which is responsible for
creating initial research hypotheses based on the research goal.
"""

import logging
import json
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class GenerationAgent(BaseAgent):
    """
    Generation Agent creates initial research hypotheses.
    
    This agent uses the language model to generate hypotheses based on a research goal,
    taking into account any domain constraints and scientist feedback.
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the generation agent.
        
        Args:
            model: Language model to use
            config: Configuration dictionary
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.generation_count = config.get("generation_count", 5)
        self.creativity = config.get("creativity", 0.7)  # Higher = more creative
        self.diversity_threshold = config.get("diversity_threshold", 0.3)
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate research hypotheses based on the research goal.
        
        Args:
            context: Dictionary containing:
                - goal: Research goal information
                - iteration: Current iteration number
                - feedback: List of feedback entries (optional)
            params: Dictionary containing:
                - count: Number of hypotheses to generate (optional, overrides config)
                - creativity: Creativity level (optional, overrides config)
                
        Returns:
            Dictionary containing:
                - hypotheses: List of generated hypothesis objects
                - metadata: Information about the generation process
        """
        # Extract parameters
        goal = context.get("goal", {})
        if not goal or not goal.get("description"):
            raise AgentExecutionError("Research goal is required for hypothesis generation")
            
        # Extract and override parameters if provided
        count = params.get("count", self.generation_count)
        creativity = params.get("creativity", self.creativity)
        iteration = context.get("iteration", 0)
        feedback = context.get("feedback", [])
        
        # Adjust model parameters based on iteration and creativity
        temperature = min(0.5 + (creativity * 0.5), 0.95)  # Higher for more creative
        
        # Create a JSON schema for the expected output
        output_schema = {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The hypothesis statement"
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Reasoning for why this hypothesis is plausible"
                            }
                        },
                        "required": ["text", "rationale"]
                    }
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of the generation approach"
                }
            },
            "required": ["hypotheses", "reasoning"]
        }
        
        # Build the prompt
        prompt = self._build_generation_prompt(
            goal=goal,
            count=count,
            iteration=iteration,
            feedback=feedback
        )
        
        # Call the model
        system_prompt = self._build_system_prompt(creativity)
        
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema
            )
            
            # Validate the response
            hypotheses = response.get("hypotheses", [])
            if not hypotheses:
                logger.warning("Generation agent returned no hypotheses")
                
            # Add metadata to the response
            result = {
                "hypotheses": hypotheses,
                "metadata": {
                    "count": len(hypotheses),
                    "creativity": creativity,
                    "iteration": iteration,
                    "reasoning": response.get("reasoning", "")
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Generation agent execution failed: {str(e)}")
            raise AgentExecutionError(f"Failed to generate hypotheses: {str(e)}")
    
    def _build_generation_prompt(self, 
                               goal: Dict[str, Any],
                               count: int,
                               iteration: int,
                               feedback: List[Dict[str, Any]]) -> str:
        """
        Build the generation prompt.
        
        Args:
            goal: Research goal dictionary
            count: Number of hypotheses to generate
            iteration: Current iteration number
            feedback: List of feedback entries
            
        Returns:
            Formatted prompt string
        """
        # Extract goal information
        goal_description = goal.get("description", "")
        domain = goal.get("domain", "")
        constraints = goal.get("constraints", [])
        background = goal.get("background", "")
        
        # Start with the goal
        prompt = f"# Research Goal\n{goal_description}\n\n"
        
        # Add domain if available
        if domain:
            prompt += f"# Domain\n{domain}\n\n"
            
        # Add background if available
        if background:
            prompt += f"# Background Information\n{background}\n\n"
            
        # Add constraints if available
        if constraints:
            prompt += "# Constraints\n"
            for constraint in constraints:
                prompt += f"- {constraint}\n"
            prompt += "\n"
            
        # Add feedback from previous iterations
        if feedback and iteration > 0:
            prompt += "# Previous Feedback\n"
            # Take the most recent feedback entries, up to 3
            recent_feedback = sorted(
                feedback, 
                key=lambda x: x.get("iteration", 0), 
                reverse=True
            )[:3]
            
            for entry in recent_feedback:
                feedback_text = entry.get("text", "")
                feedback_iter = entry.get("iteration", 0)
                prompt += f"Iteration {feedback_iter}: {feedback_text}\n\n"
                
        # Add task description
        prompt += f"# Task\n"
        prompt += f"Generate {count} scientifically plausible hypotheses for the research goal above."
        
        if iteration > 0:
            prompt += f" This is iteration {iteration}, so incorporate the feedback provided."
        
        prompt += "\n\nEach hypothesis should be specific, testable, and potentially impactful."
        prompt += "\nFor each hypothesis, provide a clear rationale explaining why it's worth investigating."
        
        return prompt
    
    def _build_system_prompt(self, creativity: float) -> str:
        """
        Build system prompt based on creativity setting.
        
        Args:
            creativity: Creativity level (0-1)
            
        Returns:
            System prompt string
        """
        if creativity > 0.8:
            tone = "highly innovative and out-of-the-box"
        elif creativity > 0.5:
            tone = "creative but grounded in scientific principles"
        else:
            tone = "conservative and strictly evidence-based"
            
        return f"""You are a scientific hypothesis generation system tasked with creating {tone} research hypotheses.
Ensure that each hypothesis:
1. Is specific and testable
2. Has clear scientific significance
3. Addresses the research goal
4. Includes a brief explanation of potential mechanisms
5. Considers constraints and domain knowledge
"""

    @classmethod
    def from_config(cls, config: Dict[str, Any], model: 'BaseModel') -> 'GenerationAgent':
        """
        Create a GenerationAgent instance from configuration.
        
        Args:
            config: Configuration dictionary
            model: Language model to use
            
        Returns:
            Configured GenerationAgent instance
        """
        return cls(model, config) 