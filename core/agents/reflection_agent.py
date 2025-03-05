"""
Reflection Agent for AI Co-Scientist

This module implements the Reflection Agent, which critiques and identifies
weaknesses in research hypotheses to guide their refinement.
"""

import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class ReflectionAgent(BaseAgent):
    """
    Reflection Agent critiques research hypotheses.
    
    This agent analyzes hypotheses for logical consistency, scientific plausibility,
    testability, novelty, and alignment with the research goal. It provides
    constructive feedback to guide the refinement process.
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the reflection agent.
        
        Args:
            model: Language model to use
            config: Configuration dictionary
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.critique_categories = config.get("critique_categories", [
            "logical_consistency", 
            "scientific_plausibility", 
            "testability", 
            "novelty", 
            "goal_alignment"
        ])
        self.detail_level = config.get("detail_level", "medium")  # low, medium, high
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Critique a research hypothesis by identifying strengths and weaknesses.
        
        Args:
            context: Dictionary containing:
                - goal: Research goal information
                - hypothesis: The hypothesis to critique
                - iteration: Current iteration number
                - feedback: List of feedback entries (optional)
            params: Dictionary containing optional configuration overrides
                
        Returns:
            Dictionary containing:
                - critiques: List of critique points
                - strengths: List of strength points
                - overall_assessment: Summary assessment
                - improvement_suggestions: List of suggested improvements
        """
        # Extract parameters
        goal = context.get("goal", {})
        hypothesis = context.get("hypothesis", {})
        
        if not goal or not hypothesis:
            raise AgentExecutionError("Research goal and hypothesis are required for reflection")
        
        # Extract text from hypothesis
        hypothesis_text = hypothesis.get("text", "")
        if not hypothesis_text:
            raise AgentExecutionError("Hypothesis text is required for reflection")
            
        # Extract optional parameters
        iteration = context.get("iteration", 0)
        detail_level = params.get("detail_level", self.detail_level)
        
        # Create a JSON schema for the expected output
        output_schema = {
            "type": "object",
            "properties": {
                "critiques": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Category of critique"
                            },
                            "point": {
                                "type": "string",
                                "description": "Critique point"
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["minor", "moderate", "major"],
                                "description": "Severity of the issue"
                            }
                        },
                        "required": ["category", "point", "severity"]
                    }
                },
                "strengths": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Strength of the hypothesis"
                    }
                },
                "overall_assessment": {
                    "type": "string",
                    "description": "Overall assessment of the hypothesis"
                },
                "improvement_suggestions": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Suggestion for improvement"
                    }
                }
            },
            "required": ["critiques", "strengths", "overall_assessment", "improvement_suggestions"]
        }
        
        # Build the prompt
        prompt = self._build_reflection_prompt(
            goal=goal,
            hypothesis=hypothesis,
            detail_level=detail_level,
            iteration=iteration
        )
        
        # Call the model
        system_prompt = self._build_system_prompt()
        
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema
            )
            
            # Process the response
            critiques = response.get("critiques", [])
            strengths = response.get("strengths", [])
            overall = response.get("overall_assessment", "")
            suggestions = response.get("improvement_suggestions", [])
            
            # Build the result
            result = {
                "critiques": critiques,
                "strengths": strengths,
                "overall_assessment": overall,
                "improvement_suggestions": suggestions,
                "metadata": {
                    "hypothesis_id": hypothesis.get("id", ""),
                    "iteration": iteration,
                    "detail_level": detail_level
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Reflection agent execution failed: {str(e)}")
            raise AgentExecutionError(f"Failed to critique hypothesis: {str(e)}")
    
    def _build_reflection_prompt(self,
                               goal: Dict[str, Any],
                               hypothesis: Dict[str, Any],
                               detail_level: str,
                               iteration: int) -> str:
        """
        Build the reflection prompt.
        
        Args:
            goal: Research goal dictionary
            hypothesis: Hypothesis dictionary
            detail_level: Level of detail for critique
            iteration: Current iteration number
            
        Returns:
            Formatted prompt string
        """
        # Extract information
        goal_description = goal.get("description", "")
        domain = goal.get("domain", "")
        constraints = goal.get("constraints", [])
        
        hypothesis_text = hypothesis.get("text", "")
        hypothesis_rationale = hypothesis.get("rationale", "")
        
        # Build the prompt
        prompt = f"# Research Goal\n{goal_description}\n\n"
        
        # Add domain if available
        if domain:
            prompt += f"# Domain\n{domain}\n\n"
            
        # Add constraints if available
        if constraints:
            prompt += "# Constraints\n"
            for constraint in constraints:
                prompt += f"- {constraint}\n"
            prompt += "\n"
            
        # Add the hypothesis
        prompt += f"# Hypothesis\n{hypothesis_text}\n\n"
        
        # Add the rationale if available
        if hypothesis_rationale:
            prompt += f"# Rationale\n{hypothesis_rationale}\n\n"
            
        # Add task description
        prompt += "# Task\n"
        prompt += "Critically evaluate the hypothesis for the following aspects:\n"
        prompt += "1. Logical consistency: Is the hypothesis internally consistent?\n"
        prompt += "2. Scientific plausibility: Is it consistent with established scientific knowledge?\n"
        prompt += "3. Testability: Can the hypothesis be empirically tested?\n"
        prompt += "4. Novelty: Does it offer a new perspective or approach?\n"
        prompt += "5. Goal alignment: How well does it address the research goal?\n\n"
        
        if detail_level == "high":
            prompt += "Provide a comprehensive, detailed critique with specific examples and references where relevant.\n"
        elif detail_level == "low":
            prompt += "Provide a concise critique highlighting only the most important points.\n"
        else:  # medium
            prompt += "Provide a balanced critique with adequate detail on key points.\n"
            
        # Additional guidance for later iterations
        if iteration > 0:
            prompt += f"\nThis is iteration {iteration}, so focus on more nuanced aspects that could be improved.\n"
            
        return prompt
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for the reflection agent.
        
        Returns:
            System prompt string
        """
        return """You are a scientific critic working with a researcher. 
Your task is to provide constructive criticism of research hypotheses.

Guidelines:
- Evaluate the hypothesis objectively, being neither overly harsh nor too lenient
- Identify both strengths and weaknesses
- Provide specific, actionable feedback that can guide improvements
- Consider scientific rigor, logical consistency, and alignment with research goals
- Be thorough in your analysis but focus on substantive issues
- Suggest concrete ways to address each weakness identified
- Use a constructive tone that encourages refinement rather than dismissal

Remember: Your goal is to help strengthen the hypothesis, not just criticize it.
Scientific progress comes through iterative refinement and addressing weaknesses.
""" 