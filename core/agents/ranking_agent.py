"""
Ranking Agent for AI Co-Scientist

This module implements the Ranking Agent, which evaluates and scores research
hypotheses based on multiple criteria to identify the most promising ones.
"""

import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class RankingAgent(BaseAgent):
    """
    Ranking Agent evaluates and scores research hypotheses.
    
    This agent applies consistent evaluation criteria across all hypotheses
    to produce a ranked list based on their potential scientific value.
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the ranking agent.
        
        Args:
            model: Language model to use
            config: Configuration dictionary
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.criteria = config.get("criteria", {
            "novelty": {
                "description": "Degree to which the hypothesis offers new ideas or approaches",
                "weight": 0.2
            },
            "plausibility": {
                "description": "Scientific plausibility and grounding in established knowledge",
                "weight": 0.3
            },
            "testability": {
                "description": "Ease of empirical testing and falsifiability",
                "weight": 0.25
            },
            "impact": {
                "description": "Potential scientific or practical impact if validated",
                "weight": 0.15
            },
            "goal_alignment": {
                "description": "Alignment with the original research goal",
                "weight": 0.1
            }
        })
        
        # Calculate total weight to ensure proper normalization
        total_weight = sum(c.get("weight", 0.0) for c in self.criteria.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow for small floating point errors
            logger.warning(f"Criteria weights do not sum to 1.0 (sum: {total_weight}). Normalizing.")
            for criterion in self.criteria:
                self.criteria[criterion]["weight"] /= total_weight
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rank and score a set of research hypotheses.
        
        Args:
            context: Dictionary containing:
                - goal: Research goal information
                - hypotheses: List of hypotheses to rank
                - iteration: Current iteration number
                - feedback: List of feedback entries (optional)
            params: Dictionary containing optional configuration overrides
                
        Returns:
            Dictionary containing:
                - ranked_hypotheses: List of hypotheses with scores
                - ranking_explanation: Explanation of the ranking rationale
                - top_hypotheses: List of top hypothesis IDs
        """
        # Extract parameters
        goal = context.get("goal", {})
        hypotheses = context.get("hypotheses", [])
        
        if not goal or not hypotheses:
            raise AgentExecutionError("Research goal and hypotheses are required for ranking")
        
        if len(hypotheses) == 0:
            raise AgentExecutionError("At least one hypothesis is required for ranking")
            
        # Extract optional parameters
        iteration = context.get("iteration", 0)
        feedback = context.get("feedback", [])
        
        # Create a JSON schema for the expected output
        output_schema = {
            "type": "object",
            "properties": {
                "ranked_hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "ID of the hypothesis"
                            },
                            "overall_score": {
                                "type": "number",
                                "description": "Overall score (0.0-10.0)"
                            },
                            "criteria_scores": {
                                "type": "object",
                                "description": "Scores for individual criteria (0.0-10.0)",
                                "additionalProperties": {
                                    "type": "number"
                                }
                            },
                            "ranking_rationale": {
                                "type": "string",
                                "description": "Rationale for the scores"
                            }
                        },
                        "required": ["id", "overall_score", "criteria_scores", "ranking_rationale"]
                    }
                },
                "ranking_explanation": {
                    "type": "string",
                    "description": "Overall explanation of the ranking process"
                },
                "top_hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "ID of a top hypothesis"
                    },
                    "description": "IDs of the top 3 hypotheses"
                }
            },
            "required": ["ranked_hypotheses", "ranking_explanation", "top_hypotheses"]
        }
        
        # Build the prompt
        prompt = self._build_ranking_prompt(
            goal=goal,
            hypotheses=hypotheses,
            criteria=self.criteria,
            iteration=iteration,
            feedback=feedback
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
            ranked_hypotheses = response.get("ranked_hypotheses", [])
            ranking_explanation = response.get("ranking_explanation", "")
            top_hypotheses = response.get("top_hypotheses", [])
            
            if not ranked_hypotheses:
                logger.warning("Ranking agent returned no ranked hypotheses")
                
            # Build the result
            result = {
                "ranked_hypotheses": ranked_hypotheses,
                "ranking_explanation": ranking_explanation,
                "top_hypotheses": top_hypotheses,
                "metadata": {
                    "iteration": iteration,
                    "criteria": list(self.criteria.keys()),
                    "hypothesis_count": len(hypotheses)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ranking agent execution failed: {str(e)}")
            raise AgentExecutionError(f"Failed to rank hypotheses: {str(e)}")
    
    def _build_ranking_prompt(self,
                            goal: Dict[str, Any],
                            hypotheses: List[Dict[str, Any]],
                            criteria: Dict[str, Dict[str, Any]],
                            iteration: int,
                            feedback: List[Dict[str, Any]]) -> str:
        """
        Build the ranking prompt.
        
        Args:
            goal: Research goal dictionary
            hypotheses: List of hypotheses to rank
            criteria: Dictionary of ranking criteria
            iteration: Current iteration number
            feedback: List of feedback entries
            
        Returns:
            Formatted prompt string
        """
        # Extract information
        goal_description = goal.get("description", "")
        domain = goal.get("domain", "")
        constraints = goal.get("constraints", [])
        
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
            
        # Add evaluation criteria
        prompt += "# Evaluation Criteria\n"
        for criterion, details in criteria.items():
            description = details.get("description", "")
            weight = details.get("weight", 0.0)
            prompt += f"- {criterion.upper()} (weight: {weight:.2f}): {description}\n"
        prompt += "\n"
        
        # Add recent feedback if available
        if feedback:
            prompt += "# Scientist Feedback\n"
            recent_feedback = sorted(
                feedback, 
                key=lambda x: x.get("iteration", 0),
                reverse=True
            )[:2]  # Only the 2 most recent feedback entries
            
            for entry in recent_feedback:
                feedback_text = entry.get("text", "")
                feedback_iter = entry.get("iteration", 0)
                
                if feedback_text:
                    prompt += f"From iteration {feedback_iter}: {feedback_text}\n\n"
        
        # Add the hypotheses to evaluate
        prompt += "# Hypotheses to Evaluate\n"
        for i, hypothesis in enumerate(hypotheses, 1):
            hyp_id = hypothesis.get("id", f"hyp{i}")
            text = hypothesis.get("text", "")
            rationale = hypothesis.get("rationale", "")
            
            prompt += f"\n## Hypothesis {i} [ID: {hyp_id}]\n"
            prompt += f"Text: {text}\n"
            if rationale:
                prompt += f"Rationale: {rationale}\n"
                
        # Add task description
        prompt += "\n# Task\n"
        prompt += "Evaluate each hypothesis according to the criteria provided. For each hypothesis:\n"
        prompt += "1. Assign a score from 0.0 to 10.0 for each criterion\n"
        prompt += "2. Calculate a weighted overall score based on the criterion weights\n"
        prompt += "3. Provide a brief rationale for the scores\n"
        prompt += "4. Rank the hypotheses from highest to lowest overall score\n"
        prompt += "5. Identify the top 3 hypotheses\n\n"
        
        prompt += "Ensure consistent and fair evaluation across all hypotheses."
        
        if iteration > 0:
            prompt += f"\nThis is iteration {iteration}, so consider how the hypotheses have evolved and improved."
        
        return prompt
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for the ranking agent.
        
        Returns:
            System prompt string
        """
        return """You are a scientific hypothesis evaluator working with a researcher. 
Your task is to objectively evaluate and rank research hypotheses based on specific criteria.

Guidelines:
- Apply the same evaluation standards consistently across all hypotheses
- Score each criterion on a scale from 0.0 (lowest) to 10.0 (highest)
- Calculate overall scores by applying the weights provided for each criterion
- Provide clear, specific rationales for your scoring decisions
- Consider both the strengths and weaknesses of each hypothesis
- Avoid being unduly influenced by writing style over substance
- Be particularly attentive to scientific merit, testability, and alignment with research goals
- Rank hypotheses based on their overall scores

Focus on helping the researcher identify the most promising hypotheses to pursue.
Your evaluation should help guide the research process toward scientifically valuable outcomes.
""" 