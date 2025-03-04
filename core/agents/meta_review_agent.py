"""
Meta-Review Agent for AI Co-Scientist

This module implements the Meta-Review Agent, which compiles final outputs from
the hypothesis generation process, providing comprehensive summaries and
experimental suggestions.
"""

import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class MetaReviewAgent(BaseAgent):
    """
    Meta-Review Agent summarizes hypothesis evolution and suggests experiments.
    
    This agent synthesizes information from the entire hypothesis generation
    and refinement process to provide a coherent narrative, highlight key
    findings, and suggest concrete experimental approaches.
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the meta-review agent.
        
        Args:
            model: Language model to use
            config: Configuration dictionary
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.output_format = config.get("output_format", "comprehensive")  # comprehensive, concise, technical
        self.include_experiments = config.get("include_experiments", True)
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a meta-review of top hypotheses and suggest experimental approaches.
        
        Args:
            context: Dictionary containing:
                - goal: Research goal information
                - hypotheses: List of top hypotheses
                - iterations: Number of iterations completed
                - feedback: List of feedback entries (optional)
            params: Dictionary containing optional configuration overrides
                
        Returns:
            Dictionary containing:
                - summary: Overall summary of the research process
                - hypothesis_reviews: Detailed reviews of top hypotheses
                - experimental_approaches: Suggested experimental approaches
                - limitations: Limitations of the hypotheses and suggested work
        """
        # Extract parameters
        goal = context.get("goal", {})
        hypotheses = context.get("hypotheses", [])
        iterations = context.get("iterations", 0)
        
        if not goal or not hypotheses:
            raise AgentExecutionError("Research goal and hypotheses are required for meta-review")
        
        if len(hypotheses) == 0:
            raise AgentExecutionError("At least one hypothesis is required for meta-review")
            
        # Extract optional parameters
        feedback = context.get("feedback", [])
        output_format = params.get("output_format", self.output_format)
        include_experiments = params.get("include_experiments", self.include_experiments)
        
        # Create a JSON schema for the expected output
        output_schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Overall summary of the research process and findings"
                },
                "hypothesis_reviews": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "hypothesis_id": {
                                "type": "string",
                                "description": "ID of the hypothesis"
                            },
                            "title": {
                                "type": "string",
                                "description": "Concise title for the hypothesis"
                            },
                            "summary": {
                                "type": "string",
                                "description": "Summary of the hypothesis"
                            },
                            "strengths": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "Key strengths of the hypothesis"
                            },
                            "limitations": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "Limitations or potential weaknesses"
                            }
                        },
                        "required": ["hypothesis_id", "title", "summary", "strengths", "limitations"]
                    }
                },
                "experimental_approaches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "hypothesis_id": {
                                "type": "string",
                                "description": "ID of the hypothesis"
                            },
                            "approach": {
                                "type": "string",
                                "description": "Detailed experimental approach"
                            },
                            "expected_outcomes": {
                                "type": "string",
                                "description": "Expected outcomes and their interpretation"
                            },
                            "required_resources": {
                                "type": "string",
                                "description": "Resources required for implementation"
                            }
                        },
                        "required": ["hypothesis_id", "approach", "expected_outcomes"]
                    }
                },
                "limitations": {
                    "type": "string",
                    "description": "Overall limitations and directions for future work"
                }
            },
            "required": ["summary", "hypothesis_reviews", "limitations"]
        }
        
        # If experiments are not included, make the field optional
        if not include_experiments:
            output_schema["required"] = ["summary", "hypothesis_reviews", "limitations"]
        
        # Build the prompt
        prompt = self._build_meta_review_prompt(
            goal=goal,
            hypotheses=hypotheses,
            iterations=iterations,
            feedback=feedback,
            output_format=output_format,
            include_experiments=include_experiments
        )
        
        # Call the model
        system_prompt = self._build_system_prompt(output_format)
        
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema
            )
            
            # Process the response
            summary = response.get("summary", "")
            hypothesis_reviews = response.get("hypothesis_reviews", [])
            experimental_approaches = response.get("experimental_approaches", [])
            limitations = response.get("limitations", "")
            
            # Build the result
            result = {
                "summary": summary,
                "hypothesis_reviews": hypothesis_reviews,
                "limitations": limitations,
                "metadata": {
                    "iterations": iterations,
                    "hypothesis_count": len(hypotheses),
                    "output_format": output_format
                }
            }
            
            # Add experimental approaches if included
            if include_experiments:
                result["experimental_approaches"] = experimental_approaches
            
            return result
            
        except Exception as e:
            logger.error(f"Meta-review agent execution failed: {str(e)}")
            raise AgentExecutionError(f"Failed to generate meta-review: {str(e)}")
    
    def _build_meta_review_prompt(self,
                                goal: Dict[str, Any],
                                hypotheses: List[Dict[str, Any]],
                                iterations: int,
                                feedback: List[Dict[str, Any]],
                                output_format: str,
                                include_experiments: bool) -> str:
        """
        Build the meta-review prompt.
        
        Args:
            goal: Research goal dictionary
            hypotheses: List of top hypotheses
            iterations: Number of iterations completed
            feedback: List of feedback entries
            output_format: Format of the output
            include_experiments: Whether to include experimental approaches
            
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
            
        # Add process summary
        prompt += f"# Process Summary\n"
        prompt += f"The hypothesis generation process went through {iterations} iterations of generation, reflection, and refinement. "
        prompt += f"The top hypotheses below represent the most promising outcomes from this process.\n\n"
        
        # Add the top hypotheses
        prompt += "# Top Hypotheses\n"
        for i, hypothesis in enumerate(hypotheses, 1):
            hyp_id = hypothesis.get("id", f"hyp{i}")
            text = hypothesis.get("text", "")
            rationale = hypothesis.get("rationale", "")
            score = hypothesis.get("score", "N/A")
            
            prompt += f"\n## Hypothesis {i} [ID: {hyp_id}] (Score: {score})\n"
            prompt += f"Text: {text}\n"
            if rationale:
                prompt += f"Rationale: {rationale}\n"
                
            # Add scores if available
            scores = hypothesis.get("scores", {})
            if scores:
                prompt += "Scores:\n"
                for criterion, score in scores.items():
                    prompt += f"- {criterion}: {score}\n"
        
        # Add task description
        prompt += "\n# Task\n"
        prompt += "Generate a meta-review of the research process and top hypotheses. Include:\n"
        prompt += "1. An overall summary of the research process and findings\n"
        prompt += "2. Detailed reviews of each top hypothesis, including strengths and limitations\n"
        
        if include_experiments:
            prompt += "3. Specific experimental approaches to test each hypothesis\n"
            prompt += "4. Overall limitations and directions for future work\n"
        else:
            prompt += "3. Overall limitations and directions for future work\n"
            
        # Add format guidance
        if output_format == "concise":
            prompt += "\nProvide a concise meta-review, focusing on key points with minimal detail.\n"
        elif output_format == "technical":
            prompt += "\nProvide a technical meta-review, with precise scientific terminology and detailed methodological considerations.\n"
        else:  # comprehensive
            prompt += "\nProvide a comprehensive meta-review, with balanced detail suitable for general scientific audience.\n"
            
        return prompt
    
    def _build_system_prompt(self, output_format: str) -> str:
        """
        Build the system prompt for the meta-review agent.
        
        Args:
            output_format: Format of the output
            
        Returns:
            System prompt string
        """
        base_prompt = """You are a scientific meta-reviewer working with a researcher. 
Your task is to synthesize information about a research process and provide a coherent
summary of the findings, along with critical analysis and suggested next steps.

Guidelines:
- Provide a comprehensive overview of the research process and its outcomes
- Highlight the strengths and limitations of each top hypothesis
- Maintain scientific accuracy and precision in your descriptions
- Be balanced in your assessment, acknowledging both potential and limitations
- Suggest concrete, feasible experimental approaches to test the hypotheses
- Consider practical aspects of implementation when suggesting experiments
"""

        if output_format == "concise":
            base_prompt += """
Format your response concisely, focusing on essential information. Use clear, 
direct language and minimize detailed explanations. Prioritize actionable insights
and key findings. Aim for brevity while maintaining clarity and scientific accuracy."""
        elif output_format == "technical":
            base_prompt += """
Format your response with technical precision, using appropriate scientific terminology
and detailed methodological considerations. Include specific technical details about
experimental design, controls, variables, and analysis methods. Your audience consists
of domain experts who require rigorous scientific depth."""
        else:  # comprehensive
            base_prompt += """
Format your response comprehensively with balanced detail. Explain concepts clearly 
for a general scientific audience. Include sufficient detail to understand the science
while maintaining readability. Strike a balance between accessibility and scientific depth."""
            
        return base_prompt 