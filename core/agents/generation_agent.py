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
    taking into account any domain constraints and scientist feedback. It employs several
    techniques including literature exploration, simulated scientific debates, iterative
    assumptions identification, and research expansion.
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
        self.debate_turns = config.get("debate_turns", 3)  # Number of debate turns
        self.debate_enabled = config.get("debate_enabled", True)  # Enable debate-based generation
        self.literature_search_enabled = config.get("literature_search_enabled", True)
        self.assumptions_identification_enabled = config.get("assumptions_identification_enabled", True)
        
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
                - generation_method: Method to use for generation ("debate", "literature", "assumptions", "expansion")
                
        Returns:
            Dictionary containing:
                - hypotheses: List of generated hypotheses
        """
        goal = context.get("goal", {})
        iteration = context.get("iteration", 0)
        feedback = context.get("feedback", [])
        
        count = params.get("count", self.generation_count)
        generation_method = params.get("generation_method", "auto")
        
        hypotheses = []
        
        try:
            # Select generation method based on context or random selection
            if generation_method == "auto":
                # In earlier iterations, prefer literature search and debates
                # In later iterations, prefer assumptions and research expansion
                if iteration < 2:
                    methods = ["debate", "literature"] 
                else:
                    methods = ["debate", "literature", "assumptions", "expansion"]
                
                import random
                generation_method = random.choice(methods)
            
            logger.info(f"Using generation method: {generation_method}")
            
            if generation_method == "debate":
                hypotheses = await self._generate_via_debate(goal, count, iteration, feedback)
            elif generation_method == "literature":
                hypotheses = await self._generate_via_literature(goal, count, iteration, feedback)
            elif generation_method == "assumptions":
                hypotheses = await self._generate_via_assumptions(goal, count, iteration, feedback)
            elif generation_method == "expansion":
                hypotheses = await self._generate_via_expansion(goal, count, iteration, feedback)
            else:
                # Fallback to standard generation
                prompt = self._build_generation_prompt(goal, count, iteration, feedback)
                system_prompt = self._build_system_prompt(self.creativity)
                
                response = await self.model.generate(prompt, system_prompt=system_prompt)
                hypotheses = self._parse_hypotheses(response)
            
            return {"hypotheses": hypotheses}
        except Exception as e:
            logger.error(f"Error in generation agent: {str(e)}")
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

    async def _generate_via_debate(self, goal: Dict[str, Any], count: int, iteration: int, feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate hypotheses using simulated scientific debate.
        
        This implements the self-play based scientific debate described in the paper,
        where multiple expert perspectives debate and refine ideas.
        
        Args:
            goal: Research goal information
            count: Number of hypotheses to generate
            iteration: Current iteration number
            feedback: List of feedback entries
            
        Returns:
            List of generated hypotheses
        """
        logger.info(f"Generating {count} hypotheses via scientific debate")
        
        # Create the debate prompt
        debate_system_prompt = """
        You are a panel of diverse scientific experts engaging in a structured scientific debate to generate novel research hypotheses.
        Each expert should represent a different perspective or discipline relevant to the research goal.
        The debate should follow these steps:
        1. Each expert introduces a potential hypothesis or angle based on their expertise
        2. Experts critique and improve each other's ideas
        3. The panel synthesizes the discussion into refined hypotheses
        4. The panel selects the most promising hypotheses that are novel, plausible, and testable
        
        Express disagreements and alternate viewpoints to ensure a thorough exploration of the hypothesis space.
        Support your arguments with scientific reasoning and potential evidence.
        """
        
        goal_text = goal.get("description", "")
        domain = goal.get("domain", "")
        constraints = goal.get("constraints", [])
        background = goal.get("background", "")
        
        debate_prompt = f"""
        RESEARCH GOAL: {goal_text}
        
        DOMAIN: {domain}
        
        BACKGROUND INFORMATION:
        {background}
        
        CONSTRAINTS:
        {json.dumps(constraints) if constraints else "No specific constraints provided."}
        
        PREVIOUS FEEDBACK:
        {json.dumps(feedback) if feedback else "No feedback provided yet."}
        
        DEBATE INSTRUCTIONS:
        Please conduct a multi-turn scientific debate among experts from diverse relevant fields to generate {count} novel research hypotheses related to the research goal.
        
        Generate hypotheses that are:
        1. Novel and not obvious from existing literature
        2. Scientifically plausible and grounded in evidence
        3. Testable through experiments
        4. Addressing the specific research goal
        
        For each hypothesis, provide:
        - A clear statement of the hypothesis
        - The scientific rationale behind it
        - Potential experimental approaches to test it
        """
        
        # Run the debate for multiple turns
        debate_transcript = ""
        
        for turn in range(self.debate_turns):
            turn_prompt = f"""
            {debate_prompt}
            
            CURRENT DEBATE TRANSCRIPT:
            {debate_transcript}
            
            DEBATE TURN {turn + 1}/{self.debate_turns}:
            """
            
            response = await self.model.generate(turn_prompt, system_prompt=debate_system_prompt)
            debate_transcript += f"\n\n--- TURN {turn + 1} ---\n{response}"
        
        # Final synthesis prompt to extract hypotheses
        synthesis_prompt = f"""
        Based on the following scientific debate transcript, extract the {count} most promising hypotheses.
        
        DEBATE TRANSCRIPT:
        {debate_transcript}
        
        For each hypothesis, provide:
        1. A title
        2. A clear statement of the hypothesis
        3. Scientific rationale
        4. Potential experimental approach
        5. Novelty assessment
        
        Format each hypothesis as a JSON object with the following structure:
        {{
            "title": "Brief title",
            "hypothesis": "Clear statement of the hypothesis",
            "rationale": "Scientific reasoning and supporting evidence",
            "experimental_approach": "How this could be tested experimentally",
            "novelty": "Assessment of how this advances beyond current knowledge"
        }}
        
        Return an array of these JSON objects.
        """
        
        synthesis_system_prompt = "You are a scientific editor synthesizing the results of a scientific debate into clear, structured research hypotheses."
        
        response = await self.model.generate(synthesis_prompt, system_prompt=synthesis_system_prompt)
        
        try:
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                json_str = json_match.group(0)
                hypotheses_data = json.loads(json_str)
            else:
                # Fallback if no JSON array is found
                logger.warning("No JSON array found in debate synthesis response")
                hypotheses_data = self._parse_hypotheses(response)
            
            # Convert to the expected format
            hypotheses = []
            for h in hypotheses_data:
                hypotheses.append({
                    "text": h.get("hypothesis", ""),
                    "title": h.get("title", ""),
                    "rationale": h.get("rationale", ""),
                    "experimental_approach": h.get("experimental_approach", ""),
                    "metadata": {
                        "generation_method": "scientific_debate",
                        "novelty_assessment": h.get("novelty", ""),
                        "debate_turns": self.debate_turns
                    }
                })
            
            return hypotheses
        except Exception as e:
            logger.error(f"Error parsing debate synthesis: {str(e)}")
            # Fallback to standard parsing if JSON extraction fails
            return self._parse_hypotheses(response)

    async def _generate_via_literature(self, goal: Dict[str, Any], count: int, iteration: int, feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate hypotheses based on literature exploration via web search."""
        # Implementation details will be added here
        logger.info("Literature-based generation not fully implemented yet")
        # Fallback to standard generation for now
        prompt = self._build_generation_prompt(goal, count, iteration, feedback)
        system_prompt = self._build_system_prompt(self.creativity)
        response = await self.model.generate(prompt, system_prompt=system_prompt)
        return self._parse_hypotheses(response)
        
    async def _generate_via_assumptions(self, goal: Dict[str, Any], count: int, iteration: int, feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate hypotheses by identifying and testing assumptions."""
        # Implementation details will be added here
        logger.info("Assumptions-based generation not fully implemented yet")
        # Fallback to standard generation for now
        prompt = self._build_generation_prompt(goal, count, iteration, feedback)
        system_prompt = self._build_system_prompt(self.creativity)
        response = await self.model.generate(prompt, system_prompt=system_prompt)
        return self._parse_hypotheses(response)
        
    async def _generate_via_expansion(self, goal: Dict[str, Any], count: int, iteration: int, feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate hypotheses by expanding on existing research directions."""
        # Implementation details will be added here
        logger.info("Expansion-based generation not fully implemented yet")
        # Fallback to standard generation for now
        prompt = self._build_generation_prompt(goal, count, iteration, feedback)
        system_prompt = self._build_system_prompt(self.creativity)
        response = await self.model.generate(prompt, system_prompt=system_prompt)
        return self._parse_hypotheses(response) 