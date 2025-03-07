"""
Ranking Agent for AI Co-Scientist

This module implements the Ranking Agent, which evaluates and scores research
hypotheses based on multiple criteria to identify the most promising ones using
both direct scoring and an Elo-based tournament system.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
import uuid

from .base_agent import BaseAgent, AgentExecutionError
from ..tournament.elo_tournament import EloTournament
from ..debate.scientific_debate import ScientificDebate, DebateFormat

logger = logging.getLogger(__name__)


class RankingAgent(BaseAgent):
    """
    Ranking Agent evaluates and scores research hypotheses.
    
    This agent applies consistent evaluation criteria across all hypotheses
    to produce a ranked list based on their potential scientific value using
    both direct scoring and an Elo-based tournament system.
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
        raw_criteria = config.get("criteria", {
            "novelty": 0.25,
            "plausibility": 0.25,
            "testability": 0.25,
            "impact": 0.25
        })
        
        # Convert flat criteria format to nested dictionary format if needed
        self.criteria = {}
        for key, value in raw_criteria.items():
            if isinstance(value, (int, float)):
                # Convert flat format (weight only) to nested dictionary
                self.criteria[key] = {
                    "description": self._get_default_description(key),
                    "weight": float(value)
                }
            else:
                # Already in nested dictionary format
                self.criteria[key] = value
        
        # Calculate total weight to ensure proper normalization
        total_weight = sum(c.get("weight", 0.0) for c in self.criteria.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow for small floating point errors
            logger.warning(f"Criteria weights do not sum to 1.0 (sum: {total_weight}). Normalizing.")
            for criterion in self.criteria:
                self.criteria[criterion]["weight"] /= total_weight
        
        # Tournament configuration
        self.use_tournament = config.get("use_tournament", True)
        self.tournament_matches = config.get("tournament_matches", 10)
        self.tournament_k_factor = config.get("tournament_k_factor", 32.0)
        self.debate_format = config.get("debate_format", "single_turn")
        
        # Initialize tournament state
        self.tournament = None
        self.debate = None
        
    def _get_default_description(self, criterion: str) -> str:
        """
        Get default description for a criterion.
        
        Args:
            criterion: Name of the criterion
            
        Returns:
            Default description
        """
        descriptions = {
            "novelty": "Degree to which the hypothesis offers new ideas or approaches",
            "plausibility": "Scientific plausibility and grounding in established knowledge",
            "testability": "Ease of empirical testing and falsifiability",
            "impact": "Potential scientific or practical impact if validated",
            "alignment": "Alignment with the research goal and context",
            # Add more default descriptions as needed
        }
        return descriptions.get(criterion, f"Evaluation of {criterion}")
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rank and score a set of research hypotheses.
        
        Args:
            context: Dictionary containing:
                - goal: Research goal information
                - hypotheses: List of hypotheses to rank
                - iteration: Current iteration number
                - feedback: List of feedback entries (optional)
                - previous_tournament: Previous tournament state (optional)
                - similarity_graph: Hypothesis similarity graph (optional)
            params: Dictionary containing optional configuration overrides
                
        Returns:
            Dictionary containing:
                - ranked_hypotheses: List of hypotheses with scores
                - ranking_explanation: Explanation of the ranking rationale
                - top_hypotheses: List of top hypothesis IDs
                - tournament_state: Current tournament state (if using tournament)
                - match_results: Results of tournament matches (if using tournament)
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
        previous_tournament = context.get("previous_tournament", None)
        similarity_graph = context.get("similarity_graph", {})
        
        # Check if we should use tournament-based ranking
        use_tournament = params.get("use_tournament", self.use_tournament)
        
        if use_tournament and len(hypotheses) > 1:
            # Use tournament-based ranking with debates
            result = await self._run_tournament_ranking(
                goal=goal,
                hypotheses=hypotheses,
                iteration=iteration,
                feedback=feedback,
                previous_tournament=previous_tournament,
                similarity_graph=similarity_graph,
                params=params
            )
        else:
            # Use direct scoring (standard approach)
            result = await self._run_direct_scoring(
                goal=goal,
                hypotheses=hypotheses,
                iteration=iteration,
                feedback=feedback,
                params=params
            )
            
        # Add metadata to the result
        result["metadata"] = {
            "iteration": iteration,
            "criteria": list(self.criteria.keys()),
            "hypothesis_count": len(hypotheses),
            "ranking_method": "tournament" if use_tournament and len(hypotheses) > 1 else "direct"
        }
        
        return result
    
    async def _run_direct_scoring(
        self,
        goal: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        iteration: int,
        feedback: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run direct scoring of hypotheses (original approach).
        
        Args:
            goal: Research goal information
            hypotheses: List of hypotheses to rank
            iteration: Current iteration number
            feedback: List of feedback entries
            params: Additional parameters
            
        Returns:
            Ranking results
        """
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
                "top_hypotheses": top_hypotheses[:min(3, len(ranked_hypotheses))]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ranking agent direct scoring failed: {str(e)}")
            raise AgentExecutionError(f"Failed to rank hypotheses: {str(e)}")
            
    async def _run_tournament_ranking(
        self,
        goal: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        iteration: int,
        feedback: List[Dict[str, Any]],
        previous_tournament: Optional[Dict[str, Any]] = None,
        similarity_graph: Optional[Dict[str, List[str]]] = None,
        params: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """
        Run tournament-based ranking of hypotheses.
        
        Args:
            goal: Research goal information
            hypotheses: List of hypotheses to rank
            iteration: Current iteration number
            feedback: List of feedback entries
            previous_tournament: Previous tournament state
            similarity_graph: Hypothesis similarity graph
            params: Additional parameters
            
        Returns:
            Tournament ranking results
        """
        # Ensure we have at least 2 hypotheses for tournament
        if len(hypotheses) < 2:
            logger.warning(f"Not enough hypotheses for tournament (found {len(hypotheses)}), using direct scoring instead")
            return await self._run_direct_scoring(goal, hypotheses, iteration, feedback, params)
        # Initialize tournament
        self.tournament = EloTournament(
            k_factor=params.get("tournament_k_factor", self.tournament_k_factor),
            prioritize_similar=True,
            prioritize_new=True,
            prioritize_top_ranked=True
        )
        
        # Initialize debate system
        self.debate = ScientificDebate(
            model_provider=self.model,
            format=DebateFormat(params.get("debate_format", self.debate_format)),
            evaluation_criteria=list(self.criteria.keys()),
            max_turns=params.get("debate_max_turns", 1),
            use_mediator=params.get("debate_use_mediator", False)
        )
        
        # Add all hypotheses to the tournament
        for hypothesis in hypotheses:
            hyp_id = hypothesis.get("id", str(uuid.uuid4()))
            # Ensure ID exists in the hypothesis
            hypothesis["id"] = hyp_id
            self.tournament.add_hypothesis(hyp_id, hypothesis)
        
        # If we have similarity information, update the tournament
        if similarity_graph:
            self.tournament.update_similarity_graph(similarity_graph)
            
        # Generate the goal description for debates
        goal_description = goal.get("description", "")
        domain = goal.get("domain", "")
        if domain:
            goal_description += f" (Domain: {domain})"
            
        # Run tournament matches
        num_matches = min(
            params.get("tournament_matches", self.tournament_matches),
            len(hypotheses) * (len(hypotheses) - 1) // 2  # Maximum possible pairs
        )
        
        match_results = []
        
        # Track pairs we've already matched to avoid duplicates
        matched_pairs: Set[Tuple[str, str]] = set()
        
        for _ in range(num_matches):
            try:
                # Get the next pair to match
                hyp1_id, hyp2_id = self.tournament.get_prioritized_match()
                
                # Skip if we've already matched this pair
                if (hyp1_id, hyp2_id) in matched_pairs or (hyp2_id, hyp1_id) in matched_pairs:
                    continue
                
                # Mark as matched
                matched_pairs.add((hyp1_id, hyp2_id))
                
                # Get the hypothesis data
                hyp1 = self.tournament.get_hypothesis_data(hyp1_id)
                hyp2 = self.tournament.get_hypothesis_data(hyp2_id)
                
                # Conduct the debate
                debate_result = await self.debate.conduct_debate(
                    hypothesis1=hyp1,
                    hypothesis2=hyp2,
                    research_goal=goal_description
                )
                
                # Record the result in the tournament
                winner_id = None
                if debate_result["winner"] == "hypothesis1":
                    winner_id = hyp1_id
                elif debate_result["winner"] == "hypothesis2":
                    winner_id = hyp2_id
                
                # Pass the evaluation for recording in match history
                self.tournament.record_match_result(
                    hypothesis1_id=hyp1_id, 
                    hypothesis2_id=hyp2_id, 
                    winner_id=winner_id,
                    evaluation=debate_result
                )
                
                # Record the debate result
                match_results.append({
                    "hypothesis1_id": hyp1_id,
                    "hypothesis2_id": hyp2_id,
                    "winner_id": winner_id,
                    "debate_id": debate_result["debate_id"],
                    "justification": debate_result["justification"]
                })
                
            except Exception as e:
                logger.error(f"Error in tournament match: {str(e)}")
                continue
        
        # Get the rankings
        rankings = self.tournament.get_rankings()
        top_hypotheses = self.tournament.get_top_hypotheses(3)
        
        # Generate ranked_hypotheses in the expected format
        ranked_hypotheses = []
        for hyp_id, elo_rating, matches_played in rankings:
            hypothesis = self.tournament.get_hypothesis_data(hyp_id)
            
            # Calculate a normalized score between 0-10 based on Elo rating
            # 1200 is baseline (5.0), range typically 1000-1600 (0-10)
            normalized_score = min(10.0, max(0.0, 5.0 + (elo_rating - 1200) / 80))
            
            # Get criteria scores from any debate results if available
            criteria_scores = {}
            match_results = self.tournament.match_history
            
            for criterion in self.criteria:
                # Start with baseline score
                criteria_scores[criterion] = normalized_score
                
            # Try to extract criteria scores from match results if available
            for match in match_results:
                if 'evaluation' in match and match.get('winner_id') == hyp_id:
                    if 'criteria_scoring' in match['evaluation']:
                        for criterion, scoring in match['evaluation']['criteria_scoring'].items():
                            if criterion in self.criteria:
                                # If the new format with hypothesis1_score and hypothesis2_score
                                if isinstance(scoring, dict) and 'hypothesis1_score' in scoring:
                                    # Determine which hypothesis this is in the match
                                    if hyp_id == match.get('hypothesis1_id'):
                                        criteria_scores[criterion] = scoring.get('hypothesis1_score', normalized_score)
                                    else:
                                        criteria_scores[criterion] = scoring.get('hypothesis2_score', normalized_score)
                                else:
                                    # Old format or string value
                                    if isinstance(scoring, (int, float)):
                                        criteria_scores[criterion] = scoring
            
            ranked_hyp = {
                "id": hyp_id,
                "overall_score": normalized_score,
                "criteria_scores": criteria_scores,
                "ranking_rationale": f"Based on {matches_played} tournament matches with Elo rating {elo_rating:.1f}"
            }
            
            ranked_hypotheses.append(ranked_hyp)
            
        # Generate explanation
        tournament_stats = self.tournament.get_tournament_statistics()
        ranking_explanation = (
            f"Ranked using Elo tournament system with {tournament_stats['total_matches']} matches. "
            f"The top hypotheses demonstrated superiority through scientific debate evaluations. "
            f"Average Elo rating: {tournament_stats['avg_rating']:.1f}."
        )
        
        # Build the final result
        result = {
            "ranked_hypotheses": ranked_hypotheses,
            "ranking_explanation": ranking_explanation,
            "top_hypotheses": top_hypotheses,
            "tournament_state": {
                "rankings": rankings,
                "matches_played": tournament_stats["total_matches"],
                "avg_rating": tournament_stats["avg_rating"]
            },
            "match_results": match_results
        }
        
        return result
    
    def _build_ranking_prompt(self,
                            goal: Dict[str, Any],
                            hypotheses: List[Dict[str, Any]],
                            criteria: Dict[str, Dict[str, Any]],
                            iteration: int,
                            feedback: List[Dict[str, Any]]) -> str:
        """
        Build the ranking prompt for direct scoring.
        
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