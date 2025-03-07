"""
Ranking Agent for AI Co-Scientist

This module implements the Ranking Agent, which evaluates and scores research
hypotheses based on multiple criteria to identify the most promising ones.
It implements an Elo-based tournament system for pairwise comparison of hypotheses.
"""

import logging
import random
import math
from typing import Dict, Any, List, Optional, Tuple, Set

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class RankingAgent(BaseAgent):
    """
    Ranking Agent evaluates and scores research hypotheses using an Elo-based tournament.
    
    This agent organizes pairwise comparisons between hypotheses in tournament matches,
    employing multi-turn scientific debates for top-ranked hypotheses and simpler 
    comparisons for lower-ranked ones. The resulting Elo ratings provide a ranking
    of hypotheses based on their scientific merit.
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
            elif isinstance(value, dict):
                # Already in nested format, just ensure it has the right structure
                self.criteria[key] = {
                    "description": value.get("description", self._get_default_description(key)),
                    "weight": float(value.get("weight", 1.0))
                }
        
        # Elo rating system parameters
        self.initial_elo = config.get("initial_elo", 1200)
        self.elo_k_factor = config.get("elo_k_factor", 32)  # Determines rating change magnitude
        self.tournament_matches = config.get("tournament_matches", 5)  # Matches per hypothesis
        self.debate_turns = config.get("debate_turns", 3)  # Number of debate turns for top hypotheses
        self.full_debate_threshold = config.get("full_debate_threshold", 0.25)  # Top % for full debates
        
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
        Rank research hypotheses through an Elo-based tournament.
        
        Args:
            context: Dictionary containing:
                - goal: Research goal information
                - hypotheses: List of hypotheses to rank
                - iteration: Current iteration number
                - feedback: List of feedback entries (optional)
                - proximity_graph: Hypothesis similarity graph (optional)
                - previous_elo_ratings: Previous Elo ratings (optional)
            params: Dictionary containing:
                - match_count: Number of tournament matches to run (optional)
                - tournament_type: Type of tournament ("full", "incremental")
                
        Returns:
            Dictionary containing:
                - ranked_hypotheses: List of hypotheses with updated Elo ratings
                - tournament_results: Details of tournament matches conducted
        """
        goal = context.get("goal", {})
        hypotheses = context.get("hypotheses", [])
        iteration = context.get("iteration", 0)
        feedback = context.get("feedback", [])
        proximity_graph = context.get("proximity_graph", {})
        previous_elo_ratings = context.get("previous_elo_ratings", {})
        
        # Extract parameters
        match_count = params.get("match_count", self.tournament_matches)
        tournament_type = params.get("tournament_type", "incremental")
        
        if not hypotheses:
            logger.warning("No hypotheses provided for ranking")
            return {"ranked_hypotheses": [], "tournament_results": []}
        
        try:
            # Initialize Elo ratings
            elo_ratings = self._initialize_elo_ratings(hypotheses, previous_elo_ratings)
            
            # Generate tournament matches
            matches = self._generate_tournament_matches(
                hypotheses, 
                match_count, 
                elo_ratings,
                proximity_graph,
                tournament_type
            )
            
            # Run the tournament
            tournament_results = await self._run_tournament(
                matches, 
                goal, 
                elo_ratings,
                iteration,
                feedback,
                context
            )
            
            # Update Elo ratings based on tournament results
            updated_ratings = self._update_elo_ratings(elo_ratings, tournament_results)
            
            # Log the updated ratings for debugging
            logger.info(f"Updated Elo ratings: {updated_ratings}")
            
            # Apply ratings to hypotheses
            ranked_hypotheses = self._apply_ratings_to_hypotheses(hypotheses, updated_ratings)
            
            # Ensure all hypotheses have a score
            for h in ranked_hypotheses:
                if "score" not in h or h["score"] == 0:
                    h_id = h.get("id", "unknown")
                    if h_id in updated_ratings:
                        h["score"] = float(updated_ratings[h_id])
                        logger.info(f"Force-set score for hypothesis {h_id} to {h['score']}")
            
            # Log how many hypotheses have non-zero scores
            non_zero_scores = sum(1 for h in ranked_hypotheses if h.get("score", 0) > 0)
            logger.info(f"{non_zero_scores} out of {len(ranked_hypotheses)} hypotheses have non-zero scores")
            
            return {
                "ranked_hypotheses": ranked_hypotheses,
                "tournament_results": tournament_results
            }
        except Exception as e:
            logger.error(f"Error in ranking agent: {str(e)}")
            raise AgentExecutionError(f"Failed to rank hypotheses: {str(e)}")
    
    def _initialize_elo_ratings(self, 
                              hypotheses: List[Dict[str, Any]], 
                              previous_ratings: Dict[str, float]) -> Dict[str, float]:
        """
        Initialize Elo ratings for all hypotheses.
        
        Args:
            hypotheses: List of hypotheses to rank
            previous_ratings: Previous Elo ratings by hypothesis ID
            
        Returns:
            Dictionary mapping hypothesis IDs to Elo ratings
        """
        ratings = {}
        
        for hypothesis in hypotheses:
            hypothesis_id = hypothesis.get("id")
            
            # If the hypothesis already has a rating, use it
            if hypothesis_id in previous_ratings:
                ratings[hypothesis_id] = previous_ratings[hypothesis_id]
            else:
                # Otherwise, assign the initial Elo rating
                ratings[hypothesis_id] = self.initial_elo
        
        return ratings
    
    def _generate_tournament_matches(self,
                                  hypotheses: List[Dict[str, Any]],
                                  match_count: int,
                                  elo_ratings: Dict[str, float],
                                  proximity_graph: Dict[str, List[str]],
                                  tournament_type: str) -> List[Tuple[str, str]]:
        """
        Generate tournament matches for hypothesis comparison.
        
        Args:
            hypotheses: List of hypotheses
            match_count: Maximum number of matches per hypothesis
            elo_ratings: Current Elo ratings
            proximity_graph: Hypothesis similarity graph
            tournament_type: Type of tournament to run
            
        Returns:
            List of match pairs (hypothesis1_id, hypothesis2_id)
        """
        hypothesis_ids = [h.get("id") for h in hypotheses]
        
        # Different match generation strategies
        if tournament_type == "full":
            # Full tournament: compare each hypothesis with every other
            matches = []
            for i, id1 in enumerate(hypothesis_ids):
                for id2 in hypothesis_ids[i+1:]:
                    matches.append((id1, id2))
            
            # If too many matches, randomly sample
            if len(matches) > match_count * len(hypothesis_ids) // 2:
                random.shuffle(matches)
                matches = matches[:match_count * len(hypothesis_ids) // 2]
                
        elif tournament_type == "incremental":
            # Incremental tournament: prioritize matches based on ratings and similarity
            matches = []
            matches_per_hypothesis = {}
            
            # Initialize match count per hypothesis
            for h_id in hypothesis_ids:
                matches_per_hypothesis[h_id] = 0
            
            # Strategy:
            # 1. Top hypotheses should participate in more matches
            # 2. Similar hypotheses should be more likely to be compared
            # 3. Each hypothesis should have a minimum number of matches
            
            # Sort by Elo rating (descending)
            sorted_ids = sorted(hypothesis_ids, key=lambda h_id: elo_ratings.get(h_id, 0), reverse=True)
            
            # First, ensure minimum matches for new hypotheses
            new_hypotheses = [h_id for h_id in hypothesis_ids if h_id not in elo_ratings]
            for new_id in new_hypotheses:
                # Match with a mix of top and random hypotheses
                candidate_opponents = sorted_ids[:5] + random.sample(hypothesis_ids, min(5, len(hypothesis_ids)))
                # Remove self
                candidate_opponents = [h_id for h_id in candidate_opponents if h_id != new_id]
                
                # Select up to 3 opponents
                selected_opponents = candidate_opponents[:min(3, len(candidate_opponents))]
                for opponent_id in selected_opponents:
                    matches.append((new_id, opponent_id))
                    matches_per_hypothesis[new_id] = matches_per_hypothesis.get(new_id, 0) + 1
                    matches_per_hypothesis[opponent_id] = matches_per_hypothesis.get(opponent_id, 0) + 1
            
            # Then, prioritize matches between similar hypotheses using proximity graph
            if proximity_graph:
                for h_id in hypothesis_ids:
                    if h_id in proximity_graph:
                        similar_ids = proximity_graph[h_id]
                        # Compare with similar hypotheses
                        for similar_id in similar_ids:
                            if similar_id in hypothesis_ids and similar_id != h_id:
                                matches.append((h_id, similar_id))
                                matches_per_hypothesis[h_id] = matches_per_hypothesis.get(h_id, 0) + 1
                                matches_per_hypothesis[similar_id] = matches_per_hypothesis.get(similar_id, 0) + 1
            
            # Finally, fill in remaining matches needed
            while sum(matches_per_hypothesis.values()) // 2 < match_count * len(hypothesis_ids) // 2:
                # Prioritize hypotheses with fewer matches
                candidates = sorted(hypothesis_ids, key=lambda h_id: matches_per_hypothesis.get(h_id, 0))
                
                if not candidates:
                    break
                
                h_id = candidates[0]
                
                # Find suitable opponent
                # Bias toward hypotheses with similar ratings for competitive matches
                sorted_by_rating_diff = sorted(
                    [opp_id for opp_id in hypothesis_ids if opp_id != h_id],
                    key=lambda opp_id: abs(elo_ratings.get(h_id, 0) - elo_ratings.get(opp_id, 0))
                )
                
                if sorted_by_rating_diff:
                    # Select from the 5 closest ratings with some randomness
                    pool = sorted_by_rating_diff[:min(5, len(sorted_by_rating_diff))]
                    opponent_id = random.choice(pool)
                    
                    matches.append((h_id, opponent_id))
                    matches_per_hypothesis[h_id] = matches_per_hypothesis.get(h_id, 0) + 1
                    matches_per_hypothesis[opponent_id] = matches_per_hypothesis.get(opponent_id, 0) + 1
                else:
                    # No suitable opponent found
                    break
        
        else:
            # Default: Random pairing
            matches = []
            for _ in range(match_count):
                if len(hypothesis_ids) < 2:
                    break
                
                id1, id2 = random.sample(hypothesis_ids, 2)
                matches.append((id1, id2))
        
        return matches
    
    async def _run_tournament(self,
                        matches: List[Tuple[str, str]],
                        goal: Dict[str, Any],
                        elo_ratings: Dict[str, float],
                        iteration: int,
                        feedback: List[Dict[str, Any]],
                        context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Run tournament matches and determine winners.
        
        Args:
            matches: List of match pairs (hypothesis1_id, hypothesis2_id)
            goal: Research goal information
            elo_ratings: Current Elo ratings
            iteration: Current iteration number
            feedback: List of feedback entries
            context: Original context with all hypotheses (optional)
            
        Returns:
            List of match results
        """
        results = []
        
        # Create a lookup dictionary of hypotheses
        all_hypotheses = {}
        
        # If context was provided, get hypotheses from it
        if context and "hypotheses" in context:
            for hypothesis in context.get("hypotheses", []):
                hyp_id = hypothesis.get("id")
                if hyp_id:
                    all_hypotheses[hyp_id] = hypothesis
            
        # Determine which hypotheses get full debate treatment
        top_hypothesis_ids = set(
            sorted(
                elo_ratings.keys(), 
                key=lambda h_id: elo_ratings.get(h_id, 0), 
                reverse=True
            )[:int(len(elo_ratings) * self.full_debate_threshold)]
        )
        
        # Run each match
        for match_id, (id1, id2) in enumerate(matches):
            h1 = all_hypotheses.get(id1)
            h2 = all_hypotheses.get(id2)
            
            if not h1 or not h2:
                logger.warning(f"Skipping match {match_id}: Hypothesis not found")
                continue
                
            # Determine match type: Full debate for top hypotheses, simple for others
            is_top_match = id1 in top_hypothesis_ids or id2 in top_hypothesis_ids
            
            if is_top_match:
                # Run multi-turn debate
                winner, rationale = await self._run_debate_match(h1, h2, goal, iteration, feedback)
            else:
                # Run simple comparison
                winner, rationale = await self._run_simple_match(h1, h2, goal, iteration)
            
            # Record result
            results.append({
                "match_id": match_id,
                "hypothesis1_id": id1,
                "hypothesis2_id": id2,
                "winner_id": winner,
                "rationale": rationale,
                "match_type": "debate" if is_top_match else "simple"
            })
            
            logger.info(f"Match {match_id}: {id1} vs {id2} - Winner: {winner}")
            
        return results
    
    async def _run_debate_match(self,
                         h1: Dict[str, Any],
                         h2: Dict[str, Any],
                         goal: Dict[str, Any],
                         iteration: int,
                         feedback: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Run a multi-turn scientific debate match between two hypotheses.
        
        Args:
            h1: First hypothesis
            h2: Second hypothesis
            goal: Research goal information
            iteration: Current iteration number
            feedback: List of feedback entries
            
        Returns:
            Tuple of (winner_id, rationale)
        """
        logger.info(f"Running debate match: {h1.get('id')} vs {h2.get('id')}")
        
        # Create the debate prompt
        debate_system_prompt = """
        You are a panel of scientific experts conducting a structured scientific debate to compare two research hypotheses.
        You will evaluate these hypotheses against multiple criteria to determine which is more promising.
        Express disagreements and support your arguments with scientific reasoning.
        Ultimately, you must reach a consensus about which hypothesis is superior.
        """
        
        goal_text = goal.get("description", "")
        domain = goal.get("domain", "")
        
        debate_prompt = f"""
        RESEARCH GOAL: {goal_text}
        
        DOMAIN: {domain}
        
        HYPOTHESES TO COMPARE:
        
        HYPOTHESIS A:
        {h1.get('text', '')}
        
        Rationale for Hypothesis A:
        {h1.get('rationale', '')}
        
        HYPOTHESIS B:
        {h2.get('text', '')}
        
        Rationale for Hypothesis B:
        {h2.get('rationale', '')}
        
        EVALUATION CRITERIA:
        1. Novelty: How original and innovative is the hypothesis?
        2. Plausibility: How scientifically plausible is the hypothesis?
        3. Testability: How feasible is it to test this hypothesis experimentally?
        4. Impact: If true, how significant would the implications be?
        
        DEBATE INSTRUCTIONS:
        Please conduct a structured scientific debate comparing these two hypotheses.
        Discuss the strengths and weaknesses of each hypothesis according to the evaluation criteria.
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
        
        # Final decision prompt
        decision_prompt = f"""
        Based on the following scientific debate transcript, determine which hypothesis is superior.
        
        DEBATE TRANSCRIPT:
        {debate_transcript}
        
        FINAL DECISION:
        Carefully weigh the arguments presented in the debate and decide whether Hypothesis A or Hypothesis B is superior.
        Provide a clear justification for your decision, summarizing the key points from the debate.
        
        Your response should be in this format:
        WINNER: [A or B]
        JUSTIFICATION: [detailed explanation]
        """
        
        decision_system_prompt = "You are a scientific judge making an impartial decision based on a scientific debate."
        
        decision_response = await self.model.generate(decision_prompt, system_prompt=decision_system_prompt)
        
        # Parse the decision
        import re
        winner_match = re.search(r'WINNER:\s*([AB])', decision_response, re.IGNORECASE)
        justification_match = re.search(r'JUSTIFICATION:\s*(.*)', decision_response, re.DOTALL)
        
        winner_letter = winner_match.group(1).upper() if winner_match else None
        justification = justification_match.group(1).strip() if justification_match else ""
        
        if winner_letter == 'A':
            return h1.get('id'), justification
        elif winner_letter == 'B':
            return h2.get('id'), justification
        else:
            # If parsing failed, make a simple decision
            logger.warning("Failed to parse debate decision, using simple comparison")
            return await self._run_simple_match(h1, h2, goal, iteration)
    
    async def _run_simple_match(self,
                         h1: Dict[str, Any],
                         h2: Dict[str, Any],
                         goal: Dict[str, Any],
                         iteration: int) -> Tuple[str, str]:
        """
        Run a simple comparison match between two hypotheses.
        
        Args:
            h1: First hypothesis
            h2: Second hypothesis
            goal: Research goal information
            iteration: Current iteration number
            
        Returns:
            Tuple of (winner_id, rationale)
        """
        logger.info(f"Running simple match: {h1.get('id')} vs {h2.get('id')}")
        
        system_prompt = """
        You are a scientific evaluator comparing two research hypotheses.
        Your task is to determine which hypothesis is more promising based on the given criteria.
        Provide a clear justification for your decision.
        """
        
        prompt = f"""
        RESEARCH GOAL: {goal.get('description', '')}
        
        HYPOTHESES TO COMPARE:
        
        HYPOTHESIS A:
        {h1.get('text', '')}
        
        Rationale for Hypothesis A:
        {h1.get('rationale', '')}
        
        HYPOTHESIS B:
        {h2.get('text', '')}
        
        Rationale for Hypothesis B:
        {h2.get('rationale', '')}
        
        EVALUATION CRITERIA:
        1. Novelty: How original and innovative is the hypothesis?
        2. Plausibility: How scientifically plausible is the hypothesis?
        3. Testability: How feasible is it to test this hypothesis experimentally?
        4. Impact: If true, how significant would the implications be?
        
        Compare these hypotheses and decide which one is superior.
        
        Your response should be in this format:
        WINNER: [A or B]
        JUSTIFICATION: [detailed explanation]
        """
        
        response = await self.model.generate(prompt, system_prompt=system_prompt)
        
        # Parse the decision
        import re
        winner_match = re.search(r'WINNER:\s*([AB])', response, re.IGNORECASE)
        justification_match = re.search(r'JUSTIFICATION:\s*(.*)', response, re.DOTALL)
        
        winner_letter = winner_match.group(1).upper() if winner_match else None
        justification = justification_match.group(1).strip() if justification_match else ""
        
        if winner_letter == 'A':
            return h1.get('id'), justification
        elif winner_letter == 'B':
            return h2.get('id'), justification
        else:
            # If parsing failed, choose randomly
            logger.warning("Failed to parse simple match decision, choosing randomly")
            return random.choice([h1.get('id'), h2.get('id')]), "Decision could not be determined"
    
    def _update_elo_ratings(self,
                          elo_ratings: Dict[str, float],
                          tournament_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Update Elo ratings based on tournament results.
        
        Args:
            elo_ratings: Current Elo ratings
            tournament_results: List of match results
            
        Returns:
            Updated Elo ratings
        """
        updated_ratings = elo_ratings.copy()
        
        for result in tournament_results:
            id1 = result.get("hypothesis1_id")
            id2 = result.get("hypothesis2_id")
            winner_id = result.get("winner_id")
            
            if not (id1 in updated_ratings and id2 in updated_ratings):
                continue
                
            # Get current ratings
            rating1 = updated_ratings[id1]
            rating2 = updated_ratings[id2]
            
            # Calculate expected scores
            expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
            expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
            
            # Calculate actual scores
            actual1 = 1.0 if winner_id == id1 else 0.0
            actual2 = 1.0 if winner_id == id2 else 0.0
            
            # Update ratings
            updated_ratings[id1] = rating1 + self.elo_k_factor * (actual1 - expected1)
            updated_ratings[id2] = rating2 + self.elo_k_factor * (actual2 - expected2)
        
        return updated_ratings
    
    def _apply_ratings_to_hypotheses(self,
                                  hypotheses: List[Dict[str, Any]],
                                  ratings: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Apply Elo ratings to hypotheses and sort them by rating.
        
        Args:
            hypotheses: List of hypotheses
            ratings: Elo ratings by hypothesis ID
            
        Returns:
            List of hypotheses with ratings applied and sorted
        """
        # Deep copy the hypotheses to avoid modifying the originals
        import copy
        ranked_hypotheses = copy.deepcopy(hypotheses)
        
        # Apply ratings
        for hypothesis in ranked_hypotheses:
            hypothesis_id = hypothesis.get("id")
            if hypothesis_id in ratings:
                # Always use integer or float value for score to avoid type conversion issues
                elo_score = float(ratings[hypothesis_id])
                hypothesis["score"] = elo_score
                
                # Add Elo rating to metadata
                if "metadata" not in hypothesis:
                    hypothesis["metadata"] = {}
                
                hypothesis["metadata"]["elo_rating"] = elo_score
                
                # Log that we're applying a score to help debug
                logger.info(f"Applied Elo score {elo_score} to hypothesis {hypothesis_id}")
        
        # Sort by rating (descending)
        ranked_hypotheses.sort(key=lambda h: h.get("score", 0), reverse=True)
        
        return ranked_hypotheses 