"""
Elo-based tournament system for ranking scientific hypotheses.

This module implements a tournament system where hypotheses compete in pairwise 
comparisons and are ranked using an Elo rating system. The system enables:
- Pairwise comparison of hypotheses through scientific debates
- Elo rating updates based on match outcomes
- Efficient matchmaking for tournament games
- Prioritization of matches to maximize information gain
"""

import math
import random
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class HypothesisEntry:
    """Represents a hypothesis in the tournament system."""
    hypothesis_id: str
    data: Any  # The actual hypothesis data
    elo_rating: float = 1200.0
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    similar_hypotheses: List[str] = field(default_factory=list)
    
    def update_rating(self, new_rating: float):
        """Update the Elo rating for this hypothesis."""
        self.elo_rating = new_rating
        self.last_updated = datetime.now()
    
    def record_match(self, won: bool):
        """Record a match result."""
        self.matches_played += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
        
    def volatility_factor(self) -> float:
        """Calculate a factor representing how volatile this rating is.
        
        Returns a higher value for entries with fewer matches, indicating
        less certainty about their true rating.
        """
        if self.matches_played == 0:
            return 2.0  # High uncertainty for untested hypotheses
        return 1.0 + (5.0 / (self.matches_played + 5))


class EloTournament:
    """Implements an Elo-based tournament system for scientific hypotheses."""
    
    def __init__(
        self, 
        k_factor: float = 32.0,
        initial_rating: float = 1200.0,
        prioritize_similar: bool = True,
        prioritize_new: bool = True,
        prioritize_top_ranked: bool = True
    ):
        """Initialize the tournament system.
        
        Args:
            k_factor: Determines how much ratings change after each match
            initial_rating: Starting Elo rating for new hypotheses
            prioritize_similar: Whether to prioritize matches between similar hypotheses
            prioritize_new: Whether to prioritize matches involving new hypotheses
            prioritize_top_ranked: Whether to prioritize matches involving top-ranked hypotheses
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.hypotheses: Dict[str, HypothesisEntry] = {}
        self.match_history: List[Dict] = []
        self.prioritize_similar = prioritize_similar
        self.prioritize_new = prioritize_new
        self.prioritize_top_ranked = prioritize_top_ranked
    
    def add_hypothesis(self, hypothesis_id: str, hypothesis_data: Any) -> None:
        """Add a new hypothesis to the tournament.
        
        Args:
            hypothesis_id: Unique identifier for the hypothesis
            hypothesis_data: The actual hypothesis content
        """
        if hypothesis_id in self.hypotheses:
            logger.warning(f"Hypothesis {hypothesis_id} already exists in tournament")
            return
        
        self.hypotheses[hypothesis_id] = HypothesisEntry(
            hypothesis_id=hypothesis_id,
            data=hypothesis_data,
            elo_rating=self.initial_rating
        )
        logger.info(f"Added hypothesis {hypothesis_id} to tournament with initial rating {self.initial_rating}")
        
    def update_similarity_graph(self, similarity_dict: Dict[str, List[str]]) -> None:
        """Update the similarity graph for hypotheses.
        
        Args:
            similarity_dict: Dictionary mapping hypothesis IDs to lists of similar hypothesis IDs
        """
        for hyp_id, similar_ids in similarity_dict.items():
            if hyp_id in self.hypotheses:
                # Filter only to include hypotheses that are in the tournament
                valid_similar = [h_id for h_id in similar_ids if h_id in self.hypotheses]
                self.hypotheses[hyp_id].similar_hypotheses = valid_similar
    
    def expected_outcome(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected outcome based on Elo ratings.
        
        Args:
            rating_a: Elo rating of first hypothesis
            rating_b: Elo rating of second hypothesis
            
        Returns:
            Expected outcome (win probability) for hypothesis A
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, winner_id: str, loser_id: str, draw: bool = False) -> Tuple[float, float]:
        """Update Elo ratings after a match.
        
        Args:
            winner_id: ID of winning hypothesis (or first hypothesis if draw)
            loser_id: ID of losing hypothesis (or second hypothesis if draw)
            draw: Whether the match was a draw
            
        Returns:
            Tuple of (new winner rating, new loser rating)
        """
        if winner_id not in self.hypotheses or loser_id not in self.hypotheses:
            raise ValueError(f"Both hypotheses must be in tournament: {winner_id}, {loser_id}")
        
        winner = self.hypotheses[winner_id]
        loser = self.hypotheses[loser_id]
        
        # Calculate expected outcomes
        winner_expected = self.expected_outcome(winner.elo_rating, loser.elo_rating)
        loser_expected = 1.0 - winner_expected
        
        # Calculate dynamic K-factor based on match count and rating difference
        # More volatile for hypotheses with fewer matches
        winner_k = self.k_factor * winner.volatility_factor()
        loser_k = self.k_factor * loser.volatility_factor()
        
        # Calculate actual outcome
        winner_actual = 0.5 if draw else 1.0
        loser_actual = 0.5 if draw else 0.0
        
        # Update ratings
        new_winner_rating = winner.elo_rating + winner_k * (winner_actual - winner_expected)
        new_loser_rating = loser.elo_rating + loser_k * (loser_actual - loser_expected)
        
        # Update entries
        winner.update_rating(new_winner_rating)
        loser.update_rating(new_loser_rating)
        
        # Record match results
        winner.record_match(not draw)
        loser.record_match(False)
        
        # Record match in history
        match_record = {
            "timestamp": datetime.now(),
            "winner_id": winner_id if not draw else None,
            "hypothesis1_id": winner_id,
            "hypothesis2_id": loser_id,
            "hypothesis1_rating_before": winner.elo_rating,
            "hypothesis2_rating_before": loser.elo_rating,
            "hypothesis1_rating_after": new_winner_rating,
            "hypothesis2_rating_after": new_loser_rating,
            "draw": draw
        }
        self.match_history.append(match_record)
        
        logger.info(f"Updated ratings after match: {winner_id} vs {loser_id}, "
                   f"Result: {'Draw' if draw else 'Win for ' + winner_id}")
        
        return new_winner_rating, new_loser_rating
    
    def record_match_result(
        self, 
        hypothesis1_id: str, 
        hypothesis2_id: str, 
        winner_id: Optional[str] = None,
        evaluation: Optional[Dict] = None
    ) -> None:
        """Record the result of a match between two hypotheses.
        
        Args:
            hypothesis1_id: ID of first hypothesis
            hypothesis2_id: ID of second hypothesis
            winner_id: ID of winning hypothesis (None for draw)
            evaluation: Optional evaluation data from the debate
        """
        if hypothesis1_id not in self.hypotheses or hypothesis2_id not in self.hypotheses:
            raise ValueError(f"Both hypotheses must be in tournament: {hypothesis1_id}, {hypothesis2_id}")
        
        # Handle case where winner_id might be None but we should default to one of the hypotheses
        if winner_id is None:
            # Default to avoiding draws which can break tournament scoring
            # We'll randomly choose a winner in this case
            winner_id = random.choice([hypothesis1_id, hypothesis2_id])
            draw = False
            logger.info(f"No winner specified, randomly selected {winner_id} as winner")
        elif winner_id not in [hypothesis1_id, hypothesis2_id]:
            logger.warning(f"Winner {winner_id} is not one of the hypotheses in the match, defaulting to {hypothesis1_id}")
            winner_id = hypothesis1_id
            draw = False
        else:
            draw = False  # If we have a valid winner_id, it's not a draw
        
        # Determine winner and loser
        loser_id = hypothesis2_id if winner_id == hypothesis1_id else hypothesis1_id
        
        # Update the ratings
        self.update_ratings(winner_id, loser_id, draw=draw)
        
        # Save evaluation data if provided
        if evaluation:
            # Add evaluation to match history
            match_record = {
                "timestamp": datetime.now(),
                "winner_id": winner_id,
                "hypothesis1_id": hypothesis1_id,
                "hypothesis2_id": hypothesis2_id,
                "evaluation": evaluation
            }
            self.match_history.append(match_record)
    
    def get_prioritized_match(self) -> Tuple[str, str]:
        """Get the next match to run based on prioritization criteria.
        
        Returns:
            Tuple of (hypothesis1_id, hypothesis2_id)
        """
        if len(self.hypotheses) < 2:
            raise ValueError("Need at least 2 hypotheses for a match")
        
        # Start with all possible pairs
        all_hypotheses = list(self.hypotheses.keys())
        all_pairs = [(a, b) for a in all_hypotheses for b in all_hypotheses if a < b]
        
        if not all_pairs:
            # Fallback if we somehow have no valid pairs
            return random.sample(all_hypotheses, 2)
        
        # Score each potential match based on our prioritization criteria
        scored_pairs = []
        
        for hyp1_id, hyp2_id in all_pairs:
            hyp1 = self.hypotheses[hyp1_id]
            hyp2 = self.hypotheses[hyp2_id]
            
            score = 0.0
            
            # Prioritize matches between hypotheses that haven't played many games
            unplayed_factor = 2.0 / (hyp1.matches_played + hyp2.matches_played + 2.0)
            score += unplayed_factor * 10
            
            # Prioritize matches between similar hypotheses if enabled
            if self.prioritize_similar and (
                hyp2_id in hyp1.similar_hypotheses or hyp1_id in hyp2.similar_hypotheses
            ):
                score += 5.0
            
            # Prioritize newer hypotheses if enabled
            if self.prioritize_new:
                now = datetime.now()
                recency1 = 1.0 / max(1, (now - hyp1.created_at).total_seconds() / 3600)  # Hours
                recency2 = 1.0 / max(1, (now - hyp2.created_at).total_seconds() / 3600)
                score += (recency1 + recency2) * 3.0
            
            # Prioritize top-ranked hypotheses if enabled
            if self.prioritize_top_ranked:
                # Higher ratings get priority
                rating_factor = (hyp1.elo_rating + hyp2.elo_rating - 2 * self.initial_rating) / 200
                score += max(0, rating_factor)
            
            # Add some randomness to prevent getting stuck in patterns
            score += random.uniform(0, 1)
            
            scored_pairs.append((score, (hyp1_id, hyp2_id)))
        
        # Sort by score (highest first)
        scored_pairs.sort(reverse=True)
        
        # Return the highest-scored pair
        return scored_pairs[0][1]
    
    def get_rankings(self) -> List[Tuple[str, float, int]]:
        """Get the current rankings of all hypotheses.
        
        Returns:
            List of tuples (hypothesis_id, elo_rating, matches_played)
            ordered by rating (highest first)
        """
        rankings = [
            (h_id, entry.elo_rating, entry.matches_played)
            for h_id, entry in self.hypotheses.items()
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_top_hypotheses(self, n: int = 10) -> List[str]:
        """Get the IDs of the top N hypotheses by rating.
        
        Args:
            n: Number of top hypotheses to return
            
        Returns:
            List of hypothesis IDs
        """
        rankings = self.get_rankings()
        return [h_id for h_id, _, _ in rankings[:n]]
    
    def get_hypothesis_data(self, hypothesis_id: str) -> Any:
        """Get the data for a specific hypothesis.
        
        Args:
            hypothesis_id: ID of the hypothesis
            
        Returns:
            The hypothesis data
        """
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis {hypothesis_id} not in tournament")
        return self.hypotheses[hypothesis_id].data
    
    def get_tournament_statistics(self) -> Dict:
        """Get statistics about the current state of the tournament.
        
        Returns:
            Dictionary with tournament statistics
        """
        if not self.hypotheses:
            return {
                "total_hypotheses": 0,
                "total_matches": 0,
                "avg_rating": self.initial_rating,
                "avg_matches_per_hypothesis": 0,
                "highest_rated": None,
                "most_matches": None,
            }
        
        total_hypotheses = len(self.hypotheses)
        total_matches = len(self.match_history)
        avg_rating = sum(h.elo_rating for h in self.hypotheses.values()) / total_hypotheses
        avg_matches = sum(h.matches_played for h in self.hypotheses.values()) / total_hypotheses
        
        # Find highest rated and most played
        highest_rated = max(self.hypotheses.items(), key=lambda x: x[1].elo_rating)
        most_matches = max(self.hypotheses.items(), key=lambda x: x[1].matches_played)
        
        return {
            "total_hypotheses": total_hypotheses,
            "total_matches": total_matches,
            "avg_rating": avg_rating,
            "avg_matches_per_hypothesis": avg_matches,
            "highest_rated": {
                "id": highest_rated[0],
                "rating": highest_rated[1].elo_rating,
                "matches": highest_rated[1].matches_played
            },
            "most_matches": {
                "id": most_matches[0],
                "rating": most_matches[1].elo_rating,
                "matches": most_matches[1].matches_played
            }
        }