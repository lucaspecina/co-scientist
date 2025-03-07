"""
Test the Elo Tournament system for the AI Co-Scientist.
"""

import unittest
import random
from core.tournament.elo_tournament import EloTournament


class TestTournament(unittest.TestCase):
    """Test cases for the Elo Tournament system."""
    
    def setUp(self):
        """Set up the tournament for testing."""
        self.tournament = EloTournament(
            k_factor=32.0,
            initial_rating=1200.0
        )
        
        # Add some test hypotheses
        self.hypotheses = {
            "hyp1": {"id": "hyp1", "text": "Hypothesis 1", "score": 0.0},
            "hyp2": {"id": "hyp2", "text": "Hypothesis 2", "score": 0.0},
            "hyp3": {"id": "hyp3", "text": "Hypothesis 3", "score": 0.0},
            "hyp4": {"id": "hyp4", "text": "Hypothesis 4", "score": 0.0},
            "hyp5": {"id": "hyp5", "text": "Hypothesis 5", "score": 0.0},
        }
        
        for hyp_id, hyp_data in self.hypotheses.items():
            self.tournament.add_hypothesis(hyp_id, hyp_data)
    
    def test_initial_ratings(self):
        """Test that initial ratings are set correctly."""
        for hyp_id in self.hypotheses:
            self.assertEqual(
                self.tournament.hypotheses[hyp_id].elo_rating, 
                1200.0
            )
    
    def test_match_result(self):
        """Test recording match results."""
        # Record a win for hyp1 against hyp2
        self.tournament.record_match_result("hyp1", "hyp2", "hyp1")
        
        # Check that ratings were updated
        self.assertGreater(
            self.tournament.hypotheses["hyp1"].elo_rating, 
            1200.0
        )
        self.assertLess(
            self.tournament.hypotheses["hyp2"].elo_rating, 
            1200.0
        )
        
        # Check that match counts were updated
        self.assertEqual(self.tournament.hypotheses["hyp1"].matches_played, 1)
        self.assertEqual(self.tournament.hypotheses["hyp2"].matches_played, 1)
        
        # Check that win/loss records were updated
        self.assertEqual(self.tournament.hypotheses["hyp1"].wins, 1)
        self.assertEqual(self.tournament.hypotheses["hyp1"].losses, 0)
        self.assertEqual(self.tournament.hypotheses["hyp2"].wins, 0)
        self.assertEqual(self.tournament.hypotheses["hyp2"].losses, 1)
    
    def test_draw_result(self):
        """Test recording a draw result."""
        # Record a draw between hyp3 and hyp4
        self.tournament.record_match_result("hyp3", "hyp4", None)
        
        # Ratings should be unchanged for a draw between equal ratings
        self.assertAlmostEqual(
            self.tournament.hypotheses["hyp3"].elo_rating, 
            1200.0,
            places=1
        )
        self.assertAlmostEqual(
            self.tournament.hypotheses["hyp4"].elo_rating, 
            1200.0,
            places=1
        )
        
        # Match counts should still be updated
        self.assertEqual(self.tournament.hypotheses["hyp3"].matches_played, 1)
        self.assertEqual(self.tournament.hypotheses["hyp4"].matches_played, 1)
    
    def test_rankings(self):
        """Test getting rankings."""
        # Create some matches with predetermined outcomes
        self.tournament.record_match_result("hyp1", "hyp2", "hyp1")
        self.tournament.record_match_result("hyp1", "hyp3", "hyp1")
        self.tournament.record_match_result("hyp2", "hyp3", "hyp2")
        self.tournament.record_match_result("hyp4", "hyp5", "hyp5")
        
        # Get rankings
        rankings = self.tournament.get_rankings()
        
        # Check that hyp1 is ranked highest
        self.assertEqual(rankings[0][0], "hyp1")
        
        # Check that hyp3 is ranked lowest among hyp1, hyp2, hyp3
        hyp3_rank = next(i for i, r in enumerate(rankings) if r[0] == "hyp3")
        hyp1_rank = next(i for i, r in enumerate(rankings) if r[0] == "hyp1")
        hyp2_rank = next(i for i, r in enumerate(rankings) if r[0] == "hyp2")
        
        self.assertGreater(hyp3_rank, hyp1_rank)
        self.assertGreater(hyp3_rank, hyp2_rank)
    
    def test_top_hypotheses(self):
        """Test getting top hypotheses."""
        # Create some matches with predetermined outcomes
        self.tournament.record_match_result("hyp1", "hyp2", "hyp1")
        self.tournament.record_match_result("hyp1", "hyp3", "hyp1")
        self.tournament.record_match_result("hyp2", "hyp3", "hyp2")
        self.tournament.record_match_result("hyp4", "hyp5", "hyp5")
        
        # Get top 2 hypotheses
        top_hypotheses = self.tournament.get_top_hypotheses(2)
        
        # hyp1 should be in the top 2
        self.assertIn("hyp1", top_hypotheses)
        
        # We should get exactly 2 hypotheses
        self.assertEqual(len(top_hypotheses), 2)
    
    def test_prioritized_match(self):
        """Test getting prioritized matches."""
        # Add similarity information
        similarity_graph = {
            "hyp1": ["hyp2"],
            "hyp2": ["hyp1"],
            "hyp3": ["hyp4"],
            "hyp4": ["hyp3"],
        }
        self.tournament.update_similarity_graph(similarity_graph)
        
        # Get a prioritized match
        match = self.tournament.get_prioritized_match()
        
        # We should get a tuple of two hypothesis IDs
        self.assertEqual(len(match), 2)
        self.assertIn(match[0], self.hypotheses.keys())
        self.assertIn(match[1], self.hypotheses.keys())
        
        # They should be different hypotheses
        self.assertNotEqual(match[0], match[1])
    
    def test_many_matches(self):
        """Test running many matches to ensure stability."""
        # Run 100 random matches
        for _ in range(100):
            hyp1, hyp2 = random.sample(list(self.hypotheses.keys()), 2)
            winner = random.choice([hyp1, hyp2, None])  # None for draw
            self.tournament.record_match_result(hyp1, hyp2, winner)
        
        # Get tournament statistics
        stats = self.tournament.get_tournament_statistics()
        
        # Check that statistics are reasonable
        self.assertEqual(stats["total_hypotheses"], 5)
        self.assertEqual(stats["total_matches"], 100)
        self.assertGreaterEqual(stats["avg_matches_per_hypothesis"], 30)  # ~40 expected


if __name__ == "__main__":
    unittest.main()