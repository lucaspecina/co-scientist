"""
Example demonstrating the Elo tournament and debate features.

This example creates a simple tournament, adds hypotheses, and simulates matches.
"""

import asyncio
from core.tournament.elo_tournament import EloTournament
from core.debate.scientific_debate import ScientificDebate, DebateFormat
from core.models.azure_openai_model import AzureOpenAIModel
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def run_example():
    """Run the tournament and debate example."""
    print("Running tournament and debate example")
    print("-" * 40)
    
    # Create a simple tournament
    tournament = EloTournament(
        k_factor=32.0,
        initial_rating=1200.0,
        prioritize_similar=True,
        prioritize_new=True,
        prioritize_top_ranked=True
    )
    
    # Create some test hypotheses
    hypotheses = {
        "hyp1": {
            "id": "hyp1",
            "title": "Hypothesis 1",
            "description": "Increased calcium levels in neurons lead to enhanced synaptic plasticity",
            "text": "Increased calcium levels in neurons lead to enhanced synaptic plasticity",
            "score": 0.0
        },
        "hyp2": {
            "id": "hyp2",
            "title": "Hypothesis 2",
            "description": "Decreased calcium levels in neurons inhibit synaptic plasticity",
            "text": "Decreased calcium levels in neurons inhibit synaptic plasticity",
            "score": 0.0
        },
        "hyp3": {
            "id": "hyp3",
            "title": "Hypothesis 3",
            "description": "Potassium channel blockers enhance synaptic plasticity by increasing calcium influx",
            "text": "Potassium channel blockers enhance synaptic plasticity by increasing calcium influx",
            "score": 0.0
        }
    }
    
    # Add hypotheses to the tournament
    for hyp_id, hyp_data in hypotheses.items():
        tournament.add_hypothesis(hyp_id, hyp_data)
    
    # Create similarity relationships
    similarity_graph = {
        "hyp1": ["hyp2"],
        "hyp2": ["hyp1"],
        "hyp3": []
    }
    tournament.update_similarity_graph(similarity_graph)
    
    # Create a model for debates
    model_config = {
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "deployment_id": os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "co-scientist-gpt-4o-mini"),
        "temperature": 0.3,
        "max_tokens": 2000
    }
    
    model = AzureOpenAIModel(model_config)
    
    # Create a debate system
    debate = ScientificDebate(
        model_provider=model,
        format=DebateFormat.SINGLE_TURN,
        max_turns=1,
        evaluation_criteria=["novelty", "plausibility", "testability"],
        use_mediator=False
    )
    
    print("Conducting scientific debates between hypotheses...")
    
    # Conduct a single debate
    research_goal = "Understand the role of calcium in synaptic plasticity"
    debate_result = await debate.conduct_debate(
        hypothesis1=hypotheses["hyp1"],
        hypothesis2=hypotheses["hyp2"],
        research_goal=research_goal
    )
    
    # Record the result in the tournament
    winner_id = None
    if debate_result["winner"] == "hypothesis1":
        winner_id = "hyp1"
    elif debate_result["winner"] == "hypothesis2":
        winner_id = "hyp2"
    
    tournament.record_match_result("hyp1", "hyp2", winner_id)
    
    # Display results
    print("\nDebate Results:")
    print("-" * 40)
    print(f"Debate ID: {debate_result['debate_id']}")
    print(f"Winner: {debate_result['winner']}")
    print(f"Justification: {debate_result['justification']}")
    
    # Display rankings
    print("\nTournament Rankings:")
    print("-" * 40)
    rankings = tournament.get_rankings()
    for i, (hyp_id, elo_rating, matches) in enumerate(rankings, 1):
        print(f"{i}. {hyp_id}: Elo {elo_rating:.1f} ({matches} matches)")
    
    # Get tournament statistics
    print("\nTournament Statistics:")
    print("-" * 40)
    stats = tournament.get_tournament_statistics()
    print(f"Total Hypotheses: {stats['total_hypotheses']}")
    print(f"Total Matches: {stats['total_matches']}")
    print(f"Average Elo Rating: {stats['avg_rating']:.1f}")
    
    return "Example completed successfully!"

if __name__ == "__main__":
    result = asyncio.run(run_example())
    print("\n" + result)