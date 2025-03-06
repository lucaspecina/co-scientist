"""
Proximity Agent for AI Co-Scientist

This module implements the Proximity Agent, which calculates similarity between
research hypotheses and builds a proximity graph for organizing related ideas.
"""

import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from .base_agent import BaseAgent, AgentExecutionError
from ..tools.literature_search import LiteratureSearch, PaperMetadata

logger = logging.getLogger(__name__)


class ProximityAgent(BaseAgent):
    """
    Proximity Agent calculates similarity between research hypotheses.
    
    This agent builds a proximity graph that represents the semantic similarity
    between different hypotheses, enabling the organization of related ideas
    and helping the Ranking agent to structure tournament matches efficiently.
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the proximity agent.
        
        Args:
            model: Language model to use
            config: Configuration dictionary
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.similarity_threshold = config.get("similarity_threshold", 0.7)  # Minimum similarity to consider
        self.max_neighbors = config.get("max_neighbors", 5)  # Maximum number of neighbors per hypothesis
        self.use_embeddings = config.get("use_embeddings", True)  # Whether to use embeddings for similarity
        self.embedding_model = config.get("embedding_model", "default")  # Model to use for embeddings
        
        # Initialize tools
        tools_config = config.get("tools", {})
        self.literature_search = None
        self._init_literature_search(tools_config.get("literature_search", {}))
        
    def _init_literature_search(self, config: Dict[str, Any]) -> None:
        """
        Initialize the literature search tool.
        
        Args:
            config: Literature search configuration
        """
        email = config.get("email", "researcher@example.com")
        api_keys = config.get("api_keys", {})
        
        try:
            self.literature_search = LiteratureSearch(
                email=email,
                api_keys=api_keys
            )
            logger.info("Literature search tool initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize literature search: {str(e)}")
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate similarity between hypotheses and build a proximity graph.
        
        Args:
            context: Dictionary containing:
                - goal: Research goal information
                - hypotheses: List of hypotheses to analyze
                - iteration: Current iteration number
            params: Dictionary containing:
                - similarity_threshold: Minimum similarity to consider (optional)
                - max_neighbors: Maximum number of neighbors per hypothesis (optional)
                
        Returns:
            Dictionary containing:
                - proximity_graph: Dictionary mapping hypothesis IDs to lists of similar hypothesis IDs
                - similarity_matrix: Matrix of similarity scores between hypotheses
                - clusters: List of hypothesis clusters
        """
        goal = context.get("goal", {})
        hypotheses = context.get("hypotheses", [])
        iteration = context.get("iteration", 0)
        
        # Extract parameters
        similarity_threshold = params.get("similarity_threshold", self.similarity_threshold)
        max_neighbors = params.get("max_neighbors", self.max_neighbors)
        
        if not hypotheses:
            logger.warning("No hypotheses provided for proximity analysis")
            return {
                "proximity_graph": {},
                "similarity_matrix": [],
                "clusters": []
            }
        
        try:
            # Calculate similarity matrix
            similarity_matrix = await self._calculate_similarity_matrix(hypotheses, goal)
            
            # Build proximity graph
            proximity_graph = self._build_proximity_graph(
                hypotheses, 
                similarity_matrix,
                similarity_threshold,
                max_neighbors
            )
            
            # Identify clusters
            clusters = self._identify_clusters(hypotheses, proximity_graph)
            
            return {
                "proximity_graph": proximity_graph,
                "similarity_matrix": similarity_matrix.tolist() if isinstance(similarity_matrix, np.ndarray) else similarity_matrix,
                "clusters": clusters
            }
        except Exception as e:
            logger.error(f"Error in proximity agent: {str(e)}")
            raise AgentExecutionError(f"Failed to build proximity graph: {str(e)}")
    
    async def _calculate_similarity_matrix(self, 
                                    hypotheses: List[Dict[str, Any]], 
                                    goal: Dict[str, Any]) -> np.ndarray:
        """
        Calculate similarity matrix between all pairs of hypotheses.
        
        Args:
            hypotheses: List of hypotheses
            goal: Research goal information
            
        Returns:
            Similarity matrix (n x n) where n is the number of hypotheses
        """
        n = len(hypotheses)
        similarity_matrix = np.zeros((n, n))
        
        # If using embeddings, calculate them first
        if self.use_embeddings and hasattr(self.model, 'get_embeddings'):
            # Extract hypothesis texts
            texts = [h.get("text", "") for h in hypotheses]
            
            try:
                # Get embeddings from the model
                embeddings = await self.model.get_embeddings(texts)
                
                # Calculate cosine similarity between all pairs
                for i in range(n):
                    for j in range(i, n):
                        if i == j:
                            similarity_matrix[i, j] = 1.0  # Self-similarity is 1.0
                        else:
                            # Cosine similarity
                            sim = self._cosine_similarity(embeddings[i], embeddings[j])
                            similarity_matrix[i, j] = sim
                            similarity_matrix[j, i] = sim  # Symmetric
                
                return similarity_matrix
            except Exception as e:
                logger.warning(f"Error calculating embeddings: {str(e)}. Falling back to LLM-based similarity.")
        
        # If embeddings are not available or failed, use LLM to calculate similarity
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Self-similarity is 1.0
                else:
                    # Calculate similarity using LLM
                    sim = await self._calculate_hypothesis_similarity(hypotheses[i], hypotheses[j], goal)
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim  # Symmetric
        
        return similarity_matrix
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Cosine similarity (0.0-1.0)
        """
        # Convert to numpy arrays
        a = np.array(v1)
        b = np.array(v2)
        
        # Calculate cosine similarity
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    async def _calculate_hypothesis_similarity(self, 
                                       h1: Dict[str, Any], 
                                       h2: Dict[str, Any],
                                       goal: Dict[str, Any]) -> float:
        """
        Calculate similarity between two hypotheses using the LLM.
        
        Args:
            h1: First hypothesis
            h2: Second hypothesis
            goal: Research goal information
            
        Returns:
            Similarity score (0.0-1.0)
        """
        system_prompt = """
        You are a scientific evaluator assessing the similarity between two research hypotheses.
        Your task is to determine how similar these hypotheses are in terms of their core ideas, approaches, and implications.
        Consider both conceptual similarity and methodological similarity.
        Provide a similarity score between 0.0 (completely different) and 1.0 (identical).
        """
        
        prompt = f"""
        RESEARCH GOAL: {goal.get('description', '')}
        
        HYPOTHESIS A:
        {h1.get('text', '')}
        
        HYPOTHESIS B:
        {h2.get('text', '')}
        
        Analyze these two hypotheses and determine their similarity on a scale from 0.0 to 1.0, where:
        - 0.0: Completely different, addressing entirely separate aspects with no overlap
        - 0.3: Minimal similarity, with only tangential connections
        - 0.5: Moderate similarity, addressing related aspects but with different approaches
        - 0.7: Substantial similarity, with overlapping core ideas but some distinct elements
        - 1.0: Nearly identical hypotheses
        
        Consider:
        1. Conceptual similarity: Do they address the same core concepts or mechanisms?
        2. Methodological similarity: Do they suggest similar experimental approaches?
        3. Implication similarity: Would they lead to similar outcomes if proven true?
        
        Provide your assessment as a single number between 0.0 and 1.0.
        """
        
        response = await self.model.generate(prompt, system_prompt=system_prompt)
        
        # Extract the similarity score
        try:
            # Look for a number between 0.0 and 1.0
            import re
            match = re.search(r'(\d+\.\d+|\d+)', response)
            if match:
                similarity = float(match.group(1))
                # Ensure it's between 0.0 and 1.0
                similarity = max(0.0, min(1.0, similarity))
                return similarity
            else:
                logger.warning(f"Could not extract similarity score from response: {response}")
                return 0.5  # Default to moderate similarity
        except Exception as e:
            logger.error(f"Error parsing similarity score: {str(e)}")
            return 0.5  # Default to moderate similarity
    
    def _build_proximity_graph(self,
                            hypotheses: List[Dict[str, Any]],
                            similarity_matrix: np.ndarray,
                            similarity_threshold: float,
                            max_neighbors: int) -> Dict[str, List[str]]:
        """
        Build a proximity graph based on hypothesis similarity.
        
        Args:
            hypotheses: List of hypotheses
            similarity_matrix: Matrix of similarity scores
            similarity_threshold: Minimum similarity to consider
            max_neighbors: Maximum number of neighbors per hypothesis
            
        Returns:
            Dictionary mapping hypothesis IDs to lists of similar hypothesis IDs
        """
        n = len(hypotheses)
        proximity_graph = {}
        
        for i in range(n):
            h_id = hypotheses[i].get("id")
            proximity_graph[h_id] = []
            
            # Get similarity scores for this hypothesis
            similarities = [(j, similarity_matrix[i, j]) for j in range(n) if i != j]
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Add top neighbors that meet the threshold
            for j, sim in similarities:
                if sim >= similarity_threshold and len(proximity_graph[h_id]) < max_neighbors:
                    neighbor_id = hypotheses[j].get("id")
                    proximity_graph[h_id].append(neighbor_id)
        
        return proximity_graph
    
    def _identify_clusters(self,
                        hypotheses: List[Dict[str, Any]],
                        proximity_graph: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Identify clusters of related hypotheses.
        
        Args:
            hypotheses: List of hypotheses
            proximity_graph: Proximity graph
            
        Returns:
            List of clusters, each containing hypothesis IDs and a representative hypothesis
        """
        # Create a lookup dictionary for hypotheses
        hypothesis_dict = {h.get("id"): h for h in hypotheses}
        
        # Track which hypotheses have been assigned to clusters
        assigned = set()
        clusters = []
        
        # Helper function for DFS to find connected components
        def dfs(node: str, cluster: Set[str]):
            if node in assigned:
                return
            
            assigned.add(node)
            cluster.add(node)
            
            # Visit neighbors
            for neighbor in proximity_graph.get(node, []):
                dfs(neighbor, cluster)
        
        # Find connected components (clusters)
        for h in hypotheses:
            h_id = h.get("id")
            
            if h_id not in assigned:
                # Start a new cluster
                cluster = set()
                dfs(h_id, cluster)
                
                if cluster:
                    # Find the most central hypothesis in the cluster
                    # (the one with the most connections to others in the cluster)
                    central_id = max(
                        cluster, 
                        key=lambda node: sum(1 for neighbor in proximity_graph.get(node, []) if neighbor in cluster)
                    )
                    
                    clusters.append({
                        "id": f"cluster_{len(clusters)}",
                        "hypothesis_ids": list(cluster),
                        "representative_id": central_id,
                        "representative": hypothesis_dict.get(central_id),
                        "size": len(cluster)
                    })
        
        # Sort clusters by size (descending)
        clusters.sort(key=lambda c: c.get("size", 0), reverse=True)
        
        return clusters 