#!/usr/bin/env python3
"""
Example of using tool integrations in the AI Co-Scientist system.

This example demonstrates how to:
1. Create a custom tool adapter
2. Register it with the system
3. Use it in a research session
"""

import asyncio
import os
import sys
import logging
import yaml
import json
from typing import Dict, Any, List, Optional
import requests
from datetime import datetime

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tools.base_tool import BaseTool
from core.tools.tool_registry import ToolRegistry
from core.models.model_factory import ModelFactory
from core.agents.generation_agent import GenerationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class PubMedSearchTool(BaseTool):
    """
    Example tool for searching PubMed for scientific literature.
    
    This is a simplified implementation that uses the PubMed API
    to search for articles related to a query.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PubMed search tool.
        
        Args:
            config (Dict[str, Any]): Configuration for the tool
        """
        super().__init__(config)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.max_results = config.get("max_results", 5)
        logger.info(f"Initialized PubMedSearchTool with max_results: {self.max_results}")
    
    async def search(self, query: str) -> Dict[str, Any]:
        """
        Search PubMed for articles related to the query.
        
        Args:
            query (str): The search query
            
        Returns:
            Dict[str, Any]: The search results
        """
        logger.info(f"Searching PubMed for: {query}")
        
        # For demonstration purposes, we'll use a simplified version of the PubMed API
        # In a real implementation, you would handle pagination, error handling, etc.
        
        # First, search for article IDs
        search_url = f"{self.base_url}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": self.max_results
        }
        
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            search_data = response.json()
            
            # Get the article IDs
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            
            if not id_list:
                logger.warning(f"No results found for query: {query}")
                return {"articles": []}
            
            # Fetch article details
            summary_url = f"{self.base_url}/esummary.fcgi"
            params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json"
            }
            
            response = requests.get(summary_url, params=params)
            response.raise_for_status()
            summary_data = response.json()
            
            # Extract article information
            articles = []
            for article_id in id_list:
                article_data = summary_data.get("result", {}).get(article_id, {})
                
                if article_data:
                    article = {
                        "id": article_id,
                        "title": article_data.get("title", ""),
                        "authors": [author.get("name", "") for author in article_data.get("authors", [])],
                        "journal": article_data.get("fulljournalname", ""),
                        "publication_date": article_data.get("pubdate", ""),
                        "abstract": article_data.get("abstract", ""),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/"
                    }
                    articles.append(article)
            
            return {"articles": articles}
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            return {"error": str(e), "articles": []}
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the tool with the given parameters.
        
        Args:
            params (Dict[str, Any]): The parameters for the tool
            
        Returns:
            Dict[str, Any]: The tool results
        """
        query = params.get("query")
        if not query:
            return {"error": "No query provided", "articles": []}
        
        return await self.search(query)

class MockPubMedSearchTool(BaseTool):
    """
    Mock version of the PubMed search tool for demonstration purposes.
    
    This tool returns predefined responses instead of making actual API calls.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the mock PubMed search tool.
        
        Args:
            config (Dict[str, Any]): Configuration for the tool
        """
        super().__init__(config)
        self.responses = config.get("responses", {})
        logger.info("Initialized MockPubMedSearchTool")
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the tool with the given parameters.
        
        Args:
            params (Dict[str, Any]): The parameters for the tool
            
        Returns:
            Dict[str, Any]: The tool results
        """
        query = params.get("query", "")
        logger.info(f"Mock searching PubMed for: {query}")
        
        # Return a predefined response based on keywords in the query
        for keyword, response in self.responses.items():
            if keyword.lower() in query.lower():
                logger.info(f"Found keyword '{keyword}' in query, returning predefined response")
                return response
        
        # Default response if no keyword matches
        logger.info("No keyword match found, returning default response")
        return self.responses.get("default", {"articles": []})

# Register the tools with the ToolRegistry
ToolRegistry.register_tool("pubmed_search", PubMedSearchTool)
ToolRegistry.register_tool("mock_pubmed_search", MockPubMedSearchTool)

async def run_with_tool_integration():
    """
    Run an example using the tool integration.
    """
    # Define mock responses for the tool
    mock_responses = {
        "alzheimer": {
            "articles": [
                {
                    "id": "12345678",
                    "title": "TREM2 in Alzheimer's Disease: Microglial Survival and Energy Metabolism",
                    "authors": ["Smith J", "Johnson A", "Williams B"],
                    "journal": "Journal of Neuroinflammation",
                    "publication_date": "2023 Jan 15",
                    "abstract": "TREM2 is a microglial receptor that has been implicated in Alzheimer's disease pathogenesis. This study demonstrates that TREM2 signaling promotes microglial survival and metabolic fitness in the presence of Aβ plaques.",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"
                },
                {
                    "id": "23456789",
                    "title": "Mitochondrial Dysfunction in Alzheimer's Disease: A Target for Therapeutic Intervention",
                    "authors": ["Brown R", "Davis C", "Miller E"],
                    "journal": "Nature Neuroscience",
                    "publication_date": "2022 Nov 10",
                    "abstract": "This review discusses the role of mitochondrial dysfunction in Alzheimer's disease and highlights potential therapeutic strategies targeting mitochondrial dynamics and bioenergetics.",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/23456789/"
                }
            ]
        },
        "cancer": {
            "articles": [
                {
                    "id": "34567890",
                    "title": "ALDH as a Therapeutic Target in Cancer Stem Cells",
                    "authors": ["Zhang L", "Chen H", "Wang Y"],
                    "journal": "Cancer Research",
                    "publication_date": "2023 Mar 22",
                    "abstract": "This study identifies ALDH1A3 as a key metabolic enzyme in cancer stem cells that contributes to therapy resistance. Selective inhibition of ALDH1A3 reduced tumor growth and prevented recurrence in patient-derived xenograft models.",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/34567890/"
                }
            ]
        },
        "default": {
            "articles": [
                {
                    "id": "98765432",
                    "title": "Recent Advances in Biomedical Research",
                    "authors": ["Johnson R", "Smith T"],
                    "journal": "Science",
                    "publication_date": "2023 Feb 05",
                    "abstract": "This review summarizes recent advances in biomedical research across multiple disciplines.",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/98765432/"
                }
            ]
        }
    }
    
    # Create tool instance
    tool_config = {
        "responses": mock_responses
    }
    pubmed_tool = ToolRegistry.create_tool("mock_pubmed_search", tool_config)
    
    # Example 1: Search for Alzheimer's disease
    print("\n" + "="*50)
    print("Searching PubMed for Alzheimer's disease:")
    print("="*50)
    
    alzheimer_results = await pubmed_tool.run({"query": "TREM2 Alzheimer's disease"})
    
    print(f"\nFound {len(alzheimer_results.get('articles', []))} articles:")
    for article in alzheimer_results.get("articles", []):
        print(f"\nTitle: {article.get('title')}")
        print(f"Authors: {', '.join(article.get('authors', []))}")
        print(f"Journal: {article.get('journal')}")
        print(f"Date: {article.get('publication_date')}")
        print(f"Abstract: {article.get('abstract')[:150]}...")
        print(f"URL: {article.get('url')}")
    
    # Example 2: Search for cancer
    print("\n" + "="*50)
    print("Searching PubMed for cancer research:")
    print("="*50)
    
    cancer_results = await pubmed_tool.run({"query": "ALDH cancer stem cells"})
    
    print(f"\nFound {len(cancer_results.get('articles', []))} articles:")
    for article in cancer_results.get("articles", []):
        print(f"\nTitle: {article.get('title')}")
        print(f"Authors: {', '.join(article.get('authors', []))}")
        print(f"Journal: {article.get('journal')}")
        print(f"Date: {article.get('publication_date')}")
        print(f"Abstract: {article.get('abstract')[:150]}...")
        print(f"URL: {article.get('url')}")
    
    # Example 3: Using the tool with a generation agent
    print("\n" + "="*50)
    print("Using the tool with a generation agent:")
    print("="*50)
    
    # Create a model instance
    model_config = {
        "provider": "custom",
        "custom": {
            "model_name": "example-model",
            "responses": {
                "default": json.dumps({
                    "hypotheses": [
                        {
                            "title": "TREM2 Modulation for Microglial Phagocytosis Enhancement",
                            "description": "Based on the PubMed search results showing TREM2's role in microglial survival and energy metabolism, enhancing TREM2 signaling may promote clearance of amyloid-beta and reduce neuroinflammation in Alzheimer's disease.",
                            "mechanism": "TREM2 activation increases phagocytic activity in microglia, facilitating amyloid-beta clearance and reducing plaque formation.",
                            "testing": "Develop small molecule TREM2 agonists and test in APP/PS1 mouse models, measuring amyloid load and cognitive performance.",
                            "novelty": "While TREM2 is a known target, specific small molecule modulators are underdeveloped compared to antibody approaches.",
                            "impact": "Could provide an orally available treatment that addresses both amyloid clearance and neuroinflammation simultaneously."
                        }
                    ]
                })
            }
        }
    }
    
    # Register the custom model with the ModelFactory if not already registered
    try:
        from examples.custom_model_provider import CustomModel
        if "custom" not in ModelFactory.registered_models:
            ModelFactory.register_model("custom", CustomModel)
    except ImportError:
        # Define a simple custom model class if not already imported
        from core.models.base_model import BaseModel
        
        class CustomModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)
                self.responses = config.get("custom", {}).get("responses", {})
            
            async def generate_json(self, prompt, system_prompt=None, schema=None, temperature=0.7, max_tokens=None):
                return json.loads(self.responses.get("default", "{}"))
        
        ModelFactory.register_model("custom", CustomModel)
    
    model = ModelFactory.create_model(model_config)
    
    # Create generation agent with tool access
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "templates",
        "generation_agent.txt"
    )
    
    agent = GenerationAgent(
        model=model,
        prompt_template_path=template_path,
        num_hypotheses=1,
        tools={"pubmed_search": pubmed_tool}
    )
    
    # Run generation with tool access
    print("\nGenerating hypothesis based on literature search...")
    
    # In a real implementation, the agent would use the tool during generation
    # For this example, we'll simulate the process
    
    # 1. First, search for literature
    search_results = await pubmed_tool.run({"query": "TREM2 Alzheimer's disease"})
    print(f"\nFound {len(search_results.get('articles', []))} relevant articles")
    
    # 2. Generate hypothesis based on search results
    generation_result = await agent.run_with_timing({
        "research_goal": "Identify novel drug targets for Alzheimer's disease",
        "domain": "neuroscience",
        "constraints": "Focus on mechanisms that can be targeted with small molecules",
        "prior_knowledge": "Recent studies have implicated neuroinflammation in Alzheimer's progression",
        "literature_search_results": search_results
    })
    
    # 3. Display the generated hypothesis
    print("\nGenerated hypothesis based on literature search:")
    for hypothesis in generation_result.get("hypotheses", []):
        print(f"\nTitle: {hypothesis.get('title')}")
        print(f"Description: {hypothesis.get('description')}")
        print(f"Mechanism: {hypothesis.get('mechanism')}")
        print(f"Testing: {hypothesis.get('testing')}")
        print(f"Novelty: {hypothesis.get('novelty')}")
        print(f"Impact: {hypothesis.get('impact')}")

async def main():
    """Main entry point for the example."""
    await run_with_tool_integration()

if __name__ == "__main__":
    asyncio.run(main()) 