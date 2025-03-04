#!/usr/bin/env python3
"""
Example of implementing and using a custom model provider in the AI Co-Scientist system.

This example demonstrates how to:
1. Create a custom model adapter
2. Register it with the ModelFactory
3. Use it for hypothesis generation
"""

import asyncio
import os
import sys
import logging
import yaml
import json
from typing import Dict, Any, List, Optional

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.base_model import BaseModel
from core.models.model_factory import ModelFactory
from core.agents.generation_agent import GenerationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class CustomModel(BaseModel):
    """
    Example of a custom model adapter.
    
    This is a simple implementation that simulates an LLM by returning
    predefined responses for demonstration purposes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the custom model.
        
        Args:
            config (Dict[str, Any]): Configuration for the model
        """
        super().__init__(config)
        self.model_name = config.get("model_name", "custom-model")
        self.responses = config.get("responses", {})
        logger.info(f"Initialized CustomModel with name: {self.model_name}")
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt (str): The prompt to generate a response for
            system_prompt (Optional[str]): System prompt to guide the model
            temperature (float): Sampling temperature
            max_tokens (Optional[int]): Maximum number of tokens to generate
            
        Returns:
            str: The generated response
        """
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        
        # For demonstration, we'll return a predefined response based on keywords in the prompt
        for keyword, response in self.responses.items():
            if keyword.lower() in prompt.lower():
                logger.info(f"Found keyword '{keyword}' in prompt, returning predefined response")
                return response
        
        # Default response if no keyword matches
        logger.info("No keyword match found, returning default response")
        return self.responses.get("default", "No response available for this prompt.")
    
    async def generate_json(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        schema: Dict[str, Any] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a JSON response to the given prompt.
        
        Args:
            prompt (str): The prompt to generate a response for
            system_prompt (Optional[str]): System prompt to guide the model
            schema (Dict[str, Any]): JSON schema for the response
            temperature (float): Sampling temperature
            max_tokens (Optional[int]): Maximum number of tokens to generate
            
        Returns:
            Dict[str, Any]: The generated JSON response
        """
        logger.info(f"Generating JSON response for prompt: {prompt[:50]}...")
        
        # For demonstration, we'll return a predefined JSON response
        # In a real implementation, you would parse the schema and generate a valid response
        
        # Example hypothesis structure
        if "hypotheses" in prompt.lower() or "generate" in prompt.lower():
            return {
                "hypotheses": [
                    {
                        "title": "TREM2 Modulation for Microglial Phagocytosis Enhancement",
                        "description": "Enhancing TREM2 signaling in microglia may promote clearance of amyloid-beta and reduce neuroinflammation in Alzheimer's disease.",
                        "mechanism": "TREM2 activation increases phagocytic activity in microglia, facilitating amyloid-beta clearance and reducing plaque formation.",
                        "testing": "Develop small molecule TREM2 agonists and test in APP/PS1 mouse models, measuring amyloid load and cognitive performance.",
                        "novelty": "While TREM2 is a known target, specific small molecule modulators are underdeveloped compared to antibody approaches.",
                        "impact": "Could provide an orally available treatment that addresses both amyloid clearance and neuroinflammation simultaneously."
                    },
                    {
                        "title": "Mitochondrial Fission Inhibition for Neuronal Protection",
                        "description": "Inhibiting excessive mitochondrial fission may protect neurons from metabolic dysfunction in Alzheimer's disease.",
                        "mechanism": "Blocking Drp1-mediated mitochondrial fission prevents mitochondrial fragmentation, maintaining energy production and reducing oxidative stress.",
                        "testing": "Screen for novel Drp1 inhibitors and evaluate in 3D neuronal cultures and transgenic mouse models.",
                        "novelty": "Targeting mitochondrial dynamics represents an underexplored approach compared to traditional amyloid and tau strategies.",
                        "impact": "May provide neuroprotection independent of disease stage, potentially effective in both early and late Alzheimer's disease."
                    }
                ]
            }
        
        # Default JSON response
        return {"result": "No specific JSON response available for this prompt."}

# Register the custom model with the ModelFactory
ModelFactory.register_model("custom", CustomModel)

async def run_with_custom_model():
    """
    Run hypothesis generation using the custom model provider.
    """
    # Define custom model configuration
    custom_model_config = {
        "provider": "custom",
        "custom": {
            "model_name": "example-model",
            "responses": {
                "alzheimer": json.dumps({
                    "hypotheses": [
                        {
                            "title": "TREM2 Modulation for Microglial Phagocytosis Enhancement",
                            "description": "Enhancing TREM2 signaling in microglia may promote clearance of amyloid-beta and reduce neuroinflammation in Alzheimer's disease.",
                            "mechanism": "TREM2 activation increases phagocytic activity in microglia, facilitating amyloid-beta clearance and reducing plaque formation.",
                            "testing": "Develop small molecule TREM2 agonists and test in APP/PS1 mouse models, measuring amyloid load and cognitive performance.",
                            "novelty": "While TREM2 is a known target, specific small molecule modulators are underdeveloped compared to antibody approaches.",
                            "impact": "Could provide an orally available treatment that addresses both amyloid clearance and neuroinflammation simultaneously."
                        },
                        {
                            "title": "Mitochondrial Fission Inhibition for Neuronal Protection",
                            "description": "Inhibiting excessive mitochondrial fission may protect neurons from metabolic dysfunction in Alzheimer's disease.",
                            "mechanism": "Blocking Drp1-mediated mitochondrial fission prevents mitochondrial fragmentation, maintaining energy production and reducing oxidative stress.",
                            "testing": "Screen for novel Drp1 inhibitors and evaluate in 3D neuronal cultures and transgenic mouse models.",
                            "novelty": "Targeting mitochondrial dynamics represents an underexplored approach compared to traditional amyloid and tau strategies.",
                            "impact": "May provide neuroprotection independent of disease stage, potentially effective in both early and late Alzheimer's disease."
                        }
                    ]
                }),
                "cancer": json.dumps({
                    "hypotheses": [
                        {
                            "title": "Targeting Cancer Stem Cell Metabolism via ALDH Inhibition",
                            "description": "Inhibiting aldehyde dehydrogenase (ALDH) enzymes may selectively target cancer stem cells by disrupting their metabolic adaptations.",
                            "mechanism": "ALDH enzymes protect cancer stem cells from oxidative stress and enable metabolic plasticity, contributing to therapy resistance.",
                            "testing": "Develop selective ALDH1A3 inhibitors and test in patient-derived xenograft models, measuring tumor regression and recurrence rates.",
                            "novelty": "While ALDH is a known cancer stem cell marker, its metabolic functions remain underexploited as therapeutic targets.",
                            "impact": "Could address therapy resistance and recurrence by eliminating the cancer stem cell population responsible for tumor initiation."
                        }
                    ]
                }),
                "default": json.dumps({
                    "hypotheses": [
                        {
                            "title": "Default Hypothesis Example",
                            "description": "This is a default hypothesis provided when no specific keywords are matched.",
                            "mechanism": "Generic mechanism description for demonstration purposes.",
                            "testing": "Standard testing approach for demonstration purposes.",
                            "novelty": "This is a placeholder for demonstration of the custom model adapter.",
                            "impact": "Limited impact as this is just a demonstration example."
                        }
                    ]
                })
            }
        }
    }
    
    # Create model instance
    model = ModelFactory.create_model(custom_model_config)
    
    # Create generation agent
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "templates",
        "generation_agent.txt"
    )
    agent = GenerationAgent(
        model=model,
        prompt_template_path=template_path,
        num_hypotheses=2
    )
    
    # Run generation for Alzheimer's disease
    print("\n" + "="*50)
    print("Generating hypotheses for Alzheimer's disease:")
    print("="*50)
    
    alzheimer_result = await agent.run_with_timing({
        "research_goal": "Identify novel drug targets for Alzheimer's disease",
        "domain": "neuroscience",
        "constraints": "Focus on mechanisms that can be targeted with small molecules",
        "prior_knowledge": "Recent studies have implicated neuroinflammation in Alzheimer's progression"
    })
    
    for i, hypothesis in enumerate(alzheimer_result.get("hypotheses", []), 1):
        print(f"\nHypothesis {i}:")
        print(f"Title: {hypothesis.get('title')}")
        print(f"Description: {hypothesis.get('description')}")
        print("-"*50)
    
    # Run generation for cancer
    print("\n" + "="*50)
    print("Generating hypotheses for cancer research:")
    print("="*50)
    
    cancer_result = await agent.run_with_timing({
        "research_goal": "Identify novel approaches to target therapy-resistant cancer cells",
        "domain": "oncology",
        "constraints": "Focus on approaches that can be combined with existing therapies",
        "prior_knowledge": "Cancer stem cells contribute to therapy resistance and recurrence"
    })
    
    for i, hypothesis in enumerate(cancer_result.get("hypotheses", []), 1):
        print(f"\nHypothesis {i}:")
        print(f"Title: {hypothesis.get('title')}")
        print(f"Description: {hypothesis.get('description')}")
        print("-"*50)

async def main():
    """Main entry point for the example."""
    await run_with_custom_model()

if __name__ == "__main__":
    asyncio.run(main()) 