"""
Example script demonstrating how to use the Ollama model adapter with AI Co-Scientist.

This example shows how to:
1. Initialize the Ollama model
2. Generate a response
3. Generate JSON output
4. Create embeddings

Prerequisites:
- Ollama installed and running (see https://ollama.ai)
- At least one model pulled (e.g., `ollama pull llama3`)

Usage:
    python ollama_example.py
"""

import asyncio
import json
import logging
import sys

# Add the project root to the Python path
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.ollama_model import OllamaModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run the example code for Ollama model."""
    
    logger.info("Initializing Ollama model...")
    
    # Initialize the model - modify these parameters as needed
    model = OllamaModel(
        model_name="llama3",  # Change to any model you've pulled
        api_base="http://localhost:11434",
        temperature=0.7,
        max_tokens=2048,
        timeout=60
    )
    
    # Example 1: Basic text generation
    prompt = "Explain the concept of 'emergence' in complex systems in 3 paragraphs"
    logger.info(f"Generating text response for: {prompt}")
    
    response = await model.generate(prompt)
    logger.info("Generated response:")
    print(f"\n{response}\n")
    
    # Example 2: JSON generation
    json_prompt = "Generate a JSON object with 3 example research hypotheses about artificial intelligence"
    schema = {
        "type": "object",
        "properties": {
            "hypotheses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "testability": {"type": "number", "minimum": 0, "maximum": 10}
                    },
                    "required": ["title", "description", "testability"]
                }
            }
        },
        "required": ["hypotheses"]
    }
    
    logger.info(f"Generating JSON response for: {json_prompt}")
    
    try:
        json_response = await model.generate_json(json_prompt, schema=schema)
        logger.info("Generated JSON:")
        print(json.dumps(json_response, indent=2))
        print()
    except Exception as e:
        logger.error(f"Error generating JSON: {e}")
    
    # Example 3: Generating embeddings
    text_for_embedding = "Artificial intelligence and machine learning"
    logger.info(f"Generating embeddings for: {text_for_embedding}")
    
    try:
        embeddings = await model.embed(text_for_embedding)
        logger.info(f"Generated embeddings of dimension {len(embeddings)}")
        print(f"First 5 embedding values: {embeddings[:5]}")
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 