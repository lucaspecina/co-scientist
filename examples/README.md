# AI Co-Scientist Examples

This directory contains example scripts demonstrating how to use the AI Co-Scientist system in various scenarios. Each example is designed to showcase different aspects of the system's capabilities.

## Overview of Examples

1. **basic_generation.py**: Demonstrates the core hypothesis generation functionality
2. **full_session.py**: Shows a complete research session workflow from start to finish
3. **ollama_example.py**: Illustrates how to use local models via Ollama
4. **tool_integration.py**: Explains how to integrate external scientific tools
5. **custom_model_provider.py**: Shows how to create a custom model provider
6. **api_server.py**: Demonstrates how to set up an API server for the system

## Detailed Examples

### Basic Hypothesis Generation

The `basic_generation.py` example demonstrates the core functionality of generating scientific hypotheses using the AI Co-Scientist system. This is the simplest entry point to understand how the system works.

```python
from core.models.model_factory import ModelFactory
from core.agents.generation_agent import GenerationAgent

# Initialize the model factory
model_factory = ModelFactory()

# Create a generation agent
generation_agent = GenerationAgent(
    model_factory=model_factory,
    model_provider="openai",
    model_name="gpt-4"
)

# Define a research goal
research_goal = "Identify molecular mechanisms that link chronic inflammation to Alzheimer's disease progression"
domain = "neuroscience"
background = "Recent studies suggest neuroinflammation plays a key role in neurodegeneration, but the exact mechanisms remain unclear."

# Generate hypotheses
hypotheses = generation_agent.generate_hypotheses(
    goal=research_goal,
    domain=domain,
    background=background,
    count=5
)

# Print the generated hypotheses
for i, hypothesis in enumerate(hypotheses, 1):
    print(f"Hypothesis {i}: {hypothesis.title}")
    print(f"Description: {hypothesis.description}")
    print(f"Mechanism: {hypothesis.mechanism}")
    print(f"Significance: {hypothesis.significance}")
    print("-" * 50)
```

**Key Concepts:**
- Creating a model factory to manage language model instances
- Initializing a specialized agent (GenerationAgent)
- Providing research context (goal, domain, background)
- Generating multiple hypotheses with a single call
- Working with structured hypothesis objects

### Full Research Session

The `full_session.py` example demonstrates a complete research session workflow, including all phases from hypothesis generation to meta-review. This example shows how the different components of the system work together.

```python
import asyncio
from core.controller import CoScientistController
from core.models.model_factory import ModelFactory
from core.memory.memory_manager import MemoryManager
from core.agents.agent_factory import AgentFactory

async def run_full_session():
    # Initialize components
    model_factory = ModelFactory()
    memory_manager = MemoryManager(connection_string="mongodb://localhost:27017/", database_name="ai_coscientist")
    agent_factory = AgentFactory(model_factory=model_factory)
    
    # Initialize the controller
    controller = CoScientistController(
        model_factory=model_factory,
        memory_manager=memory_manager,
        agent_factory=agent_factory,
        config_path="config/default_config.yaml"
    )
    
    # Start the system
    await controller.startup()
    
    # Create a new research session
    session_id = await controller.create_session(
        goal="Identify novel drug targets for treating treatment-resistant depression",
        domain="neuroscience",
        background="Current antidepressants primarily target monoamine systems, but 30-40% of patients do not respond adequately. Recent research suggests inflammation, glutamate signaling, and neuroplasticity pathways may offer new therapeutic avenues.",
        constraints="Focus on targets that could be druggable with small molecules and have potential for clinical translation within 5-10 years."
    )
    
    # Run the session
    await controller.run_session(
        session_id=session_id,
        wait_for_completion=True,
        status_callback=lambda status: print(f"Session status: {status}")
    )
    
    # Get the top hypotheses
    hypotheses = await controller.get_hypotheses(
        session_id=session_id,
        limit=3,
        include_experiments=True
    )
    
    # Print the results
    print("\nTop Hypotheses:")
    for i, hypothesis in enumerate(hypotheses, 1):
        print(f"{i}. {hypothesis.title}")
        print(f"   Score: {hypothesis.score}/10")
        print(f"   Description: {hypothesis.description[:100]}...")
        print(f"   Proposed Experiment: {hypothesis.experiment_design[:100]}...")
        print()
    
    # Shut down the system
    await controller.shutdown()

if __name__ == "__main__":
    asyncio.run(run_full_session())
```

**Key Concepts:**
- Initializing the complete system with all components
- Creating and running a research session
- Asynchronous workflow management
- Retrieving and displaying results
- Proper system startup and shutdown

### Using Local Models with Ollama

The `ollama_example.py` example demonstrates how to use the AI Co-Scientist system with local models via Ollama, which is useful for privacy-sensitive research or when working offline.

```python
from core.models.model_factory import ModelFactory
from core.agents.generation_agent import GenerationAgent
from core.config.config_loader import load_config

# Load the Ollama configuration
config = load_config("config/ollama_config.yaml")

# Initialize the model factory with Ollama configuration
model_factory = ModelFactory(config=config["models"])

# Create a generation agent using Ollama
generation_agent = GenerationAgent(
    model_factory=model_factory,
    model_provider="ollama",
    model_name="llama3"  # or any other model you have pulled in Ollama
)

# Define a research goal
research_goal = "Identify potential biomarkers for early detection of pancreatic cancer"
domain = "oncology"
background = "Pancreatic cancer is often diagnosed at late stages, leading to poor prognosis. Early detection biomarkers could significantly improve survival rates."

# Generate hypotheses
hypotheses = generation_agent.generate_hypotheses(
    goal=research_goal,
    domain=domain,
    background=background,
    count=3
)

# Print the generated hypotheses
for i, hypothesis in enumerate(hypotheses, 1):
    print(f"Hypothesis {i}: {hypothesis.title}")
    print(f"Description: {hypothesis.description}")
    print("-" * 50)
```

**Key Concepts:**
- Loading a specialized configuration for local models
- Connecting to Ollama for local model inference
- Adjusting parameters for potentially less capable local models
- Maintaining the same API regardless of model provider

### External Tool Integration

The `tool_integration.py` example demonstrates how to integrate external scientific tools and data sources with the AI Co-Scientist system, enhancing its capabilities with domain-specific functionality.

```python
from core.tools.literature_search import LiteratureSearchTool
from core.tools.data_visualization import DataVisualizationTool
from core.agents.proximity_agent import ProximityAgent
from core.models.model_factory import ModelFactory

# Initialize the model factory
model_factory = ModelFactory()

# Initialize scientific tools
literature_tool = LiteratureSearchTool(
    email="researcher@example.com",
    api_key="your_pubmed_api_key"  # Optional
)

visualization_tool = DataVisualizationTool()

# Create a proximity agent with tools
proximity_agent = ProximityAgent(
    model_factory=model_factory,
    model_provider="openai",
    model_name="gpt-4",
    tools=[literature_tool, visualization_tool]
)

# Define a hypothesis to gather evidence for
hypothesis = {
    "title": "BDNF-TrkB Signaling Deficits in Treatment-Resistant Depression",
    "description": "Reduced BDNF-TrkB signaling in the prefrontal cortex and hippocampus contributes to treatment resistance in depression by impairing neuroplasticity and cellular resilience."
}

# Gather evidence for the hypothesis
evidence_results = proximity_agent.gather_evidence(
    hypothesis=hypothesis,
    domain="neuroscience",
    max_papers=10
)

# Print the evidence summary
print("Evidence Summary:")
print(evidence_results["relevance_summary"])
print("\nSupporting Papers:")
for paper in evidence_results["supporting_papers"][:3]:
    print(f"- {paper['title']} ({paper['year']}) - Relevance: {paper['relevance_score']}/10")
print("\nContradicting Papers:")
for paper in evidence_results["contradicting_papers"][:3]:
    print(f"- {paper['title']} ({paper['year']}) - Relevance: {paper['relevance_score']}/10")
```

**Key Concepts:**
- Initializing specialized scientific tools
- Integrating tools with appropriate agents
- Gathering evidence for specific hypotheses
- Processing and presenting scientific literature results
- Working with structured evidence data

### Custom Model Provider

The `custom_model_provider.py` example demonstrates how to create a custom model provider to integrate with specialized or proprietary language models not natively supported by the system.

```python
from core.models.base_model import BaseModel
from core.models.model_factory import ModelFactory
import requests

# Define a custom model adapter
class CustomModelAdapter(BaseModel):
    def __init__(self, api_url, api_key, model_name):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
    
    async def generate(self, prompt, temperature=0.7, max_tokens=1000):
        # Make a request to your custom API
        response = requests.post(
            f"{self.api_url}/generate",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "model": self.model_name
            }
        )
        
        if response.status_code == 200:
            return response.json()["text"]
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")

# Register the custom model provider
ModelFactory.register_model_provider("custom_provider", CustomModelAdapter)

# Use the custom model provider
model_factory = ModelFactory()
model = model_factory.create_model(
    provider="custom_provider",
    model_name="my_specialized_model",
    api_url="https://api.example.com",
    api_key="your_api_key"
)

# Generate text with the custom model
result = await model.generate(
    prompt="Propose a hypothesis for how epigenetic modifications might influence treatment response in depression",
    temperature=0.8,
    max_tokens=500
)

print(result)
```

**Key Concepts:**
- Creating a custom model adapter by implementing the BaseModel interface
- Registering a new model provider with the ModelFactory
- Passing custom parameters to the model adapter
- Maintaining consistent API across different model providers
- Error handling for API interactions

### API Server

The `api_server.py` example demonstrates how to set up an API server for the AI Co-Scientist system, allowing it to be accessed via HTTP requests from web applications or other services.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import uvicorn
from core.controller import CoScientistController
from core.models.model_factory import ModelFactory
from core.memory.memory_manager import MemoryManager
from core.agents.agent_factory import AgentFactory

# Initialize FastAPI app
app = FastAPI(title="AI Co-Scientist API")

# Initialize AI Co-Scientist components
model_factory = ModelFactory()
memory_manager = MemoryManager(connection_string="mongodb://localhost:27017/", database_name="ai_coscientist")
agent_factory = AgentFactory(model_factory=model_factory)
controller = CoScientistController(
    model_factory=model_factory,
    memory_manager=memory_manager,
    agent_factory=agent_factory,
    config_path="config/default_config.yaml"
)

# Define request and response models
class SessionRequest(BaseModel):
    goal: str
    domain: str
    background: str
    constraints: str = None

class FeedbackRequest(BaseModel):
    session_id: str
    feedback: str

class SessionResponse(BaseModel):
    session_id: str
    status: str

# API routes
@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    session_id = await controller.create_session(
        goal=request.goal,
        domain=request.domain,
        background=request.background,
        constraints=request.constraints
    )
    
    # Start the session asynchronously
    asyncio.create_task(controller.run_session(session_id=session_id))
    
    status = await controller.get_session_status(session_id)
    return {"session_id": session_id, "status": status}

@app.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    status = await controller.get_session_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": status}

@app.get("/sessions/{session_id}/hypotheses")
async def get_hypotheses(session_id: str, limit: int = 10):
    hypotheses = await controller.get_hypotheses(
        session_id=session_id,
        limit=limit,
        include_experiments=True
    )
    if not hypotheses:
        raise HTTPException(status_code=404, detail="Session not found or no hypotheses available")
    return {"hypotheses": hypotheses}

@app.post("/sessions/{session_id}/feedback")
async def add_feedback(session_id: str, request: FeedbackRequest):
    success = await controller.add_feedback(
        session_id=session_id,
        feedback=request.feedback
    )
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "Feedback added successfully"}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    await controller.startup()

@app.on_event("shutdown")
async def shutdown_event():
    await controller.shutdown()

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Key Concepts:**
- Setting up a FastAPI server for the AI Co-Scientist system
- Defining API endpoints for session management and hypothesis retrieval
- Handling asynchronous operations in an API context
- Proper initialization and cleanup of system resources
- Structured request and response models

## Switching Model Providers

The AI Co-Scientist system supports multiple model providers. Here's how to switch between them:

### OpenAI

```python
agent = GenerationAgent(
    model_factory=model_factory,
    model_provider="openai",
    model_name="gpt-4"
)
```

### Google

```python
agent = GenerationAgent(
    model_factory=model_factory,
    model_provider="google",
    model_name="gemini-pro"
)
```

### Anthropic

```python
agent = GenerationAgent(
    model_factory=model_factory,
    model_provider="anthropic",
    model_name="claude-3-opus"
)
```

### Local Models (Ollama)

```python
agent = GenerationAgent(
    model_factory=model_factory,
    model_provider="ollama",
    model_name="llama3"  # or any other model you have pulled in Ollama
)
```

## Integration with the Overall System

These examples demonstrate individual components of the AI Co-Scientist system, but in a full deployment, they work together in a coordinated workflow:

1. The **Controller** orchestrates the entire process, managing sessions and coordinating agents
2. **Generation Agents** create initial hypotheses based on research goals
3. **Reflection Agents** critically evaluate each hypothesis
4. **Proximity Agents** gather evidence from scientific literature using integrated tools
5. **Evolution Agents** refine hypotheses based on critiques and evidence
6. **Ranking Agents** score and prioritize hypotheses
7. **Meta-Review Agents** synthesize findings and suggest experimental approaches

The `full_session.py` example demonstrates this complete workflow, while the other examples focus on specific aspects of the system that can be customized or extended.

## Best Practices

When using these examples as a starting point for your own implementation:

1. **Start Simple**: Begin with the basic_generation.py example to understand the core functionality
2. **Configure Appropriately**: Adjust model parameters based on your specific research domain
3. **Manage Resources**: Be mindful of API usage and costs when using commercial model providers
4. **Validate Results**: Always critically evaluate the generated hypotheses and suggested experiments
5. **Iterate**: Use the feedback mechanism to refine hypotheses over multiple iterations
6. **Extend Thoughtfully**: When adding new tools or model providers, follow the established patterns

## Next Steps

After exploring these examples, you might want to:

1. Create custom agents for specialized scientific domains
2. Integrate additional scientific databases or tools
3. Implement domain-specific evaluation criteria
4. Develop a web interface using the API server example as a foundation
5. Contribute improvements back to the AI Co-Scientist project 