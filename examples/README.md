# AI Co-Scientist Examples

This directory contains example use cases and configurations for the AI Co-Scientist system.

## Examples

### Basic Usage

```python
import asyncio
from core.models.model_factory import ModelFactory
from core.agents.generation_agent import GenerationAgent

async def generate_hypotheses():
    # Create a model instance
    model_config = {
        "provider": "openai",
        "openai": {
            "api_key": "your-api-key",
            "model_name": "gpt-4-turbo"
        }
    }
    model = ModelFactory.create_model(model_config)
    
    # Create a generation agent
    agent = GenerationAgent(
        model=model,
        prompt_template_path="config/templates/generation_agent.txt",
        num_hypotheses=3
    )
    
    # Run generation
    result = await agent.run_with_timing({
        "research_goal": "Identify novel drug targets for Alzheimer's disease",
        "domain": "neuroscience"
    })
    
    # Print results
    for hypothesis in result.get("hypotheses", []):
        print(f"Title: {hypothesis.get('title')}")
        print(f"Description: {hypothesis.get('description')}")
        print()

# Run the example
asyncio.run(generate_hypotheses())
```

### Switching Model Providers

The model-agnostic design makes it easy to switch between different LLM providers:

```python
# Using OpenAI
model_config = {
    "provider": "openai",
    "openai": {
        "api_key": "your-openai-key",
        "model_name": "gpt-4-turbo"
    }
}

# Using Google Gemini
model_config = {
    "provider": "google",
    "google": {
        "api_key": "your-google-key",
        "model_name": "gemini-pro"
    }
}

# Using Anthropic Claude
model_config = {
    "provider": "anthropic",
    "anthropic": {
        "api_key": "your-anthropic-key",
        "model_name": "claude-3-opus-20240229"
    }
}

# Using a local model
model_config = {
    "provider": "local",
    "local": {
        "model_path": "models/local_model",
        "device": "cuda"
    }
}
```

### Running a Full Session

```python
import asyncio
import yaml
from core.memory.memory_manager import create_memory_manager
from framework.queue.task_queue import create_task_queue
from framework.supervisor.supervisor import Supervisor

async def run_session():
    # Load configuration
    with open("config/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    memory_manager = create_memory_manager(config)
    task_queue = create_task_queue(config)
    
    # Initialize agents (simplified)
    agents = {}  # In a real example, you would initialize all agents
    
    # Initialize supervisor
    supervisor = Supervisor.from_config(config, agents, memory_manager, task_queue)
    
    # Start session
    session_id = await supervisor.start_session(
        research_goal="Identify novel drug targets for Alzheimer's disease",
        domain="neuroscience"
    )
    
    print(f"Started session {session_id}")

# Run the example
asyncio.run(run_session())
```

## Additional Examples

More examples will be added in the future, including:

- Tool integration examples
- Custom agent implementations
- Advanced configuration scenarios
- Web interface integration 