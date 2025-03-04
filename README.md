# AI Co-Scientist

A powerful AI system for scientific hypothesis generation, refinement, and prioritization through a collaborative, iterative process.

## Overview

AI Co-Scientist is designed to assist scientists in generating, refining, and prioritizing research hypotheses. It combines a natural language interface with a multi-agent architecture to facilitate a collaborative workflow that enhances the scientific discovery process.

The system uses a state-of-the-art language model (like GPT-4 or Gemini) to drive an ensemble of specialized agents, each with a specific role in the hypothesis generation and refinement process. Through multiple iterations and feedback integration, AI Co-Scientist helps scientists explore complex problems efficiently and thoroughly.

## Key Features

- **Natural Language Interface**: Interact with the system using plain language to define research goals and provide feedback.
- **Multi-Agent Architecture**: Specialized agents work together to generate, critique, refine, and rank hypotheses.
- **Iterative Refinement**: Hypotheses improve over multiple cycles, incorporating feedback and new evidence.
- **External Knowledge Integration**: System connects with scientific literature and databases to ground hypotheses in existing research.
- **Customizable Workflow**: Adjust the system's behavior through configuration to match your research style and domain.
- **Local Execution**: Run the entire system on your local machine using Ollama or other local model providers.

## Detailed Capabilities

### Knowledge Base & Search
- **Comprehensive Knowledge Repository**: Access to scientific literature through PubMed, arXiv, and Semantic Scholar
- **Semantic Search**: Find relevant papers and evidence based on natural language queries
- **Citation Management**: Track and format references in various citation styles
- **Evidence Grounding**: Link hypotheses to supporting literature with relevance scoring

### Hypothesis Generation & Evaluation
- **Novel Hypothesis Creation**: Generate diverse, scientifically grounded hypotheses based on research goals
- **Plausibility Scoring**: Evaluate hypotheses against existing knowledge and logical coherence
- **Novelty Assessment**: Measure how innovative each hypothesis is compared to existing research
- **Hypothesis Evolution**: Refine promising ideas through multiple iterations

### Experiment Design
- **Detailed Protocols**: Receive specific experimental designs to test hypotheses
- **Multiple Approaches**: Get alternative experimental strategies for the same hypothesis
- **Testability Assessment**: Ensure hypotheses can be tested with available technology
- **Resource Estimation**: Understand the time and equipment needed for proposed experiments

### Feedback & Learning
- **Iterative Improvement**: System learns from your feedback to generate better hypotheses
- **Result Integration**: Incorporate experimental results to refine future hypotheses
- **Customizable Constraints**: Set specific limitations or preferences for hypothesis generation
- **Context Awareness**: Maintain research continuity across multiple sessions

### Explanation & Transparency
- **Reasoning Justification**: Understand why a hypothesis was suggested or ranked highly
- **Uncertainty Indicators**: Identify potential weaknesses or gaps in hypotheses
- **Decision Traceability**: Follow the system's reasoning process through detailed logs

### Cross-Domain Flexibility
- **Domain Adaptation**: System adjusts to different scientific fields through configuration
- **Interdisciplinary Research**: Support for combining insights across multiple domains
- **Specialized Knowledge**: Access to domain-specific tools and databases when available

## System Architecture

The AI Co-Scientist system consists of several key components:

### Core Components

- **Controller**: Central orchestration system that manages the workflow and coordinates agents.
- **Model Factory**: Creates and manages language model instances for various tasks.
- **Agent Factory**: Creates specialized agent instances based on configuration.
- **Memory Manager**: Handles persistent storage of sessions, hypotheses, and results.
- **Supervisor**: Manages the scientific workflow and iterations.

### Specialized Agents

- **Generation Agent**: Creates initial research hypotheses based on the research goal.
- **Reflection Agent**: Critically evaluates hypotheses, identifying strengths and weaknesses.
- **Evolution Agent**: Refines hypotheses based on critiques and feedback.
- **Ranking Agent**: Scores and prioritizes hypotheses using multiple scientific criteria.
- **Proximity Agent**: Gathers evidence from scientific literature and external sources.
- **Meta-Review Agent**: Synthesizes findings and suggests experimental approaches.

### Complete Workflow Process

1. **Research Goal Specification**: Define your scientific question or research goal, including background and domain.
2. **Hypothesis Generation**: The system creates an initial set of diverse hypotheses based on your goal.
3. **Critical Evaluation**: Each hypothesis undergoes rigorous evaluation for scientific merit and logic.
4. **External Evidence Collection**: The system searches for supporting or contradicting evidence in the literature.
5. **Hypothesis Refinement**: Hypotheses are refined to address critiques and incorporate evidence.
6. **Multi-Criteria Ranking**: Hypotheses are scored on plausibility, novelty, testability, and impact.
7. **Experiment Design**: For promising hypotheses, experimental protocols are suggested.
8. **Feedback Integration**: Your feedback guides future iterations of the process.
9. **Iterative Improvement**: Steps 2-8 repeat for multiple cycles, improving hypotheses each time.
10. **Final Output**: The system delivers a set of top hypotheses with experimental suggestions.

## Interaction Model

The AI Co-Scientist is designed to function as a collaborative partner in the scientific process:

- **Conversational Interface**: Ask questions, provide feedback, and steer the research in natural language
- **Context Awareness**: The system maintains the thread of your research across sessions
- **Bidirectional Learning**: The system adapts to your feedback while providing new insights
- **Flexible Engagement**: Use for quick hypothesis checks or extended research campaigns
- **Collaborative Features**: Optionally share sessions with team members for collaborative research

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-co-scientist.git
   cd ai-co-scientist
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your API keys (for OpenAI, Anthropic, etc.) by creating a `config.json` file or using environment variables.

## Usage

### Command Line Interface

AI Co-Scientist provides a simple command-line interface:

```bash
# Start the system
python main.py start

# Create a new research session
python main.py create --goal "Identify novel molecular pathways involved in Alzheimer's disease" --domain "neuroscience"

# Run a session
python main.py run --session <session_id>

# Add feedback to a session
python main.py feedback --session <session_id> --text "Focus more on inflammatory pathways and their role in neurodegeneration"

# View session status
python main.py status --session <session_id>

# List all sessions
python main.py list

# View system information
python main.py info
```

### Configuration

You can customize the system's behavior by creating a configuration file (JSON or YAML) specifying:

- Model providers and parameters
- Agent-specific configurations
- Workflow settings
- Memory and storage options

Example configuration:

```json
{
  "version": "1.0.0",
  "models": {
    "default": {
      "provider": "openai",
      "model_name": "gpt-4"
    }
  },
  "agents": {
    "generation": {
      "model": {"provider": "openai", "model_name": "gpt-4"},
      "generation_count": 5,
      "creativity": 0.7
    },
    "reflection": {
      "model": {"provider": "openai", "model_name": "gpt-4"},
      "detail_level": "medium"
    }
  },
  "workflow": {
    "max_iterations": 3,
    "top_hypotheses_count": 3
  }
}
```

### Running with Local Models (Ollama)

To run the system with Ollama:

1. Install and set up Ollama from [ollama.ai](https://ollama.ai)
2. Pull your preferred model:
   ```bash
   ollama pull llama3
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```
4. Run AI Co-Scientist with Ollama configuration:
   ```bash
   python main.py --config config/ollama_config.yaml start
   ```

## Practical Example: Drug Repurposing for Alzheimer's

Here's how a complete workflow might look in practice:

1. **Goal Setting**: Scientist inputs "Identify existing FDA-approved drugs that might be repurposed for Alzheimer's disease treatment"

2. **Hypothesis Generation**: System generates multiple hypotheses:
   - "Calcium channel blockers may reduce amyloid plaque formation"
   - "Anti-inflammatory drugs could target neuroinflammation in Alzheimer's"
   - "Diabetes medications might improve brain insulin signaling"

3. **Evaluation**: System evaluates each hypothesis, noting that the anti-inflammatory approach has strong literature support

4. **Evidence Collection**: System finds 12 papers supporting the anti-inflammatory hypothesis, ranking them by relevance

5. **Refinement**: Based on literature, hypothesis evolves to "COX-2 inhibitors may reduce neuroinflammation and slow Alzheimer's progression"

6. **Experiment Design**: System suggests testing celecoxib in APP/PS1 mouse models, with specific dosing and measurement protocols

7. **Feedback**: Scientist notes resource constraints, requests in vitro alternatives

8. **Adaptation**: System suggests testing celecoxib on microglial cell cultures with Aβ challenge

9. **Final Output**: Delivers ranked hypotheses, experimental protocols, and comprehensive literature citations

## Extending the System

### Adding New Model Providers

Implement a new model adapter by:

1. Creating a new file in `core/models/` that implements the `BaseModel` interface
2. Registering the adapter in `MODEL_PROVIDER_MAP` in `core/models/model_factory.py`

### Adding New Agents

Implement a new specialized agent by:

1. Creating a new agent class in `core/agents/` that extends `BaseAgent`
2. Registering the agent in `AgentFactory._agent_registry`

### Adding External Tools

Enhance the system with new data sources or tools by:

1. Implementing a new tool in `core/tools/`
2. Integrating it with the relevant agent (typically the `ProximityAgent`)

## Project Structure

```
ai-co-scientist/
├── core/
│   ├── agents/              # Specialized agents
│   ├── models/              # Model adapters
│   ├── tools/               # External tools integration
│   ├── workflow/            # Workflow management
│   ├── memory/              # Persistence layer
│   └── controller.py        # Main controller
├── config/                  # Configuration files
├── data/                    # Data storage
├── examples/                # Example scripts
├── tests/                   # Test suite
├── main.py                  # CLI entrypoint
└── README.md                # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project builds on research in AI-assisted scientific discovery.
- Thanks to the open-source AI community for their tools and models.
