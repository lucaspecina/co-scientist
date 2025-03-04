# AI Co-Scientist

An AI-powered system to assist scientists in generating novel, plausible, and testable research hypotheses through a multi-agent collaboration framework.

## Overview

AI Co-Scientist is a model-agnostic platform that uses a "generate, debate, and evolve" methodology to help scientists formulate and refine research hypotheses. The system employs multiple specialized AI agents that work together asynchronously to generate, critique, rank, and evolve hypotheses.

## Key Features

- **Model-agnostic architecture**: Supports multiple LLM backends (e.g., Gemini, Claude, GPT-4, local models, Ollama)
- **Multi-agent system**: Specialized agents for hypothesis generation, reflection, ranking, evolution, proximity analysis, and meta-review
- **Asynchronous task execution**: Parallel processing of tasks with dynamic resource allocation
- **Tool integration**: Connects to external services (web search, AlphaFold, biomedical databases)
- **Tournament-based evolution**: Iteratively refines hypotheses through competition and improvement
- **Scientist-in-the-loop**: Natural language interface for collaborative hypothesis development

## Project Structure

```
co-scientist/
├── config/                 # Configuration files for agents and models
├── core/                   # Core system components
│   ├── models/             # Model adapters for different LLMs
│   ├── agents/             # Agent implementations
│   ├── memory/             # Persistent context storage
│   └── tools/              # External tool integrations
├── framework/              # Asynchronous execution framework
│   ├── supervisor/         # Task coordination and resource management
│   └── queue/              # Task queue implementation
├── ui/                     # User interface components
│   └── api/                # API endpoints for frontend communication
├── examples/               # Example use cases and configurations
└── tests/                  # Unit and integration tests
```

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/co-scientist.git
   cd co-scientist
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

The system can be configured through the `config/default_config.yaml` file. You can also create custom configuration files for different scenarios.

### Running with Different LLM Backends

#### Using OpenAI
```bash
# Set API key
export OPENAI_API_KEY=your_openai_api_key
# Run with default configuration (uses OpenAI by default)
python main.py
```

#### Using Anthropic Claude
```bash
# Set API key
export ANTHROPIC_API_KEY=your_anthropic_api_key
# Run with Claude as the default provider
python main.py --config config/custom_claude_config.yaml
```

#### Using Ollama (Local LLMs)

1. Install Ollama by following the instructions at [ollama.ai](https://ollama.ai)

2. Start the Ollama server:
   ```bash
   ollama serve
   ```

3. Pull the model you want to use (e.g., Llama 3):
   ```bash
   ollama pull llama3
   ```

4. Update your configuration to use Ollama:
   ```yaml
   # In config/ollama_config.yaml
   models:
     default_provider: "ollama"
     
     ollama:
       model_name: "llama3"  # or any other model you've pulled
       api_base: "http://localhost:11434"
       max_tokens: 2048
       temperature: 0.7
   ```

5. Run the AI Co-Scientist with Ollama:
   ```bash
   python main.py --config config/ollama_config.yaml
   ```

### Using Custom Models

You can also create custom configuration files to use different models for different agents:

```yaml
# Example: Using Ollama for generation and OpenAI for reflection
agents:
  generation:
    model_provider: "ollama"
  
  reflection:
    model_provider: "openai"
```

## Documentation

[Detailed documentation will go here]

## Contributing

[Contribution guidelines will go here]

## License

[License information will go here]
