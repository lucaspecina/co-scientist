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

## Getting Started with AI Co-Scientist: A Step-by-Step Guide

This guide will walk you through setting up and using the AI Co-Scientist system from scratch, explaining the interaction model and workflow in detail.

### Installation and Setup

#### Step 1: System Requirements

Ensure your system meets these requirements:
- Python 3.9 or higher
- 8GB+ RAM (16GB+ recommended for local model usage)
- MongoDB (for persistent storage)
- Redis (for task queue management)

#### Step 2: Clone the Repository

```bash
git clone https://github.com/lucaspecina/ai-co-scientist.git
cd ai-co-scientist
```

#### Step 3: Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 5: Set Up Environment Variables

Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
PUBMED_EMAIL=your_email@example.com
```

### Basic Configuration

#### Step 6: Select a Configuration File

The system comes with several pre-configured setups:

- `config/default_config.yaml`: Balanced configuration using OpenAI models
- `config/ollama_config.yaml`: Configuration for local model usage with Ollama
- `config/scientific_research_config.yaml`: Enhanced settings for intensive scientific research

For first-time users, we recommend starting with the default configuration.

### Starting the System

#### Step 7: Start Supporting Services

Start MongoDB and Redis if they're not running:

```bash
# Start MongoDB (adjust path as needed)
mongod --dbpath /path/to/data/db

# Start Redis
redis-server
```

#### Step 8: Start the AI Co-Scientist System

```bash
python main.py start
```

You should see output indicating the system has started successfully:

```
AI Co-Scientist system starting...
Loading configuration from config/default_config.yaml
Initializing model factory...
Initializing memory manager...
Initializing agent factory...
System ready!
```

### Research Workflow

#### Step 9: Create a New Research Session

```bash
python main.py create --goal "Identify molecular mechanisms that link chronic inflammation to Alzheimer's disease progression" --domain "neuroscience" --background "Recent studies suggest neuroinflammation plays a key role in neurodegeneration, but the exact mechanisms remain unclear."
```

The system will respond with a session ID:

```
Creating new research session...
Session created successfully!
Session ID: 4f7d9e2a-8c16-4c31-b5a3-57b8f90d2e4c
```

#### Step 10: Run the Research Session

```bash
python main.py run --session 4f7d9e2a-8c16-4c31-b5a3-57b8f90d2e4c
```

The system will display status updates as it progresses through the workflow phases:

```
Running session 4f7d9e2a-8c16-4c31-b5a3-57b8f90d2e4c...
Status: GENERATING [Iteration 1/4] - Generating initial hypotheses
Status: REFLECTING - Evaluating hypotheses
Status: EXTERNAL_DATA - Gathering evidence from literature
Status: EVOLVING - Refining hypotheses based on critiques and evidence
Status: RANKING - Scoring and prioritizing hypotheses
Status: AWAITING_FEEDBACK - Awaiting researcher input
```

## Detailed Workflow Explanation

Let's walk through what happens in each phase of the workflow:

### 1. Hypothesis Generation Phase

**System Actions:**
- The GenerationAgent receives your research goal, domain, and background information.
- It generates 5-8 diverse, scientifically grounded hypotheses.
- Each hypothesis includes a title, description, proposed mechanism, and potential significance.

**Example Output:**
```
Hypothesis 1: "Microglial TREM2 Dysfunction in Alzheimer's Disease"
Description: Chronic inflammation disrupts TREM2 signaling in microglia, impairing amyloid clearance and promoting neurodegeneration.
...

Hypothesis 2: "NF-κB-Mediated Synaptic Dysfunction"
Description: Persistent activation of NF-κB signaling pathways by inflammatory cytokines leads to synaptic dysfunction and cognitive decline.
...
```

### 2. Reflection Phase

**System Actions:**
- The ReflectionAgent critically evaluates each hypothesis.
- It assesses plausibility, novelty, logical coherence, and testability.
- For each hypothesis, it provides specific strengths, weaknesses, and improvement suggestions.

**Example Output:**
```
Evaluation of Hypothesis 1:
Summary: A promising hypothesis with strong grounding in recent literature.

Strengths:
- Connects established TREM2 variants with inflammatory processes
- Proposes a testable molecular mechanism

Weaknesses:
- Does not address how chronic inflammation initiates TREM2 dysfunction
- Lacks specificity regarding which inflammatory mediators are involved
- Correlation between TREM2 and amyloid clearance is established, but causation needs clarification

Suggestions:
- Specify which inflammatory cytokines might affect TREM2 function
- Consider incorporating the role of genetic variants
...
```

### 3. External Data Collection Phase

**System Actions:**
- The ProximityAgent generates search queries based on each hypothesis.
- It searches scientific databases (PubMed, arXiv, etc.) for relevant papers.
- Each paper is evaluated for relevance and categorized as supporting, contradicting, or contextual.
- A relevance summary is generated for each hypothesis.

**Example Output:**
```
Evidence for Hypothesis 1:
Relevant Papers: 12 found (8 supporting, 2 contradicting, 2 contextual)

Top Supporting Evidence:
1. "TREM2 variants and Alzheimer's disease" (2019) - Shows TREM2 dysfunction impairs microglial response to amyloid-β. Relevance: 9/10
2. "Inflammatory triggers in neurodegeneration" (2021) - Demonstrates IL-1β reduces TREM2 expression in microglia. Relevance: 8/10

Contradicting Evidence:
1. "Microglial activation in early Alzheimer's disease" (2020) - Suggests TREM2 upregulation, not dysfunction, in early disease. Relevance: 7/10

Relevance Summary:
Strong evidence supports the role of TREM2 in microglial function and amyloid clearance. Recent studies have linked inflammatory cytokines (particularly IL-1β and TNF-α) to altered TREM2 expression. However, the temporal relationship between inflammation and TREM2 dysfunction remains unclear, with some evidence suggesting TREM2 upregulation precedes dysfunction.
```

### 4. Evolution Phase

**System Actions:**
- The EvolutionAgent refines each hypothesis based on critiques and evidence.
- It addresses specific weaknesses identified in the reflection phase.
- It incorporates relevant findings from the literature.
- Multiple refined versions of each promising hypothesis are generated.

**Example Output:**
```
Refined Hypothesis 1.2: "IL-1β-Mediated TREM2 Dysfunction in Microglial Amyloid Clearance"

Description: Chronic elevation of IL-1β in Alzheimer's disease downregulates microglial TREM2 expression through NF-κB-dependent mechanisms, impairing amyloid-β phagocytosis and promoting plaque accumulation. This dysfunction is exacerbated in carriers of TREM2 variants (R47H, R62H) and establishes a feed-forward inflammatory cycle that accelerates neurodegeneration.

Improvements:
- Specifies IL-1β as the key inflammatory mediator based on literature evidence
- Incorporates genetic component (TREM2 variants) as suggested
- Proposes a specific molecular pathway (NF-κB-dependent regulation)
- Addresses the feed-forward nature of the inflammatory response

Supporting Evidence:
- Incorporates findings from Zhang et al. (2021) on IL-1β regulation of TREM2
- References Suarez-Calvet et al. (2019) work on TREM2 variants

Testability:
- Can be examined in transgenic mouse models with IL-1β modulation
- Amenable to in vitro studies with human iPSC-derived microglia
```

### 5. Ranking Phase

**System Actions:**
- The RankingAgent evaluates all hypotheses against multiple criteria.
- Each hypothesis receives scores for novelty, plausibility, testability, impact, and specificity.
- A weighted ranking is produced based on the importance of each criterion.
- The top hypotheses are identified and explanations provided for their ranking.

**Example Output:**
```
Ranking Results:

1. Hypothesis 1.2: "IL-1β-Mediated TREM2 Dysfunction in Microglial Amyloid Clearance"
   Total Score: 8.4/10
   Novelty: 7/10 - Builds on known TREM2 research but proposes novel regulatory mechanism
   Plausibility: 9/10 - Strong literature support and coherent molecular pathway
   Testability: 9/10 - Clear experimental approaches available
   Impact: 8/10 - Could inform both diagnostic and therapeutic approaches
   Specificity: 8/10 - Well-defined molecular mechanism

2. Hypothesis 3.3: "Astrocytic A1/A2 Polarization in Neurodegeneration"
   Total Score: 7.9/10
   [Detailed scores and explanation]

3. Hypothesis 2.1: "Microbiome-Induced Systemic Inflammation in Neurodegeneration"
   Total Score: 7.6/10
   [Detailed scores and explanation]
   
Complementary Approaches:
Hypotheses 1.2 and 3.3 could be productively investigated in parallel, as they examine different cell types (microglia vs. astrocytes) in the same inflammatory context.
```

### 6. Awaiting Feedback

**System Actions:**
- The system pauses and waits for your feedback.
- You can review the generated hypotheses, rankings, and supporting evidence.
- You can provide specific guidance to refine the research direction.

#### Step 11: Provide Feedback to Guide the Research

```bash
python main.py feedback --session 4f7d9e2a-8c16-4c31-b5a3-57b8f90d2e4c --text "I'm interested in the TREM2 hypothesis but would like to explore the potential role of peripheral immune cells crossing the blood-brain barrier. Can we examine how systemic inflammation might contribute to microglial dysfunction?"
```

The system acknowledges your feedback:

```
Feedback added successfully to session 4f7d9e2a-8c16-4c31-b5a3-57b8f90d2e4c
Session status updated to: GENERATING [Iteration 2/4]
```

### 7. Iterative Refinement

**System Actions:**
- The workflow repeats, incorporating your feedback.
- New hypotheses may be generated, or existing ones refined further.
- Each iteration builds on previous findings and your guidance.
- The process becomes increasingly focused on promising directions.

#### Step 12: Continue Running the Session

```bash
python main.py run --session 4f7d9e2a-8c16-4c31-b5a3-57b8f90d2e4c
```

The system continues through iterations, showing status updates:

```
Running session 4f7d9e2a-8c16-4c31-b5a3-57b8f90d2e4c...
Status: GENERATING [Iteration 2/4] - Generating hypotheses incorporating feedback
Status: REFLECTING - Evaluating hypotheses
...
```

### 8. Meta-Review Phase

**System Actions:**
- After the final iteration, the MetaReviewAgent synthesizes all findings.
- It identifies common themes and patterns across hypotheses.
- It proposes a cohesive narrative connecting the most promising ideas.
- Detailed experimental approaches are suggested for testing the top hypotheses.

**Example Output:**
```
Meta-Review Summary:

Research Journey:
This investigation began with a broad exploration of molecular mechanisms linking chronic inflammation to Alzheimer's disease. Through four iterations, the research focused increasingly on microglial dysfunction, with particular emphasis on TREM2 signaling and peripheral immune infiltration.

Key Themes:
1. Microglial phenotype shifting in response to inflammatory stimuli
2. Peripheral immune cell contribution to neuroinflammation
3. Cytokine-specific effects on phagocytic function
4. Blood-brain barrier integrity in disease progression

Experimental Approaches for Top Hypothesis:
"Peripheral Immune Cell-Mediated Microglial TREM2 Dysfunction"

Primary Experiment:
- Methodology: Chimeric mouse model with fluorescently-labeled peripheral monocytes to track infiltration into brain parenchyma in APP/PS1 mice
- Treatment groups: Systemic LPS vs. control
- Measurements: 
  * Infiltration of peripheral immune cells (flow cytometry)
  * Microglial TREM2 expression (immunohistochemistry, qPCR)
  * Amyloid clearance capacity (ex vivo phagocytosis assay)
  * Cognitive function (Morris water maze)
- Controls: Inclusion of TREM2 knockout mice; CCR2 deficient mice (impaired monocyte recruitment)
- Resources needed: Transgenic mouse lines, flow cytometry, tissue culture facilities, behavioral testing equipment
- Timeline: Approximately 12 months

Pilot Study:
- Co-culture of peripheral monocytes with primary microglia to assess direct cellular interactions
- Measure TREM2 expression changes following co-culture
- Expected duration: 3 months

Alternative Approach:
- Human studies examining CSF markers of peripheral immune cell infiltration correlated with TREM2 levels and cognitive measures
- Advantage: Direct human relevance
- Limitation: Correlative rather than causative

Ethical Considerations:
All animal experiments should conform to institutional guidelines for minimizing suffering. The number of animals should be calculated to provide statistical power while minimizing usage.
```

#### Step 13: Check Session Status and View Results

```bash
python main.py status --session 4f7d9e2a-8c16-4c31-b5a3-57b8f90d2e4c
```

```
Session Status: COMPLETED
Iterations Completed: 4/4
Top Hypotheses: 3
Last Updated: 2024-03-04 15:23:45
```

```bash
python main.py hypotheses --session 4f7d9e2a-8c16-4c31-b5a3-57b8f90d2e4c --limit 3
```

The system displays the top hypotheses with detailed information and experimental suggestions.

## Common Interaction Patterns

### Exploratory Research Mode

1. Start with a broad research goal
2. Allow the system to generate diverse hypotheses
3. Provide feedback to explore interesting directions
4. Iterate until promising areas emerge
5. Request detailed experimental designs for the most interesting hypotheses

### Targeted Investigation Mode

1. Start with a specific research question
2. Set constraints to focus the generation process
3. Provide detailed background information
4. Guide the system toward specific hypotheses refinement
5. Focus on experimental design and validation approaches

### Collaborative Review Mode

1. Input existing hypotheses for evaluation
2. Have the system critique and refine them
3. Request literature evidence for or against each hypothesis
4. Use the ranking to prioritize future research efforts
5. Generate experimental protocols for validation

## Tips for Effective Use

1. **Be Specific:** The more specific and detailed your research goal and background information, the more focused and relevant the generated hypotheses will be.

2. **Iterate Actively:** Don't expect perfect results from the first iteration. The system improves with your feedback and guidance.

3. **Provide Constructive Feedback:** Specific feedback like "Focus more on mechanism X" or "Consider the role of pathway Y" is more helpful than general comments.

4. **Review Evidence Carefully:** The system provides literature evidence, but you should verify key papers independently.

5. **Adjust Configuration:** Different research questions may benefit from different configurations. Try the scientific_research_config.yaml for deeper exploration of complex problems.

6. **Combine with Human Expertise:** The system is designed to augment, not replace, human scientific reasoning. The best results come from collaborative refinement.

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

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/lucaspecina/ai-co-scientist.git
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
