# AI Co-Scientist

A powerful AI system for scientific hypothesis generation, refinement, and prioritization through a collaborative, iterative process.

## Overview

AI Co-Scientist helps researchers generate and refine research hypotheses using AI. The system uses language models to create specialized agents that work together to explore research questions, evaluate hypotheses, and suggest experiments.

## Quick Start Guide

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lucaspecina/ai-co-scientist.git
   cd ai-co-scientist
   ```

2. Create and activate a Conda environment:
   ```bash
   conda create -n co_scientist python=3.11
   conda activate co_scientist
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

### Real-Time Monitoring

To visualize what the system is doing in real-time, you can use the built-in monitoring dashboard:

```bash
python monitor.py --session your_session_id
```

**What happens:** This launches a simple web-based dashboard that shows:
- Current system state and active agents
- Real-time logs of all operations
- Progress indicators for each workflow stage
- Live updates of hypothesis generation and refinement
- Resource usage statistics

The dashboard automatically refreshes and provides a visual representation of the entire workflow as it happens.

For a simpler terminal-based monitoring, you can use:

```bash
python monitor.py --session your_session_id --mode terminal
```

This will display a continuously updating log in your terminal with color-coded status updates.

### Basic Usage

The system works through a series of simple commands:

#### 1. Create a Research Session

```bash
python main.py create --goal "Your research question" --domain "Your domain"
```

Example:
```bash
python main.py create --goal "Develop a novel solution to the ARC-AGI benchmark by combining program synthesis with deep learning models" --domain "artificial intelligence"
```

**What happens:** The system initializes a new research session in the database, assigns it a unique ID, and stores your research goal and domain. This creates the foundation for all subsequent work.

#### 2. Run the Session

```bash
python main.py run --session your_session_id
```

Example:
```bash
python main.py run --session session_1741182035
```

**What happens:** The system activates multiple specialized AI agents that:
1. Generate initial hypotheses related to your research goal
2. Critically evaluate each hypothesis for scientific merit
3. Rank the hypotheses based on novelty, plausibility, and impact
4. Store all results in the session database

This process typically takes a few minutes as the agents work through multiple steps.

#### 3. View Top Hypotheses

Use our custom script to view the top hypotheses:

```bash
python get_top_hypotheses.py your_session_id
```

Example:
```bash
python get_top_hypotheses.py session_1741182035
```

**What happens:** The script connects to the database, retrieves the highest-ranked hypotheses for your session, and displays them in a readable format. Each hypothesis includes a title, description, and ranking score.

#### 4. Provide Feedback

You can provide feedback in two ways:

**Option 1: Direct feedback via command line**
```bash
python provide_feedback.py your_session_id "Your feedback text"
```

**Option 2: Feedback from a file (recommended for longer feedback)**
```bash
# First, create a feedback.txt file with your feedback
# Then run:
python provide_feedback_from_file.py your_session_id feedback.txt
```

**What happens:** Your feedback is stored in the session database and will guide the next iteration of hypothesis generation. The system will focus on aspects you found interesting and address concerns you raised.

#### 5. Check Session Status and Feedback

```bash
python get_session_info.py your_session_id
```

**What happens:** This script retrieves and displays comprehensive information about your session, including its current state (generating, evolving, completed, etc.), when it was created, and basic statistics.

```bash
python check_feedback.py your_session_id
```

**What happens:** This script retrieves and displays all feedback that has been provided for the session, showing you the history of your interactions with the system.

#### 6. Continue the Research Cycle

After providing feedback, you can run the session again to generate refined hypotheses:

```bash
python main.py run --session your_session_id
```

**What happens:** The system enters a new iteration cycle where it:
1. Analyzes your feedback to understand your interests and concerns
2. Refines existing hypotheses based on this feedback
3. Generates new hypotheses that better align with your research direction
4. Re-evaluates and re-ranks all hypotheses
5. Updates the session with these improved results

## How It Works: The Complete Workflow

The AI Co-Scientist follows this workflow behind the scenes:

1. **Session Creation**: 
   - Creates a database entry for your research session
   - Initializes the workflow state machine
   - Prepares the research context from your goal and domain

2. **Hypothesis Generation**: 
   - The Generation Agent creates 5-10 diverse hypotheses
   - Each hypothesis is structured with a title, description, and potential impact
   - The agent uses domain knowledge to ensure scientific validity

3. **Hypothesis Evaluation**: 
   - The Reflection Agent critically examines each hypothesis
   - It identifies strengths, weaknesses, and logical inconsistencies
   - It suggests potential improvements for each hypothesis

4. **External Evidence Collection**:
   - The Proximity Agent gathers supporting evidence
   - It identifies relevant scientific concepts and precedents
   - It grounds hypotheses in existing knowledge

5. **Hypothesis Evolution**:
   - The Evolution Agent refines hypotheses based on evaluations
   - It addresses identified weaknesses and incorporates evidence
   - It creates improved versions of promising hypotheses

6. **Ranking**: 
   - The Ranking Agent scores hypotheses on multiple criteria
   - Criteria include novelty, plausibility, testability, and impact
   - The top-scoring hypotheses are highlighted for your review

7. **Feedback Integration**: 
   - Your feedback is analyzed to extract key interests and concerns
   - The system adjusts its focus based on your guidance
   - This shapes the next iteration of the research cycle

8. **Iteration**: 
   - Steps 2-7 repeat, with each cycle building on previous work
   - Hypotheses become more refined and targeted
   - The system converges toward the most promising research directions

## System Components

The system consists of several key components:

- **Controller**: Orchestrates the workflow and manages sessions.
- **Specialized Agents**: Different agents handle generation, evaluation, ranking, etc.
- **Memory Manager**: Stores session data, hypotheses, and feedback.

## Example Workflow

Here's a complete example workflow:

1. Create a session:
   ```bash
   python main.py create --goal "Develop a novel solution to the ARC-AGI benchmark by combining program synthesis with deep learning models" --domain "artificial intelligence"
   ```

2. Run the session:
   ```bash
   python main.py run --session session_1741182035
   ```

3. View the top hypotheses:
   ```bash
   python get_top_hypotheses.py session_1741182035
   ```

4. Create a feedback.txt file with your thoughts:
   ```
   I'm particularly interested in the hypotheses that combine meta-learning techniques with program synthesis. The integration of model-agnostic meta-learning (MAML) with program synthesis seems especially promising.
   
   Please explore this direction further, focusing on:
   1. How MAML can be adapted specifically for program synthesis tasks
   2. Potential benchmarks for evaluating meta-learning approaches
   3. Ways to incorporate few-shot learning paradigms
   ```

5. Submit your feedback:
   ```bash
   python provide_feedback_from_file.py session_1741182035 feedback.txt
   ```

6. Run the session again to refine hypotheses:
   ```bash
   python main.py run --session session_1741182035
   ```

7. Check the updated hypotheses:
   ```bash
   python get_top_hypotheses.py session_1741182035
   ```

## Session Management

To help you manage your sessions effectively and prevent API overspending, we provide a comprehensive session management tool.

### Basic Session Management Commands

#### List All Sessions

View all your sessions with their status, token usage, and estimated costs:

```bash
python manage_sessions.py list
```

To filter by session state:

```bash
python manage_sessions.py list --filter active  # Show only active sessions
python manage_sessions.py list --filter stopped  # Show only stopped sessions
```

#### View Detailed Session Information

```bash
python manage_sessions.py details <session_id>
```

This provides comprehensive information including:
- Session metadata (goal, state, creation time)
- Token usage and estimated cost
- Breakdown of usage by agent and model
- Top hypotheses

#### Stop Sessions to Prevent Overspending

Stop a specific session:

```bash
python manage_sessions.py stop <session_id>
```

Stop all active sessions at once:

```bash
python manage_sessions.py stop-all
```

#### Delete Sessions

Delete a non-active session:

```bash
python manage_sessions.py delete <session_id>
```

Force deletion of an active session (use with caution):

```bash
python manage_sessions.py delete <session_id> --force
```

Delete all non-active sessions:

```bash
python manage_sessions.py delete-all
```

### Preventing API Overspending

To monitor and control your API usage:

1. **Regular monitoring**: Run `python manage_sessions.py list` to see all sessions and their token usage
2. **Stop unnecessary sessions**: Use `python manage_sessions.py stop-all` to stop all active sessions when not in use
3. **Clean up**: Use `python manage_sessions.py delete <session_id>` to remove completed or failed sessions you no longer need
4. **Check detailed costs**: Use `python manage_sessions.py details <session_id>` to see detailed token usage breakdown

For more detailed information about session management, see [SESSION_MANAGEMENT.md](SESSION_MANAGEMENT.md).

## Tips for Effective Use

1. **Be Specific**: Provide clear research goals and detailed feedback.
2. **Iterate**: The system improves with each feedback cycle.
3. **Combine Approaches**: Use the system to complement your own expertise.
4. **Review Critically**: Always evaluate the generated hypotheses with your scientific judgment.

## Troubleshooting

If you encounter issues with feedback submission in the console, use the file-based feedback method (`provide_feedback_from_file.py`), which avoids console buffer limitations.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 