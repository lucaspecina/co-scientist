"""
Supervisor Agent for AI Co-Scientist

This module implements the central workflow supervisor for coordinating the multi-agent
hypothesis generation, refinement, and selection process. It manages the iterative flow
between specialized agents, external tools, and scientist interactions.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Set, Callable, Tuple

from ..agents.base_agent import BaseAgent
from ..memory.memory_manager import MemoryManager
from ..models.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class WorkflowState(Enum):
    """Enumeration of workflow states."""
    INITIAL = "initial"
    GENERATING = "generating"
    REFLECTING = "reflecting"
    EVOLVING = "evolving"
    RANKING = "ranking"
    AWAITING_FEEDBACK = "awaiting_feedback"
    EXTERNAL_DATA = "external_data"
    META_REVIEW = "meta_review"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class Hypothesis:
    """Data class for research hypotheses."""
    id: str
    text: str
    score: float = 0.0
    rationale: str = ""
    critiques: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    references: List[Dict[str, Any]] = field(default_factory=list)
    experimental_approach: str = ""
    parent_id: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "rationale": self.rationale,
            "critiques": self.critiques,
            "evidence": self.evidence,
            "iteration": self.iteration,
            "scores": self.scores,
            "references": self.references,
            "experimental_approach": self.experimental_approach,
            "parent_id": self.parent_id,
            "generated_at": self.generated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Hypothesis':
        """Create a Hypothesis from a dictionary."""
        # Convert ISO format date back to datetime
        if isinstance(data.get("generated_at"), str):
            data["generated_at"] = datetime.fromisoformat(data["generated_at"])
        return cls(**data)


@dataclass
class ResearchGoal:
    """Data class for research goals."""
    id: str
    description: str
    domain: str
    constraints: List[str] = field(default_factory=list)
    background: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "description": self.description,
            "domain": self.domain,
            "constraints": self.constraints,
            "background": self.background,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchGoal':
        """Create a ResearchGoal from a dictionary."""
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class WorkflowSession:
    """Data class for workflow sessions."""
    id: str
    goal: ResearchGoal
    hypotheses: List[Hypothesis] = field(default_factory=list)
    iterations_completed: int = 0
    max_iterations: int = 5
    state: WorkflowState = WorkflowState.INITIAL
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    top_hypotheses: List[str] = field(default_factory=list)  # IDs of top hypotheses
    tool_usage: Dict[str, int] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "goal": self.goal.to_dict(),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "iterations_completed": self.iterations_completed,
            "max_iterations": self.max_iterations,
            "state": self.state.value,
            "feedback_history": self.feedback_history,
            "top_hypotheses": self.top_hypotheses,
            "tool_usage": self.tool_usage,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class SupervisorAgent:
    """
    Supervisor Agent coordinates the multi-agent workflow for hypothesis generation,
    refinement, and selection.
    """
    
    def __init__(self, 
                config: Dict[str, Any],
                memory_manager: MemoryManager,
                model_factory: ModelFactory = None,
                agent_registry: Dict[str, BaseAgent] = None):
        """
        Initialize the supervisor agent.
        
        Args:
            config: Configuration dictionary
            memory_manager: Memory manager for persistent storage
            model_factory: Factory for creating model instances
            agent_registry: Dictionary of registered agents
        """
        self.config = config
        self.memory_manager = memory_manager
        self.model_factory = model_factory or ModelFactory()
        self.agent_registry = agent_registry or {}
        
        # Load workflow configuration
        self.workflow_config = config.get("workflow", {})
        self.max_iterations = self.workflow_config.get("max_iterations", 5)
        self.tournament_size = self.workflow_config.get("tournament_size", 8)
        self.top_hypotheses_count = self.workflow_config.get("top_hypotheses_count", 3)
        
        # Active session tracking
        self.active_sessions = {}
        self.session_callbacks = {}
        
        logger.info("Supervisor agent initialized with max_iterations=%d, tournament_size=%d", 
                   self.max_iterations, self.tournament_size)
    
    async def create_session(self, 
                           goal_description: str, 
                           domain: str, 
                           background: str = "",
                           constraints: List[str] = None) -> str:
        """
        Create a new workflow session for a research goal.
        
        Args:
            goal_description: Description of the research goal
            domain: Scientific domain of the research
            background: Background information (optional)
            constraints: List of constraints (optional)
            
        Returns:
            Session ID
        """
        # Create a research goal
        goal_id = f"goal_{int(time.time())}"
        goal = ResearchGoal(
            id=goal_id,
            description=goal_description,
            domain=domain,
            background=background,
            constraints=constraints or []
        )
        
        # Create a workflow session
        session_id = f"session_{int(time.time())}"
        session = WorkflowSession(
            id=session_id,
            goal=goal,
            max_iterations=self.max_iterations
        )
        
        # Save to memory
        await self.memory_manager.store_session(session)
        self.active_sessions[session_id] = session
        
        logger.info(f"Created new session {session_id} for goal: {goal_description}")
        return session_id
    
    async def run_session(self, 
                        session_id: str, 
                        on_state_change: Optional[Callable] = None) -> WorkflowSession:
        """
        Run a workflow session from start to finish.
        
        Args:
            session_id: ID of the session to run
            on_state_change: Callback function for state changes
            
        Returns:
            Completed workflow session
        """
        if session_id not in self.active_sessions:
            session = await self.memory_manager.load_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            self.active_sessions[session_id] = session
        else:
            session = self.active_sessions[session_id]
        
        # Register callback if provided
        if on_state_change:
            self.session_callbacks[session_id] = on_state_change
        
        # Set initial state if new session
        if session.state == WorkflowState.INITIAL:
            await self._update_session_state(session, WorkflowState.GENERATING)
        
        # Run the workflow until completed or max iterations reached
        try:
            while session.state != WorkflowState.COMPLETED and session.state != WorkflowState.ERROR:
                if session.state == WorkflowState.GENERATING:
                    await self._run_generation_phase(session)
                    
                elif session.state == WorkflowState.REFLECTING:
                    await self._run_reflection_phase(session)
                    
                elif session.state == WorkflowState.EVOLVING:
                    await self._run_evolution_phase(session)
                    
                elif session.state == WorkflowState.RANKING:
                    await self._run_ranking_phase(session)
                    
                elif session.state == WorkflowState.EXTERNAL_DATA:
                    await self._run_external_data_phase(session)
                    
                elif session.state == WorkflowState.META_REVIEW:
                    await self._run_meta_review_phase(session)
                    
                elif session.state == WorkflowState.AWAITING_FEEDBACK:
                    # This state requires user input, so we pause here
                    logger.info(f"Session {session_id} is awaiting feedback")
                    break
                
                # Save session state after each phase
                await self.memory_manager.store_session(session)
                
            # Successfully completed
            if session.state == WorkflowState.COMPLETED:
                session.completed_at = datetime.now()
                await self.memory_manager.store_session(session)
                logger.info(f"Session {session_id} completed successfully")
                
            return session
            
        except Exception as e:
            logger.error(f"Error running session {session_id}: {str(e)}")
            await self._update_session_state(session, WorkflowState.ERROR)
            await self.memory_manager.store_session(session)
            raise
    
    async def add_feedback(self, 
                         session_id: str, 
                         feedback: str,
                         target_hypothesis_ids: List[str] = None) -> str:
        """
        Add scientist feedback to a session.
        
        Args:
            session_id: Session ID
            feedback: Feedback text
            target_hypothesis_ids: List of hypothesis IDs the feedback applies to
            
        Returns:
            Status message
        """
        session = await self._get_session(session_id)
        
        # Record the feedback
        feedback_entry = {
            "text": feedback,
            "timestamp": datetime.now().isoformat(),
            "target_hypotheses": target_hypothesis_ids,
            "iteration": session.iterations_completed
        }
        session.feedback_history.append(feedback_entry)
        
        # If we're awaiting feedback, process it and continue
        if session.state == WorkflowState.AWAITING_FEEDBACK:
            await self._update_session_state(session, WorkflowState.EVOLVING)
            
        # Save the updated session
        await self.memory_manager.store_session(session)
        
        return f"Feedback added to session {session_id}"
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get the current status of a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session status dictionary
        """
        session = await self._get_session(session_id)
        
        # Prepare a status summary
        status = {
            "id": session.id,
            "goal": session.goal.description,
            "state": session.state.value,
            "iterations_completed": session.iterations_completed,
            "max_iterations": session.max_iterations,
            "hypothesis_count": len(session.hypotheses),
            "started_at": session.started_at.isoformat(),
            "age_hours": (datetime.now() - session.started_at).total_seconds() / 3600,
            "top_hypotheses": []
        }
        
        # Add top hypotheses if available
        if session.top_hypotheses:
            for hyp_id in session.top_hypotheses:
                hyp = next((h for h in session.hypotheses if h.id == hyp_id), None)
                if hyp:
                    status["top_hypotheses"].append({
                        "id": hyp.id,
                        "text": hyp.text,
                        "score": hyp.score
                    })
        
        return status
    
    async def get_hypotheses(self, 
                           session_id: str, 
                           limit: int = 10, 
                           include_all: bool = False) -> List[Dict[str, Any]]:
        """
        Get hypotheses from a session.
        
        Args:
            session_id: Session ID
            limit: Maximum number of hypotheses to return
            include_all: Whether to include all hypotheses or just the latest iteration
            
        Returns:
            List of hypothesis dictionaries
        """
        session = await self._get_session(session_id)
        
        if include_all:
            hypotheses = session.hypotheses
        else:
            # Get the latest iteration
            latest_iter = max(h.iteration for h in session.hypotheses) if session.hypotheses else 0
            hypotheses = [h for h in session.hypotheses if h.iteration == latest_iter]
        
        # Sort by score (descending)
        hypotheses = sorted(hypotheses, key=lambda h: h.score, reverse=True)
        
        # Limit the number of results
        hypotheses = hypotheses[:limit]
        
        return [h.to_dict() for h in hypotheses]
    
    async def _get_session(self, session_id: str) -> WorkflowSession:
        """
        Get a session, either from active sessions or from memory.
        
        Args:
            session_id: Session ID
            
        Returns:
            Workflow session
            
        Raises:
            ValueError: If the session is not found
        """
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
            
        session = await self.memory_manager.load_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
            
        self.active_sessions[session_id] = session
        return session
    
    async def _update_session_state(self, 
                                  session: WorkflowSession, 
                                  new_state: WorkflowState) -> None:
        """
        Update a session's state and trigger callbacks.
        
        Args:
            session: Workflow session
            new_state: New state
        """
        old_state = session.state
        session.state = new_state
        
        logger.info(f"Session {session.id} state changed: {old_state.value} -> {new_state.value}")
        
        # Call the state change callback if registered
        if session.id in self.session_callbacks:
            try:
                callback = self.session_callbacks[session.id]
                await callback(session, old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {str(e)}")
    
    async def _run_generation_phase(self, session: WorkflowSession) -> None:
        """
        Run the hypothesis generation phase.
        
        Args:
            session: Workflow session
        """
        logger.info(f"Running generation phase for session {session.id}")
        
        # Get the generation agent
        generation_agent = self._get_agent("generation")
        if not generation_agent:
            raise ValueError("Generation agent not found in registry")
        
        # Generate initial hypotheses
        generation_count = self.workflow_config.get("generation_count", 10)
        
        # Prepare context for generation
        context = {
            "goal": session.goal.to_dict(),
            "iteration": session.iterations_completed,
            "feedback": session.feedback_history
        }
        
        # Generate hypotheses
        try:
            response = await generation_agent.execute(context, {"count": generation_count})
            
            # Process generated hypotheses
            for idx, hyp_data in enumerate(response.get("hypotheses", [])):
                hypothesis = Hypothesis(
                    id=f"hyp_{int(time.time())}_{idx}",
                    text=hyp_data.get("text", ""),
                    rationale=hyp_data.get("rationale", ""),
                    iteration=session.iterations_completed
                )
                session.hypotheses.append(hypothesis)
            
            # Move to the reflection phase
            await self._update_session_state(session, WorkflowState.REFLECTING)
            
        except Exception as e:
            logger.error(f"Error in generation phase: {str(e)}")
            await self._update_session_state(session, WorkflowState.ERROR)
    
    async def _run_reflection_phase(self, session: WorkflowSession) -> None:
        """
        Run the hypothesis reflection (critique) phase.
        
        Args:
            session: Workflow session
        """
        logger.info(f"Running reflection phase for session {session.id}")
        
        # Get the reflection agent
        reflection_agent = self._get_agent("reflection")
        if not reflection_agent:
            raise ValueError("Reflection agent not found in registry")
        
        # Get hypotheses from the current iteration
        current_iter = session.iterations_completed
        hypotheses = [h for h in session.hypotheses if h.iteration == current_iter]
        
        # Prepare tasks for parallel processing
        tasks = []
        for hypothesis in hypotheses:
            context = {
                "goal": session.goal.to_dict(),
                "hypothesis": hypothesis.to_dict(),
                "iteration": current_iter,
                "feedback": session.feedback_history
            }
            tasks.append(reflection_agent.execute(context, {}))
        
        # Process critiques in parallel
        try:
            results = await asyncio.gather(*tasks)
            
            # Update hypotheses with critiques
            for idx, hypothesis in enumerate(hypotheses):
                critiques = results[idx].get("critiques", [])
                hypothesis.critiques = critiques
            
            # Move to the external data phase to gather evidence
            await self._update_session_state(session, WorkflowState.EXTERNAL_DATA)
            
        except Exception as e:
            logger.error(f"Error in reflection phase: {str(e)}")
            await self._update_session_state(session, WorkflowState.ERROR)
    
    async def _run_external_data_phase(self, session: WorkflowSession) -> None:
        """
        Run the external data gathering phase.
        
        Args:
            session: Workflow session
        """
        logger.info(f"Running external data phase for session {session.id}")
        
        # Get the proximity agent for literature search
        proximity_agent = self._get_agent("proximity")
        if not proximity_agent:
            raise ValueError("Proximity agent not found in registry")
        
        # Get hypotheses from the current iteration
        current_iter = session.iterations_completed
        hypotheses = [h for h in session.hypotheses if h.iteration == current_iter]
        
        # Prepare tasks for parallel processing
        tasks = []
        for hypothesis in hypotheses:
            context = {
                "goal": session.goal.to_dict(),
                "hypothesis": hypothesis.to_dict(),
                "iteration": current_iter
            }
            tasks.append(proximity_agent.execute(context, {}))
        
        # Process evidence in parallel
        try:
            results = await asyncio.gather(*tasks)
            
            # Update hypotheses with evidence
            for idx, hypothesis in enumerate(hypotheses):
                hypothesis.evidence = results[idx].get("evidence", [])
                hypothesis.references = results[idx].get("references", [])
            
            # Move to the evolution phase
            await self._update_session_state(session, WorkflowState.EVOLVING)
            
        except Exception as e:
            logger.error(f"Error in external data phase: {str(e)}")
            await self._update_session_state(session, WorkflowState.ERROR)
    
    async def _run_evolution_phase(self, session: WorkflowSession) -> None:
        """
        Run the hypothesis evolution phase.
        
        Args:
            session: Workflow session
        """
        logger.info(f"Running evolution phase for session {session.id}")
        
        # Get the evolution agent
        evolution_agent = self._get_agent("evolution")
        if not evolution_agent:
            raise ValueError("Evolution agent not found in registry")
        
        # Get hypotheses from the current iteration
        current_iter = session.iterations_completed
        hypotheses = [h for h in session.hypotheses if h.iteration == current_iter]
        
        # Prepare tasks for evolution
        evolved_hypotheses = []
        for hypothesis in hypotheses:
            context = {
                "goal": session.goal.to_dict(),
                "hypothesis": hypothesis.to_dict(),
                "critiques": hypothesis.critiques,
                "evidence": hypothesis.evidence,
                "feedback": session.feedback_history,
                "iteration": current_iter
            }
            
            # Execute evolution
            try:
                response = await evolution_agent.execute(context, {})
                
                # Process evolved hypotheses
                evolutions = response.get("evolved_hypotheses", [])
                for idx, evolution in enumerate(evolutions):
                    evolved_hyp = Hypothesis(
                        id=f"hyp_{int(time.time())}_{hypothesis.id}_{idx}",
                        text=evolution.get("text", ""),
                        rationale=evolution.get("rationale", ""),
                        iteration=current_iter + 1,
                        parent_id=hypothesis.id
                    )
                    evolved_hypotheses.append(evolved_hyp)
                    
            except Exception as e:
                logger.error(f"Error evolving hypothesis {hypothesis.id}: {str(e)}")
        
        # Add evolved hypotheses to the session
        session.hypotheses.extend(evolved_hypotheses)
        
        # Move to the ranking phase
        await self._update_session_state(session, WorkflowState.RANKING)
    
    async def _run_ranking_phase(self, session: WorkflowSession) -> None:
        """
        Run the hypothesis ranking phase.
        
        Args:
            session: Workflow session
        """
        logger.info(f"Running ranking phase for session {session.id}")
        
        # Get the ranking agent
        ranking_agent = self._get_agent("ranking")
        if not ranking_agent:
            raise ValueError("Ranking agent not found in registry")
        
        # Get hypotheses from the current iteration
        current_iter = session.iterations_completed + 1  # Use the evolved hypotheses
        hypotheses = [h for h in session.hypotheses if h.iteration == current_iter]
        
        if not hypotheses:
            logger.warning(f"No hypotheses found for iteration {current_iter}")
            await self._update_session_state(session, WorkflowState.ERROR)
            return
        
        # Prepare context for ranking
        hypothesis_dicts = [h.to_dict() for h in hypotheses]
        context = {
            "goal": session.goal.to_dict(),
            "hypotheses": hypothesis_dicts,
            "iteration": current_iter,
            "feedback": session.feedback_history
        }
        
        # Execute ranking
        try:
            response = await ranking_agent.execute(context, {})
            
            # Update hypothesis scores
            ranked_hypotheses = response.get("ranked_hypotheses", [])
            logger.info(f"Received {len(ranked_hypotheses)} ranked hypotheses from the ranking agent")
            
            for ranked in ranked_hypotheses:
                hyp_id = ranked.get("id")
                if hyp_id:
                    hypothesis = next((h for h in hypotheses if h.id == hyp_id), None)
                    if hypothesis:
                        # Check for both possible field names (score and overall_score)
                        if "score" in ranked:
                            score_value = float(ranked.get("score", 0.0))
                            hypothesis.score = score_value
                            logger.info(f"Updated hypothesis {hyp_id} with score {score_value}")
                        else:
                            score_value = float(ranked.get("overall_score", 0.0))
                            hypothesis.score = score_value
                            logger.info(f"Updated hypothesis {hyp_id} with overall_score {score_value}")
                        
                        # Update criteria scores
                        hypothesis.scores = ranked.get("criteria_scores", {})
            
            # Sort hypotheses by score
            hypotheses.sort(key=lambda h: h.score, reverse=True)
            
            # Update top hypotheses
            session.top_hypotheses = [h.id for h in hypotheses[:self.top_hypotheses_count]]
            
            # Increment iteration counter
            session.iterations_completed += 1
            
            # Check if we've reached the maximum iterations
            if session.iterations_completed >= session.max_iterations:
                await self._update_session_state(session, WorkflowState.META_REVIEW)
            else:
                # Otherwise, ask for user feedback
                await self._update_session_state(session, WorkflowState.AWAITING_FEEDBACK)
            
        except Exception as e:
            logger.error(f"Error in ranking phase: {str(e)}")
            await self._update_session_state(session, WorkflowState.ERROR)
    
    async def _run_meta_review_phase(self, session: WorkflowSession) -> None:
        """
        Run the meta-review phase to generate final output.
        
        Args:
            session: Workflow session
        """
        logger.info(f"Running meta-review phase for session {session.id}")
        
        # Get the meta-review agent
        meta_review_agent = self._get_agent("meta_review")
        if not meta_review_agent:
            raise ValueError("Meta-review agent not found in registry")
        
        # Get the top hypotheses
        top_hyp_ids = session.top_hypotheses
        top_hypotheses = [h for h in session.hypotheses if h.id in top_hyp_ids]
        
        if not top_hypotheses:
            logger.warning("No top hypotheses found for meta-review")
            await self._update_session_state(session, WorkflowState.ERROR)
            return
        
        # Prepare context for meta-review
        hypothesis_dicts = [h.to_dict() for h in top_hypotheses]
        context = {
            "goal": session.goal.to_dict(),
            "hypotheses": hypothesis_dicts,
            "iterations": session.iterations_completed,
            "feedback": session.feedback_history
        }
        
        # Execute meta-review
        try:
            response = await meta_review_agent.execute(context, {})
            
            # Update top hypotheses with experimental approaches
            experimental_approaches = response.get("experimental_approaches", [])
            for approach in experimental_approaches:
                hyp_id = approach.get("hypothesis_id")
                if hyp_id:
                    hypothesis = next((h for h in top_hypotheses if h.id == hyp_id), None)
                    if hypothesis:
                        hypothesis.experimental_approach = approach.get("approach", "")
            
            # Mark session as completed
            await self._update_session_state(session, WorkflowState.COMPLETED)
            
        except Exception as e:
            logger.error(f"Error in meta-review phase: {str(e)}")
            await self._update_session_state(session, WorkflowState.ERROR)
    
    def _get_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """
        Get an agent from the registry.
        
        Args:
            agent_type: Type of agent to get
            
        Returns:
            Agent instance or None if not found
        """
        if agent_type not in self.agent_registry:
            logger.warning(f"Agent type {agent_type} not found in registry")
            return None
            
        return self.agent_registry[agent_type] 