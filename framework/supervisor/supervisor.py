"""
Supervisor for AI Co-Scientist

This module defines the Supervisor that coordinates the multi-agent workflow
and manages the asynchronous execution of tasks.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Callable

from core.agents.base_agent import BaseAgent
from core.memory.memory_manager import MemoryManager
from framework.queue.task_queue import TaskQueue

logger = logging.getLogger(__name__)


class Supervisor:
    """
    Supervisor coordinates the multi-agent workflow.
    
    It manages task distribution, execution tracking, and resource allocation
    across the different specialized agents.
    """
    
    def __init__(self,
                agents: Dict[str, List[BaseAgent]],
                memory_manager: MemoryManager,
                task_queue: TaskQueue,
                config: Dict[str, Any]):
        """
        Initialize the Supervisor.
        
        Args:
            agents: Dictionary mapping agent types to lists of agent instances
            memory_manager: Memory manager for persistent context
            task_queue: Task queue for async execution
            config: System configuration
        """
        self.agents = agents
        self.memory_manager = memory_manager
        self.task_queue = task_queue
        self.config = config
        
        # Workflow configuration
        self.max_iterations = config.get("system", {}).get("max_iterations", 5)
        self.tournament_size = config.get("system", {}).get("tournament_size", 8)
        self.top_hypotheses_count = config.get("system", {}).get("top_hypotheses_count", 3)
        
        # Session tracking
        self.current_sessions = {}
        self.session_callbacks = {}
    
    async def start_session(self, 
                          research_goal: str, 
                          domain: str = "general science",
                          constraints: str = "",
                          prior_knowledge: str = "",
                          callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> str:
        """
        Start a new research session.
        
        Args:
            research_goal: The scientific question or goal
            domain: Scientific domain (e.g., "biomedicine")
            constraints: Optional constraints or requirements
            prior_knowledge: Optional relevant prior knowledge
            callback: Optional callback function for session updates
            
        Returns:
            Session ID for tracking
        """
        session_id = str(uuid.uuid4())
        
        # Create session context
        session = {
            "id": session_id,
            "research_goal": research_goal,
            "domain": domain,
            "constraints": constraints,
            "prior_knowledge": prior_knowledge,
            "start_time": time.time(),
            "current_iteration": 0,
            "hypotheses": [],
            "status": "initializing",
            "complete": False
        }
        
        # Store in memory and track locally
        await self.memory_manager.save_session(session_id, session)
        self.current_sessions[session_id] = session
        
        if callback:
            self.session_callbacks[session_id] = callback
        
        # Start the workflow asynchronously
        asyncio.create_task(self._run_workflow(session_id))
        
        return session_id
    
    async def _run_workflow(self, session_id: str) -> None:
        """
        Run the complete workflow for a session.
        
        Args:
            session_id: The session ID
        """
        try:
            session = self.current_sessions[session_id]
            session["status"] = "running"
            await self._update_session(session_id, session)
            
            # Initial generation phase
            await self._generate_initial_hypotheses(session_id)
            
            # Iterative refinement
            for iteration in range(self.max_iterations):
                session = await self.memory_manager.get_session(session_id)
                if session.get("status") == "cancelled":
                    logger.info(f"Session {session_id} was cancelled")
                    return
                    
                session["current_iteration"] = iteration + 1
                await self._update_session(session_id, session)
                
                # Run the debate-and-evolve cycle
                await self._debate_phase(session_id)
                await self._rank_phase(session_id)
                await self._evolve_phase(session_id)
                
                # Check if we should stop early (convergence)
                if await self._check_convergence(session_id):
                    logger.info(f"Workflow converged at iteration {iteration + 1}")
                    break
            
            # Final ranking and meta-review
            await self._final_ranking(session_id)
            await self._meta_review(session_id)
            
            # Complete the session
            session = await self.memory_manager.get_session(session_id)
            session["status"] = "completed"
            session["complete"] = True
            session["end_time"] = time.time()
            await self._update_session(session_id, session)
            
        except Exception as e:
            logger.error(f"Error in workflow for session {session_id}: {e}")
            session = await self.memory_manager.get_session(session_id)
            session["status"] = "error"
            session["error"] = str(e)
            await self._update_session(session_id, session)
    
    async def _generate_initial_hypotheses(self, session_id: str) -> None:
        """
        Generate initial hypotheses using Generation Agents.
        
        Args:
            session_id: The session ID
        """
        session = await self.memory_manager.get_session(session_id)
        
        # Update status
        session["status"] = "generating_hypotheses"
        await self._update_session(session_id, session)
        
        # Create input for Generation Agents
        input_data = {
            "research_goal": session["research_goal"],
            "domain": session["domain"],
            "constraints": session["constraints"],
            "prior_knowledge": session["prior_knowledge"]
        }
        
        # Submit tasks to all available Generation Agents
        generation_agents = self.agents.get("generation", [])
        tasks = []
        
        for agent in generation_agents:
            task_id = f"generation_{uuid.uuid4()}"
            task = {
                "id": task_id,
                "session_id": session_id,
                "agent_type": "generation",
                "input": input_data,
                "status": "pending"
            }
            await self.task_queue.add_task(task_id, task)
            tasks.append(task_id)
        
        # Wait for all generation tasks to complete
        results = await self.task_queue.wait_for_tasks(tasks)
        
        # Combine all hypotheses
        all_hypotheses = []
        for result in results.values():
            if "error" in result:
                logger.warning(f"Error in generation task: {result['error']}")
                continue
                
            hypotheses = result.get("hypotheses", [])
            all_hypotheses.extend(hypotheses)
        
        # Update session with generated hypotheses
        session = await self.memory_manager.get_session(session_id)
        session["hypotheses"] = all_hypotheses
        session["status"] = "generation_complete"
        await self._update_session(session_id, session)
    
    async def _debate_phase(self, session_id: str) -> None:
        """
        Run the debate phase where Reflection Agents critique hypotheses.
        
        Args:
            session_id: The session ID
        """
        session = await self.memory_manager.get_session(session_id)
        
        # Update status
        session["status"] = "debating"
        await self._update_session(session_id, session)
        
        # Get all hypotheses
        hypotheses = session.get("hypotheses", [])
        if not hypotheses:
            logger.warning(f"No hypotheses found for session {session_id}")
            return
        
        # Submit critique tasks for each hypothesis
        reflection_agents = self.agents.get("reflection", [])
        if not reflection_agents:
            logger.warning("No reflection agents available")
            return
            
        tasks = []
        
        for hypothesis in hypotheses:
            # Round-robin distribution among reflection agents
            agent_idx = len(tasks) % len(reflection_agents)
            agent = reflection_agents[agent_idx]
            
            task_id = f"reflection_{uuid.uuid4()}"
            task = {
                "id": task_id,
                "session_id": session_id,
                "agent_type": "reflection",
                "input": {
                    "hypothesis": hypothesis,
                    "research_goal": session["research_goal"],
                    "domain": session["domain"]
                },
                "status": "pending"
            }
            await self.task_queue.add_task(task_id, task)
            tasks.append(task_id)
        
        # Wait for all critique tasks to complete
        results = await self.task_queue.wait_for_tasks(tasks)
        
        # Update hypotheses with critiques
        session = await self.memory_manager.get_session(session_id)
        updated_hypotheses = session.get("hypotheses", [])
        
        for result in results.values():
            if "error" in result:
                logger.warning(f"Error in reflection task: {result['error']}")
                continue
                
            critique = result.get("critique", {})
            hypothesis_id = result.get("hypothesis_id")
            
            # Find the matching hypothesis and add the critique
            for hypothesis in updated_hypotheses:
                if hypothesis.get("id") == hypothesis_id:
                    hypothesis["critique"] = critique
                    break
        
        # Update session with critiqued hypotheses
        session["hypotheses"] = updated_hypotheses
        session["status"] = "debate_complete"
        await self._update_session(session_id, session)
    
    async def _rank_phase(self, session_id: str) -> None:
        """
        Rank hypotheses using Ranking Agents.
        
        Args:
            session_id: The session ID
        """
        session = await self.memory_manager.get_session(session_id)
        
        # Update status
        session["status"] = "ranking"
        await self._update_session(session_id, session)
        
        # Get all hypotheses
        hypotheses = session.get("hypotheses", [])
        if not hypotheses:
            logger.warning(f"No hypotheses found for session {session_id}")
            return
        
        # Submit ranking task
        ranking_agents = self.agents.get("ranking", [])
        if not ranking_agents:
            logger.warning("No ranking agents available")
            return
        
        # Use the first available ranking agent
        agent = ranking_agents[0]
        
        task_id = f"ranking_{uuid.uuid4()}"
        task = {
            "id": task_id,
            "session_id": session_id,
            "agent_type": "ranking",
            "input": {
                "hypotheses": hypotheses,
                "research_goal": session["research_goal"]
            },
            "status": "pending"
        }
        await self.task_queue.add_task(task_id, task)
        
        # Wait for ranking task to complete
        results = await self.task_queue.wait_for_tasks([task_id])
        result = results.get(task_id, {})
        
        if "error" in result:
            logger.warning(f"Error in ranking task: {result['error']}")
            return
            
        # Update hypotheses with rankings
        ranked_hypotheses = result.get("ranked_hypotheses", [])
        
        # Update session with ranked hypotheses
        session = await self.memory_manager.get_session(session_id)
        session["hypotheses"] = ranked_hypotheses
        session["status"] = "ranking_complete"
        await self._update_session(session_id, session)
    
    async def _evolve_phase(self, session_id: str) -> None:
        """
        Evolve hypotheses using Evolution Agents.
        
        Args:
            session_id: The session ID
        """
        session = await self.memory_manager.get_session(session_id)
        
        # Update status
        session["status"] = "evolving"
        await self._update_session(session_id, session)
        
        # Get ranked hypotheses
        hypotheses = session.get("hypotheses", [])
        if not hypotheses:
            logger.warning(f"No hypotheses found for session {session_id}")
            return
        
        # Sort hypotheses by rank if available
        if hypotheses and "rank" in hypotheses[0]:
            hypotheses.sort(key=lambda h: h.get("rank", 999), reverse=False)
        
        # Take top hypotheses for evolution
        top_hypotheses = hypotheses[:min(len(hypotheses), self.tournament_size)]
        
        # Submit evolution tasks
        evolution_agents = self.agents.get("evolution", [])
        if not evolution_agents:
            logger.warning("No evolution agents available")
            return
            
        tasks = []
        
        # Process hypotheses in pairs for potential merging/evolution
        for i in range(0, len(top_hypotheses), 2):
            if i + 1 < len(top_hypotheses):
                # We have a pair
                hypothesis1 = top_hypotheses[i]
                hypothesis2 = top_hypotheses[i + 1]
                
                agent_idx = (i // 2) % len(evolution_agents)
                agent = evolution_agents[agent_idx]
                
                task_id = f"evolution_{uuid.uuid4()}"
                task = {
                    "id": task_id,
                    "session_id": session_id,
                    "agent_type": "evolution",
                    "input": {
                        "hypothesis1": hypothesis1,
                        "hypothesis2": hypothesis2,
                        "research_goal": session["research_goal"]
                    },
                    "status": "pending"
                }
                await self.task_queue.add_task(task_id, task)
                tasks.append(task_id)
            else:
                # Odd number of hypotheses, evolve the last one independently
                hypothesis = top_hypotheses[i]
                
                agent_idx = (i // 2) % len(evolution_agents)
                agent = evolution_agents[agent_idx]
                
                task_id = f"evolution_{uuid.uuid4()}"
                task = {
                    "id": task_id,
                    "session_id": session_id,
                    "agent_type": "evolution",
                    "input": {
                        "hypothesis": hypothesis,
                        "research_goal": session["research_goal"]
                    },
                    "status": "pending"
                }
                await self.task_queue.add_task(task_id, task)
                tasks.append(task_id)
        
        # Wait for all evolution tasks to complete
        results = await self.task_queue.wait_for_tasks(tasks)
        
        # Collect evolved hypotheses
        evolved_hypotheses = []
        for result in results.values():
            if "error" in result:
                logger.warning(f"Error in evolution task: {result['error']}")
                continue
                
            new_hypotheses = result.get("evolved_hypotheses", [])
            evolved_hypotheses.extend(new_hypotheses)
        
        # Update session with evolved hypotheses and keep some original ones
        session = await self.memory_manager.get_session(session_id)
        
        # Keep some of the original hypotheses that weren't in the top set
        remaining_hypotheses = [h for h in hypotheses if h not in top_hypotheses]
        
        # Combine evolved with some original ones
        combined_hypotheses = evolved_hypotheses + remaining_hypotheses
        
        session["hypotheses"] = combined_hypotheses
        session["status"] = "evolution_complete"
        await self._update_session(session_id, session)
    
    async def _check_convergence(self, session_id: str) -> bool:
        """
        Check if the hypotheses have converged.
        
        Args:
            session_id: The session ID
            
        Returns:
            True if converged, False otherwise
        """
        # This is a simplified convergence check
        # In a real implementation, you would track hypothesis changes over iterations
        
        session = await self.memory_manager.get_session(session_id)
        current_iteration = session.get("current_iteration", 0)
        
        # For now, we're not implementing early convergence
        return False
    
    async def _final_ranking(self, session_id: str) -> None:
        """
        Perform final ranking of hypotheses.
        
        Args:
            session_id: The session ID
        """
        # Similar to _rank_phase but we'll be more thorough
        await self._rank_phase(session_id)
        
        # After ranking, retain only the top hypotheses
        session = await self.memory_manager.get_session(session_id)
        hypotheses = session.get("hypotheses", [])
        
        # Sort by rank
        if hypotheses and "rank" in hypotheses[0]:
            hypotheses.sort(key=lambda h: h.get("rank", 999), reverse=False)
            
        # Keep only the top N
        top_hypotheses = hypotheses[:min(len(hypotheses), self.top_hypotheses_count)]
        
        session["top_hypotheses"] = top_hypotheses
        await self._update_session(session_id, session)
    
    async def _meta_review(self, session_id: str) -> None:
        """
        Generate meta-review using Meta-Review Agent.
        
        Args:
            session_id: The session ID
        """
        session = await self.memory_manager.get_session(session_id)
        
        # Update status
        session["status"] = "meta_review"
        await self._update_session(session_id, session)
        
        # Get top hypotheses
        top_hypotheses = session.get("top_hypotheses", [])
        if not top_hypotheses:
            logger.warning(f"No top hypotheses found for session {session_id}")
            return
        
        # Submit meta-review task
        meta_review_agents = self.agents.get("meta_review", [])
        if not meta_review_agents:
            logger.warning("No meta-review agents available")
            return
        
        # Use the first available meta-review agent
        agent = meta_review_agents[0]
        
        task_id = f"meta_review_{uuid.uuid4()}"
        task = {
            "id": task_id,
            "session_id": session_id,
            "agent_type": "meta_review",
            "input": {
                "top_hypotheses": top_hypotheses,
                "research_goal": session["research_goal"],
                "iterations": session["current_iteration"]
            },
            "status": "pending"
        }
        await self.task_queue.add_task(task_id, task)
        
        # Wait for meta-review task to complete
        results = await self.task_queue.wait_for_tasks([task_id])
        result = results.get(task_id, {})
        
        if "error" in result:
            logger.warning(f"Error in meta-review task: {result['error']}")
            return
            
        # Update session with meta-review
        meta_review = result.get("meta_review", {})
        
        session = await self.memory_manager.get_session(session_id)
        session["meta_review"] = meta_review
        session["status"] = "meta_review_complete"
        await self._update_session(session_id, session)
    
    async def _update_session(self, session_id: str, session: Dict[str, Any]) -> None:
        """
        Update session state and notify callback if registered.
        
        Args:
            session_id: The session ID
            session: Updated session data
        """
        # Update in memory
        await self.memory_manager.save_session(session_id, session)
        self.current_sessions[session_id] = session
        
        # Call callback if registered
        if session_id in self.session_callbacks and self.session_callbacks[session_id]:
            try:
                self.session_callbacks[session_id](session)
            except Exception as e:
                logger.error(f"Error in session callback: {e}")
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Get current session state.
        
        Args:
            session_id: The session ID
            
        Returns:
            Session data
        """
        return await self.memory_manager.get_session(session_id)
    
    async def cancel_session(self, session_id: str) -> None:
        """
        Cancel an ongoing session.
        
        Args:
            session_id: The session ID
        """
        session = await self.memory_manager.get_session(session_id)
        session["status"] = "cancelled"
        await self._update_session(session_id, session)
        
    @classmethod
    def from_config(cls, config: Dict[str, Any], agents: Dict[str, List[BaseAgent]], 
                   memory_manager: MemoryManager, task_queue: TaskQueue) -> 'Supervisor':
        """
        Create a Supervisor instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            agents: Dictionary of agent instances
            memory_manager: Memory manager
            task_queue: Task queue
            
        Returns:
            Configured Supervisor instance
        """
        return cls(
            agents=agents,
            memory_manager=memory_manager,
            task_queue=task_queue,
            config=config
        ) 