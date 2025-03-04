"""
Main Controller for AI Co-Scientist

This module implements the central controller that orchestrates the entire
AI Co-Scientist system, managing workflows, sessions, and user interactions.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union, Tuple

from .models.model_factory import ModelFactory
from .agents.agent_factory import AgentFactory
from .memory.memory_manager import MemoryManager
from .workflow.supervisor import SupervisorAgent, WorkflowSession, WorkflowState, ResearchGoal

logger = logging.getLogger(__name__)


class CoScientistController:
    """
    Controller for the AI Co-Scientist system.
    
    This class is the main entry point for the system, handling initialization,
    configuration, session management, and coordination of the scientific workflow.
    """
    
    def __init__(self, config_path: str = None, config: Dict[str, Any] = None):
        """
        Initialize the Co-Scientist controller.
        
        Args:
            config_path: Path to configuration file (optional)
            config: Configuration dictionary (optional, overrides config_path)
        """
        # Load configuration
        self.config = self._load_config(config_path, config)
        
        # Initialize components
        self.model_factory = ModelFactory()
        self.memory_manager = self._init_memory_manager()
        self.agent_factory = AgentFactory()
        
        # Create agents based on configuration
        self.agents = self.agent_factory.create_all_agents(
            config=self.config,
            model_factory=self.model_factory
        )
        
        # Initialize the supervisor agent
        self.supervisor = self._init_supervisor()
        
        # Active sessions
        self.active_sessions = {}
        
        # Status indicators
        self.initialized = True
        self.system_ready = False
        
        logger.info("CoScientist controller initialized")
    
    async def startup(self):
        """Start the system and initialize components."""
        try:
            # Start memory manager
            await self.memory_manager.startup()
            
            # Mark system as ready
            self.system_ready = True
            
            logger.info("CoScientist system started successfully")
            
        except Exception as e:
            self.system_ready = False
            logger.error(f"Error during system startup: {str(e)}")
            raise
    
    async def shutdown(self):
        """Shutdown the system gracefully."""
        try:
            # Clear caches
            self.model_factory.clear_cache()
            self.agent_factory.clear_cache()
            
            # Shutdown memory manager
            if self.memory_manager:
                await self.memory_manager.shutdown()
                
            # Mark system as not ready
            self.system_ready = False
            
            logger.info("CoScientist system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {str(e)}")
            raise
    
    async def create_session(self, 
                           goal_description: str,
                           domain: str,
                           background: str = "",
                           constraints: List[str] = None) -> str:
        """
        Create a new research session.
        
        Args:
            goal_description: Description of the research goal
            domain: Scientific domain of the research
            background: Background information (optional)
            constraints: List of constraints (optional)
            
        Returns:
            Session ID
        """
        if not self.system_ready:
            raise RuntimeError("System is not ready. Call startup() first.")
            
        # Create a new session through the supervisor
        session_id = await self.supervisor.create_session(
            goal_description=goal_description,
            domain=domain,
            background=background,
            constraints=constraints
        )
        
        # Store in active sessions
        self.active_sessions[session_id] = {
            "id": session_id,
            "goal": goal_description,
            "domain": domain,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info(f"Created new session {session_id} for goal: {goal_description}")
        return session_id
    
    async def run_session(self, 
                        session_id: str, 
                        wait_for_completion: bool = False,
                        status_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run a research session.
        
        Args:
            session_id: ID of the session to run
            wait_for_completion: Whether to wait for completion or return immediately
            status_callback: Callback function for status updates
            
        Returns:
            Session status dictionary
        """
        if not self.system_ready:
            raise RuntimeError("System is not ready. Call startup() first.")
            
        # Define state change handler
        async def on_state_change(session, old_state, new_state):
            # Update active sessions
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["last_updated"] = datetime.now().isoformat()
                self.active_sessions[session_id]["state"] = new_state.value
                
            # Call user callback if provided
            if status_callback:
                try:
                    await status_callback(session_id, old_state.value, new_state.value)
                except Exception as e:
                    logger.error(f"Error in status callback: {str(e)}")
        
        # Run the session
        session = await self.supervisor.run_session(
            session_id=session_id,
            on_state_change=on_state_change
        )
        
        # Get current status
        status = await self.get_session_status(session_id)
        
        # If session is not completed and we need to wait
        if wait_for_completion and session.state != WorkflowState.COMPLETED:
            # Wait for session to complete in background
            asyncio.create_task(self._wait_for_completion(session_id, status_callback))
        
        return status
    
    async def add_feedback(self, 
                         session_id: str, 
                         feedback: str,
                         target_hypothesis_ids: List[str] = None) -> Dict[str, Any]:
        """
        Add scientist feedback to a session.
        
        Args:
            session_id: Session ID
            feedback: Feedback text
            target_hypothesis_ids: List of hypothesis IDs the feedback applies to
            
        Returns:
            Session status dictionary
        """
        if not self.system_ready:
            raise RuntimeError("System is not ready. Call startup() first.")
            
        # Add feedback through the supervisor
        await self.supervisor.add_feedback(
            session_id=session_id,
            feedback=feedback,
            target_hypothesis_ids=target_hypothesis_ids
        )
        
        # Update active session
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["last_updated"] = datetime.now().isoformat()
            
        # Get current status
        status = await self.get_session_status(session_id)
        
        # If session is in awaiting feedback state, resume it
        if status.get("state") == WorkflowState.AWAITING_FEEDBACK.value:
            await self.resume_session(session_id)
            
        return status
    
    async def resume_session(self, session_id: str) -> Dict[str, Any]:
        """
        Resume a paused session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session status dictionary
        """
        if not self.system_ready:
            raise RuntimeError("System is not ready. Call startup() first.")
            
        # Run the session again (will resume from current state)
        await self.supervisor.run_session(session_id)
        
        # Update active session
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["last_updated"] = datetime.now().isoformat()
            
        # Get current status
        status = await self.get_session_status(session_id)
        
        return status
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get the current status of a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session status dictionary
        """
        if not self.system_ready:
            raise RuntimeError("System is not ready. Call startup() first.")
            
        # Get status from supervisor
        status = await self.supervisor.get_session_status(session_id)
        
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
        if not self.system_ready:
            raise RuntimeError("System is not ready. Call startup() first.")
            
        # Get hypotheses from supervisor
        hypotheses = await self.supervisor.get_hypotheses(
            session_id=session_id,
            limit=limit,
            include_all=include_all
        )
        
        return hypotheses
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions.
        
        Returns:
            List of session dictionaries
        """
        if not self.system_ready:
            raise RuntimeError("System is not ready. Call startup() first.")
            
        # Get sessions from memory manager
        session_ids = await self.memory_manager.list_sessions()
        
        # Prepare results
        sessions = []
        
        # Get status for each session
        for session_id in session_ids:
            try:
                status = await self.get_session_status(session_id)
                sessions.append(status)
            except Exception as e:
                logger.error(f"Error getting status for session {session_id}: {str(e)}")
        
        return sessions
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Success flag
        """
        if not self.system_ready:
            raise RuntimeError("System is not ready. Call startup() first.")
            
        # Delete from memory manager
        success = await self.memory_manager.delete_session(session_id)
        
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            
        return success
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            System information dictionary
        """
        # Get information about available models
        available_models = self.model_factory.get_available_models()
        
        # Get information about available agents
        available_agents = self.agent_factory.get_available_agent_types()
        
        # Build system info
        info = {
            "system_ready": self.system_ready,
            "version": self.config.get("version", "1.0.0"),
            "available_models": available_models,
            "available_agents": available_agents,
            "active_sessions": len(self.active_sessions),
            "config": {
                k: v for k, v in self.config.items() 
                if k not in ["api_keys", "credentials", "secrets"]
            }
        }
        
        return info
    
    async def _wait_for_completion(self, 
                                 session_id: str, 
                                 status_callback: Optional[Callable] = None) -> None:
        """
        Wait for a session to complete.
        
        Args:
            session_id: Session ID
            status_callback: Callback function for status updates
        """
        max_wait_time = 3600  # 1 hour max wait
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Get current status
            status = await self.get_session_status(session_id)
            
            # Check if completed
            if status.get("state") in [WorkflowState.COMPLETED.value, WorkflowState.ERROR.value]:
                if status_callback:
                    await status_callback(session_id, "waiting", status.get("state"))
                return
                
            # Wait before checking again
            await asyncio.sleep(5)
            
        # Timeout
        logger.warning(f"Timeout waiting for session {session_id} to complete")
        if status_callback:
            await status_callback(session_id, "waiting", "timeout")
    
    def _init_memory_manager(self) -> MemoryManager:
        """
        Initialize the memory manager.
        
        Returns:
            Configured memory manager
        """
        memory_config = self.config.get("memory", {})
        persistence_type = memory_config.get("persistence_type", "file")
        
        if persistence_type == "file":
            data_dir = memory_config.get("data_dir", "data")
            os.makedirs(data_dir, exist_ok=True)
            
        return MemoryManager(memory_config)
    
    def _init_supervisor(self) -> SupervisorAgent:
        """
        Initialize the supervisor agent.
        
        Returns:
            Configured supervisor agent
        """
        return SupervisorAgent(
            config=self.config,
            memory_manager=self.memory_manager,
            model_factory=self.model_factory,
            agent_registry=self.agents
        )
    
    def _load_config(self, config_path: Optional[str], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load configuration from file or dictionary.
        
        Args:
            config_path: Path to configuration file
            config: Configuration dictionary
            
        Returns:
            Loaded configuration
        """
        if config is not None:
            return config
            
        if config_path is not None and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
                
        # Use default configuration
        default_config = {
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
                },
                "evolution": {
                    "model": {"provider": "openai", "model_name": "gpt-4"},
                    "creativity_level": 0.6
                },
                "ranking": {
                    "model": {"provider": "openai", "model_name": "gpt-4"}
                },
                "proximity": {
                    "model": {"provider": "openai", "model_name": "gpt-4"},
                    "search_depth": "moderate"
                },
                "meta_review": {
                    "model": {"provider": "openai", "model_name": "gpt-4"},
                    "output_format": "comprehensive"
                }
            },
            "memory": {
                "persistence_type": "file",
                "data_dir": "data"
            },
            "workflow": {
                "max_iterations": 3,
                "top_hypotheses_count": 3
            }
        }
        
        logger.warning("No configuration provided, using default configuration")
        return default_config 