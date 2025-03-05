"""
Memory Manager for AI Co-Scientist

This module implements the persistent context storage for session data,
hypotheses, and other information that needs to be shared across agents
and retained between system restarts.
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryManager(ABC):
    """
    Abstract base class for memory management.
    """
    
    @abstractmethod
    async def startup(self) -> None:
        """
        Initialize and start the memory manager.
        This method should be called before using the memory manager.
        """
        pass
    
    @abstractmethod
    async def load_session(self, session_id: str) -> Any:
        """
        Load a workflow session from storage.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Loaded session object
        """
        pass
    
    @abstractmethod
    async def store_session(self, session: Any) -> None:
        """
        Store a workflow session.
        
        Args:
            session: Workflow session object
        """
        pass
    
    @abstractmethod
    async def save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """
        Save session data to persistent storage.
        
        Args:
            session_id: Unique session identifier
            session_data: Session data to store
        """
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve session data from storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or empty dict if not found
        """
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions in storage.
        
        Returns:
            List of session summaries
        """
        pass
    
    @abstractmethod
    async def save_hypothesis(self, hypothesis_id: str, hypothesis_data: Dict[str, Any]) -> None:
        """
        Save a hypothesis to persistent storage.
        
        Args:
            hypothesis_id: Unique hypothesis identifier
            hypothesis_data: Hypothesis data to store
        """
        pass
    
    @abstractmethod
    async def get_hypothesis(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Retrieve a hypothesis from storage.
        
        Args:
            hypothesis_id: Hypothesis identifier
            
        Returns:
            Hypothesis data or empty dict if not found
        """
        pass
    
    @abstractmethod
    async def get_hypotheses_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all hypotheses for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of hypothesis data
        """
        pass


class InMemoryMemoryManager(MemoryManager):
    """
    In-memory implementation of the memory manager.
    
    Warning: Data is lost when the application restarts.
    """
    
    def __init__(self):
        """Initialize the in-memory memory manager."""
        self.sessions = {}
        self.hypotheses = {}
    
    async def startup(self) -> None:
        """
        Initialize and start the in-memory memory manager.
        """
        pass
    
    async def load_session(self, session_id: str) -> Any:
        """
        Load a workflow session from memory.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Loaded session object
        """
        session_data = await self.get_session(session_id)
        if not session_data:
            raise ValueError(f"Session {session_id} not found")
        
        # Here you would typically reconstruct a session object from the data
        # For now, we'll just return the session data dictionary
        return session_data
    
    async def store_session(self, session: Any) -> None:
        """
        Store a workflow session in memory.
        
        Args:
            session: Workflow session object
        """
        # Convert session to dictionary and store it
        session_data = session.to_dict() if hasattr(session, 'to_dict') else session
        await self.save_session(session.id if hasattr(session, 'id') else session_data.get('id'), session_data)
    
    async def save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """
        Save session data to in-memory storage.
        
        Args:
            session_id: Unique session identifier
            session_data: Session data to store
        """
        self.sessions[session_id] = session_data.copy()
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve session data from in-memory storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or empty dict if not found
        """
        return self.sessions.get(session_id, {}).copy()
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from in-memory storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            # Also delete associated hypotheses
            self.hypotheses = {k: v for k, v in self.hypotheses.items() 
                              if v.get("session_id") != session_id}
            return True
        return False
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions in in-memory storage.
        
        Returns:
            List of session summaries
        """
        return [
            {
                "id": session_id,
                "research_goal": session.get("research_goal", ""),
                "status": session.get("status", "unknown"),
                "created_at": session.get("start_time", 0),
                "updated_at": session.get("update_time", 0),
                "hypothesis_count": len([h for h in self.hypotheses.values() 
                                       if h.get("session_id") == session_id])
            }
            for session_id, session in self.sessions.items()
        ]
    
    async def save_hypothesis(self, hypothesis_id: str, hypothesis_data: Dict[str, Any]) -> None:
        """
        Save a hypothesis to in-memory storage.
        
        Args:
            hypothesis_id: Unique hypothesis identifier
            hypothesis_data: Hypothesis data to store
        """
        self.hypotheses[hypothesis_id] = hypothesis_data.copy()
    
    async def get_hypothesis(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Retrieve a hypothesis from in-memory storage.
        
        Args:
            hypothesis_id: Hypothesis identifier
            
        Returns:
            Hypothesis data or empty dict if not found
        """
        return self.hypotheses.get(hypothesis_id, {}).copy()
    
    async def get_hypotheses_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all hypotheses for a session from in-memory storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of hypothesis data
        """
        return [h.copy() for h in self.hypotheses.values() 
                if h.get("session_id") == session_id]


class FileSystemMemoryManager(MemoryManager):
    """
    File system implementation of the memory manager.
    
    Data is stored as JSON files in the specified directory.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the file system memory manager.
        
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = data_dir
        self.sessions_dir = os.path.join(data_dir, "sessions")
        self.hypotheses_dir = os.path.join(data_dir, "hypotheses")
        
        # Create directories if they don't exist
        os.makedirs(self.sessions_dir, exist_ok=True)
        os.makedirs(self.hypotheses_dir, exist_ok=True)
    
    async def startup(self) -> None:
        """
        Initialize and start the file system memory manager.
        """
        pass
    
    async def load_session(self, session_id: str) -> Any:
        """
        Load a workflow session from the file system.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Loaded session object
        """
        session_data = await self.get_session(session_id)
        if not session_data:
            raise ValueError(f"Session {session_id} not found")
        
        # Import here to avoid circular imports
        from ..workflow.supervisor import WorkflowSession, ResearchGoal, WorkflowState, Hypothesis
        
        # Reconstruct the goal object
        goal_data = session_data.get("goal", {})
        goal = ResearchGoal.from_dict(goal_data)
        
        # Reconstruct hypotheses
        hypotheses = []
        for h_data in session_data.get("hypotheses", []):
            hypotheses.append(Hypothesis.from_dict(h_data))
        
        # Convert state string to enum
        state_str = session_data.get("state", "initial")
        state = WorkflowState(state_str)
        
        # Reconstruct the session object
        session = WorkflowSession(
            id=session_data.get("id"),
            goal=goal,
            hypotheses=hypotheses,
            iterations_completed=session_data.get("iterations_completed", 0),
            max_iterations=session_data.get("max_iterations", 5),
            state=state,
            feedback_history=session_data.get("feedback_history", []),
            top_hypotheses=session_data.get("top_hypotheses", []),
            tool_usage=session_data.get("tool_usage", {})
        )
        
        # Set timestamps if available
        if "started_at" in session_data and session_data["started_at"]:
            session.started_at = datetime.fromisoformat(session_data["started_at"])
        
        if "completed_at" in session_data and session_data["completed_at"]:
            session.completed_at = datetime.fromisoformat(session_data["completed_at"])
        
        return session
    
    async def store_session(self, session: Any) -> None:
        """
        Store a workflow session in the file system.
        
        Args:
            session: Workflow session object
        """
        # Convert session to dictionary and store it
        session_data = session.to_dict() if hasattr(session, 'to_dict') else session
        await self.save_session(session.id if hasattr(session, 'id') else session_data.get('id'), session_data)
    
    async def save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """
        Save session data to file system.
        
        Args:
            session_id: Unique session identifier
            session_data: Session data to store
        """
        # Add update timestamp
        session_data["update_time"] = time.time()
        
        # Save to file
        file_path = os.path.join(self.sessions_dir, f"{session_id}.json")
        await self._write_json_file(file_path, session_data)
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve session data from file system.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or empty dict if not found
        """
        file_path = os.path.join(self.sessions_dir, f"{session_id}.json")
        return await self._read_json_file(file_path)
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from file system.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        file_path = os.path.join(self.sessions_dir, f"{session_id}.json")
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                
                # Also delete associated hypotheses
                hypotheses = await self.get_hypotheses_by_session(session_id)
                for hypothesis in hypotheses:
                    if "id" in hypothesis:
                        hypothesis_path = os.path.join(self.hypotheses_dir, f"{hypothesis['id']}.json")
                        if os.path.exists(hypothesis_path):
                            os.remove(hypothesis_path)
                
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions in file system.
        
        Returns:
            List of session summaries
        """
        session_files = [f for f in os.listdir(self.sessions_dir) if f.endswith(".json")]
        sessions = []
        
        for file_name in session_files:
            try:
                session_id = file_name.replace(".json", "")
                file_path = os.path.join(self.sessions_dir, file_name)
                session = await self._read_json_file(file_path)
                
                # Get hypothesis count for this session
                hypothesis_count = len(await self.get_hypotheses_by_session(session_id))
                
                sessions.append({
                    "id": session_id,
                    "research_goal": session.get("research_goal", ""),
                    "status": session.get("status", "unknown"),
                    "created_at": session.get("start_time", 0),
                    "updated_at": session.get("update_time", 0),
                    "hypothesis_count": hypothesis_count
                })
            except Exception as e:
                logger.error(f"Error reading session file {file_name}: {e}")
        
        return sessions
    
    async def save_hypothesis(self, hypothesis_id: str, hypothesis_data: Dict[str, Any]) -> None:
        """
        Save a hypothesis to file system.
        
        Args:
            hypothesis_id: Unique hypothesis identifier
            hypothesis_data: Hypothesis data to store
        """
        # Add update timestamp
        hypothesis_data["update_time"] = time.time()
        
        # Save to file
        file_path = os.path.join(self.hypotheses_dir, f"{hypothesis_id}.json")
        await self._write_json_file(file_path, hypothesis_data)
    
    async def get_hypothesis(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Retrieve a hypothesis from file system.
        
        Args:
            hypothesis_id: Hypothesis identifier
            
        Returns:
            Hypothesis data or empty dict if not found
        """
        file_path = os.path.join(self.hypotheses_dir, f"{hypothesis_id}.json")
        return await self._read_json_file(file_path)
    
    async def get_hypotheses_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all hypotheses for a session from file system.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of hypothesis data
        """
        hypothesis_files = [f for f in os.listdir(self.hypotheses_dir) if f.endswith(".json")]
        hypotheses = []
        
        for file_name in hypothesis_files:
            try:
                file_path = os.path.join(self.hypotheses_dir, file_name)
                hypothesis = await self._read_json_file(file_path)
                
                if hypothesis.get("session_id") == session_id:
                    hypotheses.append(hypothesis)
            except Exception as e:
                logger.error(f"Error reading hypothesis file {file_name}: {e}")
        
        return hypotheses
    
    async def _write_json_file(self, file_path: str, data: Dict[str, Any]) -> None:
        """
        Write data to a JSON file asynchronously.
        
        Args:
            file_path: Path to the file
            data: Data to write
        """
        try:
            # We're running in an event loop, so use run_in_executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._write_json_sync, file_path, data)
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {e}")
            raise
    
    def _write_json_sync(self, file_path: str, data: Dict[str, Any]) -> None:
        """
        Synchronous implementation of JSON file writing.
        
        Args:
            file_path: Path to the file
            data: Data to write
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def _read_json_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read data from a JSON file asynchronously.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Data from the file or empty dict if file doesn't exist
        """
        if not os.path.exists(file_path):
            return {}
            
        try:
            # We're running in an event loop, so use run_in_executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._read_json_sync, file_path)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {}
    
    def _read_json_sync(self, file_path: str) -> Dict[str, Any]:
        """
        Synchronous implementation of JSON file reading.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Data from the file
        """
        with open(file_path, 'r') as f:
            return json.load(f)


class MongoDBMemoryManager(MemoryManager):
    """
    MongoDB implementation of the memory manager.
    
    Note: This is a stub implementation. In a real system, this would be fully
    implemented with connection to a MongoDB server.
    """
    
    def __init__(self, mongo_uri: str, db_name: str = "co_scientist"):
        """
        Initialize the MongoDB memory manager.
        
        Args:
            mongo_uri: MongoDB connection URI
            db_name: Database name
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        
        # This would connect to MongoDB in a real implementation
        logger.info(f"Connecting to MongoDB at {mongo_uri} (db: {db_name})")
    
    async def startup(self) -> None:
        """
        Initialize and start the MongoDB connection.
        """
        # In a real implementation, this would establish the MongoDB connection
        # For now, we'll just pass since the connection info is logged in __init__
        pass
    
    async def load_session(self, session_id: str) -> Any:
        """
        Load a workflow session from MongoDB.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Loaded session object
        """
        session_data = await self.get_session(session_id)
        if not session_data:
            raise ValueError(f"Session {session_id} not found")
        
        # Here you would typically reconstruct a session object from the data
        # For now, we'll just return the session data dictionary
        return session_data
    
    async def store_session(self, session: Any) -> None:
        """
        Store a workflow session in MongoDB.
        
        Args:
            session: Workflow session object
        """
        # Convert session to dictionary and store it
        session_data = session.to_dict() if hasattr(session, 'to_dict') else session
        await self.save_session(session.id if hasattr(session, 'id') else session_data.get('id'), session_data)
    
    async def save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """
        Save session data to MongoDB.
        
        Args:
            session_id: Unique session identifier
            session_data: Session data to store
        """
        raise NotImplementedError("MongoDB memory manager is not fully implemented")
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve session data from MongoDB.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or empty dict if not found
        """
        raise NotImplementedError("MongoDB memory manager is not fully implemented")
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from MongoDB.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("MongoDB memory manager is not fully implemented")
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions in MongoDB.
        
        Returns:
            List of session summaries
        """
        raise NotImplementedError("MongoDB memory manager is not fully implemented")
    
    async def save_hypothesis(self, hypothesis_id: str, hypothesis_data: Dict[str, Any]) -> None:
        """
        Save a hypothesis to MongoDB.
        
        Args:
            hypothesis_id: Unique hypothesis identifier
            hypothesis_data: Hypothesis data to store
        """
        raise NotImplementedError("MongoDB memory manager is not fully implemented")
    
    async def get_hypothesis(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Retrieve a hypothesis from MongoDB.
        
        Args:
            hypothesis_id: Hypothesis identifier
            
        Returns:
            Hypothesis data or empty dict if not found
        """
        raise NotImplementedError("MongoDB memory manager is not fully implemented")
    
    async def get_hypotheses_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all hypotheses for a session from MongoDB.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of hypothesis data
        """
        raise NotImplementedError("MongoDB memory manager is not fully implemented")


def create_memory_manager(config: Dict[str, Any]) -> MemoryManager:
    """
    Factory function to create the appropriate memory manager based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured MemoryManager instance
    """
    backend = config.get("memory", {}).get("backend", "memory")
    
    if backend == "mongodb":
        mongodb_config = config.get("memory", {}).get("mongodb", {})
        return MongoDBMemoryManager(
            mongo_uri=mongodb_config.get("uri", "mongodb://localhost:27017/"),
            db_name=mongodb_config.get("db_name", "co_scientist")
        )
    elif backend == "file_system":
        return FileSystemMemoryManager(
            data_dir=config.get("memory", {}).get("file_dir", "data")
        )
    else:
        # Default to in-memory
        return InMemoryMemoryManager() 