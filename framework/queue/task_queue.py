"""
Task Queue for AI Co-Scientist

This module implements the asynchronous task queue for distributing and executing
agent tasks across the system.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Union

logger = logging.getLogger(__name__)


class TaskQueue(ABC):
    """
    Abstract base class for task queue implementations.
    """
    
    @abstractmethod
    async def add_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """
        Add a new task to the queue.
        
        Args:
            task_id: Unique task identifier
            task_data: Task data including input, agent type, etc.
        """
        pass
    
    @abstractmethod
    async def get_task(self) -> Optional[Dict[str, Any]]:
        """
        Get the next available task from the queue.
        
        Returns:
            Task data or None if queue is empty
        """
        pass
    
    @abstractmethod
    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        """
        Mark a task as completed with results.
        
        Args:
            task_id: Task identifier
            result: Task execution result
        """
        pass
    
    @abstractmethod
    async def fail_task(self, task_id: str, error: str) -> None:
        """
        Mark a task as failed with error information.
        
        Args:
            task_id: Task identifier
            error: Error message
        """
        pass
    
    @abstractmethod
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the current status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status information
        """
        pass
    
    @abstractmethod
    async def wait_for_tasks(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for multiple tasks to complete.
        
        Args:
            task_ids: List of task identifiers to wait for
            timeout: Maximum time to wait in seconds (None for no timeout)
            
        Returns:
            Dictionary mapping task IDs to results
        """
        pass


class InMemoryTaskQueue(TaskQueue):
    """
    In-memory implementation of the task queue for simpler deployments.
    """
    
    def __init__(self, max_retries: int = 3):
        """
        Initialize the in-memory task queue.
        
        Args:
            max_retries: Maximum number of retry attempts for failed tasks
        """
        self.tasks = {}  # task_id -> task_data
        self.results = {}  # task_id -> result
        self.pending_tasks = asyncio.Queue()
        self.max_retries = max_retries
        
        # For tracking and monitoring
        self.completed_tasks = set()
        self.failed_tasks = set()
        self.task_completion_events = {}  # task_id -> asyncio.Event
    
    async def add_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """
        Add a new task to the queue.
        
        Args:
            task_id: Unique task identifier
            task_data: Task data including input, agent type, etc.
        """
        if task_id in self.tasks:
            logger.warning(f"Task {task_id} already exists in queue")
            return
            
        # Initialize task metadata
        task_data["status"] = "pending"
        task_data["created_at"] = time.time()
        task_data["attempts"] = 0
        
        # Store task
        self.tasks[task_id] = task_data
        
        # Create completion event
        self.task_completion_events[task_id] = asyncio.Event()
        
        # Add to pending queue
        await self.pending_tasks.put(task_id)
        
        logger.debug(f"Added task {task_id} to queue")
    
    async def get_task(self) -> Optional[Dict[str, Any]]:
        """
        Get the next available task from the queue.
        
        Returns:
            Task data or None if queue is empty
        """
        try:
            # Get next task ID from queue (blocking)
            task_id = await self.pending_tasks.get()
            
            # Get task data
            task_data = self.tasks[task_id]
            
            # Update status and attempts
            task_data["status"] = "processing"
            task_data["attempts"] += 1
            task_data["started_at"] = time.time()
            
            # Update task in storage
            self.tasks[task_id] = task_data
            
            # Return a copy with task_id included
            result = task_data.copy()
            result["task_id"] = task_id
            return result
        except Exception as e:
            logger.error(f"Error getting task from queue: {e}")
            return None
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        """
        Mark a task as completed with results.
        
        Args:
            task_id: Task identifier
            result: Task execution result
        """
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found in queue")
            return
            
        # Get task data
        task_data = self.tasks[task_id]
        
        # Update status
        task_data["status"] = "completed"
        task_data["completed_at"] = time.time()
        if "started_at" in task_data:
            task_data["execution_time"] = task_data["completed_at"] - task_data["started_at"]
        
        # Store result
        self.results[task_id] = result
        
        # Track completion
        self.completed_tasks.add(task_id)
        
        # Signal completion
        if task_id in self.task_completion_events:
            self.task_completion_events[task_id].set()
            
        # Mark queue task as done
        self.pending_tasks.task_done()
        
        logger.debug(f"Task {task_id} completed")
    
    async def fail_task(self, task_id: str, error: str) -> None:
        """
        Mark a task as failed with error information.
        
        Args:
            task_id: Task identifier
            error: Error message
        """
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found in queue")
            return
            
        # Get task data
        task_data = self.tasks[task_id]
        
        # Check if we should retry
        if task_data["attempts"] < self.max_retries:
            logger.info(f"Retrying task {task_id} (attempt {task_data['attempts']})")
            
            # Reset task status
            task_data["status"] = "pending"
            
            # Add back to queue
            await self.pending_tasks.put(task_id)
        else:
            # Max retries reached, mark as failed
            task_data["status"] = "failed"
            task_data["failed_at"] = time.time()
            task_data["error"] = error
            
            # Store error result
            self.results[task_id] = {"error": error}
            
            # Track failure
            self.failed_tasks.add(task_id)
            
            # Signal completion (with failure)
            if task_id in self.task_completion_events:
                self.task_completion_events[task_id].set()
                
            # Mark queue task as done
            self.pending_tasks.task_done()
            
            logger.warning(f"Task {task_id} failed after {task_data['attempts']} attempts: {error}")
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the current status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status information
        """
        if task_id not in self.tasks:
            return {"task_id": task_id, "status": "not_found"}
            
        # Get task data
        task_data = self.tasks[task_id]
        
        # Build status response
        status = {
            "task_id": task_id,
            "status": task_data["status"],
            "created_at": task_data.get("created_at"),
            "attempts": task_data.get("attempts", 0)
        }
        
        # Add additional fields based on status
        if task_data["status"] == "completed":
            status["completed_at"] = task_data.get("completed_at")
            status["execution_time"] = task_data.get("execution_time")
            status["result"] = self.results.get(task_id, {})
        elif task_data["status"] == "failed":
            status["failed_at"] = task_data.get("failed_at")
            status["error"] = task_data.get("error")
        elif task_data["status"] == "processing":
            status["started_at"] = task_data.get("started_at")
            
        return status
    
    async def wait_for_tasks(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for multiple tasks to complete.
        
        Args:
            task_ids: List of task identifiers to wait for
            timeout: Maximum time to wait in seconds (None for no timeout)
            
        Returns:
            Dictionary mapping task IDs to results
        """
        # Create list of events to wait for
        events = []
        for task_id in task_ids:
            if task_id not in self.task_completion_events:
                # If task doesn't exist, create a completed event
                event = asyncio.Event()
                event.set()
                self.task_completion_events[task_id] = event
            
            events.append(self.task_completion_events[task_id].wait())
        
        # Wait for all events (with timeout if specified)
        try:
            await asyncio.wait_for(asyncio.gather(*events), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for tasks: {task_ids}")
        
        # Collect results
        results = {}
        for task_id in task_ids:
            if task_id in self.results:
                results[task_id] = self.results[task_id]
            else:
                # Task not completed or failed
                status = await self.get_task_status(task_id)
                results[task_id] = {"error": f"Task not completed: {status['status']}"}
        
        return results


class RedisTaskQueue(TaskQueue):
    """
    Redis-based implementation of the task queue for distributed deployments.
    
    Note: This is a stub implementation. In a real system, this would be fully
    implemented with connection to a Redis server.
    """
    
    def __init__(self, redis_host: str, redis_port: int, redis_db: int = 0, max_retries: int = 3):
        """
        Initialize the Redis task queue.
        
        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            max_retries: Maximum number of retry attempts for failed tasks
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.max_retries = max_retries
        
        # This would connect to Redis in a real implementation
        logger.info(f"Connecting to Redis at {redis_host}:{redis_port} (db: {redis_db})")
    
    async def add_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """
        Add a new task to the queue.
        
        Args:
            task_id: Unique task identifier
            task_data: Task data including input, agent type, etc.
        """
        raise NotImplementedError("Redis task queue is not fully implemented")
    
    async def get_task(self) -> Optional[Dict[str, Any]]:
        """
        Get the next available task from the queue.
        
        Returns:
            Task data or None if queue is empty
        """
        raise NotImplementedError("Redis task queue is not fully implemented")
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        """
        Mark a task as completed with results.
        
        Args:
            task_id: Task identifier
            result: Task execution result
        """
        raise NotImplementedError("Redis task queue is not fully implemented")
    
    async def fail_task(self, task_id: str, error: str) -> None:
        """
        Mark a task as failed with error information.
        
        Args:
            task_id: Task identifier
            error: Error message
        """
        raise NotImplementedError("Redis task queue is not fully implemented")
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the current status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status information
        """
        raise NotImplementedError("Redis task queue is not fully implemented")
    
    async def wait_for_tasks(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for multiple tasks to complete.
        
        Args:
            task_ids: List of task identifiers to wait for
            timeout: Maximum time to wait in seconds (None for no timeout)
            
        Returns:
            Dictionary mapping task IDs to results
        """
        raise NotImplementedError("Redis task queue is not fully implemented")


def create_task_queue(config: Dict[str, Any]) -> TaskQueue:
    """
    Factory function to create the appropriate task queue based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured TaskQueue instance
    """
    queue_type = config.get("execution", {}).get("queue_backend", "memory")
    
    if queue_type == "redis":
        redis_config = config.get("execution", {}).get("redis", {})
        return RedisTaskQueue(
            redis_host=redis_config.get("host", "localhost"),
            redis_port=redis_config.get("port", 6379),
            redis_db=redis_config.get("db", 0),
            max_retries=config.get("execution", {}).get("retry_limit", 3)
        )
    else:
        # Default to in-memory queue
        return InMemoryTaskQueue(
            max_retries=config.get("execution", {}).get("retry_limit", 3)
        ) 