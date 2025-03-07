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
    
    This implementation supports distributed execution of tasks across multiple processes
    or machines using Redis as a centralized queue and state storage.
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
        self.task_completion_events = {}  # For local tracking
        
        # Import Redis here to avoid dependency issues if Redis is not used
        import redis
        import aioredis
        
        # Create synchronous client for initialization
        self.redis_sync = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            db=redis_db, 
            decode_responses=True
        )
        
        # Create async Redis connection
        self.redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
        
        logger.info(f"Initialized Redis task queue at {redis_host}:{redis_port} (db: {redis_db})")
    
    async def _connect(self):
        """Create or return the async Redis connection."""
        if not hasattr(self, 'redis'):
            import aioredis
            self.redis = await aioredis.from_url(
                self.redis_url, 
                decode_responses=True
            )
        return self.redis
    
    async def add_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """
        Add a new task to the queue.
        
        Args:
            task_id: Unique task identifier
            task_data: Task data including input, agent type, etc.
        """
        redis = await self._connect()
        
        # Initialize task metadata
        task_data["status"] = "pending"
        task_data["created_at"] = time.time()
        task_data["attempts"] = 0
        
        # Store task data
        await redis.hset(f"task:{task_id}", mapping=self._serialize_dict(task_data))
        
        # Add to pending queue
        await redis.lpush("pending_tasks", task_id)
        
        # Create local completion event for waiting
        self.task_completion_events[task_id] = asyncio.Event()
        
        logger.debug(f"Added task {task_id} to Redis queue")
    
    async def get_task(self) -> Optional[Dict[str, Any]]:
        """
        Get the next available task from the queue.
        
        Returns:
            Task data or None if queue is empty
        """
        redis = await self._connect()
        
        # Get next task from queue with timeout (non-blocking)
        result = await redis.brpop("pending_tasks", timeout=1)
        if not result:
            return None
            
        _, task_id = result
        
        # Get task data
        task_data_raw = await redis.hgetall(f"task:{task_id}")
        if not task_data_raw:
            logger.warning(f"Task {task_id} found in queue but no data exists")
            return None
            
        # Deserialize task data
        task_data = self._deserialize_dict(task_data_raw)
        
        # Update status and attempts
        task_data["status"] = "processing"
        task_data["attempts"] = int(task_data.get("attempts", 0)) + 1
        task_data["started_at"] = time.time()
        
        # Save updated task data
        await redis.hset(f"task:{task_id}", mapping=self._serialize_dict(task_data))
        
        # Add to processing set with expiration
        await redis.sadd("processing_tasks", task_id)
        
        # Return task data with ID
        result = task_data.copy()
        result["task_id"] = task_id
        return result
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        """
        Mark a task as completed with results.
        
        Args:
            task_id: Task identifier
            result: Task execution result
        """
        redis = await self._connect()
        
        # Get task data
        task_data_raw = await redis.hgetall(f"task:{task_id}")
        if not task_data_raw:
            logger.warning(f"Task {task_id} not found in Redis")
            return
            
        # Deserialize task data
        task_data = self._deserialize_dict(task_data_raw)
        
        # Update task status
        task_data["status"] = "completed"
        task_data["completed_at"] = time.time()
        if "started_at" in task_data:
            task_data["execution_time"] = float(task_data["completed_at"]) - float(task_data["started_at"])
        
        # Store updated task data
        await redis.hset(f"task:{task_id}", mapping=self._serialize_dict(task_data))
        
        # Store result
        await redis.hset(f"result:{task_id}", mapping=self._serialize_dict(result))
        
        # Remove from processing set
        await redis.srem("processing_tasks", task_id)
        
        # Add to completed set
        await redis.sadd("completed_tasks", task_id)
        
        # Signal local completion if event exists
        if task_id in self.task_completion_events:
            self.task_completion_events[task_id].set()
        
        logger.debug(f"Task {task_id} completed in Redis queue")
    
    async def fail_task(self, task_id: str, error: str) -> None:
        """
        Mark a task as failed with error information.
        
        Args:
            task_id: Task identifier
            error: Error message
        """
        redis = await self._connect()
        
        # Get task data
        task_data_raw = await redis.hgetall(f"task:{task_id}")
        if not task_data_raw:
            logger.warning(f"Task {task_id} not found in Redis")
            return
            
        # Deserialize task data
        task_data = self._deserialize_dict(task_data_raw)
        
        # Check if we should retry
        if int(task_data.get("attempts", 0)) < self.max_retries:
            logger.info(f"Retrying task {task_id} (attempt {task_data.get('attempts', 0)})")
            
            # Reset task status
            task_data["status"] = "pending"
            
            # Store updated task data
            await redis.hset(f"task:{task_id}", mapping=self._serialize_dict(task_data))
            
            # Add back to pending queue
            await redis.lpush("pending_tasks", task_id)
            
            # Remove from processing set
            await redis.srem("processing_tasks", task_id)
        else:
            # Max retries reached, mark as failed
            task_data["status"] = "failed"
            task_data["failed_at"] = time.time()
            task_data["error"] = error
            
            # Store updated task data
            await redis.hset(f"task:{task_id}", mapping=self._serialize_dict(task_data))
            
            # Store error result
            await redis.hset(f"result:{task_id}", mapping=self._serialize_dict({"error": error}))
            
            # Remove from processing set
            await redis.srem("processing_tasks", task_id)
            
            # Add to failed set
            await redis.sadd("failed_tasks", task_id)
            
            # Signal local completion (with failure) if event exists
            if task_id in self.task_completion_events:
                self.task_completion_events[task_id].set()
            
            logger.warning(f"Task {task_id} failed after {task_data.get('attempts', 0)} attempts: {error}")
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the current status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status information
        """
        redis = await self._connect()
        
        # Get task data
        task_data_raw = await redis.hgetall(f"task:{task_id}")
        if not task_data_raw:
            return {"task_id": task_id, "status": "not_found"}
            
        # Deserialize task data
        task_data = self._deserialize_dict(task_data_raw)
        
        # Build status response
        status = {
            "task_id": task_id,
            "status": task_data.get("status", "unknown"),
            "created_at": float(task_data.get("created_at", 0)),
            "attempts": int(task_data.get("attempts", 0))
        }
        
        # Add status-specific fields
        if task_data.get("status") == "completed":
            status["completed_at"] = float(task_data.get("completed_at", 0))
            status["execution_time"] = float(task_data.get("execution_time", 0))
            
            # Get result
            result_raw = await redis.hgetall(f"result:{task_id}")
            status["result"] = self._deserialize_dict(result_raw) if result_raw else {}
            
        elif task_data.get("status") == "failed":
            status["failed_at"] = float(task_data.get("failed_at", 0))
            status["error"] = task_data.get("error", "Unknown error")
            
        elif task_data.get("status") == "processing":
            status["started_at"] = float(task_data.get("started_at", 0))
            
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
        
        # Check status of all tasks
        redis = await self._connect()
        for task_id in task_ids:
            # Check if task exists and is already completed
            task_status = await self.get_task_status(task_id)
            if task_status.get("status") in ["completed", "failed"]:
                # Already done, create and set event
                event = asyncio.Event()
                event.set()
                self.task_completion_events[task_id] = event
            elif task_id not in self.task_completion_events:
                # Create an event for this task
                self.task_completion_events[task_id] = asyncio.Event()
                
                # Subscribe to completion using Redis pubsub
                # (not implemented in this simplified version)
                pass
            
            events.append(self.task_completion_events[task_id].wait())
        
        # Wait for all events (with timeout if specified)
        try:
            await asyncio.wait_for(asyncio.gather(*events), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for Redis tasks: {task_ids}")
        
        # Collect results
        results = {}
        for task_id in task_ids:
            status = await self.get_task_status(task_id)
            if status.get("status") in ["completed", "failed"]:
                # Get result from status
                results[task_id] = status.get("result", {})
                if "error" in status:
                    results[task_id]["error"] = status["error"]
            else:
                # Task not completed or failed
                results[task_id] = {"error": f"Task not completed: {status['status']}"}
        
        return results
    
    def _serialize_dict(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Convert dictionary values to strings for Redis storage."""
        result = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                result[key] = json.dumps(value)
            else:
                result[key] = str(value)
        return result
    
    def _deserialize_dict(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Convert Redis string values back to Python types."""
        result = {}
        for key, value in data.items():
            if key in ["created_at", "started_at", "completed_at", "failed_at", "execution_time"]:
                try:
                    result[key] = float(value)
                except (ValueError, TypeError):
                    result[key] = value
            elif key in ["attempts"]:
                try:
                    result[key] = int(value)
                except (ValueError, TypeError):
                    result[key] = value
            else:
                try:
                    # Try to parse as JSON
                    result[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, use as-is
                    result[key] = value
        return result


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