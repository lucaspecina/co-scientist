"""
Asynchronous task execution framework for the Co-Scientist system.

This module implements a scalable, asynchronous task execution framework that allows
the system to flexibly allocate computational resources to scientific reasoning tasks.
It provides:
- Distributed task queuing
- Worker process management
- Task prioritization and scheduling
- Resource allocation and monitoring
"""

import asyncio
import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Union, TypeVar, Generic, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import traceback
import json
import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# Type for the result of a task
T = TypeVar('T')

class TaskStatus(Enum):
    """Status of a task in the system."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class Task(Generic[T]):
    """Represents a task in the asynchronous task system."""
    
    task_id: str
    agent_type: str
    function: str
    args: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    result: Optional[T] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    session_id: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    timeout_seconds: int = 600  # Default 10 minute timeout
    retry_count: int = 0
    max_retries: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the task to a dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "agent_type": self.agent_type,
            "function": self.function,
            "args": self.args,
            "status": self.status.value,
            "priority": self.priority.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "worker_id": self.worker_id,
            "session_id": self.session_id,
            "depends_on": self.depends_on,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create a task from a dictionary."""
        # Convert string datetimes back to datetime objects
        for dt_field in ['created_at', 'started_at', 'completed_at']:
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        
        # Convert string enums back to enum objects
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = TaskStatus(data['status'])
        
        if 'priority' in data and isinstance(data['priority'], int):
            data['priority'] = TaskPriority(data['priority'])
        
        return cls(**data)


class TaskQueue:
    """Base abstract class for task queues."""
    
    async def initialize(self) -> None:
        """Initialize the queue."""
        pass
    
    async def close(self) -> None:
        """Close the queue connections."""
        pass
    
    async def push_task(self, task: Task) -> bool:
        """Push a task to the queue.
        
        Args:
            task: The task to push
            
        Returns:
            Success status
        """
        raise NotImplementedError
    
    async def pop_task(self, worker_id: str) -> Optional[Task]:
        """Pop a task from the queue.
        
        Args:
            worker_id: ID of the worker requesting a task
            
        Returns:
            A task if available, otherwise None
        """
        raise NotImplementedError
    
    async def complete_task(self, task_id: str, result: Any) -> bool:
        """Mark a task as completed with a result.
        
        Args:
            task_id: ID of the completed task
            result: Result of the task
            
        Returns:
            Success status
        """
        raise NotImplementedError
    
    async def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a task as failed with an error.
        
        Args:
            task_id: ID of the failed task
            error: Error message
            
        Returns:
            Success status
        """
        raise NotImplementedError
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            Success status
        """
        raise NotImplementedError
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Status of the task if found, otherwise None
        """
        raise NotImplementedError
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            The task if found, otherwise None
        """
        raise NotImplementedError
    
    async def get_session_tasks(self, session_id: str) -> List[Task]:
        """Get all tasks for a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            List of tasks for the session
        """
        raise NotImplementedError
    
    async def get_pending_tasks_count(self) -> int:
        """Get the number of pending tasks.
        
        Returns:
            Number of pending tasks
        """
        raise NotImplementedError
    
    async def get_running_tasks_count(self) -> int:
        """Get the number of running tasks.
        
        Returns:
            Number of running tasks
        """
        raise NotImplementedError
    
    async def get_queue_statistics(self) -> Dict[str, Any]:
        """Get statistics about the queue.
        
        Returns:
            Dictionary with queue statistics
        """
        raise NotImplementedError


class InMemoryTaskQueue(TaskQueue):
    """In-memory implementation of the task queue for local testing."""
    
    def __init__(self):
        """Initialize the in-memory task queue."""
        self.tasks: Dict[str, Task] = {}
        self.pending_tasks: List[Task] = []
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        self.lock = asyncio.Lock()
    
    async def push_task(self, task: Task) -> bool:
        """Push a task to the queue."""
        async with self.lock:
            self.tasks[task.task_id] = task
            self.pending_tasks.append(task)
            # Sort by priority (lower value = higher priority)
            self.pending_tasks.sort(key=lambda t: t.priority.value)
            return True
    
    async def pop_task(self, worker_id: str) -> Optional[Task]:
        """Pop a task from the queue."""
        async with self.lock:
            # Check for dependency satisfaction
            ready_tasks = [
                t for t in self.pending_tasks 
                if all(dep_id not in self.pending_tasks and dep_id not in self.running_tasks 
                       for dep_id in t.depends_on)
            ]
            
            if not ready_tasks:
                return None
            
            # Get the highest priority task
            task = ready_tasks[0]
            self.pending_tasks.remove(task)
            
            # Mark as running
            task.status = TaskStatus.RUNNING
            task.worker_id = worker_id
            task.started_at = datetime.now()
            self.running_tasks[task.task_id] = task
            
            return task
    
    async def complete_task(self, task_id: str, result: Any) -> bool:
        """Mark a task as completed."""
        async with self.lock:
            if task_id not in self.running_tasks:
                return False
            
            task = self.running_tasks.pop(task_id)
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()
            self.completed_tasks[task_id] = task
            return True
    
    async def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a task as failed."""
        async with self.lock:
            if task_id in self.running_tasks:
                task = self.running_tasks.pop(task_id)
            elif task_id in self.pending_tasks:
                task = [t for t in self.pending_tasks if t.task_id == task_id][0]
                self.pending_tasks.remove(task)
            else:
                return False
            
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = datetime.now()
            
            # Check for retry
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.error = f"Retry {task.retry_count}/{task.max_retries}: {error}"
                self.pending_tasks.append(task)
                self.pending_tasks.sort(key=lambda t: t.priority.value)
                return True
            
            self.failed_tasks[task_id] = task
            return True
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        async with self.lock:
            # Find the task in pending tasks
            pending_task = None
            for task in self.pending_tasks:
                if task.task_id == task_id:
                    pending_task = task
                    break
            
            if pending_task:
                self.pending_tasks.remove(pending_task)
                pending_task.status = TaskStatus.CANCELED
                pending_task.completed_at = datetime.now()
                self.failed_tasks[task_id] = pending_task
                return True
            
            # Check if it's running
            if task_id in self.running_tasks:
                # Can't cancel running tasks directly
                return False
            
            return False
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return None
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    async def get_session_tasks(self, session_id: str) -> List[Task]:
        """Get all tasks for a session."""
        return [task for task in self.tasks.values() if task.session_id == session_id]
    
    async def get_pending_tasks_count(self) -> int:
        """Get the number of pending tasks."""
        return len(self.pending_tasks)
    
    async def get_running_tasks_count(self) -> int:
        """Get the number of running tasks."""
        return len(self.running_tasks)
    
    async def get_queue_statistics(self) -> Dict[str, Any]:
        """Get statistics about the queue."""
        return {
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "total_tasks": len(self.tasks),
            "tasks_by_priority": {
                priority.name: len([t for t in self.pending_tasks if t.priority == priority])
                for priority in TaskPriority
            },
            "tasks_by_agent": {
                agent_type: len([t for t in self.tasks.values() if t.agent_type == agent_type])
                for agent_type in set(t.agent_type for t in self.tasks.values())
            }
        }


class RedisTaskQueue(TaskQueue):
    """Redis-based implementation of the task queue for distributed execution."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize the Redis task queue.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis = None
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the Redis connection."""
        try:
            self.redis = redis.from_url(self.redis_url)
            # Test connection
            self.redis.ping()
            self.initialized = True
            logger.info(f"Connected to Redis at {self.redis_url}")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def close(self) -> None:
        """Close the Redis connection."""
        if self.redis:
            self.redis.close()
            logger.info("Closed Redis connection")
    
    def _ensure_initialized(self) -> None:
        """Ensure the queue is initialized."""
        if not self.initialized or not self.redis:
            raise RuntimeError("Redis task queue not initialized. Call initialize() first.")
    
    def _serialize_task(self, task: Task) -> str:
        """Serialize a task to JSON."""
        return json.dumps(task.to_dict())
    
    def _deserialize_task(self, task_json: str) -> Task:
        """Deserialize a task from JSON."""
        return Task.from_dict(json.loads(task_json))
    
    async def push_task(self, task: Task) -> bool:
        """Push a task to the queue."""
        self._ensure_initialized()
        
        try:
            # Store the task details
            task_key = f"task:{task.task_id}"
            task_json = self._serialize_task(task)
            
            # Use a Redis transaction
            pipeline = self.redis.pipeline()
            
            # Store the full task
            pipeline.set(task_key, task_json)
            
            # Add to the pending queue with priority as score
            pending_queue = "queue:pending"
            pipeline.zadd(pending_queue, {task.task_id: task.priority.value})
            
            # Add to session index if session_id is provided
            if task.session_id:
                session_key = f"session:{task.session_id}:tasks"
                pipeline.sadd(session_key, task.task_id)
            
            # Add to the task list
            pipeline.sadd("tasks:all", task.task_id)
            
            # Execute the transaction
            pipeline.execute()
            
            return True
        except RedisError as e:
            logger.error(f"Failed to push task {task.task_id} to Redis: {e}")
            return False
    
    async def pop_task(self, worker_id: str) -> Optional[Task]:
        """Pop a task from the queue."""
        self._ensure_initialized()
        
        try:
            pending_queue = "queue:pending"
            running_set = "tasks:running"
            
            # Get the highest priority task (lowest score)
            # We'll use a transaction to ensure atomicity
            pipeline = self.redis.pipeline()
            
            # Get the task with the lowest score (highest priority)
            task_ids_with_scores = self.redis.zrange(
                pending_queue, 0, 0, withscores=True, desc=False
            )
            
            if not task_ids_with_scores:
                return None
            
            task_id, score = task_ids_with_scores[0]
            task_id = task_id.decode('utf-8') if isinstance(task_id, bytes) else task_id
            
            # Check for dependencies
            task_json = self.redis.get(f"task:{task_id}")
            if not task_json:
                # Task data missing, clean up the queue entry
                self.redis.zrem(pending_queue, task_id)
                return None
            
            task = self._deserialize_task(task_json.decode('utf-8') if isinstance(task_json, bytes) else task_json)
            
            # Check if any dependencies are still pending or running
            for dep_id in task.depends_on:
                dep_status = await self.get_task_status(dep_id)
                if dep_status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    # Dependency not satisfied, skip this task for now
                    return None
            
            # Remove from pending queue
            pipeline.zrem(pending_queue, task_id)
            
            # Mark as running
            task.status = TaskStatus.RUNNING
            task.worker_id = worker_id
            task.started_at = datetime.now()
            
            # Update task data
            pipeline.set(f"task:{task_id}", self._serialize_task(task))
            
            # Add to running set
            pipeline.sadd(running_set, task_id)
            
            # Execute the transaction
            pipeline.execute()
            
            return task
        except RedisError as e:
            logger.error(f"Failed to pop task from Redis: {e}")
            return None
    
    async def complete_task(self, task_id: str, result: Any) -> bool:
        """Mark a task as completed."""
        self._ensure_initialized()
        
        try:
            task_key = f"task:{task_id}"
            running_set = "tasks:running"
            completed_set = "tasks:completed"
            
            # Get the task
            task_json = self.redis.get(task_key)
            if not task_json:
                return False
            
            task = self._deserialize_task(task_json.decode('utf-8') if isinstance(task_json, bytes) else task_json)
            
            # Update the task
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()
            
            # Use a transaction
            pipeline = self.redis.pipeline()
            
            # Update task data
            pipeline.set(task_key, self._serialize_task(task))
            
            # Remove from running set
            pipeline.srem(running_set, task_id)
            
            # Add to completed set
            pipeline.sadd(completed_set, task_id)
            
            # Execute the transaction
            pipeline.execute()
            
            return True
        except RedisError as e:
            logger.error(f"Failed to complete task {task_id} in Redis: {e}")
            return False
    
    async def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a task as failed."""
        self._ensure_initialized()
        
        try:
            task_key = f"task:{task_id}"
            running_set = "tasks:running"
            pending_queue = "queue:pending"
            failed_set = "tasks:failed"
            
            # Get the task
            task_json = self.redis.get(task_key)
            if not task_json:
                return False
            
            task = self._deserialize_task(task_json.decode('utf-8') if isinstance(task_json, bytes) else task_json)
            
            # Use a transaction
            pipeline = self.redis.pipeline()
            
            # Remove from running or pending
            if task.status == TaskStatus.RUNNING:
                pipeline.srem(running_set, task_id)
            elif task.status == TaskStatus.PENDING:
                pipeline.zrem(pending_queue, task_id)
            
            # Check for retry
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.error = f"Retry {task.retry_count}/{task.max_retries}: {error}"
                
                # Update task data
                pipeline.set(task_key, self._serialize_task(task))
                
                # Add back to pending queue
                pipeline.zadd(pending_queue, {task_id: task.priority.value})
            else:
                # Mark as failed
                task.status = TaskStatus.FAILED
                task.error = error
                task.completed_at = datetime.now()
                
                # Update task data
                pipeline.set(task_key, self._serialize_task(task))
                
                # Add to failed set
                pipeline.sadd(failed_set, task_id)
            
            # Execute the transaction
            pipeline.execute()
            
            return True
        except RedisError as e:
            logger.error(f"Failed to fail task {task_id} in Redis: {e}")
            return False
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        self._ensure_initialized()
        
        try:
            task_key = f"task:{task_id}"
            pending_queue = "queue:pending"
            failed_set = "tasks:failed"
            
            # Get the task
            task_json = self.redis.get(task_key)
            if not task_json:
                return False
            
            task = self._deserialize_task(task_json.decode('utf-8') if isinstance(task_json, bytes) else task_json)
            
            # Can only cancel pending tasks
            if task.status != TaskStatus.PENDING:
                return False
            
            # Use a transaction
            pipeline = self.redis.pipeline()
            
            # Remove from pending queue
            pipeline.zrem(pending_queue, task_id)
            
            # Mark as canceled
            task.status = TaskStatus.CANCELED
            task.completed_at = datetime.now()
            
            # Update task data
            pipeline.set(task_key, self._serialize_task(task))
            
            # Add to failed set
            pipeline.sadd(failed_set, task_id)
            
            # Execute the transaction
            pipeline.execute()
            
            return True
        except RedisError as e:
            logger.error(f"Failed to cancel task {task_id} in Redis: {e}")
            return False
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        self._ensure_initialized()
        
        try:
            task_key = f"task:{task_id}"
            
            # Get the task
            task_json = self.redis.get(task_key)
            if not task_json:
                return None
            
            task = self._deserialize_task(task_json.decode('utf-8') if isinstance(task_json, bytes) else task_json)
            return task.status
        except RedisError as e:
            logger.error(f"Failed to get status for task {task_id} from Redis: {e}")
            return None
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        self._ensure_initialized()
        
        try:
            task_key = f"task:{task_id}"
            
            # Get the task
            task_json = self.redis.get(task_key)
            if not task_json:
                return None
            
            return self._deserialize_task(task_json.decode('utf-8') if isinstance(task_json, bytes) else task_json)
        except RedisError as e:
            logger.error(f"Failed to get task {task_id} from Redis: {e}")
            return None
    
    async def get_session_tasks(self, session_id: str) -> List[Task]:
        """Get all tasks for a session."""
        self._ensure_initialized()
        
        try:
            session_key = f"session:{session_id}:tasks"
            
            # Get all task IDs for the session
            task_ids = self.redis.smembers(session_key)
            task_ids = [tid.decode('utf-8') if isinstance(tid, bytes) else tid for tid in task_ids]
            
            tasks = []
            for task_id in task_ids:
                task = await self.get_task(task_id)
                if task:
                    tasks.append(task)
            
            return tasks
        except RedisError as e:
            logger.error(f"Failed to get tasks for session {session_id} from Redis: {e}")
            return []
    
    async def get_pending_tasks_count(self) -> int:
        """Get the number of pending tasks."""
        self._ensure_initialized()
        
        try:
            pending_queue = "queue:pending"
            return self.redis.zcard(pending_queue)
        except RedisError as e:
            logger.error(f"Failed to get pending tasks count from Redis: {e}")
            return 0
    
    async def get_running_tasks_count(self) -> int:
        """Get the number of running tasks."""
        self._ensure_initialized()
        
        try:
            running_set = "tasks:running"
            return self.redis.scard(running_set)
        except RedisError as e:
            logger.error(f"Failed to get running tasks count from Redis: {e}")
            return 0
    
    async def get_queue_statistics(self) -> Dict[str, Any]:
        """Get statistics about the queue."""
        self._ensure_initialized()
        
        try:
            all_tasks_set = "tasks:all"
            pending_queue = "queue:pending"
            running_set = "tasks:running"
            completed_set = "tasks:completed"
            failed_set = "tasks:failed"
            
            # Use a pipeline to get all stats in one go
            pipeline = self.redis.pipeline()
            pipeline.scard(all_tasks_set)
            pipeline.zcard(pending_queue)
            pipeline.scard(running_set)
            pipeline.scard(completed_set)
            pipeline.scard(failed_set)
            
            # Execute the pipeline
            total, pending, running, completed, failed = pipeline.execute()
            
            # Get the distribution by priority (more complex, needs multiple commands)
            priorities = {}
            for priority in TaskPriority:
                min_score = priority.value
                max_score = priority.value
                count = self.redis.zcount(pending_queue, min_score, max_score)
                priorities[priority.name] = count
            
            # Getting agent distribution would require loading all tasks, which could be expensive
            # We'll skip that for now
            
            return {
                "total_tasks": total,
                "pending_tasks": pending,
                "running_tasks": running,
                "completed_tasks": completed,
                "failed_tasks": failed,
                "tasks_by_priority": priorities
            }
        except RedisError as e:
            logger.error(f"Failed to get queue statistics from Redis: {e}")
            return {
                "error": str(e),
                "pending_tasks": 0,
                "running_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "total_tasks": 0
            }


class TaskWorker:
    """Worker process that executes tasks from the queue."""
    
    def __init__(
        self,
        worker_id: str,
        queue: TaskQueue,
        agent_registry: Dict[str, Any],
        max_concurrent_tasks: int = 3,
        poll_interval: float = 1.0,
        shutdown_timeout: float = 30.0
    ):
        """Initialize a task worker.
        
        Args:
            worker_id: Unique identifier for this worker
            queue: Task queue to pull tasks from
            agent_registry: Registry of agent implementations
            max_concurrent_tasks: Maximum number of tasks to execute concurrently
            poll_interval: Interval in seconds to poll for new tasks
            shutdown_timeout: Timeout in seconds for graceful shutdown
        """
        self.worker_id = worker_id
        self.queue = queue
        self.agent_registry = agent_registry
        self.max_concurrent_tasks = max_concurrent_tasks
        self.poll_interval = poll_interval
        self.shutdown_timeout = shutdown_timeout
        
        self.running = False
        self.tasks: Dict[str, asyncio.Task] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "start_time": None,
            "agent_stats": {}
        }
    
    async def start(self) -> None:
        """Start the worker process."""
        if self.running:
            return
        
        self.running = True
        self.stats["start_time"] = datetime.now()
        logger.info(f"Starting task worker {self.worker_id}")
        
        # Start the main loop
        asyncio.create_task(self._worker_loop())
    
    async def stop(self) -> None:
        """Stop the worker process."""
        if not self.running:
            return
        
        logger.info(f"Stopping task worker {self.worker_id}, waiting for {len(self.tasks)} tasks to complete")
        self.running = False
        
        # Wait for all tasks to complete with timeout
        if self.tasks:
            done, pending = await asyncio.wait(
                self.tasks.values(), 
                timeout=self.shutdown_timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
            
            logger.info(f"Worker {self.worker_id} shutdown complete, {len(done)} tasks completed, {len(pending)} tasks canceled")
    
    async def _worker_loop(self) -> None:
        """Main worker loop that polls for and executes tasks."""
        while self.running:
            # Check if we have capacity for more tasks
            if len(self.tasks) >= self.max_concurrent_tasks:
                # Wait a bit before checking again
                await asyncio.sleep(self.poll_interval)
                continue
            
            # Try to get a task
            task = await self.queue.pop_task(self.worker_id)
            if not task:
                # No tasks available, wait before polling again
                await asyncio.sleep(self.poll_interval)
                continue
            
            # Got a task, execute it
            logger.info(f"Worker {self.worker_id} received task {task.task_id} of type {task.agent_type}.{task.function}")
            
            # Start a new task executor
            executor = asyncio.create_task(self._execute_task(task))
            self.tasks[task.task_id] = executor
            
            # Set up a callback to handle completion
            executor.add_done_callback(lambda _: self.tasks.pop(task.task_id, None))
            
            # Small delay to prevent overloading
            await asyncio.sleep(0.1)
    
    async def _execute_task(self, task: Task) -> None:
        """Execute a single task."""
        start_time = time.time()
        
        try:
            # Get the agent from the registry
            agent_class = self.agent_registry.get(task.agent_type)
            if not agent_class:
                raise ValueError(f"Unknown agent type: {task.agent_type}")
            
            # Create an instance of the agent
            agent = agent_class()
            
            # Check if the agent has the requested function
            if not hasattr(agent, task.function) or not callable(getattr(agent, task.function)):
                raise ValueError(f"Agent {task.agent_type} does not have function {task.function}")
            
            # Get the method
            method = getattr(agent, task.function)
            
            # Execute the method with the provided arguments
            async with self.semaphore:
                if asyncio.iscoroutinefunction(method):
                    result = await method(**task.args)
                else:
                    # Run non-async functions in a thread pool
                    result = await asyncio.to_thread(method, **task.args)
            
            # Record success
            execution_time = time.time() - start_time
            await self.queue.complete_task(task.task_id, result)
            
            # Update stats
            self.stats["tasks_completed"] += 1
            self.stats["total_execution_time"] += execution_time
            
            if task.agent_type not in self.stats["agent_stats"]:
                self.stats["agent_stats"][task.agent_type] = {
                    "completed": 0, "failed": 0, "total_time": 0.0
                }
            
            self.stats["agent_stats"][task.agent_type]["completed"] += 1
            self.stats["agent_stats"][task.agent_type]["total_time"] += execution_time
            
            logger.info(f"Worker {self.worker_id} completed task {task.task_id} in {execution_time:.2f}s")
            
        except Exception as e:
            # Record the failure
            execution_time = time.time() - start_time
            error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            await self.queue.fail_task(task.task_id, error_message)
            
            # Update stats
            self.stats["tasks_failed"] += 1
            
            if task.agent_type not in self.stats["agent_stats"]:
                self.stats["agent_stats"][task.agent_type] = {
                    "completed": 0, "failed": 0, "total_time": 0.0
                }
            
            self.stats["agent_stats"][task.agent_type]["failed"] += 1
            
            logger.error(f"Worker {self.worker_id} failed task {task.task_id}: {error_message}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        uptime = 0
        if self.stats["start_time"]:
            uptime = (datetime.now() - self.stats["start_time"]).total_seconds()
        
        return {
            "worker_id": self.worker_id,
            "running": self.running,
            "current_tasks": len(self.tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "tasks_completed": self.stats["tasks_completed"],
            "tasks_failed": self.stats["tasks_failed"],
            "total_execution_time": self.stats["total_execution_time"],
            "uptime": uptime,
            "agent_stats": self.stats["agent_stats"]
        }


class TaskManager:
    """Manages the asynchronous task execution framework."""
    
    def __init__(
        self,
        agent_registry: Dict[str, Any],
        use_redis: bool = False,
        redis_url: str = "redis://localhost:6379/0",
        num_workers: int = 3,
        max_tasks_per_worker: int = 3
    ):
        """Initialize the task manager.
        
        Args:
            agent_registry: Registry of agent implementations
            use_redis: Whether to use Redis for task queuing
            redis_url: Redis connection URL if using Redis
            num_workers: Number of worker processes to start
            max_tasks_per_worker: Maximum number of tasks per worker
        """
        self.agent_registry = agent_registry
        self.use_redis = use_redis
        self.redis_url = redis_url
        self.num_workers = num_workers
        self.max_tasks_per_worker = max_tasks_per_worker
        
        # Initialize the appropriate queue
        self.queue: TaskQueue = RedisTaskQueue(redis_url) if use_redis else InMemoryTaskQueue()
        
        # Initialize worker pool
        self.workers: Dict[str, TaskWorker] = {}
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the task manager."""
        if self.initialized:
            return
        
        # Initialize the queue
        await self.queue.initialize()
        
        # Start workers
        for i in range(self.num_workers):
            worker_id = f"worker-{uuid.uuid4()}"
            worker = TaskWorker(
                worker_id=worker_id,
                queue=self.queue,
                agent_registry=self.agent_registry,
                max_concurrent_tasks=self.max_tasks_per_worker
            )
            await worker.start()
            self.workers[worker_id] = worker
        
        self.initialized = True
        logger.info(f"Task manager initialized with {self.num_workers} workers")
    
    async def shutdown(self) -> None:
        """Shut down the task manager."""
        if not self.initialized:
            return
        
        # Stop all workers
        for worker in self.workers.values():
            await worker.stop()
        
        # Close the queue
        await self.queue.close()
        
        self.initialized = False
        logger.info("Task manager shut down")
    
    def _ensure_initialized(self) -> None:
        """Ensure the task manager is initialized."""
        if not self.initialized:
            raise RuntimeError("Task manager not initialized. Call initialize() first.")
    
    async def submit_task(
        self,
        agent_type: str,
        function: str,
        args: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM,
        session_id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        timeout_seconds: int = 600,
        max_retries: int = 2,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit a task to the queue.
        
        Args:
            agent_type: Type of agent to execute the task
            function: Function to call on the agent
            args: Arguments to pass to the function
            priority: Priority of the task
            session_id: Optional session ID to associate with the task
            depends_on: Optional list of task IDs this task depends on
            timeout_seconds: Timeout in seconds for the task
            max_retries: Maximum number of retries if the task fails
            metadata: Optional metadata to associate with the task
            
        Returns:
            The ID of the submitted task
        """
        self._ensure_initialized()
        
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            agent_type=agent_type,
            function=function,
            args=args,
            priority=priority,
            session_id=session_id,
            depends_on=depends_on or [],
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            metadata=metadata or {}
        )
        
        success = await self.queue.push_task(task)
        if not success:
            raise RuntimeError(f"Failed to submit task {task_id}")
        
        logger.info(f"Submitted task {task_id} to queue: {agent_type}.{function}")
        return task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            Success status
        """
        self._ensure_initialized()
        return await self.queue.cancel_task(task_id)
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Status of the task if found, otherwise None
        """
        self._ensure_initialized()
        return await self.queue.get_task_status(task_id)
    
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get the result of a completed task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Result of the task if completed, otherwise None
        """
        self._ensure_initialized()
        task = await self.queue.get_task(task_id)
        if task and task.status == TaskStatus.COMPLETED:
            return task.result
        return None
    
    async def get_session_tasks(self, session_id: str) -> List[Task]:
        """Get all tasks for a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            List of tasks for the session
        """
        self._ensure_initialized()
        return await self.queue.get_session_tasks(session_id)
    
    async def get_queue_statistics(self) -> Dict[str, Any]:
        """Get statistics about the queue and workers.
        
        Returns:
            Dictionary with queue and worker statistics
        """
        self._ensure_initialized()
        
        queue_stats = await self.queue.get_queue_statistics()
        worker_stats = {worker_id: worker.get_stats() for worker_id, worker in self.workers.items()}
        
        return {
            "queue": queue_stats,
            "workers": worker_stats,
            "total_workers": len(self.workers),
            "active_workers": sum(1 for worker in self.workers.values() if worker.running),
        }
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[Any]:
        """Wait for a task to complete.
        
        Args:
            task_id: ID of the task to wait for
            timeout: Optional timeout in seconds
            
        Returns:
            Result of the task if completed within the timeout, otherwise None
        """
        self._ensure_initialized()
        
        start_time = time.time()
        poll_interval = 0.5  # Start with a short interval
        
        while True:
            task = await self.queue.get_task(task_id)
            
            if not task:
                # Task not found
                return None
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            
            if task.status in [TaskStatus.FAILED, TaskStatus.CANCELED, TaskStatus.TIMEOUT]:
                # Task completed but not successfully
                logger.warning(f"Task {task_id} finished with status {task.status}: {task.error}")
                return None
            
            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                logger.warning(f"Timeout waiting for task {task_id}")
                return None
            
            # Wait before polling again, with exponential backoff up to a limit
            await asyncio.sleep(min(poll_interval, 5.0))
            poll_interval *= 1.5
    
    async def submit_and_wait(
        self,
        agent_type: str,
        function: str,
        args: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM,
        session_id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        timeout_seconds: int = 600,
        max_retries: int = 2,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Submit a task and wait for its completion.
        
        This is a convenience method that combines submit_task and wait_for_task.
        
        Args:
            agent_type: Type of agent to execute the task
            function: Function to call on the agent
            args: Arguments to pass to the function
            priority: Priority of the task
            session_id: Optional session ID to associate with the task
            depends_on: Optional list of task IDs this task depends on
            timeout_seconds: Timeout in seconds for the task
            max_retries: Maximum number of retries if the task fails
            metadata: Optional metadata to associate with the task
            
        Returns:
            Result of the task if completed successfully, otherwise None
        """
        task_id = await self.submit_task(
            agent_type=agent_type,
            function=function,
            args=args,
            priority=priority,
            session_id=session_id,
            depends_on=depends_on,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            metadata=metadata
        )
        
        return await self.wait_for_task(task_id, timeout=timeout_seconds)