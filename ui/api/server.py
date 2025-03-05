"""
FastAPI Server for AI Co-Scientist

This module implements the API endpoints for interacting with the AI Co-Scientist system.
"""

import asyncio
import json
import logging
import os
import yaml
from typing import Dict, List, Optional, Any, Union

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.models.model_factory import ModelFactory
from core.agents.base_agent import BaseAgent
from core.agents.generation_agent import GenerationAgent
from core.memory.memory_manager import create_memory_manager
from framework.queue.task_queue import create_task_queue
from framework.supervisor.supervisor import Supervisor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Co-Scientist API",
    description="API for the AI Co-Scientist system",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for system components
config = {}
memory_manager = None
task_queue = None
supervisor = None
agents = {}
connected_websockets = set()


# ---------- Pydantic Models for Request/Response ----------

class ResearchGoalRequest(BaseModel):
    """Request model for starting a research session."""
    
    research_goal: str = Field(..., description="The scientific question or goal")
    domain: str = Field("general science", description="Scientific domain")
    constraints: Optional[str] = Field(None, description="Optional constraints")
    prior_knowledge: Optional[str] = Field(None, description="Optional prior knowledge")
    model_provider: Optional[str] = Field(None, description="Optional model provider override")


class SessionResponse(BaseModel):
    """Response model for session information."""
    
    session_id: str = Field(..., description="Unique session identifier")
    research_goal: str = Field(..., description="The scientific question or goal")
    status: str = Field(..., description="Current session status")
    start_time: float = Field(..., description="Session start timestamp")
    current_iteration: int = Field(0, description="Current iteration number")
    hypothesis_count: int = Field(0, description="Number of hypotheses")


class HypothesisResponse(BaseModel):
    """Response model for hypothesis information."""
    
    id: str = Field(..., description="Unique hypothesis identifier")
    title: str = Field(..., description="Hypothesis title")
    description: str = Field(..., description="Hypothesis description")
    mechanism: Optional[str] = Field(None, description="Proposed mechanism")
    rank: Optional[float] = Field(None, description="Ranking score")
    critique: Optional[Dict[str, Any]] = Field(None, description="Critique information")


class ModelProviderRequest(BaseModel):
    """Request model for changing the model provider."""
    
    provider: str = Field(..., description="Model provider name")
    api_key: Optional[str] = Field(None, description="API key for the provider")


# ---------- API Endpoints ----------

@app.get("/")
async def root():
    """Root endpoint that returns basic system information."""
    return {
        "name": "AI Co-Scientist",
        "version": "0.1.0",
        "status": "running" if supervisor else "initializing"
    }


@app.get("/config")
async def get_config():
    """Get the current system configuration (sensitive data redacted)."""
    # Create a copy of the config and redact sensitive information
    safe_config = {}
    
    for section, settings in config.items():
        if isinstance(settings, dict):
            safe_config[section] = {}
            for key, value in settings.items():
                if key in ["api_key", "password", "secret"]:
                    safe_config[section][key] = "********"
                elif isinstance(value, dict):
                    safe_config[section][key] = {}
                    for subkey, subvalue in value.items():
                        if subkey in ["api_key", "password", "secret"]:
                            safe_config[section][key][subkey] = "********"
                        else:
                            safe_config[section][key][subkey] = subvalue
                else:
                    safe_config[section][key] = value
        else:
            safe_config[section] = settings
            
    return safe_config


@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: ResearchGoalRequest):
    """
    Create a new research session.
    
    Args:
        request: Research goal and related information
        
    Returns:
        Session information
    """
    # Check if system is initialized
    if not supervisor:
        raise HTTPException(status_code=503, detail="System not yet initialized")
    
    # Override model provider if specified
    if request.model_provider:
        temp_config = config.copy()
        temp_config["models"]["default_provider"] = request.model_provider
        
        # Reinitialize agents with new config
        # This is simplified - in a real system you'd handle this more efficiently
        await initialize_system(temp_config)
    
    # Start session
    session_id = await supervisor.start_session(
        research_goal=request.research_goal,
        domain=request.domain,
        constraints=request.constraints or "",
        prior_knowledge=request.prior_knowledge or "",
        callback=session_update_callback
    )
    
    # Get initial session data
    session = await supervisor.get_session(session_id)
    
    return {
        "session_id": session_id,
        "research_goal": session["research_goal"],
        "status": session["status"],
        "start_time": session["start_time"],
        "current_iteration": session.get("current_iteration", 0),
        "hypothesis_count": len(session.get("hypotheses", []))
    }


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    Get information about a specific session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session information
    """
    # Check if system is initialized
    if not supervisor:
        raise HTTPException(status_code=503, detail="System not yet initialized")
    
    # Get session data
    session = await supervisor.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "research_goal": session["research_goal"],
        "status": session["status"],
        "start_time": session["start_time"],
        "current_iteration": session.get("current_iteration", 0),
        "hypothesis_count": len(session.get("hypotheses", []))
    }


@app.get("/sessions/{session_id}/hypotheses")
async def get_hypotheses(session_id: str):
    """
    Get hypotheses for a specific session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        List of hypotheses
    """
    # Check if system is initialized
    if not supervisor:
        raise HTTPException(status_code=503, detail="System not yet initialized")
    
    # Get session data
    session = await supervisor.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get hypotheses
    hypotheses = session.get("hypotheses", [])
    
    return {"hypotheses": hypotheses}


@app.get("/sessions/{session_id}/meta_review")
async def get_meta_review(session_id: str):
    """
    Get meta-review for a specific session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Meta-review information
    """
    # Check if system is initialized
    if not supervisor:
        raise HTTPException(status_code=503, detail="System not yet initialized")
    
    # Get session data
    session = await supervisor.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get meta-review
    meta_review = session.get("meta_review", {})
    
    return {"meta_review": meta_review}


@app.post("/model/provider")
async def set_model_provider(request: ModelProviderRequest):
    """
    Change the default model provider.
    
    Args:
        request: Model provider information
        
    Returns:
        Success message
    """
    # Check if provider is supported
    supported_providers = ["openai", "anthropic", "google", "huggingface", "local"]
    if request.provider not in supported_providers:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported provider. Supported providers: {', '.join(supported_providers)}"
        )
    
    # Update config
    global config
    config["models"]["default_provider"] = request.provider
    
    # Set API key if provided
    if request.api_key:
        config["models"][request.provider]["api_key"] = request.api_key
    
    # Reinitialize system with new config
    await initialize_system(config)
    
    return {"message": f"Model provider changed to {request.provider}"}


@app.websocket("/ws/sessions/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time session updates.
    
    Args:
        websocket: WebSocket connection
        session_id: Session identifier
    """
    await websocket.accept()
    connected_websockets.add(websocket)
    
    try:
        # Send initial session data
        if supervisor:
            session = await supervisor.get_session(session_id)
            if session:
                await websocket.send_json(session)
        
        # Keep connection open until client disconnects
        while True:
            # Wait for client messages (or disconnection)
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        connected_websockets.remove(websocket)


# ---------- System Initialization ----------

async def initialize_system(system_config: Dict[str, Any] = None):
    """
    Initialize the AI Co-Scientist system components.
    
    Args:
        system_config: Configuration dictionary (loads from file if None)
    """
    global config, memory_manager, task_queue, supervisor, agents
    
    # Load configuration if not provided
    if not system_config:
        config_path = os.environ.get("CONFIG_PATH", "config/default_config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = system_config
    
    # Initialize memory manager
    memory_manager = create_memory_manager(config)
    
    # Initialize task queue
    task_queue = create_task_queue(config)
    
    # Initialize agents
    agents = await initialize_agents(config)
    
    # Initialize supervisor
    supervisor = Supervisor.from_config(config, agents, memory_manager, task_queue)
    
    logger.info("System initialized successfully")


async def initialize_agents(config: Dict[str, Any]) -> Dict[str, List[BaseAgent]]:
    """
    Initialize agent instances based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping agent types to lists of agent instances
    """
    agents = {}
    
    # Process each agent type
    for agent_type, agent_config in config.get("agents", {}).items():
        count = agent_config.get("count", 1)
        agents[agent_type] = []
        
        # Create the specified number of instances
        for i in range(count):
            # Get a model for this agent
            model = ModelFactory.create_model_for_agent(agent_type, config)
            
            # Create agent instance (example for generation agent)
            if agent_type == "generation":
                agent = GenerationAgent.from_config(agent_config, model)
                agents[agent_type].append(agent)
            # Add other agent types here...
            else:
                logger.warning(f"Agent type {agent_type} not implemented yet")
    
    return agents


def session_update_callback(session: Dict[str, Any]):
    """
    Callback function for session updates.
    
    Args:
        session: Updated session data
    """
    # Broadcast update to connected WebSocket clients
    for websocket in connected_websockets:
        asyncio.create_task(websocket.send_json(session))


# ---------- Application Startup ----------

@app.on_event("startup")
async def startup_event():
    """Run initialization tasks on application startup."""
    try:
        await initialize_system()
    except Exception as e:
        logger.error(f"Error initializing system: {e}")


# ---------- Main Entry Point ----------

def main():
    """Main entry point for running the server directly."""
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main() 