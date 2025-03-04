#!/usr/bin/env python3
"""
Example of running and interacting with the AI Co-Scientist API server.

This example demonstrates how to:
1. Start the API server
2. Make requests to the API endpoints
3. Process and display the responses
"""

import asyncio
import os
import sys
import logging
import yaml
import json
import requests
import time
import subprocess
from urllib.parse import urljoin

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def start_api_server(host="127.0.0.1", port=8000, config_path=None):
    """
    Start the API server in a separate process.
    
    Args:
        host (str): The host to bind the server to
        port (int): The port to bind the server to
        config_path (str, optional): Path to a custom configuration file
    
    Returns:
        subprocess.Popen: The server process
    """
    # Determine the path to main.py
    main_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "main.py"
    )
    
    # Build the command
    cmd = [sys.executable, main_path, "api", "--host", host, "--port", str(port)]
    
    if config_path:
        cmd.extend(["--config", config_path])
    
    logger.info(f"Starting API server with command: {' '.join(cmd)}")
    
    # Start the server process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the server to start
    time.sleep(2)
    
    # Check if the server started successfully
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        logger.error(f"Failed to start API server: {stderr}")
        raise RuntimeError(f"Failed to start API server: {stderr}")
    
    logger.info(f"API server started on http://{host}:{port}")
    return process

def make_api_request(endpoint, method="GET", data=None, base_url="http://127.0.0.1:8000"):
    """
    Make a request to the API server.
    
    Args:
        endpoint (str): The API endpoint to request
        method (str): The HTTP method to use
        data (dict, optional): The data to send with the request
        base_url (str): The base URL of the API server
    
    Returns:
        dict: The response from the API server
    """
    url = urljoin(base_url, endpoint)
    logger.info(f"Making {method} request to {url}")
    
    if method == "GET":
        response = requests.get(url)
    elif method == "POST":
        response = requests.post(url, json=data)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")
    
    response.raise_for_status()
    return response.json()

def example_api_usage(host="127.0.0.1", port=8000):
    """
    Example of using the API server.
    
    Args:
        host (str): The host the server is running on
        port (int): The port the server is running on
    """
    base_url = f"http://{host}:{port}"
    
    # Check server status
    status = make_api_request("status", base_url=base_url)
    print("\nServer Status:")
    print(json.dumps(status, indent=2))
    
    # Generate hypotheses
    print("\nGenerating hypotheses...")
    generation_data = {
        "research_goal": "Identify novel drug targets for Alzheimer's disease",
        "domain": "neuroscience",
        "model_provider": "openai"
    }
    
    hypotheses = make_api_request(
        "generate", 
        method="POST", 
        data=generation_data,
        base_url=base_url
    )
    
    print("\nGenerated Hypotheses:")
    print(json.dumps(hypotheses, indent=2))
    
    # Start a research session
    print("\nStarting research session...")
    session_data = {
        "research_goal": "Identify novel drug targets for Alzheimer's disease",
        "domain": "neuroscience"
    }
    
    session = make_api_request(
        "sessions/start", 
        method="POST", 
        data=session_data,
        base_url=base_url
    )
    
    session_id = session.get("session_id")
    print(f"\nSession started with ID: {session_id}")
    
    # Poll session status
    print("\nPolling session status...")
    for _ in range(5):
        time.sleep(2)
        status = make_api_request(
            f"sessions/{session_id}/status",
            base_url=base_url
        )
        print(f"Session status: {status.get('status')}")
        
        if status.get("status") in ["completed", "error"]:
            break
    
    # Get session results
    print("\nGetting session results...")
    results = make_api_request(
        f"sessions/{session_id}/results",
        base_url=base_url
    )
    
    print("\nSession Results:")
    print(json.dumps(results, indent=2))

def main():
    """Main entry point for the example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run and interact with the API server")
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to a custom configuration file"
    )
    
    args = parser.parse_args()
    
    # Start the API server
    server_process = start_api_server(
        host=args.host,
        port=args.port,
        config_path=args.config
    )
    
    try:
        # Run the example
        example_api_usage(host=args.host, port=args.port)
    finally:
        # Stop the server
        logger.info("Stopping API server")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main() 