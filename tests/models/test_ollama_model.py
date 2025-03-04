"""
Test module for Ollama model integration.
"""

import asyncio
import json
import unittest
from unittest.mock import patch, MagicMock

import pytest

from core.models.ollama_model import OllamaModel


class TestOllamaModel(unittest.TestCase):
    """Test cases for the Ollama model adapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "model_name": "llama3",
            "api_base": "http://localhost:11434",
            "temperature": 0.7,
            "max_tokens": 2048,
            "timeout": 60
        }
        self.model = OllamaModel.from_config(self.config)

    def test_init(self):
        """Test initialization from config."""
        self.assertEqual(self.model.model_name, "llama3")
        self.assertEqual(self.model.api_base, "http://localhost:11434")
        self.assertEqual(self.model.temperature, 0.7)
        self.assertEqual(self.model.max_tokens, 2048)
        self.assertEqual(self.model.timeout, 60)

    @patch("aiohttp.ClientSession.post")
    async def test_generate(self, mock_post):
        """Test generate method."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {"response": "Test response from Ollama"}
        mock_post.return_value.__aenter__.return_value = mock_response

        # Call generate
        response = await self.model.generate("Test prompt")
        
        # Assertions
        self.assertEqual(response, "Test response from Ollama")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://localhost:11434/api/generate")
        self.assertEqual(json.loads(kwargs["json"]["prompt"]), "Test prompt")
        self.assertEqual(kwargs["json"]["model"], "llama3")

    @patch("aiohttp.ClientSession.post")
    async def test_generate_error(self, mock_post):
        """Test generate method with error response."""
        # Setup mock error response
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal Server Error"
        mock_post.return_value.__aenter__.return_value = mock_response

        # Call generate and expect empty response
        response = await self.model.generate("Test prompt")
        self.assertEqual(response, "")

    @patch("aiohttp.ClientSession.post")
    async def test_generate_json(self, mock_post):
        """Test generate_json method."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {"response": '{"key": "value"}'}
        mock_post.return_value.__aenter__.return_value = mock_response

        # Call generate_json
        result = await self.model.generate_json("Generate JSON", schema={})
        
        # Assertions
        self.assertEqual(result, {"key": "value"})

    @patch("aiohttp.ClientSession.post")
    async def test_generate_json_invalid(self, mock_post):
        """Test generate_json method with invalid JSON response."""
        # Setup mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {"response": "Not valid JSON"}
        mock_post.return_value.__aenter__.return_value = mock_response

        # Call generate_json and expect default value
        result = await self.model.generate_json("Generate JSON", schema={}, default={})
        self.assertEqual(result, {})

    @patch("aiohttp.ClientSession.post")
    async def test_embed(self, mock_post):
        """Test embed method."""
        # Setup mock response
        embedding_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {"embedding": embedding_values}
        mock_post.return_value.__aenter__.return_value = mock_response

        # Call embed
        result = await self.model.embed("Test text")
        
        # Assertions
        self.assertEqual(result, embedding_values)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://localhost:11434/api/embeddings")
        self.assertEqual(kwargs["json"]["prompt"], "Test text")
        self.assertEqual(kwargs["json"]["model"], "llama3")


# Run the tests with pytest
if __name__ == "__main__":
    pytest.main(["-xvs", "test_ollama_model.py"]) 