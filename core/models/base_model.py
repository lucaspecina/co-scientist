"""
Base Model Interface for AI Co-Scientist

This module defines the abstract base class for all models.
"""

import abc
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Base exception for model errors."""
    pass

class RateLimitError(ModelError):
    """Exception raised when a model provider rate limits the request."""
    pass

class TokenLimitError(ModelError):
    """Exception raised when a request exceeds token limits."""
    pass

class AuthenticationError(ModelError):
    """Exception raised when authentication fails."""
    pass

class ServiceUnavailableError(ModelError):
    """Exception raised when the model service is unavailable."""
    pass

class BaseModel(abc.ABC):
    """Abstract base class for all model implementations."""
    
    def __init__(self, **kwargs):
        """Initialize the model.
        
        Args:
            **kwargs: Model-specific initialization parameters
        """
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
        
        # Optional callback for tracking model performance
        self._on_completion: Optional[Callable] = None
    
    @abc.abstractmethod
    async def generate(self, 
                       prompt: str, 
                       system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       stop_sequences: Optional[List[str]] = None,
                       **kwargs) -> str:
        """Generate text from the model.
        
        Args:
            prompt: User prompt to generate from
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of strings that will stop generation if encountered
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text
            
        Raises:
            ModelError: If generation fails
        """
        pass
    
    @abc.abstractmethod
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        pass
    
    @abc.abstractmethod
    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for the input text.
        
        Args:
            text: Text or list of texts to embed
            
        Returns:
            Embedding vector(s) as a list of floats or list of list of floats
            
        Raises:
            ModelError: If embedding generation fails
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseModel':
        """Create a model instance from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured model instance
        """
        pass
    
    def set_completion_callback(self, callback: Callable) -> None:
        """Set a callback to be called on completion of model calls.
        
        Args:
            callback: Function to call with statistics after each model call
        """
        self._on_completion = callback
    
    async def _timed_generate(self,
                             func: Callable,
                             *args,
                             **kwargs) -> Any:
        """
        Execute a generation function with timing and error tracking.
        
        Args:
            func: The async function to call
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
            
        Raises:
            ModelError: If the function call fails
        """
        start_time = time.time()
        self.total_calls += 1
        success = False
        token_count = 0
        result = None
        
        try:
            result = await func(*args, **kwargs)
            success = True
            self.successful_calls += 1
            
            # Estimate token count if possible
            if isinstance(result, str):
                # Rough estimate: 1 token ≈ 4 characters for English text
                token_count = len(result) // 4
            elif isinstance(result, dict) and kwargs.get('return_token_count'):
                token_count = kwargs.get('return_token_count', 0)
                
            self.total_tokens += token_count
            return result
            
        except Exception as e:
            self.failed_calls += 1
            logger.error(f"Model call failed: {str(e)}")
            
            # Convert to appropriate ModelError subclass
            if "rate limit" in str(e).lower():
                raise RateLimitError(f"Rate limit exceeded: {str(e)}")
            elif "token" in str(e).lower() and ("limit" in str(e).lower() or "exceed" in str(e).lower()):
                raise TokenLimitError(f"Token limit exceeded: {str(e)}")
            elif "auth" in str(e).lower() or "key" in str(e).lower() or "credential" in str(e).lower():
                raise AuthenticationError(f"Authentication failed: {str(e)}")
            elif "unavailable" in str(e).lower() or "down" in str(e).lower() or "connect" in str(e).lower():
                raise ServiceUnavailableError(f"Service unavailable: {str(e)}")
            else:
                raise ModelError(f"Model error: {str(e)}")
                
        finally:
            elapsed_time = time.time() - start_time
            self.total_time += elapsed_time
            
            # Call the completion callback if set
            if self._on_completion:
                try:
                    self._on_completion(
                        success=success,
                        elapsed_time=elapsed_time,
                        token_count=token_count,
                        model_type=self.__class__.__name__
                    )
                except Exception as callback_error:
                    logger.warning(f"Error in completion callback: {callback_error}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this model.
        
        Returns:
            Dictionary of statistics
        """
        avg_time = 0 if self.total_calls == 0 else self.total_time / self.total_calls
        success_rate = 0 if self.total_calls == 0 else (self.successful_calls / self.total_calls) * 100
        
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": success_rate,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "average_time_per_call": avg_time,
            "model_type": self.__class__.__name__
        } 