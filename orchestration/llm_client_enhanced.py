"""
Enhanced LLM Client with dynamic temperature control and improved provider management
Extends the existing LLMClient with better mock integration and temperature controls
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
from orchestration.llm_client import LLMClient, LLMProvider
from orchestration.mock_llm import MockLLMProvider, MockLLMTestHelper

logger = logging.getLogger(__name__)

class LLMClientEnhanced(LLMClient):
    """Enhanced LLM client with temperature control and mock support"""
    
    def __init__(self, 
                 provider: str = "openai",
                 model: str = None,
                 temperature: float = 0.0,
                 max_tokens: int = 500,
                 use_mock: bool = None,
                 mock_responses_file: Optional[str] = None):
        """
        Initialize enhanced LLM client
        
        Args:
            provider: LLM provider name (openai, anthropic, huggingface, mock)
            model: Model name (provider-specific)
            temperature: Temperature for response generation (0.0-2.0)
            max_tokens: Maximum tokens in response
            use_mock: Force mock mode (auto-detects if None)
            mock_responses_file: Custom responses file for mock mode
        """
        
        # Determine mock mode
        if use_mock is None:
            # Auto-detect mock mode based on environment
            use_mock = (
                os.getenv("USE_MOCK_LLM", "").lower() == "true" or
                provider.lower() == "mock" or
                self._should_use_mock()
            )
        
        # Initialize temperature control
        self.temperature = self._validate_temperature(temperature)
        self.max_tokens = max_tokens
        self.use_mock = use_mock
        self.mock_responses_file = mock_responses_file
        
        # Set provider based on mock mode
        if use_mock:
            provider = "mock"
            
        # Initialize parent class
        super().__init__(provider=provider, model=model)
        
        # Override with mock provider if needed
        if use_mock and provider.lower() == "mock":
            self._initialize_mock_provider()
        
        logger.info(f"Enhanced LLM Client initialized - Provider: {provider}, Temperature: {temperature}, Mock: {use_mock}")
    
    def _should_use_mock(self) -> bool:
        """Auto-detect if mock mode should be used"""
        # Use mock if no API keys are available
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            return True
        
        # Use mock in test environment
        if os.getenv("ENVIRONMENT") in ["test", "unittest"]:
            return True
        
        # Use mock if specifically requested
        if os.getenv("FORCE_MOCK_LLM") == "true":
            return True
        
        return False
    
    def _validate_temperature(self, temperature: float) -> float:
        """Validate and clamp temperature value"""
        if temperature < 0:
            logger.warning(f"Temperature {temperature} too low, setting to 0.0")
            return 0.0
        elif temperature > 2:
            logger.warning(f"Temperature {temperature} too high, setting to 2.0")
            return 2.0
        return temperature
    
    def _initialize_mock_provider(self):
        """Initialize mock provider with custom settings"""
        if self.mock_responses_file:
            self.provider = MockLLMProvider(
                responses_file=self.mock_responses_file,
                default_latency_ms=50,  # Fast for testing
                model_name=f"mock-{self.model or 'gpt-4'}"
            )
        else:
            self.provider = MockLLMTestHelper.create_deterministic_mock()
        
        logger.info("Mock provider initialized for enhanced client")
    
    def set_temperature(self, temperature: float):
        """Dynamically update temperature setting"""
        old_temp = self.temperature
        self.temperature = self._validate_temperature(temperature)
        logger.debug(f"Temperature changed from {old_temp} to {self.temperature}")
    
    def set_deterministic(self):
        """Set temperature to 0 for deterministic responses"""
        self.set_temperature(0.0)
        logger.info("Set to deterministic mode (temperature=0)")
    
    def set_creative(self, temperature: float = 0.7):
        """Set temperature for creative responses"""
        self.set_temperature(temperature)
        logger.info(f"Set to creative mode (temperature={temperature})")
    
    def generate_with_options(self, 
                            prompt: str,
                            temperature: Optional[float] = None,
                            max_tokens: Optional[int] = None,
                            deterministic: bool = False,
                            **kwargs) -> Dict[str, Any]:
        """
        Generate response with flexible options
        
        Args:
            prompt: Input prompt
            temperature: Override temperature for this request
            max_tokens: Override max tokens for this request
            deterministic: Force deterministic mode (temperature=0)
            **kwargs: Additional provider-specific options
            
        Returns:
            Enhanced response with metadata
        """
        
        # Determine effective parameters
        effective_temperature = 0.0 if deterministic else (temperature if temperature is not None else self.temperature)
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Prepare generation parameters
        generation_params = {
            "temperature": effective_temperature,
            "max_tokens": effective_max_tokens,
            **kwargs
        }
        
        try:
            # Use mock provider directly if available
            if self.use_mock and hasattr(self.provider, 'generate'):
                result = self.provider.generate(prompt, **generation_params)
            else:
                # Fallback to parent generate method
                json_response = self.generate(prompt, **generation_params)
                result = json.loads(json_response) if isinstance(json_response, str) else json_response
            
            # Enhance response with additional metadata
            enhanced_result = {
                **result,
                "generation_params": {
                    "temperature": effective_temperature,
                    "max_tokens": effective_max_tokens,
                    "deterministic": deterministic,
                    "provider": self.provider_name,
                    "model": getattr(self, 'model', 'unknown')
                },
                "client_metadata": {
                    "enhanced_client": True,
                    "mock_mode": self.use_mock,
                    "generation_time": datetime.utcnow().isoformat()
                }
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Enhanced generation failed: {e}")
            return {
                "error": True,
                "error_message": str(e),
                "error_type": type(e).__name__,
                "generation_params": generation_params,
                "response": "Error occurred during generation"
            }
    
    def test_consistency(self, 
                        prompt: str, 
                        n_runs: int = 5,
                        temperature: float = 0.0) -> Dict[str, Any]:
        """
        Test response consistency across multiple runs
        
        Args:
            prompt: Test prompt
            n_runs: Number of test runs
            temperature: Temperature to use (0.0 for deterministic)
            
        Returns:
            Consistency analysis results
        """
        logger.info(f"Running consistency test with {n_runs} iterations")
        
        responses = []
        latencies = []
        
        for i in range(n_runs):
            result = self.generate_with_options(
                prompt, 
                temperature=temperature,
                deterministic=(temperature == 0.0)
            )
            
            response_text = result.get("response", "")
            responses.append(response_text)
            
            # Track latency if available
            if "latency_ms" in result:
                latencies.append(result["latency_ms"])
        
        # Analyze consistency
        unique_responses = list(set(responses))
        consistency_score = 1.0 - (len(unique_responses) - 1) / n_runs
        
        analysis = {
            "n_runs": n_runs,
            "temperature": temperature,
            "unique_responses": len(unique_responses),
            "consistency_score": consistency_score,
            "is_deterministic": len(unique_responses) == 1,
            "responses": responses,
            "unique_response_samples": unique_responses[:3],  # First 3 unique responses
            "latency_stats": {
                "mean_ms": sum(latencies) / len(latencies) if latencies else 0,
                "min_ms": min(latencies) if latencies else 0,
                "max_ms": max(latencies) if latencies else 0
            } if latencies else None
        }
        
        logger.info(f"Consistency test complete - Score: {consistency_score:.3f}, Unique responses: {len(unique_responses)}")
        return analysis
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get detailed provider information"""
        info = {
            "provider": self.provider_name,
            "model": getattr(self, 'model', None),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_mock": self.use_mock,
            "enhanced": True
        }
        
        # Add mock-specific info
        if self.use_mock and hasattr(self.provider, 'get_call_history'):
            info["mock_stats"] = self.provider.get_call_history()
        
        return info
    
    def add_mock_response(self, prompt: str, response: str):
        """Add custom response for mock mode"""
        if self.use_mock and hasattr(self.provider, 'add_response'):
            self.provider.add_response(prompt, response)
            logger.info(f"Added custom mock response for: {prompt[:50]}...")
        else:
            logger.warning("Cannot add mock response: not in mock mode or provider doesn't support it")
    
    def reset_mock_stats(self):
        """Reset mock provider statistics"""
        if self.use_mock and hasattr(self.provider, 'reset_stats'):
            self.provider.reset_stats()
            logger.info("Mock provider statistics reset")

class LLMClientFactory:
    """Factory for creating LLM clients with different configurations"""
    
    @staticmethod
    def create_deterministic_client(provider: str = "openai", 
                                  model: str = None) -> LLMClientEnhanced:
        """Create client optimized for deterministic responses"""
        return LLMClientEnhanced(
            provider=provider,
            model=model,
            temperature=0.0,
            max_tokens=500
        )
    
    @staticmethod
    def create_creative_client(provider: str = "openai",
                             model: str = None,
                             temperature: float = 0.7) -> LLMClientEnhanced:
        """Create client optimized for creative responses"""
        return LLMClientEnhanced(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=1000
        )
    
    @staticmethod
    def create_test_client(mock_responses_file: Optional[str] = None) -> LLMClientEnhanced:
        """Create client for testing with mock provider"""
        return LLMClientEnhanced(
            provider="mock",
            temperature=0.0,
            use_mock=True,
            mock_responses_file=mock_responses_file,
            max_tokens=200
        )
    
    @staticmethod
    def create_from_env() -> LLMClientEnhanced:
        """Create client from environment variables"""
        provider = os.getenv("LLM_PROVIDER", "openai")
        model = os.getenv("LLM_MODEL")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "500"))
        
        return LLMClientEnhanced(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

# Example usage and testing
if __name__ == "__main__":
    print("=== Enhanced LLM Client Test ===")
    
    # Test deterministic client
    print("\n1. Testing deterministic client:")
    deterministic_client = LLMClientFactory.create_deterministic_client(provider="mock")
    
    test_prompt = "What is the capital of France?"
    consistency_result = deterministic_client.test_consistency(test_prompt, n_runs=3)
    print(f"Consistency score: {consistency_result['consistency_score']}")
    print(f"Is deterministic: {consistency_result['is_deterministic']}")
    
    # Test temperature control
    print("\n2. Testing temperature control:")
    client = LLMClientEnhanced(provider="mock", temperature=0.5)
    client.set_deterministic()
    
    result = client.generate_with_options(test_prompt, deterministic=True)
    print(f"Response: {result.get('response', 'No response')}")
    print(f"Temperature used: {result.get('generation_params', {}).get('temperature')}")
    
    # Test provider info
    print("\n3. Provider information:")
    info = client.get_provider_info()
    for key, value in info.items():
        print(f"{key}: {value}")