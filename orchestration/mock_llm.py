"""
Mock LLM Provider for deterministic testing and unit tests
Provides consistent, reproducible responses for testing prompt templates
"""

import json
import hashlib
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from orchestration.llm_client import LLMProvider

logger = logging.getLogger(__name__)

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for deterministic testing"""
    
    def __init__(self, 
                 responses_file: Optional[str] = None,
                 default_latency_ms: float = 100,
                 model_name: str = "mock-gpt-4"):
        """
        Initialize mock provider
        
        Args:
            responses_file: JSON file with predefined responses
            default_latency_ms: Simulated response latency
            model_name: Mock model identifier
        """
        self.model_name = model_name
        self.default_latency_ms = default_latency_ms
        self.responses = {}
        self.call_count = 0
        
        if responses_file:
            self._load_responses(responses_file)
        
        # Default responses for common patterns
        self._setup_default_responses()
        
        logger.info(f"MockLLMProvider initialized with {len(self.responses)} predefined responses")
    
    def _load_responses(self, responses_file: str):
        """Load predefined responses from JSON file"""
        try:
            with open(responses_file, 'r') as f:
                self.responses = json.load(f)
                logger.info(f"Loaded {len(self.responses)} responses from {responses_file}")
        except FileNotFoundError:
            logger.warning(f"Responses file {responses_file} not found, using defaults only")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in responses file: {e}")
    
    def _setup_default_responses(self):
        """Setup default responses for common test patterns"""
        default_responses = {
            # Capital finder responses
            "france": {
                "capital": "Paris",
                "confidence": 1.0,
                "metadata": {
                    "source": "mock_geographical_database",
                    "timestamp": "2024-09-04T10:00:00Z"
                }
            },
            "united states": {
                "capital": "Washington D.C.",
                "confidence": 1.0,
                "metadata": {
                    "source": "mock_geographical_database", 
                    "timestamp": "2024-09-04T10:00:00Z"
                }
            },
            "japan": {
                "capital": "Tokyo",
                "confidence": 1.0,
                "metadata": {
                    "source": "mock_geographical_database",
                    "timestamp": "2024-09-04T10:00:00Z"
                }
            },
            
            # Summarization responses
            "summarize": "This is a comprehensive summary of the provided document, highlighting the key points and main conclusions.",
            
            # Financial planning responses
            "financial planning": {
                "recommendation": "Based on the provided information, here is a personalized financial plan",
                "risk_level": "moderate",
                "confidence": 0.9
            },
            
            # Default fallback
            "default": "I understand your request and will provide an appropriate response based on the given context."
        }
        
        # Convert to hash-based lookup
        for key, response in default_responses.items():
            prompt_hash = self._generate_hash(key.lower())
            if isinstance(response, dict):
                self.responses[prompt_hash] = json.dumps(response)
            else:
                self.responses[prompt_hash] = response
    
    def _generate_hash(self, prompt: str) -> str:
        """Generate consistent hash for prompt lookup"""
        # Normalize prompt for consistent hashing
        normalized = prompt.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _find_best_match(self, prompt: str) -> Optional[str]:
        """Find best matching response for prompt"""
        prompt_lower = prompt.lower()
        
        # Direct hash lookup first
        prompt_hash = self._generate_hash(prompt)
        if prompt_hash in self.responses:
            return self.responses[prompt_hash]
        
        # Pattern matching for flexibility
        for key_pattern in ["france", "united states", "japan", "summarize", "financial planning"]:
            if key_pattern in prompt_lower:
                pattern_hash = self._generate_hash(key_pattern)
                if pattern_hash in self.responses:
                    return self.responses[pattern_hash]
        
        # Default fallback
        default_hash = self._generate_hash("default")
        return self.responses.get(default_hash, "Mock response generated.")
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate mock response with realistic metadata"""
        self.call_count += 1
        
        # Simulate processing time
        start_time = time.time()
        time.sleep(self.default_latency_ms / 1000)  # Convert ms to seconds
        
        # Find appropriate response
        response_content = self._find_best_match(prompt)
        
        # Calculate simulated token usage
        prompt_tokens = len(prompt.split())
        completion_tokens = len(response_content.split()) if isinstance(response_content, str) else 20
        
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "response": response_content,
            "model": self.model_name,
            "provider": "mock",
            "latency_ms": elapsed_time,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens
            },
            "timestamp": datetime.utcnow().isoformat(),
            "call_count": self.call_count,
            "temperature": kwargs.get("temperature", 0),
            "deterministic": True
        }
    
    def health_check(self) -> bool:
        """Mock health check - always returns True"""
        return True
    
    def add_response(self, prompt: str, response: str):
        """Add custom response for testing"""
        prompt_hash = self._generate_hash(prompt)
        self.responses[prompt_hash] = response
        logger.debug(f"Added custom response for prompt: {prompt[:50]}...")
    
    def get_call_history(self) -> Dict[str, Any]:
        """Get mock provider usage statistics"""
        return {
            "total_calls": self.call_count,
            "available_responses": len(self.responses),
            "model": self.model_name
        }
    
    def reset_stats(self):
        """Reset call statistics"""
        self.call_count = 0

class MockLLMTestHelper:
    """Helper class for setting up mock LLM in tests"""
    
    @staticmethod
    def create_deterministic_mock() -> MockLLMProvider:
        """Create mock with deterministic responses for testing"""
        mock = MockLLMProvider(
            default_latency_ms=50,  # Fast for tests
            model_name="test-mock-gpt-4"
        )
        
        # Add common test responses
        test_responses = {
            "What is the capital of France?": json.dumps({
                "capital": "Paris",
                "confidence": 1.0
            }),
            "What is the capital of Japan?": json.dumps({
                "capital": "Tokyo", 
                "confidence": 1.0
            }),
            "Summarize this document": "This is a test summary of the document.",
            "Create a financial plan": json.dumps({
                "recommendation": "Test financial recommendation",
                "risk_level": "moderate"
            })
        }
        
        for prompt, response in test_responses.items():
            mock.add_response(prompt, response)
        
        return mock
    
    @staticmethod
    def create_mock_with_file(responses_file: str) -> MockLLMProvider:
        """Create mock with responses from file"""
        return MockLLMProvider(
            responses_file=responses_file,
            default_latency_ms=75,
            model_name="file-mock-gpt-4"
        )
    
    @staticmethod
    def save_responses_template(file_path: str):
        """Save template responses file for customization"""
        template = {
            "5d41402abc4b2a76b9719d911017c592": json.dumps({
                "capital": "Paris",
                "confidence": 1.0,
                "metadata": {
                    "source": "test_database",
                    "timestamp": "2024-09-04T10:00:00Z"
                }
            }),
            "098f6bcd4621d373cade4e832627b4f6": "This is a test summarization response.",
            "default_hash": "Default mock response for unmatched prompts."
        }
        
        with open(file_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        logger.info(f"Saved mock responses template to {file_path}")

# Example usage and testing
if __name__ == "__main__":
    # Create mock provider
    mock_provider = MockLLMProvider()
    
    # Test capital queries
    test_prompts = [
        "What is the capital of France?",
        "What is the capital of Japan?", 
        "Tell me about financial planning",
        "Unknown query test"
    ]
    
    print("=== Mock LLM Provider Test ===")
    for prompt in test_prompts:
        result = mock_provider.generate(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result['response']}")
        print(f"Latency: {result['latency_ms']:.1f}ms")
        print(f"Tokens: {result['tokens']['total']}")
    
    print(f"\n=== Provider Statistics ===")
    stats = mock_provider.get_call_history()
    print(f"Total calls: {stats['total_calls']}")
    print(f"Available responses: {stats['available_responses']}")
    print(f"Model: {stats['model']}")