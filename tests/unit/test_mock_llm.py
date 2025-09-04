"""
Unit tests for Mock LLM Provider
Tests deterministic behavior, response consistency, and mock functionality
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the components to test
from orchestration.mock_llm import MockLLMProvider, MockLLMTestHelper
from orchestration.llm_client_enhanced import LLMClientEnhanced, LLMClientFactory

class TestMockLLMProvider:
    """Test suite for MockLLMProvider"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_provider = MockLLMProvider(
            default_latency_ms=10,  # Fast for testing
            model_name="test-mock-gpt-4"
        )
    
    def test_basic_response_generation(self):
        """Test basic response generation"""
        prompt = "What is the capital of France?"
        result = self.mock_provider.generate(prompt)
        
        assert isinstance(result, dict)
        assert "response" in result
        assert "model" in result
        assert "provider" in result
        assert "latency_ms" in result
        assert "tokens" in result
        assert "timestamp" in result
        
        assert result["provider"] == "mock"
        assert result["model"] == "test-mock-gpt-4"
        assert result["latency_ms"] > 0
    
    def test_deterministic_responses(self):
        """Test that same prompt produces same response"""
        prompt = "What is the capital of France?"
        
        result1 = self.mock_provider.generate(prompt)
        result2 = self.mock_provider.generate(prompt)
        
        # Response should be identical
        assert result1["response"] == result2["response"]
        
        # But metadata may differ (timestamps, call counts)
        assert result1["timestamp"] != result2["timestamp"]
        assert result1["call_count"] != result2["call_count"]
    
    def test_pattern_matching(self):
        """Test pattern-based response matching"""
        test_cases = [
            ("france", "Paris"),
            ("France", "Paris"),  # Case insensitive
            ("FRANCE", "Paris"),
            ("What is the capital of France?", "Paris"),
            ("Tell me about France", "Paris")
        ]
        
        for prompt, expected_capital in test_cases:
            result = self.mock_provider.generate(prompt)
            response_data = json.loads(result["response"])
            assert response_data["capital"] == expected_capital
    
    def test_custom_response_addition(self):
        """Test adding custom responses"""
        custom_prompt = "What is the capital of Mars?"
        custom_response = "New Beijing"
        
        self.mock_provider.add_response(custom_prompt, custom_response)
        result = self.mock_provider.generate(custom_prompt)
        
        assert result["response"] == custom_response
    
    def test_health_check(self):
        """Test health check always returns True"""
        assert self.mock_provider.health_check() is True
    
    def test_call_statistics(self):
        """Test call statistics tracking"""
        initial_stats = self.mock_provider.get_call_history()
        assert initial_stats["total_calls"] == 0
        
        # Make some calls
        for i in range(3):
            self.mock_provider.generate(f"Test prompt {i}")
        
        final_stats = self.mock_provider.get_call_history()
        assert final_stats["total_calls"] == 3
        
        # Reset stats
        self.mock_provider.reset_stats()
        reset_stats = self.mock_provider.get_call_history()
        assert reset_stats["total_calls"] == 0
    
    def test_temperature_parameter(self):
        """Test temperature parameter handling"""
        prompt = "Test prompt"
        
        result = self.mock_provider.generate(prompt, temperature=0.5)
        assert result["temperature"] == 0.5
        assert result["deterministic"] is True  # Mock is always deterministic
    
    def test_token_counting(self):
        """Test token usage calculation"""
        short_prompt = "Hi"
        long_prompt = "This is a much longer prompt with many words to test token counting"
        
        short_result = self.mock_provider.generate(short_prompt)
        long_result = self.mock_provider.generate(long_prompt)
        
        assert short_result["tokens"]["prompt"] < long_result["tokens"]["prompt"]
        assert "total" in short_result["tokens"]
        assert short_result["tokens"]["total"] == (
            short_result["tokens"]["prompt"] + short_result["tokens"]["completion"]
        )

class TestMockLLMWithFiles:
    """Test mock LLM with response files"""
    
    def test_load_responses_from_file(self):
        """Test loading responses from JSON file"""
        # Create temporary response file
        responses = {
            "test_hash_123": "Test response from file",
            "another_hash_456": "Another test response"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(responses, f)
            temp_file = f.name
        
        try:
            # Create provider with response file
            provider = MockLLMProvider(responses_file=temp_file)
            
            # Verify responses were loaded
            stats = provider.get_call_history()
            assert stats["available_responses"] > len(responses)  # Including defaults
            
        finally:
            os.unlink(temp_file)
    
    def test_missing_response_file(self):
        """Test handling of missing response file"""
        provider = MockLLMProvider(responses_file="nonexistent_file.json")
        
        # Should still work with default responses
        result = provider.generate("What is the capital of France?")
        assert "response" in result

class TestMockLLMTestHelper:
    """Test MockLLMTestHelper utility class"""
    
    def test_create_deterministic_mock(self):
        """Test creating deterministic mock"""
        mock = MockLLMTestHelper.create_deterministic_mock()
        
        assert isinstance(mock, MockLLMProvider)
        assert mock.health_check() is True
        
        # Test predefined responses
        result = mock.generate("What is the capital of France?")
        response_data = json.loads(result["response"])
        assert response_data["capital"] == "Paris"
    
    def test_create_mock_with_file(self):
        """Test creating mock with response file"""
        # Create temporary response file
        responses = {"test_key": "test_response"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(responses, f)
            temp_file = f.name
        
        try:
            mock = MockLLMTestHelper.create_mock_with_file(temp_file)
            assert isinstance(mock, MockLLMProvider)
            
        finally:
            os.unlink(temp_file)
    
    def test_save_responses_template(self):
        """Test saving response template"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            MockLLMTestHelper.save_responses_template(temp_file)
            
            # Verify file was created and contains valid JSON
            with open(temp_file, 'r') as f:
                template = json.load(f)
            
            assert isinstance(template, dict)
            assert len(template) > 0
            
        finally:
            os.unlink(temp_file)

class TestLLMClientEnhancedWithMock:
    """Test enhanced LLM client with mock provider"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.client = LLMClientEnhanced(
            provider="mock",
            temperature=0.0,
            use_mock=True
        )
    
    def test_mock_client_initialization(self):
        """Test mock client initializes correctly"""
        assert self.client.use_mock is True
        assert self.client.temperature == 0.0
        assert self.client.provider_name == "mock"
    
    def test_deterministic_generation(self):
        """Test deterministic response generation"""
        prompt = "What is the capital of France?"
        
        result1 = self.client.generate_with_options(prompt, deterministic=True)
        result2 = self.client.generate_with_options(prompt, deterministic=True)
        
        assert result1["response"] == result2["response"]
        assert result1["generation_params"]["deterministic"] is True
        assert result1["generation_params"]["temperature"] == 0.0
    
    def test_consistency_testing(self):
        """Test consistency testing functionality"""
        prompt = "What is the capital of Japan?"
        
        consistency_result = self.client.test_consistency(
            prompt=prompt,
            n_runs=3,
            temperature=0.0
        )
        
        assert "consistency_score" in consistency_result
        assert "is_deterministic" in consistency_result
        assert consistency_result["n_runs"] == 3
        assert consistency_result["temperature"] == 0.0
        
        # With temperature=0, should be deterministic
        assert consistency_result["is_deterministic"] is True
        assert consistency_result["consistency_score"] == 1.0
    
    def test_temperature_control(self):
        """Test temperature control functionality"""
        # Test setting different temperatures
        self.client.set_temperature(0.5)
        assert self.client.temperature == 0.5
        
        self.client.set_deterministic()
        assert self.client.temperature == 0.0
        
        self.client.set_creative(0.8)
        assert self.client.temperature == 0.8
    
    def test_provider_info(self):
        """Test provider information retrieval"""
        info = self.client.get_provider_info()
        
        assert info["provider"] == "mock"
        assert info["use_mock"] is True
        assert info["enhanced"] is True
        assert "temperature" in info
        assert "mock_stats" in info
    
    def test_custom_mock_responses(self):
        """Test adding custom mock responses"""
        custom_prompt = "What is the capital of Atlantis?"
        custom_response = "Aquatica"
        
        self.client.add_mock_response(custom_prompt, custom_response)
        result = self.client.generate_with_options(custom_prompt)
        
        assert custom_response in str(result["response"])

class TestLLMClientFactory:
    """Test LLM client factory methods"""
    
    @patch.dict(os.environ, {}, clear=True)
    def test_create_deterministic_client(self):
        """Test creating deterministic client"""
        client = LLMClientFactory.create_deterministic_client(provider="mock")
        
        assert client.temperature == 0.0
        assert client.use_mock is True
    
    @patch.dict(os.environ, {}, clear=True)
    def test_create_test_client(self):
        """Test creating test client"""
        client = LLMClientFactory.create_test_client()
        
        assert client.use_mock is True
        assert client.temperature == 0.0
        assert client.provider_name == "mock"
    
    @patch.dict(os.environ, {
        "LLM_PROVIDER": "mock",
        "LLM_TEMPERATURE": "0.3",
        "LLM_MAX_TOKENS": "1000"
    })
    def test_create_from_env(self):
        """Test creating client from environment variables"""
        client = LLMClientFactory.create_from_env()
        
        assert client.temperature == 0.3
        assert client.max_tokens == 1000

class TestErrorHandling:
    """Test error handling in mock LLM"""
    
    def test_invalid_temperature_clamping(self):
        """Test invalid temperature values are clamped"""
        # Test negative temperature
        client = LLMClientEnhanced(provider="mock", temperature=-1.0)
        assert client.temperature == 0.0
        
        # Test temperature too high
        client = LLMClientEnhanced(provider="mock", temperature=5.0)
        assert client.temperature == 2.0
    
    def test_consistency_with_no_client(self):
        """Test consistency testing with no LLM client"""
        from evaluation.metrics_enhanced import StabilityMetric
        
        metric = StabilityMetric(llm_client=None)
        test_case = type('TestCase', (), {'input': 'test prompt'})()
        
        score = metric.measure(test_case)
        assert score == 0.0

# Test fixtures and utilities
@pytest.fixture
def sample_mock_provider():
    """Fixture providing a mock LLM provider"""
    return MockLLMProvider(
        default_latency_ms=5,
        model_name="test-fixture-mock"
    )

@pytest.fixture
def sample_test_client():
    """Fixture providing a test LLM client"""
    return LLMClientFactory.create_test_client()

# Integration test
def test_end_to_end_mock_integration(sample_test_client):
    """End-to-end test of mock LLM integration"""
    # Test basic generation
    result = sample_test_client.generate_with_options(
        "What is the capital of France?",
        deterministic=True
    )
    
    # Verify response structure
    assert "response" in result
    assert "generation_params" in result
    assert "client_metadata" in result
    assert result["client_metadata"]["mock_mode"] is True
    
    # Test consistency
    consistency = sample_test_client.test_consistency(
        "What is the capital of Japan?",
        n_runs=3
    )
    
    assert consistency["is_deterministic"] is True
    assert consistency["consistency_score"] == 1.0
    
    print("✅ All mock LLM tests passed!")

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])