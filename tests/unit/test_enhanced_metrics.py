"""
Unit tests for Enhanced Metrics
Tests ExactMatchMetric, StabilityMetric, SchemaValidityMetric, and LatencyMetric
"""

import pytest
import json
import tempfile
import statistics
from unittest.mock import patch, MagicMock

# Import components to test
from evaluation.metrics_enhanced import (
    ExactMatchMetric,
    StabilityMetric, 
    SchemaValidityMetric,
    LatencyMetric,
    MetricsRunner
)

# Mock test case class for testing
class MockTestCase:
    """Mock test case for metrics testing"""
    def __init__(self, input_text: str, actual_output: str, expected_output: str = None, **kwargs):
        self.input = input_text
        self.actual_output = actual_output
        self.expected_output = expected_output
        for key, value in kwargs.items():
            setattr(self, key, value)

class TestExactMatchMetric:
    """Test suite for ExactMatchMetric"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.metric = ExactMatchMetric(threshold=0.8, case_sensitive=False)
    
    def test_exact_match_success(self):
        """Test exact match returns 1.0 for identical outputs"""
        test_case = MockTestCase(
            input_text="test input",
            actual_output="Paris",
            expected_output="Paris"
        )
        
        score = self.metric.measure(test_case)
        assert score == 1.0
    
    def test_exact_match_failure(self):
        """Test exact match returns 0.0 for different outputs"""
        test_case = MockTestCase(
            input_text="test input", 
            actual_output="Paris",
            expected_output="London"
        )
        
        score = self.metric.measure(test_case)
        assert score == 0.0
    
    def test_case_insensitive_matching(self):
        """Test case insensitive matching"""
        test_case = MockTestCase(
            input_text="test input",
            actual_output="PARIS",
            expected_output="paris"
        )
        
        # Case insensitive (default)
        score = self.metric.measure(test_case)
        assert score == 1.0
        
        # Case sensitive
        case_sensitive_metric = ExactMatchMetric(case_sensitive=True)
        score_sensitive = case_sensitive_metric.measure(test_case)
        assert score_sensitive == 0.0
    
    def test_whitespace_handling(self):
        """Test whitespace is stripped for comparison"""
        test_case = MockTestCase(
            input_text="test input",
            actual_output="  Paris  ",
            expected_output="Paris"
        )
        
        score = self.metric.measure(test_case)
        assert score == 1.0
    
    def test_missing_expected_output(self):
        """Test handling of missing expected output"""
        test_case = MockTestCase(
            input_text="test input",
            actual_output="Paris"
            # No expected_output
        )
        
        score = self.metric.measure(test_case)
        assert score == 0.0
    
    def test_batch_evaluation(self):
        """Test batch evaluation functionality"""
        test_cases = [
            MockTestCase("input1", "Paris", "Paris"),
            MockTestCase("input2", "London", "London"),
            MockTestCase("input3", "Tokyo", "Berlin"),  # Mismatch
            MockTestCase("input4", "Madrid", "Madrid")
        ]
        
        results = self.metric.evaluate_batch(test_cases)
        
        assert results["metric"] == "exact_match"
        assert results["total_cases"] == 4
        assert results["matches"] == 3
        assert results["mismatches"] == 1
        assert results["pass_rate"] == 0.75
        assert results["passes_threshold"] is False  # 0.75 < 0.8 threshold
        
        # Check mismatch examples
        assert len(results["mismatch_examples"]) == 1
        assert "Tokyo" in results["mismatch_examples"][0]["actual"]
        assert "Berlin" in results["mismatch_examples"][0]["expected"]

class TestSchemaValidityMetric:
    """Test suite for SchemaValidityMetric"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.schema = {
            "type": "object",
            "properties": {
                "capital": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["capital", "confidence"],
            "additionalProperties": False
        }
        self.metric = SchemaValidityMetric(schema_dict=self.schema)
    
    def test_valid_json_schema(self):
        """Test valid JSON against schema"""
        valid_output = json.dumps({
            "capital": "Paris",
            "confidence": 0.95
        })
        
        test_case = MockTestCase("input", valid_output)
        score = self.metric.measure(test_case)
        assert score == 1.0
    
    def test_invalid_json_schema(self):
        """Test invalid JSON against schema"""
        # Missing required field
        invalid_output = json.dumps({
            "capital": "Paris"
            # Missing confidence
        })
        
        test_case = MockTestCase("input", invalid_output)
        score = self.metric.measure(test_case)
        assert score == 0.0
    
    def test_invalid_json_format(self):
        """Test invalid JSON format"""
        invalid_json = "{ capital: Paris, confidence: 0.95 }"  # Invalid JSON syntax
        
        test_case = MockTestCase("input", invalid_json)
        score = self.metric.measure(test_case)
        assert score == 0.0
    
    def test_wrong_data_types(self):
        """Test wrong data types in JSON"""
        wrong_types = json.dumps({
            "capital": "Paris",
            "confidence": "high"  # Should be number, not string
        })
        
        test_case = MockTestCase("input", wrong_types)
        score = self.metric.measure(test_case)
        assert score == 0.0
    
    def test_value_constraints(self):
        """Test value constraints (min/max)"""
        out_of_range = json.dumps({
            "capital": "Paris",
            "confidence": 1.5  # Exceeds maximum of 1
        })
        
        test_case = MockTestCase("input", out_of_range)
        score = self.metric.measure(test_case)
        assert score == 0.0
    
    def test_batch_validation(self):
        """Test batch validation functionality"""
        test_cases = [
            MockTestCase("input1", json.dumps({"capital": "Paris", "confidence": 0.95})),
            MockTestCase("input2", json.dumps({"capital": "London", "confidence": 0.88})),
            MockTestCase("input3", json.dumps({"capital": "Tokyo"})),  # Missing confidence
            MockTestCase("input4", "invalid json"),  # Invalid JSON
        ]
        
        results = self.metric.validate_batch(test_cases)
        
        assert results["metric"] == "schema_validity"
        assert results["total_cases"] == 4
        assert results["valid_count"] == 2
        assert results["invalid_count"] == 2
        assert results["pass_rate"] == 0.5
        assert len(results["validation_errors"]) == 2
    
    def test_schema_loading_from_file(self):
        """Test loading schema from file"""
        # Create temporary schema file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.schema, f)
            schema_file = f.name
        
        try:
            file_metric = SchemaValidityMetric(schema_path=schema_file)
            
            valid_output = json.dumps({"capital": "Paris", "confidence": 0.95})
            test_case = MockTestCase("input", valid_output)
            score = file_metric.measure(test_case)
            assert score == 1.0
            
        finally:
            import os
            os.unlink(schema_file)

class TestStabilityMetric:
    """Test suite for StabilityMetric"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Mock LLM client for testing
        self.mock_client = MagicMock()
        self.metric = StabilityMetric(
            n_runs=3,
            temperature=0.0,
            llm_client=self.mock_client
        )
    
    def test_perfect_consistency(self):
        """Test perfect consistency (deterministic responses)"""
        # Mock client returns same response every time
        self.mock_client.generate_with_options.return_value = {
            "response": "Paris",
            "latency_ms": 100
        }
        
        test_case = MockTestCase("What is the capital of France?", "")
        score = self.metric.measure(test_case)
        
        assert score == 1.0
        assert self.mock_client.generate_with_options.call_count == 3
    
    def test_partial_consistency(self):
        """Test partial consistency"""
        # Mock client returns different responses
        responses = [
            {"response": "Paris", "latency_ms": 100},
            {"response": "Paris", "latency_ms": 110}, 
            {"response": "Lyon", "latency_ms": 95}   # Different response
        ]
        self.mock_client.generate_with_options.side_effect = responses
        
        test_case = MockTestCase("What is the capital of France?", "")
        score = self.metric.measure(test_case)
        
        # 2 out of 3 responses are the same, so consistency = 1 - (2-1)/3 = 2/3
        assert abs(score - 2/3) < 0.01
    
    def test_no_consistency(self):
        """Test no consistency (all different responses)"""
        responses = [
            {"response": "Paris", "latency_ms": 100},
            {"response": "Lyon", "latency_ms": 110},
            {"response": "Marseille", "latency_ms": 95}
        ]
        self.mock_client.generate_with_options.side_effect = responses
        
        test_case = MockTestCase("What is the capital of France?", "")
        score = self.metric.measure(test_case)
        
        # All different: 1 - (3-1)/3 = 1/3
        assert abs(score - 1/3) < 0.01
    
    def test_error_handling(self):
        """Test handling of generation errors"""
        def side_effect(*args, **kwargs):
            raise Exception("Generation failed")
        
        self.mock_client.generate_with_options.side_effect = side_effect
        
        test_case = MockTestCase("test input", "")
        score = self.metric.measure(test_case)
        
        assert score == 0.0
    
    def test_no_client_available(self):
        """Test metric with no client available"""
        metric = StabilityMetric(llm_client=None)
        test_case = MockTestCase("test input", "")
        
        score = metric.measure(test_case)
        assert score == 0.0
    
    def test_analyze_variance(self):
        """Test variance analysis across multiple test cases"""
        # Setup mock client to return consistent responses
        self.mock_client.generate_with_options.return_value = {
            "response": "consistent",
            "latency_ms": 100
        }
        
        test_cases = [
            MockTestCase("input1", ""),
            MockTestCase("input2", "")
        ]
        
        results = self.metric.analyze_variance(test_cases)
        
        assert results["metric"] == "stability"
        assert results["mean_consistency"] == 1.0
        assert results["deterministic_count"] == 2
        assert results["total_cases"] == 2
        assert results["n_runs_per_case"] == 3

class TestLatencyMetric:
    """Test suite for LatencyMetric"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.metric = LatencyMetric(max_latency_ms=1000, threshold=0.9)
    
    def test_within_latency_limit(self):
        """Test response within latency limit"""
        test_case = MockTestCase("input", "output", latency_ms=500)
        score = self.metric.measure(test_case)
        assert score == 1.0
    
    def test_exceeds_latency_limit(self):
        """Test response exceeding latency limit"""
        test_case = MockTestCase("input", "output", latency_ms=1500)
        score = self.metric.measure(test_case)
        assert score == 0.0
    
    def test_latency_in_metadata(self):
        """Test extracting latency from metadata"""
        test_case = MockTestCase(
            "input", 
            "output",
            metadata={"latency_ms": 750}
        )
        score = self.metric.measure(test_case)
        assert score == 1.0
    
    def test_response_time_attribute(self):
        """Test extracting from response_time_ms attribute"""
        test_case = MockTestCase("input", "output", response_time_ms=600)
        score = self.metric.measure(test_case)
        assert score == 1.0
    
    def test_missing_latency_data(self):
        """Test handling missing latency data"""
        test_case = MockTestCase("input", "output")  # No latency info
        score = self.metric.measure(test_case)
        assert score == 0.0
    
    def test_performance_analysis(self):
        """Test performance analysis across multiple cases"""
        test_cases = [
            MockTestCase("input1", "output1", latency_ms=200),
            MockTestCase("input2", "output2", latency_ms=800),
            MockTestCase("input3", "output3", latency_ms=1200),  # Exceeds limit
            MockTestCase("input4", "output4", latency_ms=600)
        ]
        
        results = self.metric.analyze_performance(test_cases)
        
        assert results["metric"] == "latency"
        assert results["total_cases"] == 4
        assert results["within_limit_count"] == 3
        assert results["exceeded_limit_count"] == 1
        assert results["pass_rate"] == 0.75
        assert results["max_latency_limit_ms"] == 1000
        assert results["mean_latency_ms"] == 700  # (200+800+1200+600)/4
        assert results["min_latency_ms"] == 200
        assert results["max_latency_ms"] == 1200

class TestMetricsRunner:
    """Test suite for MetricsRunner"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.runner = MetricsRunner()
        
        # Add test metrics
        self.runner.add_metric("exact_match", ExactMatchMetric(threshold=0.8))
        
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"]
        }
        self.runner.add_metric("schema_validity", SchemaValidityMetric(schema_dict=schema))
        self.runner.add_metric("latency", LatencyMetric(max_latency_ms=1000))
    
    def test_run_all_metrics(self):
        """Test running all registered metrics"""
        test_cases = [
            MockTestCase(
                "input1", 
                json.dumps({"result": "Paris"}),
                json.dumps({"result": "Paris"}),
                latency_ms=500
            ),
            MockTestCase(
                "input2",
                json.dumps({"result": "London"}),
                json.dumps({"result": "Tokyo"}),  # Mismatch
                latency_ms=800
            )
        ]
        
        results = self.runner.run_all_metrics(test_cases)
        
        assert "timestamp" in results
        assert results["total_test_cases"] == 2
        assert results["metrics_run"] == 3
        assert "metric_results" in results
        assert "overall_summary" in results
        
        # Check individual metric results
        assert "exact_match" in results["metric_results"]
        assert "schema_validity" in results["metric_results"]
        assert "latency" in results["metric_results"]
        
        # Check overall summary
        summary = results["overall_summary"]
        assert "passing_metrics" in summary
        assert "total_metrics" in summary
        assert "overall_pass_rate" in summary
    
    def test_empty_test_cases(self):
        """Test handling of empty test case list"""
        results = self.runner.run_all_metrics([])
        
        assert results["total_test_cases"] == 0
        assert "metric_results" in results
    
    def test_metric_error_handling(self):
        """Test error handling when metric fails"""
        # Add a metric that will fail
        failing_metric = MagicMock()
        failing_metric.measure.side_effect = Exception("Metric failed")
        
        self.runner.add_metric("failing_metric", failing_metric)
        
        test_case = MockTestCase("input", "output")
        results = self.runner.run_all_metrics([test_case])
        
        assert "failing_metric" in results["metric_results"]
        assert "error" in results["metric_results"]["failing_metric"]

class TestIntegration:
    """Integration tests for metrics system"""
    
    def test_end_to_end_evaluation(self):
        """End-to-end test of metrics evaluation"""
        # Setup runner with all metrics
        runner = MetricsRunner()
        runner.add_metric("exact_match", ExactMatchMetric(threshold=0.9))
        
        schema = {
            "type": "object", 
            "properties": {"capital": {"type": "string"}},
            "required": ["capital"]
        }
        runner.add_metric("schema_validity", SchemaValidityMetric(schema_dict=schema))
        runner.add_metric("latency", LatencyMetric(max_latency_ms=2000))
        
        # Create test cases
        test_cases = [
            MockTestCase(
                "What is the capital of France?",
                json.dumps({"capital": "Paris"}),
                json.dumps({"capital": "Paris"}),
                latency_ms=150
            ),
            MockTestCase(
                "What is the capital of Japan?", 
                json.dumps({"capital": "Tokyo"}),
                json.dumps({"capital": "Tokyo"}),
                latency_ms=200
            )
        ]
        
        # Run evaluation
        results = runner.run_all_metrics(test_cases)
        
        # Verify results
        assert results["overall_summary"]["all_metrics_passing"] is True
        assert results["metric_results"]["exact_match"]["pass_rate"] == 1.0
        assert results["metric_results"]["schema_validity"]["pass_rate"] == 1.0  
        assert results["metric_results"]["latency"]["pass_rate"] == 1.0

# Pytest fixtures
@pytest.fixture
def sample_test_cases():
    """Fixture providing sample test cases"""
    return [
        MockTestCase(
            "input1",
            json.dumps({"capital": "Paris", "confidence": 0.95}),
            json.dumps({"capital": "Paris", "confidence": 0.95}),
            latency_ms=300
        ),
        MockTestCase(
            "input2", 
            json.dumps({"capital": "London", "confidence": 0.88}),
            json.dumps({"capital": "London", "confidence": 0.88}),
            latency_ms=450
        )
    ]

@pytest.fixture
def standard_schema():
    """Fixture providing standard JSON schema"""
    return {
        "type": "object",
        "properties": {
            "capital": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["capital", "confidence"]
    }

def test_metrics_with_fixtures(sample_test_cases, standard_schema):
    """Test metrics using fixtures"""
    exact_match = ExactMatchMetric()
    schema_validity = SchemaValidityMetric(schema_dict=standard_schema)
    
    # Test exact match
    exact_results = exact_match.evaluate_batch(sample_test_cases)
    assert exact_results["pass_rate"] == 1.0
    
    # Test schema validity
    schema_results = schema_validity.validate_batch(sample_test_cases)
    assert schema_results["pass_rate"] == 1.0

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])