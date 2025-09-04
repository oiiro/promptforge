"""
Enhanced evaluation metrics for PromptForge
Extends DeepEval with custom metrics for exact matching, stability, schema validation, and more
"""

import json
import jsonschema
import re
import hashlib
import statistics
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod

try:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    # Fallback if DeepEval not available
    class BaseMetric(ABC):
        def __init__(self, threshold: float = 0.5):
            self.threshold = threshold
        
        @abstractmethod
        def measure(self, test_case) -> float:
            pass
    
    class LLMTestCase:
        def __init__(self, input: str, actual_output: str, expected_output: str = None, **kwargs):
            self.input = input
            self.actual_output = actual_output
            self.expected_output = expected_output
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    DEEPEVAL_AVAILABLE = False

# Import mock client if available
try:
    from orchestration.llm_client_enhanced import LLMClientEnhanced
    LLM_CLIENT_AVAILABLE = True
except ImportError:
    LLM_CLIENT_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExactMatchMetric(BaseMetric):
    """Exact match metric for deterministic outputs"""
    
    def __init__(self, threshold: float = 0.95, case_sensitive: bool = False):
        """
        Initialize exact match metric
        
        Args:
            threshold: Minimum pass rate for batch evaluation
            case_sensitive: Whether to perform case-sensitive matching
        """
        super().__init__(threshold)
        self.case_sensitive = case_sensitive
        self.name = "exact_match"
    
    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure exact match between actual and expected output
        
        Args:
            test_case: Test case with actual and expected outputs
            
        Returns:
            1.0 for exact match, 0.0 for no match
        """
        if not hasattr(test_case, 'expected_output') or test_case.expected_output is None:
            logger.warning("No expected output provided for exact match comparison")
            return 0.0
        
        actual = str(test_case.actual_output)
        expected = str(test_case.expected_output)
        
        # Handle case sensitivity
        if not self.case_sensitive:
            actual = actual.lower()
            expected = expected.lower()
        
        # Strip whitespace for comparison
        actual = actual.strip()
        expected = expected.strip()
        
        return 1.0 if actual == expected else 0.0
    
    def evaluate_batch(self, test_cases: List[LLMTestCase]) -> Dict[str, Any]:
        """Evaluate multiple test cases and return detailed metrics"""
        if not test_cases:
            return {"error": "No test cases provided"}
        
        scores = []
        matches = 0
        mismatches = []
        
        for i, test_case in enumerate(test_cases):
            score = self.measure(test_case)
            scores.append(score)
            
            if score == 1.0:
                matches += 1
            else:
                mismatches.append({
                    "index": i,
                    "input": str(test_case.input)[:100],
                    "actual": str(test_case.actual_output)[:200],
                    "expected": str(test_case.expected_output)[:200]
                })
        
        pass_rate = sum(scores) / len(scores)
        
        return {
            "metric": "exact_match",
            "pass_rate": pass_rate,
            "passes_threshold": pass_rate >= self.threshold,
            "total_cases": len(test_cases),
            "matches": matches,
            "mismatches": len(mismatches),
            "mismatch_examples": mismatches[:5],  # First 5 mismatches
            "case_sensitive": self.case_sensitive
        }

class StabilityMetric(BaseMetric):
    """N-run consistency check for prompt stability"""
    
    def __init__(self, 
                 n_runs: int = 5, 
                 temperature: float = 0.0,
                 threshold: float = 0.8,
                 llm_client: Optional['LLMClientEnhanced'] = None):
        """
        Initialize stability metric
        
        Args:
            n_runs: Number of runs to test consistency
            temperature: Temperature for LLM generation
            threshold: Minimum consistency score to pass
            llm_client: LLM client for generating responses
        """
        super().__init__(threshold)
        self.n_runs = n_runs
        self.temperature = temperature
        self.name = "stability"
        self.llm_client = llm_client
        
        # Initialize default client if none provided and available
        if self.llm_client is None and LLM_CLIENT_AVAILABLE:
            self.llm_client = LLMClientEnhanced(
                use_mock=True,  # Use mock for testing
                temperature=self.temperature
            )
    
    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure consistency across multiple runs
        
        Args:
            test_case: Test case with input prompt
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        if not self.llm_client:
            logger.error("No LLM client available for stability testing")
            return 0.0
        
        responses = []
        
        # Generate multiple responses
        for i in range(self.n_runs):
            try:
                result = self.llm_client.generate_with_options(
                    prompt=str(test_case.input),
                    temperature=self.temperature,
                    deterministic=(self.temperature == 0.0)
                )
                
                response = result.get("response", "")
                responses.append(response)
                
            except Exception as e:
                logger.error(f"Error in stability test run {i+1}: {e}")
                responses.append("")  # Empty response for failed runs
        
        # Calculate consistency
        if not responses:
            return 0.0
        
        # Remove empty responses
        valid_responses = [r for r in responses if r.strip()]
        
        if not valid_responses:
            return 0.0
        
        # Calculate uniqueness
        unique_responses = list(set(valid_responses))
        consistency_score = 1.0 - (len(unique_responses) - 1) / len(valid_responses)
        
        return max(0.0, consistency_score)
    
    def analyze_variance(self, test_cases: List[LLMTestCase]) -> Dict[str, Any]:
        """Analyze variance across multiple test cases"""
        if not test_cases:
            return {"error": "No test cases provided"}
        
        scores = []
        detailed_results = []
        
        for test_case in test_cases:
            score = self.measure(test_case)
            scores.append(score)
            
            detailed_results.append({
                "input": str(test_case.input)[:100],
                "consistency_score": score,
                "is_deterministic": score == 1.0
            })
        
        return {
            "metric": "stability",
            "mean_consistency": statistics.mean(scores) if scores else 0.0,
            "median_consistency": statistics.median(scores) if scores else 0.0,
            "min_consistency": min(scores) if scores else 0.0,
            "max_consistency": max(scores) if scores else 0.0,
            "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "deterministic_count": sum(1 for s in scores if s == 1.0),
            "total_cases": len(test_cases),
            "n_runs_per_case": self.n_runs,
            "temperature": self.temperature,
            "passes_threshold": statistics.mean(scores) >= self.threshold if scores else False,
            "detailed_results": detailed_results
        }

class SchemaValidityMetric(BaseMetric):
    """JSON schema validation metric"""
    
    def __init__(self, schema_path: str = None, schema_dict: Dict = None, threshold: float = 1.0):
        """
        Initialize schema validity metric
        
        Args:
            schema_path: Path to JSON schema file
            schema_dict: Schema dictionary (alternative to file)
            threshold: Minimum pass rate (typically 1.0 for strict validation)
        """
        super().__init__(threshold)
        self.name = "schema_validity"
        self.schema = self._load_schema(schema_path, schema_dict)
    
    def _load_schema(self, schema_path: Optional[str], schema_dict: Optional[Dict]) -> Optional[Dict]:
        """Load JSON schema from file or dictionary"""
        if schema_dict:
            return schema_dict
        
        if schema_path:
            try:
                with open(schema_path, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(f"Error loading schema from {schema_path}: {e}")
        
        # Default schema for basic validation
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True
        }
    
    def measure(self, test_case: LLMTestCase) -> float:
        """
        Validate JSON schema compliance
        
        Args:
            test_case: Test case with actual output to validate
            
        Returns:
            1.0 if valid, 0.0 if invalid
        """
        if not self.schema:
            logger.warning("No schema available for validation")
            return 0.0
        
        try:
            # Try to parse as JSON
            if isinstance(test_case.actual_output, str):
                output_data = json.loads(test_case.actual_output)
            else:
                output_data = test_case.actual_output
            
            # Validate against schema
            jsonschema.validate(output_data, self.schema)
            return 1.0
            
        except json.JSONDecodeError:
            logger.debug("Output is not valid JSON")
            return 0.0
        except jsonschema.ValidationError as e:
            logger.debug(f"Schema validation failed: {e.message}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error in schema validation: {e}")
            return 0.0
    
    def validate_batch(self, test_cases: List[LLMTestCase]) -> Dict[str, Any]:
        """Validate multiple test cases against schema"""
        if not test_cases:
            return {"error": "No test cases provided"}
        
        valid_count = 0
        validation_errors = []
        
        for i, test_case in enumerate(test_cases):
            score = self.measure(test_case)
            
            if score == 1.0:
                valid_count += 1
            else:
                # Capture validation error details
                try:
                    if isinstance(test_case.actual_output, str):
                        output_data = json.loads(test_case.actual_output)
                        jsonschema.validate(output_data, self.schema)
                    else:
                        jsonschema.validate(test_case.actual_output, self.schema)
                except json.JSONDecodeError:
                    validation_errors.append({
                        "index": i,
                        "error_type": "json_parse_error",
                        "output_preview": str(test_case.actual_output)[:200]
                    })
                except jsonschema.ValidationError as e:
                    validation_errors.append({
                        "index": i,
                        "error_type": "schema_validation_error",
                        "error_message": e.message,
                        "error_path": list(e.path) if e.path else []
                    })
                except Exception as e:
                    validation_errors.append({
                        "index": i,
                        "error_type": "unexpected_error",
                        "error_message": str(e)
                    })
        
        pass_rate = valid_count / len(test_cases)
        
        return {
            "metric": "schema_validity",
            "pass_rate": pass_rate,
            "passes_threshold": pass_rate >= self.threshold,
            "total_cases": len(test_cases),
            "valid_count": valid_count,
            "invalid_count": len(test_cases) - valid_count,
            "validation_errors": validation_errors[:10],  # First 10 errors
            "schema_loaded": self.schema is not None
        }

class LatencyMetric(BaseMetric):
    """Response latency measurement metric"""
    
    def __init__(self, max_latency_ms: float = 2000, threshold: float = 0.95):
        """
        Initialize latency metric
        
        Args:
            max_latency_ms: Maximum acceptable latency in milliseconds
            threshold: Minimum pass rate for latency compliance
        """
        super().__init__(threshold)
        self.max_latency_ms = max_latency_ms
        self.name = "latency"
    
    def measure(self, test_case: LLMTestCase) -> float:
        """
        Check if response latency is within acceptable limits
        
        Args:
            test_case: Test case (should have latency_ms attribute)
            
        Returns:
            1.0 if within limits, 0.0 if exceeded
        """
        # Look for latency in various possible attributes
        latency_ms = None
        
        if hasattr(test_case, 'latency_ms'):
            latency_ms = test_case.latency_ms
        elif hasattr(test_case, 'metadata') and isinstance(test_case.metadata, dict):
            latency_ms = test_case.metadata.get('latency_ms')
        elif hasattr(test_case, 'response_time_ms'):
            latency_ms = test_case.response_time_ms
        
        if latency_ms is None:
            logger.warning("No latency information available in test case")
            return 0.0
        
        return 1.0 if latency_ms <= self.max_latency_ms else 0.0
    
    def analyze_performance(self, test_cases: List[LLMTestCase]) -> Dict[str, Any]:
        """Analyze performance across multiple test cases"""
        if not test_cases:
            return {"error": "No test cases provided"}
        
        latencies = []
        within_limit_count = 0
        
        for test_case in test_cases:
            # Extract latency
            latency_ms = None
            if hasattr(test_case, 'latency_ms'):
                latency_ms = test_case.latency_ms
            elif hasattr(test_case, 'metadata') and isinstance(test_case.metadata, dict):
                latency_ms = test_case.metadata.get('latency_ms')
            elif hasattr(test_case, 'response_time_ms'):
                latency_ms = test_case.response_time_ms
            
            if latency_ms is not None:
                latencies.append(latency_ms)
                if latency_ms <= self.max_latency_ms:
                    within_limit_count += 1
        
        if not latencies:
            return {"error": "No latency data found in test cases"}
        
        pass_rate = within_limit_count / len(latencies)
        
        return {
            "metric": "latency",
            "pass_rate": pass_rate,
            "passes_threshold": pass_rate >= self.threshold,
            "max_latency_limit_ms": self.max_latency_ms,
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))],
            "within_limit_count": within_limit_count,
            "total_cases": len(latencies),
            "exceeded_limit_count": len(latencies) - within_limit_count
        }

class MetricsRunner:
    """Utility class to run multiple metrics on test cases"""
    
    def __init__(self):
        self.metrics = {}
    
    def add_metric(self, name: str, metric: BaseMetric):
        """Add a metric to the runner"""
        self.metrics[name] = metric
    
    def run_all_metrics(self, test_cases: List[LLMTestCase]) -> Dict[str, Any]:
        """Run all registered metrics on test cases"""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_test_cases": len(test_cases),
            "metrics_run": len(self.metrics),
            "metric_results": {},
            "overall_summary": {}
        }
        
        # Run each metric
        for metric_name, metric in self.metrics.items():
            try:
                if hasattr(metric, 'evaluate_batch'):
                    metric_result = metric.evaluate_batch(test_cases)
                elif hasattr(metric, 'validate_batch'):
                    metric_result = metric.validate_batch(test_cases)
                elif hasattr(metric, 'analyze_performance'):
                    metric_result = metric.analyze_performance(test_cases)
                else:
                    # Fallback to individual measurement
                    scores = [metric.measure(tc) for tc in test_cases]
                    metric_result = {
                        "metric": metric_name,
                        "scores": scores,
                        "mean_score": statistics.mean(scores) if scores else 0.0,
                        "pass_rate": sum(1 for s in scores if s >= metric.threshold) / len(scores) if scores else 0.0
                    }
                
                results["metric_results"][metric_name] = metric_result
                
            except Exception as e:
                logger.error(f"Error running metric {metric_name}: {e}")
                results["metric_results"][metric_name] = {
                    "error": str(e),
                    "metric": metric_name
                }
        
        # Calculate overall summary
        passing_metrics = 0
        total_metrics = 0
        
        for metric_name, result in results["metric_results"].items():
            if "error" not in result:
                total_metrics += 1
                if result.get("passes_threshold", False):
                    passing_metrics += 1
        
        results["overall_summary"] = {
            "passing_metrics": passing_metrics,
            "total_metrics": total_metrics,
            "overall_pass_rate": passing_metrics / total_metrics if total_metrics > 0 else 0.0,
            "all_metrics_passing": passing_metrics == total_metrics
        }
        
        return results

# Example usage and testing
if __name__ == "__main__":
    print("=== Enhanced Metrics Testing ===")
    
    # Create test cases
    test_cases = [
        LLMTestCase(
            input="What is the capital of France?",
            actual_output='{"capital": "Paris", "confidence": 1.0}',
            expected_output='{"capital": "Paris", "confidence": 1.0}',
            latency_ms=150
        ),
        LLMTestCase(
            input="What is the capital of Japan?",
            actual_output='{"capital": "Tokyo", "confidence": 0.95}',
            expected_output='{"capital": "Tokyo", "confidence": 0.95}',
            latency_ms=200
        )
    ]
    
    # Test exact match metric
    exact_match = ExactMatchMetric(threshold=0.9)
    exact_results = exact_match.evaluate_batch(test_cases)
    print(f"Exact Match Results: {exact_results['pass_rate']:.2f}")
    
    # Test schema validation
    schema = {
        "type": "object",
        "properties": {
            "capital": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["capital", "confidence"]
    }
    
    schema_metric = SchemaValidityMetric(schema_dict=schema)
    schema_results = schema_metric.validate_batch(test_cases)
    print(f"Schema Validation: {schema_results['pass_rate']:.2f}")
    
    # Test latency metric
    latency_metric = LatencyMetric(max_latency_ms=300)
    latency_results = latency_metric.analyze_performance(test_cases)
    print(f"Latency Performance: {latency_results['pass_rate']:.2f}")
    
    # Test metrics runner
    runner = MetricsRunner()
    runner.add_metric("exact_match", exact_match)
    runner.add_metric("schema_validity", schema_metric)
    runner.add_metric("latency", latency_metric)
    
    all_results = runner.run_all_metrics(test_cases)
    print(f"Overall Results: {all_results['overall_summary']['overall_pass_rate']:.2f}")