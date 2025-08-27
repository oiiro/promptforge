"""
Metrics collection and monitoring for prompt orchestration
Provides comprehensive business and technical metrics
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import threading
import statistics

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: str
    tags: Dict[str, str]
    metric_type: str  # counter, gauge, histogram

class MetricsCollector:
    """Collects and aggregates metrics for prompt operations"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.gauges = {}
        self.histograms = defaultdict(list)
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # Business metrics
        self.request_count = 0
        self.error_count = 0
        self.guardrail_violations = 0
        self.response_times = []
        self.confidence_scores = []
        self.provider_stats = defaultdict(lambda: {"requests": 0, "errors": 0, "latency": []})
        
    def increment(self, metric_name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        with self.lock:
            key = self._metric_key(metric_name, tags)
            self.counters[key] += value
            
            # Record metric point
            self.metrics[metric_name].append(MetricPoint(
                name=metric_name,
                value=value,
                timestamp=datetime.utcnow().isoformat(),
                tags=tags or {},
                metric_type="counter"
            ))
            
            # Update business metrics
            if metric_name == "successful_requests":
                self.request_count += value
            elif metric_name == "failed_requests":
                self.error_count += value
            elif metric_name == "guardrail_violations":
                self.guardrail_violations += value
    
    def gauge(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric"""
        with self.lock:
            key = self._metric_key(metric_name, tags)
            self.gauges[key] = value
            
            # Record metric point
            self.metrics[metric_name].append(MetricPoint(
                name=metric_name,
                value=value,
                timestamp=datetime.utcnow().isoformat(),
                tags=tags or {},
                metric_type="gauge"
            ))
    
    def record(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value"""
        with self.lock:
            key = self._metric_key(metric_name, tags)
            self.histograms[key].append(value)
            
            # Record metric point
            self.metrics[metric_name].append(MetricPoint(
                name=metric_name,
                value=value,
                timestamp=datetime.utcnow().isoformat(),
                tags=tags or {},
                metric_type="histogram"
            ))
            
            # Update business metrics
            if metric_name == "request_latency":
                self.response_times.append(value)
            elif metric_name == "confidence_score":
                self.confidence_scores.append(value)
    
    def _metric_key(self, name: str, tags: Dict[str, str] = None) -> str:
        """Generate metric key from name and tags"""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def record_llm_call(self, provider: str, model: str, latency_ms: float, 
                       tokens: Dict[str, int], success: bool):
        """Record LLM-specific metrics"""
        tags = {"provider": provider, "model": model}
        
        # Record latency
        self.record("llm_latency", latency_ms, tags)
        
        # Record token usage
        if "total" in tokens:
            self.record("llm_tokens", tokens["total"], tags)
        
        # Record success/failure
        if success:
            self.increment("llm_success", tags=tags)
        else:
            self.increment("llm_error", tags=tags)
        
        # Update provider stats
        with self.lock:
            self.provider_stats[provider]["requests"] += 1
            self.provider_stats[provider]["latency"].append(latency_ms)
            if not success:
                self.provider_stats[provider]["errors"] += 1
    
    def record_guardrail_check(self, stage: str, rule: str, passed: bool, severity: str = None):
        """Record guardrail validation metrics"""
        tags = {"stage": stage, "rule": rule}
        
        if passed:
            self.increment("guardrail_pass", tags=tags)
        else:
            self.increment("guardrail_violation", tags=tags)
            if severity:
                tags["severity"] = severity
                self.increment("guardrail_violation_by_severity", tags=tags)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self.lock:
            uptime_seconds = time.time() - self.start_time
            
            # Calculate business metrics
            total_requests = self.request_count + self.error_count
            error_rate = self.error_count / total_requests if total_requests > 0 else 0
            
            # Response time statistics
            response_stats = {}
            if self.response_times:
                response_stats = {
                    "mean": statistics.mean(self.response_times),
                    "median": statistics.median(self.response_times),
                    "p95": self._percentile(self.response_times, 0.95),
                    "p99": self._percentile(self.response_times, 0.99),
                    "min": min(self.response_times),
                    "max": max(self.response_times)
                }
            
            # Confidence score statistics
            confidence_stats = {}
            if self.confidence_scores:
                confidence_stats = {
                    "mean": statistics.mean(self.confidence_scores),
                    "median": statistics.median(self.confidence_scores),
                    "low_confidence_rate": sum(1 for s in self.confidence_scores if s < 0.8) / len(self.confidence_scores)
                }
            
            # Provider statistics
            provider_summary = {}
            for provider, stats in self.provider_stats.items():
                if stats["requests"] > 0:
                    avg_latency = statistics.mean(stats["latency"]) if stats["latency"] else 0
                    provider_error_rate = stats["errors"] / stats["requests"]
                    
                    provider_summary[provider] = {
                        "requests": stats["requests"],
                        "error_rate": provider_error_rate,
                        "avg_latency_ms": avg_latency,
                        "availability": 1 - provider_error_rate
                    }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": uptime_seconds,
                "business_metrics": {
                    "total_requests": total_requests,
                    "successful_requests": self.request_count,
                    "failed_requests": self.error_count,
                    "error_rate": error_rate,
                    "guardrail_violations": self.guardrail_violations,
                    "requests_per_minute": total_requests / (uptime_seconds / 60) if uptime_seconds > 0 else 0
                },
                "performance_metrics": {
                    "response_time": response_stats,
                    "confidence_scores": confidence_stats
                },
                "provider_metrics": provider_summary,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "thresholds": self._check_thresholds()
            }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value"""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        
        if index >= len(sorted_data):
            return sorted_data[-1]
        
        return sorted_data[index]
    
    def _check_thresholds(self) -> Dict[str, Any]:
        """Check if metrics exceed defined thresholds"""
        thresholds = {
            "error_rate": {"threshold": 0.05, "status": "ok"},
            "response_time_p95": {"threshold": 2000, "status": "ok"},
            "low_confidence_rate": {"threshold": 0.2, "status": "ok"},
            "guardrail_violation_rate": {"threshold": 0.01, "status": "ok"}
        }
        
        total_requests = self.request_count + self.error_count
        
        # Check error rate
        if total_requests > 0:
            error_rate = self.error_count / total_requests
            if error_rate > 0.05:
                thresholds["error_rate"]["status"] = "warning"
        
        # Check response time
        if self.response_times:
            p95_latency = self._percentile(self.response_times, 0.95)
            if p95_latency > 2000:
                thresholds["response_time_p95"]["status"] = "warning"
        
        # Check confidence scores
        if self.confidence_scores:
            low_conf_rate = sum(1 for s in self.confidence_scores if s < 0.8) / len(self.confidence_scores)
            if low_conf_rate > 0.2:
                thresholds["low_confidence_rate"]["status"] = "warning"
        
        # Check guardrail violations
        if total_requests > 0:
            violation_rate = self.guardrail_violations / total_requests
            if violation_rate > 0.01:
                thresholds["guardrail_violation_rate"]["status"] = "critical"
        
        return thresholds
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        timestamp = int(time.time() * 1000)
        
        # Counters
        for key, value in self.counters.items():
            lines.append(f"prompt_counter_{key} {value} {timestamp}")
        
        # Gauges
        for key, value in self.gauges.items():
            lines.append(f"prompt_gauge_{key} {value} {timestamp}")
        
        # Histograms (simplified)
        for key, values in self.histograms.items():
            if values:
                lines.append(f"prompt_histogram_{key}_sum {sum(values)} {timestamp}")
                lines.append(f"prompt_histogram_{key}_count {len(values)} {timestamp}")
                lines.append(f"prompt_histogram_{key}_avg {statistics.mean(values)} {timestamp}")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all metrics (for testing)"""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.request_count = 0
            self.error_count = 0
            self.guardrail_violations = 0
            self.response_times.clear()
            self.confidence_scores.clear()
            self.provider_stats.clear()
            self.start_time = time.time()
    
    def get_health_score(self) -> Dict[str, Any]:
        """Calculate overall health score"""
        thresholds = self._check_thresholds()
        
        # Count threshold violations
        violations = sum(1 for t in thresholds.values() if t["status"] != "ok")
        total_checks = len(thresholds)
        
        # Calculate health score (0-100)
        health_score = max(0, 100 - (violations * 25))
        
        # Determine overall status
        if violations == 0:
            status = "healthy"
        elif any(t["status"] == "critical" for t in thresholds.values()):
            status = "critical"
        else:
            status = "warning"
        
        return {
            "health_score": health_score,
            "status": status,
            "threshold_violations": violations,
            "total_checks": total_checks,
            "details": thresholds
        }
    
    def export_json(self, filepath: str):
        """Export metrics to JSON file"""
        summary = self.get_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")

# Global metrics collector instance
metrics_collector = MetricsCollector()

# Utility functions
def increment_counter(name: str, value: int = 1, **tags):
    """Convenience function to increment counter"""
    metrics_collector.increment(name, value, tags)

def record_histogram(name: str, value: float, **tags):
    """Convenience function to record histogram value"""
    metrics_collector.record(name, value, tags)

def set_gauge(name: str, value: float, **tags):
    """Convenience function to set gauge"""
    metrics_collector.gauge(name, value, tags)

# Example usage
if __name__ == "__main__":
    # Simulate some metrics
    collector = MetricsCollector()
    
    # Record some requests
    collector.increment("successful_requests", 95)
    collector.increment("failed_requests", 5)
    
    # Record response times
    import random
    for _ in range(100):
        collector.record("request_latency", random.uniform(100, 2000))
        collector.record("confidence_score", random.uniform(0.6, 1.0))
    
    # Record LLM calls
    collector.record_llm_call(
        provider="openai",
        model="gpt-4",
        latency_ms=1200,
        tokens={"prompt": 150, "completion": 50, "total": 200},
        success=True
    )
    
    # Get summary
    summary = collector.get_summary()
    print(json.dumps(summary, indent=2))
    
    # Health score
    health = collector.get_health_score()
    print(f"\nHealth Score: {health}")
    
    # Export to file
    collector.export_json("metrics_sample.json")