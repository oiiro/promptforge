"""
Distributed tracing for prompt orchestration
Integrates with OpenTelemetry for comprehensive observability
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"OpenTelemetry not available: {e}. Tracing features disabled.")
    OPENTELEMETRY_AVAILABLE = False
    
    # Mock classes for graceful fallback
    class MockTracer:
        def start_span(self, *args, **kwargs):
            return MockSpan()
    
    class MockSpan:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def set_attribute(self, *args, **kwargs):
            pass
        def set_status(self, *args, **kwargs):
            pass
        def record_exception(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)

class TracingManager:
    """Manages distributed tracing for prompt operations"""
    
    def __init__(self, service_name: str = "prompt-orchestration"):
        """Initialize tracing with OpenTelemetry"""
        self.service_name = service_name
        self.tracer = self._initialize_tracer()
        
        # Instrument HTTP requests automatically if OpenTelemetry is available
        if OPENTELEMETRY_AVAILABLE:
            try:
                RequestsInstrumentor().instrument()
            except Exception as e:
                logger.warning(f"Failed to instrument requests: {e}")
        else:
            logger.info("OpenTelemetry not available. Tracing disabled.")
        
    def _initialize_tracer(self):
        """Set up OpenTelemetry tracer"""
        if not OPENTELEMETRY_AVAILABLE:
            logger.info("OpenTelemetry not available. Using mock tracer.")
            return MockTracer()
            
        try:
            # Create resource with service information
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": os.getenv("VERSION", "1.0.0"),
                "deployment.environment": os.getenv("ENVIRONMENT", "development")
            })
            
            # Create tracer provider
            provider = TracerProvider(resource=resource)
            
            # Add exporters
            if os.getenv("ENABLE_TRACING") == "true":
                # OTLP exporter for production
                otlp_exporter = OTLPSpanExporter(
                    endpoint=os.getenv("TRACING_ENDPOINT", "localhost:4317"),
                    insecure=True
                )
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            
            # Always add console exporter in development
            if os.getenv("ENVIRONMENT") == "development":
                console_exporter = ConsoleSpanExporter()
                provider.add_span_processor(BatchSpanProcessor(console_exporter))
            
            # Set global tracer provider
            trace.set_tracer_provider(provider)
            
            return trace.get_tracer(__name__)
            
        except Exception as e:
            logger.warning(f"Failed to initialize OpenTelemetry tracer: {e}. Using mock tracer.")
            return MockTracer()
    
    def start_span(self, name: str, request_id: str = None) -> Any:
        """Start a new trace span"""
        span = self.tracer.start_span(name)
        
        # Add standard attributes
        span.set_attribute("request.id", request_id or "unknown")
        span.set_attribute("service.name", self.service_name)
        span.set_attribute("span.start_time", datetime.utcnow().isoformat())
        
        return span
    
    def end_span(self, span: Any, metadata: Dict[str, Any] = None):
        """End a trace span with metadata"""
        if metadata:
            for key, value in metadata.items():
                # OpenTelemetry attributes must be strings, numbers, or booleans
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"custom.{key}", value)
                else:
                    span.set_attribute(f"custom.{key}", json.dumps(value))
        
        # Set status based on metadata
        if metadata and metadata.get("status") == "error":
            span.set_status(Status(StatusCode.ERROR, metadata.get("error", "Unknown error")))
        else:
            span.set_status(Status(StatusCode.OK))
        
        span.end()
    
    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Context manager for tracing operations"""
        span = self.start_span(operation_name)
        
        # Add attributes
        for key, value in attributes.items():
            span.set_attribute(key, value)
        
        start_time = time.time()
        
        try:
            yield span
            
            # Success
            elapsed_ms = (time.time() - start_time) * 1000
            span.set_attribute("duration_ms", elapsed_ms)
            span.set_status(Status(StatusCode.OK))
            
        except Exception as e:
            # Error
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
            
        finally:
            span.end()
    
    def trace_llm_call(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int):
        """Specialized tracing for LLM calls"""
        with self.trace_operation(
            "llm_call",
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        ) as span:
            return span
    
    def trace_guardrail_check(self, stage: str, passed: bool, violations: int = 0):
        """Trace guardrail validations"""
        with self.trace_operation(
            f"guardrail_{stage}",
            stage=stage,
            passed=passed,
            violation_count=violations
        ) as span:
            return span