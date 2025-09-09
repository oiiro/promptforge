"""
Langfuse Configuration and Integration for PromptForge
Production-grade LLM observability and prompt management
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import langfuse
from langfuse import Langfuse, observe
from langfuse.model import CreateScore
import structlog

logger = structlog.get_logger()

class ObservabilityLevel(Enum):
    """Observability detail levels"""
    MINIMAL = "minimal"      # Basic tracing only
    STANDARD = "standard"    # Tracing + metrics
    DETAILED = "detailed"    # Full observability with spans
    DEBUG = "debug"         # Everything including debug info

@dataclass
class LangfuseConfig:
    """Langfuse configuration for PromptForge"""
    
    # Connection settings
    public_key: str
    secret_key: str
    host: str = "https://cloud.langfuse.com"
    
    # Behavior settings
    enabled: bool = True
    observability_level: ObservabilityLevel = ObservabilityLevel.STANDARD
    flush_at: int = 10
    flush_interval: float = 10.0
    max_retries: int = 3
    timeout: int = 30
    
    # Feature flags
    enable_prompt_caching: bool = True
    enable_cost_tracking: bool = True
    enable_latency_tracking: bool = True
    enable_quality_scoring: bool = True
    enable_dataset_upload: bool = True
    
    # Sampling configuration
    sampling_rate: float = 1.0  # 1.0 = 100% sampling
    
    @classmethod
    def from_env(cls) -> "LangfuseConfig":
        """Load configuration from environment variables"""
        return cls(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            enabled=os.getenv("LANGFUSE_ENABLED", "true").lower() == "true",
            observability_level=ObservabilityLevel(
                os.getenv("LANGFUSE_OBSERVABILITY_LEVEL", "standard")
            ),
            sampling_rate=float(os.getenv("LANGFUSE_SAMPLING_RATE", "1.0"))
        )

class LangfuseObserver:
    """Central Langfuse observer for PromptForge"""
    
    def __init__(self, config: Optional[LangfuseConfig] = None):
        self.config = config or LangfuseConfig.from_env()
        self.client = None
        
        if self.config.enabled:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Langfuse client"""
        try:
            self.client = Langfuse(
                public_key=self.config.public_key,
                secret_key=self.config.secret_key,
                host=self.config.host,
                flush_at=self.config.flush_at,
                flush_interval=self.config.flush_interval,
                max_retries=self.config.max_retries,
                timeout=self.config.timeout,
                enabled=self.config.enabled
            )
            logger.info("Langfuse client initialized successfully", 
                       host=self.config.host)
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            self.config.enabled = False
    
    def create_trace(self, 
                    name: str,
                    metadata: Optional[Dict[str, Any]] = None,
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    version: Optional[str] = None,
                    release: Optional[str] = None,
                    tags: Optional[List[str]] = None) -> Any:
        """Create a new trace"""
        if not self.config.enabled or not self.client:
            return None
        
        try:
            trace = self.client.trace(
                name=name,
                metadata=metadata or {},
                user_id=user_id,
                session_id=session_id,
                version=version,
                release=release,
                tags=tags or []
            )
            
            if self.config.observability_level in [ObservabilityLevel.DETAILED, ObservabilityLevel.DEBUG]:
                logger.debug(f"Trace created: {trace.id}", name=name)
            
            return trace
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return None
    
    def score_trace(self,
                   trace_id: str,
                   name: str,
                   value: float,
                   comment: Optional[str] = None,
                   data_type: str = "NUMERIC") -> bool:
        """Add a score to a trace"""
        if not self.config.enabled or not self.client:
            return False
        
        try:
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
                data_type=data_type
            )
            
            logger.debug(f"Score added to trace {trace_id}: {name}={value}")
            return True
        except Exception as e:
            logger.error(f"Failed to score trace: {e}")
            return False
    
    def log_prompt_version(self,
                          name: str,
                          prompt: str,
                          version: str,
                          config: Optional[Dict[str, Any]] = None,
                          labels: Optional[List[str]] = None) -> bool:
        """Log a prompt version to Langfuse"""
        if not self.config.enabled or not self.client:
            return False
        
        try:
            self.client.create_prompt(
                name=name,
                prompt=prompt,
                version=version,
                config=config or {},
                labels=labels or []
            )
            
            logger.info(f"Prompt version logged: {name} v{version}")
            return True
        except Exception as e:
            logger.error(f"Failed to log prompt version: {e}")
            return False
    
    def create_dataset(self,
                      name: str,
                      description: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a new dataset in Langfuse"""
        if not self.config.enabled or not self.client:
            return None
        
        try:
            dataset = self.client.create_dataset(
                name=name,
                description=description,
                metadata=metadata or {}
            )
            logger.info(f"Dataset created: {name} (ID: {dataset.id})")
            return dataset.id
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            return None
    
    def add_dataset_item(self,
                        dataset_name: str,
                        input_data: Any,
                        expected_output: Optional[Any] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add an item to a dataset"""
        if not self.config.enabled or not self.client:
            return False
        
        try:
            self.client.create_dataset_item(
                dataset_name=dataset_name,
                input=input_data,
                expected_output=expected_output,
                metadata=metadata or {}
            )
            logger.debug(f"Item added to dataset: {dataset_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add dataset item: {e}")
            return False
    
    def track_generation(self,
                        name: str,
                        input_text: str,
                        output_text: str,
                        model: str,
                        prompt_tokens: Optional[int] = None,
                        completion_tokens: Optional[int] = None,
                        total_tokens: Optional[int] = None,
                        latency_ms: Optional[float] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Track a generation event with full metrics"""
        if not self.config.enabled or not self.client:
            return None
        
        try:
            generation = self.client.generation(
                name=name,
                input=input_text,
                output=output_text,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency=latency_ms,
                metadata=metadata or {}
            )
            
            if self.config.enable_cost_tracking and total_tokens:
                # Calculate and log cost (example rates)
                cost = self._calculate_cost(model, total_tokens)
                self.client.score(
                    trace_id=generation.trace_id,
                    name="cost_usd",
                    value=cost,
                    data_type="NUMERIC"
                )
            
            return generation.id
        except Exception as e:
            logger.error(f"Failed to track generation: {e}")
            return None
    
    def _calculate_cost(self, model: str, tokens: int) -> float:
        """Calculate cost based on model and token count"""
        # Example pricing (adjust based on actual provider rates)
        cost_per_1k_tokens = {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.002,
            "claude-3-opus": 0.015,
            "claude-3-sonnet": 0.003,
            "claude-3-haiku": 0.00025
        }
        
        rate = cost_per_1k_tokens.get(model, 0.01)
        return (tokens / 1000.0) * rate
    
    def flush(self):
        """Flush pending traces"""
        if self.client:
            self.client.flush()
    
    def shutdown(self):
        """Gracefully shutdown the observer"""
        if self.client:
            self.client.flush()
            self.client.shutdown()
            logger.info("Langfuse observer shutdown complete")

# Global observer instance
langfuse_observer = LangfuseObserver()

# Export decorators for direct use
__all__ = [
    "LangfuseConfig",
    "LangfuseObserver", 
    "langfuse_observer",
    "observe",
    "ObservabilityLevel"
]