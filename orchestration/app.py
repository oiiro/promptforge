"""
Production-grade Prompt Orchestration API
Financial services compliant FastAPI application with comprehensive monitoring
"""

import os
import json
import time
import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
import uvicorn
from dotenv import load_dotenv

# Import our modules
from llm_client import LLMClient
import sys
sys.path.append(str(Path(__file__).parent.parent))
from guardrails.validators import GuardrailOrchestrator
from observability.tracing import TracingManager
from observability.metrics import MetricsCollector

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Request/Response Models
class CapitalRequest(BaseModel):
    """Request model for capital finder"""
    country: str = Field(..., min_length=1, max_length=100)
    provider: Optional[str] = Field(None, description="LLM provider override")
    model: Optional[str] = Field(None, description="Model override")
    
    @field_validator('country')
    @classmethod
    def validate_country(cls, v):
        """Validate country input"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Country cannot be empty")
        return v.strip()

class CapitalResponse(BaseModel):
    """Response model for capital finder"""
    capital: str
    confidence: float = Field(..., ge=0, le=1)
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    environment: str
    providers: Dict[str, bool]
    timestamp: str

class EvaluationRequest(BaseModel):
    """Request model for evaluation"""
    dataset: str = Field("golden", description="Dataset to evaluate")
    provider: Optional[str] = None
    
class VersionInfo(BaseModel):
    """Version information for rollback support"""
    prompt_version: str
    guardrails_version: str
    model_version: str
    deployed_at: str

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("Starting Prompt Orchestration Service...")
    
    # Initialize components
    app.state.llm_client = LLMClient()
    app.state.guardrails = GuardrailOrchestrator()
    app.state.tracing = TracingManager() if os.getenv("ENABLE_TRACING") == "true" else None
    app.state.metrics = MetricsCollector()
    
    # Load version info
    app.state.version = VersionInfo(
        prompt_version=os.getenv("PROMPT_VERSION", "1.0.0"),
        guardrails_version="1.0.0",
        model_version=os.getenv("DEFAULT_MODEL", "gpt-4"),
        deployed_at=datetime.utcnow().isoformat()
    )
    
    # Warm up providers
    try:
        await app.state.llm_client.health_check()
        logger.info("LLM provider warmed up successfully")
    except Exception as e:
        logger.warning(f"Failed to warm up LLM provider: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Prompt Orchestration Service...")
    
    # Export metrics
    if app.state.metrics:
        metrics_report = app.state.metrics.get_summary()
        logger.info(f"Final metrics: {json.dumps(metrics_report, indent=2)}")

# Initialize FastAPI app
app = FastAPI(
    title="Prompt Orchestration Service",
    description="Financial services grade prompt management and orchestration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID for tracing"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response

# Rate limiting middleware
from collections import defaultdict
from datetime import datetime

rate_limit_storage = defaultdict(list)

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Simple rate limiting"""
    client_ip = request.client.host
    now = datetime.now()
    
    # Clean old entries
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage[client_ip]
        if now - timestamp < timedelta(minutes=1)
    ]
    
    # Check rate limit (100 requests per minute)
    if len(rate_limit_storage[client_ip]) >= 100:
        return Response(
            content=json.dumps({"error": "Rate limit exceeded"}),
            status_code=429,
            headers={"Retry-After": "60"}
        )
    
    rate_limit_storage[client_ip].append(now)
    
    response = await call_next(request)
    return response

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    # Simple token verification - implement proper auth in production
    expected_token = os.getenv("API_TOKEN", "demo-token")
    
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    
    return credentials.credentials

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    providers_health = {}
    
    # Check each provider
    for provider in ["openai", "anthropic", "huggingface", "airia"]:
        try:
            app.state.llm_client.switch_provider(provider)
            providers_health[provider] = app.state.llm_client.health_check()["healthy"]
        except:
            providers_health[provider] = False
    
    # Reset to default provider
    app.state.llm_client.switch_provider(os.getenv("DEFAULT_LLM_PROVIDER", "openai"))
    
    return HealthResponse(
        status="healthy" if any(providers_health.values()) else "unhealthy",
        version=app.state.version.prompt_version,
        environment=os.getenv("ENVIRONMENT", "development"),
        providers=providers_health,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/api/v1/capital", response_model=CapitalResponse)
async def find_capital(
    request: CapitalRequest,
    background_tasks: BackgroundTasks,
    req: Request,
    token: str = Depends(verify_token)
):
    """
    Find capital city for given country
    Includes comprehensive guardrails and monitoring
    """
    start_time = time.time()
    request_id = req.state.request_id
    
    # Start tracing
    span = None
    if app.state.tracing:
        span = app.state.tracing.start_span("find_capital", request_id)
    
    try:
        # Pre-execution guardrails
        is_valid, sanitized_input, violations = app.state.guardrails.validate_request(request.country)
        
        if not is_valid:
            # Log security violation
            logger.warning(f"Request {request_id} failed pre-validation: {violations}")
            
            if app.state.metrics:
                app.state.metrics.increment("guardrail_violations", 
                                          tags={"stage": "pre", "severity": "high"})
            
            raise HTTPException(
                status_code=400,
                detail=f"Input validation failed: {[v.message for v in violations if v.severity == 'high']}"
            )
        
        # Switch provider if specified
        if request.provider:
            app.state.llm_client.switch_provider(request.provider, request.model)
        
        # Generate response
        llm_response = app.state.llm_client.generate(sanitized_input)
        
        # Post-execution guardrails
        is_valid, violations = app.state.guardrails.validate_response(llm_response, sanitized_input)
        
        if not is_valid:
            logger.warning(f"Request {request_id} failed post-validation: {violations}")
            
            if app.state.metrics:
                app.state.metrics.increment("guardrail_violations",
                                          tags={"stage": "post", "severity": "high"})
            
            raise HTTPException(
                status_code=500,
                detail="Response validation failed"
            )
        
        # Parse response
        response_json = json.loads(llm_response)
        
        # Record metrics
        elapsed_time = (time.time() - start_time) * 1000
        
        if app.state.metrics:
            app.state.metrics.record("request_latency", elapsed_time)
            app.state.metrics.increment("successful_requests")
            
            # Check against thresholds
            if elapsed_time > 2000:
                app.state.metrics.increment("slow_requests")
            
            if response_json.get("confidence", 0) < 0.8:
                app.state.metrics.increment("low_confidence_responses")
        
        # Background audit logging
        background_tasks.add_task(
            audit_log,
            request_id,
            request.country,
            response_json,
            elapsed_time
        )
        
        # Complete tracing
        if span:
            app.state.tracing.end_span(span, {"status": "success", "latency_ms": elapsed_time})
        
        return CapitalResponse(**response_json)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request {request_id} failed: {e}")
        
        if app.state.metrics:
            app.state.metrics.increment("failed_requests")
        
        if span:
            app.state.tracing.end_span(span, {"status": "error", "error": str(e)})
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/v1/evaluate")
async def evaluate_prompt(
    request: EvaluationRequest,
    token: str = Depends(verify_token)
):
    """Run evaluation suite on specified dataset"""
    try:
        # Import evaluation module
        from evals.test_find_capital import CapitalFinderEvaluator
        
        # Run evaluation
        evaluator = CapitalFinderEvaluator()
        
        if request.provider:
            app.state.llm_client.switch_provider(request.provider)
        
        # Run appropriate test suite
        if request.dataset == "golden":
            _, score = evaluator.run_golden_dataset_tests()
            return {"dataset": "golden", "exact_match_score": score, "passed": score >= 0.95}
        elif request.dataset == "adversarial":
            pass_rate, violations = evaluator.run_adversarial_tests()
            return {"dataset": "adversarial", "pass_rate": pass_rate, "passed": pass_rate >= 0.95}
        else:
            raise ValueError(f"Unknown dataset: {request.dataset}")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/metrics")
async def get_metrics(token: str = Depends(verify_token)):
    """Get current metrics"""
    if not app.state.metrics:
        raise HTTPException(status_code=503, detail="Metrics not available")
    
    return app.state.metrics.get_summary()

@app.get("/api/v1/audit-log")
async def get_audit_log(
    limit: int = 100,
    token: str = Depends(verify_token)
):
    """Get audit log entries"""
    audit_entries = app.state.guardrails.get_audit_log()
    
    # Return latest entries
    return {
        "total": len(audit_entries),
        "entries": audit_entries[-limit:]
    }

@app.get("/api/v1/version")
async def get_version():
    """Get version information"""
    return app.state.version

@app.post("/api/v1/rollback")
async def rollback_version(
    target_version: str,
    token: str = Depends(verify_token)
):
    """Rollback to previous prompt version"""
    # In production, this would:
    # 1. Load previous prompt template
    # 2. Load previous guardrail configuration
    # 3. Update version info
    # 4. Clear caches
    
    logger.info(f"Rolling back to version {target_version}")
    
    # Simulate rollback
    app.state.version.prompt_version = target_version
    app.state.version.deployed_at = datetime.utcnow().isoformat()
    
    return {
        "status": "rolled_back",
        "version": target_version,
        "timestamp": datetime.utcnow().isoformat()
    }

# Utility functions
async def audit_log(request_id: str, input_text: str, output: Dict, latency_ms: float):
    """Background task for audit logging"""
    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_text,
        "output": output.get("capital"),
        "confidence": output.get("confidence"),
        "latency_ms": latency_ms,
        "model": output.get("metadata", {}).get("model"),
        "provider": output.get("metadata", {}).get("provider")
    }
    
    # Write to audit log file
    log_path = Path("logs/audit.jsonl")
    log_path.parent.mkdir(exist_ok=True)
    
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# Feature flags for gradual rollout
class FeatureFlags:
    """Simple feature flag system"""
    
    def __init__(self):
        self.flags = {
            "enhanced_guardrails": True,
            "multi_provider": True,
            "adversarial_defense": True,
            "detailed_metrics": True
        }
    
    def is_enabled(self, feature: str, user_id: str = None) -> bool:
        """Check if feature is enabled"""
        # Could implement percentage rollout based on user_id hash
        return self.flags.get(feature, False)

# Initialize feature flags
feature_flags = FeatureFlags()

@app.get("/api/v1/feature-flags")
async def get_feature_flags(token: str = Depends(verify_token)):
    """Get current feature flags"""
    return feature_flags.flags

@app.put("/api/v1/feature-flags/{flag_name}")
async def update_feature_flag(
    flag_name: str,
    enabled: bool,
    token: str = Depends(verify_token)
):
    """Update feature flag"""
    if flag_name in feature_flags.flags:
        feature_flags.flags[flag_name] = enabled
        logger.info(f"Feature flag {flag_name} set to {enabled}")
        return {"flag": flag_name, "enabled": enabled}
    else:
        raise HTTPException(status_code=404, detail=f"Unknown flag: {flag_name}")

# Main entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )