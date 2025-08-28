#!/usr/bin/env python3
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
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
from llm_client import LLMClient
from guardrails.validators import GuardrailOrchestrator
from observability.tracing import TracingManager
from observability.metrics import MetricsCollector
from retirement_endpoints_enhanced import (
    setup_retirement_endpoints,
    RetirementEligibilityRequest,
    RetirementEligibilityResponse
)

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TruLens integration for monitoring PII protection
try:
    from trulens.core import TruSession
    from trulens.feedback import feedback
    from trulens.core.app import App
    from trulens.providers.openai import OpenAI as TruOpenAI
    TRULENS_AVAILABLE = True
    logger.info("TruLens successfully imported for PII monitoring")
except ImportError:
    logger.warning("TruLens not available. Observability features limited.")
    TRULENS_AVAILABLE = False

# Presidio PII Protection integration
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
    logger.info("Presidio successfully imported for PII protection")
except ImportError as e:
    logger.warning(f"Presidio not available. PII protection features disabled. Error: {e}")
    PRESIDIO_AVAILABLE = False

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

# RetirementEligibilityRequest and RetirementEligibilityResponse now imported from retirement_endpoints

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
    
    # Initialize Presidio PII Protection
    if PRESIDIO_AVAILABLE:
        app.state.analyzer = AnalyzerEngine()
        app.state.anonymizer = AnonymizerEngine()
        logger.info("Presidio PII protection initialized")
    else:
        app.state.analyzer = None
        app.state.anonymizer = None
    
    # Initialize TruLens monitoring (simplified - no legacy app creation)
    if TRULENS_AVAILABLE:
        try:
            # Use configured database URL from environment
            database_url = os.getenv("TRULENS_DATABASE_URL", "sqlite:///default.sqlite")
            logger.info(f"Initializing TruLens with database: {database_url}")
            app.state.tru_session = TruSession(database_url=database_url)
            
            # Create feedback functions for PII monitoring
            app.state.pii_detection_feedback = create_pii_detection_feedback()
            app.state.anonymization_quality_feedback = create_anonymization_quality_feedback()
            logger.info("✅ TruLens monitoring initialized (no legacy app creation)")
        except Exception as e:
            logger.warning(f"Failed to initialize TruLens: {e}")
            app.state.tru_session = None
    else:
        app.state.tru_session = None
    
    # Initialize enhanced retirement eligibility apps with comprehensive TruLens feedback
    logger.info("Initializing enhanced retirement eligibility endpoints...")
    
    # Setup endpoints with comprehensive feedback functions
    app.state.tru_mock_app, app.state.tru_live_app = setup_retirement_endpoints(
        app, 
        app.state.llm_client, 
        verify_token
    )
    if app.state.tru_mock_app and app.state.tru_live_app:
        logger.info("✅ Enhanced retirement endpoints initialized with comprehensive TruLens feedback")
    else:
        logger.warning("❌ Failed to initialize enhanced retirement endpoints")
    
    # Load version info
    app.state.version = VersionInfo(
        prompt_version=os.getenv("PROMPT_VERSION", "1.0.0"),
        guardrails_version="1.0.0",
        model_version=os.getenv("DEFAULT_MODEL", "gpt-4"),
        deployed_at=datetime.utcnow().isoformat()
    )
    
    # Warm up providers with timeout
    try:
        # Add timeout to prevent hanging
        import asyncio
        await asyncio.wait_for(app.state.llm_client.health_check(), timeout=5.0)
        logger.info("LLM provider warmed up successfully")
    except asyncio.TimeoutError:
        logger.warning("LLM provider health check timed out after 5 seconds")
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

# TruLens feedback functions for PII monitoring
def create_pii_detection_feedback():
    """Create feedback function to detect PII in queries"""
    if not TRULENS_AVAILABLE:
        return None
        
    def pii_detection_score(input_text: str) -> float:
        """Score based on PII entities detected"""
        if not PRESIDIO_AVAILABLE:
            return 0.0
            
        try:
            analyzer = AnalyzerEngine()
            results = analyzer.analyze(text=input_text, language='en')
            
            # Score: 1.0 if PII found, 0.0 if none
            return 1.0 if len(results) > 0 else 0.0
        except Exception:
            return 0.0
    
    # For now, return the function directly without TruLens feedback wrapper
    # This allows PII detection to work without full TruLens integration
    return pii_detection_score

def create_anonymization_quality_feedback():
    """Create feedback function to evaluate anonymization quality"""
    if not TRULENS_AVAILABLE:
        return None
        
    def anonymization_quality_score(original: str, anonymized: str) -> float:
        """Score anonymization quality (1.0 = perfect anonymization)"""
        if not PRESIDIO_AVAILABLE:
            return 0.0
            
        try:
            analyzer = AnalyzerEngine()
            
            # Check original for PII
            original_pii = analyzer.analyze(text=original, language='en')
            
            # Check anonymized for remaining PII
            anonymized_pii = analyzer.analyze(text=anonymized, language='en')
            
            if len(original_pii) == 0:
                return 1.0  # No PII to anonymize
                
            # Perfect score if all PII anonymized
            anonymization_ratio = 1.0 - (len(anonymized_pii) / len(original_pii))
            return max(0.0, anonymization_ratio)
        except Exception:
            return 0.0
    
    # For now, return the function directly without TruLens feedback wrapper
    # This allows anonymization quality checking to work without full TruLens integration
    return anonymization_quality_score

# Presidio PII Protection utilities
class PresidioManager:
    """Manages Presidio PII protection operations"""
    
    def __init__(self, analyzer, anonymizer):
        self.analyzer = analyzer
        self.anonymizer = anonymizer
        self.anonymization_map = {}
    
    def analyze_pii(self, text: str):
        """Analyze text for PII entities"""
        if not self.analyzer:
            return []
        return self.analyzer.analyze(text=text, language='en')
    
    def anonymize_with_numbered_placeholders(self, text: str, pii_results):
        """Anonymize PII with numbered placeholders for multi-entity scenarios"""
        if not self.anonymizer or not pii_results:
            return text, {}
        
        # Filter overlapping entities and prioritize EMAIL_ADDRESS over URL
        filtered_results = self._filter_overlapping_entities(pii_results)
        # Additional filtering: Remove URL entities that are part of EMAIL_ADDRESS entities
        email_ranges = [(r.start, r.end) for r in filtered_results if r.entity_type == 'EMAIL_ADDRESS']
        final_results = []
        
        for result in filtered_results:
            if result.entity_type == 'URL':
                # Check if this URL entity is contained within any email address
                is_within_email = any(
                    email_start <= result.start < result.end <= email_end 
                    for email_start, email_end in email_ranges
                )
                if not is_within_email:
                    final_results.append(result)
            else:
                final_results.append(result)
        
        sorted_results = sorted(final_results, key=lambda x: x.start)
        
        entity_counters = {}
        anonymization_map = {}
        operators = {}
        
        for result in sorted_results:
            entity_type = result.entity_type
            
            # Increment counter for this entity type
            if entity_type not in entity_counters:
                entity_counters[entity_type] = 1
            else:
                entity_counters[entity_type] += 1
                
            counter = entity_counters[entity_type]
            
            # Create numbered placeholder based on entity type
            if entity_type == "PERSON":
                placeholder = f"<NAME_{counter}>"
            elif entity_type == "EMAIL_ADDRESS":
                placeholder = f"<EMAIL_ADDRESS_{counter}>"
            elif entity_type == "DATE_TIME":
                placeholder = f"<DATE_{counter}>"
            elif entity_type == "PHONE_NUMBER":
                placeholder = f"<PHONE_NUMBER_{counter}>"
            elif entity_type == "US_SSN":
                placeholder = f"<SSN_{counter}>"
            elif entity_type == "LOCATION":
                placeholder = f"<LOCATION_{counter}>"
            elif entity_type == "URL":
                placeholder = f"<URL_{counter}>"
            else:
                placeholder = f"<{entity_type}_{counter}>"
            
            # Store original text for deanonymization
            original_text = text[result.start:result.end]
            anonymization_map[placeholder] = original_text
            
            # Configure operator
            operators[result.entity_type] = OperatorConfig("replace", {"new_value": placeholder})
        
        # Perform anonymization
        if operators:
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=sorted_results,
                operators=operators
            )
            return anonymized_result.text, anonymization_map
        
        return text, {}
    
    def _filter_overlapping_entities(self, pii_results):
        """Filter overlapping entities with prioritization"""
        if not pii_results:
            return []
        
        # Sort by start position
        sorted_results = sorted(pii_results, key=lambda x: x.start)
        filtered = []
        
        for current in sorted_results:
            should_add = True
            
            # Check against already added entities
            for existing in filtered:
                # Check for overlap
                if (current.start < existing.end and current.end > existing.start):
                    # Overlapping entities - apply prioritization
                    if current.entity_type == "EMAIL_ADDRESS" and existing.entity_type == "URL":
                        # Replace URL with EMAIL_ADDRESS
                        filtered.remove(existing)
                        should_add = True
                        break
                    elif current.entity_type == "URL" and existing.entity_type == "EMAIL_ADDRESS":
                        # Keep EMAIL_ADDRESS, skip URL
                        should_add = False
                        break
                    elif current.entity_type == existing.entity_type:
                        # Same type - keep the longer one
                        if (current.end - current.start) > (existing.end - existing.start):
                            filtered.remove(existing)
                            should_add = True
                            break
                        else:
                            should_add = False
                            break
            
            if should_add:
                filtered.append(current)
        
        return filtered
    
    def deanonymize_text(self, text: str, anonymization_map: dict) -> str:
        """Restore original PII in anonymized text"""
        if not anonymization_map:
            return text
        
        result = text
        for placeholder, original_value in anonymization_map.items():
            result = result.replace(placeholder, original_value)
        
        return result
    
    def mock_multi_person_retirement_eligibility(self, anonymized_query: str, anonymization_map: dict):
        """Mock service for multi-person retirement eligibility with dynamic placeholder handling"""
        
        # Analyze available placeholders in the anonymization map
        name_placeholders = sorted([k for k in anonymization_map.keys() if k.startswith('<NAME_')])
        email_placeholders = sorted([k for k in anonymization_map.keys() if k.startswith('<EMAIL_ADDRESS_')])
        date_placeholders = sorted([k for k in anonymization_map.keys() if k.startswith('<DATE_')])
        
        # Determine number of persons to process
        persons_count = max(len(name_placeholders), len(email_placeholders), 1)
        
        # Build dynamic response template
        response_lines = ["Here is the eligibility confirmation:\n"]
        
        for i in range(persons_count):
            person_num = i + 1
            
            # Use available placeholders or fallback to generic ones
            name_ph = name_placeholders[i] if i < len(name_placeholders) else f"<NAME_{person_num}>"
            date_ph = date_placeholders[i] if i < len(date_placeholders) else f"<DATE_{person_num}>"
            
            # For emails, be more careful - reuse existing emails if we don't have enough
            if i < len(email_placeholders):
                email_ph = email_placeholders[i]
            else:
                # If we don't have enough email addresses, either use "N/A" or reuse the last available one
                if email_placeholders:
                    email_ph = email_placeholders[-1]  # Reuse last email address
                else:
                    email_ph = "N/A"  # No email available
            
            line = f"{person_num}. {name_ph} (born in {date_ph}) with email {email_ph} is eligible for an account with a $10,000 deposit."
            response_lines.append(line)
        
        response_template = "\n".join(response_lines)
        
        return {
            "response": response_template,
            "eligible": True,
            "deposit_amount": "10,000",
            "persons_processed": persons_count,
            "metadata": {
                "source": "multi_person_retirement_eligibility_service",
                "model": "mock-financial-multi-person-demo",
                "pii_protection": "enabled",
                "requires_deanonymization": True,
                "multi_entity_support": True,
                "numbered_placeholders_used": True,
                "placeholders_used": {
                    "names": name_placeholders,
                    "emails": email_placeholders, 
                    "dates": date_placeholders
                }
            }
        }

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

# Retirement eligibility endpoints are now handled in retirement_endpoints_enhanced.py
# They are registered via setup_retirement_endpoints() during app initialization

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

async def audit_retirement_eligibility_log(
    request_id: str, 
    input_query: str, 
    output: Dict, 
    latency_ms: float, 
    tru_record=None
):
    """Background task for retirement eligibility audit logging with PII protection tracking"""
    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "endpoint": "retirement-eligibility",
        "input_query": input_query[:100] + "..." if len(input_query) > 100 else input_query,  # Truncate for security
        "pii_detected": output.get("pii_detected", False),
        "pii_entities": output.get("pii_entities", []),
        "persons_processed": output.get("persons_processed", 0),
        "eligible": output.get("eligible", False),
        "deposit_amount": output.get("deposit_amount"),
        "anonymization_applied": output.get("anonymization_applied", False),
        "latency_ms": latency_ms,
        "trulens_monitoring": tru_record is not None,
        "metadata": {
            "source": output.get("metadata", {}).get("source"),
            "model": output.get("metadata", {}).get("model"),
            "pii_protection": output.get("metadata", {}).get("pii_protection", "unknown")
        }
    }
    
    # Add TruLens record ID if available
    if tru_record:
        try:
            log_entry["trulens_record_id"] = str(tru_record.record_id) if hasattr(tru_record, 'record_id') else "unknown"
        except Exception:
            log_entry["trulens_record_id"] = "error"
    
    # Write to specialized audit log file
    log_path = Path("logs/retirement_eligibility_audit.jsonl")
    log_path.parent.mkdir(exist_ok=True)
    
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# TruLens Dashboard endpoints
@app.get("/api/v1/trulens/dashboard")
async def get_trulens_dashboard(token: str = Depends(verify_token)):
    """Get TruLens dashboard data for PII protection monitoring"""
    if not TRULENS_AVAILABLE:
        raise HTTPException(status_code=503, detail="TruLens monitoring not available")
    
    if not app.state.tru_session:
        raise HTTPException(status_code=503, detail="TruLens session not initialized")
    
    try:
        # Try multiple approaches to get records safely
        records_df = None
        
        # Method 1: Get records from retirement apps (MockPromptForge and PromptForge)
        if app.state.tru_mock_app and app.state.tru_live_app:
            try:
                import pandas as pd
                
                all_records = []
                
                # Get records from MockPromptForge app
                try:
                    mock_result = app.state.tru_session.get_records_and_feedback(app_name="MockPromptForge")
                    if isinstance(mock_result, tuple):
                        mock_records_df, _ = mock_result
                        if not mock_records_df.empty:
                            all_records.append(mock_records_df)
                            logger.info(f"Retrieved {len(mock_records_df)} records from MockPromptForge")
                except Exception as e:
                    logger.warning(f"No MockPromptForge records found: {e}")
                
                # Get records from PromptForge app
                try:
                    live_result = app.state.tru_session.get_records_and_feedback(app_name="PromptForge")
                    if isinstance(live_result, tuple):
                        live_records_df, _ = live_result
                        if not live_records_df.empty:
                            all_records.append(live_records_df)
                            logger.info(f"Retrieved {len(live_records_df)} records from PromptForge")
                except Exception as e:
                    logger.warning(f"No PromptForge records found: {e}")
                
                # Combine all records
                if all_records:
                    records_df = pd.concat(all_records, ignore_index=True)
                    logger.info(f"✅ Combined {len(records_df)} total records from retirement apps")
                else:
                    logger.info("No records found from retirement apps")
                    
            except Exception as e1:
                logger.warning(f"Failed to get records from retirement apps: {e1}")
        else:
            logger.warning("No unified retirement apps available for record retrieval")
            
            # Method 2: Try without filters (get all records)
            try:
                result = app.state.tru_session.get_records_and_feedback()
                if isinstance(result, tuple):
                    records_df, feedback_columns = result
                    logger.info(f"Retrieved all records without filters: {len(records_df)} records")
                else:
                    records_df = result
                    logger.info(f"Retrieved all records without filters (non-tuple)")
            except Exception as e2:
                logger.warning(f"Failed to get all records: {e2}")
                
                # Method 3: Use simplified approach - return empty result for now
                logger.info("Using simplified empty result approach")
                return {
                    "total_records": 0,
                    "pii_detection_metrics": {"message": "TruLens records access currently unavailable"},
                    "anonymization_metrics": {"message": "TruLens records access currently unavailable"},
                    "recent_records": [],
                    "dashboard_url": f"http://localhost:8501" if os.getenv("TRULENS_DASHBOARD_ENABLED") == "true" else None
                }
        
        if records_df is None or records_df.empty:
            return {
                "total_records": 0,
                "pii_detection_metrics": {},
                "anonymization_metrics": {},
                "recent_records": [],
                "dashboard_url": f"http://localhost:8501" if os.getenv("TRULENS_DASHBOARD_ENABLED") == "true" else None
            }
        
        # Calculate metrics
        total_records = len(records_df)
        
        # PII detection metrics
        pii_detection_scores = records_df.get('PII Detection', [])
        pii_metrics = {
            "average_score": float(pii_detection_scores.mean()) if len(pii_detection_scores) > 0 else 0.0,
            "total_requests_with_pii": int(sum(pii_detection_scores)) if len(pii_detection_scores) > 0 else 0,
            "pii_detection_rate": float(pii_detection_scores.mean()) if len(pii_detection_scores) > 0 else 0.0
        }
        
        # Anonymization quality metrics
        anonymization_scores = records_df.get('Anonymization Quality', [])
        anonymization_metrics = {
            "average_quality": float(anonymization_scores.mean()) if len(anonymization_scores) > 0 else 0.0,
            "perfect_anonymization_count": int(sum(score == 1.0 for score in anonymization_scores)) if len(anonymization_scores) > 0 else 0,
            "anonymization_success_rate": float(sum(score >= 0.9 for score in anonymization_scores) / len(anonymization_scores)) if len(anonymization_scores) > 0 else 0.0
        }
        
        # Recent records (last 10)
        recent_records = []
        for idx, record in records_df.tail(10).iterrows():
            # Safely handle app_id which might be None - use app_name from record
            app_id_value = record.get('app_id', 'Unknown')
            if app_id_value is None:
                # Try to get app_name from the record, fallback to retirement app name
                app_id_value = record.get('app_name', 'PromptForge')
            
            recent_records.append({
                "record_id": str(record.get('record_id', idx)),
                "timestamp": str(record.get('start_time', '')),
                "pii_detection_score": float(record.get('PII Detection', 0.0)),
                "anonymization_quality_score": float(record.get('Anonymization Quality', 0.0)),
                "app_name": str(app_id_value)
            })
        
        return {
            "total_records": total_records,
            "pii_detection_metrics": pii_metrics,
            "anonymization_metrics": anonymization_metrics,
            "recent_records": recent_records,
            "dashboard_url": f"http://localhost:8501" if os.getenv("TRULENS_DASHBOARD_ENABLED") == "true" else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get TruLens dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")

@app.post("/api/v1/trulens/start-dashboard")
async def start_trulens_dashboard(token: str = Depends(verify_token)):
    """Start TruLens dashboard server"""
    if not TRULENS_AVAILABLE:
        raise HTTPException(status_code=503, detail="TruLens monitoring not available")
    
    try:
        # Note: In production, this would start the dashboard in a separate process
        # For demo purposes, we'll provide instructions
        return {
            "status": "instructions",
            "message": "To start TruLens dashboard, run: tru-ui --port 8501 in a separate terminal",
            "dashboard_url": "http://localhost:8501",
            "note": "Dashboard will show PII detection and anonymization quality metrics"
        }
    except Exception as e:
        logger.error(f"Failed to start TruLens dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard startup error: {str(e)}")

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