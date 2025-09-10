# PromptForge Architecture & Design

**Financial Services Grade Prompt Engineering SDLC Platform**

This document provides a comprehensive overview of PromptForge's architecture, technology stack, and integration patterns. All components work together to provide enterprise-grade prompt engineering capabilities with comprehensive evaluation, monitoring, and compliance features.

## Table of Contents
- [System Overview](#system-overview)
- [Core Architecture](#core-architecture)
- [Technology Stack](#technology-stack)
- [Service Integration](#service-integration)
- [Database Architecture](#database-architecture)
- [Security & Compliance](#security--compliance)
- [Development Workflow](#development-workflow)
- [Deployment & Operations](#deployment--operations)

## System Overview

PromptForge is a comprehensive platform for developing, evaluating, and monitoring AI prompts in financial services environments. The platform integrates with Langfuse for observability and DeepEval for optimization:

```
┌─────────────────────┐    ┌─────────────────────────┐
│   PromptForge API   │    │   Langfuse Dashboard    │
│   (Port 8000)      │    │   (Cloud/Self-hosted)   │
│                     │    │                         │
│ • FastAPI Server    │    │ • Trace Analytics       │
│ • Prompt Execution  │    │ • Performance Metrics   │
│ • PII Protection    │    │ • Cost Tracking         │
│ • Security Features │    │ • A/B Testing           │
└─────────┬───────────┘    └─────────┬───────────────┘
          │                          │
          └──────────┬─────────────────┘
                     │
           ┌─────────▼─────────┐
           │   Local Storage   │
           │   (SQLite + Logs) │
           │                   │
           │ • Prompt Registry │
           │ • Team Config     │
           │ • Audit Logs     │
           │ • Cache Data     │
           └───────────────────┘
```

### Key Benefits
- **Real-time Monitoring**: Langfuse provides live tracing and metrics for all prompt executions
- **Enterprise Security**: PII protection, security scanning, structured logging
- **Financial Compliance**: SOC 2 Type II ready with comprehensive audit trails
- **Developer Experience**: Hot-reload development, comprehensive documentation
- **Production Ready**: Docker support, observability, error handling

## Core Architecture

### Application Server (Port 8000)
**Location**: `orchestration/app.py`

```python
# Core FastAPI application with integrated Langfuse
from langfuse import Langfuse
from evaluation.langfuse_config import LangfuseConfig

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Langfuse client on startup
    config = LangfuseConfig()
    app.state.langfuse = config.client
    yield
    # Cleanup on shutdown
    app.state.langfuse.flush()

app = FastAPI(lifespan=lifespan)
```

**Primary Responsibilities**:
- RESTful API endpoints for prompt execution
- Microsoft Presidio PII protection integration
- Security validation and rate limiting
- OpenTelemetry distributed tracing
- Real-time observability via Langfuse
- Structured logging with compliance features

**Key Endpoints**:
- `/api/v1/prompt/execute` - Execute prompts with tracing
- `/api/v1/metrics` - Retrieve performance metrics
- `/api/v1/health` - Health check and system status
- `/docs` - Interactive API documentation

### Langfuse Observability Platform
**Access**: Cloud (https://cloud.langfuse.com) or Self-hosted

```python
# Langfuse integration with decorators
from langfuse import observe

@observe(name="prompt_execution")
def execute_prompt(prompt: str, variables: dict):
    # Automatic tracing of execution
    return llm_client.generate(prompt, variables)
```

**Primary Capabilities**:
- Web-based dashboard for trace analytics
- Real-time visualization of prompt performance
- Historical data analysis and trends
- Feedback collection and annotation
- Comparative analysis across prompt versions

**Key Features**:
- Live refresh of evaluation data from API server
- Interactive charts and performance metrics
- Feedback scoring and annotation tools
- Comparative prompt analysis
- Export capabilities for reports

## Technology Stack

### Core Framework Stack
```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                    │
├─────────────────────────────────────────────────────────┤
│ FastAPI 0.110.0+     │ Web framework & API endpoints    │
│ Rich CLI             │ Terminal UI framework            │
│ Uvicorn 0.29.0+      │ ASGI server implementation       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Evaluation Layer                     │
├─────────────────────────────────────────────────────────┤
│ Langfuse 2.0+        │ Primary observability platform   │
│ DeepEval 0.21.0+     │ Secondary evaluation metrics     │
│ Detoxify 0.5.0+      │ Toxicity detection               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    Security Layer                      │
├─────────────────────────────────────────────────────────┤
│ Presidio 2.2.35+     │ PII detection & anonymization    │
│ Cryptography 42.0+   │ Encryption & security primitives │
│ Passlib[bcrypt]      │ Password hashing                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  Observability Layer                   │
├─────────────────────────────────────────────────────────┤
│ OpenTelemetry 1.36+  │ Distributed tracing              │
│ Structlog 24.1.0+    │ Structured logging               │
│ Redis 5.0.0+         │ Session & caching                │
└─────────────────────────────────────────────────────────┘
```

### AI/ML Integration Stack
```python
# LLM Provider Integration
openai>=1.35.0          # OpenAI GPT models
anthropic>=0.25.0       # Claude models
langchain>=0.3.27       # LLM orchestration framework

# Data Processing Pipeline
pandas>=2.2.0           # Data manipulation
numpy>=1.26.0           # Numerical computing
spacy>=3.7.0            # NLP processing (Presidio dependency)

# Validation & Testing
pydantic>=2.7.0         # Data validation
jsonschema>=4.21.0      # JSON schema validation
pytest>=8.2.0           # Testing framework
```

## Service Integration

### Observability Architecture

The platform integrates with Langfuse for comprehensive observability and tracing:

```python
# Langfuse client initialization
# Application Server (orchestration/app.py)
from langfuse import Langfuse

app.state.langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY")
)
```

**Data Flow Pattern**:
1. **API Execution**: User calls `/api/v1/prompt/execute`
2. **Trace Collection**: Langfuse records execution traces
3. **Real-time Display**: Cloud dashboard shows live metrics
4. **Historical Analysis**: All traces persist for analysis

### Langfuse Integration Details

**Observability Pipeline**:
```python
# Automatic tracing with decorators
from langfuse import observe

@observe(name="prompt_execution")
def execute_prompt_with_tracing(prompt: str, model: str):
    # Langfuse automatically captures:
    # - Latency metrics
    # - Token usage and costs
    # - Input/output pairs
    # - Error traces
    # - Custom metadata
    
    trace = app.state.langfuse.trace(
        name="prompt_execution",
        metadata={"model": model}
    )
    
    # Execution is automatically traced
    response = llm_client.generate(prompt)
    trace.end(output=response)
    return response
```

**Dashboard Access**:
- **Cloud Dashboard**: `https://cloud.langfuse.com` - Full analytics
- **Self-Hosted**: Optional on-premise deployment
- **Browser Launch**: `./launch_trulens_dashboard.py` - Automated startup

### Microsoft Presidio PII Protection

**Architecture**: Layered protection with multiple detection engines

```python
# PII Detection Pipeline
presidio_analyzer = AnalyzerEngine()
presidio_anonymizer = AnonymizerEngine()

# Detection covers:
# - Financial data (SSN, credit cards, bank accounts)
# - Personal identifiers (names, emails, phone numbers)
# - Health information (HIPAA compliance)
# - Custom patterns for financial services

def protect_pii(text: str) -> str:
    # Multi-stage analysis
    results = presidio_analyzer.analyze(text, language='en')
    anonymized = presidio_anonymizer.anonymize(text, results)
    return anonymized.text
```

## Database Architecture

### SQLite Schema (TruLens Managed)
```sql
-- Core evaluation tables (managed by TruLens)
CREATE TABLE records (
    record_id TEXT PRIMARY KEY,
    app_id TEXT,
    input TEXT,
    output TEXT,
    cost REAL,
    latency REAL,
    timestamp DATETIME
);

CREATE TABLE feedback (
    feedback_id TEXT PRIMARY KEY,
    record_id TEXT,
    feedback_name TEXT,
    feedback_value REAL,
    FOREIGN KEY (record_id) REFERENCES records(record_id)
);

CREATE TABLE apps (
    app_id TEXT PRIMARY KEY,
    app_name TEXT,
    app_version TEXT,
    created_at DATETIME
);
```

### Data Retention & Performance
- **Default Retention**: Unlimited (SQLite file-based)
- **Performance**: Optimized for development and small-scale production
- **Scalability**: Can be upgraded to PostgreSQL/MySQL for enterprise scale
- **Backup**: File-based backup of `default.sqlite`

### Database Connection Management
```python
# Connection pooling and session management
class DatabaseManager:
    def __init__(self):
        self.session = TruSession()  # Auto-manages connection lifecycle
    
    def get_session(self):
        return self.session  # Thread-safe session reuse
```

## Security & Compliance

### Data Protection Framework
```python
# Multi-layered security architecture
SECURITY_LAYERS = {
    "PII_Protection": "Microsoft Presidio - Financial grade anonymization",
    "Encryption": "Cryptography 42.0+ - AES-256, RSA key management", 
    "Authentication": "Passlib bcrypt - Secure password hashing",
    "Input_Validation": "Pydantic 2.7+ - Request/response validation",
    "Rate_Limiting": "Token bucket - Configurable per-endpoint limits",
    "Audit_Logging": "Structured logs - SOC 2 Type II compliance ready"
}
```

### Financial Services Compliance
- **SOC 2 Type II**: Comprehensive audit trail and access controls
- **GDPR Ready**: PII anonymization and right-to-deletion support
- **HIPAA Compatible**: Health information protection patterns
- **PCI DSS**: Credit card and payment data protection
- **Regulatory Reporting**: Structured logs for compliance audits

### Security Monitoring
```python
# Real-time security event monitoring
from observability.security import SecurityMonitor

security_monitor = SecurityMonitor()

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Monitor for suspicious patterns
    security_monitor.analyze_request(request)
    response = await call_next(request)
    security_monitor.log_response(response)
    return response
```

## Development Workflow

### Local Development Setup
```bash
# 1. Environment Setup
git clone <repository>
cd promptforge
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 2. Dependency Installation
pip install -r requirements.txt

# 3. Service Startup (Terminal 1)
./start_server.sh
# PromptForge API available at: http://localhost:8000

# 4. Dashboard Startup (Terminal 2)  
./launch_trulens_dashboard.py
# TruLens Dashboard available at: http://localhost:8501

# 5. Development Workflow
# - Code changes trigger automatic reload (FastAPI hot-reload)
# - Evaluation data immediately visible in dashboard
# - Interactive API docs at http://localhost:8000/docs
```

### Testing Strategy
```python
# Comprehensive testing framework
TESTING_LAYERS = {
    "Unit Tests": "pytest - Core functionality validation",
    "Integration Tests": "TruLens evaluation validation", 
    "Security Tests": "PII protection and input validation",
    "Performance Tests": "Load testing and latency validation",
    "Compliance Tests": "Regulatory requirement validation"
}

# Run test suite
pytest tests/ --cov=orchestration --cov-report=html
```

### Code Quality Standards
```python
# Development tools integration
BLACK_CONFIG = "black --line-length 88"
FLAKE8_CONFIG = "flake8 --max-line-length=88" 
MYPY_CONFIG = "mypy --strict"

# Pre-commit hooks ensure:
# - Code formatting (Black)
# - Linting (Flake8)  
# - Type checking (MyPy)
# - Security scanning
# - Documentation validation
```

## Deployment & Operations

### Container Architecture
```dockerfile
# Multi-stage Docker build for production
FROM python:3.11-slim as base
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM base as app
COPY . .
EXPOSE 8000
CMD ["uvicorn", "orchestration.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Configuration
```python
# Environment-aware configuration
PRODUCTION_CONFIG = {
    "database": "postgresql://...",  # Upgrade from SQLite
    "redis_cache": "redis://...",    # Distributed caching
    "observability": "jaeger://...", # Distributed tracing
    "security": {
        "tls_enabled": True,
        "auth_required": True,
        "rate_limits": "strict"
    }
}
```

### Monitoring & Observability
```python
# Production monitoring stack
OBSERVABILITY_STACK = {
    "Metrics": "OpenTelemetry + Prometheus",
    "Tracing": "Jaeger distributed tracing",
    "Logging": "Structured logs + ELK stack",
    "Dashboards": "Grafana + TruLens dashboard",
    "Alerting": "PagerDuty integration"
}

# Health check endpoint
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "database": await check_database_health(),
        "trulens": await check_trulens_health(),
        "services": await check_external_services()
    }
```

### Scaling Considerations

**Horizontal Scaling**:
- Multiple FastAPI instances behind load balancer
- Shared PostgreSQL database for TruLens data
- Redis for distributed session management
- Container orchestration (Kubernetes/ECS)

**Performance Optimization**:
- Connection pooling for database access
- Async/await patterns throughout codebase  
- Caching of expensive operations (PII analysis)
- Background task processing for non-critical operations

## Architecture Decision Records

### ADR-001: Database-Mediated Service Integration
**Decision**: Use shared SQLite database for TruLens-PromptForge integration
**Rationale**: Enables real-time data sharing without complex service-to-service communication
**Consequences**: Simple development, easy deployment, limited to single-node scaling

### ADR-002: Microsoft Presidio for PII Protection  
**Decision**: Integrate Presidio as primary PII protection mechanism
**Rationale**: Industry-standard, financial services grade, comprehensive language support
**Consequences**: Additional dependency overhead, excellent compliance coverage

### ADR-003: FastAPI + Streamlit Dual-Server Architecture
**Decision**: Separate API server and dashboard server with shared database
**Rationale**: Separation of concerns, independent scaling, specialized UIs
**Consequences**: Two-process deployment, shared state via database, excellent user experience

---

## Quick Reference

### Service Ports
- **PromptForge API**: `http://localhost:8000`
- **TruLens Dashboard**: `http://localhost:8501`
- **API Documentation**: `http://localhost:8000/docs`

### Key Files
- `orchestration/app.py` - Main FastAPI application
- `launch_trulens_dashboard.py` - Dashboard launcher  
- `requirements.txt` - Complete dependency specification
- `setup_promptforge.py` - Automated setup script

### Common Commands
```bash
# Start API server
./start_server.sh

# Start dashboard
./launch_trulens_dashboard.py

# Run tests
pytest tests/ --cov

# Check dependencies
./setup_promptforge.py --check-only
```

This architecture provides enterprise-grade capabilities while maintaining developer productivity and operational simplicity. The shared database pattern enables seamless integration between evaluation and execution, providing real-time insights into AI system performance.