# 🔐 API Authentication Guide

## Overview

The PromptForge API server requires Bearer token authentication for all protected endpoints. This guide covers authentication requirements, usage examples, and troubleshooting.

## Authentication Requirements

### Protected Endpoints
The following endpoints require authentication:
- `/api/v1/capital` - Capital finder service
- `/api/v1/retirement-eligibility` - Multi-person retirement eligibility with PII protection
- `/api/v1/evaluate` - Prompt evaluation
- `/api/v1/metrics` - System metrics
- `/api/v1/audit-log` - Audit log access
- `/api/v1/rollback` - Version rollback
- `/api/v1/feature-flags` - Feature flag management
- `/api/v1/trulens/dashboard` - TruLens monitoring dashboard

### Public Endpoints
The following endpoints do NOT require authentication:
- `/health` - Health check
- `/docs` - API documentation (Swagger UI)
- `/openapi.json` - OpenAPI specification
- `/api/v1/version` - Version information

## Authentication Token

**Default Token:** `demo-token`

This is configured via the `API_TOKEN` environment variable:
```bash
export API_TOKEN=demo-token
```

## Usage Examples

### ✅ Correct Usage (Authenticated)

#### Capital Query
```bash
curl -X POST http://localhost:8000/api/v1/capital \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-token" \
  -d '{"country": "France"}'
```

#### Capital Query with PII Protection (Presidio)
```bash
# Query with personal information - PII will be automatically detected and anonymized
curl -X POST http://localhost:8000/api/v1/capital \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-token" \
  -d '{"country": "Hi, I'\''m John Smith and I want to know the capital of Japan"}'
```

**Response:**
```json
{
  "capital": "Paris",
  "confidence": 1.0,
  "metadata": {
    "source": "geographical_database",
    "timestamp": "2025-08-27T20:27:52.919918",
    "model": "mock-model",
    "latency_ms": 50,
    "token_usage": {"prompt": 228, "completion": 5, "total": 233}
  }
}
```

#### Multi-Person Retirement Eligibility with PII Protection
```bash
curl -X POST http://localhost:8000/api/v1/retirement-eligibility \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-token" \
  -d '{
    "query": "Evaluate retirement eligibility for John Smith, age 65, phone 555-123-4567, employed for 25 years with salary $75,000. Also Sarah Johnson, age 62, email sarah.johnson@company.com, employed for 30 years.",
    "enable_pii_protection": true,
    "enable_monitoring": true
  }'
```

**Response with PII Protection:**
```json
{
  "response": "Retirement eligibility confirmation with PII anonymized...",
  "eligible": true,
  "deposit_amount": "10,000",
  "persons_processed": 2,
  "pii_detected": true,
  "pii_entities": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
  "anonymization_applied": true,
  "metadata": {
    "pii_protection": "enabled",
    "numbered_placeholders_used": true,
    "anonymized_entities": ["<NAME_1>", "<EMAIL_ADDRESS_1>", "<PHONE_NUMBER_1>"],
    "request_id": "uuid-here",
    "processing_time_ms": 150,
    "trulens_monitoring": true
  }
}
```

#### System Metrics
```bash
curl -H "Authorization: Bearer demo-token" \
  http://localhost:8000/api/v1/metrics
```

#### Health Check (No Auth)
```bash
curl http://localhost:8000/health
```

### ❌ Incorrect Usage (Missing Authentication)

```bash
curl -X POST http://localhost:8000/api/v1/capital \
  -H "Content-Type: application/json" \
  -d '{"country": "France"}'
```

**Error Response:**
```json
{"detail": "Not authenticated"}
```

## Programming Examples

### Python with httpx
```python
import asyncio
import httpx

async def test_api():
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": "Bearer demo-token"}
        response = await client.post(
            "http://localhost:8000/api/v1/capital",
            json={"country": "France"},
            headers=headers
        )
        print(response.json())

asyncio.run(test_api())
```

### Python with requests
```python
import requests

headers = {"Authorization": "Bearer demo-token"}
response = requests.post(
    "http://localhost:8000/api/v1/capital",
    json={"country": "France"},
    headers=headers
)
print(response.json())
```

### JavaScript with fetch
```javascript
fetch('http://localhost:8000/api/v1/capital', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer demo-token'
  },
  body: JSON.stringify({country: 'France'})
})
.then(response => response.json())
.then(data => console.log(data));
```

## Testing Scripts

### Comprehensive API Testing
```bash
# Test all endpoints with proper authentication
python3 scripts/test_api_endpoints.py
```

### HTTP Client Testing
```bash
# Test using httpx client
python3 scripts/test_api_server.py
```

## Troubleshooting

### Common Issues

1. **"Not authenticated" Error**
   - **Cause**: Missing or invalid `Authorization` header
   - **Solution**: Add `-H "Authorization: Bearer demo-token"`

2. **403 Forbidden Error**
   - **Cause**: Invalid token or expired credentials
   - **Solution**: Verify token matches `API_TOKEN` environment variable

3. **Server Not Running**
   - **Cause**: API server not started
   - **Solution**: Run `./venv/bin/python3 orchestration/app.py`

4. **TruLens Dashboard 404 Not Found**
   - **Cause**: Using incorrect URL path `/api/trulens/dashboard`
   - **Solution**: Use correct path `/api/v1/trulens/dashboard`
   ```bash
   curl -H "Authorization: Bearer demo-token" http://localhost:8000/api/v1/trulens/dashboard
   ```

5. **TruLens Dashboard Service Unavailable**
   - **Cause**: TruLens not properly initialized or missing dependencies
   - **Solution**: Use alternative native dashboard access:
   ```python
   from trulens.core import TruSession
   session = TruSession()
   session.run_dashboard(port=8501)
   # Access: http://localhost:8501
   ```

### Debug Commands

#### Check Server Status
```bash
curl http://localhost:8000/health
```

#### Verify Token Configuration
```bash
# Check if server is using expected token
echo $API_TOKEN
```

#### Test Authentication Flow
```bash
# 1. Test without auth (should fail)
curl -X POST http://localhost:8000/api/v1/capital \
  -H "Content-Type: application/json" \
  -d '{"country": "France"}'

# 2. Test with auth (should succeed)
curl -X POST http://localhost:8000/api/v1/capital \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-token" \
  -d '{"country": "France"}'
```

## Production Considerations

### Security
- **Change Default Token**: Use a secure, randomly generated token in production
- **Token Rotation**: Implement regular token rotation
- **HTTPS Only**: Use HTTPS in production environments
- **Rate Limiting**: Monitor and limit requests per client

### Environment Variables
```bash
# Production example
export API_TOKEN="your-secure-production-token"
export ENVIRONMENT="production"
export ENABLE_TRACING="true"
```

### Advanced Authentication
For production deployments, consider implementing:
- JWT tokens with expiration
- OAuth2 integration
- API key management
- Role-based access control (RBAC)

### PII Protection with Presidio
The PromptForge API includes built-in PII protection using Microsoft Presidio:
- **Automatic Detection**: Identifies personal information in queries
- **Anonymization**: Replaces PII with placeholders during processing
- **Deanonymization**: Restores original context in responses
- **Compliance**: Supports GDPR, CCPA, and other privacy regulations

#### Presidio Testing Script
```bash
# Test PII anonymization and deanonymization
python3 scripts/test_presidio_capital_finder.py
```

#### PII Protection Features
- **Supported PII Types**: Names, emails, phone numbers, SSNs, credit cards
- **Configurable Policies**: Custom anonymization rules per entity type
- **Session Management**: Maintains anonymization mapping for request lifecycle
- **Audit Trail**: Logs PII detection and handling for compliance

## API Documentation

Visit the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **OpenAPI Spec**: http://localhost:8000/openapi.json

The Swagger UI provides a web interface for testing all endpoints with proper authentication.