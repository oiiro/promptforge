# 🔐 API Authentication Guide

## Overview

The PromptForge API server requires Bearer token authentication for all protected endpoints. This guide covers authentication requirements, usage examples, and troubleshooting.

## Authentication Requirements

### Protected Endpoints
The following endpoints require authentication:
- `/api/v1/capital` - Capital finder service
- `/api/v1/evaluate` - Prompt evaluation
- `/api/v1/metrics` - System metrics
- `/api/v1/audit-log` - Audit log access
- `/api/v1/rollback` - Version rollback
- `/api/v1/feature-flags` - Feature flag management

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

## API Documentation

Visit the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **OpenAPI Spec**: http://localhost:8000/openapi.json

The Swagger UI provides a web interface for testing all endpoints with proper authentication.