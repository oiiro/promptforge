# 🚀 Running the PII-Aware Capital Finder Example

This guide walks you through running the comprehensive PII protection reference architecture example that demonstrates Microsoft Presidio integration with PromptForge.

## 📋 Prerequisites

Before running the example, ensure you have:

- **Python 3.9+** installed
- **Virtual environment** capability
- **Git** for cloning (if needed)
- **Optional**: Redis server for production PII mapping (not required for basic testing)
- **Optional**: OpenAI or Anthropic API keys (MockProvider works without keys)

## 🔧 Step 1: Environment Setup

### Navigate to Project Directory
```bash
cd /Users/rohitiyer/oiiro/ai/promptforge
```

### Verify Project Structure
```bash
ls -la
# Should show:
# - examples/capital_finder_presidio.py  (our main example)
# - orchestration/llm_client.py          (enhanced LLM client)
# - presidio/                            (PII protection modules)
# - requirements.txt                     (dependencies)
# - README.md                           (project documentation)
```

### Check Python Version
```bash
python3 --version
# Should show: Python 3.9.x or higher
```

## 🛠️ Step 2: Install Dependencies

### Option A: Automated Setup (Recommended)
```bash
# Run the comprehensive setup script
python3 setup_promptforge.py

# This will:
# 1. Create virtual environment
# 2. Install all dependencies including Presidio
# 3. Set up configuration files
# 4. Run verification tests
# 5. Generate setup report
```

### Option B: Manual Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install --upgrade pip

# Install required packages
pip install openai>=1.35.0 anthropic>=0.25.0 
pip install fastapi>=0.110.0 uvicorn>=0.29.0
pip install pandas>=2.2.0 numpy>=1.26.0
pip install python-dotenv>=1.0.0 PyYAML>=6.0.1
pip install structlog>=24.1.0

# Install PII protection (optional - graceful fallback if not available)
pip install presidio-analyzer presidio-anonymizer
pip install spacy
python -m spacy download en_core_web_sm

# Install TruLens for evaluation
pip install trulens-core>=2.2.4 trulens-feedback>=2.2.4
```

## 🔑 Step 3: Configuration Setup

### Create Environment Configuration
```bash
# Copy the template (if it exists)
cp .env.example .env 2>/dev/null || true

# Or create a basic .env file
cat > .env << 'EOF'
# LLM Provider Configuration (optional - MockProvider works without keys)
DEFAULT_LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# PII Protection Configuration
ENABLE_PII_PROTECTION=true
PRESIDIO_LOG_LEVEL=INFO

# Redis Configuration (optional - uses in-memory fallback)
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=

# TruLens Configuration
TRULENS_DATABASE_URL=sqlite:///trulens_promptforge.db
TRULENS_LOG_LEVEL=INFO

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
API_HOST=localhost
API_PORT=8000
EOF
```

### Edit Configuration (Optional)
```bash
# Edit .env file to add your actual API keys
nano .env

# Note: The example works with MockProvider even without real API keys
```

## ✅ Step 4: Verify Installation

### Run Verification Script
```bash
# Activate virtual environment if not already active
source venv/bin/activate

# Run comprehensive verification
python3 verify_installation.py

# Expected output:
# 🔍 PromptForge Installation Verification
# ==================================================
# ✅ Virtual environment detected
# 📋 Testing Core Imports...
# ✅ TruLens core imports successful
# ✅ LLM client initialized - provider: mock
# ✅ TruLens configuration loaded
# ✅ Guardrails functional - validation: True
# ✅ Evaluation systems loaded
# 📊 Overall Score: 12/13 (92%)
# 🎉 EXCELLENT - PromptForge is fully operational!
```

### Quick Health Check
```bash
# Test basic imports
python3 -c "
import sys
sys.path.insert(0, '.')
from orchestration.llm_client import PIIAwareLLMClient, LLMClient
print('✅ Core imports successful')

# Test PII client initialization
try:
    client = PIIAwareLLMClient()
    print('✅ PII-aware client initialized')
except Exception as e:
    print(f'⚠️  PII client issue (still functional): {e}')

# Test basic client
client = LLMClient()
print('✅ Basic LLM client initialized')
print('🚀 Ready to run examples!')
"
```

## 🎯 Step 5: Run the PII-Aware Capital Finder Example

### Execute the Complete Example
```bash
# Make sure virtual environment is active
source venv/bin/activate

# Run the comprehensive reference architecture example
python3 examples/capital_finder_presidio.py
```

### Expected Output Flow

The example runs four demonstrations in sequence:

#### 1. **System Health Check**
```
🚀 PromptForge Reference Architecture: Capital City Finder
   with Microsoft Presidio PII Protection
   Financial Services Grade Implementation

🔧 System Health Check...
   LLM Client: ✅ Healthy
   PII Client: ✅ Healthy
```

#### 2. **Basic Usage Demonstration**
```
============================================================
🌍 BASIC CAPITAL FINDER DEMONSTRATION
============================================================

📝 Query: What is the capital of France?
⏱️  Processing time: 0.123s
💬 Response: The capital of France is Paris.

📝 Query: Tell me about Berlin, Germany's capital
⏱️  Processing time: 0.098s  
💬 Response: Berlin is the capital and largest city of Germany...

✅ Basic Usage demonstration completed
```

#### 3. **PII Protection Demonstration** 
```
============================================================
🛡️  PII PROTECTION DEMONSTRATION
============================================================

📝 Query with PII: Hi, I'm John Smith from New York. What's the capital of my home country?
⏱️  Processing time: 0.445s
🛡️  PII protection: ✅ Enabled
🔍 PII detected: 2 entities
📊 PII actions taken: MASK, REDACT
💬 Response: The capital of the United States is Washington D.C.

🧹 Session cleanup: {"status": "completed", "mappings_cleaned": 3}
```

#### 4. **Performance Monitoring Demonstration**
```
============================================================
📊 PERFORMANCE MONITORING DEMONSTRATION
============================================================

🚀 Processing 5 queries...

📈 PERFORMANCE METRICS:
   Total queries: 5
   Successful: 5
   Failed: 0
   Total wall time: 1.234s
   Average per query: 0.247s
   Throughput: 4.05 queries/second
```

#### 5. **Error Handling Demonstration**
```
============================================================
🛠️  ERROR HANDLING DEMONSTRATION
============================================================

🧪 Edge case: 
⚠️  Error handled gracefully: Invalid input detected

🧪 Edge case: SELECT * FROM capitals WHERE country='France'
✅ Processed successfully
💬 Response: I can help you find capital cities, but I cannot process SQL queries...

✅ Error Handling demonstration completed
```

#### 6. **Final Summary**
```
============================================================
🎉 ALL DEMONSTRATIONS COMPLETED
============================================================

📚 Key Features Demonstrated:
   ✅ PII detection and anonymization
   ✅ Policy-based PII handling
   ✅ Session management and cleanup
   ✅ Multi-provider LLM support
   ✅ Production-grade error handling
   ✅ Performance monitoring
   ✅ Async/await patterns

🔗 Learn more: https://github.com/oiiro/promptforge
```

## 🔍 Step 6: Understanding What Happened

### Core Architecture Components

#### **PIIAwareLLMClient Workflow**
The example demonstrates a 6-step PII protection workflow:

1. **Pre-execution Guardrails**: Input validation and security checks
2. **PII Detection**: Microsoft Presidio analyzes input for sensitive data
3. **PII Anonymization**: Sensitive data replaced with tokens/masks
4. **LLM Generation**: Clean prompt sent to language model
5. **PII Restoration**: Original sensitive data restored in response (if requested)
6. **Response Compilation**: Final response with metadata and audit trail

#### **PII Protection Policies**
The system uses configurable policies:

- **REDACT**: `"John Smith"` → `"[REDACTED]"`
- **MASK**: `"123-45-6789"` → `"XXX-XX-XXXX"`
- **HASH**: `"john@email.com"` → `"<HASH_ABC123>"`
- **TOKENIZE**: `"555-1234"` → `"<TOKEN_456>"`
- **SYNTHETIC**: `"1985-03-15"` → `"1990-01-01"`

#### **Session Management**
- Each conversation gets a unique session ID
- PII mappings stored securely with TTL expiration
- Automatic cleanup prevents data leakage
- Audit trail maintained for compliance

#### **Multi-Provider Support**
- **MockProvider**: Works without API keys for testing
- **OpenAI Provider**: GPT-3.5/GPT-4 integration
- **Anthropic Provider**: Claude integration
- **Graceful Fallbacks**: System works even if providers fail

## 🧪 Step 7: Interactive Testing

### Test Individual Components

#### Test Basic LLM Client
```bash
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
from orchestration.llm_client import LLMClient

async def test_basic():
    client = LLMClient()
    response = await client.generate_async('What is the capital of Japan?')
    print(f'Response: {response[\"content\"]}')

asyncio.run(test_basic())
"
```

#### Test PII Protection
```bash
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
from orchestration.llm_client import PIIAwareLLMClient

async def test_pii():
    client = PIIAwareLLMClient()
    response = await client.generate_with_pii_protection(
        'My name is Alice Johnson and my SSN is 123-45-6789. What is the capital of France?',
        session_id='test_session',
        restore_pii=True
    )
    print(f'PII Protected Response: {response[\"content\"]}')
    print(f'PII Metadata: {response.get(\"pii_metadata\", {})}')

asyncio.run(test_pii())
"
```

#### Test Service Class
```bash
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
from examples.capital_finder_presidio import CapitalFinderService

async def test_service():
    service = CapitalFinderService(enable_pii_protection=True)
    result = await service.find_capital_secure(
        'I live in Tokyo. What country am I in and what is its capital?',
        include_context=True,
        restore_pii=True
    )
    print(f'Service Response: {result[\"response\"][\"content\"]}')
    print(f'Processing Time: {result[\"processing_time_seconds\"]}s')
    
    # Cleanup
    await service.cleanup_session()

asyncio.run(test_service())
"
```

## 📊 Step 8: Advanced Usage

### Custom PII Policies
```bash
# Create custom PII policy test
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')

async def test_custom_policy():
    try:
        from presidio.policies import PIIPolicyEngine, PIIPolicy, PIIAction
        from presidio.middleware import PresidioMiddleware
        
        # Create custom policy
        custom_policy = PIIPolicy(
            name='strict_financial',
            entities={
                'CREDIT_CARD': PIIAction.REDACT,
                'PHONE_NUMBER': PIIAction.MASK,
                'EMAIL_ADDRESS': PIIAction.HASH,
                'PERSON': PIIAction.SYNTHETIC
            },
            metadata={'compliance': 'SOX', 'industry': 'financial'}
        )
        
        middleware = PresidioMiddleware(policy=custom_policy)
        
        test_text = 'Contact John Doe at john.doe@bank.com or call 555-123-4567 regarding credit card 4532-1234-5678-9012'
        
        result = await middleware.anonymize(test_text, session_id='custom_test')
        print(f'Original: {test_text}')
        print(f'Anonymized: {result[\"anonymized_text\"]}')
        print(f'Entities Detected: {len(result[\"entities\"])}')
        
    except ImportError:
        print('⚠️  Presidio not available - using mock example')
        print('Original: Contact [Name] at [Email] or call [Phone] regarding credit card [Card]')
        print('Anonymized: Contact <SYNTHETIC_PERSON> at <HASH_EMAIL> or call XXX-XXX-XXXX regarding credit card [REDACTED]')

asyncio.run(test_custom_policy())
"
```

### Performance Benchmarking
```bash
# Run performance benchmark
python3 -c "
import asyncio
import time
import sys
sys.path.insert(0, '.')
from examples.capital_finder_presidio import CapitalFinderService

async def benchmark():
    service = CapitalFinderService(enable_pii_protection=True)
    
    queries = [
        'What is the capital of France?',
        'Tell me about Tokyo, Japan',
        'What is the capital of Brazil?',
        'I need the capital of Egypt',
        'What is Canada\\'s capital city?'
    ] * 4  # 20 total queries
    
    print(f'🚀 Running benchmark with {len(queries)} queries...')
    start_time = time.time()
    
    tasks = [
        service.find_capital_secure(query, include_context=False)
        for query in queries
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    successful = sum(1 for r in results if not isinstance(r, Exception))
    
    print(f'📈 Benchmark Results:')
    print(f'   Queries: {len(queries)}')
    print(f'   Successful: {successful}')
    print(f'   Total Time: {total_time:.3f}s')
    print(f'   Average: {total_time/successful:.3f}s per query')
    print(f'   Throughput: {successful/total_time:.2f} queries/second')
    
    await service.cleanup_session()

asyncio.run(benchmark())
"
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### Issue: "Module not found" errors
```bash
# Solution: Ensure virtual environment is active
source venv/bin/activate

# Verify Python path
python3 -c "import sys; print('\\n'.join(sys.path))"

# Reinstall dependencies
pip install -r requirements.txt
```

#### Issue: Presidio not available
```bash
# The system gracefully falls back to mock protection
# To install Presidio:
pip install presidio-analyzer presidio-anonymizer
pip install spacy
python -m spacy download en_core_web_sm
```

#### Issue: Redis connection errors
```bash
# The system falls back to in-memory storage
# To use Redis (optional):
# 1. Install Redis server
# 2. Start Redis: redis-server
# 3. Update REDIS_URL in .env file
```

#### Issue: API key errors
```bash
# The MockProvider works without API keys
# For real LLM providers, add keys to .env:
echo "OPENAI_API_KEY=your-actual-key" >> .env
echo "ANTHROPIC_API_KEY=your-actual-key" >> .env
```

#### Issue: Performance issues
```bash
# Check system resources
python3 -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
"

# Reduce concurrency for testing
# Edit examples/capital_finder_presidio.py:
# Change batch size in demonstrate_performance_monitoring()
```

## 📚 Next Steps

### Explore Advanced Features
1. **Custom PII Policies**: Modify `presidio/policies.py`
2. **Enhanced Guardrails**: Extend `guardrails/validators.py`
3. **Custom Providers**: Add new LLM providers to `orchestration/llm_client.py`
4. **TruLens Evaluation**: Run `python3 -m evaluation.offline_evaluation`

### Integration Testing
```bash
# Start the API server
python3 orchestration/app.py

# Test API endpoints
curl -X POST http://localhost:8000/api/v1/capital \
  -H "Content-Type: application/json" \
  -d '{"country": "France"}'
```

### Production Deployment
1. **Environment Configuration**: Set up production `.env`
2. **Database Setup**: Configure PostgreSQL for TruLens
3. **Redis Setup**: Production Redis cluster
4. **Monitoring**: Set up Grafana/Prometheus
5. **Security**: Enable authentication and HTTPS

## 🎯 Key Takeaways

The PII-aware capital finder example demonstrates:

✅ **Enterprise-Grade Architecture**: Production-ready patterns and practices
✅ **Comprehensive PII Protection**: Microsoft Presidio integration with policies
✅ **Security-First Design**: Guardrails, validation, and audit trails
✅ **Performance Optimization**: Async patterns and batching
✅ **Graceful Degradation**: Works even when optional components fail
✅ **Financial Services Ready**: SOC 2, compliance, and governance features
✅ **Developer Experience**: Easy setup, clear examples, comprehensive testing

This reference architecture provides a solid foundation for building production LLM applications with enterprise-grade security and compliance requirements.