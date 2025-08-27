# PromptForge Test Scripts

This directory contains standalone test scripts for PromptForge PII protection and LLM functionality demonstrations.

## Available Scripts

### Core Testing Scripts

#### `health_check.py` 🔍
Comprehensive system health check covering:
- Virtual environment detection
- Dependency verification  
- Import validation
- LLM client health checks
- PII client functionality

```bash
python3 scripts/health_check.py
```

#### `test_basic_llm.py` 🤖
Tests basic LLM client functionality without PII protection:
- Standard LLM client initialization
- Simple query processing
- Response format validation

```bash
python3 scripts/test_basic_llm.py
```

#### `test_pii_protection.py` 🛡️
Tests PII protection capabilities:
- PIIAwareLLMClient initialization
- PII detection and anonymization
- Session management
- Response format handling

```bash
python3 scripts/test_pii_protection.py
```

#### `test_service_class.py` 🏛️
Tests the CapitalFinderService integration:
- Service-level PII protection
- Context-aware responses
- Performance monitoring
- Session cleanup

```bash
python3 scripts/test_service_class.py
```

### Advanced Testing Scripts

#### `test_custom_policy.py` 📋
Demonstrates custom PII policy creation:
- Custom PIIPolicy configuration
- PresidioMiddleware integration
- Entity-specific action mapping
- Compliance framework setup

```bash
python3 scripts/test_custom_policy.py
```

#### `performance_benchmark.py` 📊
Runs performance benchmarking tests:
- Concurrent query processing
- Throughput measurement
- Performance analysis
- Efficiency metrics

```bash
python3 scripts/performance_benchmark.py
```

#### `test_api_server.py` 🚀
Tests the FastAPI orchestration server:
- Health endpoint validation
- Metrics endpoint testing
- Capital finder API testing
- API documentation accessibility

```bash
python3 scripts/test_api_server.py
```

#### `test_api_endpoints.py` 🌐
Curl-style API endpoint testing:
- Authentication flow testing
- Multiple endpoint coverage
- Error handling validation
- Real HTTP request examples

```bash
python3 scripts/test_api_endpoints.py
```

#### `run_all_tests.py` 🧪
Comprehensive test suite runner:
- Executes all test scripts in sequence
- Captures and reports results
- Provides summary statistics
- Error handling and timeouts

```bash
python3 scripts/run_all_tests.py
```

## Prerequisites

Before running the scripts, ensure you have:

1. **Virtual Environment Active**:
   ```bash
   source venv/bin/activate
   ```

2. **Dependencies Installed**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional PII Protection**:
   ```bash
   pip install presidio-analyzer presidio-anonymizer
   pip install spacy
   python -m spacy download en_core_web_sm
   ```

## Script Features

### Error Handling
- Graceful fallbacks for missing dependencies
- Comprehensive exception handling
- Clear error messages and suggestions

### Output Formatting
- Color-coded status indicators
- Structured output with emojis
- Performance metrics and timing
- Detailed success/failure reporting

### PII Protection
- Microsoft Presidio integration
- Configurable anonymization policies
- Session-based PII mapping
- Automatic cleanup routines

### Mock Provider Support
- Works without API keys
- Structured data responses
- Guardrails integration
- Development-friendly testing

## Integration with Documentation

These scripts correspond to the inline examples in `docs/RUNNING_PII_EXAMPLE.md`:

- **Step 7 Interactive Testing** → `test_basic_llm.py`, `test_pii_protection.py`, `test_service_class.py`
- **Step 8 Advanced Usage** → `test_custom_policy.py`, `performance_benchmark.py`
- **System Health Check** → `health_check.py`
- **Comprehensive Testing** → `run_all_tests.py`

## Troubleshooting

### Common Issues

1. **Module Not Found Errors**:
   ```bash
   # Ensure virtual environment is active
   source venv/bin/activate
   
   # Verify Python path
   python3 -c "import sys; print('\n'.join(sys.path))"
   ```

2. **Presidio Not Available**:
   - Scripts gracefully fall back to mock examples
   - Install Presidio for full functionality
   
3. **Redis Connection Errors**:
   - Expected behavior - system falls back to in-memory storage
   - No action required for basic testing

### Performance Considerations

- Concurrent testing may use significant system resources
- Adjust batch sizes in `performance_benchmark.py` if needed
- Monitor system resources during benchmark tests

## Development Usage

Use these scripts for:
- **Development Testing**: Quick verification during development
- **Integration Testing**: End-to-end functionality validation  
- **Performance Analysis**: Benchmarking and optimization
- **Demonstration**: Showcasing PII protection capabilities
- **Documentation**: Live examples for documentation verification