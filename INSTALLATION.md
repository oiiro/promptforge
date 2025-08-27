# PromptForge Installation Guide

**Financial Services Grade Prompt Engineering SDLC**

Complete setup instructions for PromptForge - a production-ready framework for managing, testing, and deploying prompts in financial services environments with enterprise-grade security, compliance, and observability.

## 🚀 Quick Setup (Automated)

For the fastest setup experience, use the automated setup script:

```bash
# Clone or navigate to the PromptForge project
cd promptforge

# Run automated setup
./setup.sh

# Verify installation
./verify.sh
```

The automated setup will:
- ✅ Check system prerequisites
- ✅ Install system dependencies (macOS via Homebrew)
- ✅ Create Python virtual environment
- ✅ Install all Python dependencies
- ✅ Create configuration template
- ✅ Verify installation completeness

## 📋 Prerequisites

### System Requirements

- **Python**: 3.9+ (tested with 3.9, 3.10, 3.11, 3.13)
- **Operating System**: macOS, Linux, Windows (WSL recommended)
- **Memory**: 4GB RAM minimum (8GB recommended for ML features)
- **Disk Space**: 2GB free space

### macOS Specific Requirements

For optimal compatibility, install system dependencies:

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system libraries for XML processing
brew install libxml2 libxslt pkg-config
```

### API Keys Required

You'll need API keys from at least one LLM provider:

- **OpenAI**: Get from https://platform.openai.com/api-keys
- **Anthropic**: Get from https://console.anthropic.com/

## 🔧 Manual Setup (Step-by-Step)

### 1. Environment Setup

```bash
# Navigate to project directory
cd promptforge

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# Or: venv\Scripts\activate  # On Windows

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

#### Option A: Install All Dependencies at Once
```bash
pip install -r requirements.txt
```

#### Option B: Install in Stages (Recommended for troubleshooting)
```bash
# Core dependencies
pip install openai anthropic pydantic python-dotenv PyYAML jsonschema

# Evaluation framework
pip install deepeval pytest pytest-cov

# API framework
pip install fastapi uvicorn httpx

# Data processing
pip install pandas numpy

# Security
pip install cryptography "passlib[bcrypt]"

# Observability
pip install opentelemetry-api opentelemetry-sdk structlog

# Development tools
pip install black flake8 mypy

# Advanced AI packages (may take longer)
pip install guardrails-ai detoxify transformers torch
```

### 3. Configuration Setup

```bash
# Copy environment template
cp .env.example .env

# Edit configuration file
# Add your API keys and configure settings
nano .env  # or use your preferred editor
```

#### Required Configuration

Edit `.env` file with your settings:

```bash
# LLM Provider Configuration
DEFAULT_LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key-here

# Or use Anthropic
DEFAULT_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG_MODE=true
```

### 4. Verification

```bash
# Run installation verification
python check_installation.py

# Or use comprehensive verification
./verify.sh
```

## 🧪 Testing Your Installation

### 1. Package Verification
```bash
# Activate virtual environment
source venv/bin/activate

# Check all packages
python check_installation.py
```

Expected output:
```
🔍 PromptForge Installation Status Check
==================================================

📦 Core LLM Packages:
✅ openai               OK
✅ anthropic            OK

🧪 Evaluation Framework:
✅ deepeval             OK
✅ pytest               OK

[... all packages should show ✅ OK]
```

### 2. API Server Test
```bash
# Start the API server
python orchestration/app.py

# In another terminal, test endpoints
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "timestamp": "...", "version": "1.0.0"}
```

### 3. Core Functionality Test
```bash
# Test capital finder with demo token
curl -X POST http://localhost:8000/api/v1/capital \
  -H "Authorization: Bearer demo-token" \
  -H "Content-Type: application/json" \
  -d '{"country": "France"}'

# Expected response:
# {"capital": "Paris", "confidence": 1.0, "metadata": {...}}
```

### 4. Run Test Suite
```bash
# Make test runner executable
chmod +x ci/run_tests.sh

# Run comprehensive tests
./ci/run_tests.sh
```

## 🔍 Troubleshooting

### Common Installation Issues

#### 1. Python Version Compatibility
```bash
# Check Python version
python3 --version

# PromptForge requires Python 3.9+
# If using older version, upgrade Python first
```

#### 2. Virtual Environment Issues
```bash
# Remove existing virtual environment
rm -rf venv

# Recreate with specific Python version
python3.11 -m venv venv  # or python3.10, python3.9
source venv/bin/activate
```

#### 3. Package Installation Failures

**For lxml/xml processing errors (common on macOS):**
```bash
# Install system dependencies
brew install libxml2 libxslt
export LDFLAGS="-L$(brew --prefix libxml2)/lib -L$(brew --prefix libxslt)/lib"
export CPPFLAGS="-I$(brew --prefix libxml2)/include -I$(brew --prefix libxslt)/include"

# Then retry pip install
pip install lxml
```

**For PyTorch/ML package errors:**
```bash
# Install CPU-only version for faster installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For guardrails-ai issues:**
```bash
# Install dependencies separately
pip install detoxify transformers
pip install guardrails-ai
```

#### 4. API Key Configuration
```bash
# Check if .env file exists and has correct format
cat .env

# Verify API key format (should not contain quotes)
# Correct: OPENAI_API_KEY=sk-proj-abc123...
# Incorrect: OPENAI_API_KEY="sk-proj-abc123..."
```

#### 5. Permission Issues
```bash
# Make scripts executable
chmod +x setup.sh verify.sh ci/run_tests.sh

# If still having issues, try:
sudo chown -R $USER:$USER .
```

### Verification Commands

```bash
# Check virtual environment
which python
# Should show: /path/to/promptforge/venv/bin/python

# Verify key packages
python -c "import openai; print('OpenAI:', openai.__version__)"
python -c "import anthropic; print('Anthropic:', anthropic.__version__)"
python -c "import deepeval; print('DeepEval:', deepeval.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"

# Test LLM client
python -c "from orchestration.llm_client import LLMClient; print('LLM Client: OK')"

# Test guardrails
python -c "from guardrails.validators import GuardrailOrchestrator; print('Guardrails: OK')"
```

## 📚 Next Steps

After successful installation:

1. **Configure API Keys**: Add your LLM provider API keys to `.env`

2. **Review Configuration**: Check `config/governance.yml` for compliance settings

3. **Start Development**: 
   ```bash
   # Start API server
   python orchestration/app.py
   
   # Access API documentation
   open http://localhost:8000/docs
   ```

4. **Run Tests**: 
   ```bash
   ./ci/run_tests.sh
   ```

5. **Explore Examples**:
   - Check `prompts/find_capital/` for prompt templates
   - Review `datasets/` for test data
   - Examine `evals/` for evaluation examples

## 🆘 Getting Help

### Documentation
- **README.md**: Project overview and quick start
- **API Reference**: `docs/api_reference.md`
- **Security Guide**: `docs/security_guide.md`
- **Compliance**: `docs/compliance.md`

### Support Channels
- **Issues**: Check existing documentation first
- **Logs**: Review `logs/` directory for error details
- **Health Check**: Use `./verify.sh` for diagnostic information

### Environment Information
If you encounter issues, collect this information:

```bash
# System info
uname -a
python3 --version
pip --version

# PromptForge info
./verify.sh > installation_report.txt
```

---

**PromptForge** - Production-grade prompt engineering for financial services. Built with security, compliance, and reliability as core principles.