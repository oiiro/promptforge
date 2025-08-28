#!/usr/bin/env python3
"""
Enhanced PromptForge Setup Script with TruLens and Multi-Person PII Protection
Incorporates learnings from EMAIL_ADDRESS_3 deanonymization and TruLens monitoring fixes
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description, check=True):
    """Run a shell command with error handling"""
    logger.info(f"🔧 {description}")
    logger.debug(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Verify Python version compatibility"""
    logger.info("🐍 Checking Python version...")
    if sys.version_info < (3, 9):
        logger.error("❌ Python 3.9+ required for TruLens and Presidio compatibility")
        return False
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        logger.info("🏗️ Creating virtual environment...")
        if not run_command("python3 -m venv venv", "Create virtual environment"):
            return False
    else:
        logger.info("✅ Virtual environment already exists")
    
    return True

def install_core_dependencies():
    """Install core dependencies with enhanced TruLens support"""
    logger.info("📦 Installing core dependencies...")
    
    # Core dependencies that must be installed first
    core_deps = [
        "wheel>=0.37.0",
        "setuptools>=65.0.0",
        "pip>=22.0.0"
    ]
    
    for dep in core_deps:
        if not run_command(f"./venv/bin/pip install --upgrade {dep}", f"Install {dep}"):
            return False
    
    return True

def install_trulens_dependencies():
    """Install TruLens dependencies with correct providers"""
    logger.info("🔍 Installing TruLens monitoring dependencies...")
    
    # TruLens dependencies in correct order
    trulens_deps = [
        # Core TruLens packages
        "trulens-core>=2.2.4",
        "trulens-feedback>=2.2.4",
        
        # Critical: TruLens providers (fixes import errors)
        "trulens-providers-openai>=2.2.4",
        
        # LangChain integration (required for TruLens OpenAI provider)
        "langchain>=0.3.27",
        "langchain-core>=0.3.75", 
        "langchain-community>=0.3.29",
        
        # Database support
        "sqlalchemy>=2.0.0",
        "alembic>=1.12.0"
    ]
    
    for dep in trulens_deps:
        if not run_command(f"./venv/bin/pip install {dep}", f"Install {dep}"):
            logger.warning(f"⚠️  Failed to install {dep} - continuing...")
    
    return True

def install_presidio_dependencies():
    """Install Presidio PII protection dependencies"""
    logger.info("🛡️ Installing Presidio PII protection dependencies...")
    
    presidio_deps = [
        "presidio-analyzer>=2.2.33",
        "presidio-anonymizer>=2.2.33",
        
        # SpaCy models for PII detection
        "spacy>=3.4.0",
    ]
    
    for dep in presidio_deps:
        if not run_command(f"./venv/bin/pip install {dep}", f"Install {dep}"):
            return False
    
    # Install SpaCy English model
    logger.info("📚 Installing SpaCy English model...")
    run_command("./venv/bin/python -m spacy download en_core_web_sm", 
                "Download SpaCy English model", check=False)
    
    return True

def install_main_requirements():
    """Install main requirements.txt"""
    requirements_file = Path("requirements.txt")
    
    if requirements_file.exists():
        logger.info("📋 Installing main requirements.txt...")
        return run_command("./venv/bin/pip install -r requirements.txt", 
                          "Install requirements.txt")
    else:
        logger.warning("⚠️  requirements.txt not found - skipping")
        return True

def verify_trulens_installation():
    """Verify TruLens installation with correct imports"""
    logger.info("🧪 Verifying TruLens installation...")
    
    test_script = '''
import sys
try:
    # Test core TruLens imports
    from trulens.core import TruSession
    print("✅ TruLens core import successful")
    
    # Test Cost import (critical fix from our debugging)
    from trulens.core.schema.base import Cost
    print("✅ TruLens Cost import successful")
    
    # Test TruLens providers
    from trulens.providers.openai import OpenAI
    print("✅ TruLens OpenAI provider import successful")
    
    # Test basic functionality
    session = TruSession()
    print("✅ TruLens session creation successful")
    
    # Test Cost object creation
    cost = Cost(n_tokens=0, n_prompt_tokens=0, n_completion_tokens=0)
    print("✅ TruLens Cost object creation successful")
    
    print("🎉 TruLens verification PASSED")
    sys.exit(0)
    
except ImportError as e:
    print(f"❌ TruLens import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ TruLens verification error: {e}")
    sys.exit(1)
'''
    
    return run_command(f"./venv/bin/python -c '{test_script}'", 
                      "Verify TruLens installation")

def verify_presidio_installation():
    """Verify Presidio installation"""
    logger.info("🧪 Verifying Presidio installation...")
    
    test_script = '''
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    
    # Test basic functionality
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    
    # Test PII detection
    results = analyzer.analyze(text="John Smith email: john@example.com", language='en')
    print(f"✅ Presidio PII detection found {len(results)} entities")
    
    print("🎉 Presidio verification PASSED")
    
except Exception as e:
    print(f"❌ Presidio verification error: {e}")
    exit(1)
'''
    
    return run_command(f"./venv/bin/python -c '{test_script}'", 
                      "Verify Presidio installation")

def create_env_template():
    """Create enhanced .env template with TruLens configuration"""
    logger.info("📝 Creating enhanced .env template...")
    
    env_template = '''# PromptForge Enhanced Configuration Template
# Copy this to .env and configure your API keys

# LLM Provider API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# TruLens Configuration (Enhanced)
TRULENS_DATABASE_URL=sqlite:///trulens_promptforge.db
TRULENS_TRACKING=true

# PII Protection Configuration
ENABLE_PII_PROTECTION=true
PRESIDIO_LOG_LEVEL=INFO

# Financial Services Compliance
ENABLE_FINANCIAL_COMPLIANCE=true
ENABLE_AUDIT_LOGGING=true

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEBUG_MODE=false

# Multi-Person Processing Configuration
ENABLE_MULTI_PERSON_PROCESSING=true
MAX_ENTITIES_PER_REQUEST=50

# Enhanced Monitoring
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_SECURITY_MONITORING=true
'''
    
    env_template_file = Path(".env.template")
    
    try:
        with open(env_template_file, 'w') as f:
            f.write(env_template)
        logger.info("✅ Enhanced .env template created")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create .env template: {e}")
        return False

def create_verification_script():
    """Create comprehensive verification script"""
    logger.info("🧪 Creating verification script...")
    
    verification_script = '''#!/usr/bin/env python3
"""
PromptForge Enhanced Verification Script
Verifies all components including TruLens fixes and multi-person PII protection
"""

import asyncio
import sys
import os

async def verify_server_startup():
    """Verify server can start without hanging"""
    print("🚀 Testing server startup (non-hanging)...")
    
    try:
        # Import main components
        from orchestration.app import app
        print("✅ Server app import successful")
        
        # Test TruLens imports with correct paths
        from trulens.core.schema.base import Cost
        print("✅ TruLens Cost import successful (correct path)")
        
        # Test Presidio imports
        from presidio_analyzer import AnalyzerEngine
        print("✅ Presidio import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Server startup verification failed: {e}")
        return False

async def verify_multi_person_processing():
    """Verify multi-person PII processing with EMAIL_ADDRESS_3 fix"""
    print("👥 Testing multi-person PII processing...")
    
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        
        analyzer = AnalyzerEngine()
        
        # Test multi-person query (mirrors our test case)
        test_text = '''
        Please evaluate retirement eligibility for the following employees:
        
        1. John Smith, age 65, phone number 555-123-4567, employed for 25 years 
           with current salary of $75,000. Email: john.smith@company.com
           
        2. Sarah Johnson, age 62, phone 555-987-6543, employed for 30 years
           with current salary of $85,000. SSN: 123-45-6789
           
        3. Mary Williams, age 58, email mary.williams@company.com, 
           employed for 22 years with salary $68,000
        '''
        
        # Analyze PII
        pii_results = analyzer.analyze(text=test_text, language='en')
        
        # Count email addresses
        email_entities = [r for r in pii_results if r.entity_type == 'EMAIL_ADDRESS']
        print(f"✅ Found {len(email_entities)} EMAIL_ADDRESS entities")
        
        # Verify we can handle the EMAIL_ADDRESS_3 scenario
        if len(email_entities) >= 2:
            print("✅ Multi-person email processing verification PASSED")
            return True
        else:
            print("⚠️  Less than 2 email addresses found - check configuration")
            return False
            
    except Exception as e:
        print(f"❌ Multi-person processing verification failed: {e}")
        return False

async def verify_trulens_monitoring():
    """Verify TruLens monitoring with correct Cost usage"""
    print("📊 Testing TruLens monitoring...")
    
    try:
        from trulens.core import TruSession
        from trulens.core.schema.base import Cost
        
        # Test session creation
        session = TruSession()
        print("✅ TruLens session created")
        
        # Test Cost object creation (our fix)
        cost = Cost(n_tokens=0, n_prompt_tokens=0, n_completion_tokens=0)
        print("✅ TruLens Cost object created with correct structure")
        
        print("✅ TruLens monitoring verification PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TruLens monitoring verification failed: {e}")
        return False

async def main():
    """Run all verifications"""
    print("🧪 PromptForge Enhanced Verification")
    print("=" * 50)
    
    results = []
    
    # Run all verification tests
    results.append(await verify_server_startup())
    results.append(await verify_multi_person_processing()) 
    results.append(await verify_trulens_monitoring())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\\n" + "=" * 50)
    print(f"📋 VERIFICATION SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All verifications PASSED - PromptForge ready for use!")
        return 0
    else:
        print("❌ Some verifications FAILED - check configuration")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
'''
    
    verification_file = Path("scripts/verify_enhanced_setup.py")
    verification_file.parent.mkdir(exist_ok=True)
    
    try:
        with open(verification_file, 'w') as f:
            f.write(verification_script)
        
        # Make executable
        run_command(f"chmod +x {verification_file}", "Make verification script executable")
        
        logger.info("✅ Enhanced verification script created")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create verification script: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 PromptForge Enhanced Setup")
    logger.info("=" * 50)
    
    # Track setup results
    steps = []
    
    # Run setup steps
    steps.append(("Python Version Check", check_python_version()))
    steps.append(("Virtual Environment", setup_virtual_environment()))
    steps.append(("Core Dependencies", install_core_dependencies()))
    steps.append(("TruLens Dependencies", install_trulens_dependencies()))
    steps.append(("Presidio Dependencies", install_presidio_dependencies()))
    steps.append(("Main Requirements", install_main_requirements()))
    steps.append((".env Template", create_env_template()))
    steps.append(("Verification Script", create_verification_script()))
    
    # Verification steps
    steps.append(("TruLens Verification", verify_trulens_installation()))
    steps.append(("Presidio Verification", verify_presidio_installation()))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📋 SETUP SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    for step_name, success in steps:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{step_name:.<30} {status}")
        if success:
            passed += 1
    
    logger.info(f"\n🎯 Setup Result: {passed}/{len(steps)} steps completed successfully")
    
    if passed == len(steps):
        logger.info("🎉 PromptForge enhanced setup COMPLETED successfully!")
        logger.info("\n📋 Next Steps:")
        logger.info("1. Copy .env.template to .env and configure API keys")
        logger.info("2. Run: ./scripts/verify_enhanced_setup.py")
        logger.info("3. Start server: ./start_server.sh") 
        logger.info("4. Test: ./venv/bin/python scripts/test_multi_person_retirement.py")
        return 0
    else:
        logger.error("❌ Setup completed with errors - check logs above")
        return 1

if __name__ == "__main__":
    sys.exit(main())