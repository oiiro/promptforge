#!/usr/bin/env python3
"""
PromptForge Setup & Verification Script
Comprehensive setup and verification for financial services grade prompt engineering SDLC
"""

import os
import sys
import subprocess
import logging
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class PromptForgeSetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.venv_path = self.project_root / "venv"
        self.results = {}
        
    def run_command(self, cmd: str, check: bool = True) -> Tuple[bool, str]:
        """Run shell command and return success status and output"""
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                check=check
            )
            return True, result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return False, e.stderr.strip()
    
    def check_prerequisites(self) -> bool:
        """Check Python version and basic requirements"""
        logger.info("🔍 Checking prerequisites...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 9):
            logger.error(f"❌ Python 3.9+ required, found {python_version.major}.{python_version.minor}")
            return False
        
        logger.info(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
        
        # Check required files
        required_files = [
            "requirements.txt",
            "orchestration/llm_client.py",
            "evaluation/trulens_config.py",
            "guardrails/validators.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"❌ Missing required files: {', '.join(missing_files)}")
            return False
        
        logger.info("✅ All required project files present")
        return True
    
    def setup_virtual_environment(self) -> bool:
        """Create and activate virtual environment"""
        logger.info("🔧 Setting up virtual environment...")
        
        # Remove existing venv if present
        if self.venv_path.exists():
            logger.info("📁 Removing existing virtual environment...")
            success, output = self.run_command(f"rm -rf {self.venv_path}")
            if not success:
                logger.error(f"❌ Failed to remove existing venv: {output}")
                return False
        
        # Create new virtual environment
        success, output = self.run_command(f"python3 -m venv {self.venv_path}")
        if not success:
            logger.error(f"❌ Failed to create virtual environment: {output}")
            return False
        
        logger.info("✅ Virtual environment created")
        
        # Upgrade pip
        pip_path = self.venv_path / "bin" / "pip"
        success, output = self.run_command(f"{pip_path} install --upgrade pip")
        if not success:
            logger.warning(f"⚠️  Failed to upgrade pip: {output}")
        else:
            logger.info("✅ pip upgraded")
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies in stages"""
        logger.info("📦 Installing Python dependencies...")
        pip_path = self.venv_path / "bin" / "pip"
        
        # Core dependency groups for better error handling
        dependency_groups = [
            ("Core LLM", ["openai>=1.35.0", "anthropic>=0.25.0", "pydantic>=2.7.0", "python-dotenv>=1.0.0", "PyYAML>=6.0.1"]),
            ("TruLens Evaluation", ["trulens-core>=2.2.4", "trulens-feedback>=2.2.4"]),
            ("PII Detection", ["presidio-analyzer>=2.2.35", "presidio-anonymizer>=2.2.35", "spacy>=3.7.0"]),
            ("Session Storage", ["redis>=5.0.0"]),
            ("Additional Evaluation", ["deepeval>=0.21.0", "detoxify>=0.5.0"]),
            ("API Framework", ["fastapi>=0.110.0", "uvicorn>=0.29.0", "httpx>=0.27.0"]),
            ("Data Processing", ["pandas>=2.2.0", "numpy>=1.26.0"]),
            ("Security", ["cryptography>=42.0.0", "passlib[bcrypt]>=1.7.4", "jsonschema>=4.21.0"]),
            ("Observability", ["opentelemetry-api>=1.36.0", "opentelemetry-sdk>=1.36.0", "opentelemetry-exporter-otlp>=1.36.0", "opentelemetry-instrumentation-requests>=0.46b0", "opentelemetry-instrumentation-fastapi>=0.46b0", "opentelemetry-instrumentation-httpx>=0.46b0", "structlog>=24.1.0"]),
            ("Testing", ["pytest>=8.2.0", "pytest-cov>=5.0.0"]),
            ("Development", ["black>=24.3.0", "flake8>=7.0.0", "mypy>=1.9.0"]),
            # Skip Advanced AI packages that may have build issues on Python 3.13
            # ("Advanced AI", ["guardrails-ai>=0.4.0", "transformers>=4.35.0"])
        ]
        
        failed_groups = []
        for group_name, packages in dependency_groups:
            logger.info(f"   Installing {group_name}...")
            # Use subprocess with proper argument list to avoid shell parsing issues
            cmd_args = [str(pip_path), "install"] + packages
            try:
                result = subprocess.run(
                    cmd_args,
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=self.project_root
                )
                success = result.returncode == 0
                output = result.stdout if success else result.stderr
            except Exception as e:
                success = False
                output = str(e)
            
            if not success:
                logger.warning(f"⚠️  {group_name} installation had issues: {output}")
                failed_groups.append(group_name)
            else:
                logger.info(f"✅ {group_name} installed successfully")
        
        if failed_groups:
            logger.warning(f"⚠️  Some packages failed: {', '.join(failed_groups)}")
        else:
            logger.info("✅ All dependency groups installed successfully")
        
        # Install spaCy language model (required for Presidio PII detection)
        logger.info("🔤 Installing spaCy language model for PII detection...")
        success, output = self.run_command(f"{str(pip_path)} install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl", check=False)
        if not success:
            logger.warning(f"⚠️  spaCy language model installation failed: {output}")
            logger.info("   PII detection may not work properly without this model")
        else:
            logger.info("✅ spaCy language model installed successfully")
        
        return len(failed_groups) == 0
    
    def setup_configuration(self) -> bool:
        """Set up configuration files"""
        logger.info("🔧 Setting up configuration...")
        
        env_file = self.project_root / ".env"
        env_template = self.project_root / ".env.example"
        
        # Create .env from template if it doesn't exist
        if not env_file.exists():
            if env_template.exists():
                success, output = self.run_command(f"cp {env_template} {env_file}")
                if success:
                    logger.info("✅ Created .env file from template")
                else:
                    logger.error(f"❌ Failed to create .env file: {output}")
                    return False
            else:
                # Create basic .env file
                env_content = """# PromptForge Configuration
# Add your API keys here

# LLM Provider Configuration
DEFAULT_LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# TruLens Configuration
TRULENS_DATABASE_URL=sqlite:///trulens_promptforge.db
TRULENS_LOG_LEVEL=INFO

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
API_HOST=localhost
API_PORT=8000

# Security Configuration
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
"""
                with open(env_file, 'w') as f:
                    f.write(env_content)
                logger.info("✅ Created basic .env file")
        else:
            logger.info("ℹ️  .env file already exists")
        
        return True
    
    def verify_installation(self) -> Dict[str, bool]:
        """Comprehensive verification of the installation"""
        logger.info("🧪 Verifying installation...")
        
        python_path = self.venv_path / "bin" / "python"
        verification_results = {}
        
        # Test core imports
        test_script = """
import sys
sys.path.insert(0, '.')

results = {}

# Test TruLens imports
try:
    from trulens.core import TruSession
    from trulens.core.feedback import Feedback
    results['trulens_imports'] = True
    print("✅ TruLens core imports successful")
except Exception as e:
    results['trulens_imports'] = False
    print(f"❌ TruLens import error: {e}")
    # Try legacy import
    try:
        import trulens_eval
        results['trulens_imports'] = True
        print("✅ TruLens legacy imports successful")
    except Exception as e2:
        print(f"❌ TruLens legacy also failed: {e2}")

# Test LLM client
try:
    from orchestration.llm_client import LLMClient
    client = LLMClient()
    results['llm_client'] = True
except Exception as e:
    results['llm_client'] = False
    print(f"LLM client error: {e}")

# Test TruLens configuration
try:
    from evaluation.trulens_config import TruLensConfig
    config = TruLensConfig()
    results['trulens_config'] = True
except Exception as e:
    results['trulens_config'] = False
    print(f"TruLens config error: {e}")

# Test guardrails
try:
    from guardrails.validators import GuardrailOrchestrator
    guardrails = GuardrailOrchestrator()
    results['guardrails'] = True
except Exception as e:
    results['guardrails'] = False
    print(f"Guardrails error: {e}")

# Test evaluation systems
try:
    from evaluation.offline_evaluation import OfflineEvaluator
    from evaluation.production_monitoring import ProductionMonitor
    results['evaluation_systems'] = True
except Exception as e:
    results['evaluation_systems'] = False
    print(f"Evaluation systems error: {e}")

# Test key dependencies
dependencies = {
    'fastapi': 'fastapi',
    'pandas': 'pandas', 
    'numpy': 'numpy',
    'openai': 'openai',
    'anthropic': 'anthropic',
    'deepeval': 'deepeval',
    'detoxify': 'detoxify',
    'redis': 'redis',
    'presidio_analyzer': 'presidio_analyzer',
    'presidio_anonymizer': 'presidio_anonymizer',
    'spacy': 'spacy'
}

# Test OpenTelemetry specifically (since it was a common failure point)
try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    results['opentelemetry_instrumentation'] = True
    print("✅ OpenTelemetry instrumentation imports successful")
except Exception as e:
    results['opentelemetry_instrumentation'] = False
    print(f"❌ OpenTelemetry instrumentation error: {e}")

# Test observability/tracing module (our custom tracing module)
try:
    from observability.tracing import TracingManager
    tracer = TracingManager()
    results['tracing_manager'] = True
    print("✅ TracingManager initialization successful")
except Exception as e:
    results['tracing_manager'] = False
    print(f"❌ TracingManager error: {e}")

for name, module in dependencies.items():
    try:
        __import__(module)
        results[f'dep_{name}'] = True
    except ImportError:
        results[f'dep_{name}'] = False

print(f"VERIFICATION_RESULTS: {results}")
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            test_file = f.name
        
        try:
            success, output = self.run_command(f"{python_path} {test_file}", check=False)
            
            # Parse results from output
            for line in output.split('\n'):
                if line.startswith('VERIFICATION_RESULTS:'):
                    import ast
                    results_str = line.replace('VERIFICATION_RESULTS: ', '')
                    verification_results = ast.literal_eval(results_str)
                    break
            
        finally:
            os.unlink(test_file)
        
        return verification_results
    
    def test_api_server(self) -> bool:
        """Test API server startup"""
        logger.info("🌐 Testing API server...")
        
        python_path = self.venv_path / "bin" / "python"
        
        # Start server in background
        try:
            import signal
            import time
            
            server_process = subprocess.Popen(
                [str(python_path), "orchestration/app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(3)
            
            # Test health endpoint
            success, output = self.run_command("curl -s http://localhost:8000/health", check=False)
            
            # Clean up
            server_process.send_signal(signal.SIGTERM)
            server_process.wait(timeout=5)
            
            if success and "status" in output.lower():
                logger.info("✅ API server test successful")
                return True
            else:
                logger.warning("⚠️  API server test failed")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️  API server test failed: {e}")
            return False
    
    def generate_report(self, verification_results: Dict[str, bool], api_test: bool) -> None:
        """Generate comprehensive setup report"""
        logger.info("\n" + "="*60)
        logger.info("📋 PROMPTFORGE SETUP REPORT")
        logger.info("="*60)
        
        # Core components
        core_components = {
            'TruLens Imports': verification_results.get('trulens_imports', False),
            'LLM Client': verification_results.get('llm_client', False),
            'TruLens Configuration': verification_results.get('trulens_config', False),
            'Guardrails System': verification_results.get('guardrails', False),
            'Evaluation Systems': verification_results.get('evaluation_systems', False)
        }
        
        logger.info("\n🔧 Core Components:")
        for component, status in core_components.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"  {status_icon} {component}")
        
        # Dependencies
        dependencies = {
            'FastAPI': verification_results.get('dep_fastapi', False),
            'Pandas': verification_results.get('dep_pandas', False),
            'NumPy': verification_results.get('dep_numpy', False),
            'OpenAI': verification_results.get('dep_openai', False),
            'Anthropic': verification_results.get('dep_anthropic', False),
            'DeepEval': verification_results.get('dep_deepeval', False),
            'Detoxify': verification_results.get('dep_detoxify', False),
            'Redis': verification_results.get('dep_redis', False),
            'Presidio Analyzer': verification_results.get('dep_presidio_analyzer', False),
            'Presidio Anonymizer': verification_results.get('dep_presidio_anonymizer', False),
            'spaCy': verification_results.get('dep_spacy', False),
            'OpenTelemetry Instrumentation': verification_results.get('opentelemetry_instrumentation', False),
            'Tracing Manager': verification_results.get('tracing_manager', False)
        }
        
        logger.info("\n📦 Dependencies:")
        for dep, status in dependencies.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"  {status_icon} {dep}")
        
        # API Server
        api_icon = "✅" if api_test else "⚠️ "
        logger.info(f"\n🌐 API Server: {api_icon}")
        
        # Calculate overall score
        all_tests = list(core_components.values()) + list(dependencies.values()) + [api_test]
        passed_tests = sum(all_tests)
        total_tests = len(all_tests)
        
        logger.info(f"\n📊 Overall Score: {passed_tests}/{total_tests} tests passed")
        
        # Recommendations
        logger.info("\n💡 Next Steps:")
        if passed_tests == total_tests:
            logger.info("  🎉 PromptForge is fully ready!")
            logger.info("  🚀 Start the API server: python orchestration/app.py")
            logger.info("  📚 API docs: http://localhost:8000/docs")
        elif passed_tests > total_tests * 0.8:
            logger.info("  ⚠️  PromptForge is mostly ready")
            logger.info("  🔧 Add API keys to .env file for full functionality")
            if not verification_results.get('dep_openai', True):
                logger.info("  📦 Install missing dependencies: pip install openai anthropic")
        else:
            logger.info("  ❌ Setup incomplete - review errors above")
            logger.info("  🆘 Try running this script again")
        
        logger.info("\n📖 Documentation:")
        logger.info("  • TruLens Integration: docs/TRULENS_INTEGRATION.md")
        logger.info("  • API Reference: docs/api_reference.md") 
        logger.info("  • Security Guide: docs/security_guide.md")
        
        # Save results for future reference
        import datetime
        report_data = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'core_components': core_components,
            'dependencies': dependencies,
            'api_server': api_test,
            'score': f"{passed_tests}/{total_tests}",
            'success': passed_tests >= total_tests * 0.8
        }
        
        with open(self.project_root / "setup_report.json", 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"\n📄 Detailed report saved: setup_report.json")

    def run_setup(self) -> bool:
        """Run complete setup process"""
        logger.info("🚀 PromptForge Setup & Verification")
        logger.info("="*50)
        
        try:
            # Prerequisites
            if not self.check_prerequisites():
                return False
            
            # Virtual environment
            if not self.setup_virtual_environment():
                return False
            
            # Dependencies
            deps_success = self.install_dependencies()
            
            # Configuration
            if not self.setup_configuration():
                return False
            
            # Verification
            verification_results = self.verify_installation()
            
            # API server test
            api_test = self.test_api_server()
            
            # Generate report
            self.generate_report(verification_results, api_test)
            
            return deps_success and len(verification_results) > 0
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Setup interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False

def main():
    """Main entry point"""
    setup = PromptForgeSetup()
    success = setup.run_setup()
    
    if success:
        logger.info("\n🎉 Setup completed successfully!")
    else:
        logger.error("\n❌ Setup failed - check errors above")
        sys.exit(1)

if __name__ == "__main__":
    main()