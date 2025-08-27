#!/usr/bin/env python3
"""
PromptForge Installation Verification
Comprehensive verification of TruLens integration and system components
"""

import os
import sys
import subprocess
import logging
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class InstallationVerifier:
    def __init__(self):
        self.project_root = Path.cwd()
        self.venv_path = self.project_root / "venv"
        self.python_path = self.venv_path / "bin" / "python"
        
    def check_environment(self) -> bool:
        """Check if virtual environment and Python are available"""
        if not self.venv_path.exists():
            logger.error("❌ Virtual environment not found. Run setup_promptforge.py first.")
            return False
            
        if not self.python_path.exists():
            logger.error("❌ Python executable not found in virtual environment")
            return False
            
        logger.info("✅ Virtual environment detected")
        return True
    
    def test_core_imports(self) -> Dict[str, bool]:
        """Test all core module imports"""
        logger.info("📋 Testing Core Imports...")
        
        test_script = """
import sys
sys.path.insert(0, '.')

results = {}

# Test TruLens core imports
try:
    from trulens.core import TruSession
    from trulens.feedback import Feedback
    results['trulens_core'] = True
    print("✅ TruLens core imports successful")
except Exception as e:
    results['trulens_core'] = False
    print(f"❌ TruLens core imports failed: {e}")

# Test TruLens legacy compatibility
try:
    import trulens_eval
    results['trulens_legacy'] = True
    print("✅ TruLens legacy compatibility")
except Exception as e:
    results['trulens_legacy'] = False
    print(f"⚠️  TruLens legacy not available: {e}")

# Test LLM client
try:
    from orchestration.llm_client import LLMClient
    client = LLMClient()
    results['llm_client'] = True
    print(f"✅ LLM client initialized - provider: {client.provider}")
except Exception as e:
    results['llm_client'] = False
    print(f"❌ LLM client failed: {e}")

# Test TruLens configuration
try:
    from evaluation.trulens_config import TruLensConfig
    config = TruLensConfig()
    results['trulens_config'] = True
    print("✅ TruLens configuration loaded")
except Exception as e:
    results['trulens_config'] = False
    print(f"❌ TruLens config failed: {e}")

# Test guardrails
try:
    from guardrails.validators import GuardrailOrchestrator
    guardrails = GuardrailOrchestrator()
    is_valid, sanitized, violations = guardrails.validate_request('Test query')
    results['guardrails'] = True
    print(f"✅ Guardrails functional - validation: {is_valid}")
except Exception as e:
    results['guardrails'] = False
    print(f"❌ Guardrails failed: {e}")

# Test evaluation systems
try:
    from evaluation.offline_evaluation import OfflineEvaluator
    from evaluation.production_monitoring import ProductionMonitor
    results['evaluation_systems'] = True
    print("✅ Evaluation systems loaded")
except Exception as e:
    results['evaluation_systems'] = False
    print(f"❌ Evaluation systems failed: {e}")

# Test observability
try:
    from observability.metrics import MetricsCollector
    collector = MetricsCollector()
    collector.increment('test_metric', 1)
    results['observability'] = True
    print("✅ Observability metrics working")
except Exception as e:
    results['observability'] = False
    print(f"⚠️  Observability not available: {e}")

print(f"CORE_RESULTS:{results}")
"""
        
        results = self._run_test_script(test_script, "CORE_RESULTS:")
        return results or {}
    
    def test_dependencies(self) -> Dict[str, bool]:
        """Test key dependencies"""
        logger.info("📋 Testing Dependencies...")
        
        test_script = """
results = {}

# Key dependencies
dependencies = {
    'openai': 'OpenAI',
    'anthropic': 'Anthropic', 
    'fastapi': 'FastAPI',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'deepeval': 'DeepEval',
    'detoxify': 'Detoxify',
    'trulens.core': 'TruLens Core',
    'trulens.feedback': 'TruLens Feedback',
    'pydantic': 'Pydantic',
    'structlog': 'StructLog',
    'opentelemetry.trace': 'OpenTelemetry',
    'pytest': 'PyTest',
    'guardrails': 'Guardrails AI'
}

for module, name in dependencies.items():
    try:
        __import__(module)
        results[module] = True
        print(f"✅ {name}")
    except ImportError as e:
        results[module] = False
        print(f"❌ {name} - {e}")

print(f"DEPENDENCY_RESULTS:{results}")
"""
        
        results = self._run_test_script(test_script, "DEPENDENCY_RESULTS:")
        return results or {}
    
    def test_trulens_functionality(self) -> Dict[str, bool]:
        """Test TruLens specific functionality"""
        logger.info("📋 Testing TruLens Functionality...")
        
        test_script = """
import os
import sys
sys.path.insert(0, '.')

results = {}

try:
    from evaluation.trulens_config import TruLensConfig
    
    # Initialize TruLens config
    config = TruLensConfig()
    results['trulens_init'] = True
    print("✅ TruLens configuration initialized")
    
    # Test feedback functions
    feedback_functions = config.get_feedback_functions()
    results['feedback_functions'] = len(feedback_functions) > 0
    print(f"✅ Created {len(feedback_functions)} feedback functions")
    
    # Test database connection
    tru = config.get_tru_session()
    results['database'] = True
    print("✅ Database connection successful")
    
    # Test provider availability
    providers = config.get_available_providers()
    results['providers'] = len(providers) > 0
    print(f"✅ Available providers: {providers}")
    
except Exception as e:
    results['trulens_init'] = False
    results['feedback_functions'] = False
    results['database'] = False  
    results['providers'] = False
    print(f"❌ TruLens functionality test failed: {e}")

print(f"TRULENS_RESULTS:{results}")
"""
        
        results = self._run_test_script(test_script, "TRULENS_RESULTS:")
        return results or {}
    
    def test_api_endpoints(self) -> Dict[str, bool]:
        """Test API server and endpoints"""
        logger.info("📋 Testing API Server...")
        
        try:
            import signal
            import requests
            
            # Start server
            server_process = subprocess.Popen(
                [str(self.python_path), "orchestration/app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Wait for server to start
            time.sleep(4)
            
            results = {}
            
            # Test endpoints
            endpoints = {
                'health': 'http://localhost:8000/health',
                'docs': 'http://localhost:8000/docs',
                'openapi': 'http://localhost:8000/openapi.json'
            }
            
            for name, url in endpoints.items():
                try:
                    response = requests.get(url, timeout=5)
                    results[f'endpoint_{name}'] = response.status_code == 200
                    status = "✅" if results[f'endpoint_{name}'] else "❌"
                    logger.info(f"  {status} {name.title()} endpoint - {response.status_code}")
                except Exception as e:
                    results[f'endpoint_{name}'] = False
                    logger.warning(f"  ❌ {name.title()} endpoint failed: {e}")
            
            # Clean up server
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
            server_process.wait(timeout=5)
            
            return results
            
        except Exception as e:
            logger.warning(f"⚠️  API server test failed: {e}")
            return {'api_server': False}
    
    def test_configuration(self) -> Dict[str, bool]:
        """Test configuration files and environment"""
        logger.info("📋 Testing Configuration...")
        
        results = {}
        
        # Check .env file
        env_file = self.project_root / ".env"
        results['env_file'] = env_file.exists()
        
        if results['env_file']:
            logger.info("✅ .env file exists")
            
            # Check for API keys (without revealing them)
            with open(env_file) as f:
                env_content = f.read()
            
            results['openai_key'] = 'OPENAI_API_KEY=' in env_content and 'your-openai' not in env_content
            results['anthropic_key'] = 'ANTHROPIC_API_KEY=' in env_content and 'your-anthropic' not in env_content
            results['default_provider'] = 'DEFAULT_LLM_PROVIDER=' in env_content
            
            key_status = "✅" if results['openai_key'] else "⚠️ "
            logger.info(f"  {key_status} OpenAI API key configured")
            
            key_status = "✅" if results['anthropic_key'] else "⚠️ "
            logger.info(f"  {key_status} Anthropic API key configured")
            
            provider_status = "✅" if results['default_provider'] else "⚠️ "
            logger.info(f"  {provider_status} Default provider set")
        else:
            logger.warning("⚠️  .env file missing")
        
        # Check required directories
        required_dirs = ['orchestration', 'evaluation', 'guardrails', 'observability', 'prompts', 'datasets']
        results['project_structure'] = all((self.project_root / d).exists() for d in required_dirs)
        
        if results['project_structure']:
            logger.info("✅ Project structure complete")
        else:
            logger.warning("⚠️  Some project directories missing")
        
        return results
    
    def _run_test_script(self, script: str, result_marker: str) -> Dict[str, bool]:
        """Execute test script and parse results"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            test_file = f.name
        
        try:
            result = subprocess.run(
                [str(self.python_path), test_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse results from output
            for line in result.stdout.split('\n'):
                if line.startswith(result_marker):
                    import ast
                    results_str = line.replace(result_marker, '')
                    return ast.literal_eval(results_str)
            
            # If no results found, check stderr
            if result.stderr:
                logger.warning(f"Test script stderr: {result.stderr}")
            
            return {}
            
        except subprocess.TimeoutExpired:
            logger.warning("Test script timed out")
            return {}
        except Exception as e:
            logger.warning(f"Test script failed: {e}")
            return {}
        finally:
            os.unlink(test_file)
    
    def generate_verification_report(self, all_results: Dict[str, Dict[str, bool]]) -> None:
        """Generate comprehensive verification report"""
        logger.info("\n" + "="*60)
        logger.info("🔍 PROMPTFORGE VERIFICATION REPORT")
        logger.info("="*60)
        
        # Calculate scores for each category
        category_scores = {}
        for category, results in all_results.items():
            if results:
                passed = sum(results.values())
                total = len(results)
                category_scores[category] = (passed, total)
            else:
                category_scores[category] = (0, 1)
        
        # Display results by category
        for category, (passed, total) in category_scores.items():
            percentage = (passed / total) * 100 if total > 0 else 0
            status = "✅" if percentage >= 80 else "⚠️ " if percentage >= 50 else "❌"
            logger.info(f"\n{status} {category.replace('_', ' ').title()}: {passed}/{total} ({percentage:.0f}%)")
            
            # Show individual test results
            if category in all_results and all_results[category]:
                for test, result in all_results[category].items():
                    test_status = "✅" if result else "❌"
                    test_name = test.replace('_', ' ').title()
                    logger.info(f"    {test_status} {test_name}")
        
        # Overall score
        total_passed = sum(score[0] for score in category_scores.values())
        total_tests = sum(score[1] for score in category_scores.values())
        overall_percentage = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        
        logger.info(f"\n📊 Overall Score: {total_passed}/{total_tests} ({overall_percentage:.0f}%)")
        
        # Status and recommendations
        if overall_percentage >= 90:
            logger.info("🎉 EXCELLENT - PromptForge is fully operational!")
            logger.info("🚀 Ready for production use")
        elif overall_percentage >= 80:
            logger.info("✅ GOOD - PromptForge is mostly functional")
            logger.info("🔧 Address remaining issues for optimal performance")
        elif overall_percentage >= 60:
            logger.info("⚠️  FAIR - PromptForge has some issues")
            logger.info("🛠️  Review failed tests and missing configurations")
        else:
            logger.info("❌ POOR - PromptForge needs significant attention")
            logger.info("🆘 Run setup_promptforge.py to fix installation issues")
        
        # Specific recommendations
        logger.info("\n💡 Recommendations:")
        
        config_results = all_results.get('configuration', {})
        if not config_results.get('openai_key', True) or not config_results.get('anthropic_key', True):
            logger.info("  🔑 Add API keys to .env file for full LLM functionality")
        
        if not config_results.get('env_file', True):
            logger.info("  📄 Create .env configuration file")
        
        dependency_results = all_results.get('dependencies', {})
        missing_deps = [dep for dep, status in dependency_results.items() if not status]
        if missing_deps:
            logger.info(f"  📦 Install missing dependencies: {', '.join(missing_deps[:3])}")
        
        trulens_results = all_results.get('trulens_functionality', {})
        if not trulens_results.get('trulens_init', True):
            logger.info("  🔧 Fix TruLens configuration issues")
        
        logger.info("\n🚀 Quick Start:")
        logger.info("  • Start API: python orchestration/app.py")
        logger.info("  • API docs: http://localhost:8000/docs")
        logger.info("  • Run tests: python -m pytest evals/")
        
        # Save detailed report
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': all_results,
            'scores': category_scores,
            'overall_score': f"{total_passed}/{total_tests}",
            'percentage': overall_percentage
        }
        
        report_file = self.project_root / "verification_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"\n📄 Detailed report saved: {report_file}")
    
    def run_verification(self) -> bool:
        """Run complete verification process"""
        logger.info("🔍 PromptForge Installation Verification")
        logger.info("="*50)
        
        if not self.check_environment():
            return False
        
        # Run all verification tests
        all_results = {
            'core_imports': self.test_core_imports(),
            'dependencies': self.test_dependencies(),
            'trulens_functionality': self.test_trulens_functionality(),
            'api_endpoints': self.test_api_endpoints(),
            'configuration': self.test_configuration()
        }
        
        # Generate report
        self.generate_verification_report(all_results)
        
        # Calculate success
        total_passed = sum(sum(results.values()) for results in all_results.values() if results)
        total_tests = sum(len(results) for results in all_results.values() if results)
        success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        
        return success_rate >= 80

def main():
    """Main entry point"""
    verifier = InstallationVerifier()
    success = verifier.run_verification()
    
    if success:
        logger.info("\n✅ Verification completed successfully!")
    else:
        logger.warning("\n⚠️  Verification completed with issues - see report above")

if __name__ == "__main__":
    main()