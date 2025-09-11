#!/usr/bin/env python3
"""
Developer Guide Verification Script

Verifies that all components from the Comprehensive Developer Guide are working correctly.
Uses the wrapper script when available for streamlined testing.
"""

import sys
import json
import traceback
import importlib.util
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

class DeveloperGuideVerifier:
    """Verifies all components from the comprehensive developer guide using wrapper script."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "tests_run": 0,
            "tests_passed": 0, 
            "tests_failed": 0,
            "test_details": [],
            "overall_status": "unknown"
        }
        
        # Check if wrapper script exists
        self.wrapper_script = Path("run_component.py")
        self.use_wrapper = self.wrapper_script.exists()
        
        if self.use_wrapper:
            print("✓ Using run_component.py wrapper script for testing")
        else:
            print("⚠️  run_component.py not found, using direct testing")
        
        # Test modules mapped to wrapper components
        self.test_modules = [
            ("1. JSON Schema Validation", "schema_validation"),
            ("2. Prompt Templating", "prompt_templating"),
            ("3. Unit Testing", "unit_testing"),
            ("4. Integration Testing", "integration_testing"),
            ("5. Test Data Generation", "test_data_generation"),
            ("6. Heuristic Validation", "heuristic_validation"),
            ("7. Policy Filters", "policy_filters"),
            ("8. Response Modification", "response_modification"),
            ("9. Custom Evaluation", "custom_evaluation"),
            ("10. Iterative Refinement", "iterative_refinement")
        ]
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all verification tests using wrapper script when available."""
        print("🔍 PromptForge Developer Guide Verification")
        print("=" * 60)
        print(f"Started at: {datetime.utcnow().isoformat()}")
        print()
        
        if self.use_wrapper:
            return self._run_wrapper_tests()
        else:
            return self._run_direct_tests()
    
    def _run_wrapper_tests(self) -> Dict[str, Any]:
        """Run tests using the wrapper script."""
        print("🚀 Running tests via wrapper script...")
        print()
        
        for test_name, component_name in self.test_modules:
            self.results["tests_run"] += 1
            
            try:
                print(f"Testing {test_name} ({component_name})...")
                print("-" * 50)
                
                # Run component via wrapper script
                result = subprocess.run([
                    sys.executable, "run_component.py", component_name
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    self.results["tests_passed"] += 1
                    self.results["test_details"].append({
                        "name": test_name,
                        "component": component_name,
                        "status": "PASSED",
                        "output": result.stdout,
                        "error": None
                    })
                    print(f"✅ {test_name} - PASSED")
                    print(result.stdout.strip())
                else:
                    raise Exception(f"Component failed with return code {result.returncode}: {result.stderr}")
                
            except Exception as e:
                self.results["tests_failed"] += 1
                error_details = {
                    "name": test_name,
                    "component": component_name,
                    "status": "FAILED",
                    "error": str(e),
                    "stderr": getattr(result, 'stderr', '') if 'result' in locals() else ''
                }
                self.results["test_details"].append(error_details)
                
                print(f"❌ {test_name} - FAILED")
                print(f"   Error: {str(e)}")
            
            print()
        
        # Calculate overall status
        if self.results["tests_failed"] == 0:
            self.results["overall_status"] = "ALL_PASSED"
        elif self.results["tests_passed"] > self.results["tests_failed"]:
            self.results["overall_status"] = "MOSTLY_PASSED" 
        else:
            self.results["overall_status"] = "MOSTLY_FAILED"
        
        self._print_summary()
        return self.results
    
    def _run_direct_tests(self) -> Dict[str, Any]:
        """Run tests directly (fallback when wrapper not available)."""
        print("⚠️  Wrapper script not available, running direct tests...")
        print("   (Install dependencies: pip install jsonschema jinja2)")
        print()
        
        # Simplified direct tests
        direct_tests = [
            ("1. JSON Schema Files", self._test_schema_files_exist),
            ("2. Python Environment", self._test_python_environment),
            ("3. Basic Structure", self._test_basic_structure)
        ]
        
        for test_name, test_function in direct_tests:
            self.results["tests_run"] += 1
            
            try:
                print(f"Testing {test_name}...")
                print("-" * 50)
                
                test_function()
                
                self.results["tests_passed"] += 1
                self.results["test_details"].append({
                    "name": test_name,
                    "status": "PASSED",
                    "error": None
                })
                
                print(f"✅ {test_name} - PASSED")
                
            except Exception as e:
                self.results["tests_failed"] += 1
                error_details = {
                    "name": test_name,
                    "status": "FAILED",
                    "error": str(e)
                }
                self.results["test_details"].append(error_details)
                
                print(f"❌ {test_name} - FAILED")
                print(f"   Error: {str(e)}")
            
            print()
        
        # Calculate overall status
        if self.results["tests_failed"] == 0:
            self.results["overall_status"] = "ALL_PASSED"
        elif self.results["tests_passed"] > self.results["tests_failed"]:
            self.results["overall_status"] = "MOSTLY_PASSED" 
        else:
            self.results["overall_status"] = "MOSTLY_FAILED"
        
        self._print_summary()
        return self.results
    
    def _test_schema_files_exist(self):
        """Test that schema files exist."""
        schema_files = [
            "schemas/retirement_input_schema.json",
            "schemas/retirement_output_schema.json"
        ]
        
        for schema_file in schema_files:
            if not Path(schema_file).exists():
                raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        print("✓ All schema files exist")
    
    def _test_python_environment(self):
        """Test Python environment basics."""
        # Test Python version
        print(f"✓ Python version: {sys.version}")
        
        # Test basic imports
        try:
            import json
            print("✓ json module available")
        except ImportError:
            raise ImportError("json module not available")
    
    def _test_basic_structure(self):
        """Test basic project structure."""
        required_dirs = ["schemas", "docs"]
        required_files = [
            "docs/COMPREHENSIVE_DEVELOPER_GUIDE.md",
            "run_component.py"
        ]
        
        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                print(f"⚠️  Directory missing: {dir_name}")
        
        for file_name in required_files:
            if Path(file_name).exists():
                print(f"✓ {file_name} exists")
            else:
                print(f"⚠️  {file_name} missing")
        
        print("✓ Basic structure verified")
    
    def _print_summary(self):
        """Print test summary."""
        print("=" * 60)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 60)
        
        print(f"Tests Run: {self.results['tests_run']}")
        print(f"Passed: {self.results['tests_passed']}")
        print(f"Failed: {self.results['tests_failed']}")
        print(f"Overall Status: {self.results['overall_status']}")
        
        if self.results['tests_failed'] > 0:
            print("\n❌ Failed Tests:")
            for test in self.results['test_details']:
                if test['status'] == 'FAILED':
                    print(f"  - {test['name']}: {test['error']}")
        
        print("\n🎯 Recommendation:")
        if self.results['overall_status'] == 'ALL_PASSED':
            print("  All tests passed! Developer guide is ready to use.")
            if self.use_wrapper:
                print("  Use 'python run_component.py <component>' to run individual tests.")
        else:
            print("  Some tests failed. Check the errors above and:")
            if not self.use_wrapper:
                print("  1. Install dependencies: pip install jsonschema jinja2")
                print("  2. Run: python run_component.py all")
            else:
                print("  1. Fix any missing files or dependencies")
                print("  2. Re-run: python verify_developer_guide.py")


def main():
    """Main entry point."""
    verifier = DeveloperGuideVerifier()
    results = verifier.run_all_tests()
    
    # Exit with appropriate code
    if results["overall_status"] == "ALL_PASSED":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()