#!/usr/bin/env python3
"""
Run All Tests

Comprehensive test runner that executes all available test scripts
in sequence with proper error handling and reporting.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path


def run_script(script_path):
    """Run a Python script and capture its output."""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Test timed out after 60 seconds',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }


async def main():
    """Run all test scripts in sequence."""
    print("🧪 PromptForge Comprehensive Test Suite")
    print("=" * 50)
    
    scripts_dir = Path(__file__).parent
    
    # Define test scripts in execution order
    test_scripts = [
        ('Health Check', 'health_check.py'),
        ('Basic LLM Client', 'test_basic_llm.py'),
        ('PII Protection', 'test_pii_protection.py'),
        ('Service Class', 'test_service_class.py'),
        ('Custom Policy', 'test_custom_policy.py'),
        ('Presidio Capital Finder', 'test_presidio_capital_finder.py'),
        ('API Server', 'test_api_server.py'),
        ('API Endpoints', 'test_api_endpoints.py'),
        ('Performance Benchmark', 'performance_benchmark.py')
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_name, script_name in test_scripts:
        script_path = scripts_dir / script_name
        
        if not script_path.exists():
            print(f"\n❌ {test_name}: Script not found ({script_name})")
            results.append({
                'name': test_name,
                'success': False,
                'error': 'Script not found'
            })
            continue
        
        print(f"\n🚀 Running {test_name}...")
        print("-" * 30)
        
        start_time = time.time()
        result = run_script(script_path)
        duration = time.time() - start_time
        
        if result['success']:
            print(f"✅ {test_name}: PASSED ({duration:.2f}s)")
            if result['stdout']:
                # Show key output lines
                lines = result['stdout'].strip().split('\n')
                for line in lines[-5:]:  # Show last 5 lines
                    if line.strip():
                        print(f"   {line}")
        else:
            print(f"❌ {test_name}: FAILED ({duration:.2f}s)")
            if result['stderr']:
                print(f"   Error: {result['stderr'].strip()}")
        
        results.append({
            'name': test_name,
            'success': result['success'],
            'duration': duration,
            'stdout': result['stdout'],
            'stderr': result['stderr']
        })
    
    # Summary report
    total_duration = time.time() - total_start_time
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Total time: {total_duration:.2f}s")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    # Detailed failure report
    failures = [r for r in results if not r['success']]
    if failures:
        print(f"\n❌ FAILED TESTS:")
        print("-" * 20)
        for failure in failures:
            print(f"   {failure['name']}: {failure.get('error', 'Unknown error')}")
    else:
        print(f"\n🎉 ALL TESTS PASSED!")
    
    # Overall status
    if passed == total:
        print("✅ System is fully operational!")
        return 0
    elif passed >= total * 0.8:
        print("⚠️  System is mostly operational with minor issues")
        return 1
    else:
        print("❌ System has significant issues requiring attention")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())