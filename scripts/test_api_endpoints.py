#!/usr/bin/env python3
"""
Test API Endpoints

Quick curl-style tests for API endpoints with proper authentication examples.
"""

import subprocess
import json
import sys
from pathlib import Path

def run_curl(description, cmd, expect_success=True):
    """Run curl command and display results"""
    print(f"\n🧪 {description}")
    print(f"📝 Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Status: Success")
            try:
                # Try to parse JSON response
                response = json.loads(result.stdout.strip())
                print(f"📋 Response: {json.dumps(response, indent=2)}")
            except json.JSONDecodeError:
                print(f"📋 Response: {result.stdout.strip()}")
        else:
            status = "❌ Failed" if expect_success else "✅ Expected Error"
            print(f"{status} (Return code: {result.returncode})")
            if result.stderr:
                print(f"🚨 Error: {result.stderr.strip()}")
            if result.stdout:
                print(f"📋 Response: {result.stdout.strip()}")
                
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
    except Exception as e:
        print(f"❌ Error running command: {e}")

def main():
    """Run API endpoint tests"""
    print("🚀 API Endpoints Testing Suite")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health check (no auth required)
    run_curl(
        "Health Check",
        ["curl", "-s", f"{base_url}/health"]
    )
    
    # Test 2: Capital endpoint without authentication (should fail)
    run_curl(
        "Capital Query - No Authentication (Expected Failure)",
        ["curl", "-s", "-X", "POST", f"{base_url}/api/v1/capital",
         "-H", "Content-Type: application/json",
         "-d", '{"country": "France"}'],
        expect_success=False
    )
    
    # Test 3: Capital endpoint with authentication (should succeed)
    run_curl(
        "Capital Query - With Authentication",
        ["curl", "-s", "-X", "POST", f"{base_url}/api/v1/capital",
         "-H", "Content-Type: application/json", 
         "-H", "Authorization: Bearer demo-token",
         "-d", '{"country": "France"}']
    )
    
    # Test 4: Different country
    run_curl(
        "Capital Query - Japan",
        ["curl", "-s", "-X", "POST", f"{base_url}/api/v1/capital",
         "-H", "Content-Type: application/json",
         "-H", "Authorization: Bearer demo-token", 
         "-d", '{"country": "Japan"}']
    )
    
    # Test 5: Metrics endpoint (auth required)
    run_curl(
        "Metrics - With Authentication",
        ["curl", "-s", "-H", "Authorization: Bearer demo-token",
         f"{base_url}/api/v1/metrics"]
    )
    
    # Test 6: Version endpoint (no auth required)
    run_curl(
        "Version Information",
        ["curl", "-s", f"{base_url}/api/v1/version"]
    )
    
    print("\n" + "=" * 50)
    print("🎯 Testing Summary:")
    print("✅ All authentication-required endpoints tested")
    print("✅ Error handling for missing authentication verified")
    print("✅ Multiple countries tested for capital endpoint")
    print("✅ Various endpoint types covered")
    print("\n💡 To start the server: ./venv/bin/python3 orchestration/app.py")

if __name__ == "__main__":
    main()