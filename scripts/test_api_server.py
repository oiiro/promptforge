#!/usr/bin/env python3
"""
Test API Server

Tests the FastAPI orchestration server endpoints.
"""

import asyncio
import sys
import httpx
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_api_server():
    """Test the FastAPI server endpoints."""
    print("🚀 Testing API Server...")
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        try:
            # Test health endpoint
            print("🏥 Testing health endpoint...")
            response = await client.get(f"{base_url}/health")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                health_data = response.json()
                print(f"   Service Status: {health_data.get('status', 'unknown')}")
                print(f"   Environment: {health_data.get('environment', 'unknown')}")
                print(f"   Version: {health_data.get('version', 'unknown')}")
                
                providers = health_data.get('providers', {})
                print(f"   Providers: {sum(providers.values())}/{len(providers)} available")
            
            # Test metrics endpoint
            print("\n📊 Testing metrics endpoint...")
            response = await client.get(f"{base_url}/metrics")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                metrics_data = response.json()
                print(f"   Total Requests: {metrics_data.get('total_requests', 0)}")
                print(f"   Uptime: {metrics_data.get('uptime_seconds', 0)}s")
            
            # Test capital endpoint (mock example)
            print("\n🏛️  Testing capital endpoint...")
            test_payload = {
                "country": "France",
                "provider": "mock"
            }
            
            response = await client.post(f"{base_url}/capital", json=test_payload)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                capital_data = response.json()
                print(f"   Capital: {capital_data.get('capital', 'unknown')}")
                print(f"   Confidence: {capital_data.get('confidence', 0)}")
            elif response.status_code == 422:
                print("   ⚠️  Validation error (expected without mock provider)")
            
            # Test docs endpoint
            print("\n📚 Testing API documentation...")
            response = await client.get(f"{base_url}/docs")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print("   ✅ API documentation accessible")
            
        except httpx.ConnectError:
            print("❌ Server not running. Start it with:")
            print("   ./venv/bin/python3 orchestration/app.py")
        except Exception as e:
            print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_api_server())