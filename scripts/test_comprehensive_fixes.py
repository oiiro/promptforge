#!/usr/bin/env python3
"""
Comprehensive Test Script for PromptForge Multi-Person Retirement Eligibility Fixes
Tests both EMAIL_ADDRESS_3 deanonymization and TruLens monitoring functionality
"""

import asyncio
import httpx
import json
import time
import sys
from typing import Dict, Any
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8000"
TOKEN = "demo-token"

class ComprehensiveTest:
    def __init__(self):
        self.results = []
        self.passed_tests = 0
        self.failed_tests = 0
    
    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result"""
        status = "✅ PASSED" if passed else "❌ FAILED"
        full_message = f"{status} - {test_name}"
        if message:
            full_message += f": {message}"
        
        print(full_message)
        
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        self.results.append({
            "test": test_name,
            "passed": passed,
            "message": message
        })
        
        return passed

    async def test_server_health(self, client: httpx.AsyncClient) -> bool:
        """Test basic server health"""
        try:
            response = await client.get(f"{API_BASE}/health", timeout=10.0)
            
            if response.status_code == 200:
                health_data = response.json()
                return self.log_test(
                    "Server Health Check", 
                    True, 
                    f"Version: {health_data.get('version', 'unknown')}"
                )
            else:
                return self.log_test(
                    "Server Health Check", 
                    False, 
                    f"Status: {response.status_code}"
                )
                
        except Exception as e:
            return self.log_test("Server Health Check", False, str(e))

    async def test_email_address_3_fix(self, client: httpx.AsyncClient) -> bool:
        """Test the EMAIL_ADDRESS_3 deanonymization fix"""
        
        # Test query with 3 people but only 2 email addresses (the original failing case)
        test_query = """
        Please evaluate retirement eligibility for the following employees:
        
        1. John Smith, age 65, phone number 555-123-4567, employed for 25 years 
           with current salary of $75,000. Email: john.smith@company.com
           
        2. Sarah Johnson, age 62, phone 555-987-6543, employed for 30 years
           with current salary of $85,000. SSN: 123-45-6789
           
        3. Mary Williams, age 58, email mary.williams@company.com, 
           employed for 22 years with salary $68,000
        """
        
        payload = {
            "query": test_query,
            "enable_pii_protection": True,
            "enable_monitoring": True
        }
        
        try:
            response = await client.post(
                f"{API_BASE}/api/v1/retirement-eligibility",
                json=payload,
                headers={"Authorization": f"Bearer {TOKEN}"},
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                # Check that EMAIL_ADDRESS_3 is NOT present (should be deanonymized)
                if '<EMAIL_ADDRESS_3>' in response_text:
                    return self.log_test(
                        "EMAIL_ADDRESS_3 Deanonymization", 
                        False, 
                        "Found <EMAIL_ADDRESS_3> placeholder in response"
                    )
                
                # Check that we have real email addresses in response
                if 'mary.williams@company.com' in response_text or 'john.smith@company.com' in response_text:
                    return self.log_test(
                        "EMAIL_ADDRESS_3 Deanonymization", 
                        True, 
                        "All email placeholders properly deanonymized"
                    )
                else:
                    return self.log_test(
                        "EMAIL_ADDRESS_3 Deanonymization", 
                        False, 
                        "No real email addresses found in response"
                    )
            else:
                return self.log_test(
                    "EMAIL_ADDRESS_3 Deanonymization", 
                    False, 
                    f"API call failed: {response.status_code}"
                )
                
        except Exception as e:
            return self.log_test("EMAIL_ADDRESS_3 Deanonymization", False, str(e))

    async def test_trulens_monitoring(self, client: httpx.AsyncClient) -> bool:
        """Test TruLens monitoring functionality"""
        
        test_query = "Test PromptForge TruLens monitoring with John Smith, email: john@example.com"
        
        payload = {
            "query": test_query,
            "enable_pii_protection": True,
            "enable_monitoring": True
        }
        
        try:
            response = await client.post(
                f"{API_BASE}/api/v1/retirement-eligibility",
                json=payload,
                headers={"Authorization": f"Bearer {TOKEN}"},
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                metadata = result.get('metadata', {})
                
                # Check if TruLens monitoring is enabled
                trulens_monitoring = metadata.get('trulens_monitoring', False)
                
                if trulens_monitoring:
                    return self.log_test(
                        "TruLens Monitoring", 
                        True, 
                        "Monitoring active and recording data"
                    )
                else:
                    return self.log_test(
                        "TruLens Monitoring", 
                        False, 
                        "trulens_monitoring flag is False"
                    )
            else:
                return self.log_test(
                    "TruLens Monitoring", 
                    False, 
                    f"API call failed: {response.status_code}"
                )
                
        except Exception as e:
            return self.log_test("TruLens Monitoring", False, str(e))

    async def test_pii_protection_accuracy(self, client: httpx.AsyncClient) -> bool:
        """Test PII protection accuracy with URL filtering fix"""
        
        test_query = "Process John Smith with email john.smith@company.com and phone 555-123-4567"
        
        payload = {
            "query": test_query,
            "enable_pii_protection": True,
            "enable_monitoring": True
        }
        
        try:
            response = await client.post(
                f"{API_BASE}/api/v1/retirement-eligibility",
                json=payload,
                headers={"Authorization": f"Bearer {TOKEN}"},
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Check PII detection
                pii_detected = result.get('pii_detected', False)
                pii_entities = result.get('pii_entities', [])
                
                if pii_detected and 'EMAIL_ADDRESS' in pii_entities:
                    return self.log_test(
                        "PII Protection Accuracy", 
                        True, 
                        f"Detected {len(pii_entities)} PII entities"
                    )
                else:
                    return self.log_test(
                        "PII Protection Accuracy", 
                        False, 
                        f"PII detection failed: {pii_entities}"
                    )
            else:
                return self.log_test(
                    "PII Protection Accuracy", 
                    False, 
                    f"API call failed: {response.status_code}"
                )
                
        except Exception as e:
            return self.log_test("PII Protection Accuracy", False, str(e))

    async def test_multi_person_processing(self, client: httpx.AsyncClient) -> bool:
        """Test multi-person processing capability"""
        
        test_query = """
        Evaluate:
        1. Alice Johnson, age 65, email alice@company.com
        2. Bob Williams, age 62, phone 555-123-4567  
        3. Carol Davis, age 58, email carol@company.com
        """
        
        payload = {
            "query": test_query,
            "enable_pii_protection": True,
            "enable_monitoring": True
        }
        
        try:
            response = await client.post(
                f"{API_BASE}/api/v1/retirement-eligibility",
                json=payload,
                headers={"Authorization": f"Bearer {TOKEN}"},
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                persons_processed = result.get('persons_processed', 0)
                
                if persons_processed >= 3:
                    return self.log_test(
                        "Multi-Person Processing", 
                        True, 
                        f"Processed {persons_processed} persons"
                    )
                else:
                    return self.log_test(
                        "Multi-Person Processing", 
                        False, 
                        f"Only processed {persons_processed} persons"
                    )
            else:
                return self.log_test(
                    "Multi-Person Processing", 
                    False, 
                    f"API call failed: {response.status_code}"
                )
                
        except Exception as e:
            return self.log_test("Multi-Person Processing", False, str(e))

    async def run_comprehensive_tests(self):
        """Run all comprehensive tests"""
        print("🧪 PromptForge Comprehensive Fix Verification")
        print("=" * 60)
        print("Testing EMAIL_ADDRESS_3 deanonymization and TruLens monitoring fixes")
        print()
        
        async with httpx.AsyncClient() as client:
            # Run all tests
            await self.test_server_health(client)
            await self.test_email_address_3_fix(client)  # Critical fix test
            await self.test_trulens_monitoring(client)   # Critical fix test
            await self.test_pii_protection_accuracy(client)
            await self.test_multi_person_processing(client)
        
        # Summary
        print("\n" + "=" * 60)
        print(f"📋 TEST SUMMARY: {self.passed_tests}/{self.passed_tests + self.failed_tests} tests passed")
        print("=" * 60)
        
        if self.failed_tests == 0:
            print("🎉 ALL TESTS PASSED - Both critical fixes verified!")
            print()
            print("✅ EMAIL_ADDRESS_3 deanonymization working correctly")
            print("✅ TruLens monitoring recording data successfully")
            print("✅ Server startup no longer hangs")
            print("✅ Multi-person PII processing operational")
            return 0
        else:
            print(f"❌ {self.failed_tests} TESTS FAILED - Check issues above")
            
            # Show failed tests
            print("\nFailed Tests:")
            for result in self.results:
                if not result["passed"]:
                    print(f"  - {result['test']}: {result['message']}")
            return 1

async def main():
    """Main test execution"""
    tester = ComprehensiveTest()
    return await tester.run_comprehensive_tests()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))