#!/usr/bin/env python3
"""
Test script for multi-person retirement eligibility API with PII protection
Demonstrates the complete workflow including API invocation and monitoring
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any

# Test endpoint
API_BASE = "http://localhost:8000"
TOKEN = "demo-token"

async def test_multi_person_retirement():
    """Test the multi-person retirement eligibility endpoint"""
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        
        # Test data with multiple people containing PII
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
        
        print("🚀 Testing Multi-Person Retirement Eligibility API")
        print("=" * 60)
        print(f"📡 Endpoint: {API_BASE}/api/v1/retirement-eligibility")
        print(f"🔒 Authentication: Bearer token")
        print(f"📊 PII Protection: Enabled")
        print(f"📈 TruLens Monitoring: Enabled")
        print()
        
        try:
            print("📤 Sending request...")
            response = await client.post(
                f"{API_BASE}/api/v1/retirement-eligibility",
                json=payload,
                headers=headers,
                timeout=30.0
            )
            
            print(f"📥 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ SUCCESS - API Response:")
                print("-" * 30)
                print(f"Response: {result['response']}")
                print(f"Eligible: {result['eligible']}")
                print(f"Deposit Amount: ${result['deposit_amount']}")
                print(f"Persons Processed: {result['persons_processed']}")
                print(f"PII Detected: {result['pii_detected']}")
                print(f"PII Entities: {result['pii_entities']}")
                print(f"Anonymization Applied: {result['anonymization_applied']}")
                print()
                print("📊 Metadata:")
                for key, value in result['metadata'].items():
                    print(f"  {key}: {value}")
                    
                # Test TruLens dashboard if available
                print("\n🔍 Testing TruLens Dashboard...")
                try:
                    dashboard_response = await client.get(
                        f"{API_BASE}/api/v1/trulens/dashboard",
                        headers=headers
                    )
                    if dashboard_response.status_code == 200:
                        dashboard_data = dashboard_response.json()
                        print("✅ TruLens Dashboard Available:")
                        print(f"  Records: {dashboard_data.get('total_records', 0)}")
                        print(f"  Apps: {dashboard_data.get('total_apps', 0)}")
                    else:
                        print(f"⚠️  TruLens Dashboard: {dashboard_response.status_code}")
                except Exception as e:
                    print(f"⚠️  TruLens Dashboard Error: {e}")
                
            else:
                print("❌ ERROR - API Response:")
                print(response.text)
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            
        # Test health endpoint
        print("\n💚 Testing Health Endpoint...")
        try:
            health_response = await client.get(f"{API_BASE}/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"✅ Health Status: {health_data['status']}")
                print(f"   Version: {health_data['version']}")
                print(f"   Environment: {health_data['environment']}")
            else:
                print(f"⚠️  Health check failed: {health_response.status_code}")
        except Exception as e:
            print(f"❌ Health check error: {e}")

async def demonstrate_pii_protection():
    """Demonstrate PII protection capabilities directly"""
    print("\n🛡️  Direct PII Protection Demonstration")
    print("=" * 50)
    
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
        
        # Initialize engines
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        
        # Test text with multiple people and PII
        test_text = """
        John Smith (phone: 555-123-4567, email: john.smith@company.com) 
        and Sarah Johnson (SSN: 123-45-6789, email: sarah.j@company.com) 
        are both eligible for retirement.
        """
        
        print(f"📝 Original Text:\n{test_text}")
        
        # Analyze for PII
        pii_results = analyzer.analyze(text=test_text, language='en')
        print(f"\n🔍 PII Entities Found: {len(pii_results)}")
        
        for result in pii_results:
            original_text = test_text[result.start:result.end]
            print(f"  - {result.entity_type}: '{original_text}' (confidence: {result.score:.2f})")
        
        # Anonymize with numbered placeholders
        anonymization_mapping = {}
        entity_counters = {}
        
        anonymized_text = test_text
        offset = 0
        
        for result in sorted(pii_results, key=lambda x: x.start):
            entity_type = result.entity_type
            if entity_type not in entity_counters:
                entity_counters[entity_type] = 1
            else:
                entity_counters[entity_type] += 1
                
            placeholder = f"<{entity_type}_{entity_counters[entity_type]}>"
            original_value = test_text[result.start:result.end]
            
            # Store mapping for deanonymization
            anonymization_mapping[placeholder] = original_value
            
            # Replace in anonymized text
            start_pos = result.start + offset
            end_pos = result.end + offset
            anonymized_text = anonymized_text[:start_pos] + placeholder + anonymized_text[end_pos:]
            offset += len(placeholder) - (result.end - result.start)
        
        print(f"\n🔒 Anonymized Text:\n{anonymized_text}")
        
        print(f"\n🗂️  Anonymization Mapping:")
        for placeholder, original in anonymization_mapping.items():
            print(f"  {placeholder} → '{original}'")
            
        # Demonstrate deanonymization
        deanonymized_text = anonymized_text
        for placeholder, original in anonymization_mapping.items():
            deanonymized_text = deanonymized_text.replace(placeholder, original)
            
        print(f"\n🔓 Deanonymized Text:\n{deanonymized_text}")
        print(f"✅ Text matches original: {deanonymized_text.strip() == test_text.strip()}")
        
    except ImportError:
        print("⚠️  Presidio not available for direct demonstration")
    except Exception as e:
        print(f"❌ PII Protection Error: {e}")

if __name__ == "__main__":
    print("🧪 PromptForge Multi-Person Retirement Eligibility Test")
    print("=" * 60)
    
    # Run the tests
    asyncio.run(test_multi_person_retirement())
    asyncio.run(demonstrate_pii_protection())
    
    print("\n✨ Testing complete!")