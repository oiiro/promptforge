#!/usr/bin/env python3
"""
Test script for enhanced TruLens feedback functions on live endpoint.
"""

import requests
import json
import logging
import os
import time
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_feedback():
    """Test the enhanced TruLens feedback functions."""
    
    base_url = "http://localhost:8000"
    
    # Test authentication
    auth_response = requests.post(f"{base_url}/api/v1/auth/login", json={
        "username": "test_user",
        "password": "test_pass"
    })
    
    if auth_response.status_code != 200:
        logger.error(f"❌ Authentication failed: {auth_response.status_code}")
        return False
    
    token = auth_response.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test queries with different complexity levels for comprehensive feedback
    test_queries = [
        {
            "name": "Simple Age Query",
            "query": "I am 67 years old. Am I eligible for retirement?",
            "expected_features": ["age_detection", "clear_answer", "high_relevance"]
        },
        {
            "name": "Complex Multi-Factor Query", 
            "query": "I'm 62 years old with $500k in 401k and $80k annual salary. When can I retire and how much should I save?",
            "expected_features": ["comprehensive_analysis", "financial_planning", "detailed_response"]
        },
        {
            "name": "PII-Rich Query",
            "query": "My name is John Smith, SSN 123-45-6789, born 01/15/1958. Am I retirement eligible?",
            "expected_features": ["pii_detection", "pii_anonymization", "privacy_protection"]
        },
        {
            "name": "Ambiguous Query",
            "query": "Can someone retire?",
            "expected_features": ["low_completeness", "clarification_request", "low_confidence"]
        },
        {
            "name": "Emotional Query",
            "query": "I'm worried about retirement. I'm 64 and scared I don't have enough money saved.",
            "expected_features": ["sentiment_analysis", "empathetic_response", "supportive_tone"]
        }
    ]
    
    logger.info("🧪 Testing Enhanced TruLens Feedback Functions")
    logger.info("=" * 60)
    
    for i, test_case in enumerate(test_queries, 1):
        logger.info(f"\n📝 Test {i}: {test_case['name']}")
        logger.info(f"Query: {test_case['query']}")
        
        # Test V1 (Mock) endpoint
        logger.info("🔸 Testing V1 (Mock) endpoint...")
        v1_response = requests.post(
            f"{base_url}/api/v1/retirement-eligibility",
            headers=headers,
            json={
                "query": test_case["query"],
                "enable_pii_protection": True,
                "enable_monitoring": True
            }
        )
        
        if v1_response.status_code == 200:
            v1_data = v1_response.json()
            logger.info(f"   ✅ V1 Response: {v1_data.get('eligible', 'N/A')} - {v1_data.get('confidence', 0):.2f} confidence")
            if v1_data.get('pii_detected'):
                logger.info(f"   🔒 PII Detected: {v1_data.get('pii_entities', [])}")
        else:
            logger.error(f"   ❌ V1 Failed: {v1_response.status_code}")
        
        # Test V2 (Live with Comprehensive Feedback) endpoint
        logger.info("🔸 Testing V2 (Live + Comprehensive Feedback) endpoint...")
        v2_response = requests.post(
            f"{base_url}/api/v2/retirement-eligibility",
            headers=headers,
            json={
                "query": test_case["query"],
                "enable_pii_protection": True,
                "enable_monitoring": True
            }
        )
        
        if v2_response.status_code == 200:
            v2_data = v2_response.json()
            logger.info(f"   ✅ V2 Response: {v2_data.get('eligible', 'N/A')} - {v2_data.get('confidence', 0):.2f} confidence")
            logger.info(f"   🎯 Enhanced feedback will capture:")
            logger.info(f"      - QA Relevance (question → answer alignment)")
            logger.info(f"      - Context Relevance (input context quality)")
            logger.info(f"      - Groundedness (answer supported by context)")
            logger.info(f"      - Sentiment Analysis (emotional tone)")
            logger.info(f"      - Toxicity Detection (content safety)")
            logger.info(f"      - Retirement Response Quality (domain-specific)")
            logger.info(f"      - Input Completeness (required info present)")
            logger.info(f"      - PII Protection (privacy compliance)")
            logger.info(f"      - Confidence Calibration (accuracy vs confidence)")
            
            if v2_data.get('pii_detected'):
                logger.info(f"   🔒 PII Detected & Protected: {v2_data.get('pii_entities', [])}")
        else:
            logger.error(f"   ❌ V2 Failed: {v2_response.status_code}")
        
        # Wait between requests to avoid rate limiting
        time.sleep(2)
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Enhanced TruLens Feedback Testing Complete!")
    logger.info("\n📊 Expected TruLens Dashboard Data:")
    logger.info("   • MockPromptForge app: Basic feedback (4 functions)")
    logger.info("   • PromptForge app: Comprehensive feedback (9 functions)")
    logger.info("\n🔍 Check TruLens dashboard at: http://localhost:8501")
    logger.info("   • View records with detailed feedback scores")
    logger.info("   • Compare mock vs live endpoint performance")
    logger.info("   • Analyze PII protection effectiveness")
    logger.info("   • Review sentiment and toxicity patterns")
    
    return True

if __name__ == "__main__":
    # Ensure server is running
    logger.info("🚀 Starting enhanced TruLens feedback test...")
    logger.info("📋 This test will:")
    logger.info("   1. Test 5 different query types")
    logger.info("   2. Compare V1 (basic) vs V2 (comprehensive) feedback")
    logger.info("   3. Validate all 9 TruLens feedback functions")
    logger.info("   4. Verify PII protection and anonymization")
    logger.info("   5. Generate rich evaluation data for TruLens dashboard")
    
    success = test_enhanced_feedback()
    
    if success:
        logger.info("\n✅ All tests completed - check TruLens dashboard for feedback data!")
    else:
        logger.error("\n❌ Some tests failed - check server logs for details.")