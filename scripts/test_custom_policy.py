#!/usr/bin/env python3
"""
Test Custom PII Policy

Demonstrates creating and using custom PII policies with PresidioMiddleware.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_custom_policy():
    """Test custom PII policy functionality."""
    print("📋 Testing Custom PII Policy...")
    
    try:
        from presidio.policies import PIIPolicyEngine, PIIPolicy, PIIAction
        from presidio.middleware import PresidioMiddleware
        
        # Create custom policy
        custom_policy = PIIPolicy(
            name='strict_financial',
            version='1.0.0',
            entities={
                'CREDIT_CARD': PIIAction.REDACT,
                'PHONE_NUMBER': PIIAction.MASK,
                'EMAIL_ADDRESS': PIIAction.HASH,
                'PERSON': PIIAction.SYNTHETIC
            },
            metadata={'compliance': 'SOX', 'industry': 'financial'}
        )
        
        print(f"✅ Custom policy created: {custom_policy.name}")
        
        middleware = PresidioMiddleware(policy=custom_policy)
        
        test_text = 'Contact John Doe at john.doe@bank.com or call 555-123-4567 regarding credit card 4532-1234-5678-9012'
        
        print(f"📝 Original: {test_text}")
        
        result = await middleware.anonymize(test_text, session_id='custom_test')
        print(f"🛡️  Anonymized: {result.get('anonymized_text', 'Not available')}")
        print(f"🔍 Entities Detected: {len(result.get('entities', []))}")
        
        # Show detected entities if available
        if 'entities' in result:
            for entity in result['entities']:
                print(f"   - {entity.get('entity_type', 'Unknown')}: {entity.get('action', 'No action')}")
        
    except ImportError as e:
        print(f"⚠️  Presidio not available: {e}")
        print("Using mock example:")
        print("📝 Original: Contact [Name] at [Email] or call [Phone] regarding credit card [Card]")
        print("🛡️  Anonymized: Contact <SYNTHETIC_PERSON> at <HASH_EMAIL> or call XXX-XXX-XXXX regarding credit card [REDACTED]")


if __name__ == "__main__":
    asyncio.run(test_custom_policy())