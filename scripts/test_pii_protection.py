#!/usr/bin/env python3
"""
Test PII Protection

Tests the PIIAwareLLMClient with PII detection and anonymization.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.llm_client import PIIAwareLLMClient


async def test_pii():
    """Test PII protection functionality."""
    print("🛡️  Testing PII Protection...")
    
    client = PIIAwareLLMClient()
    
    # Test query with PII
    query = 'My name is Alice Johnson and my SSN is 123-45-6789. What is the capital of France?'
    print(f"📝 Query: {query}")
    
    response = await client.generate_with_pii_protection(
        query,
        session_id='test_session',
        restore_pii=True
    )
    
    # PIIAware client returns structured response - extract the nested response content
    inner_response = response.get('response', {})
    if 'content' in inner_response:
        print(f"💬 PII Protected Response: {inner_response['content']}")
    else:
        # MockProvider returns structured data like {"capital": "Paris", "confidence": 1.0}
        # Or guardrails may block request: "Request blocked by security guardrails"
        print(f"💬 PII Protected Response: {inner_response}")
    
    print(f"🔍 PII Metadata: {response.get('pii_metadata', {})}")
    print(f"📊 Status: {response.get('status', 'unknown')}")
    print("ℹ️  Note: SSN in query may trigger guardrails blocking for security")


if __name__ == "__main__":
    asyncio.run(test_pii())