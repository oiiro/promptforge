#!/usr/bin/env python3
"""
Test Service Class

Tests the CapitalFinderService with PII protection integration.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.capital_finder_presidio import CapitalFinderService


async def test_service():
    """Test the CapitalFinderService class."""
    print("🏛️  Testing CapitalFinderService...")
    
    service = CapitalFinderService(enable_pii_protection=True)
    
    query = 'I live in Tokyo. What country am I in and what is its capital?'
    print(f"📝 Query: {query}")
    
    result = await service.find_capital_secure(
        query,
        include_context=True,
        restore_pii=True
    )
    
    # CapitalFinderService wraps PIIAware response - extract content properly
    response_content = result["response"].get("content", result["response"])
    print(f"💬 Service Response: {response_content}")
    print(f"⏱️  Processing Time: {result['processing_time_seconds']}s")
    print(f"🔒 PII Protection: {'✅ Enabled' if result['pii_protection_enabled'] else '❌ Disabled'}")
    print(f"🆔 Session ID: {result['session_id']}")
    
    # Cleanup
    cleanup_result = await service.cleanup_session()
    print(f"🧹 Session cleanup: {cleanup_result.get('status', 'unknown')}")


if __name__ == "__main__":
    asyncio.run(test_service())