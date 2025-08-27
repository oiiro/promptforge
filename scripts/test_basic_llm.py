#!/usr/bin/env python3
"""
Test Basic LLM Client

Tests the basic LLM client functionality without PII protection.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.llm_client import LLMClient


async def test_basic():
    """Test basic LLM client functionality."""
    print("🧪 Testing Basic LLM Client...")
    
    client = LLMClient()
    response = await client.generate_async('What is the capital of Japan?')
    
    print(f"✅ Response: {response['content']}")
    print(f"📊 Provider: {response.get('provider', 'unknown')}")
    print(f"⏱️  Processing completed successfully")


if __name__ == "__main__":
    asyncio.run(test_basic())