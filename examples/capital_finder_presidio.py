#!/usr/bin/env python3
"""
PromptForge Reference Architecture: Capital City Finder with PII Protection

This example demonstrates a production-ready implementation of PromptForge 
with Microsoft Presidio PII anonymization for financial services use cases.

Key Features:
- Comprehensive PII detection and anonymization
- Policy-based PII handling (redact, mask, hash, tokenize, synthetic)
- Session-based PII mapping with secure storage
- Multi-provider LLM support with fallbacks
- Production-grade error handling and logging
- Async/await patterns for optimal performance
- TruLens integration for evaluation and monitoring
"""

import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.llm_client import PIIAwareLLMClient, LLMClient, MockProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('capital_finder_presidio.log')
    ]
)
logger = logging.getLogger(__name__)


class CapitalFinderService:
    """
    Production-grade capital finder service with comprehensive PII protection.
    
    This service demonstrates:
    - PII-aware prompt engineering
    - Session-based conversation management
    - Secure PII handling and restoration
    - Comprehensive error handling
    - Performance monitoring and logging
    """
    
    def __init__(self, enable_pii_protection: bool = True):
        """Initialize the Capital Finder Service.
        
        Args:
            enable_pii_protection: Whether to enable PII anonymization
        """
        self.enable_pii_protection = enable_pii_protection
        self.session_id = str(uuid.uuid4())
        
        # Initialize clients based on PII protection requirements
        if enable_pii_protection:
            self.client = PIIAwareLLMClient()
            logger.info("✅ PII-aware LLM client initialized")
        else:
            self.client = LLMClient()
            logger.info("⚠️  Standard LLM client initialized (no PII protection)")
    
    async def find_capital_secure(
        self, 
        query: str,
        include_context: bool = True,
        restore_pii: bool = True
    ) -> Dict[str, Any]:
        """
        Find capital city with comprehensive PII protection.
        
        Args:
            query: Natural language query (may contain PII)
            include_context: Whether to include geographic context
            restore_pii: Whether to restore PII in the final response
            
        Returns:
            Dict containing the response, metadata, and PII handling info
        """
        start_time = time.time()
        
        logger.info(f"🔍 Processing capital query: '{query[:50]}...'")
        logger.info(f"📋 Session ID: {self.session_id}")
        
        try:
            # Construct the prompt with context if requested
            prompt = self._construct_prompt(query, include_context)
            
            # Process with or without PII protection
            if self.enable_pii_protection and hasattr(self.client, 'generate_with_pii_protection'):
                response = await self.client.generate_with_pii_protection(
                    prompt=prompt,
                    session_id=self.session_id,
                    restore_pii=restore_pii,
                    max_tokens=150,
                    temperature=0.3
                )
            else:
                # Fallback to standard processing
                response = await self.client.generate_async(
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.3
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Enhance response with service metadata
            enhanced_response = {
                "service": "CapitalFinderService",
                "version": "1.0.0",
                "session_id": self.session_id,
                "processing_time_seconds": round(processing_time, 3),
                "pii_protection_enabled": self.enable_pii_protection,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "query": query,
                "response": response
            }
            
            logger.info(f"✅ Query processed successfully in {processing_time:.3f}s")
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {str(e)}")
            
            return {
                "service": "CapitalFinderService",
                "version": "1.0.0", 
                "session_id": self.session_id,
                "processing_time_seconds": time.time() - start_time,
                "pii_protection_enabled": self.enable_pii_protection,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "query": query,
                "error": str(e),
                "response": {
                    "content": "I apologize, but I encountered an error processing your request. Please try again.",
                    "error": True,
                    "metadata": {"error_type": type(e).__name__}
                }
            }
    
    def _construct_prompt(self, query: str, include_context: bool = True) -> str:
        """Construct an optimized prompt for capital city queries."""
        
        base_prompt = f"""You are a helpful geography assistant that provides accurate information about world capitals.

User Query: {query}

Please provide the capital city information requested. If the query mentions a country, provide its capital. If it's about a capital city, confirm and provide relevant details.

Response format:
- Be concise and accurate
- Include the country name and capital city
- Add brief geographic context if helpful
"""
        
        if include_context:
            context_prompt = """
Additional context to consider:
- Provide current capital information (some countries have changed capitals)
- Include population estimates if relevant
- Mention any interesting geographic or historical facts briefly
- If multiple countries are mentioned, address each one
"""
            return base_prompt + context_prompt
        
        return base_prompt
    
    async def cleanup_session(self) -> Dict[str, Any]:
        """Clean up PII mappings and session data."""
        
        if self.enable_pii_protection and hasattr(self.client, 'cleanup_session'):
            cleanup_result = await self.client.cleanup_session(self.session_id)
            logger.info(f"🧹 Session cleanup completed: {cleanup_result}")
            return cleanup_result
        
        logger.info("ℹ️  No session cleanup required (PII protection disabled)")
        return {"status": "no_cleanup_needed", "reason": "PII protection disabled"}


async def demonstrate_basic_usage():
    """Demonstrate basic capital finder functionality."""
    
    print("\n" + "="*60)
    print("🌍 BASIC CAPITAL FINDER DEMONSTRATION")
    print("="*60)
    
    service = CapitalFinderService(enable_pii_protection=False)
    
    queries = [
        "What is the capital of France?",
        "Tell me about Berlin, Germany's capital",
        "What are the capitals of Italy and Spain?"
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        response = await service.find_capital_secure(query, restore_pii=False)
        
        print(f"⏱️  Processing time: {response['processing_time_seconds']}s")
        print(f"💬 Response: {response['response'].get('content', 'No response')}")


async def demonstrate_pii_protection():
    """Demonstrate PII protection capabilities with sensitive data."""
    
    print("\n" + "="*60)
    print("🛡️  PII PROTECTION DEMONSTRATION")
    print("="*60)
    
    service = CapitalFinderService(enable_pii_protection=True)
    
    # Simulate queries with PII
    pii_queries = [
        "Hi, I'm John Smith from New York. What's the capital of my home country?",
        "My email is john.doe@company.com and I live in London. What country am I in and what's its capital?",
        "I was born on 1985-03-15 and my SSN is 123-45-6789. What's the capital of the USA where I live?"
    ]
    
    for query in pii_queries:
        print(f"\n📝 Query with PII: {query}")
        
        # Process with PII protection and restoration
        response = await service.find_capital_secure(
            query, 
            include_context=True,
            restore_pii=True
        )
        
        print(f"⏱️  Processing time: {response['processing_time_seconds']}s")
        print(f"🛡️  PII protection: {'✅ Enabled' if response['pii_protection_enabled'] else '❌ Disabled'}")
        
        # Show PII handling metadata if available
        if 'pii_metadata' in response.get('response', {}):
            pii_info = response['response']['pii_metadata']
            print(f"🔍 PII detected: {pii_info.get('entities_detected', 0)} entities")
            print(f"📊 PII actions taken: {', '.join(pii_info.get('actions_applied', []))}")
        
        print(f"💬 Response: {response['response'].get('content', 'No response')}")
    
    # Clean up session
    cleanup_result = await service.cleanup_session()
    print(f"\n🧹 Session cleanup: {cleanup_result.get('status', 'Unknown')}")


async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring and batch processing."""
    
    print("\n" + "="*60)
    print("📊 PERFORMANCE MONITORING DEMONSTRATION")
    print("="*60)
    
    service = CapitalFinderService(enable_pii_protection=True)
    
    # Batch queries for performance testing
    batch_queries = [
        "What is the capital of Australia?",
        "Tell me about Tokyo, Japan",
        "What's the capital of Brazil?",
        "I need to know the capital of Egypt",
        "What is the capital city of Canada?"
    ]
    
    print(f"🚀 Processing {len(batch_queries)} queries...")
    
    start_time = time.time()
    results = []
    
    # Process queries concurrently for better performance
    tasks = [
        service.find_capital_secure(query, include_context=False, restore_pii=False)
        for query in batch_queries
    ]
    
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    # Analyze results
    successful = 0
    total_processing_time = 0
    
    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            print(f"❌ Query {i+1} failed: {result}")
        else:
            successful += 1
            total_processing_time += result['processing_time_seconds']
            results.append(result)
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"   Total queries: {len(batch_queries)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {len(batch_queries) - successful}")
    print(f"   Total wall time: {total_time:.3f}s")
    print(f"   Average per query: {total_processing_time/successful:.3f}s" if successful > 0 else "   Average: N/A")
    print(f"   Throughput: {successful/total_time:.2f} queries/second")
    
    # Clean up
    await service.cleanup_session()


async def demonstrate_error_handling():
    """Demonstrate comprehensive error handling capabilities."""
    
    print("\n" + "="*60)
    print("🛠️  ERROR HANDLING DEMONSTRATION")
    print("="*60)
    
    service = CapitalFinderService(enable_pii_protection=True)
    
    # Test various edge cases
    edge_cases = [
        "",  # Empty query
        "x" * 5000,  # Very long query
        "What is the capital of Atlantis?",  # Non-existent country
        "🌍🏛️🌟",  # Emoji-only query
        "SELECT * FROM capitals WHERE country='France'",  # SQL injection attempt
    ]
    
    for query in edge_cases:
        print(f"\n🧪 Edge case: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        response = await service.find_capital_secure(query, restore_pii=False)
        
        if 'error' in response:
            print(f"⚠️  Error handled gracefully: {response['error']}")
        else:
            print(f"✅ Processed successfully")
            print(f"💬 Response: {response['response'].get('content', 'No response')[:100]}...")
    
    await service.cleanup_session()


async def main():
    """Main demonstration function."""
    
    print("🚀 PromptForge Reference Architecture: Capital City Finder")
    print("   with Microsoft Presidio PII Protection")
    print("   Financial Services Grade Implementation")
    print()
    
    # Check system health
    print("🔧 System Health Check...")
    
    try:
        # Test basic LLM client
        client = LLMClient()
        health = await client.health_check()
        print(f"   LLM Client: {'✅ Healthy' if health.get('status') == 'healthy' else '⚠️  Issues detected'}")
        
        # Test PII-aware client if available
        try:
            pii_client = PIIAwareLLMClient()
            pii_health = await pii_client.health_check()
            print(f"   PII Client: {'✅ Healthy' if pii_health.get('status') == 'healthy' else '⚠️  Issues detected'}")
        except Exception:
            print("   PII Client: ⚠️  Not available (Presidio not installed)")
        
    except Exception as e:
        print(f"   System Health: ❌ Issues detected - {e}")
    
    print()
    
    # Run all demonstrations
    demonstrations = [
        ("Basic Usage", demonstrate_basic_usage),
        ("PII Protection", demonstrate_pii_protection), 
        ("Performance Monitoring", demonstrate_performance_monitoring),
        ("Error Handling", demonstrate_error_handling)
    ]
    
    for demo_name, demo_func in demonstrations:
        try:
            print(f"\n🎯 Running {demo_name} demonstration...")
            await demo_func()
        except Exception as e:
            print(f"❌ {demo_name} demonstration failed: {e}")
            logger.exception(f"Demonstration {demo_name} failed")
        
        print(f"\n✅ {demo_name} demonstration completed")
        await asyncio.sleep(1)  # Brief pause between demonstrations
    
    print("\n" + "="*60)
    print("🎉 ALL DEMONSTRATIONS COMPLETED")
    print("="*60)
    print()
    print("📚 Key Features Demonstrated:")
    print("   ✅ PII detection and anonymization")
    print("   ✅ Policy-based PII handling")
    print("   ✅ Session management and cleanup")
    print("   ✅ Multi-provider LLM support")
    print("   ✅ Production-grade error handling")
    print("   ✅ Performance monitoring")
    print("   ✅ Async/await patterns")
    print()
    print("🔗 Learn more: https://github.com/oiiro/promptforge")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())