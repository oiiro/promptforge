#!/usr/bin/env python3
"""
Performance Benchmark

Runs performance benchmarking tests for the CapitalFinderService
with concurrent query processing.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.capital_finder_presidio import CapitalFinderService


async def benchmark():
    """Run performance benchmark with concurrent queries."""
    print("📊 Running Performance Benchmark...")
    
    service = CapitalFinderService(enable_pii_protection=True)
    
    queries = [
        'What is the capital of France?',
        'Tell me about Tokyo, Japan',
        'What is the capital of Brazil?',
        'I need the capital of Egypt',
        'What is Canada\'s capital city?'
    ] * 4  # 20 total queries
    
    print(f"🚀 Processing {len(queries)} queries concurrently...")
    start_time = time.time()
    
    # Process queries concurrently for better performance
    tasks = [
        service.find_capital_secure(query, include_context=False, restore_pii=False)
        for query in queries
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    # Analyze results
    successful = 0
    failed = 0
    total_processing_time = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ Query {i+1} failed: {result}")
            failed += 1
        else:
            successful += 1
            total_processing_time += result['processing_time_seconds']
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"   Total queries: {len(queries)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Total wall time: {total_time:.3f}s")
    print(f"   Average per query: {total_processing_time/successful:.3f}s" if successful > 0 else "   Average: N/A")
    print(f"   Throughput: {successful/total_time:.2f} queries/second")
    
    # Performance analysis
    if successful > 0:
        efficiency = (total_processing_time / total_time) * 100
        print(f"   Concurrency efficiency: {efficiency:.1f}%")
        
        if successful/total_time > 5:
            print("✅ Excellent performance (>5 queries/sec)")
        elif successful/total_time > 2:
            print("✅ Good performance (>2 queries/sec)")
        else:
            print("⚠️  Consider performance optimization")
    
    # Clean up
    await service.cleanup_session()
    print("🧹 Session cleanup completed")


if __name__ == "__main__":
    asyncio.run(benchmark())