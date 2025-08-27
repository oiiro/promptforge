#!/usr/bin/env python3
"""
Health Check Script

Comprehensive system health check for PromptForge components including
LLM client, PII protection, and dependencies.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.llm_client import LLMClient, PIIAwareLLMClient


async def check_basic_imports():
    """Check basic imports and dependencies."""
    print("📋 Checking basic imports...")
    
    checks = []
    
    try:
        import structlog
        checks.append("✅ structlog imported successfully")
    except ImportError:
        checks.append("❌ structlog import failed")
    
    try:
        from orchestration.llm_client import LLMClient
        checks.append("✅ LLMClient imported successfully")
    except ImportError as e:
        checks.append(f"❌ LLMClient import failed: {e}")
    
    try:
        from presidio.policies import PIIPolicy, PIIAction
        checks.append("✅ PII policies imported successfully")
    except ImportError:
        checks.append("⚠️  PII policies not available (optional)")
    
    try:
        from examples.capital_finder_presidio import CapitalFinderService
        checks.append("✅ CapitalFinderService imported successfully")
    except ImportError as e:
        checks.append(f"❌ CapitalFinderService import failed: {e}")
    
    return checks


async def check_llm_client():
    """Check basic LLM client functionality."""
    print("🤖 Checking LLM client...")
    
    try:
        client = LLMClient()
        health = await client.health_check()
        
        if health.get('status') == 'healthy':
            return ["✅ LLM Client: Healthy"]
        else:
            return [f"⚠️  LLM Client: {health}"]
            
    except Exception as e:
        return [f"❌ LLM Client error: {e}"]


async def check_pii_client():
    """Check PII-aware client functionality."""
    print("🛡️  Checking PII-aware client...")
    
    try:
        pii_client = PIIAwareLLMClient()
        pii_health = await pii_client.health_check()
        
        if pii_health.get('status') == 'healthy':
            return ["✅ PII Client: Healthy"]
        else:
            return [f"⚠️  PII Client: {pii_health}"]
            
    except Exception as e:
        return [f"❌ PII Client error: {e}"]


async def check_virtual_environment():
    """Check if running in virtual environment."""
    import os
    
    if 'VIRTUAL_ENV' in os.environ:
        venv_path = os.environ['VIRTUAL_ENV']
        return [f"✅ Virtual environment detected: {venv_path}"]
    else:
        return ["⚠️  No virtual environment detected (recommended to use venv)"]


def check_dependencies():
    """Check critical dependencies availability."""
    print("📦 Checking dependencies...")
    
    checks = []
    dependencies = [
        ('asyncio', 'Python async support'),
        ('pathlib', 'Path handling'),
        ('json', 'JSON processing'),
        ('uuid', 'UUID generation'),
        ('time', 'Time utilities')
    ]
    
    for module, description in dependencies:
        try:
            __import__(module)
            checks.append(f"✅ {module}: Available")
        except ImportError:
            checks.append(f"❌ {module}: Missing ({description})")
    
    return checks


async def main():
    """Run comprehensive health check."""
    print("🔍 PromptForge Health Check")
    print("=" * 50)
    
    all_checks = []
    
    # Basic system checks
    all_checks.extend(check_dependencies())
    all_checks.extend(await check_virtual_environment())
    
    # Import checks
    all_checks.extend(await check_basic_imports())
    
    # Service health checks
    all_checks.extend(await check_llm_client())
    all_checks.extend(await check_pii_client())
    
    # Display results
    print("\n📊 Health Check Results:")
    print("-" * 30)
    
    passed = 0
    total = len(all_checks)
    
    for check in all_checks:
        print(f"   {check}")
        if check.startswith("✅"):
            passed += 1
    
    # Overall score
    score = (passed / total) * 100 if total > 0 else 0
    print(f"\n📈 Overall Health Score: {passed}/{total} ({score:.1f}%)")
    
    if score >= 90:
        print("🎉 EXCELLENT - System is fully operational!")
    elif score >= 75:
        print("✅ GOOD - System is mostly operational with minor issues")
    elif score >= 50:
        print("⚠️  FAIR - System has some issues that may affect functionality")
    else:
        print("❌ POOR - System has significant issues requiring attention")


if __name__ == "__main__":
    asyncio.run(main())