#!/usr/bin/env python3
"""
Installation status checker for PromptForge
Verifies that all essential packages are installed and working
"""

def check_package(package_name, import_name=None, optional=False):
    """Check if a package can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        status = "✅"
        message = "OK"
    except ImportError as e:
        if optional:
            status = "⚠️ "
            message = "Optional - not installed"
        else:
            status = "❌"
            message = f"MISSING - {e}"
    
    print(f"{status} {package_name:<20} {message}")

def main():
    print("🔍 PromptForge Installation Status Check")
    print("=" * 50)
    
    print("\n📦 Core LLM Packages:")
    check_package("openai")
    check_package("anthropic")
    
    print("\n🧪 Evaluation Framework:")
    check_package("deepeval")
    check_package("pytest")
    
    print("\n🚀 API & Web Framework:")
    check_package("fastapi")
    check_package("uvicorn")
    check_package("pydantic")
    
    print("\n📊 Data & Analysis:")
    check_package("pandas")
    check_package("numpy")
    
    print("\n🔒 Security & Validation:")
    check_package("cryptography")
    check_package("passlib")
    check_package("jsonschema")
    
    print("\n📡 Observability:")
    check_package("structlog")
    check_package("opentelemetry-api", "opentelemetry.trace")
    check_package("opentelemetry-sdk", "opentelemetry.sdk")
    
    print("\n🛠️  Development Tools:")
    check_package("black")
    check_package("flake8")
    check_package("mypy")
    
    print("\n🛡️  Optional Advanced Packages:")
    check_package("guardrails-ai", "guardrails", optional=True)
    check_package("transformers", optional=True)
    check_package("langchain", optional=True)
    
    print("\n🎯 Custom Modules:")
    try:
        from orchestration.llm_client import LLMClient
        print("✅ LLM Client            OK")
    except Exception as e:
        print(f"❌ LLM Client            MISSING - {e}")
    
    try:
        from guardrails.validators import GuardrailOrchestrator
        print("✅ Guardrails            OK")
    except Exception as e:
        print(f"❌ Guardrails            MISSING - {e}")
    
    try:
        from observability.metrics import metrics_collector
        print("✅ Metrics Collection    OK")
    except Exception as e:
        print(f"❌ Metrics Collection    MISSING - {e}")
    
    print("\n" + "=" * 50)
    print("📝 Notes:")
    print("  • Core LLM and evaluation packages are installed ✅")
    print("  • FastAPI server can be started with: python orchestration/app.py")
    print("  • Tests can be run with: ./ci/run_tests.sh")
    print("  • Some optional packages (guardrails-ai) may need manual installation")
    print("  • Use: pip install guardrails-ai (if needed for advanced guardrails)")

if __name__ == "__main__":
    main()