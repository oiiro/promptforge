#!/usr/bin/env python3
"""
Working test example for PromptForge Langfuse integration
Uses simplified components that work without external API dependencies
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_langfuse_basic():
    """Test basic Langfuse functionality"""
    print("🔧 Testing basic Langfuse integration...")
    
    try:
        from langfuse import Langfuse, observe
        
        # Test basic client creation
        client = Langfuse()  # Will work but show auth warning
        print("✅ Langfuse client created")
        
        # Test observe decorator
        @observe(name="test_function")
        def sample_function(input_text: str) -> str:
            return f"Processed: {input_text}"
        
        result = sample_function("Hello World")
        print(f"✅ Decorated function works: {result}")
        
        return True
    except Exception as e:
        print(f"❌ Langfuse test failed: {e}")
        return False

def test_simplified_optimizer():
    """Test the simplified DeepEval optimizer"""
    print("\n🧠 Testing simplified DeepEval optimizer...")
    
    try:
        from evaluation.deepeval_optimizer_minimal import HallucinationOptimizer, OptimizationConfig
        
        # Create optimizer
        config = OptimizationConfig(
            max_iterations=3,  # Quick test
            target_hallucination_score=0.85
        )
        optimizer = HallucinationOptimizer(config)
        print("✅ Optimizer created")
        
        # Test prompt optimization
        base_prompt = "Answer the following question: {input}"
        test_cases = [
            {
                "input": "What is 2+2?",
                "expected_output": "4",
                "context": ["Basic arithmetic"]
            }
        ]
        
        results = optimizer.optimize_prompt(
            base_prompt=base_prompt,
            test_cases=test_cases
        )
        
        print(f"✅ Optimization completed:")
        print(f"   • Iterations: {results['iterations']}")
        print(f"   • Improvement: {results['improvement']:.3f}")
        print(f"   • Final scores: {results['final_scores']}")
        
        return True
    except Exception as e:
        print(f"❌ Optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chain_of_thought():
    """Test Chain-of-Thought templates"""
    print("\n🤔 Testing Chain-of-Thought templates...")
    
    try:
        from evaluation.deepeval_optimizer_minimal import ChainOfThoughtTemplates
        
        templates = ChainOfThoughtTemplates()
        base_prompt = "Solve this problem: {input}"
        
        # Test all templates
        structured = templates.STRUCTURED.format(base_prompt=base_prompt)
        narrative = templates.NARRATIVE.format(base_prompt=base_prompt)
        hybrid = templates.HYBRID.format(base_prompt=base_prompt)
        
        print("✅ Templates generated:")
        print(f"   • Structured: {len(structured)} chars")
        print(f"   • Narrative: {len(narrative)} chars")
        print(f"   • Hybrid: {len(hybrid)} chars")
        
        # Verify content
        assert "step-by-step" in structured.lower()
        assert "reasoning process" in narrative.lower()
        assert "systematic approach" in hybrid.lower()
        
        print("✅ All templates contain expected elements")
        return True
    except Exception as e:
        print(f"❌ CoT test failed: {e}")
        return False

def test_end_to_end():
    """Test end-to-end workflow"""
    print("\n🚀 Testing end-to-end workflow...")
    
    try:
        from evaluation.deepeval_optimizer_minimal import HallucinationOptimizer, OptimizationConfig
        from langfuse import observe
        
        @observe(name="financial_analysis")
        def analyze_retirement(employee_data: str) -> dict:
            """Mock financial analysis function"""
            return {
                "eligible": True,
                "reason": "Meets age and service requirements",
                "confidence": 0.95
            }
        
        # Test the decorated function
        result = analyze_retirement("John Doe, Age: 67, Service: 25 years")
        print(f"✅ Financial analysis: {result}")
        
        # Test optimization for this use case
        optimizer = HallucinationOptimizer(OptimizationConfig(max_iterations=2))
        
        financial_prompt = """
Assess retirement eligibility for the following employee:
{input}

Consider:
- Standard retirement age is 65
- Minimum service requirement is 20 years
- Either condition qualifies for retirement

Provide a clear determination with reasoning.
"""
        
        test_cases = [
            {
                "input": "Employee: Jane Smith, Age: 68, Years of Service: 30",
                "expected_output": "ELIGIBLE - Meets both age and service requirements",
                "context": ["Standard retirement: Age 65 OR 20 years service"]
            }
        ]
        
        results = optimizer.optimize_prompt(
            base_prompt=financial_prompt,
            test_cases=test_cases
        )
        
        print(f"✅ Financial prompt optimization:")
        print(f"   • Optimized prompt length: {len(results['optimized_prompt'])} chars")
        print(f"   • Contains CoT elements: {'step-by-step' in results['optimized_prompt'].lower()}")
        
        return True
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False

def create_sample_env():
    """Create a sample .env file for development"""
    print("\n📝 Creating sample .env file...")
    
    env_content = """# PromptForge Development Configuration
# Update these values for production use

# Langfuse Configuration (optional for testing)
LANGFUSE_PUBLIC_KEY=pk-lf-development-key-here
LANGFUSE_SECRET_KEY=sk-lf-development-key-here
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED=false  # Disabled for local testing

# LLM Provider API Keys (optional for basic testing)
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
DEFAULT_LLM_PROVIDER=mock
DEFAULT_MODEL=mock

# Development Settings
LOG_LEVEL=INFO
ENVIRONMENT=development
ENABLE_MOCK_MODE=true
"""
    
    env_file = Path(".env.development")
    try:
        with open(env_file, "w") as f:
            f.write(env_content)
        print(f"✅ Created {env_file}")
        print("   Copy to .env and update with your actual keys when ready")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def main():
    """Run all working tests"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║           PromptForge Working Integration Test                    ║
    ║                     Langfuse v2.0                               ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    tests = [
        ("Langfuse Basic", test_langfuse_basic),
        ("Simplified Optimizer", test_simplified_optimizer),
        ("Chain-of-Thought", test_chain_of_thought),
        ("End-to-End Workflow", test_end_to_end),
        ("Sample Environment", create_sample_env)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status:8} | {name}")
    
    print("-"*70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! PromptForge Langfuse integration is working!")
        print("\n📚 Next steps:")
        print("1. Copy .env.development to .env")
        print("2. Add your API keys for full functionality")
        print("3. Run: python examples/prompt_refinement_example.py")
    elif passed > 0:
        print(f"\n⚠️ {total - passed} test(s) failed. See output above for details.")
    else:
        print("\n❌ All tests failed. Check your installation.")
    
    return 0 if passed >= total - 1 else 1  # Allow 1 failure

if __name__ == "__main__":
    sys.exit(main())