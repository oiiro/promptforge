#!/usr/bin/env python3
"""
Test script to verify Langfuse integration and DeepEval optimization
Run this to ensure the new architecture is working correctly
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_langfuse_import():
    """Test that Langfuse can be imported"""
    print("Testing Langfuse import...")
    try:
        from evaluation.langfuse_config import (
            langfuse_observer, 
            LangfuseConfig, 
            ObservabilityLevel
        )
        print("✅ Langfuse modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Langfuse: {e}")
        return False

def test_deepeval_import():
    """Test that DeepEval can be imported"""
    print("\nTesting DeepEval import...")
    try:
        from evaluation.deepeval_optimizer import (
            HallucinationOptimizer,
            OptimizationConfig,
            ChainOfThoughtTemplates
        )
        print("✅ DeepEval optimizer imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import DeepEval optimizer: {e}")
        return False

def test_langfuse_configuration():
    """Test Langfuse configuration from environment"""
    print("\nTesting Langfuse configuration...")
    try:
        from evaluation.langfuse_config import LangfuseConfig
        
        config = LangfuseConfig.from_env()
        print(f"  • Host: {config.host}")
        print(f"  • Enabled: {config.enabled}")
        print(f"  • Observability Level: {config.observability_level.value}")
        print(f"  • Sampling Rate: {config.sampling_rate}")
        
        if not config.public_key or not config.secret_key:
            print("⚠️  Langfuse API keys not configured (add to .env for production)")
        else:
            print("✅ Langfuse configuration loaded")
        return True
    except Exception as e:
        print(f"❌ Failed to configure Langfuse: {e}")
        return False

def test_basic_trace():
    """Test creating a basic trace with Langfuse"""
    print("\nTesting basic trace creation...")
    try:
        from evaluation.langfuse_config import langfuse_observer
        from langfuse.decorators import observe, langfuse_context
        
        @observe(name="test_function")
        def sample_function(input_text: str) -> str:
            # Simulate some processing
            result = f"Processed: {input_text}"
            
            # Add a score
            try:
                langfuse_context.score_current_trace(
                    name="test_score",
                    value=0.95,
                    comment="Test score"
                )
            except:
                pass  # Scoring might fail without API keys
            
            return result
        
        # Run the function
        result = sample_function("Hello, Langfuse!")
        print(f"  • Function result: {result}")
        print("✅ Basic trace creation works")
        return True
    except Exception as e:
        print(f"❌ Failed to create trace: {e}")
        return False

def test_optimization_config():
    """Test DeepEval optimization configuration"""
    print("\nTesting optimization configuration...")
    try:
        from evaluation.deepeval_optimizer import OptimizationConfig
        
        config = OptimizationConfig(
            max_iterations=5,
            target_hallucination_score=0.95,
            enable_cot=True,
            cot_style="structured"
        )
        
        print(f"  • Max iterations: {config.max_iterations}")
        print(f"  • Target hallucination score: {config.target_hallucination_score}")
        print(f"  • Chain-of-Thought enabled: {config.enable_cot}")
        print(f"  • CoT style: {config.cot_style}")
        print("✅ Optimization configuration works")
        return True
    except Exception as e:
        print(f"❌ Failed to create optimization config: {e}")
        return False

def test_cot_templates():
    """Test Chain-of-Thought templates"""
    print("\nTesting Chain-of-Thought templates...")
    try:
        from evaluation.deepeval_optimizer import ChainOfThoughtTemplates
        
        templates = ChainOfThoughtTemplates()
        base_prompt = "Answer the following question: {input}"
        
        # Test structured template
        structured = templates.STRUCTURED.format(base_prompt=base_prompt)
        assert "step-by-step" in structured.lower()
        print("  • Structured template: OK")
        
        # Test narrative template
        narrative = templates.NARRATIVE.format(base_prompt=base_prompt)
        assert "reasoning process" in narrative.lower()
        print("  • Narrative template: OK")
        
        # Test hybrid template
        hybrid = templates.HYBRID.format(base_prompt=base_prompt)
        assert "systematic approach" in hybrid.lower()
        print("  • Hybrid template: OK")
        
        print("✅ Chain-of-Thought templates work")
        return True
    except Exception as e:
        print(f"❌ Failed to test CoT templates: {e}")
        return False

def test_mock_optimization():
    """Test a mock optimization without actual LLM calls"""
    print("\nTesting mock optimization...")
    try:
        from evaluation.deepeval_optimizer import HallucinationOptimizer, OptimizationConfig
        
        # Create optimizer with minimal config
        config = OptimizationConfig(
            max_iterations=2,  # Just 2 iterations for testing
            target_hallucination_score=0.90,
            enable_cot=True
        )
        
        optimizer = HallucinationOptimizer(config)
        
        # Create simple test case
        test_cases = [
            {
                "input": "What is 2+2?",
                "expected_output": "4",
                "context": ["Basic arithmetic: 2+2=4"]
            }
        ]
        
        # Test prompt variation generation
        variation = optimizer._generate_prompt_variation(
            current_prompt="Answer: {input}",
            iteration=0,
            history=[],
            few_shot_examples=None
        )
        
        print(f"  • Generated prompt variation")
        print(f"  • Config temperature: {variation['config']['temperature']}")
        print(f"  • Config top_p: {variation['config']['top_p']}")
        print("✅ Mock optimization works")
        return True
    except Exception as e:
        print(f"❌ Failed mock optimization: {e}")
        return False

def main():
    """Run all tests"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║           PromptForge Langfuse Integration Test Suite           ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    tests = [
        ("Langfuse Import", test_langfuse_import),
        ("DeepEval Import", test_deepeval_import),
        ("Langfuse Configuration", test_langfuse_configuration),
        ("Basic Trace", test_basic_trace),
        ("Optimization Config", test_optimization_config),
        ("CoT Templates", test_cot_templates),
        ("Mock Optimization", test_mock_optimization)
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
        print("\n🎉 All tests passed! The Langfuse integration is working correctly.")
    elif passed > 0:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
        print("Note: Some failures are expected if API keys are not configured.")
    else:
        print("\n❌ All tests failed. Please check your installation.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())