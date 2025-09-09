#!/usr/bin/env python3
"""
PromptForge Example: Financial Analysis Prompt Refinement
Demonstrates optimization for low hallucination and Chain-of-Thought reasoning
using DeepEval and Langfuse integration
"""

import json
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.deepeval_optimizer_minimal import HallucinationOptimizer, OptimizationConfig
# Langfuse integration via @observe decorators
from orchestration.llm_client import LLMClient
from langfuse import observe
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class FinancialAnalysisPromptOptimizer:
    """
    Example implementation of prompt optimization for financial analysis
    with strict factual requirements and CoT reasoning
    """
    
    def __init__(self):
        self.optimizer = HallucinationOptimizer(
            config=OptimizationConfig(
                max_iterations=10,
                target_hallucination_score=0.98,  # Very strict for financial domain
                target_faithfulness_score=0.95,
                target_relevancy_score=0.90,
                enable_cot=True,
                cot_style="structured",
                temperature_range=(0.0, 0.2),  # Keep temperature low for consistency
                top_p_range=(0.9, 0.95),
            )
        )
        self.llm_client = LLMClient()
    
    @observe(name="optimize_financial_prompt")
    def optimize_retirement_eligibility_prompt(self) -> Dict[str, Any]:
        """
        Optimize a retirement eligibility assessment prompt
        for maximum factual accuracy and clear reasoning
        """
        
        # Base prompt (before optimization)
        base_prompt = """
Assess retirement eligibility based on the following information:
{input}

Provide eligibility status and reasoning.
"""
        
        # Few-shot examples for learning
        few_shot_examples = [
            {
                "input": "Employee: Sarah Wilson, Age: 68, Years of Service: 15",
                "context": "Standard retirement age is 65, minimum service requirement is 10 years",
                "output": "ELIGIBLE - Meets both age requirement (68 > 65) and service requirement (15 > 10)",
                "reasoning": "Age 68 exceeds minimum 65, and 15 years exceeds minimum 10 years service"
            },
            {
                "input": "Employee: Tom Davis, Age: 50, Years of Service: 32",
                "context": "Early retirement available with 30+ years regardless of age",
                "output": "ELIGIBLE - Qualifies for early retirement with 32 years of service",
                "reasoning": "32 years exceeds the 30-year threshold for early retirement"
            },
            {
                "input": "Employee: Lisa Chen, Age: 40, Years of Service: 8",
                "context": "Minimum requirements: Age 65 OR 20 years of service",
                "output": "NOT ELIGIBLE - Neither age (40 < 65) nor service (8 < 20) requirements met",
                "reasoning": "Age 40 is below 65, and 8 years is below 20 years minimum service"
            }
        ]
        
        # Prepare test cases with ground truth
        test_cases = [
            {
                "input": "Employee: John Smith, Age: 67, Years of Service: 30, Salary: $85,000, Retirement Plan: 401k",
                "expected_output": "ELIGIBLE: Meets age requirement (67 >= 65) and service requirement (30 >= 20)",
                "context": [
                    "Retirement eligibility requirements: Minimum age 65 OR minimum 20 years of service",
                    "Full benefits available at age 67 with 20+ years of service",
                    "401k plans follow standard retirement rules"
                ]
            },
            {
                "input": "Employee: Jane Doe, Age: 55, Years of Service: 35, Salary: $120,000, Retirement Plan: Pension",
                "expected_output": "ELIGIBLE: Meets service requirement (35 >= 30) for early retirement",
                "context": [
                    "Early retirement available with 30+ years of service regardless of age",
                    "Pension plans allow retirement after 30 years of service",
                    "Benefits may be reduced for early retirement before age 65"
                ]
            },
            {
                "input": "Employee: Bob Johnson, Age: 45, Years of Service: 10, Salary: $70,000, Retirement Plan: 401k",
                "expected_output": "NOT ELIGIBLE: Does not meet age (45 < 65) or service (10 < 20) requirements",
                "context": [
                    "Minimum requirements: Age 65 OR 20 years of service",
                    "No early retirement options available for 401k plans with less than 20 years",
                    "Must meet at least one requirement for eligibility"
                ]
            },
            {
                "input": "Employee: Alice Brown, Age: 62, Years of Service: 25, Salary: $95,000, Retirement Plan: Hybrid",
                "expected_output": "ELIGIBLE: Meets service requirement (25 >= 20) and qualifies for early retirement at age 62",
                "context": [
                    "Hybrid plans allow early retirement at age 62 with 20+ years of service",
                    "Full benefits begin at age 65",
                    "Early retirement results in reduced benefits (3% per year before 65)"
                ]
            },
            {
                "input": "Employee: Robert Green, Age: 70, Years of Service: 5, Salary: $60,000, Retirement Plan: 401k",
                "expected_output": "ELIGIBLE: Meets age requirement (70 >= 65) despite limited service",
                "context": [
                    "Standard retirement age is 65 years",
                    "Service requirement is waived when age requirement is met",
                    "Age 70 qualifies for full retirement benefits"
                ]
            }
        ]
        
        # Context documents (company retirement policy)
        context_documents = [
            """
            RETIREMENT ELIGIBILITY POLICY - COMPREHENSIVE GUIDELINES
            
            1. Standard Retirement Eligibility:
               - Minimum age: 65 years (full benefits)
               - Minimum service: 20 years (vesting requirement)
               - Meeting EITHER condition qualifies for retirement
               - Age 67+ with any service time = full benefits
            
            2. Early Retirement Options:
               - Age 62+ with 20+ years of service (benefits reduced 3% per year before 65)
               - Any age with 30+ years of service (immediate eligibility, full benefits)
               - Age 60+ with 25+ years for hybrid plans only
            
            3. Plan-Specific Rules:
               - 401k Plans: Follow standard eligibility rules, no special provisions
               - Pension Plans: 30 years service allows immediate retirement at any age
               - Hybrid Plans: Early retirement available at age 62 with 20+ years
               - All plans: Mandatory retirement at age 72
            
            4. Benefit Calculation Factors:
               - Full benefits begin at age 67 (100% benefit rate)
               - Early retirement reduction: 3% per year before age 67
               - Service multiplier: 1.5% per year of service
               - Maximum benefit: 80% of final average salary
            
            5. Special Circumstances:
               - Disability retirement: Available at any age with 5+ years service
               - Deferred vested: 10+ years service, benefits begin at age 65
               - Rule of 85: Age + Service >= 85 qualifies for unreduced benefits
            """,
            """
            FACTUAL ASSESSMENT REQUIREMENTS FOR RETIREMENT ELIGIBILITY
            
            All retirement eligibility assessments must adhere to these standards:
            
            1. Data Accuracy:
               - State specific age and service values explicitly
               - Reference applicable policy sections by number
               - Provide clear YES/NO or ELIGIBLE/NOT ELIGIBLE determination
               - Include numerical comparisons (e.g., "67 >= 65")
            
            2. Reasoning Requirements:
               - Explain reasoning with direct policy citations
               - Show mathematical comparisons for all thresholds
               - Identify which specific rule applies (standard, early, special)
               - Avoid assumptions not directly supported by stated policy
            
            3. Prohibited Practices:
               - Do not infer information not explicitly provided
               - Do not make recommendations beyond eligibility determination
               - Do not estimate or approximate values
               - Do not consider factors outside stated policy
            
            4. Required Output Format:
               - Status: [ELIGIBLE/NOT ELIGIBLE]
               - Primary Reason: [Specific rule met/not met]
               - Supporting Data: [Age comparison, Service comparison]
               - Policy Reference: [Section number and rule]
            """
        ]
        
        # Run optimization
        print("\n" + "="*80)
        print("PROMPTFORGE: FINANCIAL PROMPT OPTIMIZATION")
        print("="*80)
        print("\nOptimizing retirement eligibility assessment prompt...")
        print(f"Target Scores:")
        print(f"  - Hallucination: ≥{self.optimizer.config.target_hallucination_score:.2%}")
        print(f"  - Faithfulness: ≥{self.optimizer.config.target_faithfulness_score:.2%}")
        print(f"  - Relevancy: ≥{self.optimizer.config.target_relevancy_score:.2%}")
        print("\n" + "-"*80)
        
        optimization_result = self.optimizer.optimize_prompt(
            base_prompt=base_prompt,
            test_cases=test_cases,
            context_documents=context_documents,
            few_shot_examples=few_shot_examples
        )
        
        # Display results
        self._display_optimization_results(optimization_result)
        
        # Test the optimized prompt with new cases
        self._test_optimized_prompt(
            optimization_result["optimized_prompt"],
            optimization_result["configuration"]
        )
        
        # Note: Langfuse logging handled automatically by @observe decorators
        
        # Analyze optimization history
        analysis = self.optimizer.analyze_optimization_history()
        self._display_optimization_analysis(analysis)
        
        return optimization_result
    
    def _display_optimization_results(self, results: Dict[str, Any]):
        """Display optimization results in a formatted manner"""
        
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)
        
        print(f"\n📊 Performance Summary:")
        print(f"  • Iterations completed: {results['iterations']}")
        print(f"  • Score improvement: {results['improvement']:.4f} ({results['improvement']*100:.1f}%)")
        
        print("\n📈 Final Scores:")
        for metric, score in results["final_scores"].items():
            status = "✅" if score >= 0.9 else "⚠️" if score >= 0.8 else "❌"
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {status} {metric:20s}: {bar} {score:.4f}")
        
        print("\n📝 Optimized Prompt:")
        print("-" * 40)
        print(results["optimized_prompt"])
        print("-" * 40)
        
        print("\n⚙️ Optimal Configuration:")
        for key, value in results["configuration"].items():
            if isinstance(value, float):
                print(f"  • {key}: {value:.3f}")
            else:
                print(f"  • {key}: {value}")
        
        if results["history"]:
            print("\n📊 Score Evolution:")
            df = pd.DataFrame([
                {
                    "Iter": h["iteration"],
                    "Halluc.": f"{h['scores'].get('hallucination', 0):.3f}",
                    "Faith.": f"{h['scores'].get('faithfulness', 0):.3f}",
                    "Factual": f"{h['scores'].get('factual_consistency', 0):.3f}",
                    "Relev.": f"{h['scores'].get('relevancy', 0):.3f}",
                    "Composite": f"{h['composite_score']:.3f}"
                }
                for h in results["history"]
            ])
            print(df.to_string(index=False))
    
    @observe(name="test_optimized_prompt")
    def _test_optimized_prompt(self, prompt: str, config: Dict[str, Any]):
        """Test the optimized prompt with new cases"""
        
        print("\n" + "="*80)
        print("TESTING OPTIMIZED PROMPT")
        print("="*80)
        
        test_cases = [
            {
                "name": "Edge Case: Exactly at threshold",
                "input": """
Employee: Michael Green
Age: 65
Years of Service: 20
Salary: $92,000
Retirement Plan: 401k
Department: Engineering
Performance Rating: Exceeds Expectations
""",
                "expected": "ELIGIBLE (meets both minimum thresholds)"
            },
            {
                "name": "Complex Case: Rule of 85",
                "input": """
Employee: Patricia White
Age: 58
Years of Service: 27
Salary: $105,000
Retirement Plan: Hybrid
Department: Finance
""",
                "expected": "ELIGIBLE (Rule of 85: 58+27=85)"
            },
            {
                "name": "Near Miss Case",
                "input": """
Employee: David Lee
Age: 64
Years of Service: 19
Salary: $88,000
Retirement Plan: 401k
""",
                "expected": "NOT ELIGIBLE (neither threshold met)"
            }
        ]
        
        for test_case in test_cases:
            print(f"\n🧪 Test: {test_case['name']}")
            print(f"Input:{test_case['input']}")
            print(f"Expected: {test_case['expected']}")
            
            # Generate response using optimized prompt
            try:
                response = self.llm_client.generate(
                    prompt.format(input=test_case['input']),
                    temperature=config.get("temperature", 0.1),
                    top_p=config.get("top_p", 0.95)
                )
                print(f"\n🤖 Generated Response:")
                print(response)
            except Exception as e:
                print(f"\n⚠️ Mock Response (LLM not available):")
                print(f"ELIGIBLE - Based on provided criteria...")
            
            # Evaluate the response
            try:
                from deepeval.test_case import LLMTestCase
                from deepeval.metrics import HallucinationMetric
                
                llm_test_case = LLMTestCase(
                    input=test_case['input'],
                    actual_output=response if 'response' in locals() else "Mock response",
                    context=[
                        "Standard retirement: Age 65 OR 20 years service",
                        "Rule of 85: Age + Service >= 85 for unreduced benefits"
                    ]
                )
                
                hallucination_metric = HallucinationMetric(threshold=0.95)
                hallucination_metric.measure(llm_test_case)
                
                print(f"\n📊 Hallucination Score: {hallucination_metric.score:.4f}")
                if hasattr(hallucination_metric, 'reason') and hallucination_metric.reason:
                    print(f"📝 Evaluation Reason: {hallucination_metric.reason}")
            except Exception as e:
                print(f"\n📊 Evaluation skipped (DeepEval not configured): {e}")
            
            print("-" * 40)
    
    def _display_optimization_analysis(self, analysis: Dict[str, Any]):
        """Display analysis of optimization history"""
        
        print("\n" + "="*80)
        print("OPTIMIZATION ANALYSIS")
        print("="*80)
        
        if "error" in analysis:
            print(f"⚠️ {analysis['error']}")
            return
        
        print(f"\n📊 Optimization Summary:")
        print(f"  • Total iterations: {analysis['total_iterations']}")
        print(f"  • Best iteration: #{analysis['best_iteration']}")
        
        print("\n📈 Metric Improvements:")
        for metric, trends in analysis.get("metric_trends", {}).items():
            improvement = trends["improvement"]
            symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
            color = "🟢" if improvement > 0.1 else "🟡" if improvement > 0 else "🔴"
            
            print(f"\n  {metric.upper()}:")
            print(f"    {color} Change: {improvement:+.4f} {symbol}")
            print(f"    • Initial: {trends['initial']:.4f}")
            print(f"    • Final:   {trends['final']:.4f}")
            print(f"    • Best:    {trends['max']:.4f}")

def main():
    """Run the prompt optimization example"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    PROMPTFORGE v2.0                              ║
    ║  Financial Prompt Optimization with Langfuse Integration         ║
    ║  Using DeepEval for Low Hallucination & CoT Reasoning           ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Langfuse is configured via environment variables and @observe decorators
    print("\n🔧 Configuration:")
    print(f"  • Langfuse: Enabled via @observe decorators")
    print(f"  • Integration: Live tracing with actual API keys")
    
    # Create optimizer
    print("\n🚀 Initializing optimizer...")
    optimizer = FinancialAnalysisPromptOptimizer()
    
    # Run optimization
    print("\n🔄 Starting optimization process...")
    results = optimizer.optimize_retirement_eligibility_prompt()
    
    # Save results
    output_file = "optimization_results.json"
    with open(output_file, "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        json_safe_results = json.loads(json.dumps(results, default=str))
        json.dump(json_safe_results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    # Create summary report
    summary = {
        "timestamp": datetime.now().isoformat(),
        "optimization_completed": True,
        "iterations": results["iterations"],
        "improvement": results["improvement"],
        "final_scores": results["final_scores"],
        "configuration": results["configuration"]
    }
    
    summary_file = "optimization_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary saved to {summary_file}")
    
    # Display final message
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*80)
    print("\n🎯 Key Achievements:")
    
    if results["final_scores"].get("hallucination", 0) >= 0.95:
        print("  ✅ Hallucination score exceeds 95% threshold")
    if results["final_scores"].get("faithfulness", 0) >= 0.90:
        print("  ✅ Faithfulness score exceeds 90% threshold")
    if results["final_scores"].get("relevancy", 0) >= 0.85:
        print("  ✅ Relevancy score exceeds 85% threshold")
    
    print("\n📚 Next Steps:")
    print("  1. Review the optimized prompt in 'optimization_results.json'")
    print("  2. Test with production data")
    print("  3. Monitor performance in Langfuse dashboard")
    print("  4. Iterate based on real-world feedback")
    
    print("\n👋 Optimization complete - check Langfuse dashboard for traces")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Optimization interrupted by user")
        pass
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        pass
        sys.exit(1)