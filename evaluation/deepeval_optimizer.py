"""
DeepEval Optimizer for Low Hallucination and Chain-of-Thought Reasoning
Integrated with Langfuse for comprehensive observability
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from deepeval import evaluate
from deepeval.metrics import (
    HallucinationMetric,
    FactualConsistencyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase
from langfuse import observe
from evaluation.langfuse_config import langfuse_observer
import structlog

logger = structlog.get_logger()

@dataclass
class OptimizationConfig:
    """Configuration for prompt optimization"""
    max_iterations: int = 10
    target_hallucination_score: float = 0.95  # 95% factual accuracy
    target_faithfulness_score: float = 0.90
    target_relevancy_score: float = 0.85
    enable_cot: bool = True
    cot_style: str = "structured"  # structured, narrative, or hybrid
    temperature_range: Tuple[float, float] = (0.0, 0.3)
    top_p_range: Tuple[float, float] = (0.9, 1.0)
    
    # Advanced optimization settings
    use_few_shot: bool = True
    few_shot_examples: int = 3
    use_self_consistency: bool = True
    self_consistency_samples: int = 3
    use_verification_step: bool = True

class ChainOfThoughtTemplates:
    """Collection of Chain-of-Thought templates for different reasoning styles"""
    
    STRUCTURED = """
Let's approach this step-by-step:

1. First, I'll identify the key facts from the context
2. Next, I'll analyze what's being asked
3. Then, I'll reason through the solution
4. Finally, I'll provide the answer based on evidence

{base_prompt}

Step-by-step reasoning:
"""
    
    NARRATIVE = """
To answer this question accurately, I need to carefully consider the available information and think through the logical connections.

{base_prompt}

My reasoning process:
"""
    
    HYBRID = """
# Task Analysis
{base_prompt}

## Systematic Approach:
1. **Context Review**: What facts are provided?
2. **Question Analysis**: What specifically is being asked?
3. **Evidence Gathering**: What information supports the answer?
4. **Logical Deduction**: How do the facts lead to the conclusion?
5. **Answer Formulation**: Clear, factual response based on evidence

## Reasoning:
"""
    
    VERIFICATION = """
{enhanced_prompt}

## Verification Steps:
Before finalizing the answer, let me verify:
- ✓ Is every claim supported by the provided context?
- ✓ Have I avoided assumptions beyond the given information?
- ✓ Is my reasoning logical and traceable?
- ✓ Does my answer directly address the question?
"""
    
    FACTUAL_CONSTRAINT = """
{enhanced_prompt}

**Important Constraints:**
- Base responses strictly on provided information
- If information doesn't support a claim, explicitly state uncertainty
- Distinguish between facts and inferences
- Cite specific evidence for each conclusion
"""

class HallucinationOptimizer:
    """Optimize prompts for minimal hallucination using DeepEval"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.iteration_history = []
        self.templates = ChainOfThoughtTemplates()
        
        # Initialize metrics
        self.hallucination_metric = HallucinationMetric(
            threshold=self.config.target_hallucination_score,
            model="gpt-4",
            include_reason=True
        )
        
        self.faithfulness_metric = FaithfulnessMetric(
            threshold=self.config.target_faithfulness_score,
            model="gpt-4",
            include_reason=True
        )
        
        self.factual_consistency_metric = FactualConsistencyMetric(
            threshold=0.9,
            model="gpt-4",
            include_reason=True
        )
        
        self.relevancy_metric = AnswerRelevancyMetric(
            threshold=self.config.target_relevancy_score,
            model="gpt-4",
            include_reason=True
        )
        
        self.contextual_precision_metric = ContextualPrecisionMetric(
            threshold=0.85,
            model="gpt-4",
            include_reason=True
        )
        
        self.contextual_recall_metric = ContextualRecallMetric(
            threshold=0.85,
            model="gpt-4",
            include_reason=True
        )
    
    @observe(name="optimize_prompt")
    def optimize_prompt(self,
                        base_prompt: str,
                        test_cases: List[Dict[str, Any]],
                        context_documents: Optional[List[str]] = None,
                        few_shot_examples: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Optimize a prompt for low hallucination and high factual accuracy
        
        Args:
            base_prompt: Initial prompt template
            test_cases: List of test inputs and expected outputs
            context_documents: Optional context for RAG scenarios
            few_shot_examples: Optional examples for few-shot learning
        
        Returns:
            Optimized prompt configuration and metrics
        """
        best_prompt = base_prompt
        best_score = 0.0
        best_config = {}
        
        # Log optimization start to Langfuse
        if langfuse_observer.client:
            trace_id = langfuse_observer.client.get_current_trace_id()
            if trace_id:
                langfuse_observer.client.create_event(
                    name="optimization_start",
                    metadata={
                        "base_prompt_length": len(base_prompt),
                        "test_cases": len(test_cases),
                        "max_iterations": self.config.max_iterations,
                        "optimization_config": self.config.__dict__
                    }
                )
        
        for iteration in range(self.config.max_iterations):
            logger.info(f"Optimization iteration {iteration + 1}/{self.config.max_iterations}")
            
            # Generate prompt variation
            prompt_variation = self._generate_prompt_variation(
                best_prompt, 
                iteration,
                self.iteration_history,
                few_shot_examples
            )
            
            # Test the prompt
            test_results = self._evaluate_prompt(
                prompt_variation,
                test_cases,
                context_documents
            )
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(test_results)
            
            # Track iteration
            iteration_data = {
                "iteration": iteration + 1,
                "prompt": prompt_variation,
                "scores": test_results,
                "composite_score": composite_score
            }
            self.iteration_history.append(iteration_data)
            
            # Log iteration results to Langfuse
            if langfuse_observer.client:
                langfuse_observer.client.score_current_trace(
                    name=f"iteration_{iteration + 1}_composite",
                    value=composite_score,
                    comment=f"Scores: {test_results}"
                )
            
            # Update best if improved
            if composite_score > best_score:
                best_score = composite_score
                best_prompt = prompt_variation["prompt"]
                best_config = prompt_variation["config"]
                
                logger.info(f"New best score: {best_score:.4f}")
                
                # Log improvement
                langfuse_context.score_current_trace(
                    name="optimization_improvement",
                    value=composite_score,
                    comment=f"Iteration {iteration + 1} - New best"
                )
            
            # Check if targets met
            if self._targets_met(test_results):
                logger.info("Optimization targets met!")
                langfuse_context.update_current_trace(
                    metadata={"targets_met": True, "final_iteration": iteration + 1}
                )
                break
        
        # Calculate final improvement
        initial_score = self.iteration_history[0]["composite_score"] if self.iteration_history else 0
        improvement = best_score - initial_score
        
        # Log final results
        langfuse_context.score_current_trace(
            name="final_composite_score",
            value=best_score
        )
        langfuse_context.score_current_trace(
            name="total_improvement",
            value=improvement
        )
        
        return {
            "optimized_prompt": best_prompt,
            "configuration": best_config,
            "final_scores": self.iteration_history[-1]["scores"] if self.iteration_history else {},
            "improvement": improvement,
            "iterations": len(self.iteration_history),
            "history": self.iteration_history
        }
    
    def _generate_prompt_variation(self, 
                                   current_prompt: str,
                                   iteration: int,
                                   history: List[Dict],
                                   few_shot_examples: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Generate a prompt variation with CoT enhancements"""
        
        enhanced_prompt = current_prompt
        
        # Add few-shot examples if enabled
        if self.config.use_few_shot and few_shot_examples and iteration > 1:
            examples_text = self._format_few_shot_examples(few_shot_examples[:self.config.few_shot_examples])
            enhanced_prompt = f"{examples_text}\n\n{enhanced_prompt}"
        
        # Apply Chain-of-Thought template
        if self.config.enable_cot:
            cot_style = self._select_cot_style(iteration)
            template = getattr(self.templates, cot_style.upper(), self.templates.STRUCTURED)
            enhanced_prompt = template.format(base_prompt=enhanced_prompt)
        
        # Add factuality constraints progressively
        if iteration > 2:
            enhanced_prompt = self.templates.FACTUAL_CONSTRAINT.format(
                enhanced_prompt=enhanced_prompt
            )
        
        # Add verification step for later iterations
        if self.config.use_verification_step and iteration > 4:
            enhanced_prompt = self.templates.VERIFICATION.format(
                enhanced_prompt=enhanced_prompt
            )
        
        # Add self-consistency instruction
        if self.config.use_self_consistency and iteration > 6:
            enhanced_prompt += """

## Self-Consistency Check:
Generate multiple reasoning paths and select the most consistent answer."""
        
        # Determine temperature and top_p
        temperature = np.interp(
            iteration / max(1, self.config.max_iterations - 1),
            [0, 1],
            self.config.temperature_range
        )
        
        top_p = np.interp(
            iteration / max(1, self.config.max_iterations - 1),
            [0, 1],
            self.config.top_p_range
        )
        
        return {
            "prompt": enhanced_prompt,
            "config": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "cot_style": cot_style if self.config.enable_cot else None,
                "iteration": iteration,
                "few_shot": self.config.use_few_shot and few_shot_examples is not None,
                "verification": self.config.use_verification_step and iteration > 4,
                "self_consistency": self.config.use_self_consistency and iteration > 6
            }
        }
    
    def _select_cot_style(self, iteration: int) -> str:
        """Select Chain-of-Thought style based on iteration"""
        if self.config.cot_style != "auto":
            return self.config.cot_style
        
        # Rotate through styles for variety
        styles = ["structured", "narrative", "hybrid"]
        return styles[iteration % len(styles)]
    
    def _format_few_shot_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format few-shot examples for inclusion in prompt"""
        formatted = "## Examples:\n\n"
        for i, example in enumerate(examples, 1):
            formatted += f"### Example {i}:\n"
            formatted += f"**Input**: {example.get('input', '')}\n"
            formatted += f"**Context**: {example.get('context', '')}\n"
            formatted += f"**Output**: {example.get('output', '')}\n"
            formatted += f"**Reasoning**: {example.get('reasoning', '')}\n\n"
        return formatted
    
    @observe(name="evaluate_prompt")
    def _evaluate_prompt(self,
                        prompt_variation: Dict[str, Any],
                        test_cases: List[Dict[str, Any]],
                        context_documents: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate a prompt variation using DeepEval metrics"""
        
        scores = {
            "hallucination": [],
            "faithfulness": [],
            "factual_consistency": [],
            "relevancy": [],
            "contextual_precision": [],
            "contextual_recall": []
        }
        
        for i, test_case in enumerate(test_cases):
            # Create DeepEval test case
            llm_test_case = LLMTestCase(
                input=test_case["input"],
                actual_output=self._generate_output(
                    prompt_variation["prompt"],
                    test_case["input"],
                    prompt_variation["config"]
                ),
                expected_output=test_case.get("expected_output"),
                context=context_documents or test_case.get("context", []),
                retrieval_context=test_case.get("retrieval_context", [])
            )
            
            # Evaluate metrics with error handling
            metric_results = self._evaluate_test_case(llm_test_case)
            
            # Aggregate scores
            for metric, score in metric_results.items():
                if metric in scores:
                    scores[metric].append(score)
            
            # Log individual test case scores
            langfuse_context.score_current_trace(
                name=f"test_case_{i}_hallucination",
                value=metric_results.get("hallucination", 0.0)
            )
        
        # Calculate average scores
        avg_scores = {
            metric: np.mean(values) if values else 0.0
            for metric, values in scores.items()
        }
        
        # Log average scores to Langfuse
        for metric, score in avg_scores.items():
            langfuse_context.score_current_trace(
                name=f"avg_{metric}",
                value=score
            )
        
        return avg_scores
    
    def _evaluate_test_case(self, test_case: LLMTestCase) -> Dict[str, float]:
        """Evaluate a single test case with all metrics"""
        results = {}
        
        # Hallucination metric
        try:
            self.hallucination_metric.measure(test_case)
            results["hallucination"] = self.hallucination_metric.score
        except Exception as e:
            logger.error(f"Hallucination metric failed: {e}")
            results["hallucination"] = 0.0
        
        # Faithfulness metric
        try:
            self.faithfulness_metric.measure(test_case)
            results["faithfulness"] = self.faithfulness_metric.score
        except Exception as e:
            logger.error(f"Faithfulness metric failed: {e}")
            results["faithfulness"] = 0.0
        
        # Factual consistency metric
        try:
            self.factual_consistency_metric.measure(test_case)
            results["factual_consistency"] = self.factual_consistency_metric.score
        except Exception as e:
            logger.error(f"Factual consistency metric failed: {e}")
            results["factual_consistency"] = 0.0
        
        # Relevancy metric
        try:
            self.relevancy_metric.measure(test_case)
            results["relevancy"] = self.relevancy_metric.score
        except Exception as e:
            logger.error(f"Relevancy metric failed: {e}")
            results["relevancy"] = 0.0
        
        # Contextual precision metric
        try:
            self.contextual_precision_metric.measure(test_case)
            results["contextual_precision"] = self.contextual_precision_metric.score
        except Exception as e:
            logger.error(f"Contextual precision metric failed: {e}")
            results["contextual_precision"] = 0.0
        
        # Contextual recall metric
        try:
            self.contextual_recall_metric.measure(test_case)
            results["contextual_recall"] = self.contextual_recall_metric.score
        except Exception as e:
            logger.error(f"Contextual recall metric failed: {e}")
            results["contextual_recall"] = 0.0
        
        return results
    
    def _generate_output(self, prompt: str, input_text: str, config: Dict[str, Any]) -> str:
        """Generate output using the LLM with given configuration"""
        # This would integrate with your LLM client
        # For now, returning a placeholder
        try:
            from orchestration.llm_client import LLMClient
            
            client = LLMClient()
            
            # Apply self-consistency if enabled
            if config.get("self_consistency") and self.config.self_consistency_samples > 1:
                outputs = []
                for _ in range(self.config.self_consistency_samples):
                    output = client.generate(
                        prompt.format(input=input_text),
                        temperature=config.get("temperature", 0.1),
                        top_p=config.get("top_p", 0.95)
                    )
                    outputs.append(output)
                
                # Select most consistent output (simplified: most common)
                from collections import Counter
                return Counter(outputs).most_common(1)[0][0]
            else:
                return client.generate(
                    prompt.format(input=input_text),
                    temperature=config.get("temperature", 0.1),
                    top_p=config.get("top_p", 0.95)
                )
        except Exception as e:
            logger.error(f"Failed to generate output: {e}")
            return f"Generated response for: {input_text[:50]}..."
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted composite score"""
        weights = {
            "hallucination": 0.30,      # High priority
            "factual_consistency": 0.25,
            "faithfulness": 0.15,
            "relevancy": 0.10,
            "contextual_precision": 0.10,
            "contextual_recall": 0.10
        }
        
        composite = sum(
            scores.get(metric, 0.0) * weight
            for metric, weight in weights.items()
        )
        
        return composite
    
    def _targets_met(self, scores: Dict[str, float]) -> bool:
        """Check if optimization targets are met"""
        return (
            scores.get("hallucination", 0) >= self.config.target_hallucination_score and
            scores.get("faithfulness", 0) >= self.config.target_faithfulness_score and
            scores.get("relevancy", 0) >= self.config.target_relevancy_score
        )
    
    @observe(name="analyze_optimization_history")
    def analyze_optimization_history(self) -> Dict[str, Any]:
        """Analyze the optimization history for insights"""
        if not self.iteration_history:
            return {"error": "No optimization history available"}
        
        analysis = {
            "total_iterations": len(self.iteration_history),
            "best_iteration": max(self.iteration_history, key=lambda x: x["composite_score"])["iteration"],
            "score_progression": [h["composite_score"] for h in self.iteration_history],
            "metric_trends": {}
        }
        
        # Analyze trends for each metric
        metrics = ["hallucination", "faithfulness", "factual_consistency", "relevancy"]
        for metric in metrics:
            values = [h["scores"].get(metric, 0) for h in self.iteration_history]
            analysis["metric_trends"][metric] = {
                "initial": values[0] if values else 0,
                "final": values[-1] if values else 0,
                "max": max(values) if values else 0,
                "min": min(values) if values else 0,
                "improvement": (values[-1] - values[0]) if values else 0
            }
        
        # Log analysis to Langfuse
        langfuse_context.update_current_trace(
            metadata={"optimization_analysis": analysis}
        )
        
        return analysis