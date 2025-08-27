"""
TruLens-PromptForge PII Integration
Seamless integration of PII monitoring with existing TruLens evaluation pipeline
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from contextlib import contextmanager

from trulens_eval import Tru, TruChain, Feedback, feedback
from trulens_eval.feedback import Groundedness
from trulens_eval.app import App
from trulens_eval.schema import Record, FeedbackResult

from monitoring.pii_metrics import (
    PIIFeedbackProvider, PIIMonitoringDashboard, PIIFeedbackLoop,
    PIIMetrics, setup_pii_monitoring
)
from presidio.middleware import PresidioMiddleware
from orchestration.llm_client import PIIAwareLLMClient

logger = logging.getLogger(__name__)

class PIIAwareTruChain:
    """
    TruChain wrapper that automatically captures PII metrics
    Extends existing PromptForge TruLens integration
    """
    
    def __init__(self, 
                 llm_client: PIIAwareLLMClient,
                 tru_instance: Tru,
                 app_name: str = "pii_aware_prompt_chain"):
        self.llm_client = llm_client
        self.tru = tru_instance
        self.app_name = app_name
        
        # Setup PII monitoring components
        self.pii_provider, self.pii_dashboard, self.pii_feedback_loop = setup_pii_monitoring(
            tru_instance, llm_client.presidio
        )
        
        # Define PII-specific feedback functions
        self.pii_feedbacks = self._create_pii_feedbacks()
        
        # Create TruChain app with PII monitoring
        self.app = self._create_pii_aware_app()
        
        logger.info(f"PIIAwareTruChain initialized for app: {app_name}")
    
    def _create_pii_feedbacks(self) -> List[Feedback]:
        """Create comprehensive PII feedback functions for TruLens"""
        
        feedbacks = []
        
        # 1. PII Masking Effectiveness
        pii_masking_feedback = Feedback(
            self._evaluate_pii_masking,
            name="pii_masking_effectiveness",
            higher_is_better=True
        ).on(
            prompt=lambda record: record.main_input,
            response=lambda record: record.main_output,
            session_id=lambda record: record.session_id
        )
        feedbacks.append(pii_masking_feedback)
        
        # 2. PII Leakage Detection  
        pii_leakage_feedback = Feedback(
            self._evaluate_pii_leakage,
            name="pii_leakage_detection", 
            higher_is_better=True
        ).on(
            response=lambda record: record.main_output,
            context=lambda record: getattr(record, 'pii_context', {})
        )
        feedbacks.append(pii_leakage_feedback)
        
        # 3. PII Restoration Accuracy
        pii_restoration_feedback = Feedback(
            self._evaluate_pii_restoration,
            name="pii_restoration_accuracy",
            higher_is_better=True
        ).on(
            session_id=lambda record: record.session_id,
            restored_text=lambda record: record.main_output,
            original_entities=lambda record: getattr(record, 'original_pii_entities', [])
        )
        feedbacks.append(pii_restoration_feedback)
        
        # 4. PII Processing Latency
        pii_latency_feedback = Feedback(
            self._evaluate_pii_latency,
            name="pii_processing_latency",
            higher_is_better=False  # Lower latency is better
        ).on(
            session_id=lambda record: record.session_id
        )
        feedbacks.append(pii_latency_feedback)
        
        # 5. Policy Compliance
        policy_compliance_feedback = Feedback(
            self._evaluate_policy_compliance,
            name="pii_policy_compliance",
            higher_is_better=True
        ).on(
            session_id=lambda record: record.session_id,
            prompt=lambda record: record.main_input,
            response=lambda record: record.main_output
        )
        feedbacks.append(policy_compliance_feedback)
        
        return feedbacks
    
    def _create_pii_aware_app(self) -> TruChain:
        """Create TruChain app with PII monitoring enabled"""
        
        # Create the base app
        app = TruChain(
            chain=self._chain_wrapper,
            app_id=self.app_name,
            feedbacks=self.pii_feedbacks
        )
        
        return app
    
    def _chain_wrapper(self, prompt: str, **kwargs) -> str:
        """
        Wrapper function that adds PII context to TruLens records
        """
        session_id = kwargs.get('session_id', f"session_{datetime.now().isoformat()}")
        
        # Execute with PII processing
        start_time = datetime.now()
        response = self.llm_client.generate(prompt, session_id=session_id, **kwargs)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Capture PII metrics for this execution
        pii_metadata = getattr(self.llm_client, '_last_pii_metadata', {})
        if pii_metadata:
            metrics = PIIMetrics(
                session_id=session_id,
                timestamp=datetime.now(),
                pii_entities_detected=len(pii_metadata.get('detected_entities', [])),
                pii_entities_masked=len(pii_metadata.get('masked_entities', [])),
                pii_entities_restored=len(pii_metadata.get('restored_entities', [])),
                masking_accuracy=pii_metadata.get('masking_accuracy', 1.0),
                restoration_accuracy=pii_metadata.get('restoration_accuracy', 1.0),
                pii_leakage_count=pii_metadata.get('leakage_count', 0),
                processing_latency_ms=processing_time,
                policy_version=pii_metadata.get('policy_version', 'unknown'),
                user_authorized=pii_metadata.get('user_authorized', False),
                entities_by_type=pii_metadata.get('entities_by_type', {})
            )
            
            # Record metrics in dashboard
            self.pii_dashboard.record_pii_execution(metrics)
        
        return response
    
    def _evaluate_pii_masking(self, prompt: str, response: str, session_id: str) -> float:
        """TruLens feedback function for PII masking effectiveness"""
        return self.pii_provider.pii_masking_effectiveness(prompt, response, session_id)
    
    def _evaluate_pii_leakage(self, response: str, context: Dict) -> float:
        """TruLens feedback function for PII leakage detection"""
        return self.pii_provider.pii_leakage_detection(response, context)
    
    def _evaluate_pii_restoration(self, session_id: str, restored_text: str, 
                                  original_entities: List[Dict]) -> float:
        """TruLens feedback function for PII restoration accuracy"""
        return self.pii_provider.pii_restoration_accuracy(
            session_id, restored_text, original_entities
        )
    
    def _evaluate_pii_latency(self, session_id: str) -> float:
        """TruLens feedback function for PII processing latency"""
        # Get latest metrics for this session
        for metrics in reversed(self.pii_dashboard.metrics_history):
            if metrics.session_id == session_id:
                return metrics.processing_latency_ms
        return 0.0
    
    def _evaluate_policy_compliance(self, session_id: str, prompt: str, response: str) -> float:
        """TruLens feedback function for PII policy compliance"""
        # Evaluate compliance with current PII policies
        try:
            # Check if all required entities were properly handled
            analysis = self.llm_client.presidio.analyzer.analyze(text=prompt, language='en')
            policy = self.llm_client.presidio.policy_engine.get_active_policy()
            
            compliance_score = 1.0
            
            for entity_result in analysis:
                entity_type = entity_result.entity_type
                required_action = policy.entities.get(entity_type)
                
                if required_action and not self._verify_action_taken(
                    entity_result, required_action, prompt, response
                ):
                    compliance_score -= 0.1  # Penalty for non-compliance
            
            return max(0.0, compliance_score)
            
        except Exception as e:
            logger.error(f"Error evaluating policy compliance: {e}")
            return 0.0
    
    def _verify_action_taken(self, entity_result, required_action, prompt: str, response: str) -> bool:
        """Verify that required PII action was taken"""
        # Simplified verification - would be more sophisticated in production
        entity_text = prompt[entity_result.start:entity_result.end]
        
        # Check if entity appears masked in response context
        if required_action.value in ['redact', 'mask']:
            return entity_text not in response
        elif required_action.value == 'tokenize':
            # Check for tokenized format
            return f"[{entity_result.entity_type}]" in response or entity_text not in response
        
        return True  # Default to compliant for other actions
    
    def run_evaluation(self, test_prompts: List[str], 
                      session_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation including PII metrics
        
        Args:
            test_prompts: List of prompts to evaluate
            session_metadata: Optional metadata for the evaluation session
            
        Returns:
            Comprehensive evaluation results including PII metrics
        """
        logger.info(f"Starting PII-aware evaluation with {len(test_prompts)} prompts")
        
        evaluation_results = {
            "evaluation_id": f"eval_{datetime.now().isoformat()}",
            "timestamp": datetime.now().isoformat(),
            "total_prompts": len(test_prompts),
            "pii_results": [],
            "trulens_results": [],
            "summary_metrics": {},
            "incidents": [],
            "recommendations": []
        }
        
        # Track metrics before evaluation
        initial_metrics_count = len(self.pii_dashboard.metrics_history)
        
        # Run evaluation through TruChain
        for i, prompt in enumerate(test_prompts):
            try:
                session_id = f"eval_session_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Execute through TruChain (automatically captures feedback)
                with self.app as recording:
                    response = self._chain_wrapper(
                        prompt, 
                        session_id=session_id,
                        **(session_metadata or {})
                    )
                
                # Capture TruLens record
                record = recording.get()
                evaluation_results["trulens_results"].append({
                    "session_id": session_id,
                    "prompt": prompt,
                    "response": response,
                    "feedback_scores": {
                        fb.name: fb.result.score if fb.result else None 
                        for fb in record.feedback_results or []
                    }
                })
                
                logger.info(f"Completed evaluation {i+1}/{len(test_prompts)}")
                
            except Exception as e:
                logger.error(f"Error evaluating prompt {i}: {e}")
                evaluation_results["trulens_results"].append({
                    "session_id": f"failed_session_{i}",
                    "prompt": prompt,
                    "error": str(e)
                })
        
        # Collect new PII metrics from evaluation
        new_metrics = self.pii_dashboard.metrics_history[initial_metrics_count:]
        evaluation_results["pii_results"] = [
            {
                "session_id": m.session_id,
                "entities_detected": m.pii_entities_detected,
                "entities_masked": m.pii_entities_masked,
                "masking_accuracy": m.masking_accuracy,
                "restoration_accuracy": m.restoration_accuracy,
                "leakage_count": m.pii_leakage_count,
                "latency_ms": m.processing_latency_ms
            }
            for m in new_metrics
        ]
        
        # Calculate summary metrics
        if new_metrics:
            evaluation_results["summary_metrics"] = {
                "avg_masking_accuracy": sum(m.masking_accuracy for m in new_metrics) / len(new_metrics),
                "avg_restoration_accuracy": sum(m.restoration_accuracy for m in new_metrics) / len(new_metrics),
                "total_pii_detected": sum(m.pii_entities_detected for m in new_metrics),
                "total_leakages": sum(m.pii_leakage_count for m in new_metrics),
                "avg_latency_ms": sum(m.processing_latency_ms for m in new_metrics) / len(new_metrics)
            }
        
        # Get recent incidents
        recent_incidents = [i for i in self.pii_dashboard.incidents if not i.resolved]
        evaluation_results["incidents"] = [
            {
                "incident_id": i.incident_id,
                "severity": i.severity.value,
                "type": i.incident_type,
                "affected_entities": i.affected_entities
            }
            for i in recent_incidents[-5:]  # Last 5 incidents
        ]
        
        # Trigger feedback loop analysis
        feedback_analysis = self.pii_feedback_loop.analyze_performance_and_improve(days=1)
        evaluation_results["recommendations"] = feedback_analysis.get("improvements_triggered", [])
        
        logger.info("PII-aware evaluation completed")
        return evaluation_results
    
    def get_pii_dashboard(self) -> Dict[str, Any]:
        """Get current PII monitoring dashboard data"""
        return self.pii_dashboard.get_pii_dashboard_data()
    
    def trigger_improvements(self) -> Dict[str, Any]:
        """Manually trigger PII improvement analysis"""
        return self.pii_feedback_loop.analyze_performance_and_improve()

class PIIPromptEvaluationPipeline:
    """
    Complete evaluation pipeline integrating PII monitoring with existing PromptForge
    Designed to work with existing promptfoo and DeepEval integrations
    """
    
    def __init__(self, config_path: str = "pii_evaluation_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.tru = Tru()
        self.presidio_middleware = PresidioMiddleware(self.config.get("presidio", {}))
        self.llm_client = PIIAwareLLMClient(
            presidio_middleware=self.presidio_middleware,
            **self.config.get("llm_client", {})
        )
        
        # Setup PII-aware TruChain
        self.pii_chain = PIIAwareTruChain(
            self.llm_client, 
            self.tru,
            app_name=self.config.get("app_name", "pii_evaluation_pipeline")
        )
        
    def _load_config(self) -> Dict[str, Any]:
        """Load evaluation configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for PII evaluation pipeline"""
        return {
            "app_name": "pii_evaluation_pipeline",
            "presidio": {
                "redis_host": "localhost",
                "redis_port": 6379,
                "policy_name": "financial_services_standard"
            },
            "llm_client": {
                "model_name": "gpt-4",
                "temperature": 0.1
            },
            "evaluation": {
                "parallel_executions": 5,
                "timeout_seconds": 30,
                "include_adversarial": True
            }
        }
    
    def run_comprehensive_evaluation(self, 
                                   dataset_paths: Dict[str, str],
                                   output_path: str = "pii_evaluation_results.json") -> Dict[str, Any]:
        """
        Run comprehensive PII evaluation across multiple datasets
        
        Args:
            dataset_paths: Dict mapping dataset names to file paths
            output_path: Path to save results
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting comprehensive PII evaluation pipeline")
        
        all_results = {
            "pipeline_id": f"pii_eval_{datetime.now().isoformat()}",
            "timestamp": datetime.now().isoformat(),
            "datasets_evaluated": list(dataset_paths.keys()),
            "dataset_results": {},
            "overall_summary": {},
            "pii_dashboard": {},
            "improvement_recommendations": []
        }
        
        # Evaluate each dataset
        for dataset_name, dataset_path in dataset_paths.items():
            logger.info(f"Evaluating dataset: {dataset_name}")
            
            try:
                # Load test prompts
                prompts = self._load_dataset(dataset_path)
                
                # Run evaluation
                dataset_results = self.pii_chain.run_evaluation(
                    prompts,
                    session_metadata={"dataset": dataset_name}
                )
                
                all_results["dataset_results"][dataset_name] = dataset_results
                
            except Exception as e:
                logger.error(f"Error evaluating dataset {dataset_name}: {e}")
                all_results["dataset_results"][dataset_name] = {"error": str(e)}
        
        # Generate overall summary
        all_results["overall_summary"] = self._calculate_overall_summary(all_results["dataset_results"])
        
        # Get final dashboard state
        all_results["pii_dashboard"] = self.pii_chain.get_pii_dashboard()
        
        # Trigger improvement analysis
        improvements = self.pii_chain.trigger_improvements()
        all_results["improvement_recommendations"] = improvements
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Comprehensive evaluation complete. Results saved to: {output_path}")
        return all_results
    
    def _load_dataset(self, dataset_path: str) -> List[str]:
        """Load test prompts from dataset file"""
        import pandas as pd
        
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
            # Assume 'prompt' or 'input' column contains test prompts
            prompt_column = 'prompt' if 'prompt' in df.columns else 'input'
            return df[prompt_column].tolist()
        elif dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
                return [item['prompt'] for item in data if 'prompt' in item]
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    def _calculate_overall_summary(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall evaluation summary across all datasets"""
        summary = {
            "total_datasets": len(dataset_results),
            "successful_datasets": 0,
            "failed_datasets": 0,
            "total_prompts_evaluated": 0,
            "overall_pii_metrics": {
                "avg_masking_accuracy": 0.0,
                "avg_restoration_accuracy": 0.0,
                "total_incidents": 0,
                "total_leakages": 0
            }
        }
        
        valid_results = []
        
        for dataset_name, results in dataset_results.items():
            if "error" in results:
                summary["failed_datasets"] += 1
                continue
                
            summary["successful_datasets"] += 1
            summary["total_prompts_evaluated"] += results.get("total_prompts", 0)
            
            if "summary_metrics" in results and results["summary_metrics"]:
                valid_results.append(results["summary_metrics"])
        
        # Calculate averages from valid results
        if valid_results:
            summary["overall_pii_metrics"]["avg_masking_accuracy"] = sum(
                r.get("avg_masking_accuracy", 0) for r in valid_results
            ) / len(valid_results)
            
            summary["overall_pii_metrics"]["avg_restoration_accuracy"] = sum(
                r.get("avg_restoration_accuracy", 0) for r in valid_results
            ) / len(valid_results)
            
            summary["overall_pii_metrics"]["total_leakages"] = sum(
                r.get("total_leakages", 0) for r in valid_results
            )
            
            summary["overall_pii_metrics"]["total_incidents"] = sum(
                len(results.get("incidents", [])) for results in dataset_results.values()
                if "error" not in results
            )
        
        return summary

# Integration helper functions
def create_pii_aware_trulens_app(llm_client: PIIAwareLLMClient, 
                                 app_name: str = "pii_prompt_app") -> PIIAwareTruChain:
    """
    Helper function to create PII-aware TruLens app
    
    Args:
        llm_client: PIIAwareLLMClient instance
        app_name: Name for the TruLens app
        
    Returns:
        PIIAwareTruChain ready for evaluation
    """
    tru = Tru()
    return PIIAwareTruChain(llm_client, tru, app_name)

def run_pii_evaluation_suite(config_path: str, 
                           datasets: Dict[str, str],
                           output_dir: str = "pii_evaluation_results") -> str:
    """
    Run complete PII evaluation suite
    
    Args:
        config_path: Path to evaluation configuration
        datasets: Dict mapping dataset names to file paths
        output_dir: Directory to save results
        
    Returns:
        Path to results file
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = PIIPromptEvaluationPipeline(config_path)
    results_path = os.path.join(output_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    pipeline.run_comprehensive_evaluation(datasets, results_path)
    return results_path

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        "app_name": "financial_services_pii_evaluation",
        "presidio": {
            "redis_host": "localhost",
            "redis_port": 6379,
            "policy_name": "financial_services_standard"
        }
    }
    
    # Save example config
    with open("pii_evaluation_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("TruLens PII Integration - Ready for comprehensive evaluation")