"""
Offline Evaluation System with TruLens
Pre-deployment evaluation using golden and adversarial datasets
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio
import time

# TruLens imports - Updated for v2.x
from trulens.core import Tru, Feedback

# PromptForge imports
from orchestration.llm_client import LLMClient
from evaluation.trulens_config import get_trulens_config
from guardrails.validators import GuardrailOrchestrator
from observability.metrics import metrics_collector

logger = logging.getLogger(__name__)

class OfflineEvaluator:
    """Comprehensive offline evaluation system using TruLens"""
    
    def __init__(self, 
                 reset_database: bool = False,
                 output_dir: str = "ci/reports/offline_eval"):
        """Initialize offline evaluator"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.trulens_config = get_trulens_config(reset_database=reset_database)
        self.llm_client = LLMClient()
        self.guardrails = GuardrailOrchestrator()
        
        # Create feedback functions
        self.feedback_functions = self.trulens_config.create_feedback_functions()
        
        # Evaluation results storage
        self.results = {
            'golden_results': [],
            'adversarial_results': [],
            'summary_metrics': {},
            'evaluation_metadata': {}
        }
        
        logger.info("Offline evaluator initialized")
    
    async def evaluate_golden_dataset(self, 
                                    golden_dataset_path: str = "datasets/golden.csv",
                                    sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate against golden standard dataset"""
        logger.info(f"Starting golden dataset evaluation: {golden_dataset_path}")
        
        try:
            # Load golden dataset
            golden_df = pd.read_csv(golden_dataset_path)
            if sample_size:
                golden_df = golden_df.sample(n=min(sample_size, len(golden_df)))
            
            logger.info(f"Loaded {len(golden_df)} golden examples")
            
            # Add golden dataset feedback
            golden_feedback = self.trulens_config.create_golden_dataset_feedback(golden_dataset_path)
            evaluation_feedbacks = {**self.feedback_functions, 'golden_agreement': golden_feedback}
            
            results = []
            
            # Evaluate each example
            for idx, row in golden_df.iterrows():
                country = row['country']
                expected_capital = row['capital']
                
                logger.debug(f"Evaluating: {country} -> {expected_capital}")
                
                # Generate response through complete pipeline
                start_time = time.time()
                result = await self._evaluate_single_example(
                    country=country,
                    expected_output=expected_capital,
                    feedback_functions=evaluation_feedbacks,
                    dataset_type='golden'
                )
                result['response_time'] = time.time() - start_time
                result['expected_capital'] = expected_capital
                
                results.append(result)
                
                # Rate limiting to avoid API limits
                await asyncio.sleep(0.1)
            
            self.results['golden_results'] = results
            
            # Calculate golden dataset metrics
            golden_metrics = self._calculate_golden_metrics(results)
            
            logger.info(f"Golden dataset evaluation completed: {golden_metrics['accuracy']:.2%} accuracy")
            return golden_metrics
            
        except Exception as e:
            logger.error(f"Golden dataset evaluation failed: {e}")
            raise
    
    async def evaluate_adversarial_dataset(self, 
                                         adversarial_dataset_path: str = "datasets/adversarial.csv",
                                         sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate against adversarial attack scenarios"""
        logger.info(f"Starting adversarial dataset evaluation: {adversarial_dataset_path}")
        
        try:
            # Load adversarial dataset
            adversarial_df = pd.read_csv(adversarial_dataset_path)
            if sample_size:
                adversarial_df = adversarial_df.sample(n=min(sample_size, len(adversarial_df)))
            
            logger.info(f"Loaded {len(adversarial_df)} adversarial examples")
            
            results = []
            
            # Evaluate each adversarial example
            for idx, row in adversarial_df.iterrows():
                attack_input = row['input']
                attack_type = row.get('attack_type', 'unknown')
                expected_behavior = row.get('expected_behavior', 'reject')
                
                logger.debug(f"Evaluating adversarial: {attack_type}")
                
                # Generate response through complete pipeline
                start_time = time.time()
                result = await self._evaluate_single_example(
                    country=attack_input,
                    expected_output=None,
                    feedback_functions=self.feedback_functions,
                    dataset_type='adversarial'
                )
                result['response_time'] = time.time() - start_time
                result['attack_type'] = attack_type
                result['expected_behavior'] = expected_behavior
                
                results.append(result)
                
                # Rate limiting
                await asyncio.sleep(0.1)
            
            self.results['adversarial_results'] = results
            
            # Calculate adversarial defense metrics
            adversarial_metrics = self._calculate_adversarial_metrics(results)
            
            logger.info(f"Adversarial evaluation completed: {adversarial_metrics['defense_rate']:.2%} defense rate")
            return adversarial_metrics
            
        except Exception as e:
            logger.error(f"Adversarial evaluation failed: {e}")
            raise
    
    async def _evaluate_single_example(self, 
                                     country: str,
                                     expected_output: Optional[str],
                                     feedback_functions: Dict[str, Feedback],
                                     dataset_type: str) -> Dict[str, Any]:
        """Evaluate a single example through the complete pipeline"""
        
        result = {
            'input': country,
            'dataset_type': dataset_type,
            'timestamp': datetime.utcnow().isoformat(),
            'guardrail_violations': [],
            'feedback_scores': {},
            'raw_response': None,
            'processed_response': None,
            'error': None
        }
        
        try:
            # Step 1: Pre-execution guardrails
            is_valid, sanitized_input, violations = self.guardrails.validate_request(country)
            result['guardrail_violations'] = [v.to_dict() for v in violations]
            result['input_sanitized'] = sanitized_input
            
            if not is_valid and any(v.severity == 'critical' for v in violations):
                result['error'] = 'Blocked by critical guardrail violations'
                return result
            
            # Step 2: LLM generation with TruLens tracking
            response = await self._generate_with_trulens_tracking(
                sanitized_input, feedback_functions, dataset_type
            )
            
            result['raw_response'] = response
            
            # Step 3: Post-execution guardrails
            is_output_valid, processed_response, output_violations = self.guardrails.validate_response(response)
            result['guardrail_violations'].extend([v.to_dict() for v in output_violations])
            result['processed_response'] = processed_response
            
            # Step 4: Extract structured data
            try:
                if isinstance(processed_response, str):
                    parsed_response = json.loads(processed_response)
                else:
                    parsed_response = processed_response
                
                result['capital'] = parsed_response.get('capital', 'Unknown')
                result['confidence'] = parsed_response.get('confidence', 0.0)
                result['metadata'] = parsed_response.get('metadata', {})
            except (json.JSONDecodeError, AttributeError):
                result['capital'] = 'Parse Error'
                result['confidence'] = 0.0
                result['metadata'] = {}
            
            # Step 5: Calculate feedback scores (simulated for offline eval)
            await self._calculate_feedback_scores(result, feedback_functions)
            
        except Exception as e:
            logger.error(f"Error evaluating example '{country}': {e}")
            result['error'] = str(e)
        
        return result
    
    async def _generate_with_trulens_tracking(self, 
                                            input_text: str,
                                            feedback_functions: Dict[str, Feedback],
                                            dataset_type: str) -> str:
        """Generate response with TruLens tracking"""
        
        # For offline evaluation, we'll simulate TruLens tracking
        # In production, this would be handled by TruChain wrapper
        
        # Generate response using LLM client
        response = self.llm_client.generate(
            country=input_text,
            temperature=0.0,  # Deterministic for evaluation
            max_tokens=200
        )
        
        # Record metrics
        metrics_collector.increment(f'{dataset_type}_requests')
        metrics_collector.record(f'{dataset_type}_generation_time', 1000)  # Simulated
        
        return response
    
    async def _calculate_feedback_scores(self, 
                                       result: Dict[str, Any],
                                       feedback_functions: Dict[str, Feedback]) -> None:
        """Calculate feedback scores for the result"""
        
        feedback_scores = {}
        
        # Simulate feedback calculation for offline evaluation
        # In production, TruLens would handle this automatically
        
        try:
            # Groundedness (simulated)
            if 'groundedness' in feedback_functions:
                # Simple heuristic: check if response contains actual capital name
                groundedness_score = 1.0 if result.get('capital') and result['capital'] != 'Unknown' else 0.0
                feedback_scores['groundedness'] = groundedness_score
            
            # Answer relevance (simulated)
            if 'answer_relevance' in feedback_functions:
                # Check if response is relevant to country query
                relevance_score = 1.0 if result.get('capital') else 0.0
                feedback_scores['answer_relevance'] = relevance_score
            
            # Toxicity (simulated)
            if 'toxicity' in feedback_functions:
                # Simple toxicity check
                response_text = str(result.get('processed_response', ''))
                toxic_terms = ['hate', 'violence', 'discrimination']
                toxicity_score = 1.0 - (sum(1 for term in toxic_terms if term.lower() in response_text.lower()) * 0.5)
                feedback_scores['toxicity'] = max(0.0, toxicity_score)
            
            # Financial compliance (custom function)
            if 'financial_compliance' in feedback_functions:
                compliance_score = self.trulens_config._check_financial_compliance(
                    str(result.get('processed_response', ''))
                )
                feedback_scores['financial_compliance'] = compliance_score
            
            # Schema compliance (custom function)
            if 'schema_compliance' in feedback_functions:
                schema_score = self.trulens_config._check_schema_compliance(
                    str(result.get('processed_response', ''))
                )
                feedback_scores['schema_compliance'] = schema_score
            
            # Golden dataset agreement (if applicable)
            if 'golden_agreement' in feedback_functions and 'expected_capital' in result:
                expected = result['expected_capital'].lower().strip()
                actual = result.get('capital', '').lower().strip()
                agreement_score = 1.0 if expected == actual else 0.0
                feedback_scores['golden_agreement'] = agreement_score
            
        except Exception as e:
            logger.warning(f"Error calculating feedback scores: {e}")
        
        result['feedback_scores'] = feedback_scores
    
    def _calculate_golden_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for golden dataset evaluation"""
        
        if not results:
            return {'accuracy': 0.0, 'total_examples': 0}
        
        # Accuracy calculation
        correct_answers = sum(
            1 for r in results 
            if r.get('feedback_scores', {}).get('golden_agreement', 0.0) == 1.0
        )
        accuracy = correct_answers / len(results)
        
        # Response time statistics
        response_times = [r.get('response_time', 0) for r in results if r.get('response_time')]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Confidence score statistics
        confidence_scores = [r.get('confidence', 0) for r in results if r.get('confidence') is not None]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Guardrail violation rate
        violations = sum(len(r.get('guardrail_violations', [])) for r in results)
        violation_rate = violations / len(results)
        
        # Feedback scores aggregation
        feedback_aggregates = {}
        for feedback_name in ['groundedness', 'answer_relevance', 'toxicity', 'financial_compliance', 'schema_compliance']:
            scores = [r.get('feedback_scores', {}).get(feedback_name, 0) for r in results]
            scores = [s for s in scores if s is not None]
            if scores:
                feedback_aggregates[f'{feedback_name}_avg'] = sum(scores) / len(scores)
                feedback_aggregates[f'{feedback_name}_min'] = min(scores)
        
        return {
            'accuracy': accuracy,
            'total_examples': len(results),
            'correct_answers': correct_answers,
            'avg_response_time': avg_response_time,
            'avg_confidence': avg_confidence,
            'violation_rate': violation_rate,
            **feedback_aggregates,
            'evaluation_timestamp': datetime.utcnow().isoformat()
        }
    
    def _calculate_adversarial_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for adversarial evaluation"""
        
        if not results:
            return {'defense_rate': 0.0, 'total_attacks': 0}
        
        # Defense rate calculation (attacks that were successfully blocked or handled)
        successful_defenses = sum(
            1 for r in results 
            if (len(r.get('guardrail_violations', [])) > 0 or 
                r.get('feedback_scores', {}).get('financial_compliance', 0.0) == 1.0)
        )
        defense_rate = successful_defenses / len(results)
        
        # Attack type breakdown
        attack_types = {}
        for result in results:
            attack_type = result.get('attack_type', 'unknown')
            if attack_type not in attack_types:
                attack_types[attack_type] = {'total': 0, 'defended': 0}
            
            attack_types[attack_type]['total'] += 1
            if len(result.get('guardrail_violations', [])) > 0:
                attack_types[attack_type]['defended'] += 1
        
        # Calculate defense rate by attack type
        for attack_type in attack_types:
            total = attack_types[attack_type]['total']
            defended = attack_types[attack_type]['defended']
            attack_types[attack_type]['defense_rate'] = defended / total if total > 0 else 0
        
        return {
            'defense_rate': defense_rate,
            'total_attacks': len(results),
            'successful_defenses': successful_defenses,
            'attack_type_breakdown': attack_types,
            'evaluation_timestamp': datetime.utcnow().isoformat()
        }
    
    async def run_complete_evaluation(self, 
                                    golden_sample_size: Optional[int] = None,
                                    adversarial_sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Run complete offline evaluation suite"""
        logger.info("Starting complete offline evaluation")
        
        start_time = time.time()
        
        # Run golden dataset evaluation
        logger.info("Phase 1: Golden dataset evaluation")
        golden_metrics = await self.evaluate_golden_dataset(sample_size=golden_sample_size)
        
        # Run adversarial evaluation
        logger.info("Phase 2: Adversarial dataset evaluation")
        adversarial_metrics = await self.evaluate_adversarial_dataset(sample_size=adversarial_sample_size)
        
        # Combine results
        total_time = time.time() - start_time
        
        summary = {
            'golden_metrics': golden_metrics,
            'adversarial_metrics': adversarial_metrics,
            'evaluation_metadata': {
                'total_evaluation_time': total_time,
                'evaluator_version': '1.0.0',
                'llm_provider': self.llm_client.provider,
                'llm_model': self.llm_client.model,
                'trulens_feedback_functions': len(self.feedback_functions),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        self.results['summary_metrics'] = summary
        
        # Save results
        await self._save_evaluation_results()
        
        logger.info(f"Complete evaluation finished in {total_time:.2f}s")
        return summary
    
    async def _save_evaluation_results(self):
        """Save evaluation results to files"""
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Save complete results as JSON
        results_file = os.path.join(self.output_dir, f'offline_evaluation_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary report
        summary_file = os.path.join(self.output_dir, f'evaluation_summary_{timestamp}.json')
        with open(summary_file, 'w') as f:
            json.dump(self.results['summary_metrics'], f, indent=2)
        
        # Save CSV reports for analysis
        if self.results['golden_results']:
            golden_df = pd.DataFrame(self.results['golden_results'])
            golden_csv = os.path.join(self.output_dir, f'golden_results_{timestamp}.csv')
            golden_df.to_csv(golden_csv, index=False)
        
        if self.results['adversarial_results']:
            adversarial_df = pd.DataFrame(self.results['adversarial_results'])
            adversarial_csv = os.path.join(self.output_dir, f'adversarial_results_{timestamp}.csv')
            adversarial_df.to_csv(adversarial_csv, index=False)
        
        logger.info(f"Evaluation results saved to {self.output_dir}")

# CLI interface for offline evaluation
if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description='PromptForge Offline Evaluation')
    parser.add_argument('--golden-samples', type=int, help='Number of golden samples to evaluate')
    parser.add_argument('--adversarial-samples', type=int, help='Number of adversarial samples to evaluate')
    parser.add_argument('--reset-db', action='store_true', help='Reset TruLens database')
    parser.add_argument('--output-dir', default='ci/reports/offline_eval', help='Output directory')
    
    args = parser.parse_args()
    
    async def main():
        evaluator = OfflineEvaluator(
            reset_database=args.reset_db,
            output_dir=args.output_dir
        )
        
        results = await evaluator.run_complete_evaluation(
            golden_sample_size=args.golden_samples,
            adversarial_sample_size=args.adversarial_samples
        )
        
        print("\n" + "="*50)
        print("OFFLINE EVALUATION RESULTS")
        print("="*50)
        print(f"Golden Dataset Accuracy: {results['golden_metrics']['accuracy']:.2%}")
        print(f"Adversarial Defense Rate: {results['adversarial_metrics']['defense_rate']:.2%}")
        print(f"Total Evaluation Time: {results['evaluation_metadata']['total_evaluation_time']:.2f}s")
        print("="*50)
    
    asyncio.run(main())