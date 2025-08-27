"""
Production-Grade Quality Checks for PII Processing

Comprehensive evaluation framework for PII detection, anonymization,
and de-anonymization quality with metrics and reporting.
"""

import asyncio
import csv
import json
import logging
import statistics
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .middleware import PresidioMiddleware
from .policies import PIIAction, PIIPolicy


logger = logging.getLogger(__name__)


@dataclass
class PIITestCase:
    """Test case for PII quality evaluation"""
    id: str
    text: str
    expected_entities: List[Dict[str, Any]]  # [{type, start, end, value}]
    category: str  # "golden", "edge_case", "adversarial"
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QualityMetrics:
    """Quality metrics for PII processing evaluation"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    processing_time_ms: float
    
    # PII-specific metrics
    entities_detected: int
    entities_expected: int
    entities_correctly_detected: int
    
    # Round-trip metrics (for reversible operations)
    round_trip_accuracy: Optional[float] = None
    restoration_rate: Optional[float] = None
    
    # Adversarial defense metrics
    adversarial_defense_rate: Optional[float] = None
    pii_leakage_incidents: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting"""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "processing_time_ms": self.processing_time_ms,
            "entities_detected": self.entities_detected,
            "entities_expected": self.entities_expected,
            "entities_correctly_detected": self.entities_correctly_detected,
            "round_trip_accuracy": self.round_trip_accuracy,
            "restoration_rate": self.restoration_rate,
            "adversarial_defense_rate": self.adversarial_defense_rate,
            "pii_leakage_incidents": self.pii_leakage_incidents
        }


class PIIQualityEvaluator:
    """
    Production-grade quality evaluator for PII processing
    
    Provides comprehensive testing including:
    - Detection accuracy on golden datasets
    - Round-trip anonymization accuracy
    - Adversarial prompt injection defense
    - Performance benchmarking
    """
    
    def __init__(self, presidio_middleware: PresidioMiddleware):
        """Initialize evaluator with Presidio middleware"""
        self.presidio = presidio_middleware
        self.test_cases = []
        self.results = []
    
    def load_test_datasets(self, datasets_dir: str = "datasets") -> Dict[str, int]:
        """Load test datasets from CSV files"""
        
        datasets_path = Path(datasets_dir)
        if not datasets_path.exists():
            logger.error(f"Datasets directory not found: {datasets_dir}")
            raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")
        
        loaded_counts = {}
        
        # Load golden dataset
        golden_file = datasets_path / "golden.csv"
        if golden_file.exists():
            count = self._load_csv_dataset(golden_file, "golden")
            loaded_counts["golden"] = count
            logger.info(f"Loaded {count} golden test cases")
        
        # Load edge cases
        edge_cases_file = datasets_path / "edge_cases.csv"
        if edge_cases_file.exists():
            count = self._load_csv_dataset(edge_cases_file, "edge_case")
            loaded_counts["edge_cases"] = count
            logger.info(f"Loaded {count} edge case test cases")
        
        # Load adversarial cases
        adversarial_file = datasets_path / "adversarial.csv"
        if adversarial_file.exists():
            count = self._load_csv_dataset(adversarial_file, "adversarial")
            loaded_counts["adversarial"] = count
            logger.info(f"Loaded {count} adversarial test cases")
        
        total_loaded = sum(loaded_counts.values())
        logger.info(f"Total test cases loaded: {total_loaded}")
        
        return loaded_counts
    
    def _load_csv_dataset(self, file_path: Path, category: str) -> int:
        """Load test cases from CSV file"""
        
        count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Parse expected entities from JSON string
                    expected_entities = []
                    if row.get('expected_entities'):
                        try:
                            expected_entities = json.loads(row['expected_entities'])
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse expected_entities for {row.get('id', 'unknown')}: {e}")
                    
                    test_case = PIITestCase(
                        id=row['id'],
                        text=row['text'],
                        expected_entities=expected_entities,
                        category=category,
                        description=row.get('description'),
                        metadata={
                            'difficulty': row.get('difficulty', 'medium'),
                            'language': row.get('language', 'en'),
                            'domain': row.get('domain', 'general')
                        }
                    )
                    
                    self.test_cases.append(test_case)
                    count += 1
                    
        except Exception as e:
            logger.error(f"Failed to load dataset {file_path}: {e}")
            raise
        
        return count
    
    async def evaluate_detection_quality(
        self, 
        policy_name: str = "financial_services_standard",
        test_categories: Optional[List[str]] = None
    ) -> Dict[str, QualityMetrics]:
        """
        Evaluate PII detection quality across test datasets
        
        Returns metrics by test category (golden, edge_case, adversarial)
        """
        
        if not self.test_cases:
            raise ValueError("No test cases loaded. Call load_test_datasets() first.")
        
        if test_categories is None:
            test_categories = ["golden", "edge_case", "adversarial"]
        
        results_by_category = {}
        
        for category in test_categories:
            category_test_cases = [tc for tc in self.test_cases if tc.category == category]
            
            if not category_test_cases:
                logger.warning(f"No test cases found for category: {category}")
                continue
            
            logger.info(f"Evaluating {len(category_test_cases)} {category} test cases...")
            
            # Run detection evaluation for this category
            metrics = await self._evaluate_category_detection(
                category_test_cases, policy_name
            )
            
            results_by_category[category] = metrics
            
            logger.info(
                f"{category} results - Precision: {metrics.precision:.3f}, "
                f"Recall: {metrics.recall:.3f}, F1: {metrics.f1_score:.3f}"
            )
        
        return results_by_category
    
    async def _evaluate_category_detection(
        self, test_cases: List[PIITestCase], policy_name: str
    ) -> QualityMetrics:
        """Evaluate detection quality for a specific category"""
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_processing_time = 0
        
        total_detected = 0
        total_expected = 0
        correctly_detected = 0
        
        pii_leakage_incidents = 0
        
        for test_case in test_cases:
            start_time = time.time()
            
            # Run anonymization to get detection results
            anonymized_text, metadata = await self.presidio.anonymize(
                text=test_case.text,
                session_id=f"eval_{test_case.id}",
                policy_name=policy_name
            )
            
            processing_time = (time.time() - start_time) * 1000
            total_processing_time += processing_time
            
            # Analyze results
            detected_entities = metadata.get('entities_processed', [])
            expected_entities = test_case.expected_entities
            
            total_detected += len(detected_entities)
            total_expected += len(expected_entities)
            
            # Calculate precision/recall metrics
            tp, fp, fn, correct = self._calculate_detection_metrics(
                detected_entities, expected_entities, test_case.text
            )
            
            true_positives += tp
            false_positives += fp
            false_negatives += fn
            correctly_detected += correct
            
            # Check for PII leakage (PII remains in anonymized text)
            if self._check_pii_leakage(anonymized_text, expected_entities, test_case.text):
                pii_leakage_incidents += 1
            
            # Cleanup session
            await self.presidio.cleanup_session(f"eval_{test_case.id}")
        
        # Calculate overall metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = correctly_detected / total_expected if total_expected > 0 else 0.0
        
        avg_processing_time = total_processing_time / len(test_cases) if test_cases else 0.0
        
        return QualityMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            processing_time_ms=avg_processing_time,
            entities_detected=total_detected,
            entities_expected=total_expected,
            entities_correctly_detected=correctly_detected,
            pii_leakage_incidents=pii_leakage_incidents
        )
    
    def _calculate_detection_metrics(
        self, 
        detected: List[Dict], 
        expected: List[Dict], 
        original_text: str
    ) -> Tuple[int, int, int, int]:
        """Calculate true positives, false positives, false negatives"""
        
        true_positives = 0
        false_positives = 0
        correctly_detected = 0
        
        # Create sets of (entity_type, start, end) for comparison
        detected_set = {
            (entity['type'], entity['start'], entity['end'])
            for entity in detected
        }
        
        expected_set = {
            (entity['type'], entity['start'], entity['end'])
            for entity in expected
        }
        
        # True positives: detected and expected
        true_positives = len(detected_set.intersection(expected_set))
        correctly_detected = true_positives
        
        # False positives: detected but not expected
        false_positives = len(detected_set - expected_set)
        
        # False negatives: expected but not detected
        false_negatives = len(expected_set - detected_set)
        
        return true_positives, false_positives, false_negatives, correctly_detected
    
    def _check_pii_leakage(
        self, anonymized_text: str, expected_entities: List[Dict], original_text: str
    ) -> bool:
        """Check if any expected PII entities remain in anonymized text"""
        
        for entity in expected_entities:
            start, end = entity['start'], entity['end']
            original_value = original_text[start:end]
            
            # Simple check - if original PII value appears in anonymized text
            if original_value in anonymized_text:
                return True
        
        return False
    
    async def evaluate_round_trip_quality(
        self, 
        policy_name: str = "financial_services_standard",
        sample_size: int = 100
    ) -> QualityMetrics:
        """
        Evaluate round-trip anonymization -> de-anonymization accuracy
        
        Only tests reversible operations (TOKENIZE, MASK)
        """
        
        # Filter test cases with reversible PII entities
        reversible_test_cases = []
        policy = self.presidio.policy_engine.get_policy(policy_name)
        
        if not policy:
            raise ValueError(f"Policy not found: {policy_name}")
        
        reversible_actions = {PIIAction.TOKENIZE, PIIAction.MASK}
        
        for test_case in self.test_cases:
            for entity in test_case.expected_entities:
                entity_type = entity.get('type')
                action = policy.entities.get(entity_type, policy.default_action)
                
                if action in reversible_actions:
                    reversible_test_cases.append(test_case)
                    break
        
        if not reversible_test_cases:
            logger.warning("No test cases with reversible PII entities found")
            return QualityMetrics(
                precision=0.0, recall=0.0, f1_score=0.0, accuracy=0.0,
                processing_time_ms=0.0, entities_detected=0, entities_expected=0,
                entities_correctly_detected=0, round_trip_accuracy=0.0,
                restoration_rate=0.0
            )
        
        # Sample test cases if needed
        if len(reversible_test_cases) > sample_size:
            import random
            reversible_test_cases = random.sample(reversible_test_cases, sample_size)
        
        logger.info(f"Testing round-trip quality with {len(reversible_test_cases)} test cases")
        
        total_processing_time = 0
        successful_round_trips = 0
        total_entities_restored = 0
        total_entities_reversible = 0
        
        for test_case in reversible_test_cases:
            start_time = time.time()
            session_id = f"roundtrip_{test_case.id}"
            
            try:
                # Step 1: Anonymize
                anonymized_text, anon_metadata = await self.presidio.anonymize(
                    text=test_case.text,
                    session_id=session_id,
                    policy_name=policy_name
                )
                
                # Step 2: De-anonymize
                deanonymized_text, deanon_metadata = await self.presidio.deanonymize(
                    anonymized_text=anonymized_text,
                    session_id=session_id,
                    policy_name=policy_name
                )
                
                processing_time = (time.time() - start_time) * 1000
                total_processing_time += processing_time
                
                # Check round-trip accuracy
                if self._calculate_text_similarity(test_case.text, deanonymized_text) > 0.95:
                    successful_round_trips += 1
                
                # Count restored entities
                restored_entities = deanon_metadata.get('restored_entities', 0)
                reversible_entities = anon_metadata.get('reversible_entities', 0)
                
                total_entities_restored += restored_entities
                total_entities_reversible += reversible_entities
                
            except Exception as e:
                logger.error(f"Round-trip test failed for {test_case.id}: {e}")
                continue
                
            finally:
                # Cleanup
                await self.presidio.cleanup_session(session_id)
        
        # Calculate metrics
        round_trip_accuracy = successful_round_trips / len(reversible_test_cases) if reversible_test_cases else 0.0
        restoration_rate = total_entities_restored / total_entities_reversible if total_entities_reversible > 0 else 0.0
        avg_processing_time = total_processing_time / len(reversible_test_cases) if reversible_test_cases else 0.0
        
        return QualityMetrics(
            precision=0.0, recall=0.0, f1_score=0.0, accuracy=round_trip_accuracy,
            processing_time_ms=avg_processing_time,
            entities_detected=0, entities_expected=total_entities_reversible,
            entities_correctly_detected=total_entities_restored,
            round_trip_accuracy=round_trip_accuracy,
            restoration_rate=restoration_rate
        )
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simplified implementation)"""
        
        if text1 == text2:
            return 1.0
        
        # Simple character-level similarity
        if not text1 or not text2:
            return 0.0
        
        # Use Levenshtein distance approximation
        longer = text1 if len(text1) > len(text2) else text2
        shorter = text2 if len(text1) > len(text2) else text1
        
        if len(longer) == 0:
            return 1.0
        
        # Simple similarity calculation
        matches = sum(c1 == c2 for c1, c2 in zip(text1, text2))
        max_len = max(len(text1), len(text2))
        
        return matches / max_len
    
    async def evaluate_adversarial_defense(
        self, policy_name: str = "financial_services_standard"
    ) -> QualityMetrics:
        """
        Evaluate defense against adversarial prompt injection attacks
        
        Tests ability to detect PII in manipulated/obfuscated text
        """
        
        adversarial_cases = [tc for tc in self.test_cases if tc.category == "adversarial"]
        
        if not adversarial_cases:
            logger.warning("No adversarial test cases found")
            return QualityMetrics(
                precision=0.0, recall=0.0, f1_score=0.0, accuracy=0.0,
                processing_time_ms=0.0, entities_detected=0, entities_expected=0,
                entities_correctly_detected=0, adversarial_defense_rate=0.0
            )
        
        logger.info(f"Testing adversarial defense with {len(adversarial_cases)} test cases")
        
        successful_defenses = 0
        total_processing_time = 0
        pii_leakage_incidents = 0
        
        for test_case in adversarial_cases:
            start_time = time.time()
            session_id = f"adversarial_{test_case.id}"
            
            try:
                anonymized_text, metadata = await self.presidio.anonymize(
                    text=test_case.text,
                    session_id=session_id,
                    policy_name=policy_name
                )
                
                processing_time = (time.time() - start_time) * 1000
                total_processing_time += processing_time
                
                # Check if PII was detected and anonymized
                entities_detected = metadata.get('entities_found', 0)
                expected_entities = len(test_case.expected_entities)
                
                # Successful defense if we detected at least some PII
                if entities_detected > 0 and expected_entities > 0:
                    detection_rate = entities_detected / expected_entities
                    if detection_rate >= 0.5:  # Detected at least 50% of expected PII
                        successful_defenses += 1
                
                # Check for PII leakage
                if self._check_pii_leakage(anonymized_text, test_case.expected_entities, test_case.text):
                    pii_leakage_incidents += 1
                    
            except Exception as e:
                logger.error(f"Adversarial test failed for {test_case.id}: {e}")
                continue
                
            finally:
                await self.presidio.cleanup_session(session_id)
        
        # Calculate defense rate
        defense_rate = successful_defenses / len(adversarial_cases) if adversarial_cases else 0.0
        avg_processing_time = total_processing_time / len(adversarial_cases) if adversarial_cases else 0.0
        
        return QualityMetrics(
            precision=0.0, recall=0.0, f1_score=0.0, accuracy=defense_rate,
            processing_time_ms=avg_processing_time,
            entities_detected=0, entities_expected=0, entities_correctly_detected=0,
            adversarial_defense_rate=defense_rate,
            pii_leakage_incidents=pii_leakage_incidents
        )
    
    async def evaluate_comprehensive_quality(
        self, 
        policy_name: str = "financial_services_standard",
        output_path: str = "pii_quality_report.json"
    ) -> Dict[str, Any]:
        """
        Run comprehensive quality evaluation suite
        
        Includes detection quality, round-trip accuracy, and adversarial defense
        """
        
        logger.info("Starting comprehensive PII quality evaluation...")
        
        start_time = time.time()
        
        # 1. Detection Quality Evaluation
        logger.info("Evaluating detection quality...")
        detection_results = await self.evaluate_detection_quality(policy_name)
        
        # 2. Round-trip Quality Evaluation  
        logger.info("Evaluating round-trip quality...")
        roundtrip_results = await self.evaluate_round_trip_quality(policy_name)
        
        # 3. Adversarial Defense Evaluation
        logger.info("Evaluating adversarial defense...")
        adversarial_results = await self.evaluate_adversarial_defense(policy_name)
        
        total_time = (time.time() - start_time) * 1000
        
        # Compile comprehensive report
        report = {
            "evaluation_metadata": {
                "policy_name": policy_name,
                "timestamp": int(time.time()),
                "total_evaluation_time_ms": round(total_time, 2),
                "total_test_cases": len(self.test_cases),
                "presidio_version": "2.2.0",  # Update based on actual version
            },
            "detection_quality": {
                category: metrics.to_dict() 
                for category, metrics in detection_results.items()
            },
            "round_trip_quality": roundtrip_results.to_dict(),
            "adversarial_defense": adversarial_results.to_dict(),
            "overall_summary": self._generate_overall_summary(
                detection_results, roundtrip_results, adversarial_results
            )
        }
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive quality evaluation completed. Report saved to {output_path}")
        
        return report
    
    def _generate_overall_summary(
        self,
        detection_results: Dict[str, QualityMetrics],
        roundtrip_results: QualityMetrics,
        adversarial_results: QualityMetrics
    ) -> Dict[str, Any]:
        """Generate overall quality summary"""
        
        # Calculate average detection metrics
        detection_metrics = list(detection_results.values())
        avg_precision = statistics.mean([m.precision for m in detection_metrics]) if detection_metrics else 0.0
        avg_recall = statistics.mean([m.recall for m in detection_metrics]) if detection_metrics else 0.0
        avg_f1 = statistics.mean([m.f1_score for m in detection_metrics]) if detection_metrics else 0.0
        
        # Overall quality score (weighted combination)
        quality_score = (
            avg_f1 * 0.4 +  # Detection quality (40%)
            (roundtrip_results.round_trip_accuracy or 0.0) * 0.3 +  # Round-trip quality (30%)
            (adversarial_results.adversarial_defense_rate or 0.0) * 0.3  # Adversarial defense (30%)
        )
        
        # Risk assessment
        total_leakage = sum(m.pii_leakage_incidents for m in detection_metrics) + adversarial_results.pii_leakage_incidents
        
        risk_level = "HIGH" if total_leakage > 5 or quality_score < 0.7 else \
                    "MEDIUM" if total_leakage > 0 or quality_score < 0.85 else \
                    "LOW"
        
        return {
            "overall_quality_score": round(quality_score, 3),
            "average_detection_precision": round(avg_precision, 3),
            "average_detection_recall": round(avg_recall, 3),
            "average_detection_f1": round(avg_f1, 3),
            "round_trip_accuracy": round(roundtrip_results.round_trip_accuracy or 0.0, 3),
            "adversarial_defense_rate": round(adversarial_results.adversarial_defense_rate or 0.0, 3),
            "total_pii_leakage_incidents": total_leakage,
            "risk_level": risk_level,
            "production_ready": quality_score >= 0.85 and total_leakage == 0,
            "recommendations": self._generate_recommendations(
                quality_score, total_leakage, detection_results
            )
        }
    
    def _generate_recommendations(
        self, 
        quality_score: float, 
        total_leakage: int, 
        detection_results: Dict[str, QualityMetrics]
    ) -> List[str]:
        """Generate recommendations for improving PII processing quality"""
        
        recommendations = []
        
        if quality_score < 0.7:
            recommendations.append("Overall quality score is below acceptable threshold (70%). Consider policy tuning.")
        
        if total_leakage > 0:
            recommendations.append(f"PII leakage detected ({total_leakage} incidents). Review anonymization configuration.")
        
        # Check individual category performance
        for category, metrics in detection_results.items():
            if metrics.precision < 0.8:
                recommendations.append(f"Low precision in {category} dataset ({metrics.precision:.2f}). Reduce false positives.")
            
            if metrics.recall < 0.8:
                recommendations.append(f"Low recall in {category} dataset ({metrics.recall:.2f}). Improve entity detection.")
        
        if not recommendations:
            recommendations.append("PII processing quality meets production standards.")
        
        return recommendations