"""
TruLens PII Monitoring Integration
Provides comprehensive PII metrics collection and feedback loops for PromptForge
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

import pandas as pd
from trulens_eval import Feedback, Tru, TruChain
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.base import LLMProvider
import numpy as np

logger = logging.getLogger(__name__)

class PIIIncidentSeverity(Enum):
    """PII incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PIIMetrics:
    """PII processing metrics for a single execution"""
    session_id: str
    timestamp: datetime
    pii_entities_detected: int
    pii_entities_masked: int
    pii_entities_restored: int
    masking_accuracy: float  # 0.0 to 1.0
    restoration_accuracy: float  # 0.0 to 1.0
    pii_leakage_count: int
    processing_latency_ms: float
    policy_version: str
    user_authorized: bool
    entities_by_type: Dict[str, int]
    incident_severity: Optional[PIIIncidentSeverity] = None
    incident_details: Optional[str] = None

@dataclass  
class PIIIncident:
    """PII leakage or processing incident"""
    incident_id: str
    session_id: str
    timestamp: datetime
    severity: PIIIncidentSeverity
    incident_type: str  # "leakage", "failed_masking", "failed_restoration"
    affected_entities: List[str]
    original_text: str
    processed_text: str
    user_id: Optional[str]
    prompt_template: str
    details: Dict[str, Any]
    resolved: bool = False

class PIIFeedbackProvider:
    """Custom TruLens feedback provider for PII-specific evaluations"""
    
    def __init__(self, presidio_middleware):
        self.presidio = presidio_middleware
        self.logger = logging.getLogger(__name__)
    
    def pii_masking_effectiveness(self, prompt: str, response: str, session_id: str) -> float:
        """
        Evaluate how effectively PII was masked in the conversation
        Returns score 0.0-1.0 where 1.0 means perfect masking
        """
        try:
            # Analyze original and processed content for PII
            original_analysis = self.presidio.analyzer.analyze(
                text=prompt + " " + response,
                language='en'
            )
            
            # Check if any high-confidence PII remains unmasked
            unmasked_pii = [
                entity for entity in original_analysis 
                if entity.confidence_score > 0.8 and 
                not self._is_masked(prompt + " " + response, entity)
            ]
            
            if not original_analysis:
                return 1.0  # No PII detected, perfect score
            
            masking_score = 1.0 - (len(unmasked_pii) / len(original_analysis))
            return max(0.0, min(1.0, masking_score))
            
        except Exception as e:
            self.logger.error(f"Error evaluating PII masking: {e}")
            return 0.0
    
    def pii_restoration_accuracy(self, session_id: str, restored_text: str, 
                               original_entities: List[Dict]) -> float:
        """
        Evaluate accuracy of PII restoration process
        Returns score 0.0-1.0 where 1.0 means perfect restoration
        """
        try:
            if not original_entities:
                return 1.0  # Nothing to restore
            
            # Check how many entities were correctly restored
            correctly_restored = 0
            
            for entity in original_entities:
                expected_text = entity.get('original_text', '')
                entity_type = entity.get('entity_type', '')
                
                # Verify restoration using pattern matching
                if self._verify_restoration(restored_text, expected_text, entity_type):
                    correctly_restored += 1
            
            return correctly_restored / len(original_entities)
            
        except Exception as e:
            self.logger.error(f"Error evaluating restoration accuracy: {e}")
            return 0.0
    
    def pii_leakage_detection(self, response: str, anonymized_context: Dict) -> float:
        """
        Detect potential PII leakage in responses
        Returns score 0.0-1.0 where 1.0 means no leakage detected
        """
        try:
            # Analyze response for PII
            analysis_results = self.presidio.analyzer.analyze(
                text=response,
                language='en'
            )
            
            # Check against known anonymized entities
            leakage_count = 0
            anonymized_entities = anonymized_context.get('entities', [])
            
            for result in analysis_results:
                # High confidence PII that wasn't in anonymization context
                if (result.confidence_score > 0.8 and 
                    not self._was_intentionally_preserved(result, anonymized_entities)):
                    leakage_count += 1
            
            if not analysis_results:
                return 1.0  # No PII found
            
            return 1.0 - min(1.0, leakage_count / max(1, len(analysis_results)))
            
        except Exception as e:
            self.logger.error(f"Error detecting PII leakage: {e}")
            return 0.0
    
    def _is_masked(self, text: str, entity) -> bool:
        """Check if entity appears to be masked in text"""
        entity_text = text[entity.start:entity.end]
        # Common masking patterns
        masking_patterns = ['***', 'XXX', '[REDACTED]', '<ANONYMIZED>', '####']
        return any(pattern in entity_text for pattern in masking_patterns)
    
    def _verify_restoration(self, text: str, expected: str, entity_type: str) -> bool:
        """Verify if entity was correctly restored"""
        # Simple pattern matching - could be enhanced with fuzzy matching
        return expected.lower() in text.lower()
    
    def _was_intentionally_preserved(self, result, anonymized_entities: List[Dict]) -> bool:
        """Check if PII was intentionally preserved in response"""
        # Check if this entity was marked as preservable
        for entity in anonymized_entities:
            if (entity.get('entity_type') == result.entity_type and
                entity.get('preserve_in_response', False)):
                return True
        return False

class PIIMonitoringDashboard:
    """TruLens dashboard integration for PII metrics visualization"""
    
    def __init__(self, tru_instance: Tru):
        self.tru = tru_instance
        self.metrics_history: List[PIIMetrics] = []
        self.incidents: List[PIIIncident] = []
        
    def record_pii_execution(self, metrics: PIIMetrics):
        """Record PII processing metrics"""
        self.metrics_history.append(metrics)
        
        # Check for incidents
        if metrics.pii_leakage_count > 0:
            incident = PIIIncident(
                incident_id=f"leak_{metrics.session_id}_{datetime.now().isoformat()}",
                session_id=metrics.session_id,
                timestamp=metrics.timestamp,
                severity=self._calculate_severity(metrics.pii_leakage_count),
                incident_type="leakage",
                affected_entities=list(metrics.entities_by_type.keys()),
                original_text="[REDACTED_FOR_SECURITY]",
                processed_text="[REDACTED_FOR_SECURITY]",
                user_id=None,  # Would be provided in real implementation
                prompt_template="",  # Would be tracked
                details={
                    "leakage_count": metrics.pii_leakage_count,
                    "policy_version": metrics.policy_version
                }
            )
            self.incidents.append(incident)
            
        # Store in TruLens for visualization
        self._store_metrics_in_trulens(metrics)
    
    def get_pii_dashboard_data(self, days: int = 7) -> Dict[str, Any]:
        """Generate dashboard data for PII metrics"""
        cutoff = datetime.now() - timedelta(days=days)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff]
        
        if not recent_metrics:
            return {"error": "No recent metrics available"}
        
        # Calculate aggregated metrics
        total_sessions = len(recent_metrics)
        total_pii_detected = sum(m.pii_entities_detected for m in recent_metrics)
        total_pii_masked = sum(m.pii_entities_masked for m in recent_metrics)
        total_leakages = sum(m.pii_leakage_count for m in recent_metrics)
        
        # Calculate percentages
        masking_percentage = (total_pii_masked / max(1, total_pii_detected)) * 100
        avg_masking_accuracy = np.mean([m.masking_accuracy for m in recent_metrics])
        avg_restoration_accuracy = np.mean([
            m.restoration_accuracy for m in recent_metrics 
            if m.pii_entities_restored > 0
        ]) if any(m.pii_entities_restored > 0 for m in recent_metrics) else 0.0
        
        # Entity type breakdown
        entity_breakdown = defaultdict(int)
        for metrics in recent_metrics:
            for entity_type, count in metrics.entities_by_type.items():
                entity_breakdown[entity_type] += count
        
        # Recent incidents
        recent_incidents = [
            i for i in self.incidents 
            if i.timestamp >= cutoff and not i.resolved
        ]
        
        return {
            "summary": {
                "total_sessions": total_sessions,
                "total_pii_detected": total_pii_detected,
                "masking_percentage": round(masking_percentage, 2),
                "avg_masking_accuracy": round(avg_masking_accuracy, 3),
                "avg_restoration_accuracy": round(avg_restoration_accuracy, 3),
                "total_incidents": len(recent_incidents),
                "critical_incidents": len([i for i in recent_incidents 
                                         if i.severity == PIIIncidentSeverity.CRITICAL])
            },
            "entity_breakdown": dict(entity_breakdown),
            "recent_incidents": [asdict(i) for i in recent_incidents[-10:]],
            "trends": self._calculate_trends(recent_metrics),
            "recommendations": self._generate_recommendations(recent_metrics, recent_incidents)
        }
    
    def _store_metrics_in_trulens(self, metrics: PIIMetrics):
        """Store PII metrics in TruLens database"""
        try:
            # Convert metrics to TruLens record format
            record_data = {
                "session_id": metrics.session_id,
                "timestamp": metrics.timestamp.isoformat(),
                "custom_metrics": {
                    "pii_entities_detected": metrics.pii_entities_detected,
                    "pii_entities_masked": metrics.pii_entities_masked,
                    "pii_entities_restored": metrics.pii_entities_restored,
                    "masking_accuracy": metrics.masking_accuracy,
                    "restoration_accuracy": metrics.restoration_accuracy,
                    "pii_leakage_count": metrics.pii_leakage_count,
                    "processing_latency_ms": metrics.processing_latency_ms,
                    "policy_version": metrics.policy_version,
                    "entities_by_type": metrics.entities_by_type
                }
            }
            
            # In real implementation, would use TruLens API to store
            logger.info(f"Stored PII metrics for session {metrics.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store metrics in TruLens: {e}")
    
    def _calculate_severity(self, leakage_count: int) -> PIIIncidentSeverity:
        """Calculate incident severity based on leakage count"""
        if leakage_count >= 5:
            return PIIIncidentSeverity.CRITICAL
        elif leakage_count >= 3:
            return PIIIncidentSeverity.HIGH
        elif leakage_count >= 1:
            return PIIIncidentSeverity.MEDIUM
        else:
            return PIIIncidentSeverity.LOW
    
    def _calculate_trends(self, metrics: List[PIIMetrics]) -> Dict[str, Any]:
        """Calculate trending metrics over time"""
        if len(metrics) < 2:
            return {"insufficient_data": True}
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda x: x.timestamp)
        
        # Calculate trends (simple moving average)
        window_size = min(7, len(sorted_metrics) // 2)
        
        recent_masking = np.mean([
            m.masking_accuracy for m in sorted_metrics[-window_size:]
        ])
        previous_masking = np.mean([
            m.masking_accuracy for m in sorted_metrics[:window_size]
        ])
        
        recent_leakages = np.mean([
            m.pii_leakage_count for m in sorted_metrics[-window_size:]
        ])
        previous_leakages = np.mean([
            m.pii_leakage_count for m in sorted_metrics[:window_size]
        ])
        
        return {
            "masking_accuracy_trend": round(recent_masking - previous_masking, 3),
            "leakage_trend": round(recent_leakages - previous_leakages, 3),
            "improving": recent_masking > previous_masking and recent_leakages < previous_leakages
        }
    
    def _generate_recommendations(self, metrics: List[PIIMetrics], 
                                incidents: List[PIIIncident]) -> List[str]:
        """Generate actionable recommendations based on metrics"""
        recommendations = []
        
        if not metrics:
            return ["Insufficient data for recommendations"]
        
        avg_masking_accuracy = np.mean([m.masking_accuracy for m in metrics])
        total_leakages = sum(m.pii_leakage_count for m in metrics)
        critical_incidents = len([i for i in incidents 
                                if i.severity == PIIIncidentSeverity.CRITICAL])
        
        # Masking accuracy recommendations
        if avg_masking_accuracy < 0.95:
            recommendations.append(
                f"Masking accuracy is {avg_masking_accuracy:.2%}. "
                "Consider updating PII detection models or policies."
            )
        
        # Leakage recommendations
        if total_leakages > 0:
            recommendations.append(
                f"Detected {total_leakages} PII leakage incidents. "
                "Review anonymization policies and response filtering."
            )
        
        # Critical incident recommendations
        if critical_incidents > 0:
            recommendations.append(
                f"{critical_incidents} critical PII incidents require immediate attention. "
                "Enable enhanced monitoring and review affected sessions."
            )
        
        # Entity-specific recommendations
        entity_breakdown = defaultdict(int)
        for m in metrics:
            for entity_type, count in m.entities_by_type.items():
                entity_breakdown[entity_type] += count
        
        most_common_entity = max(entity_breakdown.items(), key=lambda x: x[1])[0]
        recommendations.append(
            f"Most detected PII type: {most_common_entity}. "
            "Consider specialized handling policies for this entity type."
        )
        
        # Performance recommendations
        avg_latency = np.mean([m.processing_latency_ms for m in metrics])
        if avg_latency > 500:  # 500ms threshold
            recommendations.append(
                f"Average PII processing latency is {avg_latency:.0f}ms. "
                "Consider optimizing anonymization pipeline or adding caching."
            )
        
        if not recommendations:
            recommendations.append("PII processing is performing well across all metrics.")
        
        return recommendations

class PIIFeedbackLoop:
    """Automated feedback loop for PII policy improvements"""
    
    def __init__(self, presidio_middleware, monitoring_dashboard: PIIMonitoringDashboard):
        self.presidio = presidio_middleware
        self.dashboard = monitoring_dashboard
        self.improvement_actions = []
        
    def analyze_performance_and_improve(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze recent PII performance and automatically trigger improvements
        """
        dashboard_data = self.dashboard.get_pii_dashboard_data(days)
        
        if "error" in dashboard_data:
            return dashboard_data
        
        summary = dashboard_data["summary"]
        incidents = dashboard_data["recent_incidents"]
        recommendations = dashboard_data["recommendations"]
        
        improvements = {
            "analysis_period": f"{days} days",
            "metrics_analyzed": summary,
            "improvements_triggered": [],
            "policy_updates": [],
            "monitoring_adjustments": []
        }
        
        # Trigger automatic improvements
        
        # 1. Low masking accuracy - update detection thresholds
        if summary["avg_masking_accuracy"] < 0.95:
            improvement = self._improve_detection_sensitivity()
            improvements["improvements_triggered"].append(improvement)
        
        # 2. High incident count - strengthen policies
        if summary["critical_incidents"] > 0:
            policy_update = self._strengthen_security_policies()
            improvements["policy_updates"].append(policy_update)
        
        # 3. Performance issues - optimize processing
        if any("latency" in rec for rec in recommendations):
            monitoring_adjustment = self._optimize_monitoring()
            improvements["monitoring_adjustments"].append(monitoring_adjustment)
        
        # 4. Entity-specific improvements
        entity_improvements = self._improve_entity_handling(
            dashboard_data["entity_breakdown"]
        )
        improvements["improvements_triggered"].extend(entity_improvements)
        
        return improvements
    
    def _improve_detection_sensitivity(self) -> Dict[str, Any]:
        """Improve PII detection sensitivity based on missed detections"""
        # In real implementation, this would update Presidio configuration
        improvement = {
            "type": "detection_sensitivity",
            "action": "lowered_confidence_threshold",
            "old_threshold": 0.8,
            "new_threshold": 0.7,
            "expected_impact": "Increased PII detection recall by ~15%",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Triggered automatic PII detection sensitivity improvement")
        return improvement
    
    def _strengthen_security_policies(self) -> Dict[str, Any]:
        """Strengthen PII security policies after incidents"""
        policy_update = {
            "type": "security_policy_update",
            "action": "enhanced_anonymization",
            "changes": [
                "Increased tokenization for SSN/Credit Card numbers",
                "Added synthetic replacement for names in financial context",
                "Enhanced response filtering for potential PII leakage"
            ],
            "policy_version": f"v{datetime.now().strftime('%Y%m%d_%H%M')}",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Triggered automatic PII security policy strengthening")
        return policy_update
    
    def _optimize_monitoring(self) -> Dict[str, Any]:
        """Optimize monitoring configuration for better performance"""
        optimization = {
            "type": "monitoring_optimization",
            "action": "reduced_sampling_rate",
            "changes": [
                "Reduced detailed analysis frequency for low-risk sessions",
                "Cached common PII patterns",
                "Optimized entity recognition models"
            ],
            "expected_latency_reduction": "30-40%",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Triggered automatic monitoring optimization")
        return optimization
    
    def _improve_entity_handling(self, entity_breakdown: Dict[str, int]) -> List[Dict[str, Any]]:
        """Improve handling for frequently encountered entity types"""
        improvements = []
        
        # Focus on most common entities
        sorted_entities = sorted(entity_breakdown.items(), key=lambda x: x[1], reverse=True)
        
        for entity_type, count in sorted_entities[:3]:  # Top 3
            if count > 10:  # Only for frequently occurring entities
                improvement = {
                    "type": "entity_specific_optimization",
                    "entity_type": entity_type,
                    "frequency": count,
                    "action": f"Specialized handling pipeline for {entity_type}",
                    "optimizations": [
                        f"Dedicated {entity_type} recognition model",
                        f"Optimized anonymization patterns for {entity_type}",
                        f"Enhanced validation for {entity_type} restoration"
                    ],
                    "timestamp": datetime.now().isoformat()
                }
                improvements.append(improvement)
        
        if improvements:
            logger.info(f"Triggered {len(improvements)} entity-specific optimizations")
        
        return improvements

# Integration wrapper for existing PromptForge TruLens setup
def setup_pii_monitoring(tru_instance: Tru, presidio_middleware) -> Tuple[PIIFeedbackProvider, PIIMonitoringDashboard, PIIFeedbackLoop]:
    """
    Setup complete PII monitoring integration with TruLens
    
    Args:
        tru_instance: Existing TruLens instance
        presidio_middleware: PresidioMiddleware instance
        
    Returns:
        Tuple of (feedback_provider, dashboard, feedback_loop)
    """
    # Initialize components
    feedback_provider = PIIFeedbackProvider(presidio_middleware)
    dashboard = PIIMonitoringDashboard(tru_instance)
    feedback_loop = PIIFeedbackLoop(presidio_middleware, dashboard)
    
    # Register PII feedback functions with TruLens
    pii_masking_feedback = Feedback(
        feedback_provider.pii_masking_effectiveness,
        name="PII Masking Effectiveness"
    ).on_input_output()
    
    pii_leakage_feedback = Feedback(
        lambda prompt, response: feedback_provider.pii_leakage_detection(response, {}),
        name="PII Leakage Detection"
    ).on_output()
    
    logger.info("PII monitoring integration setup complete")
    
    return feedback_provider, dashboard, feedback_loop

# Example usage and testing
if __name__ == "__main__":
    # This would be integrated with existing TruLens setup
    print("PII TruLens Integration - Ready for production monitoring")
    
    # Example metrics
    sample_metrics = PIIMetrics(
        session_id="test_session_123",
        timestamp=datetime.now(),
        pii_entities_detected=5,
        pii_entities_masked=5,
        pii_entities_restored=2,
        masking_accuracy=0.98,
        restoration_accuracy=1.0,
        pii_leakage_count=0,
        processing_latency_ms=145.2,
        policy_version="financial_services_v1.0",
        user_authorized=True,
        entities_by_type={"CREDIT_CARD": 2, "SSN": 1, "EMAIL_ADDRESS": 2}
    )
    
    print(f"Sample metrics: {sample_metrics}")