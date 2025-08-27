"""
Production Monitoring with TruLens
Continuous monitoring of production LLM calls with real-time feedback
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading

# TruLens imports - Updated for v2.x
from trulens.core import Tru, Feedback, Select

# PromptForge imports
from orchestration.llm_client import LLMClient
from evaluation.trulens_config import get_trulens_config
from observability.metrics import metrics_collector

logger = logging.getLogger(__name__)

@dataclass
class ProductionCall:
    """Record of a production LLM call"""
    call_id: str
    timestamp: str
    input_text: str
    output_text: str
    response_time_ms: float
    llm_provider: str
    llm_model: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    api_key_hash: Optional[str] = None
    guardrail_violations: List[Dict] = None
    feedback_scores: Dict[str, float] = None
    metadata: Dict[str, Any] = None

class ProductionMonitor:
    """Real-time production monitoring with TruLens feedback"""
    
    def __init__(self, 
                 enable_realtime_feedback: bool = True,
                 feedback_batch_size: int = 10,
                 feedback_interval_seconds: int = 30):
        """Initialize production monitor"""
        
        self.enable_realtime_feedback = enable_realtime_feedback
        self.feedback_batch_size = feedback_batch_size
        self.feedback_interval_seconds = feedback_interval_seconds
        
        # Initialize TruLens
        self.trulens_config = get_trulens_config()
        self.tru = self.trulens_config.tru
        
        # Create feedback functions
        self.feedback_functions = self.trulens_config.create_feedback_functions()
        
        # Production call storage
        self.pending_feedback = []
        self.call_history = defaultdict(list)  # Store by time buckets
        
        # Monitoring thread
        self._monitoring_active = False
        self._monitoring_thread = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_calls': 0,
            'total_feedback_evaluations': 0,
            'avg_response_time': 0.0,
            'error_rate': 0.0,
            'last_update': datetime.utcnow()
        }
        
        logger.info("Production monitor initialized")
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        
        logger.info("Production monitoring stopped")
    
    def record_production_call(self, 
                             call_id: str,
                             input_text: str,
                             output_text: str,
                             response_time_ms: float,
                             llm_provider: str,
                             llm_model: str,
                             user_id: Optional[str] = None,
                             session_id: Optional[str] = None,
                             guardrail_violations: List[Dict] = None,
                             metadata: Dict[str, Any] = None) -> ProductionCall:
        """Record a production LLM call for monitoring"""
        
        production_call = ProductionCall(
            call_id=call_id,
            timestamp=datetime.utcnow().isoformat(),
            input_text=input_text,
            output_text=output_text,
            response_time_ms=response_time_ms,
            llm_provider=llm_provider,
            llm_model=llm_model,
            user_id=user_id,
            session_id=session_id,
            guardrail_violations=guardrail_violations or [],
            metadata=metadata or {}
        )
        
        # Add to pending feedback queue
        if self.enable_realtime_feedback:
            self.pending_feedback.append(production_call)
        
        # Store in time-bucketed history
        hour_bucket = datetime.utcnow().strftime('%Y-%m-%d-%H')
        self.call_history[hour_bucket].append(production_call)
        
        # Update performance metrics
        self._update_performance_metrics(production_call)
        
        # Record metrics
        metrics_collector.increment('production_calls_total')
        metrics_collector.record('production_response_time', response_time_ms)
        metrics_collector.increment(f'production_calls_by_provider', tags={'provider': llm_provider})
        
        logger.debug(f"Recorded production call: {call_id}")
        return production_call
    
    def _monitoring_loop(self):
        """Background monitoring loop for processing feedback"""
        logger.info("Starting monitoring loop")
        
        while self._monitoring_active:
            try:
                # Process pending feedback in batches
                if len(self.pending_feedback) >= self.feedback_batch_size:
                    batch = self.pending_feedback[:self.feedback_batch_size]
                    self.pending_feedback = self.pending_feedback[self.feedback_batch_size:]
                    
                    # Process batch asynchronously
                    asyncio.run(self._process_feedback_batch(batch))
                
                # Sleep until next interval
                time.sleep(self.feedback_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    async def _process_feedback_batch(self, batch: List[ProductionCall]):
        """Process a batch of production calls for feedback"""
        logger.debug(f"Processing feedback batch of {len(batch)} calls")
        
        try:
            for call in batch:
                feedback_scores = await self._calculate_production_feedback(call)
                call.feedback_scores = feedback_scores
                
                # Check for alerts
                await self._check_feedback_alerts(call, feedback_scores)
                
                # Update metrics
                self.performance_metrics['total_feedback_evaluations'] += 1
            
            logger.debug(f"Processed feedback for {len(batch)} calls")
            
        except Exception as e:
            logger.error(f"Error processing feedback batch: {e}")
    
    async def _calculate_production_feedback(self, call: ProductionCall) -> Dict[str, float]:
        """Calculate feedback scores for a production call"""
        feedback_scores = {}
        
        try:
            # Groundedness feedback
            if 'groundedness' in self.feedback_functions:
                # Simple heuristic for production
                has_structured_output = self._is_structured_output(call.output_text)
                groundedness_score = 1.0 if has_structured_output else 0.5
                feedback_scores['groundedness'] = groundedness_score
            
            # Answer relevance
            if 'answer_relevance' in self.feedback_functions:
                relevance_score = self._calculate_relevance_score(call.input_text, call.output_text)
                feedback_scores['answer_relevance'] = relevance_score
            
            # Toxicity
            if 'toxicity' in self.feedback_functions:
                toxicity_score = self._calculate_toxicity_score(call.output_text)
                feedback_scores['toxicity'] = toxicity_score
            
            # Financial compliance
            if 'financial_compliance' in self.feedback_functions:
                compliance_score = self.trulens_config._check_financial_compliance(call.output_text)
                feedback_scores['financial_compliance'] = compliance_score
            
            # Schema compliance
            if 'schema_compliance' in self.feedback_functions:
                schema_score = self.trulens_config._check_schema_compliance(call.output_text)
                feedback_scores['schema_compliance'] = schema_score
            
            # Response time feedback (custom)
            response_time_score = self._calculate_response_time_score(call.response_time_ms)
            feedback_scores['response_time'] = response_time_score
            
            # Record feedback metrics
            for feedback_name, score in feedback_scores.items():
                metrics_collector.record(f'feedback_score_{feedback_name}', score)
            
        except Exception as e:
            logger.error(f"Error calculating production feedback for call {call.call_id}: {e}")
        
        return feedback_scores
    
    def _is_structured_output(self, output_text: str) -> bool:
        """Check if output is properly structured JSON"""
        try:
            parsed = json.loads(output_text)
            return isinstance(parsed, dict) and 'capital' in parsed
        except:
            return False
    
    def _calculate_relevance_score(self, input_text: str, output_text: str) -> float:
        """Calculate relevance score (simplified)"""
        try:
            # Simple heuristic: check if output contains country-related terms
            country_terms = ['capital', 'city', 'country', 'nation']
            output_lower = output_text.lower()
            
            relevant_terms = sum(1 for term in country_terms if term in output_lower)
            return min(1.0, relevant_terms / 2.0)  # Normalize to 0-1
            
        except:
            return 0.5
    
    def _calculate_toxicity_score(self, output_text: str) -> float:
        """Calculate toxicity score (simplified)"""
        try:
            toxic_terms = [
                'hate', 'violence', 'discrimination', 'offensive', 
                'inappropriate', 'harmful', 'toxic', 'abuse'
            ]
            
            output_lower = output_text.lower()
            toxic_count = sum(1 for term in toxic_terms if term in output_lower)
            
            # Return inverse toxicity score (1.0 = not toxic, 0.0 = very toxic)
            return max(0.0, 1.0 - (toxic_count * 0.3))
            
        except:
            return 1.0  # Assume non-toxic if can't calculate
    
    def _calculate_response_time_score(self, response_time_ms: float) -> float:
        """Calculate response time score"""
        # Target: under 2000ms = 1.0, over 5000ms = 0.0
        if response_time_ms <= 2000:
            return 1.0
        elif response_time_ms >= 5000:
            return 0.0
        else:
            # Linear interpolation between 2000ms and 5000ms
            return 1.0 - ((response_time_ms - 2000) / 3000)
    
    async def _check_feedback_alerts(self, call: ProductionCall, feedback_scores: Dict[str, float]):
        """Check feedback scores against alert thresholds"""
        
        # Define alert thresholds
        alert_thresholds = {
            'toxicity': {'min': 0.8, 'severity': 'critical'},
            'financial_compliance': {'min': 0.9, 'severity': 'critical'},
            'groundedness': {'min': 0.7, 'severity': 'warning'},
            'answer_relevance': {'min': 0.6, 'severity': 'warning'},
            'schema_compliance': {'min': 0.8, 'severity': 'warning'},
            'response_time': {'min': 0.5, 'severity': 'warning'}
        }
        
        alerts = []
        
        for metric, threshold in alert_thresholds.items():
            if metric in feedback_scores:
                score = feedback_scores[metric]
                if score < threshold['min']:
                    alert = {
                        'call_id': call.call_id,
                        'metric': metric,
                        'score': score,
                        'threshold': threshold['min'],
                        'severity': threshold['severity'],
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    alerts.append(alert)
                    
                    # Record alert metric
                    metrics_collector.increment(f'production_alert_{metric}', tags={
                        'severity': threshold['severity']
                    })
        
        if alerts:
            await self._handle_alerts(alerts)
    
    async def _handle_alerts(self, alerts: List[Dict]):
        """Handle feedback alerts"""
        for alert in alerts:
            if alert['severity'] == 'critical':
                logger.error(f"CRITICAL ALERT: {alert}")
            else:
                logger.warning(f"Warning alert: {alert}")
            
            # Could integrate with external alerting systems here
            # e.g., Slack, PagerDuty, email notifications
    
    def _update_performance_metrics(self, call: ProductionCall):
        """Update performance metrics with new call data"""
        self.performance_metrics['total_calls'] += 1
        
        # Update average response time (rolling average)
        current_avg = self.performance_metrics['avg_response_time']
        total_calls = self.performance_metrics['total_calls']
        
        new_avg = ((current_avg * (total_calls - 1)) + call.response_time_ms) / total_calls
        self.performance_metrics['avg_response_time'] = new_avg
        
        # Update error rate (based on guardrail violations)
        if call.guardrail_violations:
            critical_violations = sum(1 for v in call.guardrail_violations 
                                    if v.get('severity') == 'critical')
            if critical_violations > 0:
                # Recalculate error rate
                # This is simplified - in production, use a sliding window
                pass
        
        self.performance_metrics['last_update'] = datetime.utcnow()
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get real-time monitoring dashboard data"""
        
        # Calculate recent performance
        recent_calls = self._get_recent_calls(hours=1)
        
        dashboard_data = {
            'overview': {
                'total_calls': self.performance_metrics['total_calls'],
                'total_feedback_evaluations': self.performance_metrics['total_feedback_evaluations'],
                'avg_response_time_ms': self.performance_metrics['avg_response_time'],
                'monitoring_active': self._monitoring_active,
                'pending_feedback_queue': len(self.pending_feedback)
            },
            'recent_performance': {
                'last_hour_calls': len(recent_calls),
                'last_hour_avg_response_time': self._calculate_avg_response_time(recent_calls),
                'last_hour_error_rate': self._calculate_error_rate(recent_calls)
            },
            'feedback_scores': self._get_recent_feedback_scores(),
            'alerts': self._get_recent_alerts(),
            'provider_breakdown': self._get_provider_breakdown(recent_calls),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return dashboard_data
    
    def _get_recent_calls(self, hours: int = 1) -> List[ProductionCall]:
        """Get calls from the last N hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_calls = []
        
        for hour_bucket, calls in self.call_history.items():
            bucket_time = datetime.strptime(hour_bucket, '%Y-%m-%d-%H')
            if bucket_time >= cutoff_time:
                recent_calls.extend(calls)
        
        return recent_calls
    
    def _calculate_avg_response_time(self, calls: List[ProductionCall]) -> float:
        """Calculate average response time for calls"""
        if not calls:
            return 0.0
        
        return sum(call.response_time_ms for call in calls) / len(calls)
    
    def _calculate_error_rate(self, calls: List[ProductionCall]) -> float:
        """Calculate error rate based on guardrail violations"""
        if not calls:
            return 0.0
        
        error_calls = sum(1 for call in calls 
                         if call.guardrail_violations and 
                         any(v.get('severity') == 'critical' for v in call.guardrail_violations))
        
        return error_calls / len(calls)
    
    def _get_recent_feedback_scores(self) -> Dict[str, Dict[str, float]]:
        """Get recent feedback score statistics"""
        recent_calls = self._get_recent_calls(hours=24)
        
        feedback_stats = {}
        
        # Collect all feedback scores
        all_scores = defaultdict(list)
        for call in recent_calls:
            if call.feedback_scores:
                for metric, score in call.feedback_scores.items():
                    all_scores[metric].append(score)
        
        # Calculate statistics for each metric
        for metric, scores in all_scores.items():
            if scores:
                feedback_stats[metric] = {
                    'avg': sum(scores) / len(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
        
        return feedback_stats
    
    def _get_recent_alerts(self) -> List[Dict]:
        """Get recent alerts (placeholder - would integrate with alert system)"""
        # This would typically pull from an alert management system
        return []
    
    def _get_provider_breakdown(self, calls: List[ProductionCall]) -> Dict[str, Dict]:
        """Get breakdown of calls by LLM provider"""
        provider_stats = defaultdict(lambda: {'calls': 0, 'avg_response_time': 0, 'total_time': 0})
        
        for call in calls:
            provider = call.llm_provider
            provider_stats[provider]['calls'] += 1
            provider_stats[provider]['total_time'] += call.response_time_ms
        
        # Calculate averages
        for provider in provider_stats:
            stats = provider_stats[provider]
            if stats['calls'] > 0:
                stats['avg_response_time'] = stats['total_time'] / stats['calls']
            del stats['total_time']  # Remove intermediate calculation
        
        return dict(provider_stats)
    
    def export_monitoring_data(self, 
                             hours: int = 24,
                             output_file: str = None) -> Dict[str, Any]:
        """Export monitoring data for analysis"""
        
        if output_file is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            output_file = f"ci/reports/production_monitoring_{timestamp}.json"
        
        # Get data for export
        recent_calls = self._get_recent_calls(hours=hours)
        dashboard_data = self.get_monitoring_dashboard()
        
        export_data = {
            'monitoring_summary': dashboard_data,
            'raw_calls': [asdict(call) for call in recent_calls],
            'export_metadata': {
                'export_time': datetime.utcnow().isoformat(),
                'hours_included': hours,
                'total_calls_exported': len(recent_calls)
            }
        }
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Monitoring data exported to {output_file}")
        return export_data

# Global production monitor instance
production_monitor = None

def get_production_monitor() -> ProductionMonitor:
    """Get or create global production monitor"""
    global production_monitor
    
    if production_monitor is None:
        production_monitor = ProductionMonitor(
            enable_realtime_feedback=os.getenv('ENABLE_PRODUCTION_MONITORING', 'true').lower() == 'true',
            feedback_batch_size=int(os.getenv('FEEDBACK_BATCH_SIZE', '10')),
            feedback_interval_seconds=int(os.getenv('FEEDBACK_INTERVAL_SECONDS', '30'))
        )
    
    return production_monitor

# Example usage
if __name__ == "__main__":
    import uuid
    
    # Initialize monitor
    monitor = ProductionMonitor()
    monitor.start_monitoring()
    
    # Simulate some production calls
    for i in range(5):
        call_id = str(uuid.uuid4())
        monitor.record_production_call(
            call_id=call_id,
            input_text=f"Test country {i}",
            output_text=f'{{"capital": "Test City {i}", "confidence": 0.9}}',
            response_time_ms=1000 + (i * 100),
            llm_provider="openai",
            llm_model="gpt-4-turbo",
            user_id=f"user_{i}"
        )
    
    # Wait a bit for processing
    time.sleep(35)
    
    # Get dashboard
    dashboard = monitor.get_monitoring_dashboard()
    print(json.dumps(dashboard, indent=2))
    
    # Stop monitoring
    monitor.stop_monitoring()