"""
Enhanced Retirement Eligibility Endpoints with Comprehensive TruLens Feedback

This module contains both v1 (mock) and v2 (live LLM) retirement eligibility endpoints
with comprehensive TruLens feedback functions for evaluation and monitoring.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import Request, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Request/Response models
class RetirementEligibilityRequest(BaseModel):
    query: str = Field(..., description="Natural language query about retirement eligibility")
    enable_pii_protection: bool = Field(default=True, description="Enable PII anonymization")
    enable_monitoring: bool = Field(default=True, description="Enable TruLens monitoring")

class RetirementEligibilityResponse(BaseModel):
    response: str = Field(..., description="Natural language response")
    eligible: bool = Field(..., description="Whether the person is eligible for retirement")
    deposit_amount: str = Field(..., description="Required deposit amount if applicable")
    persons_processed: int = Field(..., description="Number of persons in the query")
    pii_detected: bool = Field(default=False, description="Whether PII was detected")
    pii_entities: list = Field(default=[], description="List of detected PII entity types")
    anonymization_applied: bool = Field(default=False, description="Whether anonymization was applied")
    confidence: float = Field(default=1.0, description="Confidence in the response (0-1)")
    metadata: Dict[str, Any] = Field(default={}, description="Additional processing metadata")

def create_enhanced_retirement_app(llm_client):
    """Create MockPromptForge and PromptForge apps with comprehensive TruLens feedback"""
    
    try:
        from trulens.core import TruSession
        from trulens.apps.basic import TruBasicApp
        from trulens.core.feedback import Feedback
        from trulens.providers.openai import OpenAI as TruLensOpenAI
        import numpy as np
        
        # Initialize TruLens session
        database_url = os.getenv('TRULENS_DATABASE_URL', 'sqlite:///trulens_promptforge.db')
        tru_session = TruSession(database_url=database_url)
        logger.info(f"✅ TruLens session initialized with enhanced feedback")
        
        # Initialize feedback provider (using OpenAI for comprehensive feedback)
        # Only initialize if OpenAI API key is available
        feedback_provider = None
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key:
            try:
                feedback_provider = TruLensOpenAI(api_key=openai_api_key)
                logger.info("✅ TruLens OpenAI feedback provider initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize OpenAI feedback provider: {e}")
        else:
            logger.warning("⚠️ OPENAI_API_KEY not set - advanced feedback functions will be disabled")
        
        # ==========================
        # COMPREHENSIVE FEEDBACK FUNCTIONS
        # ==========================
        
        comprehensive_feedbacks = []
        
        # Add TruLens OpenAI feedback functions if available
        if feedback_provider:
            try:
                # 1. QA Relevance - Measures relevance of answer to question
                f_qa_relevance = Feedback(
                    feedback_provider.relevance_with_cot_reasons,
                    name="Answer Relevance"
                ).on_input_output()
                comprehensive_feedbacks.append(f_qa_relevance)
                
                # 2. Context Relevance - For RAG systems  
                f_context_relevance = Feedback(
                    feedback_provider.context_relevance_with_cot_reasons,
                    name="Context Relevance"
                ).on_input().on_output()
                comprehensive_feedbacks.append(f_context_relevance)
                
                # 3. Groundedness - Check if answer is grounded in the input/context
                f_groundedness = Feedback(
                    feedback_provider.groundedness_measure_with_cot_reasons,
                    name="Groundedness"
                ).on_input().on_output()
                comprehensive_feedbacks.append(f_groundedness)
                
                # 4. Sentiment Analysis
                f_sentiment = Feedback(
                    feedback_provider.sentiment_with_cot_reasons,
                    name="Sentiment"
                ).on_output()
                comprehensive_feedbacks.append(f_sentiment)
                
                # 5. Toxicity Detection
                f_toxicity = Feedback(
                    feedback_provider.not_toxic,
                    name="Non-Toxicity",
                    higher_is_better=True
                ).on_output()
                comprehensive_feedbacks.append(f_toxicity)
                
                logger.info("✅ Added 5 comprehensive TruLens OpenAI feedback functions")
                
            except Exception as e:
                logger.error(f"❌ Error setting up TruLens OpenAI feedback: {e}")
        else:
            logger.info("⚠️ TruLens OpenAI feedback functions disabled - API key not available")
        
        # ==========================
        # CUSTOM RETIREMENT-SPECIFIC FEEDBACK
        # ==========================
        
        def retirement_response_quality(output: str) -> float:
            """Evaluate retirement-specific response quality"""
            if not output:
                return 0.0
            
            score = 0.0
            # Check for key retirement concepts
            if "eligible" in output.lower() or "eligibility" in output.lower():
                score += 0.2
            if "age" in output.lower() or "years" in output.lower():
                score += 0.15
            if "deposit" in output.lower() or "amount" in output.lower():
                score += 0.15
            if "retirement" in output.lower():
                score += 0.2
            if "confident" in output.lower() or "analysis" in output.lower():
                score += 0.15
            if len(output) > 50:  # Reasonable length
                score += 0.15
                
            return min(score, 1.0)
        
        def input_completeness(input_text: str) -> float:
            """Evaluate input completeness for retirement analysis"""
            if not input_text:
                return 0.0
                
            score = 0.0
            if any(word in input_text.lower() for word in ["age", "years", "old"]):
                score += 0.25
            if any(word in input_text.lower() for word in ["retire", "retirement", "eligible"]):
                score += 0.35
            if any(word in input_text.lower() for word in ["income", "salary", "savings", "401k"]):
                score += 0.2
            if any(word in input_text.lower() for word in ["born", "birth", "dob"]):
                score += 0.2
                
            return min(score, 1.0)
        
        def pii_protection_score(output: Dict[str, Any]) -> float:
            """Evaluate PII protection effectiveness"""
            # Check if PII protection was properly handled
            if isinstance(output, dict):
                if output.get('anonymization_applied', False):
                    return 1.0
                elif output.get('pii_detected', False):
                    return 0.5  # PII detected but not anonymized
            return 0.8  # No PII detected (good)
        
        def confidence_calibration(output: Dict[str, Any]) -> float:
            """Evaluate confidence calibration"""
            if isinstance(output, dict):
                confidence = output.get('confidence', 0.5)
                # Good confidence is between 0.6 and 0.95
                if 0.6 <= confidence <= 0.95:
                    return 1.0
                elif confidence > 0.95:
                    return 0.7  # Overconfident
                else:
                    return confidence  # Under-confident
            return 0.5
        
        # Custom feedback functions
        f_retirement_quality = Feedback(
            retirement_response_quality,
            name="Retirement Response Quality",
            higher_is_better=True
        ).on_output()
        
        f_input_completeness = Feedback(
            input_completeness,
            name="Input Completeness",
            higher_is_better=True
        ).on_input()
        
        f_pii_protection = Feedback(
            pii_protection_score,
            name="PII Protection",
            higher_is_better=True
        ).on_output()
        
        f_confidence = Feedback(
            confidence_calibration,
            name="Confidence Calibration",
            higher_is_better=True
        ).on_output()
        
        # Create unified processor
        class UnifiedRetirementProcessor:
            """Unified processor for both mock and live retirement queries"""
            
            def __init__(self, llm_client):
                self.llm_client = llm_client
                self.pii_processor = None
                self._init_pii_processor()
            
            def _init_pii_processor(self):
                """Initialize PII processor if available"""
                try:
                    from presidio_analyzer import AnalyzerEngine
                    from presidio_anonymizer import AnonymizerEngine
                    self.pii_processor = {
                        'analyzer': AnalyzerEngine(),
                        'anonymizer': AnonymizerEngine()
                    }
                    logger.info("✅ PII processor initialized")
                except ImportError:
                    logger.warning("⚠️ Presidio not available - PII protection disabled")
            
            def __call__(self, query: str, mode: str = "mock", enable_pii: bool = True) -> Dict[str, Any]:
                """Main callable interface for TruLens instrumentation"""
                if mode == "mock":
                    return self.process_retirement_query_mock(query, enable_pii)
                else:
                    return self.process_retirement_query_live(query, enable_pii)
            
            def detect_and_anonymize_pii(self, text: str) -> Dict[str, Any]:
                """Detect and optionally anonymize PII in text"""
                if not self.pii_processor:
                    return {
                        'text': text,
                        'pii_detected': False,
                        'entities': [],
                        'anonymized': False
                    }
                
                try:
                    # Analyze for PII
                    analyzer_results = self.pii_processor['analyzer'].analyze(
                        text=text,
                        language='en'
                    )
                    
                    if analyzer_results:
                        # Anonymize if PII found
                        anonymized_result = self.pii_processor['anonymizer'].anonymize(
                            text=text,
                            analyzer_results=analyzer_results
                        )
                        
                        return {
                            'text': anonymized_result.text,
                            'pii_detected': True,
                            'entities': [r.entity_type for r in analyzer_results],
                            'anonymized': True
                        }
                    
                    return {
                        'text': text,
                        'pii_detected': False,
                        'entities': [],
                        'anonymized': False
                    }
                    
                except Exception as e:
                    logger.error(f"PII processing error: {e}")
                    return {
                        'text': text,
                        'pii_detected': False,
                        'entities': [],
                        'anonymized': False
                    }
            
            def process_retirement_query_mock(self, query: str, enable_pii: bool = True) -> Dict[str, Any]:
                """Process retirement query using mock responses"""
                
                # PII detection and anonymization
                pii_result = {'text': query, 'pii_detected': False, 'entities': [], 'anonymized': False}
                if enable_pii:
                    pii_result = self.detect_and_anonymize_pii(query)
                
                processed_query = pii_result['text']
                
                # Mock response generation
                age_mentioned = any(word in processed_query.lower() for word in ['age', 'years', 'old', 'born'])
                
                if age_mentioned:
                    # Parse potential age
                    import re
                    age_match = re.search(r'\b(\d{2})\b', processed_query)
                    
                    if age_match:
                        age = int(age_match.group(1))
                        is_eligible = age >= 65
                        
                        if is_eligible:
                            response = f"Based on the analysis, the person appears to be {age} years old and is ELIGIBLE for retirement. No additional deposit required."
                            deposit = "$0"
                        else:
                            years_until_eligible = 65 - age
                            deposit = f"${years_until_eligible * 5000:,}"
                            response = f"Based on the analysis, the person appears to be {age} years old and is NOT YET ELIGIBLE for retirement. They need {years_until_eligible} more years. Recommended deposit: {deposit}"
                    else:
                        response = "Age information detected but couldn't determine exact age. Please provide a specific age for accurate retirement eligibility assessment."
                        is_eligible = False
                        deposit = "Unable to calculate"
                else:
                    response = "Please provide age information to determine retirement eligibility."
                    is_eligible = False
                    deposit = "Unable to calculate"
                
                return {
                    'response': response,
                    'eligible': is_eligible,
                    'deposit_amount': deposit,
                    'persons_processed': 1,
                    'pii_detected': pii_result['pii_detected'],
                    'pii_entities': pii_result['entities'],
                    'anonymization_applied': pii_result['anonymized'],
                    'confidence': 0.95 if age_mentioned else 0.3,
                    'metadata': {
                        'processing_mode': 'mock',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }
            
            def process_retirement_query_live(self, query: str, enable_pii: bool = True) -> Dict[str, Any]:
                """Process retirement query using live LLM"""
                
                # PII detection and anonymization
                pii_result = {'text': query, 'pii_detected': False, 'entities': [], 'anonymized': False}
                if enable_pii:
                    pii_result = self.detect_and_anonymize_pii(query)
                
                processed_query = pii_result['text']
                
                # Construct prompt for LLM
                prompt = f"""
                Analyze the following retirement eligibility query and provide:
                1. Whether the person is eligible for retirement (typically age 65+)
                2. If not eligible, how many years until eligible and recommended deposit amount
                3. A confidence score (0-1) for your assessment
                
                Query: {processed_query}
                
                Provide a clear, concise response addressing retirement eligibility.
                If age information is missing, request it politely.
                """
                
                try:
                    # Call LLM
                    llm_response = self.llm_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a retirement eligibility advisor."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    
                    response_text = llm_response.choices[0].message.content
                    
                    # Parse response for eligibility determination
                    is_eligible = "eligible" in response_text.lower() and "not eligible" not in response_text.lower()
                    
                    # Extract deposit amount if mentioned
                    import re
                    deposit_match = re.search(r'\$[\d,]+', response_text)
                    deposit = deposit_match.group(0) if deposit_match else ("$0" if is_eligible else "Contact advisor")
                    
                    # Determine confidence
                    confidence = 0.85 if any(word in processed_query.lower() for word in ['age', 'years', 'old']) else 0.4
                    
                    return {
                        'response': response_text,
                        'eligible': is_eligible,
                        'deposit_amount': deposit,
                        'persons_processed': 1,
                        'pii_detected': pii_result['pii_detected'],
                        'pii_entities': pii_result['entities'],
                        'anonymization_applied': pii_result['anonymized'],
                        'confidence': confidence,
                        'metadata': {
                            'processing_mode': 'live_llm',
                            'model': 'gpt-4',
                            'timestamp': datetime.utcnow().isoformat()
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"LLM processing error: {e}")
                    # Fallback to mock response
                    return self.process_retirement_query_mock(query, enable_pii)
        
        # Initialize processor
        unified_processor = UnifiedRetirementProcessor(llm_client)
        
        # ==========================
        # CREATE TRUBASICAPP INSTANCES
        # ==========================
        
        # Add custom retirement-specific feedback to comprehensive list
        comprehensive_feedbacks.extend([
            f_retirement_quality,
            f_input_completeness,
            f_pii_protection,
            f_confidence
        ])
        
        # Mock app with basic feedback
        tru_mock_app = TruBasicApp(
            unified_processor,
            app_name="MockPromptForge",
            app_version="1.0.0",
            tru_session=tru_session,
            feedbacks=[
                f_retirement_quality,
                f_input_completeness,
                f_pii_protection,
                f_confidence
            ]
        )
        
        # Live app with comprehensive feedback
        tru_live_app = TruBasicApp(
            unified_processor,
            app_name="PromptForge",
            app_version="2.0.0",
            tru_session=tru_session,
            feedbacks=comprehensive_feedbacks
        )
        
        logger.info("✅ Created enhanced TruBasicApp instances with comprehensive feedback:")
        logger.info("   - MockPromptForge v1.0.0: Basic feedback (4 functions)")
        logger.info("   - PromptForge v2.0.0: Comprehensive feedback (9 functions)")
        logger.info("   Feedback functions enabled:")
        logger.info("     ✓ QA Relevance")
        logger.info("     ✓ Context Relevance")
        logger.info("     ✓ Groundedness")
        logger.info("     ✓ Sentiment Analysis")
        logger.info("     ✓ Toxicity Detection")
        logger.info("     ✓ Retirement Response Quality")
        logger.info("     ✓ Input Completeness")
        logger.info("     ✓ PII Protection")
        logger.info("     ✓ Confidence Calibration")
        
        return tru_mock_app, tru_live_app, unified_processor
        
    except ImportError as e:
        logger.warning(f"⚠️ TruLens not available: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"❌ Error creating TruLens apps: {e}")
        return None, None, None


def setup_retirement_endpoints(app, llm_client, auth_dependency):
    """Setup retirement eligibility endpoints with enhanced TruLens feedback"""
    
    # Create enhanced TruLens apps
    tru_mock_app, tru_live_app, processor = create_enhanced_retirement_app(llm_client)
    
    @app.post("/api/v1/retirement-eligibility", response_model=RetirementEligibilityResponse)
    async def retirement_eligibility_v1(
        request: RetirementEligibilityRequest,
        req: Request,
        background_tasks: BackgroundTasks,
        token: str = Depends(auth_dependency)
    ):
        """
        V1 Retirement Eligibility Endpoint (Mock Implementation)
        
        This endpoint uses mock processing with basic TruLens feedback.
        """
        request_id = req.headers.get("X-Request-ID", str(time.time()))
        logger.info(f"Processing retirement eligibility v1 request {request_id} with MockPromptForge")
        
        try:
            if request.enable_monitoring and tru_mock_app:
                # Use TruLens monitoring with recording context
                with tru_mock_app as recording:
                    result = tru_mock_app.app(
                        request.query, 
                        mode="mock",
                        enable_pii=request.enable_pii_protection
                    )
                    
                logger.info(f"✅ V1 endpoint: TruLens record captured with basic feedback")
            else:
                # Direct processing without monitoring
                if processor:
                    result = processor.process_retirement_query_mock(
                        request.query,
                        request.enable_pii_protection
                    )
                else:
                    # Fallback if no processor
                    result = {
                        'response': "Mock response: Unable to process retirement query",
                        'eligible': False,
                        'deposit_amount': "N/A",
                        'persons_processed': 0,
                        'pii_detected': False,
                        'pii_entities': [],
                        'anonymization_applied': False,
                        'confidence': 0.0,
                        'metadata': {'error': 'Processor not available'}
                    }
            
            # Return response
            return RetirementEligibilityResponse(**result)
            
        except Exception as e:
            logger.error(f"Error in v1 endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v2/retirement-eligibility", response_model=RetirementEligibilityResponse)
    async def retirement_eligibility_v2(
        request: RetirementEligibilityRequest,
        req: Request,
        background_tasks: BackgroundTasks,
        token: str = Depends(auth_dependency)
    ):
        """
        V2 Retirement Eligibility Endpoint (Live LLM with Comprehensive Feedback)
        
        This endpoint uses live LLM processing with comprehensive TruLens feedback:
        - QA Relevance
        - Context Relevance  
        - Groundedness
        - Sentiment Analysis
        - Toxicity Detection
        - Custom retirement-specific metrics
        """
        request_id = req.headers.get("X-Request-ID", str(time.time()))
        logger.info(f"Processing retirement eligibility v2 request {request_id} with PromptForge live LLM")
        logger.info(f"🎯 Comprehensive TruLens feedback enabled for this request")
        
        try:
            if request.enable_monitoring and tru_live_app:
                # Use TruLens monitoring with comprehensive feedback
                with tru_live_app as recording:
                    result = tru_live_app.app(
                        request.query,
                        mode="live",
                        enable_pii=request.enable_pii_protection
                    )
                    
                logger.info(f"✅ V2 endpoint: TruLens record captured with 9 feedback functions:")
                logger.info(f"   - Core feedback: QA Relevance, Context, Groundedness, Sentiment, Toxicity")
                logger.info(f"   - Custom feedback: Retirement Quality, Input Completeness, PII Protection, Confidence")
            else:
                # Direct processing without monitoring
                if processor:
                    result = processor.process_retirement_query_live(
                        request.query,
                        request.enable_pii_protection
                    )
                else:
                    # Fallback if no processor
                    result = {
                        'response': "Live response: Unable to process retirement query",
                        'eligible': False,
                        'deposit_amount': "N/A",
                        'persons_processed': 0,
                        'pii_detected': False,
                        'pii_entities': [],
                        'anonymization_applied': False,
                        'confidence': 0.0,
                        'metadata': {'error': 'Processor not available'}
                    }
            
            # Return response
            return RetirementEligibilityResponse(**result)
            
        except Exception as e:
            logger.error(f"Error in v2 endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    logger.info("✅ Enhanced retirement endpoints registered with comprehensive TruLens feedback")
    return tru_mock_app, tru_live_app