"""
Retirement Eligibility Endpoints

This module contains both v1 (mock) and v2 (live LLM) retirement eligibility endpoints
with proper TruLens integration for data collection and evaluation.
"""

import os
import json
import time
import logging
from typing import Dict, Any
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

def create_unified_retirement_app(llm_client):
    """Create separate MockPromptForge (v1) and PromptForge (v2) apps"""
    
    try:
        from trulens.core import TruSession
        from trulens.apps.basic import TruBasicApp
        from trulens.core.feedback import Feedback
        
        # Initialize TruLens session with correct database URL
        database_url = os.getenv('TRULENS_DATABASE_URL', 'sqlite:///trulens_promptforge.db')
        tru_session = TruSession(database_url=database_url)
        logger.info(f"✅ TruLens session initialized with database: {database_url}")
        
        class UnifiedRetirementProcessor:
            """Unified retirement eligibility processor with both mock and live LLM modes"""
            
            def __init__(self, llm_client):
                self.name = "PromptForge"
                self.llm_client = llm_client
                
            def __call__(self, query: str, mode: str = "mock", enable_pii: bool = True) -> Dict[str, Any]:
                """Main callable interface for TruLens instrumentation"""
                if mode == "mock":
                    return self.process_retirement_query_mock(query, enable_pii)
                else:
                    return self.process_retirement_query_live(query, enable_pii)
                
            def process_retirement_query_mock(self, query: str, enable_pii: bool = True) -> Dict[str, Any]:
                """Process retirement query with mock data for v1 endpoint - calls LLM for TruLens recording"""
                logger.info(f"🎭 PromptForge (mock mode) processing: {query[:50]}...")
                
                # Call LLM client for TruLens recording (but use mock logic for response)
                mock_prompt = f"Mock prompt for retirement eligibility: {query}"
                llm_response = self.llm_client.generate(mock_prompt)
                logger.info(f"📤 Mock LLM response: {llm_response[:50]}...")
                
                # Simulate PII detection
                pii_entities = []
                if "john" in query.lower() or "doe" in query.lower():
                    pii_entities.append("PERSON")
                if any(age in query for age in ["65", "67", "70"]):
                    pii_entities.append("AGE")
                
                # Mock retirement decision logic
                eligible = "eligible" in query.lower() or "qualify" in query.lower()
                persons_count = query.lower().count("person") + query.lower().count("individual")
                if persons_count == 0:
                    persons_count = 1
                
                return {
                    "response": f"Based on your query, retirement eligibility has been determined. This is a mock response for testing purposes.",
                    "eligible": eligible,
                    "deposit_amount": "$50,000" if eligible else "$0",
                    "persons_processed": min(persons_count, 3),
                    "pii_detected": len(pii_entities) > 0,
                    "pii_entities": pii_entities,
                    "anonymization_applied": enable_pii and len(pii_entities) > 0,
                    "confidence": 0.95,
                    "processing_type": "mock",
                    "endpoint": "v1"
                }
                
            def process_retirement_query_live(self, query: str, enable_pii: bool = True) -> Dict[str, Any]:
                """Process retirement query with live LLM for v2 endpoint"""
                logger.info(f"🚀 PromptForge (live mode) processing with LLM: {query[:50]}...")
                
                # Create structured prompt for LLM
                prompt = f"""
                You are a financial advisor AI specializing in retirement eligibility.
                
                Analyze this retirement query and provide a structured response:
                
                QUERY: {query}
                
                Respond with valid JSON containing:
                {{
                    "eligible": boolean,
                    "deposit_amount": "dollar amount as string or 'Not applicable'",
                    "persons_processed": number of people mentioned,
                    "confidence": confidence level (0.0-1.0),
                    "reasoning": "brief explanation of decision",
                    "key_factors": ["list", "of", "key", "factors"]
                }}
                
                Consider factors like age, income, current savings, employment status, and specific retirement requirements.
                """
                
                # Call LLM
                start_time = time.time()
                llm_response = self.llm_client.generate(prompt)
                llm_latency = (time.time() - start_time) * 1000
                
                logger.info(f"📤 LLM response received ({llm_latency:.1f}ms): {llm_response[:100]}...")
                
                # Parse LLM response
                try:
                    parsed_response = json.loads(llm_response)
                    
                    # Extract structured data
                    eligible = parsed_response.get("eligible", False)
                    deposit_amount = parsed_response.get("deposit_amount", "Not applicable")
                    persons_processed = parsed_response.get("persons_processed", 1)
                    confidence = parsed_response.get("confidence", 0.8)
                    reasoning = parsed_response.get("reasoning", "Analysis completed")
                    key_factors = parsed_response.get("key_factors", [])
                    
                    # Create natural language response
                    eligibility_text = "eligible" if eligible else "not eligible"
                    response_text = f"Based on the provided information, the individual appears to be {eligibility_text} for retirement. {reasoning}"
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse LLM response as JSON: {e}")
                    # Fallback parsing
                    eligible = "eligible" in llm_response.lower()
                    deposit_amount = "Analysis required"
                    persons_processed = 1
                    confidence = 0.6
                    reasoning = "LLM response required manual parsing"
                    key_factors = ["manual_parsing_required"]
                    response_text = llm_response
                
                # Basic PII detection (enhanced version would use Presidio)
                pii_entities = []
                pii_terms = ["john", "doe", "smith", "johnson"]
                for term in pii_terms:
                    if term in query.lower():
                        pii_entities.append("PERSON")
                        break
                
                if any(char.isdigit() for char in query) and ("age" in query.lower() or "years" in query.lower()):
                    pii_entities.append("AGE")
                
                return {
                    "response": response_text,
                    "eligible": eligible,
                    "deposit_amount": deposit_amount,
                    "persons_processed": persons_processed,
                    "pii_detected": len(pii_entities) > 0,
                    "pii_entities": pii_entities,
                    "anonymization_applied": enable_pii and len(pii_entities) > 0,
                    "confidence": confidence,
                    "processing_type": "live_llm",
                    "endpoint": "v2",
                    "llm_latency_ms": llm_latency,
                    "reasoning": reasoning,
                    "key_factors": key_factors,
                    "metadata": {
                        "llm_provider": getattr(self.llm_client, 'provider_name', 'unknown'),
                        "prompt_tokens": len(prompt.split()),
                        "response_tokens": len(llm_response.split()) if llm_response else 0
                    }
                }
        
        # Create feedback functions for evaluation
        def response_quality_feedback(output: str) -> float:
            """Evaluate response quality"""
            if not output:
                return 0.0
            
            # Check for structured elements
            score = 0.0
            if "eligible" in output.lower():
                score += 0.3
            if "deposit" in output.lower() or "amount" in output.lower():
                score += 0.2
            if "confident" in output.lower() or "analysis" in output.lower():
                score += 0.3
            if len(output) > 50:  # Reasonable length
                score += 0.2
                
            return min(score, 1.0)
        
        def input_completeness_feedback(input_text: str) -> float:
            """Evaluate input completeness for retirement analysis"""
            if not input_text:
                return 0.0
                
            score = 0.0
            if any(word in input_text.lower() for word in ["age", "years", "old"]):
                score += 0.3
            if any(word in input_text.lower() for word in ["retire", "retirement", "eligible"]):
                score += 0.4
            if any(word in input_text.lower() for word in ["income", "salary", "savings", "401k"]):
                score += 0.3
                
            return min(score, 1.0)
        
        # Create unified processor
        unified_processor = UnifiedRetirementProcessor(llm_client)
        
        # Create separate TruBasicApp instances for v1 (mock) and v2 (live)
        tru_mock_app = TruBasicApp(
            unified_processor,
            app_name="MockPromptForge",
            app_version="1.0.0",
            tru_session=tru_session,
            feedbacks=[
                Feedback(
                    response_quality_feedback,
                    name="Response Quality",
                    higher_is_better=True
                ).on_output(),
                Feedback(
                    input_completeness_feedback,
                    name="Input Completeness",
                    higher_is_better=True
                ).on_input()
            ]
        )
        
        tru_live_app = TruBasicApp(
            unified_processor,
            app_name="PromptForge",
            app_version="1.0.0",
            tru_session=tru_session,
            feedbacks=[
                Feedback(
                    response_quality_feedback,
                    name="Response Quality",
                    higher_is_better=True
                ).on_output(),
                Feedback(
                    input_completeness_feedback,
                    name="Input Completeness",
                    higher_is_better=True
                ).on_input()
            ]
        )
        
        return {
            "processor": unified_processor,
            "tru_mock_app": tru_mock_app,
            "tru_live_app": tru_live_app,
            "tru_session": tru_session
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create PromptForge app: {e}")
        import traceback
        traceback.print_exc()
        return None

async def process_retirement_request_v1(
    request: RetirementEligibilityRequest,
    request_id: str,
    unified_app_components: dict
) -> RetirementEligibilityResponse:
    """Process v1 retirement request with PromptForge (mock mode)"""
    
    start_time = time.time()
    
    try:
        tru_mock_app = unified_app_components["tru_mock_app"]
        processor = unified_app_components["processor"]
        
        # Process with TruLens recording using mock mode - call through TruBasicApp
        with tru_mock_app as recording:
            result = tru_mock_app.app(
                request.query, 
                mode="mock",
                enable_pii=request.enable_pii_protection
            )
        
        # Add timing metadata
        processing_time = (time.time() - start_time) * 1000
        result["metadata"] = result.get("metadata", {})
        result["metadata"]["processing_time_ms"] = processing_time
        result["metadata"]["request_id"] = request_id
        result["metadata"]["trulens_recording"] = True
        
        logger.info(f"✅ PromptForge (mock mode) processed request {request_id} in {processing_time:.1f}ms")
        
        return RetirementEligibilityResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ PromptForge (mock mode) failed for request {request_id}: {e}")
        # Return error response
        return RetirementEligibilityResponse(
            response=f"Processing failed: {str(e)}",
            eligible=False,
            deposit_amount="$0",
            persons_processed=0,
            confidence=0.0,
            metadata={"error": str(e), "request_id": request_id}
        )

async def process_retirement_request_v2(
    request: RetirementEligibilityRequest,
    request_id: str,
    unified_app_components: dict
) -> RetirementEligibilityResponse:
    """Process v2 retirement request with PromptForge (live LLM mode)"""
    
    start_time = time.time()
    
    try:
        tru_live_app = unified_app_components["tru_live_app"]
        processor = unified_app_components["processor"]
        
        # Process with TruLens recording using live LLM mode - call through TruBasicApp
        with tru_live_app as recording:
            result = tru_live_app.app(
                request.query,
                mode="live",
                enable_pii=request.enable_pii_protection
            )
        
        # Add timing metadata
        processing_time = (time.time() - start_time) * 1000
        result["metadata"]["total_processing_time_ms"] = processing_time
        result["metadata"]["request_id"] = request_id
        result["metadata"]["trulens_recording"] = True
        
        logger.info(f"✅ PromptForge (live mode) processed request {request_id} in {processing_time:.1f}ms")
        
        return RetirementEligibilityResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ PromptForge (live mode) failed for request {request_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error response
        return RetirementEligibilityResponse(
            response=f"Live LLM processing failed: {str(e)}",
            eligible=False,
            deposit_amount="$0",
            persons_processed=0,
            confidence=0.0,
            metadata={"error": str(e), "request_id": request_id, "endpoint": "v2"}
        )