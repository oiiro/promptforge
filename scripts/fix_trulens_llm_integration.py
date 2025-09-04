#!/usr/bin/env python3
"""
Fix TruLens LLM Integration for Proper Data Collection

This script creates a proper TruVirtual app wrapper around your LLM client
to capture actual LLM inputs, outputs, costs, and performance metrics.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "orchestration"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_trulens_wrapped_llm_client():
    """Create a TruLens-wrapped LLM client for proper data collection"""
    
    try:
        from trulens.core import TruSession
        from trulens.apps.virtual import VirtualApp
        from trulens.core.app import App
        from trulens.feedback import feedback
        from trulens.providers.openai import OpenAI as TruOpenAI
        from orchestration.llm_client import LLMClient
        
        # Initialize TruLens session with consolidated database
        os.environ['TRULENS_DATABASE_URL'] = 'sqlite:///trulens_promptforge.db'
        tru_session = TruSession(database_url='sqlite:///trulens_promptforge.db')
        logger.info("✅ TruLens session initialized with consolidated database")
        
        # Get existing app_id from database directly
        import sqlite3
        conn = sqlite3.connect('trulens_promptforge.db')
        cursor = conn.cursor()
        cursor.execute("SELECT app_id FROM trulens_apps WHERE app_name = 'promptforge' LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        
        if result:
            target_app_id = result[0]
            logger.info(f"✅ Found existing app with ID: {target_app_id}")
        else:
            logger.error("❌ No existing promptforge app found in database.")
            return None
        
        # Initialize your actual LLM client
        llm_client = LLMClient(provider="openai")  # or your preferred provider
        logger.info("✅ LLM client initialized")
        
        # Create wrapper functions that TruLens can instrument
        def llm_generate_wrapper(country: str, **kwargs) -> str:
            """Wrapper function for LLM generation that TruLens can instrument"""
            logger.info(f"🎯 LLM Generate called with country: {country}")
            response = llm_client.generate(country, **kwargs)
            logger.info(f"📤 LLM Response generated: {response[:100]}...")
            return response
        
        # Create TruVirtual app with proper instrumentation
        virtual_app = VirtualApp(
            app_name="promptforge",
            app_version="1.0.0"
        )
        
        # Add the main callable method
        virtual_app = virtual_app.with_wrapped_method(
            component_name="llm_client",
            callable=llm_generate_wrapper,
            method_name="generate"
        )
        
        logger.info("✅ TruVirtual app created with LLM wrapper")
        
        # Create TruLens feedback functions for evaluation
        def input_quality_feedback(input_text: str) -> float:
            """Evaluate input quality"""
            if not input_text or len(input_text.strip()) < 3:
                return 0.0
            if len(input_text) > 1000:
                return 0.5  # Too long
            return 1.0
        
        def response_relevance_feedback(input_text: str, output_text: str) -> float:
            """Evaluate response relevance"""
            if not output_text:
                return 0.0
            
            # Check if response contains expected JSON structure
            if "capital" in output_text.lower() and "{" in output_text:
                return 1.0
            return 0.5
        
        def response_completeness_feedback(output_text: str) -> float:
            """Evaluate response completeness"""
            if not output_text:
                return 0.0
            
            required_fields = ["capital", "country", "confidence"]
            found_fields = sum(1 for field in required_fields if field in output_text.lower())
            return found_fields / len(required_fields)
        
        # Create TruApp with feedback functions
        try:
            from trulens.apps.basic import TruBasicApp
            
            tru_app = TruBasicApp(
                app=virtual_app,
                app_name="promptforge",
                app_version="1.0.0",
                feedbacks=[
                    feedback.Feedback(
                        input_quality_feedback,
                        name="Input Quality",
                        higher_is_better=True
                    ).on_input(),
                    feedback.Feedback(
                        response_relevance_feedback,
                        name="Response Relevance",
                        higher_is_better=True
                    ).on_input_output(),
                    feedback.Feedback(
                        response_completeness_feedback,
                        name="Response Completeness",
                        higher_is_better=True
                    ).on_output()
                ],
                app_id=target_app_id  # Use existing app_id for consistency
            )
            
            # Register with session
            tru_session.register(tru_app)
            logger.info("✅ TruApp registered with comprehensive feedback functions")
            
        except Exception as tru_app_error:
            logger.warning(f"Failed to create TruApp: {tru_app_error}")
            # Fallback to basic virtual app
            tru_app = virtual_app
        
        return {
            "tru_session": tru_session,
            "tru_app": tru_app,
            "llm_client": llm_client,
            "wrapped_generate": llm_generate_wrapper
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create TruLens wrapped LLM client: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_trulens_wrapped_client():
    """Test the TruLens wrapped client to ensure proper data collection"""
    
    logger.info("🧪 Testing TruLens wrapped LLM client...")
    
    components = create_trulens_wrapped_llm_client()
    if not components:
        logger.error("❌ Failed to create wrapped client")
        return False
    
    tru_session = components["tru_session"]
    tru_app = components["tru_app"]
    wrapped_generate = components["wrapped_generate"]
    
    try:
        # Test call through TruLens wrapper
        logger.info("🎯 Making test LLM call through TruLens wrapper...")
        
        # This should be instrumented by TruLens and capture full data
        with tru_app as recording:
            response = wrapped_generate("France")
        
        logger.info(f"📤 Test response: {response}")
        
        # Check if record was created with proper data
        import time
        time.sleep(2)  # Allow time for record processing
        
        # Query recent records
        import sqlite3
        conn = sqlite3.connect('trulens_promptforge.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT record_id, input, output, 
                   CASE WHEN record_json = '{}' OR record_json = '' THEN 'EMPTY' ELSE 'HAS_DATA' END as record_status,
                   datetime(ts, 'unixepoch') as timestamp
            FROM trulens_records 
            ORDER BY ts DESC LIMIT 1
        """)
        
        latest_record = cursor.fetchone()
        if latest_record:
            record_id, input_data, output_data, record_status, timestamp = latest_record
            logger.info(f"📊 Latest record: {record_id}")
            logger.info(f"   Input: {'CAPTURED' if input_data else 'MISSING'}")
            logger.info(f"   Output: {'CAPTURED' if output_data else 'MISSING'}")
            logger.info(f"   Record JSON: {record_status}")
            logger.info(f"   Timestamp: {timestamp}")
            
            if input_data and output_data and record_status == 'HAS_DATA':
                logger.info("✅ TruLens is now capturing complete LLM data!")
                return True
            else:
                logger.warning("⚠️ TruLens record created but missing data")
                return False
        else:
            logger.error("❌ No records found in database")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    """Main execution function"""
    logger.info("🛠️ TruLens LLM Integration Fix")
    logger.info("=" * 50)
    
    # Test the wrapped client
    success = test_trulens_wrapped_client()
    
    if success:
        logger.info("\n" + "=" * 50)
        logger.info("✅ TRULENS LLM INTEGRATION FIX COMPLETE")
        logger.info("=" * 50)
        logger.info("🎉 TruLens is now properly capturing LLM data!")
        logger.info("")
        logger.info("🔄 Next steps:")
        logger.info("1. Update your API endpoints to use the wrapped LLM client")
        logger.info("2. Replace manual add_record() calls with proper TruApp recording")
        logger.info("3. Test API calls to see rich data in TruLens dashboard")
        logger.info("4. Dashboard will show complete inputs, outputs, and metrics")
        return 0
    else:
        logger.error("\n" + "=" * 50)
        logger.error("❌ TRULENS LLM INTEGRATION FIX FAILED")
        logger.error("=" * 50)
        logger.error("⚠️ Manual intervention required")
        return 1

if __name__ == "__main__":
    exit(main())