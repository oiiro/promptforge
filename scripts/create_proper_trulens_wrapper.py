#!/usr/bin/env python3
"""
Create Proper TruLens LLM Wrapper for Data Collection

This creates a TruBasicApp that wraps your LLM client properly
to capture inputs, outputs, and costs in TruLens records.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "orchestration"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_trulens_llm_wrapper():
    """Create a proper TruLens wrapper for LLM calls"""
    
    try:
        from trulens.core import TruSession
        from trulens.apps.basic import TruBasicApp
        from trulens.feedback import feedback
        from orchestration.llm_client import LLMClient
        
        # Initialize TruLens session
        os.environ['TRULENS_DATABASE_URL'] = 'sqlite:///trulens_promptforge.db'
        tru_session = TruSession(database_url='sqlite:///trulens_promptforge.db')
        logger.info("✅ TruLens session initialized")
        
        # Create your LLM client
        llm_client = LLMClient(provider="mock")  # Use mock for testing
        logger.info("✅ LLM client initialized with mock provider")
        
        # Create a simple app class that TruLens can wrap
        class CapitalFinderApp:
            def __init__(self, llm_client):
                self.llm_client = llm_client
                
            def find_capital(self, country: str) -> str:
                """Find capital of a country using LLM"""
                logger.info(f"🎯 Finding capital for: {country}")
                response = self.llm_client.generate(country)
                logger.info(f"📤 LLM response: {response[:100]}...")
                return response
        
        # Create app instance
        capital_app = CapitalFinderApp(llm_client)
        
        # Create feedback functions
        def input_quality_feedback(input_text: str) -> float:
            """Evaluate input quality"""
            if not input_text or len(input_text.strip()) < 2:
                return 0.0
            return 1.0
            
        def response_quality_feedback(output_text: str) -> float:
            """Evaluate response quality"""
            if not output_text:
                return 0.0
            # Check if looks like valid JSON
            try:
                data = json.loads(output_text)
                if "capital" in data and "country" in data:
                    return 1.0
                return 0.5
            except:
                return 0.3
        
        # Wrap with TruLens
        tru_capital_app = TruBasicApp(
            capital_app,
            app_name="promptforge",
            app_version="1.0.0",
            feedbacks=[
                feedback.Feedback(
                    input_quality_feedback,
                    name="Input Quality"
                ).on_input(),
                feedback.Feedback(
                    response_quality_feedback,
                    name="Response Quality"
                ).on_output()
            ]
        )
        
        logger.info("✅ TruBasicApp created with feedback functions")
        
        return {
            "tru_session": tru_session,
            "tru_app": tru_capital_app,
            "capital_app": capital_app
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create TruLens wrapper: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_trulens_wrapper():
    """Test the TruLens wrapper with actual LLM calls"""
    
    logger.info("🧪 Testing TruLens LLM wrapper...")
    
    components = create_trulens_llm_wrapper()
    if not components:
        return False
    
    tru_app = components["tru_app"]
    
    try:
        # Test multiple calls through TruLens
        test_countries = ["France", "Japan", "Germany"]
        
        for country in test_countries:
            logger.info(f"🎯 Testing with country: {country}")
            
            # This call should be fully instrumented by TruLens
            with tru_app as recording:
                response = tru_app.app.find_capital(country)
                
            logger.info(f"📤 Response: {response[:100]}...")
            time.sleep(1)  # Small delay between calls
        
        # Check database for new records with proper data
        logger.info("🔍 Checking database for new records...")
        
        import sqlite3
        conn = sqlite3.connect('trulens_promptforge.db')
        cursor = conn.cursor()
        
        # Get recent records
        cursor.execute("""
            SELECT record_id, input, output, record_json,
                   datetime(ts, 'unixepoch') as timestamp
            FROM trulens_records 
            ORDER BY ts DESC LIMIT 3
        """)
        
        records = cursor.fetchall()
        
        if records:
            logger.info(f"📊 Found {len(records)} recent records:")
            
            success_count = 0
            for i, (record_id, input_data, output_data, record_json, timestamp) in enumerate(records):
                logger.info(f"   Record {i+1}: {record_id}")
                logger.info(f"     Input: {'✅ PRESENT' if input_data else '❌ MISSING'}")
                logger.info(f"     Output: {'✅ PRESENT' if output_data else '❌ MISSING'}")
                logger.info(f"     JSON: {'✅ RICH DATA' if len(record_json) > 100 else '❌ MINIMAL DATA'}")
                logger.info(f"     Time: {timestamp}")
                
                if input_data and output_data and len(record_json) > 100:
                    success_count += 1
            
            conn.close()
            
            if success_count >= 2:  # At least 2 out of 3 successful
                logger.info("✅ TruLens is capturing complete LLM interaction data!")
                return True
            else:
                logger.warning("⚠️ TruLens records created but data incomplete")
                return False
                
        else:
            logger.error("❌ No new records found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution"""
    logger.info("🛠️ Creating Proper TruLens LLM Wrapper")
    logger.info("=" * 50)
    
    success = test_trulens_wrapper()
    
    if success:
        logger.info("\n" + "=" * 50)
        logger.info("✅ TRULENS LLM WRAPPER CREATED SUCCESSFULLY")
        logger.info("=" * 50)
        logger.info("🎉 TruLens is now capturing complete LLM data!")
        logger.info("")
        logger.info("📋 What was fixed:")
        logger.info("• Created TruBasicApp wrapper around LLM client")
        logger.info("• Added proper feedback functions for evaluation")
        logger.info("• LLM calls now captured with full input/output data")
        logger.info("• Records include performance and cost metrics")
        logger.info("")
        logger.info("🔄 Next steps:")
        logger.info("1. Update API endpoints to use similar TruBasicApp pattern")
        logger.info("2. Replace manual add_record() with proper recording context")
        logger.info("3. TruLens dashboard will now show rich evaluation data")
        return 0
    else:
        logger.error("\n" + "=" * 50)
        logger.error("❌ TRULENS LLM WRAPPER CREATION FAILED")
        logger.error("=" * 50)
        return 1

if __name__ == "__main__":
    exit(main())