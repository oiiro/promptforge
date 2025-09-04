#!/usr/bin/env python3
"""
Simple TruLens LLM Wrapper - Focus on Data Collection

This creates a minimal working TruBasicApp that captures LLM inputs and outputs properly.
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

def create_simple_trulens_app():
    """Create a simple TruLens app that captures LLM data properly"""
    
    try:
        from trulens.core import TruSession
        from trulens.apps.basic import TruBasicApp
        from orchestration.llm_client import LLMClient
        
        # Initialize TruLens session
        os.environ['TRULENS_DATABASE_URL'] = 'sqlite:///trulens_promptforge.db'
        tru_session = TruSession(database_url='sqlite:///trulens_promptforge.db')
        logger.info("✅ TruLens session initialized")
        
        # Create LLM client
        llm_client = LLMClient(provider="mock")  # Use mock for reliable testing
        logger.info("✅ LLM client initialized")
        
        # Create simple app class
        class SimpleCapitalApp:
            def __init__(self, llm_client):
                self.llm_client = llm_client
                
            def find_capital(self, country: str) -> str:
                """Main method that TruLens will instrument"""
                logger.info(f"🎯 Processing request for: {country}")
                
                # This is the actual LLM call that should be captured
                response = self.llm_client.generate(country)
                
                logger.info(f"📤 Generated response: {response[:100]}...")
                return response
        
        # Create app instance
        capital_app = SimpleCapitalApp(llm_client)
        
        # Wrap with TruLens (minimal version - no feedback functions for now)
        tru_app = TruBasicApp(
            capital_app,
            app_name="promptforge_simple",
            app_version="1.0.0"
        )
        
        logger.info("✅ TruBasicApp created successfully")
        
        return {
            "tru_session": tru_session,
            "tru_app": tru_app,
            "capital_app": capital_app
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create TruLens app: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_simple_app():
    """Test the simple TruLens app"""
    
    logger.info("🧪 Testing simple TruLens app...")
    
    components = create_simple_trulens_app()
    if not components:
        return False
    
    tru_app = components["tru_app"]
    
    try:
        # Make test calls
        test_countries = ["Brazil", "Canada"]
        
        for country in test_countries:
            logger.info(f"🎯 Testing: {country}")
            
            # Use TruLens recording context
            with tru_app as recording:
                response = tru_app.app.find_capital(country)
                
            logger.info(f"📤 Response received: {len(response)} characters")
            time.sleep(1)
        
        # Check database for records
        logger.info("🔍 Checking database for new records...")
        
        import sqlite3
        conn = sqlite3.connect('trulens_promptforge.db')
        cursor = conn.cursor()
        
        # Check records for the simple app
        cursor.execute("""
            SELECT r.record_id, r.input, r.output, 
                   length(r.record_json) as json_size,
                   datetime(r.ts, 'unixepoch') as timestamp,
                   a.app_name
            FROM trulens_records r
            JOIN trulens_apps a ON r.app_id = a.app_id
            WHERE a.app_name = 'promptforge_simple'
            ORDER BY r.ts DESC LIMIT 2
        """)
        
        records = cursor.fetchall()
        conn.close()
        
        if records:
            logger.info(f"📊 Found {len(records)} records for simple app:")
            
            for record_id, input_data, output_data, json_size, timestamp, app_name in records:
                logger.info(f"   📋 {record_id}")
                logger.info(f"      Input: {'✅ PRESENT' if input_data else '❌ MISSING'}")
                logger.info(f"      Output: {'✅ PRESENT' if output_data else '❌ MISSING'}")
                logger.info(f"      JSON size: {json_size} bytes")
                logger.info(f"      Time: {timestamp}")
                
                if input_data and output_data:
                    logger.info("      ✅ Complete data captured!")
                    return True
                else:
                    logger.warning("      ⚠️ Missing input or output data")
                    
        logger.error("❌ No complete records found")
        return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution"""
    logger.info("🛠️ Simple TruLens LLM Wrapper Test")
    logger.info("=" * 45)
    
    success = test_simple_app()
    
    if success:
        logger.info("\n" + "=" * 45)
        logger.info("✅ SIMPLE TRULENS APP WORKS!")
        logger.info("=" * 45)
        logger.info("🎉 TruLens is capturing LLM input/output data!")
        logger.info("")
        logger.info("🔄 This proves the concept. Next steps:")
        logger.info("1. Apply this pattern to your API endpoints")
        logger.info("2. Replace manual record creation with TruBasicApp")
        logger.info("3. TruLens will capture rich interaction data")
        return 0
    else:
        logger.error("\n" + "=" * 45)
        logger.error("❌ SIMPLE TRULENS APP FAILED")
        logger.error("=" * 45)
        return 1

if __name__ == "__main__":
    exit(main())