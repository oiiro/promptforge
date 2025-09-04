#!/usr/bin/env python3
"""
Start TruLens Dashboard with Consolidated Database
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Set environment to use consolidated database
os.environ['TRULENS_DATABASE_URL'] = 'sqlite:///trulens_promptforge.db'

def start_dashboard():
    """Start TruLens dashboard with consolidated database"""
    print("🚀 Starting TruLens Dashboard with consolidated database...")
    print("📊 Database: sqlite:///trulens_promptforge.db")
    
    try:
        from trulens.core import TruSession
        from trulens.dashboard.run import run_dashboard
        
        # Create session with consolidated database
        session = TruSession(database_url='sqlite:///trulens_promptforge.db')
        print(f"✅ Connected to TruLens database")
        
        # Check record count
        try:
            result = session.get_records_and_feedback(app_name="promptforge")
            if isinstance(result, tuple):
                records_df, feedback_columns = result
            else:
                records_df = result
                feedback_columns = []
            
            print(f"✅ Found {len(records_df)} records for promptforge app")
        except Exception as e:
            print(f"⚠️ Warning: Could not get record count: {e}")
        
        # Start dashboard
        print("🌐 Starting dashboard on http://localhost:8501")
        print("📋 Dashboard will show all consolidated records")
        run_dashboard(
            session=session,
            port=8501,
            force=True,
            _dev=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(start_dashboard())