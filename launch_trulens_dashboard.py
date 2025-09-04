#!/usr/bin/env ./venv/bin/python
"""
TruLens Dashboard Launcher for Browser Access
Starts the native TruLens dashboard on port 8501
"""

import sys
import os
from pathlib import Path

# Add venv/bin to PATH so TruLens can find streamlit
current_dir = Path(__file__).parent
venv_bin = current_dir / "venv" / "bin"
if str(venv_bin) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"

def launch_trulens_dashboard():
    """Launch TruLens dashboard for browser access"""
    print("🚀 Starting TruLens Dashboard...")
    print("=" * 50)
    
    try:
        from trulens.core import TruSession
        from trulens.dashboard.run import run_dashboard
        
        # Initialize TruLens session with consolidated database
        print("🔧 Initializing TruLens session...")
        database_url = 'sqlite:///trulens_promptforge.db'
        os.environ['TRULENS_DATABASE_URL'] = database_url
        session = TruSession(database_url=database_url)
        
        print("✅ TruLens session created successfully")
        print("📊 Database: SQLite (trulens_promptforge.db)")
        
        # Verify record count
        try:
            result = session.get_records_and_feedback(app_name="promptforge")
            if isinstance(result, tuple):
                records_df, feedback_columns = result
            else:
                records_df = result
                feedback_columns = []
            print(f"📋 Records available: {len(records_df)} records")
        except Exception as e:
            print(f"⚠️ Could not verify records: {e}")
        
        print("")
        print("🌐 Starting dashboard...")
        print("   URL: http://localhost:8501")
        print("   Press Ctrl+C to stop")
        print("")
        
        # Start the dashboard using the new API
        run_dashboard(session, port=8501)
        
    except ImportError as e:
        print(f"❌ TruLens import error: {e}")
        print("💡 Try installing: pip install trulens-core")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    launch_trulens_dashboard()