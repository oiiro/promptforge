#!/usr/bin/env python3
"""
TruLens Dashboard Launcher for Browser Access
Starts the native TruLens dashboard on port 8501
"""

import sys
import os
from pathlib import Path

def launch_trulens_dashboard():
    """Launch TruLens dashboard for browser access"""
    print("🚀 Starting TruLens Dashboard...")
    print("=" * 50)
    
    try:
        from trulens.core import TruSession
        
        # Initialize TruLens session
        print("🔧 Initializing TruLens session...")
        session = TruSession()
        
        print("✅ TruLens session created successfully")
        print(f"📊 Database: {session.connector.database_url}")
        print("")
        print("🌐 Starting dashboard...")
        print("   URL: http://localhost:8501")
        print("   Press Ctrl+C to stop")
        print("")
        
        # Start the dashboard
        session.run_dashboard(port=8501)
        
    except ImportError as e:
        print(f"❌ TruLens import error: {e}")
        print("💡 Try installing: pip install trulens-core")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    launch_trulens_dashboard()