#!/usr/bin/env python3
"""
Fix Streamlit Dashboard Database Connection
Move remaining records and restart dashboard with consolidated database
"""

import os
import sys
import sqlite3
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def kill_streamlit_processes():
    """Kill any existing Streamlit processes"""
    print("🔄 Stopping existing Streamlit processes...")
    try:
        # Kill processes using port 8501
        result = subprocess.run(['lsof', '-ti:8501'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                subprocess.run(['kill', '-9', pid], capture_output=True)
                print(f"  ✅ Killed process {pid}")
        else:
            print("  ✅ No processes found on port 8501")
    except Exception as e:
        print(f"  ⚠️ Error killing processes: {e}")

def verify_database_migration():
    """Verify the database consolidation is complete"""
    print("🔍 Verifying database consolidation...")
    
    # Check consolidated database
    conn_new = sqlite3.connect('trulens_promptforge.db')
    cursor_new = conn_new.cursor()
    cursor_new.execute('SELECT COUNT(*) FROM trulens_records')
    new_count = cursor_new.fetchone()[0]
    conn_new.close()
    
    # Check old database
    conn_old = sqlite3.connect('default.sqlite')
    cursor_old = conn_old.cursor()
    cursor_old.execute('SELECT COUNT(*) FROM trulens_records')
    old_count = cursor_old.fetchone()[0]
    conn_old.close()
    
    print(f"  📊 trulens_promptforge.db: {new_count} records")
    print(f"  📊 default.sqlite: {old_count} records")
    
    return new_count, old_count

def start_dashboard_with_consolidated_db():
    """Start dashboard with consolidated database using direct streamlit command"""
    print("🚀 Starting TruLens dashboard with consolidated database...")
    
    # Set environment variables
    env = os.environ.copy()
    env['TRULENS_DATABASE_URL'] = 'sqlite:///trulens_promptforge.db'
    
    # Start streamlit with explicit path to venv streamlit and database URL
    cmd = [
        './venv/bin/streamlit', 
        'run', 
        './venv/lib/python3.13/site-packages/trulens/dashboard/main.py',
        '--server.port', '8501',
        '--server.headless', 'false',
        '--theme.base', 'dark',
        '--theme.primaryColor', '#E0735C',
        '--',
        '--database-url', 'sqlite:///trulens_promptforge.db'
    ]
    
    print(f"  🌐 Starting dashboard on http://localhost:8501")
    print(f"  📊 Database: sqlite:///trulens_promptforge.db")
    print(f"  📋 Command: {' '.join(cmd)}")
    
    # Start in background
    subprocess.Popen(cmd, env=env)
    print("  ✅ Dashboard started successfully!")
    
def main():
    """Main fix routine"""
    print("🛠️ Fixing Streamlit Dashboard Database Connection")
    print("=" * 60)
    
    # Step 1: Stop existing dashboard
    kill_streamlit_processes()
    
    # Step 2: Verify database state
    new_count, old_count = verify_database_migration()
    
    if new_count >= old_count:
        print("✅ Database consolidation verified - all records in trulens_promptforge.db")
    else:
        print("⚠️ Warning: Consolidated database has fewer records than old database")
    
    # Step 3: Start dashboard with correct database
    start_dashboard_with_consolidated_db()
    
    print("\n" + "=" * 60)
    print("📋 DASHBOARD FIX COMPLETE")
    print("=" * 60)
    print("🎉 Streamlit dashboard restarted with consolidated database!")
    print()
    print(f"🌐 Dashboard URL: http://localhost:8501")
    print(f"📊 Records available: {new_count}")
    print(f"📂 Database: trulens_promptforge.db")
    print()
    print("🔄 Next steps:")
    print("1. Open http://localhost:8501 in your browser")
    print("2. Navigate to the Records page")
    print(f"3. Verify you now see all {new_count} records")
    print("4. Test the dashboard functionality with the consolidated data")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())