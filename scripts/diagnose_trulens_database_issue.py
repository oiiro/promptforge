#!/usr/bin/env python3
"""
TruLens Database Issue Diagnostic Script
Diagnose why API creates records but dashboard doesn't show them
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def check_database_files():
    """Check all SQLite databases in the project"""
    print("🔍 Checking database files...")
    
    databases = [
        'default.sqlite',
        'trulens_promptforge.db',
        'trulens.db'
    ]
    
    for db_name in databases:
        if os.path.exists(db_name):
            print(f"\n📊 Database: {db_name}")
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()
            
            try:
                # Check records table
                cursor.execute('SELECT COUNT(*) FROM trulens_records')
                record_count = cursor.fetchone()[0]
                print(f"  📝 Records: {record_count}")
                
                if record_count > 0:
                    # Get sample records with more details
                    cursor.execute('''
                        SELECT app_id, record_id, ts, 
                               SUBSTR(input, 1, 100) as input_preview,
                               SUBSTR(output, 1, 100) as output_preview
                        FROM trulens_records 
                        ORDER BY ts DESC 
                        LIMIT 3
                    ''')
                    records = cursor.fetchall()
                    
                    for i, (app_id, record_id, ts, input_preview, output_preview) in enumerate(records):
                        print(f"    Record {i+1}:")
                        print(f"      app_id: {app_id}")
                        print(f"      record_id: {record_id}")
                        print(f"      timestamp: {ts}")
                        print(f"      input preview: {input_preview}...")
                
                # Check apps table
                cursor.execute('SELECT COUNT(*) FROM trulens_apps')
                app_count = cursor.fetchone()[0]
                print(f"  📱 Apps: {app_count}")
                
                if app_count > 0:
                    cursor.execute('SELECT app_id, app_name FROM trulens_apps')
                    apps = cursor.fetchall()
                    for app_id, app_name in apps:
                        print(f"    App: '{app_name}' (id: {app_id})")
                        
                        # Check if records exist for this app
                        cursor.execute('SELECT COUNT(*) FROM trulens_records WHERE app_id = ?', (app_id,))
                        app_record_count = cursor.fetchone()[0]
                        print(f"         Records for this app: {app_record_count}")
                
            except Exception as e:
                print(f"  ❌ Error reading database: {e}")
            
            conn.close()
        else:
            print(f"❌ Database {db_name} does not exist")

def check_environment_config():
    """Check environment configuration"""
    print(f"\n⚙️ Environment Configuration:")
    
    # Read .env file
    env_vars = {}
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
                    os.environ[key] = value
    except Exception as e:
        print(f"❌ Error reading .env: {e}")
    
    trulens_url = env_vars.get('TRULENS_DATABASE_URL', 'sqlite:///default.sqlite')
    print(f"  TRULENS_DATABASE_URL: {trulens_url}")
    
    # Extract database file from URL
    if trulens_url.startswith('sqlite:///'):
        db_file = trulens_url.replace('sqlite:///', '')
        exists = os.path.exists(db_file)
        print(f"  Database file: {db_file} (exists: {exists})")
        return db_file
    
    return None

def test_trulens_session_connectivity():
    """Test TruLens session connectivity"""
    print(f"\n🧪 Testing TruLens Session Connectivity:")
    
    try:
        from trulens.core import TruSession
        
        # Test with environment database URL
        env_db_url = os.getenv('TRULENS_DATABASE_URL', 'sqlite:///default.sqlite')
        print(f"  Testing with env URL: {env_db_url}")
        
        session = TruSession(database_url=env_db_url)
        
        # Try to get records without app name filter first
        print("  Testing get_records_and_feedback() without app_name...")
        try:
            result = session.get_records_and_feedback()
            if isinstance(result, tuple):
                records_df, feedback_columns = result
                print(f"    ✅ All records: {len(records_df)} records, {len(feedback_columns)} feedback columns")
            else:
                records_df = result
                print(f"    ✅ All records: {len(records_df)} records (non-tuple return)")
                
            # Show some record details if available
            if hasattr(records_df, 'iterrows') and len(records_df) > 0:
                print("    Sample records:")
                for idx, row in records_df.head(3).iterrows():
                    app_id = row.get('app_id', 'N/A')
                    input_val = str(row.get('input', 'N/A'))[:50]
                    print(f"      Record {idx}: app_id={app_id}, input={input_val}...")
                    
        except Exception as e:
            print(f"    ❌ Error getting all records: {e}")
        
        # Try with app_name filter
        print("  Testing get_records_and_feedback() with app_name='promptforge'...")
        try:
            result = session.get_records_and_feedback(app_name="promptforge")
            if isinstance(result, tuple):
                records_df, feedback_columns = result
                print(f"    ✅ Promptforge records: {len(records_df)} records, {len(feedback_columns)} feedback columns")
            else:
                records_df = result
                print(f"    ✅ Promptforge records: {len(records_df)} records (non-tuple return)")
        except Exception as e:
            print(f"    ❌ Error getting promptforge records: {e}")
            
        # List all apps
        print("  Testing list_apps()...")
        try:
            apps = session.list_apps()
            print(f"    ✅ Found {len(apps)} apps:")
            for i, app in enumerate(apps):
                print(f"      App {i+1}: {app}")
        except Exception as e:
            print(f"    ❌ Error listing apps: {e}")
            
    except Exception as e:
        print(f"❌ TruLens session error: {e}")
        import traceback
        traceback.print_exc()

def test_dashboard_endpoint():
    """Test the dashboard endpoint directly"""
    print(f"\n🌐 Testing Dashboard Endpoint:")
    
    try:
        import httpx
        import asyncio
        
        async def test_endpoint():
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(
                        "http://localhost:8000/api/v1/trulens/dashboard",
                        headers={"Authorization": "Bearer demo-token"},
                        timeout=10.0
                    )
                    
                    print(f"  Status Code: {response.status_code}")
                    if response.status_code == 200:
                        data = response.json()
                        print(f"  ✅ Dashboard data: {data}")
                    else:
                        print(f"  ❌ Response: {response.text}")
                        
                except Exception as e:
                    print(f"  ❌ Request error: {e}")
        
        asyncio.run(test_endpoint())
        
    except ImportError:
        print("  ⚠️ httpx not available - skipping endpoint test")
    except Exception as e:
        print(f"  ❌ Endpoint test error: {e}")

def consolidate_databases():
    """Offer to consolidate records from multiple databases"""
    print(f"\n🔧 Database Consolidation Options:")
    
    # Find databases with records
    db_files = ['default.sqlite', 'trulens_promptforge.db']
    db_with_records = []
    
    for db_file in db_files:
        if os.path.exists(db_file):
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            try:
                cursor.execute('SELECT COUNT(*) FROM trulens_records')
                count = cursor.fetchone()[0]
                if count > 0:
                    db_with_records.append((db_file, count))
            except:
                pass
            conn.close()
    
    if len(db_with_records) > 1:
        print("  📊 Multiple databases found with records:")
        for db_file, count in db_with_records:
            print(f"    {db_file}: {count} records")
        
        print(f"\n  💡 Recommendation:")
        print(f"    1. Choose one primary database (recommended: trulens_promptforge.db)")
        print(f"    2. Update TRULENS_DATABASE_URL in .env to point to chosen database")
        print(f"    3. Run migration script to consolidate records")
        print(f"    4. Restart server to use unified database")
        
        return db_with_records
    elif len(db_with_records) == 1:
        db_file, count = db_with_records[0]
        print(f"  ✅ Single database with records: {db_file} ({count} records)")
        
        # Check if .env points to the right database
        env_db = os.getenv('TRULENS_DATABASE_URL', 'sqlite:///default.sqlite')
        expected_file = env_db.replace('sqlite:///', '')
        
        if expected_file != db_file:
            print(f"  ⚠️ Configuration mismatch!")
            print(f"    .env points to: {expected_file}")
            print(f"    Records are in: {db_file}")
            print(f"    → Update .env to: TRULENS_DATABASE_URL=sqlite:///{db_file}")
    else:
        print("  ❌ No databases found with records")
    
    return db_with_records

def main():
    """Main diagnostic routine"""
    print("🚀 TruLens Database Issue Diagnostic")
    print("=" * 60)
    print("Investigating why API creates records but dashboard shows none")
    print()
    
    # Run all diagnostic checks
    check_database_files()
    configured_db = check_environment_config()
    test_trulens_session_connectivity()
    test_dashboard_endpoint()
    db_with_records = consolidate_databases()
    
    # Final recommendations
    print(f"\n{'='*60}")
    print("📋 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    
    if len(db_with_records) > 1:
        print("🔴 ISSUE IDENTIFIED: Multiple databases with records")
        print("   → API and dashboard are using different databases")
        print()
        print("🛠️ SOLUTION:")
        print("   1. Choose trulens_promptforge.db as primary database")
        print("   2. Update .env: TRULENS_DATABASE_URL=sqlite:///trulens_promptforge.db")
        print("   3. Run consolidation script to merge records")
        print("   4. Restart server")
    elif len(db_with_records) == 1:
        db_file, count = db_with_records[0]
        env_db = os.getenv('TRULENS_DATABASE_URL', 'sqlite:///default.sqlite').replace('sqlite:///', '')
        
        if env_db != db_file:
            print("🟡 ISSUE IDENTIFIED: Configuration mismatch")
            print(f"   → .env points to {env_db}, but records are in {db_file}")
            print()
            print("🛠️ SOLUTION:")
            print(f"   1. Update .env: TRULENS_DATABASE_URL=sqlite:///{db_file}")
            print("   2. Restart server")
        else:
            print("🟢 DATABASE CONFIGURATION LOOKS CORRECT")
            print("   → Issue might be in record retrieval logic")
            print()
            print("🛠️ NEXT STEPS:")
            print("   1. Check server logs for TruLens errors")
            print("   2. Test dashboard endpoint directly")
            print("   3. Verify app_name filtering logic")
    else:
        print("🔴 ISSUE IDENTIFIED: No records found in any database")
        print("   → Records are not being created properly")
        print()
        print("🛠️ SOLUTION:")
        print("   1. Check TruLens integration in API endpoint")
        print("   2. Verify TruSession initialization")
        print("   3. Test record creation with diagnostic calls")

if __name__ == "__main__":
    main()