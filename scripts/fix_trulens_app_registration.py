#!/usr/bin/env python3
"""
Fix TruLens App Registration
Register the missing app for existing records to resolve dashboard display issues
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up environment
from dotenv import load_dotenv
load_dotenv()

def fix_app_registration():
    """Register missing app in TruLens database"""
    print("🔧 Fixing TruLens app registration...")
    
    # We know records are in default.sqlite since that's where TruLens defaulted to
    database_path = "default.sqlite"
    
    if not os.path.exists(database_path):
        print(f"❌ Database file {database_path} not found")
        return False
    
    try:
        # Initialize TruLens with the correct database
        from trulens.core import TruSession
        from trulens.apps.virtual import TruVirtual
        
        session = TruSession()  # This will use default.sqlite
        print(f"✅ Connected to TruLens database")
        
        # Check current state directly from database since get_records_and_feedback() fails
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Get record count and unique app_ids
        cursor.execute('SELECT COUNT(*) FROM trulens_records')
        record_count = cursor.fetchone()[0]
        print(f"📊 Found {record_count} records in database")
        
        cursor.execute('SELECT DISTINCT app_id FROM trulens_records')
        unique_app_ids = [row[0] for row in cursor.fetchall()]
        print(f"📱 Unique app_ids in records: {unique_app_ids}")
        
        cursor.execute('SELECT COUNT(*) FROM trulens_apps')
        app_count = cursor.fetchone()[0]
        print(f"📱 Current apps in database: {app_count}")
        
        conn.close()
        
        # Register missing apps
        if unique_app_ids:
            for app_id in unique_app_ids:
                try:
                    # Create virtual app structure
                    virtual_app = {
                        "llm": {
                            "provider": "openai",
                            "model": "gpt-4"
                        },
                        "type": "multi-person-retirement-eligibility",
                        "description": "PII-protected multi-person retirement eligibility processing"
                    }
                    
                    # Create TruVirtual app
                    app = TruVirtual(
                        app_name=app_id,
                        app_id=app_id,
                        app_version="1.0.0",
                        app=virtual_app
                    )
                    
                    # Register with session
                    session.add_app(app=app)
                    print(f"✅ Registered TruVirtual app: {app_id}")
                    
                except Exception as e:
                    print(f"⚠️  Failed to register app {app_id}: {e}")
                    # Continue with other apps
        
        # Verify registration and fix record app_ids
        print("\n🧪 Verifying app registration and fixing record references...")
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM trulens_apps')
        app_count = cursor.fetchone()[0]
        print(f"✅ Apps in database: {app_count}")
        
        if app_count > 0:
            cursor.execute('SELECT app_id, app_name FROM trulens_apps')
            apps = cursor.fetchall()
            print("📱 Registered apps:")
            for app_id, app_name in apps:
                print(f"  - {app_id} ({app_name})")
                
                # Update records to use the correct app_id hash instead of app_name
                print(f"🔧 Updating records to use app_id hash {app_id}...")
                cursor.execute('UPDATE trulens_records SET app_id = ? WHERE app_id = ?', 
                             (app_id, app_name))
                updated_count = cursor.rowcount
                print(f"✅ Updated {updated_count} records")
            
            conn.commit()
        
        conn.close()
        
        # Test record retrieval
        print("\n🧪 Testing record retrieval...")
        records_after = session.get_records_and_feedback()
        if isinstance(records_after, tuple):
            records_df_after, _ = records_after
        else:
            records_df_after = records_after
            
        print(f"✅ Successfully retrieved {len(records_df_after)} records")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix app registration: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_fix():
    """Verify the fix by testing dashboard endpoint functionality"""
    print("\n🧪 Verifying fix with dashboard endpoint test...")
    
    try:
        from trulens.core import TruSession
        session = TruSession()
        
        # Test the same logic as our dashboard endpoint
        app_name = "multi-person-retirement-eligibility"
        
        try:
            result = session.get_records_and_feedback(app_name=app_name)
            if isinstance(result, tuple):
                records_df, feedback_columns = result
                print(f"✅ Dashboard endpoint test: {len(records_df)} records, {len(feedback_columns)} feedback columns")
            else:
                records_df = result
                print(f"✅ Dashboard endpoint test: {len(records_df)} records")
                
            return len(records_df) > 0
            
        except Exception as e:
            print(f"❌ Dashboard endpoint test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TruLens App Registration Fix")
    print("=" * 50)
    
    success = fix_app_registration()
    
    if success:
        print("\n✅ App registration completed successfully!")
        
        if verify_fix():
            print("✅ Verification passed - dashboard should now work!")
            print("\n📋 Next steps:")
            print("1. Restart the PromptForge server")
            print("2. Test the dashboard endpoint: GET /api/v1/dashboard")
            print("3. Start TruLens dashboard: trulens")
            sys.exit(0)
        else:
            print("⚠️  Registration completed but verification failed")
            sys.exit(1)
    else:
        print("❌ App registration failed")
        sys.exit(1)