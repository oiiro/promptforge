#!/usr/bin/env python3
"""
Update TruLens App Name to PromptForge
Updates existing database records and registers new app with 'promptforge' name
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

def update_app_name_to_promptforge():
    """Update app name from multi-person-retirement-eligibility to promptforge"""
    print("🔧 Updating TruLens app name to 'promptforge'...")
    
    # We know records are in default.sqlite
    database_path = "default.sqlite"
    
    if not os.path.exists(database_path):
        print(f"❌ Database file {database_path} not found")
        return False
    
    try:
        # Initialize TruLens
        from trulens.core import TruSession
        from trulens.apps.virtual import TruVirtual
        
        session = TruSession()  # This will use default.sqlite
        print(f"✅ Connected to TruLens database")
        
        # Check current state
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Get current apps and records count
        cursor.execute('SELECT COUNT(*) FROM trulens_records')
        record_count = cursor.fetchone()[0]
        print(f"📊 Found {record_count} records in database")
        
        cursor.execute('SELECT COUNT(*) FROM trulens_apps')
        app_count = cursor.fetchone()[0]
        print(f"📱 Current apps in database: {app_count}")
        
        # Show current apps
        if app_count > 0:
            cursor.execute('SELECT app_id, app_name FROM trulens_apps')
            apps = cursor.fetchall()
            print("📱 Current apps:")
            for app_id, app_name in apps:
                print(f"  - {app_id} ({app_name})")
        
        # Create new promptforge app
        print(f"\n🔧 Creating new 'promptforge' app...")
        
        virtual_app = {
            "llm": {
                "provider": "openai", 
                "model": "gpt-4"
            },
            "type": "promptforge",
            "description": "PromptForge PII-protected multi-person processing system",
            "version": "1.0.0"
        }
        
        # Create TruVirtual app
        promptforge_app = TruVirtual(
            app_name="promptforge",
            app_id="promptforge", 
            app_version="1.0.0",
            app=virtual_app
        )
        
        # Register with session
        session.add_app(app=promptforge_app)
        print(f"✅ Registered new TruVirtual app: promptforge")
        
        # Get the new app's hash ID
        cursor.execute('SELECT app_id FROM trulens_apps WHERE app_name = ?', ('promptforge',))
        new_app_result = cursor.fetchone()
        if new_app_result:
            new_app_id = new_app_result[0]
            print(f"✅ New app registered with ID: {new_app_id}")
            
            # Update all existing records to use the new app_id
            cursor.execute('UPDATE trulens_records SET app_id = ?', (new_app_id,))
            updated_count = cursor.rowcount
            print(f"✅ Updated {updated_count} records to use new promptforge app_id")
            
            conn.commit()
        else:
            print("❌ Failed to retrieve new app ID")
            return False
        
        # Clean up old apps (optional)
        print(f"\n🧹 Cleaning up old apps...")
        cursor.execute('DELETE FROM trulens_apps WHERE app_name != ?', ('promptforge',))
        deleted_count = cursor.rowcount
        print(f"✅ Cleaned up {deleted_count} old app entries")
        conn.commit()
        
        conn.close()
        
        # Verify the update
        print("\n🧪 Verifying update...")
        records_after = session.get_records_and_feedback(app_name="promptforge")
        if isinstance(records_after, tuple):
            records_df_after, _ = records_after
        else:
            records_df_after = records_after
            
        print(f"✅ Successfully retrieved {len(records_df_after)} records for promptforge app")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to update app name: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_promptforge_app():
    """Verify the promptforge app works correctly"""
    print("\n🧪 Verifying promptforge app functionality...")
    
    try:
        from trulens.core import TruSession
        session = TruSession()
        
        # Test retrieval with promptforge app name
        result = session.get_records_and_feedback(app_name="promptforge")
        if isinstance(result, tuple):
            records_df, feedback_columns = result
            print(f"✅ PromptForge app test: {len(records_df)} records, {len(feedback_columns)} feedback columns")
        else:
            records_df = result
            print(f"✅ PromptForge app test: {len(records_df)} records")
            
        return len(records_df) > 0
        
    except Exception as e:
        print(f"❌ PromptForge app verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TruLens App Name Update to PromptForge")
    print("=" * 50)
    
    success = update_app_name_to_promptforge()
    
    if success:
        print("\n✅ App name update completed successfully!")
        
        if verify_promptforge_app():
            print("✅ Verification passed - promptforge app is working!")
            print("\n📋 Next steps:")
            print("1. Restart the PromptForge server (it will use 'promptforge' app name)")
            print("2. Test the dashboard endpoint: GET /api/v1/dashboard")
            print("3. Test API calls will now be tracked under 'promptforge' app")
            sys.exit(0)
        else:
            print("⚠️  Update completed but verification failed")
            sys.exit(1)
    else:
        print("❌ App name update failed")
        sys.exit(1)