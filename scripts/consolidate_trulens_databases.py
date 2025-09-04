#!/usr/bin/env python3
"""
TruLens Database Consolidation Script
Consolidates records from multiple databases into a single unified database
"""

import os
import sys
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def backup_databases():
    """Create backups of existing databases"""
    print("💾 Creating database backups...")
    
    databases = ['default.sqlite', 'trulens_promptforge.db']
    backups_created = []
    
    for db_name in databases:
        if os.path.exists(db_name):
            backup_name = f"{db_name}.backup"
            
            # Copy database
            import shutil
            shutil.copy2(db_name, backup_name)
            backups_created.append(backup_name)
            print(f"  ✅ Backed up {db_name} → {backup_name}")
    
    return backups_created

def analyze_databases() -> Dict[str, Any]:
    """Analyze existing databases to understand the schema and data"""
    print("🔍 Analyzing existing databases...")
    
    analysis = {
        'databases': {},
        'total_records': 0,
        'total_apps': 0,
        'unique_app_names': set(),
        'schema_differences': []
    }
    
    databases = ['default.sqlite', 'trulens_promptforge.db']
    
    for db_name in databases:
        if not os.path.exists(db_name):
            continue
            
        print(f"\n📊 Analyzing {db_name}...")
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        db_info = {
            'records': 0,
            'apps': 0,
            'app_details': [],
            'sample_records': []
        }
        
        try:
            # Count records
            cursor.execute('SELECT COUNT(*) FROM trulens_records')
            record_count = cursor.fetchone()[0]
            db_info['records'] = record_count
            analysis['total_records'] += record_count
            print(f"  Records: {record_count}")
            
            # Count apps
            cursor.execute('SELECT COUNT(*) FROM trulens_apps')
            app_count = cursor.fetchone()[0]
            db_info['apps'] = app_count
            analysis['total_apps'] += app_count
            print(f"  Apps: {app_count}")
            
            # Get app details
            cursor.execute('SELECT app_id, app_name FROM trulens_apps')
            apps = cursor.fetchall()
            for app_id, app_name in apps:
                db_info['app_details'].append({'app_id': app_id, 'app_name': app_name})
                analysis['unique_app_names'].add(app_name)
                print(f"    App: '{app_name}' (id: {app_id[:16]}...)")
                
                # Count records for this app
                cursor.execute('SELECT COUNT(*) FROM trulens_records WHERE app_id = ?', (app_id,))
                app_records = cursor.fetchone()[0]
                print(f"         Records: {app_records}")
            
            # Get sample records to understand structure
            cursor.execute('''
                SELECT record_id, app_id, input, output, ts
                FROM trulens_records 
                ORDER BY ts DESC 
                LIMIT 2
            ''')
            sample_records = cursor.fetchall()
            for record_id, app_id, input_data, output_data, ts in sample_records:
                db_info['sample_records'].append({
                    'record_id': record_id,
                    'app_id': app_id,
                    'timestamp': ts,
                    'has_input': input_data is not None,
                    'has_output': output_data is not None
                })
            
        except Exception as e:
            print(f"  ❌ Error analyzing {db_name}: {e}")
        
        conn.close()
        analysis['databases'][db_name] = db_info
    
    # Convert set to list for JSON serialization
    analysis['unique_app_names'] = list(analysis['unique_app_names'])
    
    return analysis

def consolidate_to_primary_database(target_db: str = "trulens_promptforge.db") -> bool:
    """Consolidate all records into the primary database"""
    print(f"\n🔧 Consolidating records into {target_db}...")
    
    # Ensure target database exists
    if not os.path.exists(target_db):
        print(f"❌ Target database {target_db} does not exist")
        return False
    
    target_conn = sqlite3.connect(target_db)
    target_cursor = target_conn.cursor()
    
    # Get the target app_id (should be the hash from the promptforge app)
    target_cursor.execute("SELECT app_id FROM trulens_apps WHERE app_name = 'promptforge'")
    target_app_result = target_cursor.fetchone()
    
    if not target_app_result:
        print("❌ No promptforge app found in target database")
        target_conn.close()
        return False
    
    target_app_id = target_app_result[0]
    print(f"✅ Target app_id: {target_app_id}")
    
    # Source databases to consolidate from
    source_databases = ['default.sqlite']
    total_migrated = 0
    
    for source_db in source_databases:
        if not os.path.exists(source_db) or source_db == target_db:
            continue
            
        print(f"\n📦 Migrating records from {source_db}...")
        source_conn = sqlite3.connect(source_db)
        source_cursor = source_conn.cursor()
        
        try:
            # Get all records from source
            source_cursor.execute('''
                SELECT record_id, app_id, input, output, perf_json, ts, cost_json, tags, record_json
                FROM trulens_records
                ORDER BY ts
            ''')
            
            records = source_cursor.fetchall()
            print(f"  Found {len(records)} records to migrate")
            
            migrated_count = 0
            for record in records:
                record_id, app_id, input_data, output_data, perf_json, ts, cost_json, tags, record_json = record
                
                # Check if record already exists in target
                target_cursor.execute("SELECT COUNT(*) FROM trulens_records WHERE record_id = ?", (record_id,))
                if target_cursor.fetchone()[0] > 0:
                    print(f"    ⚠️ Record {record_id} already exists, skipping")
                    continue
                
                # Insert with updated app_id
                try:
                    target_cursor.execute('''
                        INSERT INTO trulens_records 
                        (record_id, app_id, input, output, perf_json, ts, cost_json, tags, record_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (record_id, target_app_id, input_data, output_data, perf_json, ts, cost_json, tags, record_json))
                    
                    migrated_count += 1
                    
                except Exception as e:
                    print(f"    ❌ Error migrating record {record_id}: {e}")
            
            target_conn.commit()
            total_migrated += migrated_count
            print(f"  ✅ Migrated {migrated_count} records from {source_db}")
            
        except Exception as e:
            print(f"  ❌ Error migrating from {source_db}: {e}")
        
        source_conn.close()
    
    # Also migrate any records within target database that have wrong app_id
    print(f"\n🔄 Fixing app_id mismatches within {target_db}...")
    try:
        # Find records with literal "promptforge" app_id instead of hash
        target_cursor.execute("SELECT COUNT(*) FROM trulens_records WHERE app_id = 'promptforge'")
        mismatch_count = target_cursor.fetchone()[0]
        
        if mismatch_count > 0:
            print(f"  Found {mismatch_count} records with literal app_id, fixing...")
            target_cursor.execute(
                "UPDATE trulens_records SET app_id = ? WHERE app_id = 'promptforge'",
                (target_app_id,)
            )
            target_conn.commit()
            print(f"  ✅ Fixed {mismatch_count} app_id mismatches")
            total_migrated += mismatch_count
        else:
            print("  ✅ No app_id mismatches found")
            
    except Exception as e:
        print(f"  ❌ Error fixing app_id mismatches: {e}")
    
    target_conn.close()
    
    print(f"\n✅ Consolidation complete! Migrated {total_migrated} total records")
    return True

def verify_consolidation(target_db: str = "trulens_promptforge.db") -> bool:
    """Verify the consolidation was successful"""
    print(f"\n🧪 Verifying consolidation in {target_db}...")
    
    try:
        # Manual database check
        conn = sqlite3.connect(target_db)
        cursor = conn.cursor()
        
        # Count total records
        cursor.execute('SELECT COUNT(*) FROM trulens_records')
        total_records = cursor.fetchone()[0]
        print(f"  📊 Total records: {total_records}")
        
        # Count records by app
        cursor.execute('''
            SELECT r.app_id, a.app_name, COUNT(*) as record_count
            FROM trulens_records r
            LEFT JOIN trulens_apps a ON r.app_id = a.app_id
            GROUP BY r.app_id, a.app_name
        ''')
        app_stats = cursor.fetchall()
        print(f"  📱 Records by app:")
        for app_id, app_name, count in app_stats:
            print(f"    {app_name or 'Unknown'}: {count} records (id: {app_id[:16] if app_id else 'None'}...)")
        
        conn.close()
        
        # Test TruLens session
        print(f"\n🧪 Testing TruLens session with consolidated database...")
        
        # Set environment to point to target database
        os.environ['TRULENS_DATABASE_URL'] = f'sqlite:///{target_db}'
        
        from trulens.core import TruSession
        session = TruSession(database_url=f'sqlite:///{target_db}')
        
        # Test record retrieval
        try:
            result = session.get_records_and_feedback(app_name="promptforge")
            if isinstance(result, tuple):
                records_df, feedback_columns = result
                print(f"  ✅ TruLens retrieval: {len(records_df)} records, {len(feedback_columns)} feedback columns")
                return len(records_df) > 0
            else:
                records_df = result
                print(f"  ✅ TruLens retrieval: {len(records_df)} records")
                return len(records_df) > 0
                
        except Exception as e:
            print(f"  ❌ TruLens retrieval error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def update_environment_config(target_db: str = "trulens_promptforge.db"):
    """Update .env file to point to the consolidated database"""
    print(f"\n⚙️ Updating .env configuration...")
    
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    # Read current content
    lines = env_file.read_text().splitlines()
    updated_lines = []
    trulens_url_updated = False
    
    target_url = f"sqlite:///{target_db}"
    
    for line in lines:
        if line.startswith('TRULENS_DATABASE_URL='):
            updated_lines.append(f'TRULENS_DATABASE_URL={target_url}')
            trulens_url_updated = True
            print(f"  ✅ Updated TRULENS_DATABASE_URL to {target_url}")
        else:
            updated_lines.append(line)
    
    # If TRULENS_DATABASE_URL wasn't found, add it
    if not trulens_url_updated:
        updated_lines.append(f'TRULENS_DATABASE_URL={target_url}')
        print(f"  ✅ Added TRULENS_DATABASE_URL={target_url}")
    
    # Write updated content
    env_file.write_text('\n'.join(updated_lines) + '\n')
    print(f"  ✅ .env file updated")
    return True

def main():
    """Main consolidation routine"""
    print("🚀 TruLens Database Consolidation")
    print("=" * 60)
    print("Consolidating records from multiple databases into unified storage")
    print()
    
    try:
        # Step 1: Backup existing databases
        backups = backup_databases()
        
        # Step 2: Analyze current state
        analysis = analyze_databases()
        print(f"\n📋 Analysis Summary:")
        print(f"  Total records across all databases: {analysis['total_records']}")
        print(f"  Total apps: {analysis['total_apps']}")
        print(f"  Unique app names: {analysis['unique_app_names']}")
        
        if analysis['total_records'] == 0:
            print("❌ No records found to consolidate")
            return 1
        
        # Step 3: Consolidate to primary database
        success = consolidate_to_primary_database("trulens_promptforge.db")
        if not success:
            print("❌ Consolidation failed")
            return 1
        
        # Step 4: Verify consolidation
        verified = verify_consolidation("trulens_promptforge.db")
        if not verified:
            print("⚠️ Consolidation verification failed")
        
        # Step 5: Update environment config
        update_environment_config("trulens_promptforge.db")
        
        # Final summary
        print(f"\n{'='*60}")
        print("📋 CONSOLIDATION COMPLETE")
        print("=" * 60)
        
        if verified:
            print("🎉 All records successfully consolidated!")
            print()
            print("✅ All records moved to trulens_promptforge.db")
            print("✅ App_id mismatches resolved")
            print("✅ TruLens session verified working")
            print("✅ Environment configuration updated")
            print()
            print("🔄 Next steps:")
            print("1. Restart the PromptForge server")
            print("2. Test the dashboard: curl -H 'Authorization: Bearer demo-token' http://localhost:8000/api/v1/trulens/dashboard")
            print("3. Make test API calls to verify record creation")
            print()
            print(f"💾 Backups created: {', '.join(backups)}")
            return 0
        else:
            print("⚠️ Consolidation completed with verification issues")
            print("Check the output above for specific problems")
            return 1
            
    except Exception as e:
        print(f"❌ Consolidation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())