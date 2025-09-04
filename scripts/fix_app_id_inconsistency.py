#!/usr/bin/env python3
"""
Fix TruLens App ID Inconsistency
Updates literal 'promptforge' app_id records to use the hashed app_id format
"""

import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_app_id_inconsistency():
    """Fix app_id inconsistency between literal and hashed formats"""
    
    db_path = Path("trulens_promptforge.db")
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False
    
    logger.info("🔧 Fixing TruLens app_id inconsistency...")
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # First, identify the target hashed app_id
        cursor.execute("""
            SELECT app_id FROM trulens_apps 
            WHERE app_name = 'promptforge'
            LIMIT 1
        """)
        result = cursor.fetchone()
        
        if not result:
            logger.error("No promptforge app found in trulens_apps table")
            return False
        
        target_app_id = result[0]
        logger.info(f"📊 Target app_id: {target_app_id}")
        
        # Count records with literal app_id
        cursor.execute("""
            SELECT COUNT(*) FROM trulens_records 
            WHERE app_id = 'promptforge'
        """)
        literal_count = cursor.fetchone()[0]
        
        # Count records with hashed app_id
        cursor.execute("""
            SELECT COUNT(*) FROM trulens_records 
            WHERE app_id = ?
        """, (target_app_id,))
        hashed_count = cursor.fetchone()[0]
        
        logger.info(f"📋 Found {literal_count} records with literal app_id 'promptforge'")
        logger.info(f"📋 Found {hashed_count} records with hashed app_id '{target_app_id}'")
        
        if literal_count == 0:
            logger.info("✅ No inconsistency found - all records already use hashed app_id")
            return True
        
        # Update literal app_id records to use hashed app_id
        cursor.execute("""
            UPDATE trulens_records 
            SET app_id = ?
            WHERE app_id = 'promptforge'
        """, (target_app_id,))
        
        updated_count = cursor.rowcount
        conn.commit()
        
        logger.info(f"✅ Updated {updated_count} records to use hashed app_id")
        
        # Verify the fix
        cursor.execute("""
            SELECT COUNT(*) FROM trulens_records 
            WHERE app_id = ?
        """, (target_app_id,))
        final_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM trulens_records 
            WHERE app_id = 'promptforge'
        """)
        remaining_literal = cursor.fetchone()[0]
        
        logger.info(f"📊 Verification:")
        logger.info(f"   Total records with hashed app_id: {final_count}")
        logger.info(f"   Remaining literal app_id records: {remaining_literal}")
        
        if remaining_literal == 0:
            logger.info("🎉 App ID inconsistency successfully resolved!")
            return True
        else:
            logger.error(f"⚠️  Still have {remaining_literal} records with literal app_id")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error fixing app_id inconsistency: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def main():
    """Main execution function"""
    logger.info("🛠️  TruLens App ID Inconsistency Fix")
    logger.info("=" * 50)
    
    success = fix_app_id_inconsistency()
    
    if success:
        logger.info("\n" + "=" * 50)
        logger.info("✅ APP ID INCONSISTENCY FIX COMPLETE")
        logger.info("=" * 50)
        logger.info("🎉 All TruLens records now use consistent hashed app_id!")
        logger.info("🔄 Next steps:")
        logger.info("1. Restart the TruLens dashboard")
        logger.info("2. Verify all 19 records are now visible")
        logger.info("3. Test new API calls to ensure they appear immediately")
        return 0
    else:
        logger.error("\n" + "=" * 50)
        logger.error("❌ APP ID INCONSISTENCY FIX FAILED")
        logger.error("=" * 50)
        return 1

if __name__ == "__main__":
    exit(main())