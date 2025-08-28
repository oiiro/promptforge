# TruLens Integration Troubleshooting Guide

**Complete guide to resolving TruLens dashboard errors and integration issues**

## 🚨 Critical Fixes Applied

### Issue 1: Dashboard NoneType app_id Error

**Error**: `{"detail":"Dashboard error: 'NoneType' object has no attribute 'app_id'"}`

**Root Cause**: TruLens records existed in database but no corresponding app objects were registered.

**Solution Applied**:
```python
# Fixed in orchestration/app.py
# 1. Enhanced error handling for dashboard endpoint
# 2. Proper tuple unpacking for get_records_and_feedback()
# 3. Automated app registration with TruVirtual

try:
    result = app.state.tru_session.get_records_and_feedback(app_name="promptforge")
    if isinstance(result, tuple):
        records_df, feedback_columns = result
    else:
        records_df = result
        feedback_columns = []
except Exception as e:
    logger.error(f"TruLens dashboard error: {str(e)}")
    return {"error": "Dashboard temporarily unavailable"}
```

### Issue 2: Tuple Object Attribute Error

**Error**: `'tuple' object has no attribute 'empty'`

**Root Cause**: `get_records_and_feedback()` returns tuple (DataFrame, feedback_columns) not just DataFrame.

**Solution Applied**:
```python
# Proper tuple unpacking pattern
if isinstance(result, tuple):
    records_df, feedback_columns = result
    logger.info(f"Retrieved {len(records_df)} records, {len(feedback_columns)} feedback columns")
else:
    records_df = result
    feedback_columns = []
```

### Issue 3: App Name Inconsistency

**Error**: Records not showing in dashboard despite successful creation

**Root Cause**: App name "multi-person-retirement-eligibility" vs "promptforge" mismatch.

**Solution Applied**:
1. Updated all app references to "promptforge" in `orchestration/app.py`
2. Created `scripts/update_app_name_to_promptforge.py` migration script
3. Updated test scripts to use "promptforge" branding

### Issue 4: Missing App Registration

**Error**: Records exist but no app objects causing retrieval failures

**Solution Applied**:
1. Created `scripts/fix_trulens_app_registration.py` 
2. Implemented TruVirtual app pattern instead of abstract App class
3. Automated app registration during server startup

## 🛠️ Automated Fix Scripts

### 1. Fix App Registration
```bash
python scripts/fix_trulens_app_registration.py
```

**What it does**:
- Registers missing TruVirtual apps for existing records
- Updates record references to use correct app_id hashes
- Verifies registration and tests retrieval

### 2. Update App Name to PromptForge
```bash
python scripts/update_app_name_to_promptforge.py
```

**What it does**:
- Creates new "promptforge" app in TruLens database
- Migrates all existing records to use promptforge app_id
- Cleans up old app entries

### 3. Comprehensive Test
```bash
python scripts/test_comprehensive_fixes.py
```

**What it tests**:
- EMAIL_ADDRESS_3 deanonymization fix
- TruLens monitoring functionality
- PII protection accuracy
- Multi-person processing capability

## 📊 Verification Steps

### 1. Check Database State
```python
import sqlite3
conn = sqlite3.connect("default.sqlite")
cursor = conn.cursor()

# Check records count
cursor.execute('SELECT COUNT(*) FROM trulens_records')
record_count = cursor.fetchone()[0]
print(f"Records: {record_count}")

# Check apps count
cursor.execute('SELECT COUNT(*) FROM trulens_apps')
app_count = cursor.fetchone()[0]
print(f"Apps: {app_count}")

# Check app names
cursor.execute('SELECT app_id, app_name FROM trulens_apps')
apps = cursor.fetchall()
for app_id, app_name in apps:
    print(f"  - {app_id} ({app_name})")

conn.close()
```

### 2. Test Dashboard Endpoint
```bash
# With Bearer token authentication
curl -H "Authorization: Bearer demo-token" \
     http://localhost:8000/api/v1/trulens/dashboard

# Expected response: JSON with records and feedback data
```

### 3. Test Record Creation
```bash
curl -X POST http://localhost:8000/api/v1/retirement-eligibility \
  -H "Authorization: Bearer demo-token" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Test PromptForge TruLens with John, email: john@example.com",
    "enable_monitoring": true
  }'
```

## 🔧 Manual Recovery Procedures

### Reset TruLens Database
```python
import os
from trulens.core import TruSession

# Option 1: Reset to clean state
if os.path.exists("default.sqlite"):
    os.remove("default.sqlite")

# Option 2: Start fresh with new session
session = TruSession(database_url="sqlite:///trulens_promptforge_clean.db")
```

### Rebuild App Registration
```python
from trulens.core import TruSession
from trulens.apps.virtual import TruVirtual

session = TruSession()

# Create PromptForge app
virtual_app = {
    "llm": {"provider": "openai", "model": "gpt-4"},
    "type": "promptforge",
    "description": "PromptForge PII-protected multi-person processing system"
}

promptforge_app = TruVirtual(
    app_name="promptforge",
    app_id="promptforge", 
    app_version="1.0.0",
    app=virtual_app
)

session.add_app(app=promptforge_app)
```

## 📋 Common Error Messages & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `'NoneType' object has no attribute 'app_id'` | Missing app registration | Run `fix_trulens_app_registration.py` |
| `'tuple' object has no attribute 'empty'` | Incorrect tuple handling | Fixed in latest app.py version |
| `No records found` | App name mismatch | Run `update_app_name_to_promptforge.py` |
| `Dashboard not loading` | API endpoint issues | Check `/api/v1/trulens/dashboard` path |
| `503 Service Unavailable` | TruLens not initialized | Restart server, check TruSession init |

## 🔍 Debugging Tips

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# In orchestration/app.py
logger.setLevel(logging.DEBUG)
```

### Check TruLens Version Compatibility
```python
import trulens
print(f"TruLens version: {trulens.__version__}")
# Should be: 2.2.4 or compatible
```

### Verify Database Connection
```python
from trulens.core import TruSession

try:
    session = TruSession()
    print("✅ TruLens session created successfully")
    
    # Test basic operation
    apps = session.list_apps()
    print(f"✅ Found {len(apps)} apps")
    
except Exception as e:
    print(f"❌ TruLens connection failed: {e}")
```

## 🚀 Production Checklist

Before deploying to production, verify:

- [ ] Dashboard endpoint responds with 200 status
- [ ] Records are being created (check logs for "added record")
- [ ] App registration is complete (promptforge app exists)
- [ ] Error handling is working (no server crashes)
- [ ] Database migrations are applied
- [ ] All fix scripts have been executed
- [ ] Comprehensive tests pass

## 📞 Support Escalation

If issues persist after applying all fixes:

1. **Check server logs** for detailed error traces
2. **Run comprehensive test script** to identify specific failures
3. **Verify TruLens version** compatibility (2.2.4 recommended)
4. **Consider database reset** if corruption suspected
5. **Review environment variables** for configuration issues

---

**Last Updated**: 2024-08-28  
**Fix Scripts Version**: 1.0.0  
**Verified Compatible**: TruLens v2.2.4