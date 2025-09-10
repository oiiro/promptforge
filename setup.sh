#!/bin/bash
# PromptForge Simple Setup Script
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 PromptForge Enterprise Setup${NC}"

# Check Python
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    echo -e "${RED}❌ Python 3.8+ required${NC}"
    exit 1
fi

# Setup virtual environment
echo -e "${BLUE}📦 Setting up environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
pip install -q --upgrade pip setuptools wheel
pip install -q -r requirements.txt

# Initialize project
echo -e "${BLUE}🔧 Initializing project...${NC}"
python3 -c "
import os, json, sqlite3
from pathlib import Path

# Create directories
dirs = ['prompts/_registry', 'prompts/_templates', 'data', 'logs']
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)

# Create basic database
conn = sqlite3.connect('data/promptforge.db')
conn.execute('CREATE TABLE IF NOT EXISTS prompts (id TEXT PRIMARY KEY, team TEXT, name TEXT, version TEXT)')
conn.commit()
conn.close()

# Create basic config if missing
if not os.path.exists('.env'):
    with open('.env', 'w') as f:
        f.write('ENVIRONMENT=development\\nENABLE_MOCK_MODE=true\\nLOG_LEVEL=INFO\\n')

print('✅ Project initialized')
"

# Install CLI tools
echo -e "${BLUE}🛠️  Installing CLI tools...${NC}"
cd tools/cli && pip install -q -e . && cd ../..

# Verify installation
echo -e "${BLUE}✅ Verifying installation...${NC}"
python3 -c "
try:
    import langfuse, fastapi, click, rich
    print('✅ Core dependencies ready')
except ImportError as e:
    print(f'⚠️  Some optional dependencies missing: {e}')

try:
    from promptforge_cli import cli
    print('✅ CLI tools ready')
except ImportError:
    print('⚠️  CLI tools may need PATH update')
"

echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo "Quick start:"
echo "  source venv/bin/activate"
echo "  export PROMPTFORGE_TEAM=platform"
echo "  promptforge --help"