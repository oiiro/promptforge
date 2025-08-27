#!/bin/bash
#
# PromptForge Server Startup Script
# Ensures proper virtual environment activation and server startup
#

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting PromptForge API Server${NC}"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Run setup first:${NC}"
    echo "   python3 setup_promptforge.py"
    exit 1
fi

# Check if FastAPI is installed in venv
if ! ./venv/bin/python -c "import fastapi" 2>/dev/null; then
    echo -e "${RED}❌ FastAPI not found in virtual environment. Run setup:${NC}"
    echo "   python3 setup_promptforge.py"
    exit 1
fi

# Check if port 8000 is in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port 8000 is already in use. Stopping existing server...${NC}"
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Start the server using virtual environment Python
echo -e "${GREEN}🔧 Starting FastAPI server with virtual environment Python...${NC}"
echo "   URL: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "   Health Check: http://localhost:8000/health"
echo ""
echo -e "${YELLOW}📋 Press Ctrl+C to stop the server${NC}"
echo ""

# Run the server
exec ./venv/bin/python orchestration/app.py