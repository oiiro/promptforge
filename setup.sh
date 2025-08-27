#!/bin/bash

# PromptForge Setup Script - Enhanced Python Version
# Automated setup for financial services grade prompt engineering SDLC
set -e

echo "🚀 PromptForge Setup - Financial Services Grade Prompt Engineering SDLC"
echo "======================================================================="

# Check if we're in the right directory
if [[ ! -f "requirements.txt" ]] || [[ ! -f "README.md" ]]; then
    echo "❌ Error: Please run this script from the promptforge project directory"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Python 3 is required but not installed"
    echo "   Install Python 3.9+ from https://python.org"
    exit 1
fi

# Run the enhanced Python setup script
echo "🔄 Running enhanced Python setup script..."
echo ""

if python3 setup_promptforge.py; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "🔍 For detailed verification, run: python3 verify_installation.py"
    echo "🚀 To start the API server: source venv/bin/activate && python orchestration/app.py"
    echo "📚 API documentation: http://localhost:8000/docs"
else
    echo ""
    echo "❌ Setup encountered issues. Check the output above for details."
    echo "🔍 Run verification: python3 verify_installation.py"
    exit 1
fi