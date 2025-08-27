#!/bin/bash

# PromptForge Verification Script - Enhanced Python Version
# Comprehensive verification of installation and functionality
set -e

echo "🔍 PromptForge Verification Suite"
echo "=================================="

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

# Check for virtual environment
if [[ ! -d "venv" ]]; then
    echo "❌ Virtual environment not found"
    echo "   Run ./setup.sh or python3 setup_promptforge.py first"
    exit 1
fi

# Run the enhanced Python verification script
echo "🔄 Running comprehensive verification..."
echo ""

if python3 verify_installation.py; then
    echo ""
    echo "🎉 Verification completed!"
    echo ""
    echo "🚀 Quick Start Commands:"
    echo "   Start API server: source venv/bin/activate && python orchestration/app.py"
    echo "   Run tests: source venv/bin/activate && python -m pytest evals/"
    echo "   API documentation: http://localhost:8000/docs"
    echo ""
    echo "📄 Check verification_report.json for detailed results"
else
    echo ""
    echo "⚠️  Verification completed with issues"
    echo "🔧 Review the output above and verification_report.json"
    echo "🆘 If issues persist, run: python3 setup_promptforge.py"
fi