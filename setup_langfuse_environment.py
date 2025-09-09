#!/usr/bin/env python3
"""
PromptForge Langfuse Environment Setup Script
Automatically configures the environment for Langfuse integration
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major != 3 or version.minor < 9:
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Please use Python 3.9 or higher")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("📦 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_venv_python():
    """Get path to virtual environment Python"""
    if os.name == "nt":  # Windows
        return Path("venv/Scripts/python.exe")
    else:  # Unix/Linux/MacOS
        return Path("venv/bin/python")

def install_dependencies():
    """Install Langfuse and other dependencies"""
    print("\n📦 Installing dependencies...")
    
    venv_python = get_venv_python()
    
    if not venv_python.exists():
        print("❌ Virtual environment Python not found")
        return False
    
    # Core dependencies for Langfuse integration
    langfuse_deps = [
        "langfuse>=2.0.0",
        "deepeval>=0.21.0", 
        "pydantic>=2.7.0",
        "python-dotenv>=1.0.0",
        "structlog>=23.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0"
    ]
    
    try:
        # Upgrade pip first
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install Langfuse dependencies
        for dep in langfuse_deps:
            print(f"  Installing {dep}...")
            subprocess.run([str(venv_python), "-m", "pip", "install", dep], 
                          check=True, capture_output=True)
        
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create .env file with Langfuse configuration"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("📝 Creating .env file...")
    
    env_content = """# PromptForge Langfuse Configuration
# Replace with your actual Langfuse API keys from https://cloud.langfuse.com

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key-here
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED=true
LANGFUSE_OBSERVABILITY_LEVEL=standard
LANGFUSE_SAMPLING_RATE=1.0

# LLM Provider API Keys (add your actual keys)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-4

# DeepEval Configuration
DEEPEVAL_TESTSET_RATIO=0.8
DEEPEVAL_METRICS_MODEL=gpt-4

# Security & Compliance
ENABLE_PII_REDACTION=true
ENABLE_AUDIT_LOGGING=true
ENABLE_FINANCIAL_COMPLIANCE=true

# Development Settings
LOG_LEVEL=INFO
ENVIRONMENT=development
"""
    
    try:
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ .env file created")
        print("   📝 Please update with your actual API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def verify_installation():
    """Verify that the installation works"""
    print("\n🔍 Verifying installation...")
    
    venv_python = get_venv_python()
    
    test_script = """
import sys
try:
    import langfuse
    print("✅ Langfuse imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Langfuse: {e}")
    sys.exit(1)

try:
    import deepeval
    print("✅ DeepEval imported successfully")
except ImportError as e:
    print(f"❌ Failed to import DeepEval: {e}")
    sys.exit(1)

try:
    from langfuse.decorators import observe, langfuse_context
    print("✅ Langfuse decorators available")
except ImportError as e:
    print(f"❌ Failed to import Langfuse decorators: {e}")
    sys.exit(1)

print("✅ All imports successful - Langfuse integration ready!")
"""
    
    try:
        result = subprocess.run([str(venv_python), "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Verification failed:")
        print(e.stdout)
        print(e.stderr)
        return False

def create_quickstart_script():
    """Create a quickstart script"""
    print("\n📝 Creating quickstart script...")
    
    quickstart_content = '''#!/usr/bin/env python3
"""
PromptForge Langfuse Quickstart
Run this script to test your Langfuse integration
"""

import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def main():
    print("🚀 PromptForge Langfuse Quickstart")
    print("=" * 50)
    
    # Check configuration
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    
    if not public_key.startswith("pk-lf-") or public_key.endswith("-here"):
        print("⚠️  Please update LANGFUSE_PUBLIC_KEY in .env file")
        print("   Get your keys from: https://cloud.langfuse.com")
        return
    
    if not secret_key.startswith("sk-lf-") or secret_key.endswith("-here"):
        print("⚠️  Please update LANGFUSE_SECRET_KEY in .env file")
        return
    
    print("✅ Langfuse configuration looks good!")
    
    # Test basic functionality
    try:
        from langfuse import Langfuse
        from langfuse.decorators import observe, langfuse_context
        
        # Initialize client
        langfuse = Langfuse()
        
        @observe(name="quickstart_test")
        def test_function(message: str) -> str:
            result = f"Processed: {message}"
            
            # Add a score
            langfuse_context.score_current_trace(
                name="test_score",
                value=0.95,
                comment="Quickstart test"
            )
            
            return result
        
        # Run test
        result = test_function("Hello from PromptForge!")
        print(f"✅ Test result: {result}")
        
        # Flush traces
        langfuse.flush()
        
        print("✅ Langfuse integration working!")
        print(f"🌐 Check your traces at: {os.getenv('LANGFUSE_HOST')}")
        
    except Exception as e:
        print(f"❌ Error testing Langfuse: {e}")

if __name__ == "__main__":
    main()
'''
    
    try:
        with open("quickstart_langfuse.py", "w") as f:
            f.write(quickstart_content)
        print("✅ quickstart_langfuse.py created")
        return True
    except Exception as e:
        print(f"❌ Failed to create quickstart script: {e}")
        return False

def main():
    """Main setup function"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              PromptForge Langfuse Environment Setup             ║
    ║                        Version 2.0                              ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Creating .env file", create_env_file),
        ("Verifying installation", verify_installation),
        ("Creating quickstart script", create_quickstart_script)
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if step_func():
            success_count += 1
        else:
            print(f"❌ Setup failed at step: {step_name}")
            break
    
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    
    if success_count == len(steps):
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Activate virtual environment:")
        if os.name == "nt":
            print("   .\\venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("2. Update API keys in .env file")
        print("3. Run quickstart: python quickstart_langfuse.py")
        print("4. Run example: python examples/prompt_refinement_example.py")
        
        return 0
    else:
        print(f"❌ Setup failed ({success_count}/{len(steps)} steps completed)")
        return 1

if __name__ == "__main__":
    sys.exit(main())