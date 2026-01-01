#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated dependency installation script
Automatically detects and installs all required dependencies
"""

import subprocess
import sys
import platform

def print_header(text):
    """Print header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def print_step(text):
    """Print step"""
    print(f"  {text}")

def run_command(cmd, description=""):
    """Run command"""
    if description:
        print_step(description)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=False,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  Error: {e}")
        return False

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"  Python version: {version_str}")
    
    if version.major >= 3 and version.minor >= 8:
        print("  Python version OK (>= 3.8)\n")
        return True
    else:
        print("  ERROR: Python 3.8 or higher required\n")
        return False

def check_os():
    """Check operating system"""
    print_header("Checking Operating System")
    
    os_type = platform.system()
    print(f"  OS: {os_type}")
    
    if os_type == "Windows":
        print("  Windows system detected\n")
        return "windows"
    elif os_type == "Darwin":
        print("  macOS system detected\n")
        return "macos"
    elif os_type == "Linux":
        print("  Linux system detected\n")
        return "linux"
    else:
        print(f"  ERROR: Unsupported OS: {os_type}\n")
        return None

def upgrade_pip():
    """Upgrade pip"""
    print_header("Upgrading pip")
    
    cmd = f"{sys.executable} -m pip install --upgrade pip"
    return run_command(cmd, "Upgrading pip...")

def install_dependencies():
    """Install all dependencies"""
    print_header("Installing Dependencies")
    
    dependencies = [
        ("PyQt5==5.15.9", "PyQt5 GUI Framework"),
        ("PyQt5-sip==12.13.0", "PyQt5 SIP Bindings"),
        ("pandas>=1.3.0", "Data Processing"),
        ("numpy>=1.20.0", "Numerical Computing"),
        ("matplotlib>=3.4.0", "Chart Rendering"),
        ("huggingface-hub>=0.10.0", "HuggingFace Data Source"),
    ]
    
    failed = []
    
    for package, description in dependencies:
        print(f"\n  Installing {description} ({package})")
        cmd = f"{sys.executable} -m pip install {package}"
        success = run_command(cmd)
        
        if not success:
            failed.append((package, description))
            print(f"    ERROR: Failed to install {package}")
        else:
            print(f"    OK: {package}")
    
    return failed

def verify_imports():
    """Verify all modules can be imported"""
    print_header("Verifying Imports")
    
    modules = [
        ("PyQt5.QtWidgets", "PyQt5.QtWidgets"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib.pyplot", "matplotlib"),
        ("huggingface_hub", "huggingface_hub"),
    ]
    
    failed = []
    
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  OK: {display_name}")
        except ImportError as e:
            print(f"  ERROR: {display_name}: {e}")
            failed.append((module_name, display_name))
    
    return failed

def create_test_script():
    """Create test script"""
    print_header("Creating Test Script")
    
    test_code = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test all dependencies"""

import sys

def test_imports():
    modules = {
        "PyQt5.QtWidgets": "PyQt5",
        "pandas": "pandas",
        "numpy": "numpy",
        "matplotlib.pyplot": "matplotlib",
        "huggingface_hub": "huggingface_hub",
    }
    
    failed = []
    for module_name, display_name in modules.items():
        try:
            __import__(module_name)
            print(f"  OK: {display_name}")
        except ImportError as e:
            print(f"  ERROR: {display_name}: {e}")
            failed.append(display_name)
    
    return len(failed) == 0

if __name__ == "__main__":
    print("="*60)
    print("  Dependency Verification")
    print("="*60)
    print()
    
    if test_imports():
        print()
        print("  All dependencies verified successfully!")
        print("  Run: python model_ensemble_gui.py")
        sys.exit(0)
    else:
        print()
        print("  Some dependencies failed verification")
        sys.exit(1)
'''
    
    with open("test_dependencies.py", "w", encoding="utf-8") as f:
        f.write(test_code)
    
    print("  Created: test_dependencies.py")

def main():
    """Main function"""
    print("\n")
    print("  Crypto Currency K-Line Analysis System")
    print("  Automated Dependency Installation")
    print()
    
    # 1. Check Python version
    if not check_python_version():
        return False
    
    # 2. Check operating system
    os_type = check_os()
    if not os_type:
        return False
    
    # 3. Upgrade pip
    if not upgrade_pip():
        print("  WARNING: pip upgrade failed, continuing...")
    
    # 4. Install dependencies
    failed_deps = install_dependencies()
    
    if failed_deps:
        print("\n  WARNING: Some dependencies failed to install:")
        for package, description in failed_deps:
            print(f"    - {package} ({description})")
    
    # 5. Verify imports
    failed_imports = verify_imports()
    
    # 6. Create test script
    create_test_script()
    
    # 7. Summary
    print_header("Installation Summary")
    
    if not failed_imports:
        print("  SUCCESS: All dependencies installed and verified!")
        print("\n  Next steps:")
        print("    1. Run: python model_ensemble_gui.py")
        print("    2. Or verify: python test_dependencies.py")
        return True
    else:
        print("  WARNING: The following modules failed to import:")
        for module_name, display_name in failed_imports:
            print(f"    - {display_name}")
        print("\n  Please refer to INSTALL_GUIDE.md for troubleshooting")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
