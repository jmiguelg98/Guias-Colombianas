#!/usr/bin/env python3
"""
Installation script for PDF to Markdown Converter
Automates the setup process and checks for dependencies
"""

import os
import sys
import subprocess
import platform
import shutil

def run_command(command, description=""):
    """Run a system command and return success status"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description or 'Command completed successfully'}")
            return True
        else:
            print(f"✗ {description or 'Command failed'}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error running command: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    if sys.version_info < (3, 7):
        print("✗ Python 3.7 or higher is required")
        return False
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
        return True

def install_pip_packages():
    """Install required Python packages"""
    print("\nInstalling Python dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("✗ requirements.txt not found")
        return False
    
    # Install packages
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Python packages installed"
    )
    
    return success

def check_tesseract():
    """Check if Tesseract is installed"""
    print("\nChecking Tesseract OCR...")
    
    # Check if tesseract command exists
    if shutil.which('tesseract'):
        print("✓ Tesseract is installed")
        return True
    else:
        print("✗ Tesseract is not installed")
        return False

def install_tesseract():
    """Provide instructions for installing Tesseract"""
    print("\nTesseract installation instructions:")
    
    system = platform.system().lower()
    
    if system == 'darwin':  # macOS
        print("For macOS, run:")
        print("  brew install tesseract")
        print("  (Requires Homebrew: https://brew.sh/)")
        
    elif system == 'linux':
        print("For Ubuntu/Debian, run:")
        print("  sudo apt-get update")
        print("  sudo apt-get install tesseract-ocr tesseract-ocr-spa")
        print("\nFor other Linux distributions, check your package manager.")
        
    elif system == 'windows':
        print("For Windows:")
        print("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Install and make sure it's added to PATH")
        
    else:
        print("Please install Tesseract OCR for your operating system")
        
    print("\nAfter installing Tesseract, run this script again to verify.")

def test_installation():
    """Test the installation by running basic tests"""
    print("\nTesting installation...")
    
    try:
        # Test Python imports
        import fitz  # PyMuPDF
        import pdfplumber
        from PIL import Image
        import pytesseract
        import pandas as pd
        import numpy as np
        print("✓ All Python dependencies can be imported")
        
        # Test Tesseract
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✓ Tesseract version: {version}")
        except pytesseract.TesseractNotFoundError:
            print("✗ Tesseract not found in PATH")
            return False
        
        # Test our converter
        from pdf_to_markdown_converter import PDFToMarkdownConverter
        converter = PDFToMarkdownConverter()
        print("✓ PDF converter can be imported and initialized")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False

def main():
    """Main installation process"""
    print("PDF to Markdown Converter - Installation Script")
    print("=" * 55)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install pip packages
    if not install_pip_packages():
        print("\nInstallation failed. Please check the errors above.")
        return
    
    # Check Tesseract
    if not check_tesseract():
        install_tesseract()
        print("\nPlease install Tesseract and run this script again.")
        return
    
    # Test installation
    if test_installation():
        print("\n" + "=" * 55)
        print("✓ Installation completed successfully!")
        print("\nYou can now run the converter with:")
        print("  python pdf_to_markdown_converter.py")
        print("\nOr test it with:")
        print("  python test_converter.py")
        
    else:
        print("\n" + "=" * 55)
        print("✗ Installation test failed.")
        print("Please check the errors above and try again.")

if __name__ == "__main__":
    main()