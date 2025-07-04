#!/usr/bin/env python3
"""
Simple Path Tester for macOS
============================

This script helps diagnose file path issues on macOS.
Run this first to make sure your PDF file can be found.
"""

import os
from pathlib import Path

def test_file_path():
    """Test if we can find and access a PDF file."""
    print("🔍 macOS File Path Tester")
    print("=" * 40)
    print("This will help diagnose why your PDF file isn't being found.")
    print()
    
    # Get current working directory
    cwd = Path.cwd()
    print(f"📁 Current working directory: {cwd}")
    
    # Get home directory
    home = Path.home()
    print(f"🏠 Home directory: {home}")
    
    # List common locations for PDF files
    print("\n📄 Looking for PDF files in common locations:")
    
    common_locations = [
        home / "Downloads",
        home / "Documents", 
        home / "Desktop",
        cwd,
    ]
    
    for location in common_locations:
        if location.exists():
            print(f"\n📂 {location}:")
            try:
                pdf_files = list(location.glob("*.pdf"))
                if pdf_files:
                    for pdf_file in pdf_files[:5]:  # Show first 5
                        print(f"   ✅ {pdf_file.name}")
                        print(f"      Full path: {pdf_file}")
                else:
                    print("   (no PDF files found)")
            except Exception as e:
                print(f"   ❌ Error accessing folder: {e}")
    
    # Interactive test
    print("\n" + "="*40)
    print("🧪 Interactive Path Test")
    print("="*40)
    
    while True:
        path_input = input("\nEnter a file path to test (or 'quit' to exit): ").strip()
        
        if path_input.lower() == 'quit':
            break
        
        if not path_input:
            continue
        
        print(f"\n🔍 Testing path: '{path_input}'")
        
        # Clean up the path
        cleaned_path = path_input.strip('"\'')
        print(f"   After removing quotes: '{cleaned_path}'")
        
        # Expand home directory
        if cleaned_path.startswith('~'):
            expanded_path = os.path.expanduser(cleaned_path)
            print(f"   After expanding ~: '{expanded_path}'")
        else:
            expanded_path = cleaned_path
        
        # Create Path object
        try:
            file_path = Path(expanded_path).resolve()
            print(f"   Resolved path: '{file_path}'")
        except Exception as e:
            print(f"   ❌ Error creating path: {e}")
            continue
        
        # Test if exists
        if file_path.exists():
            print(f"   ✅ File exists!")
            
            # Test if it's a file
            if file_path.is_file():
                print(f"   ✅ It's a file")
                
                # Test if it's a PDF
                if file_path.suffix.lower() == '.pdf':
                    print(f"   ✅ It's a PDF file")
                    
                    # Test if readable
                    try:
                        with open(file_path, 'rb') as f:
                            f.read(1024)
                        print(f"   ✅ File is readable")
                        print(f"   📊 File size: {file_path.stat().st_size:,} bytes")
                    except Exception as e:
                        print(f"   ❌ Cannot read file: {e}")
                else:
                    print(f"   ❌ Not a PDF file (extension: {file_path.suffix})")
            else:
                print(f"   ❌ Not a file (might be a directory)")
        else:
            print(f"   ❌ File does not exist")
            
            # Check if parent directory exists
            parent = file_path.parent
            if parent.exists():
                print(f"   📁 Parent directory exists: {parent}")
                print(f"   📄 Files in parent directory:")
                try:
                    for file in parent.iterdir():
                        if file.is_file():
                            print(f"      - {file.name}")
                        if len(list(parent.iterdir())) > 10:
                            print(f"      ... and more files")
                            break
                except Exception as e:
                    print(f"      Error listing files: {e}")
            else:
                print(f"   ❌ Parent directory doesn't exist: {parent}")

if __name__ == "__main__":
    test_file_path()