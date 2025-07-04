#!/usr/bin/env python3
"""
PDF to Markdown Converter
=========================
A simple script to convert PDF files to markdown format while preserving tables and structure.

Requirements:
- Python 3.6+
- pymupdf4llm (install with: pip install pymupdf4llm)

Usage:
    python pdf_to_markdown_converter.py

Author: AI Assistant
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def install_dependencies():
    """Install required dependencies if not available"""
    try:
        import pymupdf4llm
        return True
    except ImportError:
        print("📦 Installing required dependencies...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf4llm"])
            import pymupdf4llm
            print("✅ Dependencies installed successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Please run: pip install pymupdf4llm")
            return False

def get_pdf_path():
    """Get PDF file path from user"""
    while True:
        print("\n" + "="*60)
        print("📁 PDF FILE SELECTION")
        print("="*60)
        print("💡 Tips:")
        print("   • Drag and drop your PDF file into this terminal window")
        print("   • Or type the full path to your PDF file")
        print("   • Press Ctrl+C to exit")
        print()
        
        try:
            pdf_path = input("Enter the path to your PDF file: ").strip()
            
            # Remove quotes if present (from drag and drop)
            pdf_path = pdf_path.strip('"\'')
            
            # Expand user home directory
            pdf_path = os.path.expanduser(pdf_path)
            
            # Convert to Path object
            pdf_file = Path(pdf_path)
            
            # Check if file exists
            if not pdf_file.exists():
                print(f"❌ File not found: {pdf_file}")
                continue
            
            # Check if it's a PDF
            if not pdf_file.suffix.lower() == '.pdf':
                print(f"❌ File is not a PDF: {pdf_file}")
                continue
            
            print(f"✅ PDF file found: {pdf_file}")
            return pdf_file
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

def get_output_folder(pdf_path):
    """Get output folder path from user"""
    print("\n" + "="*60)
    print("� OUTPUT FOLDER SELECTION")
    print("="*60)
    print("Where would you like to save the markdown file?")
    print()
    print("1. Same folder as the PDF file")
    print("2. Choose a different folder")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            
            if choice == "1":
                return pdf_path.parent
            elif choice == "2":
                folder_path = input("Enter the output folder path: ").strip()
                folder_path = folder_path.strip('"\'')
                folder_path = os.path.expanduser(folder_path)
                
                output_folder = Path(folder_path)
                
                # Create folder if it doesn't exist
                output_folder.mkdir(parents=True, exist_ok=True)
                print(f"✅ Output folder: {output_folder}")
                return output_folder
            else:
                print("❌ Please enter 1 or 2")
                continue
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

def convert_pdf_to_markdown(pdf_path, output_folder):
    """Convert PDF to markdown"""
    try:
        import pymupdf4llm
        
        # Generate output filename
        output_filename = pdf_path.stem + ".md"
        output_path = output_folder / output_filename
        
        print("\n" + "="*60)
        print("🔄 CONVERTING PDF TO MARKDOWN")
        print("="*60)
        print(f"📄 Input:  {pdf_path}")
        print(f"📝 Output: {output_path}")
        print()
        print("⏳ Converting... (this may take a while for large PDFs)")
        
        # Convert PDF to markdown
        markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {pdf_path.stem}\n\n")
            f.write(f"*Converted from PDF on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write("---\n\n")
            f.write(markdown_text)
        
        # Show results
        file_size = output_path.stat().st_size
        print("\n" + "="*60)
        print("🎉 CONVERSION COMPLETED!")
        print("="*60)
        print(f"✅ Markdown file created: {output_path}")
        
        if file_size > 1024 * 1024:
            print(f"📊 File size: {file_size / (1024 * 1024):.2f} MB")
        elif file_size > 1024:
            print(f"📊 File size: {file_size / 1024:.2f} KB")
        else:
            print(f"📊 File size: {file_size} bytes")
        
        # Count lines
        with open(output_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f)
        print(f"📄 Lines: {line_count:,}")
        
        # Check for tables
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            table_count = content.count('|---')
        if table_count > 0:
            print(f"📊 Tables found: {table_count}")
        
        print(f"\n🎯 Your PDF has been successfully converted!")
        print(f"📁 Location: {output_path.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        print("💡 Possible causes:")
        print("   - PDF is password protected")
        print("   - PDF is corrupted")
        print("   - Insufficient memory for very large PDFs")
        print("   - Missing dependencies")
        return False

def main():
    """Main function"""
    print("🚀 PDF to Markdown Converter")
    print("=" * 60)
    print("Convert PDF files to markdown while preserving tables and structure")
    print()
    
    # Check dependencies
    if not install_dependencies():
        return
    
    # Get PDF file
    pdf_path = get_pdf_path()
    
    # Get output folder
    output_folder = get_output_folder(pdf_path)
    
    # Convert
    success = convert_pdf_to_markdown(pdf_path, output_folder)
    
    if success:
        print("\n✨ Conversion completed successfully!")
    else:
        print("\n💥 Conversion failed!")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()