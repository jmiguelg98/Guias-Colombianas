#!/usr/bin/env python3
"""
PDF to Markdown Converter using MegaParse
==========================================

This script converts PDF files to markdown format while preserving:
- Text content
- Tables 
- Images
- Figures
- Headers and footers
- Table of contents

Requirements:
- megaparse library
- OpenAI or Anthropic API key
- poppler (for image processing)
- tesseract (for OCR)
- libmagic (Mac only)

Usage:
    python pdf_to_markdown_converter.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Optional
import argparse
from datetime import datetime

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        from megaparse import MegaParse
        from langchain_openai import ChatOpenAI
        print("✓ MegaParse and dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nTo install required packages, run:")
        print("pip install megaparse langchain-openai")
        return False

def setup_environment():
    """Setup environment variables for API keys."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  OPENAI_API_KEY environment variable not found.")
        api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print("✓ API key set for this session")
        else:
            print("⚠️  No API key provided. MegaParse Vision features will be limited.")
    else:
        print("✓ OpenAI API key found in environment")
    
    return api_key

def browse_for_pdf_file() -> Optional[Path]:
    """Try to open a file browser on macOS."""
    try:
        if platform.system() == "Darwin":  # macOS
            # Use AppleScript to open file dialog
            script = '''
            tell application "System Events"
                set theFile to choose file with prompt "Select a PDF file:" of type {"pdf"}
                return POSIX path of theFile
            end tell
            '''
            
            result = subprocess.run(['osascript', '-e', script], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                file_path = result.stdout.strip()
                if file_path:
                    return Path(file_path)
    except Exception as e:
        print(f"Could not open file browser: {e}")
    
    return None

def get_pdf_path() -> Path:
    """Get PDF file path from user input."""
    while True:
        print("\n" + "="*50)
        print("📁 PDF File Selection")
        print("="*50)
        print("How would you like to select your PDF file?")
        print("1. Browse and select (opens file dialog)")
        print("2. Enter the file path manually")
        print()
        
        method = input("Enter your choice (1 or 2): ").strip()
        
        if method == "1":
            # Try to use file browser
            pdf_file = browse_for_pdf_file()
            if pdf_file and pdf_file.exists():
                print(f"✅ Selected file: {pdf_file}")
                return pdf_file
            else:
                print("❌ File selection cancelled or failed. Let's try manual entry.")
                # Fall through to manual entry
        
        # Manual entry (method == "2" or fallback from method == "1")
        print("\n💡 Tips for entering the path:")
        print("   • You can drag and drop the file into the terminal")
        print("   • Use Tab completion for easier typing")
        print("   • Path should start with / or ~")
        print("   • Example: /Users/yourname/Documents/file.pdf")
        print("   • Or: ~/Documents/file.pdf")
        print()
        
        pdf_path = input("Enter the full path to your PDF file: ").strip()
        
        # Debug: Show what we received
        print(f"🔍 Debug: Received path: '{pdf_path}'")
        
        # Remove quotes if present (common when dragging files)
        pdf_path = pdf_path.strip('"\'')
        print(f"🔍 Debug: After removing quotes: '{pdf_path}'")
        
        # Handle empty input
        if not pdf_path:
            print("❌ No path entered. Please try again.")
            continue
        
        # Expand user home directory (~)
        if pdf_path.startswith('~'):
            pdf_path = os.path.expanduser(pdf_path)
            print(f"🔍 Debug: After expanding ~: '{pdf_path}'")
        
        # Convert to Path object and resolve any relative paths
        try:
            pdf_file = Path(pdf_path).resolve()
            print(f"🔍 Debug: Resolved path: '{pdf_file}'")
        except Exception as e:
            print(f"❌ Invalid path format: {e}")
            continue
        
        # Check if file exists
        if not pdf_file.exists():
            print(f"❌ File does not exist: {pdf_file}")
            print("🔍 Troubleshooting:")
            print(f"   • Check if the file is at: {pdf_file}")
            print("   • Make sure you have permission to access the file")
            print("   • Try dragging and dropping the file into the terminal")
            
            # Check if parent directory exists
            if pdf_file.parent.exists():
                print(f"   • Parent directory exists: {pdf_file.parent}")
                print("   • Files in that directory:")
                try:
                    for file in pdf_file.parent.iterdir():
                        if file.suffix.lower() == '.pdf':
                            print(f"     - {file.name}")
                except Exception as e:
                    print(f"     Error listing files: {e}")
            else:
                print(f"   • Parent directory does not exist: {pdf_file.parent}")
            continue
        
        # Check if it's a PDF file
        if not pdf_file.suffix.lower() == '.pdf':
            print(f"❌ File is not a PDF: {pdf_file}")
            print(f"   File extension: {pdf_file.suffix}")
            continue
        
        # Check if we can read the file
        try:
            with open(pdf_file, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
        except PermissionError:
            print(f"❌ Permission denied reading file: {pdf_file}")
            continue
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            continue
        
        print(f"✅ PDF file found and accessible: {pdf_file}")
        return pdf_file

def get_output_folder() -> Path:
    """Get output folder path from user input."""
    while True:
        print("\n" + "="*50)
        print("📁 Output Folder Selection")
        print("="*50)
        print("Where do you want to save the markdown file?")
        print("1. Same folder as the PDF file")
        print("2. Specify a different folder")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            return None  # Will use PDF's parent directory
        elif choice == "2":
            print("\n💡 Tips for entering the folder path:")
            print("   • You can drag and drop the folder into the terminal")
            print("   • Use Tab completion for easier typing")
            print("   • Example: /Users/yourname/Desktop/converted_docs")
            print("   • Or: ~/Desktop/converted_docs")
            print()
            
            folder_path = input("Enter the full path to the output folder: ").strip()
            
            # Debug: Show what we received
            print(f"🔍 Debug: Received folder path: '{folder_path}'")
            
            # Remove quotes if present
            folder_path = folder_path.strip('"\'')
            print(f"🔍 Debug: After removing quotes: '{folder_path}'")
            
            # Handle empty input
            if not folder_path:
                print("❌ No folder path entered. Please try again.")
                continue
            
            # Expand user home directory (~)
            if folder_path.startswith('~'):
                folder_path = os.path.expanduser(folder_path)
                print(f"🔍 Debug: After expanding ~: '{folder_path}'")
            
            # Convert to Path object and resolve
            try:
                output_folder = Path(folder_path).resolve()
                print(f"🔍 Debug: Resolved folder path: '{output_folder}'")
            except Exception as e:
                print(f"❌ Invalid folder path format: {e}")
                continue
            
            # Create folder if it doesn't exist
            try:
                output_folder.mkdir(parents=True, exist_ok=True)
                print(f"✅ Output folder ready: {output_folder}")
                return output_folder
            except Exception as e:
                print(f"❌ Cannot create or access folder: {e}")
                print("🔍 Troubleshooting:")
                print(f"   • Check if you have permission to create folders at: {output_folder}")
                print("   • Try using a different location like ~/Desktop")
                continue
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")

def convert_pdf_to_markdown(pdf_path: Path, output_folder: Optional[Path] = None, use_vision: bool = False) -> Path:
    """Convert PDF to markdown using MegaParse."""
    
    # Import here to avoid issues if dependencies aren't installed
    from megaparse import MegaParse
    
    # Determine output folder
    if output_folder is None:
        output_folder = pdf_path.parent
    
    # Generate output filename
    output_filename = f"{pdf_path.stem}_converted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    output_path = output_folder / output_filename
    
    print(f"\n🔄 Converting PDF to markdown...")
    print(f"   Input:  {pdf_path}")
    print(f"   Output: {output_path}")
    
    try:
        # Initialize MegaParse
        megaparse = MegaParse()
        
        # Convert the PDF
        print("   Processing... This may take a while depending on PDF size.")
        response = megaparse.load(str(pdf_path))
        
        # Save to markdown file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response)
        
        print(f"✅ Conversion completed successfully!")
        print(f"   Markdown file saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        print("\nTroubleshooting tips:")
        print("- Ensure your PDF file is not corrupted")
        print("- Check that you have sufficient disk space")
        print("- Verify your API key is correct (if using vision mode)")
        print("- Try with a smaller PDF file first")
        sys.exit(1)

def convert_pdf_with_vision(pdf_path: Path, output_folder: Optional[Path] = None) -> Path:
    """Convert PDF to markdown using MegaParse Vision (higher quality)."""
    
    # Import here to avoid issues if dependencies aren't installed
    from megaparse.parser.megaparse_vision import MegaParseVision
    from langchain_openai import ChatOpenAI
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for Vision mode")
        return None
    
    # Determine output folder
    if output_folder is None:
        output_folder = pdf_path.parent
    
    # Generate output filename
    output_filename = f"{pdf_path.stem}_vision_converted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    output_path = output_folder / output_filename
    
    print(f"\n🔄 Converting PDF to markdown using Vision mode...")
    print(f"   Input:  {pdf_path}")
    print(f"   Output: {output_path}")
    
    try:
        # Initialize the model
        model = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
        parser = MegaParseVision(model=model)
        
        # Convert the PDF
        print("   Processing with AI vision... This may take longer but provides higher quality.")
        response = parser.convert(str(pdf_path))
        
        # Save to markdown file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response)
        
        print(f"✅ Vision conversion completed successfully!")
        print(f"   Markdown file saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error during vision conversion: {e}")
        print("\nFalling back to standard conversion...")
        return convert_pdf_to_markdown(pdf_path, output_folder)

def main():
    """Main function to run the PDF to Markdown converter."""
    print("=" * 60)
    print("PDF to Markdown Converter using MegaParse")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install the required packages and try again.")
        sys.exit(1)
    
    # Setup environment
    api_key = setup_environment()
    
    # Get PDF file path
    pdf_path = get_pdf_path()
    
    # Get output folder
    output_folder = get_output_folder()
    
    # Ask about conversion mode
    if api_key:
        print("\nChoose conversion mode:")
        print("1. Standard conversion (faster)")
        print("2. Vision conversion (higher quality, uses AI, costs API credits)")
        
        mode_choice = input("Enter your choice (1 or 2): ").strip()
        use_vision = mode_choice == "2"
    else:
        use_vision = False
    
    # Convert the PDF
    try:
        if use_vision:
            output_path = convert_pdf_with_vision(pdf_path, output_folder)
        else:
            output_path = convert_pdf_to_markdown(pdf_path, output_folder)
        
        if output_path:
            # Display file info
            file_size = output_path.stat().st_size
            print(f"\n📄 Output file information:")
            print(f"   Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"   Location: {output_path.absolute()}")
            
            # Ask if user wants to open the file
            if input("\nWould you like to open the output folder? (y/n): ").lower().startswith('y'):
                try:
                    if platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", output_folder])
                    elif platform.system() == "Windows":
                        subprocess.run(["explorer", output_folder])
                    else:  # Linux
                        subprocess.run(["xdg-open", output_folder])
                except Exception as e:
                    print(f"Could not open folder automatically: {e}")
                    print(f"You can manually navigate to: {output_folder}")
    
    except KeyboardInterrupt:
        print("\n\n❌ Conversion cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()