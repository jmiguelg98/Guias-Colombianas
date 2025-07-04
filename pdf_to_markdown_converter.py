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

def get_pdf_path() -> Path:
    """Get PDF file path from user input."""
    while True:
        pdf_path = input("\nEnter the full path to your PDF file: ").strip()
        
        # Remove quotes if present
        pdf_path = pdf_path.strip('"\'')
        
        # Convert to Path object
        pdf_file = Path(pdf_path)
        
        # Check if file exists
        if not pdf_file.exists():
            print(f"❌ File does not exist: {pdf_file}")
            continue
        
        # Check if it's a PDF file
        if not pdf_file.suffix.lower() == '.pdf':
            print(f"❌ File is not a PDF: {pdf_file}")
            continue
        
        print(f"✓ PDF file found: {pdf_file}")
        return pdf_file

def get_output_folder() -> Path:
    """Get output folder path from user input."""
    while True:
        print("\nWhere do you want to save the markdown file?")
        print("1. Same folder as the PDF file")
        print("2. Specify a different folder")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            return None  # Will use PDF's parent directory
        elif choice == "2":
            folder_path = input("Enter the full path to the output folder: ").strip()
            folder_path = folder_path.strip('"\'')
            
            output_folder = Path(folder_path)
            
            # Create folder if it doesn't exist
            try:
                output_folder.mkdir(parents=True, exist_ok=True)
                print(f"✓ Output folder ready: {output_folder}")
                return output_folder
            except Exception as e:
                print(f"❌ Cannot create or access folder: {e}")
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
                import subprocess
                import platform
                
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