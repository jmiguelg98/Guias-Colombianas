#!/usr/bin/env python3
"""
Simple PDF to Markdown converter using pymupdf4llm
Converts CONSENSO_COMPLETO.pdf to markdown format while preserving tables and structure
"""

import pymupdf4llm
from pathlib import Path
from datetime import datetime

def convert_pdf_to_markdown():
    """Convert the PDF file in the workspace to markdown"""
    
    # File paths
    pdf_path = Path("CONSENSO_COMPLETO.pdf")
    output_path = Path("CONSENSO_COMPLETO.md")
    
    print("🔄 Starting PDF to Markdown conversion...")
    print(f"📄 Input PDF: {pdf_path}")
    print(f"📝 Output Markdown: {output_path}")
    
    # Check if PDF exists
    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        return
    
    try:
        print("✅ pymupdf4llm loaded successfully")
        
        # Convert PDF to markdown
        print("🔄 Converting PDF to markdown (this may take a while for large PDFs)...")
        
        # Use pymupdf4llm to convert PDF to markdown
        # This library is specifically designed for high-quality PDF to markdown conversion
        # preserving tables, headers, formatting, and structure
        markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
        
        # Write the result to markdown file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {pdf_path.stem}\n\n")
            f.write(f"*Converted from PDF on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write("---\n\n")
            f.write(markdown_text)
        
        print(f"✅ Conversion completed successfully!")
        print(f"📝 Markdown file saved as: {output_path}")
        
        # Show file size
        file_size = output_path.stat().st_size
        if file_size > 1024 * 1024:
            print(f"📊 Output file size: {file_size / (1024 * 1024):.2f} MB")
        elif file_size > 1024:
            print(f"📊 Output file size: {file_size / 1024:.2f} KB")
        else:
            print(f"📊 Output file size: {file_size} bytes")
            
        print(f"\n🎉 Your PDF has been successfully converted to markdown!")
        print(f"📁 You can now find the markdown file at: {output_path.absolute()}")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        print("💡 This might be due to:")
        print("   - Complex PDF formatting")
        print("   - PDF corruption or protection")
        print("   - Memory limitations for very large PDFs")

if __name__ == "__main__":
    convert_pdf_to_markdown()