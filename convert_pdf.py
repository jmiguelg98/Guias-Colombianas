#!/usr/bin/env python3
"""
Simple PDF to Markdown converter using MegaParse
Converts CONSENSO_COMPLETO.pdf to markdown format
"""

import os
import asyncio
from pathlib import Path
from datetime import datetime

async def convert_pdf_to_markdown():
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
        # Import megaparse
        from megaparse.sdk import MegaParseSDK
        from megaparse.sdk.megaparse_sdk.utils.type import ParserType, StrategyEnum, Language
        
        print("✅ MegaParseSDK imported successfully")
        
        # Initialize MegaParseSDK
        parser = MegaParseSDK()
        print("✅ MegaParseSDK initialized")
        
        # Convert PDF to markdown
        print("🔄 Converting PDF to markdown (this may take a while)...")
        
        # Upload and parse the PDF
        response = await parser.file.upload(
            file_path=str(pdf_path),
            method=ParserType.UNSTRUCTURED,  # Using unstructured parser
            strategy=StrategyEnum.AUTO,
            check_table=True,  # Enable table detection
            language=Language.ENGLISH
        )
        
        # Check if the response is successful
        if response.status_code == 200:
            result = response.json()
            
            # Extract the parsed content
            if 'content' in result:
                content = result['content']
            elif 'text' in result:
                content = result['text']
            else:
                content = str(result)
                
            # Write the result to markdown file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# {pdf_path.stem}\n\n")
                f.write(f"*Converted on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write("---\n\n")
                f.write(content)
        else:
            print(f"❌ Upload failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        print(f"✅ Conversion completed successfully!")
        print(f"📝 Markdown file saved as: {output_path}")
        print(f"📊 File size: {output_path.stat().st_size / 1024:.1f} KB")
        
    except ImportError as e:
        print(f"❌ Error importing megaparse: {e}")
        print("💡 Make sure you have activated the virtual environment and installed megaparse")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        print("💡 This might be due to:")
        print("   - Complex PDF formatting")
        print("   - Missing API keys for advanced features")
        print("   - PDF corruption or protection")

if __name__ == "__main__":
    asyncio.run(convert_pdf_to_markdown())