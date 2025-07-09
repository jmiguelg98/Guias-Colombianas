#!/usr/bin/env python3
"""
CLI Medical PDF Converter - WORKING VERSION
ACTUALLY analyzes medical content with meaningful descriptions
Converts tables to LLM-friendly format and provides real clinical analysis
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from PDF_to_markdown_converter_AI_WORKING import WorkingAIAnalyzer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pdf_converter_cli_working.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='WORKING AI Clinical Guideline Analyzer - CLI Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single PDF with AI analysis
  python PDF_to_markdown_converter_CLI_WORKING.py -i "guideline.pdf" -o "output/" -k "your-api-key"
  
  # Convert entire directory
  python PDF_to_markdown_converter_CLI_WORKING.py -i "pdf_folder/" -o "markdown_output/" -k "your-api-key"
  
  # Show processing details
  python PDF_to_markdown_converter_CLI_WORKING.py -i "pdf_folder/" -o "output/" -k "your-api-key" -v

Features:
  ✅ ACTUALLY analyzes medical content with meaningful descriptions
  📋 Converts tables to LLM-friendly markdown format  
  🌳 Provides step-by-step flowchart analysis
  🎯 Real medical terminology extraction
  💰 Cost tracking (approx $0.50-3.00 per document)
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input PDF file or directory containing PDFs')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for markdown files')
    parser.add_argument('-k', '--api-key', required=True,
                       help='OpenAI API key for AI analysis')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = WorkingAIAnalyzer()
    analyzer.set_openai_key(args.api_key)
    
    print("🚀 WORKING AI Clinical Guideline Analyzer - CLI")
    print("=" * 60)
    print("✅ ACTUALLY analyzes medical content")
    print("📋 Converts tables to LLM-friendly format")
    print("🌳 Provides meaningful flowchart descriptions")
    print("🎯 Real medical content extraction")
    print("=" * 60)
    
    try:
        if input_path.is_file():
            # Single file processing
            print(f"\n📄 Processing single file: {input_path.name}")
            success = analyzer.convert_pdf_with_working_ai(str(input_path), str(output_path))
            if success:
                print(f"✅ Successfully processed {input_path.name}")
                print(f"💰 Cost: ${analyzer.current_cost_estimate:.2f}")
                print(f"🎯 Medical content found: {analyzer.medical_content_found} images")
                print(f"📋 Tables converted: {analyzer.tables_converted}")
                print(f"🌳 Flowcharts analyzed: {analyzer.flowcharts_analyzed}")
            else:
                print(f"❌ Failed to process {input_path.name}")
                sys.exit(1)
        else:
            # Directory processing
            print(f"\n📁 Processing directory: {input_path}")
            results = analyzer.convert_batch(str(input_path), str(output_path))
            
            print(f"\n🎯 PROCESSING COMPLETE!")
            print(f"✅ Successful: {results['success']}")
            print(f"❌ Failed: {results['failed']}")
            print(f"📊 Total processed: {results['total']}")
            print(f"💰 Estimated total cost: ${analyzer.current_cost_estimate:.2f}")
            
            if results['failed'] > 0:
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()