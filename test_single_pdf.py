#!/usr/bin/env python3
"""
Simple test script to verify the FIXED AI analysis works on a single PDF
"""

import sys
import os
from pathlib import Path

def test_single_pdf():
    """Test the fixed AI analysis on a single PDF"""
    
    # Get inputs
    print("🔧 FIXED AI Test - Single PDF")
    print("=" * 40)
    
    pdf_path = input("Enter PDF file path: ").strip()
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    output_dir = input("Enter output directory (or press Enter for current): ").strip()
    if not output_dir:
        output_dir = "."
    
    api_key = input("Enter OpenAI API key: ").strip()
    if not api_key:
        print("❌ API key required")
        return False
    
    # Import the fixed analyzer
    try:
        from PDF_to_markdown_converter_AI_FIXED_FINAL import FixedMedicalAnalyzer
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure PDF_to_markdown_converter_AI_FIXED_FINAL.py is in the same directory")
        return False
    
    # Initialize analyzer
    analyzer = FixedMedicalAnalyzer()
    analyzer.set_openai_key(api_key)
    
    print(f"\n🚀 Testing FIXED AI analysis on: {Path(pdf_path).name}")
    print("🔧 This version should NOT give refusal responses!")
    
    # Convert PDF
    success = analyzer.convert_pdf_with_fixed_ai(pdf_path, output_dir)
    
    if success:
        print(f"\n✅ SUCCESS!")
        print(f"💰 Cost: ${analyzer.current_cost_estimate:.2f}")
        print(f"🎯 Medical content found: {analyzer.medical_content_found} images")
        print(f"📋 Tables converted: {analyzer.tables_converted}")
        print(f"🌳 Flowcharts analyzed: {analyzer.flowcharts_analyzed}")
        print(f"✅ Successful analyses: {analyzer.successful_analyses}")
        print(f"❌ Failed analyses: {analyzer.failed_analyses}")
        
        if analyzer.failed_analyses > 0:
            print(f"\n⚠️  Some analyses failed - check the log file for details")
            
        return True
    else:
        print(f"\n❌ FAILED to convert PDF")
        return False

if __name__ == "__main__":
    test_single_pdf()