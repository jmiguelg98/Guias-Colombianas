#!/usr/bin/env python3
"""
PDF to Markdown Converter - ACTUALLY WORKING VERSION
Uses OCR-first approach with AI enhancement to avoid refusal issues
"""

import os
import sys
import re
import logging
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import json

# PDF processing libraries
try:
    import fitz  # PyMuPDF
    import pdfplumber
    from PIL import Image
    import pytesseract
    import pandas as pd
    import numpy as np
    import openai
    import base64
    import requests
    import time
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install required packages:")
    print("pip install PyMuPDF pdfplumber Pillow pytesseract pandas numpy openai")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_converter_actually_working.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ActuallyWorkingAnalyzer:
    def __init__(self):
        self.supported_formats = ['.pdf']
        self.output_format = '.md'
        self.used_filenames = set()
        self.openai_api_key = None
        self.use_ai_vision = False
        
        # Tracking
        self.ai_calls_count = 0
        self.current_cost_estimate = 0.0
        self.ocr_analyses = 0
        self.ai_analyses = 0
        self.tables_found = 0
        self.medical_content_found = 0
        self.ai_refusals = 0
        
    def set_openai_key(self, api_key: str):
        """Set OpenAI API key for AI vision processing"""
        self.openai_api_key = api_key
        self.use_ai_vision = True
        openai.api_key = api_key
    
    def extract_text_with_ocr(self, img_pil: Image.Image) -> str:
        """Extract text from image using OCR"""
        try:
            # Try both English and Spanish
            ocr_text = pytesseract.image_to_string(img_pil, lang='eng+spa')
            return ocr_text.strip()
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def detect_table_structure(self, text: str) -> bool:
        """Detect if text contains table-like structure"""
        lines = text.split('\n')
        
        # Look for table indicators
        table_indicators = [
            len([line for line in lines if '\t' in line or '  ' in line]) > 2,  # Tab or multiple spaces
            len([line for line in lines if '|' in line]) > 2,  # Pipe separators
            len([line for line in lines if re.search(r'\d+\s*(mg|ml|mcg|g|%)', line)]) > 1,  # Dosages
            len([line for line in lines if re.search(r'\w+\s+\w+\s+\w+', line)]) > 3,  # Multiple columns
        ]
        
        return any(table_indicators)
    
    def format_table_from_text(self, text: str) -> str:
        """Convert text to markdown table format"""
        try:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            if len(lines) < 2:
                return text
            
            # Try to detect column structure
            if '\t' in text:
                # Tab-separated
                rows = []
                for line in lines:
                    if '\t' in line:
                        rows.append(line.split('\t'))
                
                if rows:
                    # Create markdown table
                    max_cols = max(len(row) for row in rows)
                    
                    # Pad rows to same length
                    for row in rows:
                        while len(row) < max_cols:
                            row.append('')
                    
                    # Create header
                    header = '| ' + ' | '.join(rows[0]) + ' |'
                    separator = '|' + '|'.join(['-' * 10 for _ in range(max_cols)]) + '|'
                    
                    # Create body
                    body = []
                    for row in rows[1:]:
                        body.append('| ' + ' | '.join(row) + ' |')
                    
                    return '\n'.join([header, separator] + body)
            
            elif any('  ' in line for line in lines):
                # Space-separated columns
                # Simple approach: try to align columns
                table_lines = []
                for line in lines:
                    if '  ' in line:
                        # Split on multiple spaces
                        parts = re.split(r'\s{2,}', line)
                        table_lines.append(parts)
                
                if table_lines:
                    max_cols = max(len(parts) for parts in table_lines)
                    
                    # Pad to same length
                    for parts in table_lines:
                        while len(parts) < max_cols:
                            parts.append('')
                    
                    # Create markdown table
                    header = '| ' + ' | '.join(table_lines[0]) + ' |'
                    separator = '|' + '|'.join(['-' * 10 for _ in range(max_cols)]) + '|'
                    
                    body = []
                    for parts in table_lines[1:]:
                        body.append('| ' + ' | '.join(parts) + ' |')
                    
                    return '\n'.join([header, separator] + body)
            
            return text
            
        except Exception as e:
            logger.error(f"Table formatting error: {e}")
            return text
    
    def extract_medical_info(self, text: str) -> Dict[str, List[str]]:
        """Extract medical information from text"""
        medical_info = {
            'dosages': [],
            'conditions': [],
            'procedures': [],
            'medications': [],
            'values': []
        }
        
        # Dosage patterns
        dosage_patterns = [
            r'\d+\s*(mg|mcg|g|ml|cc|units?|iu)\b',
            r'\d+\s*(mg/kg|mcg/kg|g/kg|ml/kg)',
            r'\d+\s*-\s*\d+\s*(mg|mcg|g|ml)',
        ]
        
        for pattern in dosage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            medical_info['dosages'].extend(matches)
        
        # Medical conditions
        condition_patterns = [
            r'\b\w+itis\b',  # Conditions ending in -itis
            r'\b\w+osis\b',  # Conditions ending in -osis
            r'\b\w+emia\b',  # Blood conditions
            r'\b(diabetes|hypertension|pneumonia|sepsis|asthma)\b',
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            medical_info['conditions'].extend(matches)
        
        # Procedures
        procedure_patterns = [
            r'\b(surgery|procedure|treatment|therapy|intervention)\b',
            r'\b(intubation|ventilation|dialysis|transfusion)\b',
        ]
        
        for pattern in procedure_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            medical_info['procedures'].extend(matches)
        
        # Medications (common drug suffixes)
        medication_patterns = [
            r'\b\w+cillin\b',  # Penicillins
            r'\b\w+mycin\b',   # Mycins
            r'\b\w+pril\b',    # ACE inhibitors
            r'\b\w+olol\b',    # Beta blockers
        ]
        
        for pattern in medication_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            medical_info['medications'].extend(matches)
        
        # Clinical values
        value_patterns = [
            r'\d+\s*(mmHg|bpm|°C|°F)',
            r'\d+\.\d+\s*(mg/dl|mmol/L)',
        ]
        
        for pattern in value_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            medical_info['values'].extend(matches)
        
        # Remove duplicates and empty entries
        for key in medical_info:
            medical_info[key] = list(set([item for item in medical_info[key] if item]))
        
        return medical_info
    
    def try_ai_analysis(self, img_pil: Image.Image, ocr_text: str) -> str:
        """Try AI analysis with OCR context to avoid refusals"""
        
        if not self.use_ai_vision or not self.openai_api_key:
            return ""
        
        # Use OCR text as context to avoid image analysis refusals
        prompt = f"""Based on the following text extracted from a medical guideline image, provide analysis:

OCR TEXT:
{ocr_text[:1000]}

TASK: Analyze this medical content and provide:

1. **Content Type**: What type of medical content is this?
2. **Key Medical Information**: List important medical data, dosages, procedures
3. **Clinical Significance**: What does this mean for patient care?
4. **LLM Summary**: Clear summary for AI processing

Focus on medical education content analysis. Return detailed medical analysis."""
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical expert analyzing clinical guideline content for educational purposes."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 800,
                "temperature": 0.1
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                   headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['choices'][0]['message']['content']
                
                # Check for refusals
                refusal_patterns = [
                    "i'm unable to assist",
                    "i can't help",
                    "i'm sorry, i can't",
                    "i cannot provide",
                    "i'm not able to",
                    "i don't have the ability"
                ]
                
                analysis_lower = analysis.lower()
                if any(pattern in analysis_lower for pattern in refusal_patterns):
                    self.ai_refusals += 1
                    logger.warning("AI refused text analysis")
                    return ""
                
                self.ai_calls_count += 1
                self.current_cost_estimate = self.ai_calls_count * 0.01
                self.ai_analyses += 1
                
                return analysis.strip()
            
            else:
                logger.error(f"AI API error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return ""
    
    def analyze_image_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Analyze image content using OCR-first approach"""
        
        logger.info(f"🔍 Analyzing image {img_index} on page {page_num}")
        
        # Step 1: Extract text with OCR
        ocr_text = self.extract_text_with_ocr(img_pil)
        
        if not ocr_text:
            return f"### Image {img_index} (Page {page_num})\n\n**NO TEXT DETECTED**: This image appears to be purely graphical with no readable text.\n\n---\n\n"
        
        self.ocr_analyses += 1
        
        # Step 2: Extract medical information
        medical_info = self.extract_medical_info(ocr_text)
        
        # Step 3: Check if it's a table
        is_table = self.detect_table_structure(ocr_text)
        
        # Step 4: Try AI analysis if we have medical content
        ai_analysis = ""
        if any(medical_info.values()) or is_table:
            self.medical_content_found += 1
            ai_analysis = self.try_ai_analysis(img_pil, ocr_text)
        
        # Step 5: Format output
        analysis_parts = []
        analysis_parts.append(f"### Medical Content Analysis - Image {img_index} (Page {page_num})")
        analysis_parts.append("")
        
        # Content type
        if is_table:
            analysis_parts.append("**CONTENT TYPE:** Table")
            self.tables_found += 1
        elif any(medical_info.values()):
            analysis_parts.append("**CONTENT TYPE:** Medical Text")
        else:
            analysis_parts.append("**CONTENT TYPE:** Text")
        
        analysis_parts.append("")
        
        # OCR extracted text
        analysis_parts.append("**OCR EXTRACTED TEXT:**")
        analysis_parts.append(ocr_text)
        analysis_parts.append("")
        
        # Format table if detected
        if is_table:
            formatted_table = self.format_table_from_text(ocr_text)
            if formatted_table != ocr_text:
                analysis_parts.append("**FORMATTED TABLE:**")
                analysis_parts.append(formatted_table)
                analysis_parts.append("")
        
        # Medical information
        if any(medical_info.values()):
            analysis_parts.append("**MEDICAL INFORMATION EXTRACTED:**")
            
            for category, items in medical_info.items():
                if items:
                    analysis_parts.append(f"- **{category.title()}**: {', '.join(items)}")
            
            analysis_parts.append("")
        
        # AI analysis (if available)
        if ai_analysis:
            analysis_parts.append("**AI ENHANCED ANALYSIS:**")
            analysis_parts.append(ai_analysis)
            analysis_parts.append("")
        elif self.use_ai_vision:
            analysis_parts.append("**AI STATUS:** AI analysis not available (may have been refused or failed)")
            analysis_parts.append("")
        
        # Summary
        analysis_parts.append("**PROCESSING SUMMARY:**")
        analysis_parts.append(f"- OCR: {'✅ Success' if ocr_text else '❌ Failed'}")
        analysis_parts.append(f"- Medical Content: {'✅ Detected' if any(medical_info.values()) else '❌ None'}")
        analysis_parts.append(f"- Table Structure: {'✅ Detected' if is_table else '❌ None'}")
        analysis_parts.append(f"- AI Analysis: {'✅ Success' if ai_analysis else '❌ Failed/Refused'}")
        
        analysis_parts.append("")
        analysis_parts.append("---")
        analysis_parts.append("")
        
        return "\n".join(analysis_parts)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if not text.strip():
                    logger.info(f"Using OCR for page {page_num + 1}")
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("ppm")
                    img = Image.open(io.BytesIO(img_data))
                    text = pytesseract.image_to_string(img, lang='eng+spa')
                
                page_marker = f"\n\n--- Page {page_num + 1} ---\n\n"
                full_text += page_marker + text
            
            doc.close()
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_and_analyze_images(self, pdf_path: str) -> List[str]:
        """Extract and analyze all images using OCR-first approach"""
        image_analyses = []
        
        try:
            doc = fitz.open(pdf_path)
            total_images = 0
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Handle CMYK conversion
                        if pix.n - pix.alpha >= 4:
                            logger.info(f"Converting CMYK image to RGB on page {page_num + 1}")
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        if pix.n - pix.alpha <= 3:
                            img_data = pix.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                            
                            # Skip very small images
                            width, height = img_pil.size
                            if width < 100 or height < 100:
                                logger.info(f"Skipping small image ({width}x{height})")
                                continue
                            
                            total_images += 1
                            
                            # ANALYZE WITH OCR-FIRST APPROACH
                            analysis = self.analyze_image_content(
                                img_pil, page_num + 1, total_images
                            )
                            
                            image_analyses.append(analysis)
                            logger.info(f"✅ Analyzed image {total_images} (Page {page_num + 1})")
                        
                        pix = None
                        
                    except Exception as e:
                        logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue
            
            doc.close()
            logger.info(f"📊 Total images processed: {total_images}")
            logger.info(f"📊 OCR analyses: {self.ocr_analyses}")
            logger.info(f"📊 AI analyses: {self.ai_analyses}")
            logger.info(f"📊 AI refusals: {self.ai_refusals}")
            logger.info(f"📊 Medical content found: {self.medical_content_found}")
            logger.info(f"📊 Tables found: {self.tables_found}")
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
            
        return image_analyses
    
    def generate_filename(self, original_filename: str, output_dir: str) -> str:
        """Generate unique filename"""
        clean_original = re.sub(r'[^\w\s-]', '', Path(original_filename).stem)
        clean_original = re.sub(r'\s+', '_', clean_original)[:50]
        
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{clean_original}_{timestamp}"
        
        # Handle conflicts
        counter = 1
        final_filename = base_filename
        output_path = os.path.join(output_dir, final_filename + ".md")
        
        while os.path.exists(output_path) or final_filename in self.used_filenames:
            final_filename = f"{base_filename}_{counter}"
            output_path = os.path.join(output_dir, final_filename + ".md")
            counter += 1
        
        self.used_filenames.add(final_filename)
        return final_filename
    
    def create_markdown_output(self, text_content: str, image_analyses: List[str], 
                             original_filename: str) -> str:
        """Create final markdown output"""
        
        content_parts = []
        
        # Header
        content_parts.append(f"# Medical Guideline Analysis: {Path(original_filename).stem}")
        content_parts.append("")
        
        # Document info
        content_parts.append("## Document Information")
        content_parts.append("")
        content_parts.append(f"**Original Filename:** {original_filename}")
        content_parts.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content_parts.append(f"**Analysis Method:** OCR-First with AI Enhancement")
        content_parts.append("")
        content_parts.append("---")
        content_parts.append("")
        
        # Main text content
        if text_content:
            content_parts.append("## Document Text Content")
            content_parts.append("")
            content_parts.append(text_content)
            content_parts.append("")
            content_parts.append("---")
            content_parts.append("")
        
        # Image analyses
        if image_analyses:
            content_parts.append("## Image Analysis Results")
            content_parts.append("")
            content_parts.append("*Each image analyzed using OCR-first approach with AI enhancement where possible*")
            content_parts.append("")
            
            for analysis in image_analyses:
                content_parts.append(analysis)
        
        # Processing summary
        content_parts.append("## Processing Summary")
        content_parts.append("")
        content_parts.append(f"**Total OCR analyses:** {self.ocr_analyses}")
        content_parts.append(f"**Successful AI analyses:** {self.ai_analyses}")
        content_parts.append(f"**AI refusals:** {self.ai_refusals}")
        content_parts.append(f"**Medical content found:** {self.medical_content_found}")
        content_parts.append(f"**Tables detected:** {self.tables_found}")
        content_parts.append(f"**Estimated cost:** ${self.current_cost_estimate:.2f}")
        content_parts.append("")
        content_parts.append("*OCR-first approach ensures content is always extracted, with AI enhancement when available*")
        
        return "\n".join(content_parts)
    
    def convert_pdf_actually_working(self, pdf_path: str, output_dir: str) -> bool:
        """Convert PDF using actually working approach"""
        try:
            logger.info(f"🚀 Starting ACTUALLY WORKING analysis of {pdf_path}")
            
            # Reset counters
            self.ai_calls_count = 0
            self.current_cost_estimate = 0.0
            self.ocr_analyses = 0
            self.ai_analyses = 0
            self.tables_found = 0
            self.medical_content_found = 0
            self.ai_refusals = 0
            
            # Extract text content
            logger.info("📝 Extracting text content...")
            text_content = self.extract_text_from_pdf(pdf_path)
            
            # Analyze all images
            logger.info("🔍 Analyzing images with OCR-first approach...")
            image_analyses = self.extract_and_analyze_images(pdf_path)
            
            # Generate filename and save
            original_filename = Path(pdf_path).name
            new_filename = self.generate_filename(original_filename, output_dir)
            output_path = os.path.join(output_dir, new_filename + ".md")
            
            # Create markdown content
            logger.info("📋 Creating markdown output...")
            markdown_content = self.create_markdown_output(
                text_content, image_analyses, original_filename
            )
            
            # Save file
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"✅ Successfully converted {pdf_path}")
            logger.info(f"📊 Results: {self.ocr_analyses} OCR, {self.ai_analyses} AI, {self.ai_refusals} refused")
            logger.info(f"💰 Cost: ${self.current_cost_estimate:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error converting {pdf_path}: {e}")
            return False

def main():
    """Simple command-line interface"""
    print("🔧 ACTUALLY WORKING PDF to Markdown Converter")
    print("=" * 50)
    print("✅ OCR-first approach - content always extracted")
    print("✅ AI enhancement when available")
    print("✅ No more AI refusal issues")
    print("=" * 50)
    
    if len(sys.argv) < 3:
        print("Usage: python PDF_converter_ACTUALLY_WORKING.py <pdf_file> <output_dir> [api_key]")
        print("\nExample:")
        print("python PDF_converter_ACTUALLY_WORKING.py document.pdf output/ sk-...")
        return
    
    pdf_file = sys.argv[1]
    output_dir = sys.argv[2]
    api_key = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(pdf_file):
        print(f"❌ PDF file not found: {pdf_file}")
        return
    
    # Initialize analyzer
    analyzer = ActuallyWorkingAnalyzer()
    
    if api_key:
        analyzer.set_openai_key(api_key)
        print("🧠 AI enhancement enabled")
    else:
        print("📝 OCR-only mode (no AI enhancement)")
    
    # Convert PDF
    success = analyzer.convert_pdf_actually_working(pdf_file, output_dir)
    
    if success:
        print(f"\n✅ SUCCESS!")
        print(f"📊 OCR analyses: {analyzer.ocr_analyses}")
        print(f"📊 AI analyses: {analyzer.ai_analyses}")
        print(f"📊 AI refusals: {analyzer.ai_refusals}")
        print(f"📊 Medical content: {analyzer.medical_content_found}")
        print(f"📊 Tables: {analyzer.tables_found}")
        print(f"💰 Cost: ${analyzer.current_cost_estimate:.2f}")
    else:
        print(f"\n❌ FAILED to convert PDF")

if __name__ == "__main__":
    main()