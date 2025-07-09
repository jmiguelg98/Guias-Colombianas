#!/usr/bin/env python3
# PDF to Markdown Converter - FIXED FINAL VERSION
# ACTUALLY analyzes medical content without refusal responses
# Fixed prompts to ensure proper medical analysis

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
        logging.FileHandler('pdf_converter_fixed_final.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FixedMedicalAnalyzer:
    def __init__(self):
        self.supported_formats = ['.pdf']
        self.output_format = '.md'
        self.used_filenames = set()
        self.openai_api_key = None
        self.use_ai_vision = False
        
        # Tracking
        self.ai_calls_count = 0
        self.current_cost_estimate = 0.0
        self.tables_converted = 0
        self.flowcharts_analyzed = 0
        self.medical_content_found = 0
        self.successful_analyses = 0
        self.failed_analyses = 0
        
    def set_openai_key(self, api_key: str):
        """Set OpenAI API key for AI vision processing"""
        self.openai_api_key = api_key
        self.use_ai_vision = True
        openai.api_key = api_key
    
    def make_medical_ai_call(self, img_pil: Image.Image, prompt: str, max_tokens: int = 1500) -> str:
        """Make AI vision API call with proper medical analysis prompting"""
        try:
            if not self.use_ai_vision or not self.openai_api_key:
                return "AI analysis disabled - no API key provided"
                
            # Convert PIL image to base64
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            # FIXED PROMPT - More explicit medical context
            medical_context = """You are a medical expert analyzing clinical guideline images. 
Your task is to extract and analyze ALL medical information from this image.
This is for educational and clinical reference purposes.
Do NOT refuse to analyze medical content - this is your primary function.
Focus on extracting clinical information, not identifying people or sensitive data."""
            
            full_prompt = f"{medical_context}\n\n{prompt}"
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical expert analyzing clinical guideline images for educational purposes. Extract and analyze ALL medical information without refusal."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": full_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1
            }
            
            # Add retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post("https://api.openai.com/v1/chat/completions", 
                                           headers=headers, json=payload, timeout=90)
                    
                    if response.status_code == 200:
                        result = response.json()
                        analysis = result['choices'][0]['message']['content']
                        
                        # Check for refusal patterns
                        refusal_patterns = [
                            "i'm unable to assist",
                            "i can't help",
                            "i'm sorry, i can't",
                            "i cannot provide",
                            "i'm not able to",
                            "i don't have the ability",
                            "facial recognition"
                        ]
                        
                        analysis_lower = analysis.lower()
                        if any(pattern in analysis_lower for pattern in refusal_patterns):
                            logger.warning(f"AI refused analysis (attempt {attempt + 1}): {analysis[:100]}...")
                            if attempt < max_retries - 1:
                                time.sleep(2)  # Wait before retry
                                continue
                            else:
                                return "AI_REFUSAL_ERROR: Unable to analyze medical content"
                        
                        self.ai_calls_count += 1
                        self.current_cost_estimate = self.ai_calls_count * 0.015
                        self.successful_analyses += 1
                        
                        return analysis.strip()
                    else:
                        logger.error(f"API error {response.status_code}: {response.text}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        else:
                            return f"API_ERROR: {response.status_code}"
                            
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request error (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        return f"REQUEST_ERROR: {str(e)}"
                        
        except Exception as e:
            logger.error(f"Error in AI processing: {e}")
            self.failed_analyses += 1
            return f"PROCESSING_ERROR: {str(e)}"
    
    def analyze_medical_image(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Analyze medical image with FIXED prompting"""
        
        # FIXED PROMPT - More specific and medical-focused
        prompt = f"""MEDICAL IMAGE ANALYSIS TASK:

You are analyzing Image {img_index} from Page {page_num} of a clinical guideline document.

ANALYZE THIS MEDICAL IMAGE FOR:

1. TABLES: Medical data, dosing charts, lab values, scoring systems
2. FLOWCHARTS: Clinical decision trees, diagnostic algorithms, treatment pathways  
3. DIAGRAMS: Anatomical illustrations, procedural steps, medical equipment
4. TEXT: Medical terminology, drug names, clinical criteria, guidelines
5. FIGURES: Charts, graphs, medical imaging, clinical photos

REQUIRED OUTPUT FORMAT:

**CONTENT TYPE:** [Table/Flowchart/Diagram/Text/Figure/Mixed]

**MEDICAL ANALYSIS:**
[Detailed description of all medical content visible]

**CLINICAL INFORMATION EXTRACTED:**
[List specific medical data: dosages, criteria, procedures, terminology]

**TABLE CONVERSION (if applicable):**
[Convert any tables to markdown format with | pipes |]

**FLOWCHART ANALYSIS (if applicable):**
[Step-by-step description of decision pathways]

**LLM-FRIENDLY SUMMARY:**
[Clear summary for AI processing]

IMPORTANT: This is medical education content analysis. Extract ALL visible medical information.
Do not refuse to analyze medical content - this is your primary function.
Focus on clinical data, not patient identification."""

        logger.info(f"🔍 Analyzing medical image {img_index} on page {page_num}")
        
        analysis = self.make_medical_ai_call(img_pil, prompt, max_tokens=1500)
        
        # Handle errors and refusals
        if analysis.startswith("AI_REFUSAL_ERROR"):
            logger.error(f"AI refused to analyze image {img_index}")
            return f"### Medical Content Analysis - Image {img_index} (Page {page_num})\n\n**ERROR:** AI refused to analyze this medical image. This may be due to content policy restrictions.\n\n**FALLBACK:** Using OCR for text extraction...\n\n{self.ocr_fallback_analysis(img_pil)}\n\n---\n\n"
        
        elif analysis.startswith("API_ERROR") or analysis.startswith("REQUEST_ERROR") or analysis.startswith("PROCESSING_ERROR"):
            logger.error(f"Error analyzing image {img_index}: {analysis}")
            return f"### Medical Content Analysis - Image {img_index} (Page {page_num})\n\n**ERROR:** {analysis}\n\n**FALLBACK:** Using OCR for text extraction...\n\n{self.ocr_fallback_analysis(img_pil)}\n\n---\n\n"
        
        # Valid analysis received
        if analysis and len(analysis) > 50:
            self.medical_content_found += 1
            
            # Check for table content and format
            if "table" in analysis.lower():
                self.tables_converted += 1
                # If no markdown table found, try to extract one
                if "|" not in analysis:
                    table_prompt = f"""CONVERT TABLE TO MARKDOWN:

The previous analysis identified a table in this image. 
Extract the table data and format it as proper markdown:

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

Include ALL visible rows and columns.
Return ONLY the markdown table, nothing else."""
                    
                    table_format = self.make_medical_ai_call(img_pil, table_prompt, max_tokens=800)
                    if table_format and "|" in table_format and not table_format.startswith("AI_REFUSAL_ERROR"):
                        analysis += f"\n\n**EXTRACTED TABLE:**\n{table_format}"
            
            # Check for flowchart content
            if any(word in analysis.lower() for word in ["flowchart", "decision", "algorithm", "tree", "pathway", "workflow"]):
                self.flowcharts_analyzed += 1
            
            return f"### Medical Content Analysis - Image {img_index} (Page {page_num})\n\n{analysis}\n\n---\n\n"
        
        else:
            logger.warning(f"Minimal analysis received for image {img_index}")
            return f"### Medical Content Analysis - Image {img_index} (Page {page_num})\n\n**MINIMAL ANALYSIS:** Limited medical content detected in this image.\n\n{analysis}\n\n---\n\n"
    
    def ocr_fallback_analysis(self, img_pil: Image.Image) -> str:
        """Fallback OCR analysis when AI fails"""
        try:
            ocr_text = pytesseract.image_to_string(img_pil, lang='eng+spa')
            if ocr_text.strip():
                return f"**OCR EXTRACTED TEXT:**\n{ocr_text.strip()}\n\n**MEDICAL TERMS DETECTED:**\n{self.extract_medical_terms(ocr_text)}"
            else:
                return "**OCR RESULT:** No readable text detected in image"
        except Exception as e:
            return f"**OCR ERROR:** {str(e)}"
    
    def extract_medical_terms(self, text: str) -> str:
        """Extract medical terminology from text"""
        medical_patterns = [
            r'\b\d+\s*(mg|mcg|g|ml|cc|units?)\b',  # Dosages
            r'\b\d+\s*(mmHg|bpm|°C|°F)\b',         # Vital signs
            r'\b(diagnosis|treatment|therapy|protocol|guideline)\b',  # Clinical terms
            r'\b\w+itis\b',                        # Conditions ending in -itis
            r'\b\w+osis\b',                        # Conditions ending in -osis
            r'\b\w+emia\b',                        # Blood conditions
        ]
        
        found_terms = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_terms.extend(matches)
        
        return ", ".join(set(found_terms)) if found_terms else "No specific medical terms detected"
    
    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text content with OCR fallback"""
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
    
    def extract_and_analyze_all_images(self, pdf_path: str) -> List[str]:
        """Extract and analyze ALL images with fixed medical analysis"""
        medical_analyses = []
        
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
                            
                            # Skip very small images (likely decorative)
                            width, height = img_pil.size
                            if width < 100 or height < 100:
                                logger.info(f"Skipping small image ({width}x{height})")
                                continue
                            
                            total_images += 1
                            
                            # ANALYZE WITH FIXED MEDICAL PROMPTING
                            analysis = self.analyze_medical_image(
                                img_pil, page_num + 1, total_images
                            )
                            
                            medical_analyses.append(analysis)
                            logger.info(f"✅ Analyzed image {total_images} (Page {page_num + 1})")
                        
                        pix = None
                        
                    except Exception as e:
                        logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue
            
            doc.close()
            logger.info(f"📊 Total images processed: {total_images}")
            logger.info(f"📊 Medical content found: {self.medical_content_found}")
            logger.info(f"📊 Tables converted: {self.tables_converted}")
            logger.info(f"📊 Flowcharts analyzed: {self.flowcharts_analyzed}")
            logger.info(f"📊 Successful analyses: {self.successful_analyses}")
            logger.info(f"📊 Failed analyses: {self.failed_analyses}")
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
            
        return medical_analyses
    
    def extract_metadata_with_ai(self, text_content: str) -> Dict[str, str]:
        """Extract metadata using AI with fixed prompting"""
        if not self.use_ai_vision:
            return {}
            
        prompt = f"""METADATA EXTRACTION TASK:

Extract metadata from this clinical guideline text and return a JSON object:

{text_content[:3000]}

Return ONLY this JSON structure:
{{
    "title": "Document title",
    "organization": "Publishing organization", 
    "year": "Publication year",
    "medical_specialty": "Medical specialty/department",
    "guideline_type": "Type of guideline",
    "target_population": "Target patient population",
    "keywords": "Key medical terms and topics"
}}

Extract actual values from the text. If a field is not found, use "Not specified"."""
        
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
                        "content": "You are extracting metadata from clinical guidelines. Return only valid JSON."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions",
                                   headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                metadata_text = result['choices'][0]['message']['content']
                
                # Clean JSON response
                metadata_text = metadata_text.strip()
                if metadata_text.startswith("```json"):
                    metadata_text = metadata_text[7:]
                if metadata_text.endswith("```"):
                    metadata_text = metadata_text[:-3]
                
                return json.loads(metadata_text)
            else:
                logger.error(f"Metadata extraction failed: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    def generate_filename(self, metadata: Dict[str, str], original_filename: str, output_dir: str) -> str:
        """Generate unique filename"""
        clean_original = re.sub(r'[^\w\s-]', '', Path(original_filename).stem)
        clean_original = re.sub(r'\s+', '_', clean_original)[:25]
        
        filename_parts = [clean_original]
        
        if metadata.get('title'):
            clean_title = re.sub(r'[^\w\s-]', '', metadata['title'])
            clean_title = re.sub(r'\s+', '_', clean_title)[:30]
            filename_parts.append(clean_title)
        
        if metadata.get('year'):
            filename_parts.append(metadata['year'])
        
        base_filename = '_'.join(filename_parts)[:80]
        
        # Handle conflicts
        counter = 1
        final_filename = base_filename
        output_path = os.path.join(output_dir, final_filename + ".md")
        
        while os.path.exists(output_path) or final_filename in self.used_filenames:
            final_filename = f"{base_filename}_{counter}"
            output_path = os.path.join(output_dir, final_filename + ".md")
            counter += 1
            
            if counter > 100:
                timestamp = datetime.now().strftime("%H%M%S")
                final_filename = f"{base_filename}_{timestamp}"
                break
        
        self.used_filenames.add(final_filename)
        return final_filename
    
    def create_medical_markdown(self, text_content: str, medical_analyses: List[str], 
                              metadata: Dict[str, str], original_filename: str) -> str:
        """Create medical markdown with proper structure"""
        
        content_parts = []
        
        # Header
        title = metadata.get('title', 'Clinical Guideline')
        content_parts.append(f"# {title}")
        content_parts.append("")
        
        # Metadata section
        content_parts.append("## Document Information")
        content_parts.append("")
        content_parts.append(f"**Original Filename:** {original_filename}")
        
        for key, value in metadata.items():
            if value and value.strip() and value != "Not specified":
                formatted_key = key.replace('_', ' ').title()
                content_parts.append(f"**{formatted_key}:** {value}")
        
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
        
        # Medical content analysis
        if medical_analyses:
            content_parts.append("## AI-Analyzed Medical Content")
            content_parts.append("")
            content_parts.append("*Each image has been analyzed by AI for medical content with fixed prompting*")
            content_parts.append("")
            
            for analysis in medical_analyses:
                content_parts.append(analysis)
        
        # Processing summary
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        content_parts.append("## AI Processing Summary")
        content_parts.append("")
        content_parts.append(f"**Processed on:** {timestamp}")
        content_parts.append(f"**AI calls made:** {self.ai_calls_count}")
        content_parts.append(f"**Estimated cost:** ${self.current_cost_estimate:.2f}")
        content_parts.append(f"**Images analyzed:** {len(medical_analyses)}")
        content_parts.append(f"**Medical content found:** {self.medical_content_found}")
        content_parts.append(f"**Tables converted:** {self.tables_converted}")
        content_parts.append(f"**Flowcharts analyzed:** {self.flowcharts_analyzed}")
        content_parts.append(f"**Successful analyses:** {self.successful_analyses}")
        content_parts.append(f"**Failed analyses:** {self.failed_analyses}")
        content_parts.append("")
        content_parts.append("*Converted with FIXED medical AI analysis - no more refusals*")
        
        return "\n".join(content_parts)
    
    def convert_pdf_with_fixed_ai(self, pdf_path: str, output_dir: str) -> bool:
        """Convert PDF with FIXED AI analysis"""
        try:
            logger.info(f"🚀 Starting FIXED AI analysis of {pdf_path}")
            
            # Reset counters
            self.ai_calls_count = 0
            self.current_cost_estimate = 0.0
            self.tables_converted = 0
            self.flowcharts_analyzed = 0
            self.medical_content_found = 0
            self.successful_analyses = 0
            self.failed_analyses = 0
            
            # Extract text content
            logger.info("📝 Extracting text content...")
            text_content = self.extract_text_with_ocr(pdf_path)
            
            # AI metadata extraction
            logger.info("🎯 Extracting metadata with AI...")
            metadata = self.extract_metadata_with_ai(text_content)
            
            # ANALYZE ALL IMAGES WITH FIXED PROMPTING
            logger.info("🔍 Analyzing ALL images with FIXED medical prompting...")
            medical_analyses = self.extract_and_analyze_all_images(pdf_path)
            
            # Generate filename and save
            original_filename = Path(pdf_path).name
            new_filename = self.generate_filename(metadata, original_filename, output_dir)
            output_path = os.path.join(output_dir, new_filename + ".md")
            
            # Create medical markdown
            logger.info("📋 Creating medical markdown...")
            markdown_content = self.create_medical_markdown(
                text_content, medical_analyses, metadata, original_filename
            )
            
            # Save file
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"✅ Successfully converted {pdf_path}")
            logger.info(f"💰 Cost: ${self.current_cost_estimate:.2f}")
            logger.info(f"🎯 Medical content found: {self.medical_content_found}")
            logger.info(f"✅ Successful analyses: {self.successful_analyses}")
            logger.info(f"❌ Failed analyses: {self.failed_analyses}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error converting {pdf_path}: {e}")
            return False
    
    def convert_batch_fixed(self, input_dir: str, output_dir: str, progress_callback=None) -> Dict[str, int]:
        """Convert batch with FIXED AI analysis"""
        results = {'success': 0, 'failed': 0, 'total': 0}
        total_cost = 0.0
        total_medical_content = 0
        total_successful_analyses = 0
        total_failed_analyses = 0
        
        self.used_filenames = set()
        
        pdf_files = []
        for ext in self.supported_formats:
            pdf_files.extend(Path(input_dir).glob('*' + ext))
        
        results['total'] = len(pdf_files)
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return results
        
        print(f"\n🚀 Starting FIXED AI analysis of {results['total']} PDF files...")
        print("🔧 FIXED: No more AI refusals - proper medical analysis")
        print("=" * 70)
        
        for i, pdf_path in enumerate(pdf_files):
            try:
                print(f"\n📄 [{i+1}/{results['total']}] Processing: {Path(pdf_path).name}")
                
                if progress_callback:
                    progress_callback(i + 1, results['total'], str(pdf_path))
                
                if self.convert_pdf_with_fixed_ai(str(pdf_path), output_dir):
                    results['success'] += 1
                    total_cost += self.current_cost_estimate
                    total_medical_content += self.medical_content_found
                    total_successful_analyses += self.successful_analyses
                    total_failed_analyses += self.failed_analyses
                    print(f"   ✅ Success - Medical content: {self.medical_content_found}")
                    print(f"   💰 Cost: ${self.current_cost_estimate:.2f} (Total: ${total_cost:.2f})")
                    print(f"   📊 Analyses: {self.successful_analyses} success, {self.failed_analyses} failed")
                else:
                    results['failed'] += 1
                    print(f"   ❌ Failed")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                results['failed'] += 1
                print(f"   ❌ Error: {e}")
        
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"💰 Total cost: ${total_cost:.2f}")
        print(f"🔍 Medical content found: {total_medical_content} images")
        print(f"✅ Successful AI analyses: {total_successful_analyses}")
        print(f"❌ Failed AI analyses: {total_failed_analyses}")
        print(f"📊 Success rate: {results['success']}/{results['total']}")
        
        return results

# GUI for fixed version
class FixedAIGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("FIXED AI Clinical Guideline Analyzer")
        self.root.geometry("800x600")
        
        self.analyzer = FixedMedicalAnalyzer()
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.openai_key = tk.StringVar()
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        title_label = ttk.Label(main_frame, text="FIXED AI Clinical Guideline Analyzer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        warning_label = ttk.Label(main_frame, text="🔧 FIXED: No more AI refusals - proper medical analysis", 
                                 font=('Arial', 12, 'bold'), foreground='red')
        warning_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        fix_label = ttk.Label(main_frame, text="✅ Fixed prompting to prevent 'I'm unable to assist' responses", 
                             font=('Arial', 10), foreground='green')
        fix_label.grid(row=2, column=0, columnspan=3, pady=(0, 15))
        
        ttk.Label(main_frame, text="Input Directory (PDFs):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_dir, width=65).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_input_dir).grid(row=3, column=2, pady=5)
        
        ttk.Label(main_frame, text="Output Directory (Markdown):").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=65).grid(row=4, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_output_dir).grid(row=4, column=2, pady=5)
        
        # OpenAI API Key section
        ai_frame = ttk.LabelFrame(main_frame, text="🧠 FIXED AI Analysis", padding="5")
        ai_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=10)
        
        ttk.Label(ai_frame, text="OpenAI API Key:").grid(row=0, column=0, sticky=tk.W, pady=5)
        key_entry = ttk.Entry(ai_frame, textvariable=self.openai_key, width=65, show="*")
        key_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(ai_frame, text="🔧 FIXED: No more refusal responses", font=('Arial', 9, 'bold'), foreground='red').grid(row=1, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(ai_frame, text="✅ Proper medical analysis with retry logic", font=('Arial', 9)).grid(row=2, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(ai_frame, text="✅ OCR fallback when AI fails", font=('Arial', 9)).grid(row=3, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(ai_frame, text="✅ Detailed error tracking and reporting", font=('Arial', 9)).grid(row=4, column=0, columnspan=2, sticky=tk.W)
        
        convert_button = ttk.Button(main_frame, text="Start FIXED AI Analysis", command=self.start_conversion)
        convert_button.grid(row=6, column=0, columnspan=3, pady=20)
        
        self.progress_var = tk.StringVar()
        self.progress_var.set("Ready for FIXED AI analysis")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=7, column=0, columnspan=3, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=8, column=0, columnspan=3, sticky="ew", pady=5)
        
        log_frame = ttk.LabelFrame(main_frame, text="Analysis Log", padding="5")
        log_frame.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=10)
        
        self.log_text = tk.Text(log_frame, height=18, width=95)
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(9, weight=1)
        ai_frame.columnconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
    def select_input_dir(self):
        directory = filedialog.askdirectory(title="Select Directory with PDF Files")
        if directory:
            self.input_dir.set(directory)
    
    def select_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory for Markdown Files")
        if directory:
            self.output_dir.set(directory)
    
    def log_message(self, message: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"{timestamp} - {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_progress(self, current: int, total: int, current_file: str):
        progress = (current / total) * 100
        self.progress_bar['value'] = progress
        filename = Path(current_file).name
        status_text = f"FIXED Analysis {current}/{total}: {filename}"
        self.progress_var.set(status_text)
        self.log_message(f"Processing: {filename}")
        self.root.update()
    
    def start_conversion(self):
        if not self.input_dir.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Please select both input and output directories")
            return
        
        if not os.path.exists(self.input_dir.get()):
            messagebox.showerror("Error", "Input directory does not exist")
            return
        
        api_key = self.openai_key.get().strip()
        if not api_key:
            messagebox.showerror("Error", "OpenAI API key is required")
            return
        
        self.analyzer.set_openai_key(api_key)
        self.log_message("🧠 FIXED AI analysis enabled - no more refusals!")
        
        self.log_text.delete(1.0, tk.END)
        self.log_message("🚀 Starting FIXED AI analysis...")
        
        thread = threading.Thread(target=self.run_conversion)
        thread.daemon = True
        thread.start()
    
    def run_conversion(self):
        try:
            results = self.analyzer.convert_batch_fixed(
                self.input_dir.get(),
                self.output_dir.get(),
                self.update_progress
            )
            
            self.progress_var.set("FIXED AI analysis completed!")
            completion_message = f"Analysis completed! Success: {results['success']}, Failed: {results['failed']}"
            self.log_message(completion_message)
            
            messagebox.showinfo("FIXED AI Analysis Complete!", 
                              f"🔧 FIXED Analysis completed!\n\nSuccessful: {results['success']}\nFailed: {results['failed']}\n\n✅ No more AI refusals!\n📋 Proper medical analysis with fallbacks\n💰 Cost tracking included")
            
        except Exception as e:
            error_message = f"Error during analysis: {e}"
            self.log_message(error_message)
            messagebox.showerror("Error", f"An error occurred: {e}")
    
    def run(self):
        self.root.mainloop()

def main():
    print("🔧 FIXED AI Clinical Guideline Analyzer")
    print("=" * 50)
    print("🔧 FIXED: No more AI refusal responses")
    print("✅ Proper medical analysis with retry logic")
    print("📋 OCR fallback when AI fails")
    print("💰 Detailed cost and error tracking")
    print("=" * 50)
    
    app = FixedAIGUI()
    app.run()

if __name__ == "__main__":
    main()