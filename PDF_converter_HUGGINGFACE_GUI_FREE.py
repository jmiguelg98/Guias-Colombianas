#!/usr/bin/env python3
"""
PDF to Markdown Converter - FREE HUGGING FACE GUI VERSION
Uses open-source vision-language models with easy-to-use GUI
Completely FREE - no API costs!
"""

import os
import sys
import re
import logging
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

# PDF processing libraries
try:
    import fitz  # PyMuPDF
    import pdfplumber
    from PIL import Image
    import pytesseract
    import pandas as pd
    import numpy as np
    
    # Hugging Face libraries
    import torch
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        AutoProcessor, AutoModelForCausalLM,
        pipeline
    )
    import warnings
    warnings.filterwarnings("ignore")
    
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install required packages:")
    print("pip install PyMuPDF pdfplumber Pillow pytesseract pandas numpy")
    print("pip install torch torchvision transformers accelerate")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_converter_huggingface_gui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HuggingFaceVisionAnalyzer:
    """Free vision analysis using Hugging Face models"""
    
    def __init__(self, progress_callback=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        self.progress_callback = progress_callback
        
        # Model components
        self.blip_processor = None
        self.blip_model = None
        self.llava_processor = None
        self.llava_model = None
        
        # Analysis tracking
        self.analyses_performed = 0
        self.tables_found = 0
        self.flowcharts_found = 0
        self.figures_found = 0
        self.text_content_found = 0
        
        self.log_message(f"🤗 Hugging Face Analyzer initialized on {self.device}")
        
    def log_message(self, message: str):
        """Send message to GUI if callback available"""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message)
        
    def load_models(self):
        """Load the vision-language models"""
        if self.models_loaded:
            return
            
        self.log_message("🔄 Loading Hugging Face vision models...")
        self.log_message("⚠️  First time setup may take a few minutes to download models")
        
        try:
            # Load BLIP-2 for general image understanding
            self.log_message("📥 Loading BLIP-2 model...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.blip_model = self.blip_model.to(self.device)
            
            self.log_message("✅ BLIP-2 model loaded successfully")
            
            # Try to load LLaVA for detailed analysis (optional)
            try:
                self.log_message("📥 Loading LLaVA model for detailed analysis...")
                self.llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
                self.llava_model = AutoModelForCausalLM.from_pretrained(
                    "llava-hf/llava-1.5-7b-hf",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                if self.device == "cpu":
                    self.llava_model = self.llava_model.to(self.device)
                    
                self.log_message("✅ LLaVA model loaded successfully")
                
            except Exception as e:
                self.log_message(f"⚠️  LLaVA model not available: {e}")
                self.log_message("🔄 Continuing with BLIP-2 only")
                self.llava_model = None
                self.llava_processor = None
            
            self.models_loaded = True
            self.log_message("🎉 Vision models ready for analysis!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.log_message(f"❌ Error loading models: {e}")
            self.log_message("💡 Try running: pip install torch torchvision transformers accelerate")
            raise
    
    def classify_image_content(self, img_pil: Image.Image) -> str:
        """Classify image content type using BLIP-2"""
        try:
            if not self.models_loaded:
                self.load_models()
            
            # Classification prompt
            prompt = "What type of medical content is shown in this image: table, flowchart, figure, or text?"
            
            inputs = self.blip_processor(img_pil, prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.blip_model.generate(**inputs, max_length=50, num_beams=3)
            
            response = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean and categorize response
            response_lower = response.lower()
            
            if "table" in response_lower:
                return "TABLE"
            elif "flowchart" in response_lower or "flow" in response_lower or "algorithm" in response_lower:
                return "FLOWCHART"
            elif "figure" in response_lower or "diagram" in response_lower:
                return "FIGURE"
            elif "text" in response_lower:
                return "TEXT"
            else:
                return "MIXED"
                
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return "UNKNOWN"
    
    def analyze_with_detailed_model(self, img_pil: Image.Image, prompt: str) -> str:
        """Use LLaVA for detailed analysis if available, otherwise BLIP-2"""
        try:
            if self.llava_model is not None and self.llava_processor is not None:
                # Use LLaVA for detailed analysis
                inputs = self.llava_processor(prompt, img_pil, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.llava_model.generate(**inputs, max_length=512, num_beams=3)
                
                response = self.llava_processor.decode(outputs[0], skip_special_tokens=True)
                return response
                
            elif self.blip_model is not None and self.blip_processor is not None:
                # Fallback to BLIP-2
                inputs = self.blip_processor(img_pil, prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.blip_model.generate(**inputs, max_length=256, num_beams=3)
                
                response = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
                return response
            else:
                return "Models not loaded yet"
                
        except Exception as e:
            logger.error(f"Error in detailed analysis: {e}")
            return f"Error analyzing image: {e}"
    
    def analyze_table_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Analyze table content using vision models"""
        
        prompt = """Analyze this medical table image in detail. Extract:
1. All table headers and column names
2. All data values, numbers, dosages, ranges
3. Any units of measurement
4. Clinical significance of the data
5. Key medical information

Describe what medical information this table contains and its clinical purpose."""
        
        self.log_message(f"📊 Analyzing table {img_index} on page {page_num} with Hugging Face")
        
        analysis = self.analyze_with_detailed_model(img_pil, prompt)
        self.tables_found += 1
        
        # Try to extract structured data using OCR as supplement
        try:
            ocr_text = pytesseract.image_to_string(img_pil, lang='eng+spa')
            if ocr_text.strip():
                analysis += f"\n\n**OCR Extracted Text:**\n{ocr_text.strip()}"
                
                # Try to format as table if structured
                lines = ocr_text.strip().split('\n')
                if len(lines) > 2 and any('\t' in line or '  ' in line for line in lines):
                    analysis += f"\n\n**Structured Data Detected:**\nThis appears to contain tabular data that could be formatted as a markdown table."
        
        except Exception as e:
            logger.warning(f"OCR failed for table: {e}")
        
        return f"### Table Analysis - Image {img_index} (Page {page_num})\n\n**Content Type:** Table\n\n**AI Analysis:**\n{analysis}\n\n---\n\n"
    
    def analyze_flowchart_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Analyze flowchart/decision tree content"""
        
        prompt = """Analyze this medical flowchart or decision tree. Describe:
1. The clinical decision process shown
2. Each decision point and criteria
3. Different pathways and outcomes
4. Any medical values, thresholds, or criteria mentioned
5. The clinical purpose and how healthcare providers would use this

Map out the decision-making process step by step."""
        
        self.log_message(f"🌳 Analyzing flowchart {img_index} on page {page_num} with Hugging Face")
        
        analysis = self.analyze_with_detailed_model(img_pil, prompt)
        self.flowcharts_found += 1
        
        # Supplement with OCR for text elements
        try:
            ocr_text = pytesseract.image_to_string(img_pil, lang='eng+spa')
            if ocr_text.strip():
                analysis += f"\n\n**Text Elements Extracted:**\n{ocr_text.strip()}"
        except Exception as e:
            logger.warning(f"OCR failed for flowchart: {e}")
        
        return f"### Flowchart Analysis - Image {img_index} (Page {page_num})\n\n**Content Type:** Flowchart/Decision Tree\n\n**AI Analysis:**\n{analysis}\n\n---\n\n"
    
    def analyze_figure_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Analyze figure/diagram content"""
        
        prompt = """Analyze this medical figure or diagram. Describe:
1. What medical concept, anatomy, or process is illustrated
2. All visible text, labels, and annotations
3. Key visual elements and their relationships
4. Clinical or educational purpose
5. Any measurements, scales, or reference values shown

Provide a detailed description that would help someone understand the medical content."""
        
        self.log_message(f"🔬 Analyzing figure {img_index} on page {page_num} with Hugging Face")
        
        analysis = self.analyze_with_detailed_model(img_pil, prompt)
        self.figures_found += 1
        
        # Supplement with OCR for any text
        try:
            ocr_text = pytesseract.image_to_string(img_pil, lang='eng+spa')
            if ocr_text.strip():
                analysis += f"\n\n**Text/Labels Extracted:**\n{ocr_text.strip()}"
        except Exception as e:
            logger.warning(f"OCR failed for figure: {e}")
        
        return f"### Figure Analysis - Image {img_index} (Page {page_num})\n\n**Content Type:** Figure/Diagram\n\n**AI Analysis:**\n{analysis}\n\n---\n\n"
    
    def analyze_text_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Analyze text content in images"""
        
        prompt = """Analyze this medical text content. Extract and describe:
1. All visible text content
2. Key medical recommendations or guidelines
3. Important clinical information
4. Any structured lists, bullet points, or sections
5. Medical terminology and its context

Provide a clear summary of the medical information presented."""
        
        self.log_message(f"📝 Analyzing text content {img_index} on page {page_num} with Hugging Face")
        
        analysis = self.analyze_with_detailed_model(img_pil, prompt)
        self.text_content_found += 1
        
        # OCR is essential for text content
        try:
            ocr_text = pytesseract.image_to_string(img_pil, lang='eng+spa')
            if ocr_text.strip():
                analysis += f"\n\n**Complete Text Extraction:**\n{ocr_text.strip()}"
        except Exception as e:
            logger.warning(f"OCR failed for text: {e}")
        
        return f"### Text Content Analysis - Image {img_index} (Page {page_num})\n\n**Content Type:** Text\n\n**AI Analysis:**\n{analysis}\n\n---\n\n"
    
    def analyze_mixed_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Analyze mixed content"""
        
        prompt = """This image contains mixed medical content (combination of tables, text, figures, etc.). 
Analyze and describe:
1. All different types of content present
2. How the different elements relate to each other
3. Key medical information from each component
4. Overall clinical purpose and significance
5. Any data, values, or recommendations shown

Provide a comprehensive analysis of all content types."""
        
        self.log_message(f"🔄 Analyzing mixed content {img_index} on page {page_num} with Hugging Face")
        
        analysis = self.analyze_with_detailed_model(img_pil, prompt)
        
        # OCR for comprehensive text extraction
        try:
            ocr_text = pytesseract.image_to_string(img_pil, lang='eng+spa')
            if ocr_text.strip():
                analysis += f"\n\n**Full Text Extraction:**\n{ocr_text.strip()}"
        except Exception as e:
            logger.warning(f"OCR failed for mixed content: {e}")
        
        return f"### Mixed Content Analysis - Image {img_index} (Page {page_num})\n\n**Content Type:** Mixed\n\n**AI Analysis:**\n{analysis}\n\n---\n\n"
    
    def process_image(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Process image with appropriate analysis method"""
        try:
            # First classify the content
            content_type = self.classify_image_content(img_pil)
            
            self.log_message(f"🔍 Image {img_index} (Page {page_num}) classified as: {content_type}")
            
            self.analyses_performed += 1
            
            # Route to appropriate analysis method
            if content_type == "TABLE":
                return self.analyze_table_content(img_pil, page_num, img_index)
            elif content_type == "FLOWCHART":
                return self.analyze_flowchart_content(img_pil, page_num, img_index)
            elif content_type == "FIGURE":
                return self.analyze_figure_content(img_pil, page_num, img_index)
            elif content_type == "TEXT":
                return self.analyze_text_content(img_pil, page_num, img_index)
            else:
                return self.analyze_mixed_content(img_pil, page_num, img_index)
                
        except Exception as e:
            logger.error(f"Error processing image {img_index}: {e}")
            return f"### Image {img_index} (Page {page_num})\n\n**Error:** Failed to analyze image - {e}\n\n---\n\n"
    
    def get_analysis_summary(self) -> Dict[str, int]:
        """Get summary of analysis performed"""
        return {
            'total_analyses': self.analyses_performed,
            'tables_found': self.tables_found,
            'flowcharts_found': self.flowcharts_found,
            'figures_found': self.figures_found,
            'text_content_found': self.text_content_found,
            'cost': 0.0  # FREE!
        }

class FreePDFAnalyzer:
    """Main PDF analyzer using free Hugging Face models"""
    
    def __init__(self, progress_callback=None):
        self.supported_formats = ['.pdf']
        self.output_format = '.md'
        self.used_filenames = set()
        self.progress_callback = progress_callback
        
        # Initialize vision analyzer
        self.vision_analyzer = HuggingFaceVisionAnalyzer(progress_callback)
        
        # Tracking
        self.files_processed = 0
        self.total_images_analyzed = 0
        self.extracted_links = []
        
    def log_message(self, message: str):
        """Send message to GUI if callback available"""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message)
    
    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text content with OCR fallback"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if not text.strip():
                    self.log_message(f"Using OCR for page {page_num + 1}")
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("ppm")
                    img = Image.open(io.BytesIO(img_data))
                    text = pytesseract.image_to_string(img, lang='eng+spa')
                
                page_marker = f"\n\n--- Page {page_num + 1} ---\n\n"
                full_text += page_marker + text
                
                # Extract links
                self.extract_links_from_text(text)
            
            doc.close()
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_links_from_text(self, text: str):
        """Extract URLs and links from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        doi_pattern = r'doi:?\s*10\.\d+/[^\s]+'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        urls = re.findall(url_pattern, text)
        dois = re.findall(doi_pattern, text, re.IGNORECASE)
        emails = re.findall(email_pattern, text)
        
        self.extracted_links.extend([f"URL: {url}" for url in urls])
        self.extracted_links.extend([f"DOI: {doi}" for doi in dois])
        self.extracted_links.extend([f"Email: {email}" for email in emails])
    
    def extract_and_analyze_images(self, pdf_path: str) -> List[str]:
        """Extract and analyze all images using free models"""
        image_analyses = []
        
        try:
            doc = fitz.open(pdf_path)
            total_images = 0
            
            self.log_message("🔍 Starting FREE image analysis...")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Handle CMYK conversion
                        if pix.n - pix.alpha >= 4:
                            self.log_message(f"Converting CMYK image to RGB on page {page_num + 1}")
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        if pix.n - pix.alpha <= 3:
                            img_data = pix.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                            
                            # Skip very small images
                            width, height = img_pil.size
                            if width < 100 or height < 100:
                                self.log_message(f"Skipping small image ({width}x{height})")
                                continue
                            
                            total_images += 1
                            self.total_images_analyzed += 1
                            
                            self.log_message(f"🆓 FREE Analysis: Processing image {total_images} (Cost: $0.00)")
                            
                            # Analyze with FREE models
                            analysis = self.vision_analyzer.process_image(
                                img_pil, page_num + 1, total_images
                            )
                            
                            image_analyses.append(analysis)
                            
                        pix = None
                        
                    except Exception as e:
                        logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue
            
            doc.close()
            
            # Print summary
            summary = self.vision_analyzer.get_analysis_summary()
            self.log_message("🎉 FREE IMAGE ANALYSIS COMPLETE!")
            self.log_message(f"📊 Total images analyzed: {total_images}")
            self.log_message(f"📋 Tables found: {summary['tables_found']}")
            self.log_message(f"🌳 Flowcharts found: {summary['flowcharts_found']}")
            self.log_message(f"🔬 Figures found: {summary['figures_found']}")
            self.log_message(f"📝 Text content found: {summary['text_content_found']}")
            self.log_message(f"💰 Total cost: $0.00 (FREE!)")
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
            
        return image_analyses
    
    def generate_filename(self, original_filename: str, output_dir: str) -> str:
        """Generate unique filename"""
        clean_original = re.sub(r'[^\w\s-]', '', Path(original_filename).stem)
        clean_original = re.sub(r'\s+', '_', clean_original)[:50]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{clean_original}_HF_{timestamp}"
        
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
        """Create comprehensive markdown output"""
        
        content_parts = []
        
        # Header
        content_parts.append(f"# Medical Guideline Analysis: {Path(original_filename).stem}")
        content_parts.append("")
        content_parts.append("*Analyzed with FREE Hugging Face Vision Models*")
        content_parts.append("")
        
        # Document info
        content_parts.append("## Document Information")
        content_parts.append("")
        content_parts.append(f"**Original Filename:** {original_filename}")
        content_parts.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content_parts.append(f"**Analysis Method:** Hugging Face Transformers (FREE)")
        content_parts.append(f"**Models Used:** BLIP-2, LLaVA (if available)")
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
            content_parts.append("## FREE AI Image Analysis Results")
            content_parts.append("")
            content_parts.append("*Each image analyzed using open-source Hugging Face vision models*")
            content_parts.append("")
            
            for analysis in image_analyses:
                content_parts.append(analysis)
        
        # Links section
        if self.extracted_links:
            content_parts.append("## Links and References")
            content_parts.append("")
            for link in set(self.extracted_links):  # Remove duplicates
                content_parts.append(f"- {link}")
            content_parts.append("")
            content_parts.append("---")
            content_parts.append("")
        
        # Processing summary
        summary = self.vision_analyzer.get_analysis_summary()
        content_parts.append("## FREE Processing Summary")
        content_parts.append("")
        content_parts.append(f"**Analysis Method:** Hugging Face Transformers (Open Source)")
        content_parts.append(f"**Total Cost:** $0.00 (Completely FREE!)")
        content_parts.append(f"**Images Analyzed:** {summary['total_analyses']}")
        content_parts.append(f"**Tables Found:** {summary['tables_found']}")
        content_parts.append(f"**Flowcharts Found:** {summary['flowcharts_found']}")
        content_parts.append(f"**Figures Found:** {summary['figures_found']}")
        content_parts.append(f"**Text Content Found:** {summary['text_content_found']}")
        content_parts.append(f"**Links Extracted:** {len(set(self.extracted_links))}")
        content_parts.append("")
        content_parts.append("**Models Information:**")
        content_parts.append("- BLIP-2 (Salesforce): Image understanding and classification")
        if self.vision_analyzer.llava_model is not None:
            content_parts.append("- LLaVA-1.5: Detailed image analysis")
        content_parts.append("- Tesseract OCR: Text extraction")
        content_parts.append("")
        content_parts.append("🎉 **This analysis was completely FREE using open-source models!**")
        content_parts.append("")
        content_parts.append("*No API costs, no usage limits, fully open-source medical image analysis*")
        
        return "\n".join(content_parts)
    
    def convert_pdf_free(self, pdf_path: str, output_dir: str) -> bool:
        """Convert PDF using FREE Hugging Face models"""
        try:
            self.log_message(f"🚀 Starting FREE analysis of {Path(pdf_path).name}")
            self.log_message("🆓 Starting FREE Hugging Face analysis - NO COSTS!")
            
            self.files_processed += 1
            self.extracted_links = []
            
            # Extract text content
            self.log_message("📝 Extracting text content...")
            text_content = self.extract_text_with_ocr(pdf_path)
            
            # Analyze all images with FREE models
            self.log_message("🔍 Analyzing images with FREE Hugging Face models...")
            image_analyses = self.extract_and_analyze_images(pdf_path)
            
            # Generate filename and save
            original_filename = Path(pdf_path).name
            new_filename = self.generate_filename(original_filename, output_dir)
            output_path = os.path.join(output_dir, new_filename + ".md")
            
            # Create markdown content
            self.log_message("📋 Creating markdown output...")
            markdown_content = self.create_markdown_output(
                text_content, image_analyses, original_filename
            )
            
            # Save file
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            self.log_message(f"✅ SUCCESS! File saved as: {new_filename}.md")
            self.log_message("💰 Total cost: $0.00 (FREE!)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error converting {pdf_path}: {e}")
            self.log_message(f"❌ Error: {e}")
            return False
    
    def convert_batch_free(self, input_dir: str, output_dir: str, progress_callback=None) -> Dict[str, int]:
        """Convert batch of PDFs with FREE analysis"""
        results = {'success': 0, 'failed': 0, 'total': 0}
        
        self.used_filenames = set()
        
        pdf_files = []
        for ext in self.supported_formats:
            pdf_files.extend(Path(input_dir).glob('*' + ext))
        
        results['total'] = len(pdf_files)
        
        if not pdf_files:
            self.log_message(f"No PDF files found in {input_dir}")
            return results
        
        self.log_message(f"🆓 FREE BATCH PROCESSING:")
        self.log_message(f"📄 Files to process: {results['total']}")
        self.log_message(f"💸 Total cost: $0.00 (COMPLETELY FREE!)")
        
        for i, pdf_path in enumerate(pdf_files):
            try:
                self.log_message(f"📄 [{i+1}/{results['total']}] Processing: {Path(pdf_path).name}")
                
                if progress_callback:
                    progress_callback(i + 1, results['total'], str(pdf_path))
                
                if self.convert_pdf_free(str(pdf_path), output_dir):
                    results['success'] += 1
                    self.log_message(f"   ✅ Success - Cost: $0.00")
                else:
                    results['failed'] += 1
                    self.log_message(f"   ❌ Failed")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                results['failed'] += 1
                self.log_message(f"   ❌ Error: {e}")
        
        self.log_message(f"🎉 FREE BATCH PROCESSING COMPLETE!")
        self.log_message(f"✅ Successful: {results['success']}/{results['total']}")
        self.log_message(f"💰 Total cost: $0.00 (FREE!)")
        
        return results

# GUI for FREE Hugging Face version
class FreeHuggingFaceGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("FREE Hugging Face PDF Analyzer - No Costs!")
        self.root.geometry("850x650")
        
        self.analyzer = None
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        title_label = ttk.Label(main_frame, text="FREE Hugging Face PDF Analyzer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        free_label = ttk.Label(main_frame, text="🆓 COMPLETELY FREE - No API costs ever!", 
                              font=('Arial', 12, 'bold'), foreground='green')
        free_label.grid(row=1, column=0, columnspan=3, pady=(0, 5))
        
        models_label = ttk.Label(main_frame, text="🤗 Uses open-source Hugging Face vision models", 
                                font=('Arial', 10), foreground='blue')
        models_label.grid(row=2, column=0, columnspan=3, pady=(0, 15))
        
        ttk.Label(main_frame, text="Input Directory (PDFs):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_dir, width=70).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_input_dir).grid(row=3, column=2, pady=5)
        
        ttk.Label(main_frame, text="Output Directory (Markdown):").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=70).grid(row=4, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_output_dir).grid(row=4, column=2, pady=5)
        
        # FREE models info section
        models_frame = ttk.LabelFrame(main_frame, text="🆓 FREE Open Source Models", padding="5")
        models_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=10)
        
        ttk.Label(models_frame, text="🆓 No API key needed - everything runs locally!", font=('Arial', 10, 'bold'), foreground='green').grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(models_frame, text="🤗 BLIP-2: Image understanding and classification", font=('Arial', 9)).grid(row=1, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(models_frame, text="🔬 LLaVA-1.5: Detailed medical image analysis", font=('Arial', 9)).grid(row=2, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(models_frame, text="📝 Tesseract OCR: Text extraction from images", font=('Arial', 9)).grid(row=3, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(models_frame, text="⚠️ First run downloads models (~2-4GB)", font=('Arial', 9), foreground='orange').grid(row=4, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(models_frame, text="⚡ GPU recommended but CPU works too", font=('Arial', 9)).grid(row=5, column=0, columnspan=2, sticky=tk.W)
        
        convert_button = ttk.Button(main_frame, text="Start FREE Analysis", command=self.start_conversion)
        convert_button.grid(row=6, column=0, columnspan=3, pady=20)
        
        self.progress_var = tk.StringVar()
        self.progress_var.set("Ready for FREE analysis - No costs!")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=7, column=0, columnspan=3, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=8, column=0, columnspan=3, sticky="ew", pady=5)
        
        log_frame = ttk.LabelFrame(main_frame, text="FREE Analysis Log", padding="5")
        log_frame.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=10)
        
        self.log_text = tk.Text(log_frame, height=18, width=100)
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(9, weight=1)
        models_frame.columnconfigure(1, weight=1)
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
        status_text = f"FREE Analysis {current}/{total}: {filename}"
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
        
        self.log_text.delete(1.0, tk.END)
        self.log_message("🆓 Starting FREE Hugging Face analysis...")
        self.log_message("💰 Cost: $0.00 - Completely FREE!")
        
        thread = threading.Thread(target=self.run_conversion)
        thread.daemon = True
        thread.start()
    
    def run_conversion(self):
        try:
            # Initialize analyzer with callback
            self.analyzer = FreePDFAnalyzer(progress_callback=self.log_message)
            
            results = self.analyzer.convert_batch_free(
                self.input_dir.get(),
                self.output_dir.get(),
                self.update_progress
            )
            
            self.progress_var.set("FREE analysis completed!")
            completion_message = f"FREE Analysis completed! Success: {results['success']}, Failed: {results['failed']}"
            self.log_message(completion_message)
            
            summary = self.analyzer.vision_analyzer.get_analysis_summary()
            
            messagebox.showinfo("FREE Analysis Complete!", 
                              f"🆓 FREE Analysis completed!\n\n"
                              f"Successful: {results['success']}\n"
                              f"Failed: {results['failed']}\n"
                              f"Total Cost: $0.00 (FREE!)\n\n"
                              f"📊 Images analyzed: {summary['total_analyses']}\n"
                              f"📋 Tables: {summary['tables_found']}\n"
                              f"🌳 Flowcharts: {summary['flowcharts_found']}\n"
                              f"🔬 Figures: {summary['figures_found']}\n"
                              f"📝 Text content: {summary['text_content_found']}\n\n"
                              f"🎉 Save thousands with FREE analysis!")
            
        except Exception as e:
            error_message = f"Error during FREE analysis: {e}"
            self.log_message(error_message)
            messagebox.showerror("Error", f"An error occurred: {e}")
    
    def run(self):
        self.root.mainloop()

def main():
    print("🆓 FREE PDF ANALYZER - HUGGING FACE GUI")
    print("=" * 50)
    print("🎉 Completely FREE - No API costs!")
    print("🤗 Open-source Hugging Face models")
    print("💾 Runs locally on your machine")
    print("📊 Same analysis quality as paid versions")
    print("=" * 50)
    
    app = FreeHuggingFaceGUI()
    app.run()

if __name__ == "__main__":
    main()