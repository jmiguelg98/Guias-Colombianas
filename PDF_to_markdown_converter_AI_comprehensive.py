#!/usr/bin/env python3
# PDF to Markdown Converter - AI COMPREHENSIVE VERSION
# Complete AI-powered analysis for clinical guidelines with specialized interpretation
# Tables, flowcharts, decision trees, metadata extraction, and full content analysis

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
        logging.FileHandler('pdf_converter_comprehensive.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensivePDFAnalyzer:
    def __init__(self):
        self.supported_formats = ['.pdf']
        self.output_format = '.md'
        self.used_filenames = set()
        self.openai_api_key = None
        self.use_ai_vision = False
        
        # Comprehensive tracking
        self.ai_calls_count = 0
        self.current_cost_estimate = 0.0
        self.tables_processed = 0
        self.flowcharts_processed = 0
        self.figures_processed = 0
        self.links_extracted = 0
        
        # Content organization
        self.document_metadata = {}
        self.extracted_links = []
        self.structured_content = {
            'title': '',
            'metadata': {},
            'main_text': '',
            'tables': [],
            'flowcharts': [],
            'figures': [],
            'references': [],
            'links': []
        }
        
    def set_openai_key(self, api_key: str):
        """Set OpenAI API key for AI vision processing"""
        self.openai_api_key = api_key
        self.use_ai_vision = True
        openai.api_key = api_key
    
    def make_ai_vision_call(self, img_pil: Image.Image, prompt: str, max_tokens: int = 1000) -> str:
        """Make AI vision API call with custom prompt"""
        try:
            if not self.use_ai_vision or not self.openai_api_key:
                return ""
                
            # Convert PIL image to base64
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                   headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                description = result['choices'][0]['message']['content']
                
                self.ai_calls_count += 1
                self.current_cost_estimate = self.ai_calls_count * 0.01275
                
                return description
            else:
                logger.warning(f"AI vision API error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Error in AI vision processing: {e}")
            return ""
    
    def classify_image_content(self, img_pil: Image.Image) -> str:
        """Use AI to classify what type of content the image contains"""
        prompt = """Analyze this image from a clinical guideline and classify its primary content type. 
        Respond with ONLY ONE of these categories:
        - TABLE: If it contains tabular data, charts, or structured data
        - FLOWCHART: If it contains decision trees, algorithms, or process flows
        - FIGURE: If it contains diagrams, anatomical images, or illustrations  
        - TEXT: If it primarily contains text content
        - MIXED: If it contains multiple types of content
        
        Just respond with the single word category."""
        
        classification = self.make_ai_vision_call(img_pil, prompt, max_tokens=50)
        return classification.strip().upper()
    
    def extract_table_data(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Specialized AI analysis for tables"""
        prompt = """This image contains a table from a clinical guideline. Please:

        1. Extract ALL data from the table including headers, rows, and any footnotes
        2. Present the data in a clear, structured format
        3. Explain the clinical significance of the data
        4. Include any dosages, ranges, criteria, or clinical values
        5. Note any relationships between columns/rows
        6. If there are units of measurement, include them
        7. Convert abbreviations to full terms when possible

        Format your response as:
        **Table Content:**
        [Detailed extraction of all table data]

        **Clinical Interpretation:**
        [Explanation of what this table means clinically]

        **Key Data Points:**
        [List of important values, ranges, criteria from the table]"""
        
        self.tables_processed += 1
        logger.info(f"🔍 AI analyzing table {img_index} on page {page_num}")
        
        analysis = self.make_ai_vision_call(img_pil, prompt, max_tokens=1200)
        return f"### Table {self.tables_processed} (Page {page_num})\n\n{analysis}\n\n"
    
    def extract_flowchart_logic(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Specialized AI analysis for flowcharts and decision trees"""
        prompt = """This image contains a flowchart or clinical decision tree from a medical guideline. Please:

        1. Map out the complete decision pathway step by step
        2. Identify all decision points and their criteria  
        3. List all possible outcomes and endpoints
        4. Explain the clinical logic behind each decision
        5. Note any specific values, thresholds, or criteria mentioned
        6. Describe patient pathways through the algorithm
        7. Include any contraindications or warnings shown

        Format your response as:
        **Decision Tree Overview:**
        [Summary of the algorithm's purpose]

        **Step-by-Step Pathway:**
        [Detailed walkthrough of each decision point]

        **Clinical Decision Points:**
        [List of specific criteria, values, thresholds]

        **Patient Outcomes:**
        [Description of possible endpoints and treatments]

        **Implementation Notes:**
        [Any special considerations or warnings]"""
        
        self.flowcharts_processed += 1
        logger.info(f"🌳 AI analyzing flowchart/decision tree {img_index} on page {page_num}")
        
        analysis = self.make_ai_vision_call(img_pil, prompt, max_tokens=1500)
        return f"### Clinical Decision Algorithm {self.flowcharts_processed} (Page {page_num})\n\n{analysis}\n\n"
    
    def extract_figure_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Specialized AI analysis for figures and diagrams"""
        prompt = """This image contains a figure, diagram, or illustration from a clinical guideline. Please:

        1. Describe all visual elements in detail
        2. Explain the clinical or educational purpose
        3. Extract any text, labels, or annotations
        4. Describe relationships between elements
        5. Note any anatomical structures, pathways, or processes shown
        6. Include any measurements, scales, or reference values
        7. Explain the clinical relevance

        Format your response as:
        **Figure Description:**
        [Detailed description of visual elements]

        **Clinical Context:**
        [Explanation of clinical relevance and purpose]

        **Key Information:**
        [Important text, values, or annotations from the figure]"""
        
        self.figures_processed += 1
        logger.info(f"📊 AI analyzing figure {img_index} on page {page_num}")
        
        analysis = self.make_ai_vision_call(img_pil, prompt, max_tokens=1000)
        return f"### Figure {self.figures_processed} (Page {page_num})\n\n{analysis}\n\n"
    
    def extract_text_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Specialized AI analysis for text content"""
        prompt = """This image contains text content from a clinical guideline. Please:

        1. Extract ALL visible text accurately
        2. Maintain the structure and formatting
        3. Identify any headings, bullet points, or sections
        4. Note any emphasized text (bold, italics, etc.)
        5. Include any citations or references mentioned
        6. Preserve any lists, numbered items, or hierarchical structure
        7. Extract any URLs, links, or contact information

        Format your response as:
        **Extracted Text:**
        [Complete text extraction maintaining structure]

        **Key Points:**
        [Summary of main clinical recommendations or information]

        **Links/References:**
        [Any URLs, citations, or references found]"""
        
        logger.info(f"📝 AI extracting text content from image {img_index} on page {page_num}")
        
        analysis = self.make_ai_vision_call(img_pil, prompt, max_tokens=1200)
        return analysis
    
    def analyze_mixed_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """AI analysis for images with mixed content types"""
        prompt = """This image contains mixed content (tables, text, figures, etc.) from a clinical guideline. Please:

        1. Identify and separate each type of content
        2. Extract tables with all data and structure
        3. Extract all text content maintaining formatting
        4. Describe any figures, diagrams, or visual elements
        5. Note the relationships between different content elements
        6. Extract any flowcharts or decision processes
        7. Include all clinical data, values, and recommendations

        Format your response by content type:
        **Tables:**
        [Extract any tabular data]

        **Text Content:**
        [Extract all text maintaining structure]

        **Figures/Diagrams:**
        [Describe visual elements]

        **Clinical Recommendations:**
        [Key clinical information and guidance]"""
        
        logger.info(f"🔄 AI analyzing mixed content {img_index} on page {page_num}")
        
        analysis = self.make_ai_vision_call(img_pil, prompt, max_tokens=1500)
        return f"### Mixed Content (Page {page_num})\n\n{analysis}\n\n"
    
    def extract_comprehensive_metadata(self, first_pages_text: str, all_images_analysis: str) -> Dict[str, str]:
        """Use AI to extract comprehensive metadata from document content"""
        prompt = f"""Analyze this clinical guideline content and extract comprehensive metadata. 

        TEXT CONTENT:
        {first_pages_text[:3000]}

        IMAGE ANALYSIS:
        {all_images_analysis[:2000]}

        Extract and return ONLY a JSON object with these fields:
        {{
            "title": "Full document title",
            "organization": "Publishing organization/society",
            "year": "Publication year", 
            "authors": "Primary authors or contributors",
            "guideline_type": "Type of guideline (consensus, recommendation, protocol, etc.)",
            "medical_specialty": "Primary medical specialty addressed",
            "target_population": "Target patient population",
            "evidence_level": "Level of evidence mentioned",
            "keywords": "Key medical terms and topics",
            "doi": "DOI if mentioned",
            "issn": "ISSN if mentioned",
            "version": "Version or edition if mentioned",
            "language": "Primary language",
            "geographic_scope": "Geographic applicability",
            "clinical_areas": "Specific clinical areas covered"
        }}

        Return ONLY the JSON object, no other text."""
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 800
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                   headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                metadata_text = result['choices'][0]['message']['content']
                
                # Parse JSON response
                try:
                    metadata = json.loads(metadata_text)
                    logger.info("🎯 AI extracted comprehensive metadata")
                    return metadata
                except json.JSONDecodeError:
                    logger.warning("AI metadata response was not valid JSON")
                    return {}
            else:
                logger.warning(f"AI metadata extraction failed: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error in AI metadata extraction: {e}")
            return {}
    
    def extract_links_from_text(self, text: str) -> List[str]:
        """Extract URLs and links from text content"""
        # Pattern for URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
        # Pattern for DOIs
        doi_pattern = r'doi:?\s*10\.\d+/[^\s]+'
        
        # Pattern for email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        links = []
        
        # Find URLs
        urls = re.findall(url_pattern, text)
        links.extend([f"URL: {url}" for url in urls])
        
        # Find DOIs
        dois = re.findall(doi_pattern, text, re.IGNORECASE)
        links.extend([f"DOI: {doi}" for doi in dois])
        
        # Find emails
        emails = re.findall(email_pattern, text)
        links.extend([f"Email: {email}" for email in emails])
        
        self.links_extracted += len(links)
        return links
    
    def process_comprehensive_image(self, img_pil: Image.Image, page_num: int, img_index: int) -> Tuple[str, str]:
        """Comprehensive AI processing of each image"""
        try:
            # First, classify the image content
            content_type = self.classify_image_content(img_pil)
            logger.info(f"📋 Image {img_index} on page {page_num} classified as: {content_type}")
            
            # Process based on content type
            if content_type == "TABLE":
                analysis = self.extract_table_data(img_pil, page_num, img_index)
                return analysis, "table"
                
            elif content_type == "FLOWCHART":
                analysis = self.extract_flowchart_logic(img_pil, page_num, img_index)
                return analysis, "flowchart"
                
            elif content_type == "FIGURE":
                analysis = self.extract_figure_content(img_pil, page_num, img_index)
                return analysis, "figure"
                
            elif content_type == "TEXT":
                analysis = self.extract_text_content(img_pil, page_num, img_index)
                return f"### Text Content (Page {page_num})\n\n{analysis}\n\n", "text"
                
            elif content_type == "MIXED":
                analysis = self.analyze_mixed_content(img_pil, page_num, img_index)
                return analysis, "mixed"
                
            else:
                # Fallback to general analysis
                analysis = self.extract_figure_content(img_pil, page_num, img_index)
                return analysis, "general"
                
        except Exception as e:
            logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
            return f"### Image {img_index} (Page {page_num})\n\n**Error processing image**\n\n", "error"
    
    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text content with OCR fallback"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if not text.strip():
                    logger.info(f"No text found on page {page_num + 1}, using OCR")
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("ppm")
                    img = Image.open(io.BytesIO(img_data))
                    text = pytesseract.image_to_string(img, lang='eng')
                
                page_marker = f"\n\n--- Page {page_num + 1} ---\n\n"
                full_text += page_marker + text
                
                # Extract links from each page
                page_links = self.extract_links_from_text(text)
                self.extracted_links.extend(page_links)
            
            doc.close()
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_images_and_analyze(self, pdf_path: str) -> Dict[str, List[str]]:
        """Extract and comprehensively analyze all images"""
        content_sections = {
            'tables': [],
            'flowcharts': [],
            'figures': [],
            'text_content': [],
            'mixed_content': []
        }
        
        try:
            doc = fitz.open(pdf_path)
            
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
                            
                            # Comprehensive AI analysis
                            analysis, content_type = self.process_comprehensive_image(
                                img_pil, page_num + 1, img_index + 1
                            )
                            
                            # Categorize content
                            if content_type == "table":
                                content_sections['tables'].append(analysis)
                            elif content_type == "flowchart":
                                content_sections['flowcharts'].append(analysis)
                            elif content_type == "figure":
                                content_sections['figures'].append(analysis)
                            elif content_type == "text":
                                content_sections['text_content'].append(analysis)
                            elif content_type == "mixed":
                                content_sections['mixed_content'].append(analysis)
                            else:
                                content_sections['figures'].append(analysis)
                        
                        pix = None
                        
                    except Exception as e:
                        logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
            
        return content_sections
    
    def generate_filename(self, metadata: Dict[str, str], original_filename: str, output_dir: str) -> str:
        """Generate unique filename with conflict resolution"""
        clean_original = re.sub(r'[^\w\s-]', '', Path(original_filename).stem)
        clean_original = re.sub(r'\s+', '_', clean_original)[:25]
        
        filename_parts = [clean_original]
        
        # Use AI-extracted metadata for better filenames
        if 'title' in metadata and metadata['title']:
            clean_title = re.sub(r'[^\w\s-]', '', metadata['title'])
            clean_title = re.sub(r'\s+', '_', clean_title)[:30]
            filename_parts.append(clean_title)
        
        if 'organization' in metadata and metadata['organization']:
            clean_org = re.sub(r'[^\w\s-]', '', metadata['organization'])
            clean_org = re.sub(r'\s+', '_', clean_org)[:20]
            filename_parts.append(clean_org)
        
        if 'year' in metadata and metadata['year']:
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
    
    def create_comprehensive_markdown(self, text_content: str, image_content: Dict[str, List[str]], 
                                    metadata: Dict[str, str], original_filename: str) -> str:
        """Create comprehensive, organized markdown content"""
        
        content_parts = []
        
        # Document header with AI-extracted metadata
        title = metadata.get('title', 'Clinical Guideline')
        content_parts.append(f"# {title}")
        content_parts.append("")
        
        # Comprehensive document information
        content_parts.append("## Document Information")
        content_parts.append("")
        content_parts.append(f"**Original Filename:** {original_filename}")
        
        for key, value in metadata.items():
            if value and value.strip():
                formatted_key = key.replace('_', ' ').title()
                content_parts.append(f"**{formatted_key}:** {value}")
        
        content_parts.append("")
        content_parts.append("---")
        content_parts.append("")
        
        # Links and references section
        if self.extracted_links:
            content_parts.append("## Links and References")
            content_parts.append("")
            for link in self.extracted_links:
                content_parts.append(f"- {link}")
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
        
        # Tables section
        if image_content['tables']:
            content_parts.append("## Clinical Tables and Data")
            content_parts.append("")
            content_parts.append("*AI-analyzed tables with clinical interpretation*")
            content_parts.append("")
            for table in image_content['tables']:
                content_parts.append(table)
            content_parts.append("---")
            content_parts.append("")
        
        # Flowcharts and decision trees
        if image_content['flowcharts']:
            content_parts.append("## Clinical Decision Trees and Algorithms")
            content_parts.append("")
            content_parts.append("*AI-analyzed decision pathways and clinical algorithms*")
            content_parts.append("")
            for flowchart in image_content['flowcharts']:
                content_parts.append(flowchart)
            content_parts.append("---")
            content_parts.append("")
        
        # Figures and diagrams
        if image_content['figures']:
            content_parts.append("## Figures and Clinical Diagrams")
            content_parts.append("")
            content_parts.append("*AI-analyzed figures with clinical context*")
            content_parts.append("")
            for figure in image_content['figures']:
                content_parts.append(figure)
            content_parts.append("---")
            content_parts.append("")
        
        # Additional text content from images
        if image_content['text_content']:
            content_parts.append("## Additional Text Content")
            content_parts.append("")
            content_parts.append("*Text content extracted from images*")
            content_parts.append("")
            for text in image_content['text_content']:
                content_parts.append(text)
            content_parts.append("---")
            content_parts.append("")
        
        # Mixed content
        if image_content['mixed_content']:
            content_parts.append("## Mixed Content Analysis")
            content_parts.append("")
            content_parts.append("*Complex content with multiple elements*")
            content_parts.append("")
            for mixed in image_content['mixed_content']:
                content_parts.append(mixed)
            content_parts.append("---")
            content_parts.append("")
        
        # Processing summary
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        content_parts.append("## Processing Summary")
        content_parts.append("")
        content_parts.append(f"**Processed on:** {timestamp}")
        content_parts.append(f"**AI Vision calls:** {self.ai_calls_count}")
        content_parts.append(f"**Estimated cost:** ${self.current_cost_estimate:.2f}")
        content_parts.append(f"**Tables analyzed:** {self.tables_processed}")
        content_parts.append(f"**Flowcharts analyzed:** {self.flowcharts_processed}")
        content_parts.append(f"**Figures analyzed:** {self.figures_processed}")
        content_parts.append(f"**Links extracted:** {self.links_extracted}")
        content_parts.append("")
        content_parts.append("*Converted from PDF to Markdown using comprehensive AI analysis*")
        
        return "\n".join(content_parts)
    
    def convert_pdf_comprehensive(self, pdf_path: str, output_dir: str) -> bool:
        """Comprehensive PDF to Markdown conversion with full AI analysis"""
        try:
            logger.info(f"🚀 Starting comprehensive analysis of {pdf_path}")
            
            # Reset counters for this document
            self.ai_calls_count = 0
            self.current_cost_estimate = 0.0
            self.tables_processed = 0
            self.flowcharts_processed = 0
            self.figures_processed = 0
            self.links_extracted = 0
            self.extracted_links = []
            
            # Extract text content and links
            logger.info("📝 Extracting text content...")
            text_content = self.extract_text_with_ocr(pdf_path)
            
            # Comprehensive image analysis
            logger.info("🔍 Performing comprehensive AI image analysis...")
            image_content = self.extract_images_and_analyze(pdf_path)
            
            # AI-powered metadata extraction
            logger.info("🎯 Extracting metadata with AI...")
            all_image_analysis = ""
            for section in image_content.values():
                all_image_analysis += " ".join(section)
            
            ai_metadata = self.extract_comprehensive_metadata(text_content[:5000], all_image_analysis[:3000])
            
            # Generate filename and create markdown
            original_filename = Path(pdf_path).name
            new_filename = self.generate_filename(ai_metadata, original_filename, output_dir)
            output_path = os.path.join(output_dir, new_filename + ".md")
            
            # Create comprehensive markdown
            logger.info("📋 Creating comprehensive markdown...")
            markdown_content = self.create_comprehensive_markdown(
                text_content, image_content, ai_metadata, original_filename
            )
            
            # Save the file
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"✅ Successfully converted {pdf_path} to {output_path}")
            logger.info(f"💰 Cost for this document: ${self.current_cost_estimate:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error converting {pdf_path}: {e}")
            return False
    
    def convert_batch(self, input_dir: str, output_dir: str, progress_callback=None) -> Dict[str, int]:
        """Convert batch of PDFs with comprehensive analysis"""
        results = {'success': 0, 'failed': 0, 'total': 0}
        total_cost = 0.0
        
        self.used_filenames = set()
        
        pdf_files = []
        for ext in self.supported_formats:
            pdf_files.extend(Path(input_dir).glob('*' + ext))
        
        results['total'] = len(pdf_files)
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return results
        
        print(f"\n🚀 Starting comprehensive AI analysis of {results['total']} PDF files...")
        print("=" * 70)
        
        for i, pdf_path in enumerate(pdf_files):
            try:
                print(f"\n📄 [{i+1}/{results['total']}] Processing: {Path(pdf_path).name}")
                
                if progress_callback:
                    progress_callback(i + 1, results['total'], str(pdf_path))
                
                if self.convert_pdf_comprehensive(str(pdf_path), output_dir):
                    results['success'] += 1
                    total_cost += self.current_cost_estimate
                    print(f"   ✅ Success (Document cost: ${self.current_cost_estimate:.2f}, Total: ${total_cost:.2f})")
                else:
                    results['failed'] += 1
                    print(f"   ❌ Failed")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                results['failed'] += 1
                print(f"   ❌ Error: {e}")
        
        print(f"\n💰 Total estimated cost: ${total_cost:.2f}")
        return results

# GUI Class for the comprehensive version
class ComprehensivePDFGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Comprehensive AI PDF Analyzer - Clinical Guidelines")
        self.root.geometry("800x600")
        
        self.analyzer = ComprehensivePDFAnalyzer()
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.openai_key = tk.StringVar()
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        title_label = ttk.Label(main_frame, text="Comprehensive AI Clinical Guideline Analyzer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        warning_label = ttk.Label(main_frame, text="🧠 Complete AI analysis: Tables, Flowcharts, Metadata, Links", 
                                 font=('Arial', 11), foreground='blue')
        warning_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        ttk.Label(main_frame, text="Input Directory (PDFs):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_dir, width=60).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_input_dir).grid(row=2, column=2, pady=5)
        
        ttk.Label(main_frame, text="Output Directory (Markdown):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=60).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_output_dir).grid(row=3, column=2, pady=5)
        
        # OpenAI API Key section
        ai_frame = ttk.LabelFrame(main_frame, text="🧠 Comprehensive AI Analysis (Required)", padding="5")
        ai_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)
        
        ttk.Label(ai_frame, text="OpenAI API Key:").grid(row=0, column=0, sticky=tk.W, pady=5)
        key_entry = ttk.Entry(ai_frame, textvariable=self.openai_key, width=60, show="*")
        key_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(ai_frame, text="🔍 Specialized analysis for tables, flowcharts, and figures", font=('Arial', 9)).grid(row=1, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(ai_frame, text="🎯 AI-powered metadata extraction and link detection", font=('Arial', 9)).grid(row=2, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(ai_frame, text="📋 Comprehensive clinical content interpretation", font=('Arial', 9)).grid(row=3, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(ai_frame, text="💰 Cost varies by content complexity (~$0.50-5.00 per document)", font=('Arial', 9), foreground='red').grid(row=4, column=0, columnspan=2, sticky=tk.W)
        
        convert_button = ttk.Button(main_frame, text="Start Comprehensive Analysis", command=self.start_conversion)
        convert_button.grid(row=5, column=0, columnspan=3, pady=20)
        
        self.progress_var = tk.StringVar()
        self.progress_var.set("Ready for comprehensive clinical guideline analysis")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=6, column=0, columnspan=3, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=7, column=0, columnspan=3, sticky="ew", pady=5)
        
        log_frame = ttk.LabelFrame(main_frame, text="Analysis Log", padding="5")
        log_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=10)
        
        self.log_text = tk.Text(log_frame, height=18, width=90)
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(8, weight=1)
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
        status_text = f"Analyzing {current}/{total}: {filename}"
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
            messagebox.showerror("Error", "OpenAI API key is required for comprehensive analysis")
            return
        
        self.analyzer.set_openai_key(api_key)
        self.log_message("🧠 Comprehensive AI analysis enabled!")
        
        self.log_text.delete(1.0, tk.END)
        self.log_message("🚀 Starting comprehensive clinical guideline analysis...")
        
        thread = threading.Thread(target=self.run_conversion)
        thread.daemon = True
        thread.start()
    
    def run_conversion(self):
        try:
            results = self.analyzer.convert_batch(
                self.input_dir.get(),
                self.output_dir.get(),
                self.update_progress
            )
            
            self.progress_var.set("Comprehensive analysis completed!")
            completion_message = f"Analysis completed! Success: {results['success']}, Failed: {results['failed']}"
            self.log_message(completion_message)
            
            dialog_message = f"""🧠 Comprehensive AI Analysis Complete!

📊 RESULTS:
Total files: {results['total']}
Successful: {results['success']}
Failed: {results['failed']}

🎯 All clinical content analyzed:
• Tables interpreted with clinical context
• Flowcharts mapped to decision pathways  
• Figures analyzed for clinical relevance
• Metadata extracted with AI
• Links and references identified
• Content organized for LLM processing

All files saved with comprehensive analysis!"""
            
            messagebox.showinfo("Comprehensive Analysis Complete!", dialog_message)
            
        except Exception as e:
            error_message = f"Error during analysis: {e}"
            self.log_message(error_message)
            messagebox.showerror("Error", f"An error occurred: {e}")
    
    def run(self):
        self.root.mainloop()

def main():
    print("Comprehensive AI Clinical Guideline Analyzer")
    print("=" * 50)
    print("🧠 Complete AI-powered analysis for clinical guidelines")
    print("🔍 Tables, flowcharts, figures, metadata, and links")
    print("📋 Specialized interpretation for medical content")
    print("💰 Comprehensive analysis with real-time cost tracking")
    print("=" * 50)
    
    app = ComprehensivePDFGUI()
    app.run()

if __name__ == "__main__":
    main()