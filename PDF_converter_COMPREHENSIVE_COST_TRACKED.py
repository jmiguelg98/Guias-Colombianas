#!/usr/bin/env python3
# PDF to Markdown Converter - COMPREHENSIVE VERSION WITH COST TRACKING
# Complete AI-powered analysis for clinical guidelines with detailed cost monitoring
# This is the $2800 version that actually works - with cost tracking added

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
        logging.FileHandler('pdf_converter_comprehensive_cost.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CostTracker:
    """Detailed cost tracking for AI calls"""
    def __init__(self):
        self.calls_by_type = {
            'classification': 0,
            'table_analysis': 0,
            'flowchart_analysis': 0,
            'figure_analysis': 0,
            'text_analysis': 0,
            'mixed_analysis': 0,
            'metadata_extraction': 0
        }
        
        self.costs_by_type = {
            'classification': 0.0,
            'table_analysis': 0.0,
            'flowchart_analysis': 0.0,
            'figure_analysis': 0.0,
            'text_analysis': 0.0,
            'mixed_analysis': 0.0,
            'metadata_extraction': 0.0
        }
        
        self.total_calls = 0
        self.total_cost = 0.0
        self.cost_per_call = 0.01275  # Updated GPT-4V cost
        self.start_time = None
        self.call_log = []
        
    def start_tracking(self):
        """Start cost tracking session"""
        self.start_time = datetime.now()
        logger.info("💰 Cost tracking started")
        
    def log_ai_call(self, call_type: str, prompt_length: int, response_length: int, cost: float):
        """Log individual AI call with details"""
        self.calls_by_type[call_type] += 1
        self.costs_by_type[call_type] += cost
        self.total_calls += 1
        self.total_cost += cost
        
        call_record = {
            'timestamp': datetime.now().isoformat(),
            'type': call_type,
            'prompt_length': prompt_length,
            'response_length': response_length,
            'cost': cost,
            'cumulative_cost': self.total_cost
        }
        
        self.call_log.append(call_record)
        
        logger.info(f"💰 AI Call: {call_type} - Cost: ${cost:.4f} - Total: ${self.total_cost:.2f}")
        
    def get_cost_summary(self) -> Dict:
        """Get detailed cost summary"""
        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'total_calls': self.total_calls,
            'total_cost': self.total_cost,
            'cost_per_call_avg': self.total_cost / max(self.total_calls, 1),
            'duration_seconds': duration,
            'calls_by_type': self.calls_by_type.copy(),
            'costs_by_type': self.costs_by_type.copy(),
            'call_log': self.call_log.copy()
        }
    
    def print_cost_summary(self):
        """Print detailed cost summary"""
        print(f"\n💰 DETAILED COST SUMMARY:")
        print(f"{'='*50}")
        print(f"Total AI Calls: {self.total_calls}")
        print(f"Total Cost: ${self.total_cost:.2f}")
        print(f"Average Cost per Call: ${self.total_cost / max(self.total_calls, 1):.4f}")
        
        print(f"\n📊 CALLS BY TYPE:")
        for call_type, count in self.calls_by_type.items():
            cost = self.costs_by_type[call_type]
            if count > 0:
                print(f"  {call_type.replace('_', ' ').title()}: {count} calls, ${cost:.2f}")
        
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            print(f"\n⏱️  Processing Time: {duration:.1f} seconds")
            print(f"💸 Cost per Minute: ${(self.total_cost / max(duration/60, 1)):.2f}")

class ComprehensivePDFAnalyzer:
    def __init__(self):
        self.supported_formats = ['.pdf']
        self.output_format = '.md'
        self.used_filenames = set()
        self.openai_api_key = None
        self.use_ai_vision = False
        
        # Cost tracking
        self.cost_tracker = CostTracker()
        
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
        self.cost_tracker.start_tracking()
    
    def make_ai_vision_call(self, img_pil: Image.Image, prompt: str, call_type: str, max_tokens: int = 1000) -> str:
        """Make AI vision API call with detailed cost tracking"""
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
                
                # Calculate actual cost based on tokens (approximation)
                prompt_length = len(prompt)
                response_length = len(description)
                estimated_cost = self.cost_tracker.cost_per_call  # Base cost per call
                
                # Log the AI call with cost tracking
                self.cost_tracker.log_ai_call(call_type, prompt_length, response_length, estimated_cost)
                
                self.ai_calls_count += 1
                self.current_cost_estimate = self.cost_tracker.total_cost
                
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
        
        classification = self.make_ai_vision_call(img_pil, prompt, 'classification', max_tokens=50)
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
        logger.info(f"📊 AI analyzing table {img_index} on page {page_num}")
        
        analysis = self.make_ai_vision_call(img_pil, prompt, 'table_analysis', max_tokens=1200)
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
        
        analysis = self.make_ai_vision_call(img_pil, prompt, 'flowchart_analysis', max_tokens=1500)
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
        logger.info(f"🔬 AI analyzing figure {img_index} on page {page_num}")
        
        analysis = self.make_ai_vision_call(img_pil, prompt, 'figure_analysis', max_tokens=1000)
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
        
        analysis = self.make_ai_vision_call(img_pil, prompt, 'text_analysis', max_tokens=1200)
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
        
        analysis = self.make_ai_vision_call(img_pil, prompt, 'mixed_analysis', max_tokens=1500)
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
                
                # Log this AI call
                self.cost_tracker.log_ai_call('metadata_extraction', len(prompt), len(metadata_text), self.cost_tracker.cost_per_call)
                self.ai_calls_count += 1
                self.current_cost_estimate = self.cost_tracker.total_cost
                
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
        """Comprehensive AI processing of each image with cost tracking"""
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
                    text = pytesseract.image_to_string(img, lang='eng+spa')
                
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
        """Extract and comprehensively analyze all images with cost tracking"""
        content_sections = {
            'tables': [],
            'flowcharts': [],
            'figures': [],
            'text_content': [],
            'mixed_content': []
        }
        
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
                                continue
                            
                            total_images += 1
                            
                            # Print cost update for each image
                            print(f"💰 Processing image {total_images}: Current cost ${self.cost_tracker.total_cost:.2f}")
                            
                            # Comprehensive AI analysis with cost tracking
                            analysis, content_type = self.process_comprehensive_image(
                                img_pil, page_num + 1, total_images
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
                            
                            logger.info(f"✅ Processed image {total_images} ({content_type}) - Cost: ${self.cost_tracker.total_cost:.2f}")
                        
                        pix = None
                        
                    except Exception as e:
                        logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue
            
            doc.close()
            
            # Print final image processing summary
            print(f"\n📊 IMAGE PROCESSING COMPLETE:")
            print(f"  Total images processed: {total_images}")
            print(f"  Tables: {len(content_sections['tables'])}")
            print(f"  Flowcharts: {len(content_sections['flowcharts'])}")
            print(f"  Figures: {len(content_sections['figures'])}")
            print(f"  Text content: {len(content_sections['text_content'])}")
            print(f"  Mixed content: {len(content_sections['mixed_content'])}")
            print(f"  💰 Total cost so far: ${self.cost_tracker.total_cost:.2f}")
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
            
        return content_sections
    
    def generate_filename(self, metadata: Dict[str, str], original_filename: str, output_dir: str) -> str:
        """Generate unique filename preserving original name"""
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
    
    def create_comprehensive_markdown(self, text_content: str, image_content: Dict[str, List[str]], 
                                    metadata: Dict[str, str], original_filename: str) -> str:
        """Create comprehensive markdown with cost tracking summary"""
        
        content_parts = []
        
        # Header
        title = metadata.get('title', 'Clinical Guideline Analysis')
        content_parts.append(f"# {title}")
        content_parts.append("")
        
        # Document information
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
        
        # Main text content
        if text_content:
            content_parts.append("## Document Text Content")
            content_parts.append("")
            content_parts.append(text_content)
            content_parts.append("")
            content_parts.append("---")
            content_parts.append("")
        
        # Tables section
        if image_content.get('tables'):
            content_parts.append("## Tables and Structured Data")
            content_parts.append("")
            content_parts.append("*AI-analyzed tables with clinical interpretation*")
            content_parts.append("")
            
            for table in image_content['tables']:
                content_parts.append(table)
        
        # Flowcharts section
        if image_content.get('flowcharts'):
            content_parts.append("## Clinical Decision Trees and Flowcharts")
            content_parts.append("")
            content_parts.append("*AI-analyzed decision pathways and clinical algorithms*")
            content_parts.append("")
            
            for flowchart in image_content['flowcharts']:
                content_parts.append(flowchart)
        
        # Figures section
        if image_content.get('figures'):
            content_parts.append("## Figures and Diagrams")
            content_parts.append("")
            content_parts.append("*AI-analyzed figures, illustrations, and visual content*")
            content_parts.append("")
            
            for figure in image_content['figures']:
                content_parts.append(figure)
        
        # Text content section
        if image_content.get('text_content'):
            content_parts.append("## Extracted Text Content")
            content_parts.append("")
            content_parts.append("*AI-extracted text from images*")
            content_parts.append("")
            
            for text in image_content['text_content']:
                content_parts.append(text)
        
        # Mixed content section
        if image_content.get('mixed_content'):
            content_parts.append("## Mixed Content Analysis")
            content_parts.append("")
            content_parts.append("*AI-analyzed complex content with multiple elements*")
            content_parts.append("")
            
            for mixed in image_content['mixed_content']:
                content_parts.append(mixed)
        
        # Links section
        if self.extracted_links:
            content_parts.append("## Links and References")
            content_parts.append("")
            for link in self.extracted_links:
                content_parts.append(f"- {link}")
            content_parts.append("")
            content_parts.append("---")
            content_parts.append("")
        
        # Cost tracking summary
        cost_summary = self.cost_tracker.get_cost_summary()
        content_parts.append("## Processing Summary with Cost Tracking")
        content_parts.append("")
        content_parts.append(f"**Processing Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content_parts.append(f"**Total AI Calls:** {cost_summary['total_calls']}")
        content_parts.append(f"**Total Cost:** ${cost_summary['total_cost']:.2f}")
        content_parts.append(f"**Average Cost per Call:** ${cost_summary['cost_per_call_avg']:.4f}")
        content_parts.append("")
        
        # Detailed cost breakdown
        content_parts.append("**Cost Breakdown by AI Function:**")
        for call_type, cost in cost_summary['costs_by_type'].items():
            if cost > 0:
                calls = cost_summary['calls_by_type'][call_type]
                content_parts.append(f"- {call_type.replace('_', ' ').title()}: {calls} calls, ${cost:.2f}")
        
        content_parts.append("")
        content_parts.append(f"**Content Analysis Results:**")
        content_parts.append(f"- Tables processed: {self.tables_processed}")
        content_parts.append(f"- Flowcharts processed: {self.flowcharts_processed}")
        content_parts.append(f"- Figures processed: {self.figures_processed}")
        content_parts.append(f"- Links extracted: {self.links_extracted}")
        content_parts.append("")
        
        if cost_summary['duration_seconds'] > 0:
            content_parts.append(f"**Processing Time:** {cost_summary['duration_seconds']:.1f} seconds")
            content_parts.append(f"**Cost per Minute:** ${(cost_summary['total_cost'] / max(cost_summary['duration_seconds']/60, 1)):.2f}")
        
        content_parts.append("")
        content_parts.append("*This is the comprehensive AI analysis version - every image analyzed with specialized AI*")
        
        return "\n".join(content_parts)
    
    def convert_pdf_comprehensive(self, pdf_path: str, output_dir: str) -> bool:
        """Convert PDF with comprehensive AI analysis and cost tracking"""
        try:
            logger.info(f"🚀 Starting comprehensive AI analysis of {pdf_path}")
            print(f"💰 Starting comprehensive analysis - this will use AI on EVERY image")
            
            # Reset counters
            self.ai_calls_count = 0
            self.current_cost_estimate = 0.0
            self.tables_processed = 0
            self.flowcharts_processed = 0
            self.figures_processed = 0
            self.links_extracted = 0
            self.extracted_links = []
            
            # Start cost tracking
            if not self.cost_tracker.start_time:
                self.cost_tracker.start_tracking()
            
            # Extract text content
            logger.info("📝 Extracting text content...")
            text_content = self.extract_text_with_ocr(pdf_path)
            
            # Analyze all images comprehensively
            logger.info("🔍 Starting comprehensive AI analysis of all images...")
            print(f"💰 Image analysis starting - cost will increase with each image")
            
            image_content = self.extract_images_and_analyze(pdf_path)
            
            # Extract metadata using AI
            logger.info("🎯 Extracting metadata with AI...")
            all_content_sample = ""
            for content_list in image_content.values():
                for content in content_list:
                    all_content_sample += content[:500] + "\n"
            
            metadata = self.extract_comprehensive_metadata(text_content, all_content_sample)
            
            # Generate filename and save
            original_filename = Path(pdf_path).name
            new_filename = self.generate_filename(metadata, original_filename, output_dir)
            output_path = os.path.join(output_dir, new_filename + ".md")
            
            # Create comprehensive markdown
            logger.info("📋 Creating comprehensive markdown...")
            markdown_content = self.create_comprehensive_markdown(
                text_content, image_content, metadata, original_filename
            )
            
            # Save file
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            # Print final cost summary
            self.cost_tracker.print_cost_summary()
            
            logger.info(f"✅ Successfully converted {pdf_path}")
            logger.info(f"💰 Final cost: ${self.cost_tracker.total_cost:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error converting {pdf_path}: {e}")
            return False
    
    def convert_batch(self, input_dir: str, output_dir: str, progress_callback=None) -> Dict[str, int]:
        """Convert batch of PDFs with comprehensive analysis and cost tracking"""
        results = {'success': 0, 'failed': 0, 'total': 0}
        batch_total_cost = 0.0
        
        self.used_filenames = set()
        
        pdf_files = []
        for ext in self.supported_formats:
            pdf_files.extend(Path(input_dir).glob('*' + ext))
        
        results['total'] = len(pdf_files)
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return results
        
        # Estimate total cost
        estimated_total_cost = results['total'] * 20  # Rough estimate
        print(f"\n💰 BATCH PROCESSING COST ESTIMATE:")
        print(f"📄 Files to process: {results['total']}")
        print(f"💸 Estimated total cost: ${estimated_total_cost:.2f} (rough estimate)")
        print(f"⚠️  This is the comprehensive version - every image gets AI analysis")
        print("=" * 70)
        
        for i, pdf_path in enumerate(pdf_files):
            try:
                print(f"\n📄 [{i+1}/{results['total']}] Processing: {Path(pdf_path).name}")
                
                if progress_callback:
                    progress_callback(i + 1, results['total'], str(pdf_path))
                
                # Reset cost tracker for each file
                self.cost_tracker = CostTracker()
                
                if self.convert_pdf_comprehensive(str(pdf_path), output_dir):
                    results['success'] += 1
                    file_cost = self.cost_tracker.total_cost
                    batch_total_cost += file_cost
                    
                    print(f"   ✅ Success - File cost: ${file_cost:.2f}")
                    print(f"   💰 Batch total so far: ${batch_total_cost:.2f}")
                    print(f"   📊 AI calls: {self.cost_tracker.total_calls}")
                    
                else:
                    results['failed'] += 1
                    print(f"   ❌ Failed")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                results['failed'] += 1
                print(f"   ❌ Error: {e}")
        
        # Final batch summary
        print(f"\n💰 FINAL BATCH COST SUMMARY:")
        print(f"={'='*50}")
        print(f"📄 Files processed: {results['success']}/{results['total']}")
        print(f"💸 Total batch cost: ${batch_total_cost:.2f}")
        print(f"📊 Average cost per file: ${batch_total_cost / max(results['success'], 1):.2f}")
        print(f"⏱️  Processing complete")
        
        return results

def main():
    """Simple command-line interface for comprehensive analysis"""
    print("💰 COMPREHENSIVE AI PDF ANALYZER - WITH COST TRACKING")
    print("=" * 60)
    print("⚠️  WARNING: This analyzes EVERY image with AI")
    print("💸 Expected cost: $15-30 per document")
    print("🎯 This is the version that actually works!")
    print("=" * 60)
    
    if len(sys.argv) < 4:
        print("Usage: python PDF_converter_COMPREHENSIVE_COST_TRACKED.py <pdf_file> <output_dir> <api_key>")
        print("\nExample:")
        print("python PDF_converter_COMPREHENSIVE_COST_TRACKED.py document.pdf output/ sk-...")
        return
    
    pdf_file = sys.argv[1]
    output_dir = sys.argv[2]
    api_key = sys.argv[3]
    
    if not os.path.exists(pdf_file):
        print(f"❌ PDF file not found: {pdf_file}")
        return
    
    # Initialize analyzer
    analyzer = ComprehensivePDFAnalyzer()
    analyzer.set_openai_key(api_key)
    
    # Convert PDF
    print(f"\n🚀 Starting comprehensive analysis of: {Path(pdf_file).name}")
    print("💰 Cost tracking enabled - you'll see real-time costs")
    
    success = analyzer.convert_pdf_comprehensive(pdf_file, output_dir)
    
    if success:
        print(f"\n✅ SUCCESS!")
        print(f"💰 Final cost: ${analyzer.cost_tracker.total_cost:.2f}")
        print(f"📊 AI calls made: {analyzer.cost_tracker.total_calls}")
        print(f"📋 Tables: {analyzer.tables_processed}")
        print(f"🌳 Flowcharts: {analyzer.flowcharts_processed}")  
        print(f"🔬 Figures: {analyzer.figures_processed}")
        print(f"🔗 Links: {analyzer.links_extracted}")
    else:
        print(f"\n❌ FAILED to convert PDF")

if __name__ == "__main__":
    main()