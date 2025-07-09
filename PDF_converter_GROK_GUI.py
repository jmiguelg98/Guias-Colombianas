#!/usr/bin/env python3
"""
PDF to Markdown Converter - GROK VISION GUI VERSION
Uses Grok's advanced vision capabilities for medical image analysis
Comprehensive analysis of flowcharts, tables, figures, and layouts
"""

import os
import sys
import re
import logging
import io
import base64
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import requests
import json

# PDF processing libraries
try:
    import fitz  # PyMuPDF
    from PIL import Image
    import pytesseract
    
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install: pip install PyMuPDF Pillow pytesseract")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GrokCostTracker:
    """Track Grok API costs in real-time"""
    
    def __init__(self):
        self.total_cost = 0.0
        self.image_calls = 0
        self.text_calls = 0
        
        # Grok pricing (approximate - check current rates)
        self.image_analysis_cost = 0.015  # Per image analysis
        self.text_analysis_cost = 0.002   # Per text analysis
    
    def add_image_analysis(self):
        """Add cost for image analysis"""
        self.image_calls += 1
        cost = self.image_analysis_cost
        self.total_cost += cost
        return cost
    
    def add_text_analysis(self):
        """Add cost for text analysis"""
        self.text_calls += 1
        cost = self.text_analysis_cost
        self.total_cost += cost
        return cost
    
    def get_summary(self) -> Dict:
        """Get cost summary"""
        return {
            'total_cost': round(self.total_cost, 3),
            'image_calls': self.image_calls,
            'text_calls': self.text_calls,
            'avg_per_image': round(self.total_cost / max(self.image_calls, 1), 3)
        }

class GrokVisionAnalyzer:
    """Grok-powered medical image analysis"""
    
    def __init__(self, api_key: str, progress_callback=None):
        self.api_key = api_key
        self.progress_callback = progress_callback
        self.cost_tracker = GrokCostTracker()
        
        # Grok API endpoint
        self.base_url = "https://api.x.ai/v1"
        
        # Analysis tracking
        self.tables_found = 0
        self.flowcharts_found = 0
        self.figures_found = 0
        self.text_content_found = 0
        self.mixed_content_found = 0
        
        self.log_message("🚀 Grok Vision Analyzer initialized")
    
    def log_message(self, message: str):
        """Send message to GUI"""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message)
    
    def encode_image_base64(self, img_pil: Image.Image) -> str:
        """Convert PIL image to base64"""
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def call_grok_vision(self, img_pil: Image.Image, prompt: str) -> str:
        """Call Grok vision API"""
        try:
            img_base64 = self.encode_image_base64(img_pil)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "grok-vision-beta",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['choices'][0]['message']['content']
                cost = self.cost_tracker.add_image_analysis()
                self.log_message(f"💰 Grok analysis cost: ${cost:.3f} (Total: ${self.cost_tracker.total_cost:.3f})")
                return analysis
            else:
                error_msg = f"Grok API error {response.status_code}: {response.text}"
                self.log_message(f"❌ {error_msg}")
                return f"Error: {error_msg}"
                
        except Exception as e:
            logger.error(f"Grok API call failed: {e}")
            return f"Error calling Grok: {e}"
    
    def classify_image_content(self, img_pil: Image.Image) -> str:
        """Classify medical image content using Grok"""
        prompt = """Look at this medical image and classify it as one of these types:
- TABLE: Contains structured data in rows/columns (lab results, dosages, etc.)
- FLOWCHART: Shows decision trees, algorithms, or process flows
- FIGURE: Anatomical diagrams, charts, graphs, or illustrations  
- TEXT: Primarily text content (guidelines, recommendations)
- MIXED: Combination of multiple content types

Respond with just the classification: TABLE, FLOWCHART, FIGURE, TEXT, or MIXED"""
        
        response = self.call_grok_vision(img_pil, prompt)
        
        # Parse response
        response_upper = response.upper()
        if "TABLE" in response_upper:
            return "TABLE"
        elif "FLOWCHART" in response_upper:
            return "FLOWCHART"
        elif "FIGURE" in response_upper:
            return "FIGURE"
        elif "TEXT" in response_upper:
            return "TEXT"
        elif "MIXED" in response_upper:
            return "MIXED"
        else:
            return "MIXED"  # Default fallback
    
    def analyze_table_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Comprehensive table analysis with Grok"""
        
        prompt = """Analyze this medical table in comprehensive detail:

EXTRACT ALL DATA:
1. Table headers and column names
2. Every data value, number, dosage, range
3. Units of measurement 
4. Row labels and categories
5. Any footnotes or annotations

MEDICAL ANALYSIS:
6. What clinical information does this table provide?
7. How would healthcare providers use this data?
8. What are the key medical insights?
9. Are there normal/abnormal ranges indicated?
10. What treatment decisions could this inform?

FORMAT: Provide a structured analysis that preserves all data while explaining clinical significance."""
        
        self.log_message(f"📊 Analyzing medical table {img_index} (Page {page_num}) with Grok")
        
        analysis = self.call_grok_vision(img_pil, prompt)
        self.tables_found += 1
        
        # Add OCR supplement
        try:
            ocr_text = pytesseract.image_to_string(img_pil, lang='eng')
            if ocr_text.strip():
                analysis += f"\n\n**OCR EXTRACTED DATA:**\n{ocr_text.strip()}"
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
        
        return f"### 📊 Medical Table Analysis - Image {img_index} (Page {page_num})\n\n**Content Type:** Table\n\n**Grok Analysis:**\n{analysis}\n\n---\n\n"
    
    def analyze_flowchart_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Comprehensive flowchart analysis with Grok"""
        
        prompt = """Analyze this medical flowchart/algorithm in complete detail:

DECISION PROCESS MAPPING:
1. Map out every decision point and pathway
2. Identify entry points and exit conditions
3. List all criteria, thresholds, and decision rules
4. Trace each possible pathway from start to finish

CLINICAL IMPLEMENTATION:
5. What medical condition or situation does this address?
6. When would clinicians use this algorithm?
7. What are the key decision criteria?
8. What are the recommended actions/treatments?
9. How does this improve patient care?

STEP-BY-STEP BREAKDOWN:
10. Provide a numbered workflow that clinicians could follow

Make this analysis practical for healthcare providers to implement."""
        
        self.log_message(f"🌳 Analyzing medical flowchart {img_index} (Page {page_num}) with Grok")
        
        analysis = self.call_grok_vision(img_pil, prompt)
        self.flowcharts_found += 1
        
        # Add OCR for text elements
        try:
            ocr_text = pytesseract.image_to_string(img_pil, lang='eng')
            if ocr_text.strip():
                analysis += f"\n\n**TEXT ELEMENTS:**\n{ocr_text.strip()}"
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
        
        return f"### 🌳 Medical Flowchart Analysis - Image {img_index} (Page {page_num})\n\n**Content Type:** Flowchart/Algorithm\n\n**Grok Analysis:**\n{analysis}\n\n---\n\n"
    
    def analyze_figure_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Comprehensive figure analysis with Grok"""
        
        prompt = """Analyze this medical figure/diagram with complete detail:

VISUAL CONTENT ANALYSIS:
1. What medical concept, anatomy, or process is illustrated?
2. Identify all labeled structures, components, or elements
3. Describe spatial relationships and layouts
4. Note any arrows, connections, or flow indicators
5. Extract all visible text, labels, and annotations

CLINICAL SIGNIFICANCE:
6. What is the educational or clinical purpose?
7. How does this support medical understanding?
8. What key medical principles does it demonstrate?
9. Are there normal vs. abnormal comparisons?
10. How would this be used in medical practice?

Provide a detailed description that would help someone understand the medical content without seeing the image."""
        
        self.log_message(f"🔬 Analyzing medical figure {img_index} (Page {page_num}) with Grok")
        
        analysis = self.call_grok_vision(img_pil, prompt)
        self.figures_found += 1
        
        # Add OCR for labels
        try:
            ocr_text = pytesseract.image_to_string(img_pil, lang='eng')
            if ocr_text.strip():
                analysis += f"\n\n**LABELS/TEXT EXTRACTED:**\n{ocr_text.strip()}"
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
        
        return f"### 🔬 Medical Figure Analysis - Image {img_index} (Page {page_num})\n\n**Content Type:** Figure/Diagram\n\n**Grok Analysis:**\n{analysis}\n\n---\n\n"
    
    def analyze_text_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Comprehensive text analysis with Grok"""
        
        prompt = """Analyze this medical text content comprehensively:

TEXT EXTRACTION & ORGANIZATION:
1. Extract all visible text content accurately
2. Identify headers, subheadings, and sections
3. Preserve bullet points, numbered lists, and formatting
4. Note any emphasized or highlighted text

MEDICAL CONTENT ANALYSIS:
5. What are the key medical recommendations?
6. What clinical guidelines are provided?
7. Are there dosages, contraindications, or warnings?
8. What patient populations are addressed?
9. How should healthcare providers apply this information?

Provide both the complete text and medical analysis."""
        
        self.log_message(f"📝 Analyzing medical text {img_index} (Page {page_num}) with Grok")
        
        analysis = self.call_grok_vision(img_pil, prompt)
        self.text_content_found += 1
        
        # OCR is critical for text
        try:
            ocr_text = pytesseract.image_to_string(img_pil, lang='eng')
            if ocr_text.strip():
                analysis += f"\n\n**COMPLETE TEXT EXTRACTION:**\n{ocr_text.strip()}"
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
        
        return f"### 📝 Medical Text Analysis - Image {img_index} (Page {page_num})\n\n**Content Type:** Text Content\n\n**Grok Analysis:**\n{analysis}\n\n---\n\n"
    
    def analyze_mixed_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Comprehensive mixed content analysis with Grok"""
        
        prompt = """This image contains mixed medical content (tables, text, figures, etc.). Provide comprehensive analysis:

CONTENT IDENTIFICATION:
1. Identify all different content types present
2. Describe the layout and organization
3. Note how different elements relate to each other

COMPREHENSIVE EXTRACTION:
4. Extract all tabular data and structures
5. Map any flowcharts or decision processes  
6. Describe all figures, diagrams, or illustrations
7. Extract all text content and recommendations

INTEGRATED ANALYSIS:
8. How do the different elements work together?
9. What is the overall clinical message?
10. How would healthcare providers use this integrated information?

Provide a unified analysis that captures all content types and their relationships."""
        
        self.log_message(f"🔄 Analyzing mixed medical content {img_index} (Page {page_num}) with Grok")
        
        analysis = self.call_grok_vision(img_pil, prompt)
        self.mixed_content_found += 1
        
        # Comprehensive OCR
        try:
            ocr_text = pytesseract.image_to_string(img_pil, lang='eng')
            if ocr_text.strip():
                analysis += f"\n\n**COMPLETE TEXT EXTRACTION:**\n{ocr_text.strip()}"
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
        
        return f"### 🔄 Mixed Content Analysis - Image {img_index} (Page {page_num})\n\n**Content Type:** Mixed Content\n\n**Grok Analysis:**\n{analysis}\n\n---\n\n"
    
    def process_image(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Process image with Grok's comprehensive analysis"""
        try:
            # First classify with Grok
            content_type = self.classify_image_content(img_pil)
            self.log_message(f"🔍 Image {img_index} (Page {page_num}) classified as: {content_type}")
            
            # Route to specialized analysis
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
            return f"### Image {img_index} (Page {page_num})\n\n**Error:** {e}\n\n---\n\n"
    
    def get_analysis_summary(self) -> Dict:
        """Get comprehensive analysis summary"""
        cost_summary = self.cost_tracker.get_summary()
        return {
            'total_analyses': self.cost_tracker.image_calls,
            'tables_found': self.tables_found,
            'flowcharts_found': self.flowcharts_found,
            'figures_found': self.figures_found,
            'text_content_found': self.text_content_found,
            'mixed_content_found': self.mixed_content_found,
            'total_cost': cost_summary['total_cost'],
            'avg_per_image': cost_summary['avg_per_image']
        }

class GrokPDFAnalyzer:
    """Main PDF analyzer using Grok Vision"""
    
    def __init__(self, api_key: str, progress_callback=None):
        self.api_key = api_key
        self.progress_callback = progress_callback
        self.vision_analyzer = GrokVisionAnalyzer(api_key, progress_callback)
        
        self.supported_formats = ['.pdf']
        self.used_filenames = set()
        self.extracted_links = []
        
    def log_message(self, message: str):
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
                    text = pytesseract.image_to_string(img, lang='eng')
                
                page_marker = f"\n\n--- Page {page_num + 1} ---\n\n"
                full_text += page_marker + text
                
                # Extract links
                self.extract_links_from_text(text)
            
            doc.close()
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    def extract_links_from_text(self, text: str):
        """Extract URLs and references"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        doi_pattern = r'doi:?\s*10\.\d+/[^\s]+'
        
        urls = re.findall(url_pattern, text)
        dois = re.findall(doi_pattern, text, re.IGNORECASE)
        
        self.extracted_links.extend([f"URL: {url}" for url in urls])
        self.extracted_links.extend([f"DOI: {doi}" for doi in dois])
    
    def extract_and_analyze_images(self, pdf_path: str) -> List[str]:
        """Extract and analyze images with Grok"""
        image_analyses = []
        
        try:
            doc = fitz.open(pdf_path)
            total_images = 0
            
            self.log_message("🚀 Starting Grok image analysis...")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha >= 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        if pix.n - pix.alpha <= 3:
                            img_data = pix.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                            
                            width, height = img_pil.size
                            if width < 100 or height < 100:
                                continue
                            
                            total_images += 1
                            
                            # Analyze with Grok
                            analysis = self.vision_analyzer.process_image(
                                img_pil, page_num + 1, total_images
                            )
                            
                            image_analyses.append(analysis)
                            
                        pix = None
                        
                    except Exception as e:
                        logger.error(f"Error processing image: {e}")
                        continue
            
            doc.close()
            
            # Print summary
            summary = self.vision_analyzer.get_analysis_summary()
            self.log_message("🎉 GROK ANALYSIS COMPLETE!")
            self.log_message(f"📊 Images analyzed: {total_images}")
            self.log_message(f"📋 Tables: {summary['tables_found']}")
            self.log_message(f"🌳 Flowcharts: {summary['flowcharts_found']}")
            self.log_message(f"🔬 Figures: {summary['figures_found']}")
            self.log_message(f"📝 Text content: {summary['text_content_found']}")
            self.log_message(f"💰 Total cost: ${summary['total_cost']:.2f}")
            
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            
        return image_analyses
    
    def convert_pdf_with_grok(self, pdf_path: str, output_dir: str) -> bool:
        """Convert PDF using Grok Vision"""
        try:
            self.log_message(f"🚀 Starting Grok analysis of {Path(pdf_path).name}")
            
            self.extracted_links = []
            
            # Extract text
            self.log_message("📝 Extracting text content...")
            text_content = self.extract_text_with_ocr(pdf_path)
            
            # Analyze images with Grok
            self.log_message("🔍 Analyzing images with Grok Vision...")
            image_analyses = self.extract_and_analyze_images(pdf_path)
            
            # Create output
            original_filename = Path(pdf_path).name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_name = re.sub(r'[^\w\s-]', '', Path(original_filename).stem)[:50]
            new_filename = f"{clean_name}_GROK_{timestamp}"
            output_path = os.path.join(output_dir, new_filename + ".md")
            
            # Create markdown
            markdown_content = self.create_markdown_output(
                text_content, image_analyses, original_filename
            )
            
            # Save file
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            summary = self.vision_analyzer.get_analysis_summary()
            self.log_message(f"✅ SUCCESS! Saved as: {new_filename}.md")
            self.log_message(f"💰 Total cost: ${summary['total_cost']:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error converting {pdf_path}: {e}")
            self.log_message(f"❌ Error: {e}")
            return False
    
    def create_markdown_output(self, text_content: str, image_analyses: List[str], 
                             original_filename: str) -> str:
        """Create comprehensive markdown output"""
        
        content_parts = []
        
        # Header
        content_parts.append(f"# Medical Guideline Analysis: {Path(original_filename).stem}")
        content_parts.append("")
        content_parts.append("*Analyzed with Grok Vision AI*")
        content_parts.append("")
        
        # Document info
        content_parts.append("## Document Information")
        content_parts.append("")
        content_parts.append(f"**Original Filename:** {original_filename}")
        content_parts.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content_parts.append(f"**Analysis Method:** Grok Vision API")
        content_parts.append("")
        content_parts.append("---")
        content_parts.append("")
        
        # Main text
        if text_content:
            content_parts.append("## Document Text Content")
            content_parts.append("")
            content_parts.append(text_content)
            content_parts.append("")
            content_parts.append("---")
            content_parts.append("")
        
        # Image analyses
        if image_analyses:
            content_parts.append("## 🚀 Grok Vision Analysis Results")
            content_parts.append("")
            content_parts.append("*Each image comprehensively analyzed using Grok's advanced vision capabilities*")
            content_parts.append("")
            
            for analysis in image_analyses:
                content_parts.append(analysis)
        
        # Links
        if self.extracted_links:
            content_parts.append("## Links and References")
            content_parts.append("")
            for link in set(self.extracted_links):
                content_parts.append(f"- {link}")
            content_parts.append("")
            content_parts.append("---")
            content_parts.append("")
        
        # Summary
        summary = self.vision_analyzer.get_analysis_summary()
        content_parts.append("## 🚀 Grok Processing Summary")
        content_parts.append("")
        content_parts.append(f"**Analysis Method:** Grok Vision API")
        content_parts.append(f"**Total Cost:** ${summary['total_cost']:.2f}")
        content_parts.append(f"**Images Analyzed:** {summary['total_analyses']}")
        content_parts.append(f"**Tables Found:** {summary['tables_found']}")
        content_parts.append(f"**Flowcharts Found:** {summary['flowcharts_found']}")
        content_parts.append(f"**Figures Found:** {summary['figures_found']}")
        content_parts.append(f"**Text Content Found:** {summary['text_content_found']}")
        content_parts.append(f"**Mixed Content Found:** {summary['mixed_content_found']}")
        content_parts.append(f"**Average Cost per Image:** ${summary['avg_per_image']:.3f}")
        content_parts.append("")
        content_parts.append("**Grok Capabilities:**")
        content_parts.append("- Advanced vision understanding")
        content_parts.append("- Medical content specialization")
        content_parts.append("- Layout and structure analysis")
        content_parts.append("- Comprehensive flowchart interpretation")
        content_parts.append("- Clinical context understanding")
        
        return "\n".join(content_parts)

# GUI for Grok Version
class GrokPDFAnalyzerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Grok Vision PDF Analyzer - Medical AI Analysis")
        self.root.geometry("900x700")
        
        self.analyzer = None
        self.api_key = tk.StringVar()
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Title
        title_label = ttk.Label(main_frame, text="🚀 Grok Vision PDF Analyzer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, text="Advanced Medical Image Analysis with Grok AI", 
                                  font=('Arial', 12), foreground='blue')
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 15))
        
        # API Key
        ttk.Label(main_frame, text="Grok API Key:").grid(row=2, column=0, sticky=tk.W, pady=5)
        api_entry = ttk.Entry(main_frame, textvariable=self.api_key, width=70, show="*")
        api_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Directories
        ttk.Label(main_frame, text="Input Directory (PDFs):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_dir, width=70).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_input_dir).grid(row=3, column=2, pady=5)
        
        ttk.Label(main_frame, text="Output Directory:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=70).grid(row=4, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_output_dir).grid(row=4, column=2, pady=5)
        
        # Grok info
        info_frame = ttk.LabelFrame(main_frame, text="🚀 Grok Vision Capabilities", padding="5")
        info_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=10)
        
        ttk.Label(info_frame, text="🔬 Advanced medical image understanding", font=('Arial', 9)).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(info_frame, text="📊 Comprehensive table analysis and data extraction", font=('Arial', 9)).grid(row=1, column=0, sticky=tk.W)
        ttk.Label(info_frame, text="🌳 Detailed flowchart and algorithm interpretation", font=('Arial', 9)).grid(row=2, column=0, sticky=tk.W)
        ttk.Label(info_frame, text="🔍 Layout analysis and structure recognition", font=('Arial', 9)).grid(row=3, column=0, sticky=tk.W)
        ttk.Label(info_frame, text="💰 Cost tracking: ~$0.015 per image analysis", font=('Arial', 9)).grid(row=4, column=0, sticky=tk.W)
        
        # Convert button
        convert_button = ttk.Button(main_frame, text="Start Grok Analysis", command=self.start_conversion)
        convert_button.grid(row=6, column=0, columnspan=3, pady=20)
        
        # Progress
        self.progress_var = tk.StringVar()
        self.progress_var.set("Ready for Grok analysis")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=7, column=0, columnspan=3, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=8, column=0, columnspan=3, sticky="ew", pady=5)
        
        # Log
        log_frame = ttk.LabelFrame(main_frame, text="Grok Analysis Log", padding="5")
        log_frame.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=10)
        
        self.log_text = tk.Text(log_frame, height=20, width=100)
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(9, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
    def select_input_dir(self):
        directory = filedialog.askdirectory(title="Select Directory with PDF Files")
        if directory:
            self.input_dir.set(directory)
    
    def select_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
    
    def log_message(self, message: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"{timestamp} - {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update()
    
    def start_conversion(self):
        if not self.api_key.get():
            messagebox.showerror("Error", "Please enter your Grok API key")
            return
            
        if not self.input_dir.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Please select input and output directories")
            return
        
        self.log_text.delete(1.0, tk.END)
        self.log_message("🚀 Starting Grok Vision analysis...")
        
        thread = threading.Thread(target=self.run_conversion)
        thread.daemon = True
        thread.start()
    
    def run_conversion(self):
        try:
            self.analyzer = GrokPDFAnalyzer(self.api_key.get(), self.log_message)
            
            pdf_files = list(Path(self.input_dir.get()).glob('*.pdf'))
            total_files = len(pdf_files)
            
            if not pdf_files:
                self.log_message("No PDF files found!")
                return
            
            success_count = 0
            
            for i, pdf_path in enumerate(pdf_files):
                self.progress_bar['value'] = (i / total_files) * 100
                self.progress_var.set(f"Processing {i+1}/{total_files}: {pdf_path.name}")
                
                if self.analyzer.convert_pdf_with_grok(str(pdf_path), self.output_dir.get()):
                    success_count += 1
                
                self.root.update()
            
            self.progress_var.set("Grok analysis completed!")
            
            total_cost = self.analyzer.vision_analyzer.cost_tracker.total_cost
            summary = self.analyzer.vision_analyzer.get_analysis_summary()
            
            messagebox.showinfo("Analysis Complete!", 
                              f"🚀 Grok Analysis completed!\n\n"
                              f"Files processed: {success_count}/{total_files}\n"
                              f"Total cost: ${total_cost:.2f}\n\n"
                              f"📊 Images analyzed: {summary['total_analyses']}\n"
                              f"📋 Tables: {summary['tables_found']}\n"
                              f"🌳 Flowcharts: {summary['flowcharts_found']}\n"
                              f"🔬 Figures: {summary['figures_found']}\n"
                              f"📝 Text content: {summary['text_content_found']}")
            
        except Exception as e:
            self.log_message(f"❌ Error: {e}")
            messagebox.showerror("Error", f"Analysis failed: {e}")
    
    def run(self):
        self.root.mainloop()

def main():
    print("🚀 GROK VISION PDF ANALYZER")
    print("=" * 50)
    print("🔬 Advanced medical image analysis")
    print("📊 Comprehensive table interpretation")
    print("🌳 Detailed flowchart analysis")
    print("💰 Real-time cost tracking")
    print("=" * 50)
    
    app = GrokPDFAnalyzerGUI()
    app.run()

if __name__ == "__main__":
    main()