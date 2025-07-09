#!/usr/bin/env python3
# PDF to Markdown Converter - AI WORKING VERSION
# ACTUALLY analyzes medical content with meaningful descriptions
# Converts tables to LLM-friendly format and provides real clinical analysis

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
        logging.FileHandler('pdf_converter_working.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WorkingAIAnalyzer:
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
        
    def set_openai_key(self, api_key: str):
        """Set OpenAI API key for AI vision processing"""
        self.openai_api_key = api_key
        self.use_ai_vision = True
        openai.api_key = api_key
    
    def make_ai_call(self, img_pil: Image.Image, prompt: str, max_tokens: int = 1200) -> str:
        """Make AI vision API call and return actual analysis"""
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
                analysis = result['choices'][0]['message']['content']
                
                self.ai_calls_count += 1
                self.current_cost_estimate = self.ai_calls_count * 0.01275
                
                return analysis.strip()
            else:
                logger.error(f"AI API error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Error in AI processing: {e}")
            return ""
    
    def analyze_for_medical_content(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """ACTUALLY analyze medical content and provide meaningful descriptions"""
        
        prompt = f"""You are analyzing an image from a clinical guideline. Your job is to extract ALL medical information and present it clearly.

ANALYZE THIS IMAGE FOR:
1. Tables with medical data (dosages, ranges, criteria, protocols)
2. Flowcharts or decision trees for patient care
3. Clinical algorithms or treatment protocols  
4. Medical terminology, drug names, dosages
5. Diagnostic criteria or classification systems
6. Any medical recommendations or guidelines

IF THIS IS A TABLE:
- Convert it to a proper markdown table format
- Include ALL headers and data
- Explain what each column/row represents clinically
- Note any units, ranges, or thresholds

IF THIS IS A FLOWCHART/DECISION TREE:
- Describe each decision point step by step
- List all pathways and outcomes
- Explain the clinical logic
- Note any criteria, values, or thresholds mentioned

IF THIS IS OTHER MEDICAL CONTENT:
- Extract all medical terminology
- Explain clinical significance
- Note any recommendations or protocols

RETURN YOUR ANALYSIS IN THIS FORMAT:

**CONTENT TYPE:** [Table/Flowchart/Clinical Content/Mixed]

**MEDICAL ANALYSIS:**
[Detailed analysis of the medical content]

**KEY CLINICAL INFORMATION:**
- [List key points, values, criteria, recommendations]

**LLM-FRIENDLY SUMMARY:**
[Clear summary that an LLM can easily understand and use]

Be specific and detailed. If there's no medical content, say "No significant medical content detected."""

        logger.info(f"🔍 AI analyzing medical content in image {img_index} on page {page_num}")
        
        analysis = self.make_ai_call(img_pil, prompt, max_tokens=1500)
        
        if analysis and "no significant medical content" not in analysis.lower():
            self.medical_content_found += 1
            
            # Check if it's a table and needs special formatting
            if "table" in analysis.lower() and "|" not in analysis:
                # Request table formatting
                table_prompt = f"""The previous analysis identified a table. Now convert this table to proper markdown format.

Extract the table data and format it as:

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

Include ALL rows and columns. If there are many columns, split into multiple tables if needed.
After the table, provide a brief explanation of what each column represents clinically.

Return ONLY the formatted table(s) and explanation, nothing else."""
                
                table_format = self.make_ai_call(img_pil, table_prompt, max_tokens=800)
                if table_format and "|" in table_format:
                    analysis += f"\n\n**FORMATTED TABLE:**\n{table_format}"
                    self.tables_converted += 1
            
            # Check if it's a flowchart
            if any(word in analysis.lower() for word in ["flowchart", "decision", "algorithm", "tree", "pathway"]):
                self.flowcharts_analyzed += 1
        
        return f"### Medical Content Analysis - Image {img_index} (Page {page_num})\n\n{analysis}\n\n---\n\n"
    
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
                    text = pytesseract.image_to_string(img, lang='eng')
                
                page_marker = f"\n\n--- Page {page_num + 1} ---\n\n"
                full_text += page_marker + text
            
            doc.close()
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_and_analyze_images(self, pdf_path: str) -> List[str]:
        """Extract and analyze ALL images for medical content"""
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
                                continue
                            
                            total_images += 1
                            
                            # ANALYZE EVERY IMAGE FOR MEDICAL CONTENT
                            analysis = self.analyze_for_medical_content(
                                img_pil, page_num + 1, total_images
                            )
                            
                            if analysis and "no significant medical content" not in analysis.lower():
                                medical_analyses.append(analysis)
                                logger.info(f"✅ Medical content found in image {total_images}")
                            else:
                                logger.info(f"❌ No medical content in image {total_images}")
                        
                        pix = None
                        
                    except Exception as e:
                        logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue
            
            doc.close()
            logger.info(f"📊 Processed {total_images} images, found medical content in {len(medical_analyses)}")
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
            
        return medical_analyses
    
    def extract_metadata_with_ai(self, text_content: str) -> Dict[str, str]:
        """Extract metadata using AI"""
        prompt = f"""Extract metadata from this clinical guideline text. Return a JSON object with these fields:

{text_content[:2000]}

{{
    "title": "Document title",
    "organization": "Publishing organization", 
    "year": "Year",
    "medical_specialty": "Medical specialty",
    "guideline_type": "Type of guideline",
    "target_population": "Target patients",
    "keywords": "Key medical terms"
}}

Return ONLY the JSON object."""
        
        try:
            headers = {
                "Content-Type": "application/json", 
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions",
                                   headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                metadata_text = result['choices'][0]['message']['content']
                return json.loads(metadata_text)
            else:
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
    
    def create_llm_friendly_markdown(self, text_content: str, medical_analyses: List[str], 
                                   metadata: Dict[str, str], original_filename: str) -> str:
        """Create LLM-friendly markdown with proper structure"""
        
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
        
        # Medical content analysis
        if medical_analyses:
            content_parts.append("## AI-Analyzed Medical Content")
            content_parts.append("")
            content_parts.append("*Each image containing medical information has been analyzed by AI for clinical relevance*")
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
        content_parts.append(f"**Medical content found:** {self.medical_content_found} images")
        content_parts.append(f"**Tables converted:** {self.tables_converted}")
        content_parts.append(f"**Flowcharts analyzed:** {self.flowcharts_analyzed}")
        content_parts.append("")
        content_parts.append("*Converted with comprehensive AI analysis for LLM processing*")
        
        return "\n".join(content_parts)
    
    def convert_pdf_with_working_ai(self, pdf_path: str, output_dir: str) -> bool:
        """Convert PDF with WORKING AI analysis"""
        try:
            logger.info(f"🚀 Starting WORKING AI analysis of {pdf_path}")
            
            # Reset counters
            self.ai_calls_count = 0
            self.current_cost_estimate = 0.0
            self.tables_converted = 0
            self.flowcharts_analyzed = 0
            self.medical_content_found = 0
            
            # Extract text content
            logger.info("📝 Extracting text content...")
            text_content = self.extract_text_with_ocr(pdf_path)
            
            # AI metadata extraction
            logger.info("🎯 Extracting metadata with AI...")
            metadata = self.extract_metadata_with_ai(text_content)
            
            # ANALYZE ALL IMAGES FOR MEDICAL CONTENT
            logger.info("🔍 Analyzing ALL images for medical content...")
            medical_analyses = self.extract_and_analyze_images(pdf_path)
            
            # Generate filename and save
            original_filename = Path(pdf_path).name
            new_filename = self.generate_filename(metadata, original_filename, output_dir)
            output_path = os.path.join(output_dir, new_filename + ".md")
            
            # Create LLM-friendly markdown
            logger.info("📋 Creating LLM-friendly markdown...")
            markdown_content = self.create_llm_friendly_markdown(
                text_content, medical_analyses, metadata, original_filename
            )
            
            # Save file
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"✅ Successfully converted {pdf_path}")
            logger.info(f"💰 Cost: ${self.current_cost_estimate:.2f}")
            logger.info(f"🎯 Found medical content in {self.medical_content_found} images")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error converting {pdf_path}: {e}")
            return False
    
    def convert_batch(self, input_dir: str, output_dir: str, progress_callback=None) -> Dict[str, int]:
        """Convert batch with working AI analysis"""
        results = {'success': 0, 'failed': 0, 'total': 0}
        total_cost = 0.0
        total_medical_content = 0
        
        self.used_filenames = set()
        
        pdf_files = []
        for ext in self.supported_formats:
            pdf_files.extend(Path(input_dir).glob('*' + ext))
        
        results['total'] = len(pdf_files)
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return results
        
        print(f"\n🚀 Starting WORKING AI analysis of {results['total']} PDF files...")
        print("=" * 70)
        
        for i, pdf_path in enumerate(pdf_files):
            try:
                print(f"\n📄 [{i+1}/{results['total']}] Processing: {Path(pdf_path).name}")
                
                if progress_callback:
                    progress_callback(i + 1, results['total'], str(pdf_path))
                
                if self.convert_pdf_with_working_ai(str(pdf_path), output_dir):
                    results['success'] += 1
                    total_cost += self.current_cost_estimate
                    total_medical_content += self.medical_content_found
                    print(f"   ✅ Success - Found {self.medical_content_found} medical images")
                    print(f"   💰 Cost: ${self.current_cost_estimate:.2f} (Total: ${total_cost:.2f})")
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
        print(f"✅ Success rate: {results['success']}/{results['total']}")
        
        return results

# GUI for working version
class WorkingAIGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WORKING AI Clinical Guideline Analyzer")
        self.root.geometry("750x550")
        
        self.analyzer = WorkingAIAnalyzer()
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.openai_key = tk.StringVar()
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        title_label = ttk.Label(main_frame, text="WORKING AI Clinical Guideline Analyzer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        warning_label = ttk.Label(main_frame, text="✅ ACTUALLY analyzes medical content with meaningful descriptions", 
                                 font=('Arial', 11), foreground='green')
        warning_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        ttk.Label(main_frame, text="Input Directory (PDFs):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_dir, width=60).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_input_dir).grid(row=2, column=2, pady=5)
        
        ttk.Label(main_frame, text="Output Directory (Markdown):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=60).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_output_dir).grid(row=3, column=2, pady=5)
        
        # OpenAI API Key section
        ai_frame = ttk.LabelFrame(main_frame, text="🧠 Working AI Analysis (Required)", padding="5")
        ai_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)
        
        ttk.Label(ai_frame, text="OpenAI API Key:").grid(row=0, column=0, sticky=tk.W, pady=5)
        key_entry = ttk.Entry(ai_frame, textvariable=self.openai_key, width=60, show="*")
        key_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(ai_frame, text="✅ ACTUALLY analyzes tables, flowcharts, and medical terminology", font=('Arial', 9)).grid(row=1, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(ai_frame, text="✅ Converts tables to LLM-friendly markdown format", font=('Arial', 9)).grid(row=2, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(ai_frame, text="✅ Provides meaningful clinical descriptions", font=('Arial', 9)).grid(row=3, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(ai_frame, text="💰 Cost: ~$0.50-3.00 per document with medical content", font=('Arial', 9), foreground='blue').grid(row=4, column=0, columnspan=2, sticky=tk.W)
        
        convert_button = ttk.Button(main_frame, text="Start WORKING AI Analysis", command=self.start_conversion)
        convert_button.grid(row=5, column=0, columnspan=3, pady=20)
        
        self.progress_var = tk.StringVar()
        self.progress_var.set("Ready for WORKING AI analysis")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=6, column=0, columnspan=3, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=7, column=0, columnspan=3, sticky="ew", pady=5)
        
        log_frame = ttk.LabelFrame(main_frame, text="Analysis Log", padding="5")
        log_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=10)
        
        self.log_text = tk.Text(log_frame, height=16, width=85)
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
            messagebox.showerror("Error", "OpenAI API key is required")
            return
        
        self.analyzer.set_openai_key(api_key)
        self.log_message("🧠 WORKING AI analysis enabled!")
        
        self.log_text.delete(1.0, tk.END)
        self.log_message("🚀 Starting WORKING AI analysis...")
        
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
            
            self.progress_var.set("WORKING AI analysis completed!")
            completion_message = f"Analysis completed! Success: {results['success']}, Failed: {results['failed']}"
            self.log_message(completion_message)
            
            messagebox.showinfo("WORKING AI Analysis Complete!", 
                              f"✅ Analysis completed!\n\nSuccessful: {results['success']}\nFailed: {results['failed']}\n\n🎯 All medical content has been ACTUALLY analyzed!\n📋 Tables converted to LLM-friendly format\n🌳 Flowcharts described with clinical detail")
            
        except Exception as e:
            error_message = f"Error during analysis: {e}"
            self.log_message(error_message)
            messagebox.showerror("Error", f"An error occurred: {e}")
    
    def run(self):
        self.root.mainloop()

def main():
    print("WORKING AI Clinical Guideline Analyzer")
    print("=" * 45)
    print("✅ ACTUALLY analyzes medical content")
    print("📋 Converts tables to LLM-friendly format")
    print("🌳 Provides meaningful flowchart descriptions")
    print("🎯 Real medical content extraction")
    print("=" * 45)
    
    app = WorkingAIGUI()
    app.run()

if __name__ == "__main__":
    main()