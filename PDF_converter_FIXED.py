#!/usr/bin/env python3
# PDF to Markdown Converter for Colombian Clinical Guidelines - FIXED VERSION
# Converts PDFs (normal and scanned) to LLM-friendly markdown files
# FIXED: Prevents filename conflicts with unique naming

import os
import sys
import re
import logging
import io
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
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
        logging.FileHandler('pdf_converter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PDFToMarkdownConverter:
    def __init__(self):
        self.supported_formats = ['.pdf']
        self.output_format = '.md'
        self.used_filenames = set()  # Track used filenames to prevent conflicts
        self.openai_api_key = None
        self.use_ai_vision = False
        
    def extract_text_with_ocr(self, pdf_path: str) -> str:
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
                    text = pytesseract.image_to_string(img, lang='spa+eng')
                
                page_marker = "\n\n--- Page " + str(page_num + 1) + " ---\n\n"
                full_text += page_marker + text
            
            doc.close()
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[str]:
        tables_markdown = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    for table_num, table in enumerate(tables):
                        if table:
                            markdown_table = self.table_to_markdown(table)
                            table_header = "\n\n### Table " + str(table_num + 1) + " (Page " + str(page_num + 1) + ")\n\n"
                            table_content = table_header + markdown_table + "\n\n"
                            tables_markdown.append(table_content)
                            
        except Exception as e:
            logger.error(f"Error extracting tables from {pdf_path}: {e}")
            
        return tables_markdown
    
    def table_to_markdown(self, table: List[List[str]]) -> str:
        if not table:
            return ""
        
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned_table.append(cleaned_row)
        
        if not cleaned_table:
            return ""
        
        markdown_lines = []
        header_row = "| " + " | ".join(cleaned_table[0]) + " |"
        markdown_lines.append(header_row)
        
        separator = "| " + " | ".join(["---"] * len(cleaned_table[0])) + " |"
        markdown_lines.append(separator)
        
        for row in cleaned_table[1:]:
            data_row = "| " + " | ".join(row) + " |"
            markdown_lines.append(data_row)
        
        return "\n".join(markdown_lines)
    
    def extract_images_and_describe(self, pdf_path: str) -> List[str]:
        image_descriptions = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Handle CMYK and other color spaces
                        if pix.n - pix.alpha >= 4:  # CMYK or other multi-channel
                            logger.info(f"Converting CMYK/multi-channel image to RGB on page {page_num + 1}")
                            # Convert to RGB color space
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        # Only process if we have a valid RGB/grayscale image
                        if pix.n - pix.alpha <= 3:
                            try:
                                img_data = pix.tobytes("png")
                                img_pil = Image.open(io.BytesIO(img_data))
                                
                                # Use hybrid OCR + AI processing
                                image_content = self.process_image_hybrid(img_pil, page_num + 1, img_index + 1)
                                
                                if image_content and image_content != "**Error processing image**":
                                    figure_header = "\n\n### Figure " + str(img_index + 1) + " (Page " + str(page_num + 1) + ")\n\n"
                                    figure_content = figure_header + image_content + "\n\n"
                                    image_descriptions.append(figure_content)
                                    
                            except Exception as img_error:
                                logger.warning(f"Could not process image {img_index} on page {page_num + 1}: {img_error}")
                                # Add placeholder for unprocessable image
                                figure_header = "\n\n### Figure " + str(img_index + 1) + " (Page " + str(page_num + 1) + ")\n\n"
                                placeholder_content = "**Image detected but could not be processed**\n\n"
                                figure_content = figure_header + placeholder_content
                                image_descriptions.append(figure_content)
                        
                        pix = None
                        
                    except Exception as e:
                        logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
            
        return image_descriptions
    
    def set_openai_key(self, api_key: str):
        """Set OpenAI API key for AI vision processing"""
        self.openai_api_key = api_key
        self.use_ai_vision = True
        openai.api_key = api_key
    
    def is_likely_flowchart(self, img_pil: Image.Image) -> bool:
        """Detect if image is likely a flowchart or diagram"""
        try:
            # Basic heuristics for flowchart detection
            width, height = img_pil.size
            aspect_ratio = width / height
            
            # Flowcharts often have certain characteristics
            if width < 200 or height < 200:  # Too small
                return False
                
            # Check if image has geometric shapes (basic detection)
            # This is a simple heuristic - flowcharts tend to have structured layouts
            return 0.5 <= aspect_ratio <= 3.0  # Reasonable aspect ratio for flowcharts
        except:
            return False
    
    def get_ai_image_description(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Get AI description of image using OpenAI GPT-4 Vision"""
        try:
            if not self.use_ai_vision or not self.openai_api_key:
                return ""
                
            # Convert PIL image to base64
            import io
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this medical image from Colombian clinical guidelines. If it's a flowchart or decision tree, describe the clinical workflow, decision points, patient pathways, and recommendations. If it's text or other content, extract and describe all visible information. Focus on clinical relevance and workflow logic."
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
                "max_tokens": 500
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                   headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                description = result['choices'][0]['message']['content']
                logger.info(f"AI vision successful for image {img_index} on page {page_num}")
                return description
            else:
                logger.warning(f"AI vision API error for image {img_index} on page {page_num}: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Error in AI vision processing for image {img_index} on page {page_num}: {e}")
            return ""
    
    def process_image_hybrid(self, img_pil: Image.Image, page_num: int, img_index: int) -> str:
        """Process image with hybrid OCR + AI approach"""
        try:
            # Step 1: Try OCR first
            ocr_text = pytesseract.image_to_string(img_pil, lang='spa+eng').strip()
            
            # Step 2: Check if we should use AI vision
            use_ai = False
            reason = ""
            
            if self.is_likely_flowchart(img_pil):
                use_ai = True
                reason = "Detected flowchart/diagram"
            elif len(ocr_text) < 20:  # Poor OCR results
                use_ai = True
                reason = "Poor OCR results"
            
            # Step 3: Get AI description if needed
            if use_ai and self.use_ai_vision:
                logger.info(f"Using AI vision for image {img_index} on page {page_num}: {reason}")
                ai_description = self.get_ai_image_description(img_pil, page_num, img_index)
                
                if ai_description:
                    content = f"**AI Analysis:** {ai_description}"
                    if ocr_text:  # Include OCR text if available
                        content += f"\n\n**OCR Text:** {ocr_text}"
                    return content
            
            # Step 4: Fall back to OCR or placeholder
            if ocr_text:
                return f"**OCR Text:** {ocr_text}"
            else:
                return "**Image detected but could not be processed**"
                
        except Exception as e:
            logger.error(f"Error in hybrid image processing: {e}")
            return "**Error processing image**"
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        metadata = {
            'title': '',
            'entity': '',
            'year': '',
            'author': '',
            'subject': '',
            'keywords': ''
        }
        
        try:
            doc = fitz.open(pdf_path)
            pdf_metadata = doc.metadata
            
            if pdf_metadata:
                metadata['title'] = pdf_metadata.get('title', '')
                metadata['author'] = pdf_metadata.get('author', '')
                metadata['subject'] = pdf_metadata.get('subject', '')
                metadata['keywords'] = pdf_metadata.get('keywords', '')
            
            first_pages_text = ""
            for page_num in range(min(3, len(doc))):
                page = doc.load_page(page_num)
                first_pages_text += page.get_text()
            
            doc.close()
            
            entity_patterns = [
                r'Asociación Colombiana de [A-Za-z\s]+',
                r'Sociedad Colombiana de [A-Za-z\s]+',
                r'Ministerio de Salud y Protección Social',
                r'Instituto Nacional de Salud',
                r'IETS',
                r'Colciencias',
                r'Academia Nacional de Medicina',
                r'Federación Médica Colombiana',
                r'ASCOFAME'
            ]
            
            for pattern in entity_patterns:
                match = re.search(pattern, first_pages_text, re.IGNORECASE)
                if match:
                    metadata['entity'] = match.group()
                    break
            
            year_matches = re.findall(r'\b(19|20)\d{2}\b', first_pages_text)
            if year_matches:
                years = [int(year) for year in year_matches]
                current_year = datetime.now().year
                valid_years = [year for year in years if 1990 <= year <= current_year]
                if valid_years:
                    metadata['year'] = str(max(valid_years))
            
            if not metadata['title']:
                lines = first_pages_text.split('\n')
                for line in lines[:10]:
                    line = line.strip()
                    if len(line) > 20 and len(line) < 200:
                        keywords = ['guía', 'consenso', 'recomendación', 'protocolo']
                        if any(keyword in line.lower() for keyword in keywords):
                            metadata['title'] = line
                            break
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
        
        return metadata
    
    def generate_filename(self, metadata: Dict[str, str], original_filename: str, output_dir: str) -> str:
        """Generate unique filename with conflict resolution"""
        
        # Start with clean original filename (first 25 characters)
        clean_original = re.sub(r'[^\w\s-]', '', Path(original_filename).stem)
        clean_original = re.sub(r'\s+', '_', clean_original)[:25]
        
        filename_parts = [clean_original]  # Always include original filename
        
        # Add title if available (shortened)
        title = metadata.get('title', '').strip()
        if title:
            clean_title = re.sub(r'[^\w\s-]', '', title)
            clean_title = re.sub(r'\s+', '_', clean_title)[:25]  # Limit length
            if clean_title and clean_title not in clean_original:
                filename_parts.append(clean_title)
        
        # Add entity if available (shortened)
        entity = metadata.get('entity', '').strip()
        if entity:
            clean_entity = re.sub(r'[^\w\s-]', '', entity)
            clean_entity = re.sub(r'\s+', '_', clean_entity)[:20]  # Limit length
            if clean_entity:
                filename_parts.append(clean_entity)
        
        # Add year
        year = metadata.get('year', '').strip()
        if year:
            filename_parts.append(year)
        
        # Create base filename
        base_filename = '_'.join(filename_parts)
        
        # Ensure not too long
        if len(base_filename) > 80:
            base_filename = base_filename[:80]
        
        # Check for conflicts and add counter if needed
        counter = 1
        final_filename = base_filename
        output_path = os.path.join(output_dir, final_filename + ".md")
        
        while os.path.exists(output_path) or final_filename in self.used_filenames:
            final_filename = base_filename + "_" + str(counter)
            output_path = os.path.join(output_dir, final_filename + ".md")
            counter += 1
            
            # Safety valve - don't loop forever
            if counter > 100:
                timestamp = datetime.now().strftime("%H%M%S")
                final_filename = base_filename + "_" + timestamp
                break
        
        # Track this filename as used
        self.used_filenames.add(final_filename)
        
        return final_filename
    
    def create_markdown_content(self, metadata: Dict[str, str], text_content: str, 
                              tables: List[str], images: List[str], original_filename: str) -> str:
        
        title = metadata.get('title', 'Clinical Guideline')
        entity = metadata.get('entity', 'Unknown')
        year = metadata.get('year', 'Unknown')
        author = metadata.get('author', 'Not specified')
        subject = metadata.get('subject', 'Not specified')
        keywords = metadata.get('keywords', 'Not specified')
        
        content_parts = []
        
        content_parts.append("# " + title)
        content_parts.append("")
        content_parts.append("## Document Information")
        content_parts.append("")
        content_parts.append("**Original Filename:** " + original_filename)
        content_parts.append("**Entity:** " + entity)
        content_parts.append("**Year:** " + year)
        content_parts.append("**Author:** " + author)
        content_parts.append("**Subject:** " + subject)
        content_parts.append("**Keywords:** " + keywords)
        content_parts.append("")
        content_parts.append("---")
        content_parts.append("")
        content_parts.append("## Document Content")
        content_parts.append("")
        
        if text_content:
            content_parts.append("### Main Text Content")
            content_parts.append("")
            content_parts.append(text_content)
            content_parts.append("")
        
        if tables:
            content_parts.append("## Tables")
            content_parts.append("")
            for table in tables:
                content_parts.append(table)
        
        if images:
            content_parts.append("## Figures and Images")
            content_parts.append("")
            for image in images:
                content_parts.append(image)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        content_parts.append("")
        content_parts.append("---")
        content_parts.append("")
        content_parts.append("*Document processed on " + timestamp + "*")
        content_parts.append("*Converted from PDF to Markdown for LLM processing*")
        
        return "\n".join(content_parts)
    
    def convert_pdf_to_markdown(self, pdf_path: str, output_dir: str) -> bool:
        try:
            logger.info(f"Converting {pdf_path}")
            
            metadata = self.extract_metadata(pdf_path)
            original_filename = Path(pdf_path).name
            new_filename = self.generate_filename(metadata, original_filename, output_dir)
            output_path = os.path.join(output_dir, new_filename + ".md")
            
            logger.info("Extracting text content...")
            text_content = self.extract_text_with_ocr(pdf_path)
            
            logger.info("Extracting tables...")
            tables = self.extract_tables_from_pdf(pdf_path)
            
            logger.info("Extracting and processing images...")
            images = self.extract_images_and_describe(pdf_path)
            
            markdown_content = self.create_markdown_content(
                metadata, text_content, tables, images, original_filename
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Successfully converted {pdf_path} to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting {pdf_path}: {e}")
            return False
    
    def convert_batch(self, input_dir: str, output_dir: str, progress_callback=None) -> Dict[str, int]:
        results = {'success': 0, 'failed': 0, 'total': 0}
        
        # Reset used filenames for new batch
        self.used_filenames = set()
        
        pdf_files = []
        for ext in self.supported_formats:
            pdf_files.extend(Path(input_dir).glob('*' + ext))
        
        results['total'] = len(pdf_files)
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return results
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, pdf_path in enumerate(pdf_files):
            try:
                if progress_callback:
                    progress_callback(i + 1, results['total'], str(pdf_path))
                
                if self.convert_pdf_to_markdown(str(pdf_path), output_dir):
                    results['success'] += 1
                else:
                    results['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                results['failed'] += 1
        
        return results

class PDFConverterGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PDF to Markdown Converter - AI ENHANCED (Flowcharts + No Overwrites)")
        self.root.geometry("600x500")
        
        self.converter = PDFToMarkdownConverter()
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.openai_key = tk.StringVar()
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        title_label = ttk.Label(main_frame, text="PDF to Markdown Converter - AI ENHANCED", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        warning_label = ttk.Label(main_frame, text="🤖 AI Vision for flowcharts + No filename conflicts!", 
                                 font=('Arial', 10), foreground='green')
        warning_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        ttk.Label(main_frame, text="Input Directory (PDFs):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_dir, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_input_dir).grid(row=2, column=2, pady=5)
        
        ttk.Label(main_frame, text="Output Directory (Markdown):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_output_dir).grid(row=3, column=2, pady=5)
        
        # OpenAI API Key section
        ai_frame = ttk.LabelFrame(main_frame, text="🤖 AI Vision (Optional - for Flowcharts)", padding="5")
        ai_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)
        
        ttk.Label(ai_frame, text="OpenAI API Key:").grid(row=0, column=0, sticky=tk.W, pady=5)
        key_entry = ttk.Entry(ai_frame, textvariable=self.openai_key, width=50, show="*")
        key_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(ai_frame, text="✅ Enhances flowchart analysis", font=('Arial', 9)).grid(row=1, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(ai_frame, text="✅ Hybrid: OCR first, AI for complex images", font=('Arial', 9)).grid(row=2, column=0, columnspan=2, sticky=tk.W)
        
        convert_button = ttk.Button(main_frame, text="Convert PDFs (AI-Enhanced)", command=self.start_conversion)
        convert_button.grid(row=5, column=0, columnspan=3, pady=20)
        
        self.progress_var = tk.StringVar()
        self.progress_var.set("Ready to convert - AI enhanced!")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=6, column=0, columnspan=3, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=7, column=0, columnspan=3, sticky="ew", pady=5)
        
        log_frame = ttk.LabelFrame(main_frame, text="Conversion Log", padding="5")
        log_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=10)
        
        self.log_text = tk.Text(log_frame, height=15, width=70)
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
        log_entry = timestamp + " - " + message + "\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_progress(self, current: int, total: int, current_file: str):
        progress = (current / total) * 100
        self.progress_bar['value'] = progress
        filename = Path(current_file).name
        status_text = "Processing " + str(current) + "/" + str(total) + ": " + filename
        self.progress_var.set(status_text)
        self.log_message("Processing: " + filename)
        self.root.update()
    
    def start_conversion(self):
        if not self.input_dir.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Please select both input and output directories")
            return
        
        if not os.path.exists(self.input_dir.get()):
            messagebox.showerror("Error", "Input directory does not exist")
            return
        
        # Setup OpenAI API key if provided
        api_key = self.openai_key.get().strip()
        if api_key:
            self.converter.set_openai_key(api_key)
            self.log_message("🤖 AI Vision enabled for flowchart analysis!")
        else:
            self.log_message("📝 Using OCR-only mode (no AI vision)")
        
        self.log_text.delete(1.0, tk.END)
        self.log_message("🔧 AI-ENHANCED VERSION - Preventing filename conflicts!")
        
        thread = threading.Thread(target=self.run_conversion)
        thread.daemon = True
        thread.start()
    
    def run_conversion(self):
        try:
            self.log_message("Starting PDF to Markdown conversion with conflict prevention...")
            
            results = self.converter.convert_batch(
                self.input_dir.get(),
                self.output_dir.get(),
                self.update_progress
            )
            
            ai_status = "with AI flowchart analysis" if self.converter.use_ai_vision else "with OCR only"
            self.progress_var.set(f"Conversion completed {ai_status} - All files saved safely!")
            success_count = results['success']
            failed_count = results['failed']
            completion_message = "Conversion completed! Success: " + str(success_count) + ", Failed: " + str(failed_count)
            self.log_message(completion_message)
            
            dialog_lines = [
                "✅ AI-Enhanced conversion completed with NO overwrites!",
                "",
                "Total files: " + str(results['total']),
                "Successful: " + str(results['success']),
                "Failed: " + str(results['failed']),
                "",
                "🤖 Flowcharts analyzed with AI vision" if self.converter.use_ai_vision else "📝 OCR-only processing used",
                "All files saved with unique names!"
            ]
            dialog_message = "\n".join(dialog_lines)
            
            messagebox.showinfo("AI-Enhanced Conversion Complete!", dialog_message)
            
        except Exception as e:
            error_message = "Error during conversion: " + str(e)
            self.log_message(error_message)
            messagebox.showerror("Error", "An error occurred during conversion: " + str(e))
    
    def run(self):
        self.root.mainloop()

def main():
    if not hasattr(sys, 'version_info'):
        print("Error: Python version check failed")
        sys.exit(1)
    
    print("PDF to Markdown Converter for Colombian Clinical Guidelines - AI ENHANCED")
    print("=" * 75)
    print("🤖 AI Vision for flowcharts + No filename conflicts!")
    print("🔧 Hybrid: OCR first, then AI for complex images")
    print("=" * 75)
    
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        print("Warning: Tesseract OCR not found. Please install Tesseract:")
        print("- macOS: brew install tesseract")
        print("- Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("\nContinuing without OCR support...")
    
    app = PDFConverterGUI()
    app.run()

if __name__ == "__main__":
    main()