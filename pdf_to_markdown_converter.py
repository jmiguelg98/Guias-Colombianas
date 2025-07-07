#!/usr/bin/env python3
"""
PDF to Markdown Converter for Colombian Clinical Guidelines
Converts PDFs (normal and scanned) to LLM-friendly markdown files
"""

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
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install required packages:")
    print("pip install PyMuPDF pdfplumber Pillow pytesseract pandas numpy")
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
    """Main converter class for PDF to Markdown conversion"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
        self.output_format = '.md'
        
    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR for scanned documents"""
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
                
                full_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
            
            doc.close()
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract tables from PDF and convert to markdown format"""
        tables_markdown = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    for table_num, table in enumerate(tables):
                        if table:
                            markdown_table = self.table_to_markdown(table)
                            table_text = f"\n\n### Table {table_num + 1} (Page {page_num + 1})\n\n{markdown_table}\n\n"
                            tables_markdown.append(table_text)
                            
        except Exception as e:
            logger.error(f"Error extracting tables from {pdf_path}: {e}")
            
        return tables_markdown
    
    def table_to_markdown(self, table: List[List[str]]) -> str:
        """Convert table data to markdown format"""
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
        """Extract images from PDF and convert to text descriptions using OCR"""
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
                        
                        if pix.n - pix.alpha < 4:
                            img_data = pix.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                            img_text = pytesseract.image_to_string(img_pil, lang='spa+eng')
                            
                            if img_text.strip():
                                fig_text = f"\n\n### Figure {img_index + 1} (Page {page_num + 1})\n\n"
                                fig_text += f"**Image Content (OCR):**\n{img_text.strip()}\n\n"
                                image_descriptions.append(fig_text)
                        
                        pix = None
                        
                    except Exception as e:
                        logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
            
        return image_descriptions
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        """Extract metadata from PDF with focus on Colombian clinical guidelines"""
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
    
    def generate_filename(self, metadata: Dict[str, str], original_filename: str) -> str:
        """Generate filename based on metadata"""
        title = metadata.get('title', '').strip()
        entity = metadata.get('entity', '').strip()
        year = metadata.get('year', '').strip()
        
        if not any([title, entity, year]):
            return Path(original_filename).stem
        
        filename_parts = []
        
        if title:
            clean_title = re.sub(r'[^\w\s-]', '', title)
            clean_title = re.sub(r'\s+', '_', clean_title)
            if len(clean_title) > 50:
                clean_title = clean_title[:50]
            filename_parts.append(clean_title)
        
        if entity:
            clean_entity = re.sub(r'[^\w\s-]', '', entity)
            clean_entity = re.sub(r'\s+', '_', clean_entity)
            if len(clean_entity) > 30:
                clean_entity = clean_entity[:30]
            filename_parts.append(clean_entity)
        
        if year:
            filename_parts.append(year)
        
        filename = '_'.join(filename_parts)
        
        if len(filename) > 100:
            filename = filename[:100]
        
        return filename or Path(original_filename).stem
    
    def create_markdown_content(self, metadata: Dict[str, str], text_content: str, 
                              tables: List[str], images: List[str], original_filename: str) -> str:
        """Create formatted markdown content"""
        
        title = metadata.get('title', 'Clinical Guideline')
        entity = metadata.get('entity', 'Unknown')
        year = metadata.get('year', 'Unknown')
        author = metadata.get('author', 'Not specified')
        subject = metadata.get('subject', 'Not specified')
        keywords = metadata.get('keywords', 'Not specified')
        
        content_parts = []
        
        # Header
        content_parts.append(f"# {title}\n\n")
        content_parts.append("## Document Information\n\n")
        content_parts.append(f"**Original Filename:** {original_filename}\n")
        content_parts.append(f"**Entity:** {entity}\n")
        content_parts.append(f"**Year:** {year}\n")
        content_parts.append(f"**Author:** {author}\n")
        content_parts.append(f"**Subject:** {subject}\n")
        content_parts.append(f"**Keywords:** {keywords}\n\n")
        content_parts.append("---\n\n")
        content_parts.append("## Document Content\n\n")
        
        # Main text
        if text_content:
            content_parts.append("### Main Text Content\n\n")
            content_parts.append(text_content)
            content_parts.append("\n\n")
        
        # Tables
        if tables:
            content_parts.append("## Tables\n\n")
            content_parts.extend(tables)
        
        # Images
        if images:
            content_parts.append("## Figures and Images\n\n")
            content_parts.extend(images)
        
        # Footer
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        content_parts.append(f"\n\n---\n\n*Document processed on {timestamp}*\n")
        content_parts.append("*Converted from PDF to Markdown for LLM processing*\n")
        
        return ''.join(content_parts)
    
    def convert_pdf_to_markdown(self, pdf_path: str, output_dir: str) -> bool:
        """Convert a single PDF to markdown format"""
        try:
            logger.info(f"Converting {pdf_path}")
            
            metadata = self.extract_metadata(pdf_path)
            original_filename = Path(pdf_path).name
            new_filename = self.generate_filename(metadata, original_filename)
            output_path = os.path.join(output_dir, f"{new_filename}.md")
            
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
        """Convert multiple PDFs in batch"""
        results = {'success': 0, 'failed': 0, 'total': 0}
        
        pdf_files = []
        for ext in self.supported_formats:
            pdf_files.extend(Path(input_dir).glob(f'*{ext}'))
        
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
    """GUI for PDF to Markdown converter"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PDF to Markdown Converter - Colombian Clinical Guidelines")
        self.root.geometry("600x500")
        
        self.converter = PDFToMarkdownConverter()
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        title_label = ttk.Label(main_frame, text="PDF to Markdown Converter", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        ttk.Label(main_frame, text="Input Directory (PDFs):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_dir, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_input_dir).grid(row=1, column=2, pady=5)
        
        ttk.Label(main_frame, text="Output Directory (Markdown):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.select_output_dir).grid(row=2, column=2, pady=5)
        
        convert_button = ttk.Button(main_frame, text="Convert PDFs", command=self.start_conversion)
        convert_button.grid(row=3, column=0, columnspan=3, pady=20)
        
        self.progress_var = tk.StringVar()
        self.progress_var.set("Ready to convert")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=4, column=0, columnspan=3, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)
        
        log_frame = ttk.LabelFrame(main_frame, text="Conversion Log", padding="5")
        log_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=10)
        
        self.log_text = tk.Text(log_frame, height=15, width=70)
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
    def select_input_dir(self):
        """Select input directory"""
        directory = filedialog.askdirectory(title="Select Directory with PDF Files")
        if directory:
            self.input_dir.set(directory)
    
    def select_output_dir(self):
        """Select output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory for Markdown Files")
        if directory:
            self.output_dir.set(directory)
    
    def log_message(self, message: str):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_progress(self, current: int, total: int, current_file: str):
        """Update progress bar and status"""
        progress = (current / total) * 100
        self.progress_bar['value'] = progress
        self.progress_var.set(f"Processing {current}/{total}: {Path(current_file).name}")
        self.log_message(f"Processing: {Path(current_file).name}")
        self.root.update()
    
    def start_conversion(self):
        """Start the conversion process"""
        if not self.input_dir.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Please select both input and output directories")
            return
        
        if not os.path.exists(self.input_dir.get()):
            messagebox.showerror("Error", "Input directory does not exist")
            return
        
        self.log_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self.run_conversion)
        thread.daemon = True
        thread.start()
    
    def run_conversion(self):
        """Run the conversion process"""
        try:
            self.log_message("Starting PDF to Markdown conversion...")
            
            results = self.converter.convert_batch(
                self.input_dir.get(),
                self.output_dir.get(),
                self.update_progress
            )
            
            self.progress_var.set("Conversion completed!")
            self.log_message(f"Conversion completed! Success: {results['success']}, Failed: {results['failed']}")
            
            success_msg = f"Conversion completed!\n\n"
            success_msg += f"Total files: {results['total']}\n"
            success_msg += f"Successful: {results['success']}\n"
            success_msg += f"Failed: {results['failed']}"
            
            messagebox.showinfo("Conversion Complete", success_msg)
            
        except Exception as e:
            self.log_message(f"Error during conversion: {e}")
            messagebox.showerror("Error", f"An error occurred during conversion: {e}")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def main():
    """Main function"""
    if not hasattr(sys, 'version_info'):
        print("Error: Python version check failed")
        sys.exit(1)
    
    print("PDF to Markdown Converter for Colombian Clinical Guidelines")
    print("=" * 60)
    
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