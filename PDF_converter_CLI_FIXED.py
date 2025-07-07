#!/usr/bin/env python3
# PDF to Markdown Converter - Command Line FIXED VERSION
# NO GUI dependencies - Works on any system
# FIXED: Prevents filename conflicts with unique naming

import os
import sys
import re
import logging
import io
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# PDF processing libraries
try:
    import fitz  # PyMuPDF
    import pdfplumber
    from PIL import Image
    import pytesseract
    import pandas as pd
    import numpy as np
    print("✅ All required libraries available")
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Please install with: pip install PyMuPDF pdfplumber Pillow pytesseract pandas numpy")
    sys.exit(1)

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pdf_converter.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class PDFToMarkdownConverterCLI:
    def __init__(self):
        self.supported_formats = ['.pdf']
        self.output_format = '.md'
        self.used_filenames = set()  # Track used filenames to prevent conflicts
        
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
                
                page_marker = f"\n\n--- Page {page_num + 1} ---\n\n"
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
                            table_header = f"\n\n### Table {table_num + 1} (Page {page_num + 1})\n\n"
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
                        
                        if pix.n - pix.alpha < 4:
                            img_data = pix.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                            img_text = pytesseract.image_to_string(img_pil, lang='spa+eng')
                            
                            if img_text.strip():
                                figure_header = f"\n\n### Figure {img_index + 1} (Page {page_num + 1})\n\n"
                                ocr_content = f"**Image Content (OCR):**\n{img_text.strip()}\n\n"
                                figure_content = figure_header + ocr_content
                                image_descriptions.append(figure_content)
                        
                        pix = None
                        
                    except Exception as e:
                        logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
            
        return image_descriptions
    
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
        """Generate unique filename with conflict resolution - FIXED VERSION"""
        
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
            final_filename = f"{base_filename}_{counter}"
            output_path = os.path.join(output_dir, final_filename + ".md")
            counter += 1
            
            # Safety valve - don't loop forever
            if counter > 100:
                timestamp = datetime.now().strftime("%H%M%S")
                final_filename = f"{base_filename}_{timestamp}"
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
        
        content_parts.append(f"# {title}")
        content_parts.append("")
        content_parts.append("## Document Information")
        content_parts.append("")
        content_parts.append(f"**Original Filename:** {original_filename}")
        content_parts.append(f"**Entity:** {entity}")
        content_parts.append(f"**Year:** {year}")
        content_parts.append(f"**Author:** {author}")
        content_parts.append(f"**Subject:** {subject}")
        content_parts.append(f"**Keywords:** {keywords}")
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
        content_parts.append(f"*Document processed on {timestamp}*")
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
            
            logger.info(f"✅ Successfully converted {pdf_path} to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error converting {pdf_path}: {e}")
            return False
    
    def convert_batch(self, input_dir: str, output_dir: str) -> Dict[str, int]:
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
        
        print(f"\n🔄 Processing {results['total']} PDF files...")
        print("=" * 60)
        
        for i, pdf_path in enumerate(pdf_files):
            try:
                print(f"\n📄 [{i+1}/{results['total']}] Processing: {Path(pdf_path).name}")
                
                if self.convert_pdf_to_markdown(str(pdf_path), output_dir):
                    results['success'] += 1
                    print(f"   ✅ Success")
                else:
                    results['failed'] += 1
                    print(f"   ❌ Failed")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                results['failed'] += 1
                print(f"   ❌ Error: {e}")
        
        return results

def main():
    print("PDF to Markdown Converter - FIXED CLI VERSION")
    print("=" * 50)
    print("🔧 This version prevents filename conflicts!")
    print("=" * 50)
    
    if len(sys.argv) != 3:
        print("\nUsage: python3 PDF_converter_CLI_FIXED.py <input_directory> <output_directory>")
        print("\nExample:")
        print("  python3 PDF_converter_CLI_FIXED.py /path/to/pdfs /path/to/output")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)
    
    # Check for Tesseract OCR
    try:
        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR found")
    except pytesseract.TesseractNotFoundError:
        print("⚠️ Warning: Tesseract OCR not found. Install with:")
        print("   macOS: brew install tesseract")
        print("   Ubuntu: sudo apt-get install tesseract-ocr")
        print("   Continuing without OCR support...")
    
    print(f"\n📂 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    converter = PDFToMarkdownConverterCLI()
    
    try:
        results = converter.convert_batch(input_dir, output_dir)
        
        print("\n" + "=" * 60)
        print("🎉 CONVERSION COMPLETE - NO OVERWRITES!")
        print("=" * 60)
        print(f"📊 Total files: {results['total']}")
        print(f"✅ Successful: {results['success']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"📂 Output directory: {output_dir}")
        print(f"📋 Log file: pdf_converter.log")
        
        if results['success'] > 0:
            print(f"\n🎯 All {results['success']} files saved with unique names!")
            print("   No overwrites or conflicts occurred.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()