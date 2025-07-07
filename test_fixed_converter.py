#!/usr/bin/env python3
# Test script for the FIXED PDF converter - Command Line Version
# Tests filename conflict resolution

import os
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict
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
    print(f"⚠️ Missing library: {e}")
    print("This test will still check filename generation logic")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_converter_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PDFToMarkdownConverterCLI:
    def __init__(self):
        self.supported_formats = ['.pdf']
        self.output_format = '.md'
        self.used_filenames = set()  # Track used filenames to prevent conflicts
        
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

def test_filename_generation():
    """Test the filename generation with various scenarios"""
    
    print("🧪 Testing Filename Generation (FIXED VERSION)")
    print("=" * 60)
    
    converter = PDFToMarkdownConverterCLI()
    output_dir = "/tmp/test_output"
    
    # Test cases that previously caused conflicts
    test_cases = [
        # Case 1: Similar pneumology docs (caused "Neumo" conflicts)
        {
            'original': 'Neumonia_comunitaria_2021.pdf',
            'metadata': {'title': 'Guía Neumonía', 'entity': 'Sociedad Colombiana', 'year': '2021'}
        },
        {
            'original': 'Neumonia_nosocomial_2020.pdf', 
            'metadata': {'title': 'Guía Neumonía', 'entity': 'Sociedad Colombiana', 'year': '2020'}
        },
        {
            'original': 'Neumonia_pediatrica.pdf',
            'metadata': {'title': 'Guía Neumonía', 'entity': 'Sociedad Colombiana', 'year': '2019'}
        },
        
        # Case 2: Ministry of Health docs (caused "Ministerio de Salud" conflicts)
        {
            'original': 'Hipertension_arterial.pdf',
            'metadata': {'title': 'Guía Hipertensión', 'entity': 'Ministerio de Salud y Protección Social', 'year': '2023'}
        },
        {
            'original': 'Diabetes_mellitus.pdf',
            'metadata': {'title': 'Guía Diabetes', 'entity': 'Ministerio de Salud y Protección Social', 'year': '2023'}
        },
        {
            'original': 'Obesidad_adultos.pdf',
            'metadata': {'title': 'Guía Obesidad', 'entity': 'Ministerio de Salud y Protección Social', 'year': '2022'}
        },
        
        # Case 3: Identical metadata (worst case)
        {
            'original': 'document1.pdf',
            'metadata': {'title': 'Guía Clínica', 'entity': 'Sociedad Médica', 'year': '2023'}
        },
        {
            'original': 'document2.pdf',
            'metadata': {'title': 'Guía Clínica', 'entity': 'Sociedad Médica', 'year': '2023'}
        },
        {
            'original': 'document3.pdf',
            'metadata': {'title': 'Guía Clínica', 'entity': 'Sociedad Médica', 'year': '2023'}
        }
    ]
    
    generated_filenames = []
    
    for i, test_case in enumerate(test_cases, 1):
        original = test_case['original']
        metadata = test_case['metadata']
        
        filename = converter.generate_filename(metadata, original, output_dir)
        generated_filenames.append(filename)
        
        print(f"{i:2d}. Original: {original}")
        print(f"    Metadata: {metadata}")
        print(f"    Generated: {filename}.md")
        print()
    
    # Check for any duplicates
    print("🔍 Checking for Conflicts:")
    print("-" * 30)
    
    unique_filenames = set(generated_filenames)
    
    if len(unique_filenames) == len(generated_filenames):
        print("✅ SUCCESS: All filenames are unique!")
        print(f"   Generated {len(generated_filenames)} unique filenames")
    else:
        print("❌ FAILURE: Found duplicate filenames!")
        duplicates = len(generated_filenames) - len(unique_filenames)
        print(f"   Found {duplicates} duplicates")
        
        # Show duplicates
        from collections import Counter
        counts = Counter(generated_filenames)
        for filename, count in counts.items():
            if count > 1:
                print(f"   DUPLICATE: {filename} (appears {count} times)")
    
    print()
    print("📋 All Generated Filenames:")
    print("-" * 30)
    for i, filename in enumerate(generated_filenames, 1):
        print(f"{i:2d}. {filename}.md")

def test_with_existing_files():
    """Test conflict resolution when files already exist"""
    
    print("\n🗂️ Testing with Existing Files")
    print("=" * 40)
    
    # Create test directory and files
    test_dir = "/tmp/test_existing"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create some existing files
    existing_files = [
        "Neumonia_comunitaria_Guia_Neumo_Sociedad_Colombiana_2021.md",
        "Hipertension_arterial_Guia_Hiper_Ministerio_de_Salud_2023.md"
    ]
    
    for filename in existing_files:
        with open(os.path.join(test_dir, filename), 'w') as f:
            f.write("test content")
    
    print(f"Created existing files: {existing_files}")
    
    converter = PDFToMarkdownConverterCLI()
    
    # Test with same metadata that would create conflicts
    test_metadata = {
        'title': 'Guía Neumonía',
        'entity': 'Sociedad Colombiana', 
        'year': '2021'
    }
    
    new_filename = converter.generate_filename(test_metadata, "Neumonia_comunitaria.pdf", test_dir)
    print(f"New filename (should avoid conflict): {new_filename}.md")
    
    # Clean up
    import shutil
    shutil.rmtree(test_dir)
    print("✅ Test completed and cleaned up")

if __name__ == "__main__":
    test_filename_generation()
    test_with_existing_files()
    
    print("\n" + "=" * 60)
    print("🎯 CONCLUSION: The FIXED version should prevent all conflicts!")
    print("   - Original filename always included")
    print("   - Auto-numbering for conflicts") 
    print("   - Tracks used names in session")
    print("   - Checks existing files")
    print("=" * 60)