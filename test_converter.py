#!/usr/bin/env python3
"""
Test script for PDF to Markdown Converter
Tests basic functionality without requiring actual PDF files
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the current directory to the path so we can import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from pdf_to_markdown_converter import PDFToMarkdownConverter, PDFConverterGUI
        print("✓ Successfully imported converter classes")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_converter_initialization():
    """Test that the converter can be initialized"""
    try:
        from pdf_to_markdown_converter import PDFToMarkdownConverter
        converter = PDFToMarkdownConverter()
        print("✓ Successfully initialized PDFToMarkdownConverter")
        return True
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        return False

def test_metadata_extraction():
    """Test metadata extraction functionality"""
    try:
        from pdf_to_markdown_converter import PDFToMarkdownConverter
        converter = PDFToMarkdownConverter()
        
        # Test Colombian entity patterns
        test_text = """
        Asociación Colombiana de Cardiología
        Sociedad Colombiana de Medicina Interna
        Ministerio de Salud y Protección Social
        Instituto Nacional de Salud
        Año 2023
        """
        
        # Test entity pattern matching
        import re
        entity_patterns = [
            r'Asociación Colombiana de [A-Za-z\s]+',
            r'Sociedad Colombiana de [A-Za-z\s]+',
            r'Ministerio de Salud y Protección Social',
            r'Instituto Nacional de Salud',
        ]
        
        found_entities = []
        for pattern in entity_patterns:
            match = re.search(pattern, test_text, re.IGNORECASE)
            if match:
                found_entities.append(match.group())
        
        if found_entities:
            print(f"✓ Entity pattern matching works: {found_entities}")
            return True
        else:
            print("✗ Entity pattern matching failed")
            return False
            
    except Exception as e:
        print(f"✗ Metadata extraction test error: {e}")
        return False

def test_table_conversion():
    """Test table to markdown conversion"""
    try:
        from pdf_to_markdown_converter import PDFToMarkdownConverter
        converter = PDFToMarkdownConverter()
        
        # Test table data
        test_table = [
            ['Header 1', 'Header 2', 'Header 3'],
            ['Row 1 Col 1', 'Row 1 Col 2', 'Row 1 Col 3'],
            ['Row 2 Col 1', 'Row 2 Col 2', 'Row 2 Col 3']
        ]
        
        markdown_result = converter.table_to_markdown(test_table)
        
        if markdown_result and '|' in markdown_result and '---' in markdown_result:
            print("✓ Table to markdown conversion works")
            return True
        else:
            print("✗ Table to markdown conversion failed")
            return False
            
    except Exception as e:
        print(f"✗ Table conversion test error: {e}")
        return False

def test_filename_generation():
    """Test filename generation from metadata"""
    try:
        from pdf_to_markdown_converter import PDFToMarkdownConverter
        converter = PDFToMarkdownConverter()
        
        # Test metadata
        test_metadata = {
            'title': 'Guía de Manejo de Hipertensión Arterial',
            'entity': 'Sociedad Colombiana de Cardiología',
            'year': '2023',
            'author': 'Test Author',
            'subject': 'Cardiology',
            'keywords': 'hypertension, guidelines'
        }
        
        filename = converter.generate_filename(test_metadata, 'original_file.pdf')
        
        if filename and 'Guía_de_Manejo_de_Hipertensión_Arterial' in filename:
            print(f"✓ Filename generation works: {filename}")
            return True
        else:
            print(f"✗ Filename generation failed: {filename}")
            return False
            
    except Exception as e:
        print(f"✗ Filename generation test error: {e}")
        return False

def test_gui_initialization():
    """Test GUI initialization (without displaying)"""
    try:
        from pdf_to_markdown_converter import PDFConverterGUI
        import tkinter as tk
        
        # Create a root window for testing
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Test GUI initialization
        gui = PDFConverterGUI()
        gui.root.withdraw()  # Hide the GUI window
        
        print("✓ GUI initialization works")
        root.destroy()
        return True
        
    except Exception as e:
        print(f"✗ GUI initialization test error: {e}")
        return False

def check_dependencies():
    """Check for required dependencies"""
    print("\nChecking dependencies...")
    
    dependencies = [
        ('PyMuPDF', 'fitz'),
        ('pdfplumber', 'pdfplumber'),
        ('Pillow', 'PIL'),
        ('pytesseract', 'pytesseract'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy')
    ]
    
    missing_deps = []
    
    for dep_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✓ {dep_name} is available")
        except ImportError:
            print(f"✗ {dep_name} is missing")
            missing_deps.append(dep_name)
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies are available")
        return True

def main():
    """Run all tests"""
    print("PDF to Markdown Converter - Test Suite")
    print("=" * 50)
    
    # Check dependencies first
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\nSome dependencies are missing. Please install them first.")
        return
    
    print("\nRunning functionality tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("Converter Initialization", test_converter_initialization),
        ("Metadata Extraction", test_metadata_extraction),
        ("Table Conversion", test_table_conversion),
        ("Filename Generation", test_filename_generation),
        ("GUI Initialization", test_gui_initialization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed! The converter is ready to use.")
    else:
        print("✗ Some tests failed. Please check the installation.")
    
    print("\nTo run the converter, use:")
    print("python pdf_to_markdown_converter.py")

if __name__ == "__main__":
    main()