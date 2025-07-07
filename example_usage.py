#!/usr/bin/env python3
"""
Example usage of the PDF to Markdown Converter
Shows both programmatic and GUI usage
"""

import os
import sys
from pathlib import Path

# Make sure we can import our converter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def example_programmatic_usage():
    """Example of using the converter programmatically"""
    print("Example: Programmatic Usage")
    print("-" * 30)
    
    try:
        from pdf_to_markdown_converter import PDFToMarkdownConverter
        
        # Initialize the converter
        converter = PDFToMarkdownConverter()
        
        # Example paths (adjust these to your actual paths)
        input_directory = "/path/to/your/pdf/files"
        output_directory = "/path/to/output/markdown/files"
        
        print(f"Input directory: {input_directory}")
        print(f"Output directory: {output_directory}")
        
        # Check if directories exist (for demo purposes)
        if not os.path.exists(input_directory):
            print("⚠️  Input directory doesn't exist - this is just an example")
            print("   Replace with your actual PDF directory path")
            return
        
        # Convert all PDFs in the directory
        print("Starting conversion...")
        
        def progress_callback(current, total, filename):
            print(f"Processing {current}/{total}: {Path(filename).name}")
        
        results = converter.convert_batch(
            input_directory,
            output_directory,
            progress_callback=progress_callback
        )
        
        print(f"\nConversion completed!")
        print(f"Total files: {results['total']}")
        print(f"Successful: {results['success']}")
        print(f"Failed: {results['failed']}")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure to install dependencies first: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error: {e}")

def example_single_file_conversion():
    """Example of converting a single PDF file"""
    print("\nExample: Single File Conversion")
    print("-" * 30)
    
    try:
        from pdf_to_markdown_converter import PDFToMarkdownConverter
        
        # Initialize the converter
        converter = PDFToMarkdownConverter()
        
        # Example file paths
        pdf_file = "/path/to/your/clinical_guideline.pdf"
        output_dir = "/path/to/output/directory"
        
        print(f"PDF file: {pdf_file}")
        print(f"Output directory: {output_dir}")
        
        if not os.path.exists(pdf_file):
            print("⚠️  PDF file doesn't exist - this is just an example")
            print("   Replace with your actual PDF file path")
            return
        
        # Convert the single file
        success = converter.convert_pdf_to_markdown(pdf_file, output_dir)
        
        if success:
            print("✓ Conversion successful!")
        else:
            print("✗ Conversion failed!")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure to install dependencies first: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error: {e}")

def example_gui_usage():
    """Example of using the GUI"""
    print("\nExample: GUI Usage")
    print("-" * 30)
    
    try:
        from pdf_to_markdown_converter import PDFConverterGUI
        
        print("Launching GUI...")
        print("1. Select your PDF input directory")
        print("2. Select your markdown output directory")
        print("3. Click 'Convert PDFs'")
        print("4. Monitor progress in the log area")
        
        # Launch the GUI
        app = PDFConverterGUI()
        app.run()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure to install dependencies first: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error: {e}")

def example_metadata_extraction():
    """Example of extracting metadata from Colombian clinical guidelines"""
    print("\nExample: Metadata Extraction")
    print("-" * 30)
    
    try:
        from pdf_to_markdown_converter import PDFToMarkdownConverter
        
        # Initialize the converter
        converter = PDFToMarkdownConverter()
        
        # Example PDF file
        pdf_file = "/path/to/colombian_clinical_guideline.pdf"
        
        if not os.path.exists(pdf_file):
            print("⚠️  PDF file doesn't exist - this is just an example")
            print("   Replace with your actual PDF file path")
            
            # Show example metadata that would be extracted
            print("\nExample metadata that would be extracted:")
            example_metadata = {
                'title': 'Guía de Práctica Clínica para el Manejo de la Hipertensión Arterial',
                'entity': 'Sociedad Colombiana de Cardiología',
                'year': '2023',
                'author': 'Comité de Hipertensión Arterial',
                'subject': 'Hipertensión Arterial',
                'keywords': 'hipertensión, guía clínica, tratamiento'
            }
            
            for key, value in example_metadata.items():
                print(f"  {key}: {value}")
            
            # Show how filename would be generated
            filename = converter.generate_filename(example_metadata, "original.pdf")
            print(f"\nGenerated filename: {filename}.md")
            
            return
        
        # Extract metadata from actual file
        metadata = converter.extract_metadata(pdf_file)
        
        print("Extracted metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # Generate filename
        filename = converter.generate_filename(metadata, pdf_file)
        print(f"\nGenerated filename: {filename}.md")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure to install dependencies first: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main function showing all examples"""
    print("PDF to Markdown Converter - Usage Examples")
    print("=" * 50)
    
    print("\nChoose an example to run:")
    print("1. Programmatic batch conversion")
    print("2. Single file conversion")
    print("3. GUI usage")
    print("4. Metadata extraction example")
    print("5. Run all examples")
    print("0. Exit")
    
    try:
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == "1":
            example_programmatic_usage()
        elif choice == "2":
            example_single_file_conversion()
        elif choice == "3":
            example_gui_usage()
        elif choice == "4":
            example_metadata_extraction()
        elif choice == "5":
            example_programmatic_usage()
            example_single_file_conversion()
            example_metadata_extraction()
            print("\nTo run the GUI, use: python pdf_to_markdown_converter.py")
        elif choice == "0":
            print("Goodbye!")
            return
        else:
            print("Invalid choice. Please try again.")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()