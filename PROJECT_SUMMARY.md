# PDF to Markdown Converter - Project Summary

## Overview

I've created a comprehensive Python application that converts PDF documents (both normal and scanned) to LLM-friendly markdown files. The system is specifically designed for Colombian clinical guidelines with intelligent metadata extraction and file naming.

## Key Features Implemented

### 🔧 Core Functionality
- **Universal PDF Support**: Handles both regular text PDFs and scanned documents
- **OCR Integration**: Uses Tesseract OCR for extracting text from scanned documents and images
- **Table Extraction**: Automatically detects and converts PDF tables to markdown format
- **Image Processing**: Extracts text from images and figures using OCR
- **Batch Processing**: Converts multiple PDFs simultaneously with progress tracking

### 🏥 Colombian Clinical Guidelines Specialization
- **Metadata Extraction**: Automatically identifies Colombian medical entities:
  - Asociación Colombiana de [Specialty]
  - Sociedad Colombiana de [Specialty]
  - Ministerio de Salud y Protección Social
  - Instituto Nacional de Salud
  - IETS, Colciencias, Academia Nacional de Medicina, etc.
- **Intelligent Naming**: Generates filenames based on extracted metadata (title, entity, year)
- **Year Detection**: Automatically extracts publication years from document content

### 🖥️ User Interface
- **GUI Application**: Easy-to-use graphical interface with file/folder selection
- **Progress Tracking**: Real-time progress bar and logging
- **Error Handling**: Comprehensive error reporting and logging
- **Cross-Platform**: Works on macOS, Linux, and Windows

## Files Created

### 1. `pdf_to_markdown_converter.py` (22KB, 587 lines)
**Main application file** containing:
- `PDFToMarkdownConverter` class with all conversion logic
- `PDFConverterGUI` class for the graphical interface
- OCR processing, table extraction, metadata extraction
- Intelligent filename generation
- Comprehensive error handling and logging

### 2. `requirements.txt` (95B, 6 lines)
**Dependencies file** listing all required Python packages:
- PyMuPDF (PDF processing)
- pdfplumber (table extraction)
- Pillow (image processing)
- pytesseract (OCR)
- pandas & numpy (data processing)

### 3. `README.md` (5.0KB, 178 lines)
**Comprehensive documentation** including:
- Installation instructions for all platforms
- Usage examples (GUI and programmatic)
- Troubleshooting guide
- File naming conventions
- Supported Colombian medical entities
- System requirements

### 4. `install.py` (5.0KB, 166 lines)
**Automated installation script** that:
- Checks Python version compatibility
- Installs Python dependencies
- Verifies Tesseract OCR installation
- Provides platform-specific installation instructions
- Tests the installation

### 5. `test_converter.py` (7.0KB, 228 lines)
**Test suite** that verifies:
- All required dependencies are available
- Converter initialization works
- Metadata extraction functions properly
- Table conversion works correctly
- Filename generation is functional
- GUI can be initialized

### 6. `example_usage.py` (7.3KB, 218 lines)
**Usage examples** demonstrating:
- Programmatic batch conversion
- Single file conversion
- GUI usage
- Metadata extraction examples
- Interactive menu system

## Technical Implementation

### PDF Processing Pipeline
1. **Text Extraction**: Uses PyMuPDF to extract text directly from PDFs
2. **OCR Fallback**: When no text is found, uses Tesseract OCR on rendered pages
3. **Table Detection**: pdfplumber identifies and extracts table structures
4. **Image Processing**: Extracts embedded images and applies OCR
5. **Metadata Analysis**: Parses document metadata and content for Colombian entities

### Metadata Extraction Algorithm
- Searches PDF metadata fields (title, author, subject, keywords)
- Analyzes first 3 pages of content for Colombian medical entities
- Uses regex patterns to identify organizations and years
- Extracts publication years with validity checking (1990-current year)
- Generates clean, structured metadata for LLM consumption

### Output Format
Generated markdown files include:
- Document header with extracted metadata
- Main text content with page markers
- Tables converted to markdown format
- Images converted to text descriptions via OCR
- Processing timestamp and source information

## Installation & Usage

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Tesseract OCR (platform-specific)
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-spa

# 3. Run the application
python pdf_to_markdown_converter.py
```

### Automated Installation
```bash
python install.py
```

### Testing
```bash
python test_converter.py
```

## File Naming Examples

The system generates intelligent filenames based on extracted metadata:

- `Guía_Manejo_Hipertensión_Arterial_Sociedad_Colombiana_de_Cardiología_2023.md`
- `Consenso_Diabetes_Mellitus_Ministerio_de_Salud_2022.md`
- `Protocolo_Manejo_COVID19_Instituto_Nacional_de_Salud_2021.md`

## Error Handling & Logging

- Comprehensive logging to `pdf_converter.log`
- GUI displays real-time progress and errors
- Graceful handling of corrupted PDFs
- Continues processing even if individual files fail
- Detailed error reporting for troubleshooting

## Cross-Platform Compatibility

- **macOS**: Full support with Homebrew Tesseract installation
- **Linux**: Supports Ubuntu/Debian and other distributions
- **Windows**: Works with Tesseract Windows installer
- **Python 3.7+**: Compatible with modern Python versions

## Performance Considerations

- Processes PDFs in sequence to avoid memory issues
- Uses OCR only when necessary (scanned documents)
- Optimized table detection algorithms
- Progress callbacks for user feedback
- Memory-efficient image processing

## Future Enhancement Possibilities

- Additional Colombian medical entity patterns
- Advanced table layout recognition
- Multi-language OCR support
- Cloud processing integration
- API endpoint for web integration
- Advanced metadata extraction using AI/ML

## Quality Assurance

- Comprehensive test suite covering all major functions
- Error handling for edge cases
- Input validation and sanitization
- Cross-platform testing considerations
- Dependency version management

This system provides a robust, user-friendly solution for converting Colombian clinical guidelines from PDF to LLM-friendly markdown format while preserving all important content and metadata.