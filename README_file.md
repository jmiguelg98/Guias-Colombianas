# PDF to Markdown Converter for Colombian Clinical Guidelines

A comprehensive Python script that converts PDF documents (both normal and scanned) to LLM-friendly markdown files. Specifically designed for Colombian clinical guidelines with metadata extraction and intelligent file naming.

## Features

- **Universal PDF Support**: Handles both normal text PDFs and scanned documents
- **OCR Capability**: Uses Tesseract OCR for scanned documents and images
- **Table Extraction**: Converts PDF tables to markdown format
- **Image Processing**: Extracts text from images and figures using OCR
- **Metadata Extraction**: Automatically extracts Colombian medical entities, years, and document information
- **Intelligent Naming**: Names output files based on extracted metadata (title, entity, year)
- **Batch Processing**: Converts multiple PDFs at once
- **User-Friendly GUI**: Easy-to-use interface for selecting input and output directories
- **Cross-Platform**: Works on macOS, Linux, and Windows

## Requirements

### System Dependencies

**Tesseract OCR** (Required for scanned documents and image processing):

- **macOS**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr tesseract-ocr-spa`
- **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

### Python Dependencies

- Python 3.7 or higher
- See `requirements.txt` for specific package versions

## Installation

1. **Clone or download the script files**:
   ```bash
   # Download the files to your desired directory
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR** (see system dependencies above)

4. **Verify installation**:
   ```bash
   python pdf_to_markdown_converter.py
   ```

## Usage

### GUI Mode (Recommended)

1. **Run the script**:
   ```bash
   python pdf_to_markdown_converter.py
   ```

2. **Select directories**:
   - Choose the input directory containing your PDF files
   - Choose the output directory for markdown files

3. **Start conversion**:
   - Click "Convert PDFs" to begin the process
   - Monitor progress in the log area

### Command Line Mode

For advanced users, you can import and use the converter programmatically:

```python
from pdf_to_markdown_converter import PDFToMarkdownConverter

converter = PDFToMarkdownConverter()
results = converter.convert_batch("/path/to/pdfs", "/path/to/output")
print(f"Converted {results['success']} files successfully")
```

## Output Format

The converter creates markdown files with the following structure:

```markdown
# [Document Title]

## Document Information

**Original Filename:** original_file.pdf
**Entity:** [Extracted Colombian Medical Entity]
**Year:** [Publication Year]
**Author:** [Author Information]
**Subject:** [Subject/Keywords]

---

## Document Content

### Main Text Content
[Extracted text with page markers]

## Tables
[Converted tables in markdown format]

## Figures and Images
[OCR-extracted text from images]

---
*Document processed on [timestamp]*
```

## File Naming Convention

Output files are named using the following pattern:
```
[Title]_[Entity]_[Year].md
```

For example:
- `Guía_Manejo_Hipertensión_Arterial_Sociedad_Colombiana_de_Cardiología_2023.md`
- `Consenso_Diabetes_Mellitus_Ministerio_de_Salud_2022.md`

## Supported Colombian Medical Entities

The script automatically recognizes and extracts the following Colombian medical entities:

- Asociación Colombiana de [Specialty]
- Sociedad Colombiana de [Specialty]
- Ministerio de Salud y Protección Social
- Instituto Nacional de Salud
- IETS (Instituto de Evaluación Tecnológica en Salud)
- Colciencias
- Academia Nacional de Medicina
- Federación Médica Colombiana
- ASCOFAME

## Troubleshooting

### Common Issues

1. **"Tesseract not found" error**:
   - Ensure Tesseract is installed and in your system PATH
   - Try reinstalling Tesseract

2. **Poor OCR quality**:
   - Ensure your PDFs are high-quality scans
   - Consider preprocessing scanned documents

3. **Memory issues with large PDFs**:
   - Process files individually for very large documents
   - Increase system memory if possible

4. **Missing dependencies**:
   - Run `pip install -r requirements.txt` again
   - Check for any installation errors

### Log Files

The script creates a log file (`pdf_converter.log`) that contains detailed information about the conversion process. Check this file for troubleshooting.

## Limitations

- Very large PDFs (>100MB) may take significant time to process
- OCR quality depends on the original document quality
- Some complex table layouts may not convert perfectly
- Metadata extraction works best with standard Colombian clinical guideline formats

## Contributing

Feel free to improve this script by:
- Adding support for additional medical entities
- Improving OCR accuracy
- Enhancing table extraction
- Adding new output formats

## License

This script is provided as-is for educational and research purposes. Please ensure you have appropriate rights to convert and process your PDF documents.