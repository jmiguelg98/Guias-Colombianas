# PDF to Markdown Converter

A powerful Python script that converts PDF files to markdown format while preserving tables, images, figures, and text formatting. Built using MegaParse for high-quality document conversion.

## Features

- ✅ **High-quality conversion**: Preserves text, tables, images, and figures
- ✅ **Interactive interface**: User-friendly prompts for file paths and options
- ✅ **Flexible output**: Choose your output folder or use the same folder as the PDF
- ✅ **Two conversion modes**: Standard and AI Vision (higher quality)
- ✅ **Error handling**: Comprehensive error checking and troubleshooting
- ✅ **Cross-platform**: Works on macOS, Linux, and Windows

## Requirements

### System Dependencies

**macOS:**
- poppler (for PDF processing)
- tesseract (for OCR)
- libmagic (Mac requirement)

**Linux:**
- poppler-utils
- tesseract-ocr
- libmagic

**Windows:**
- poppler (install via conda or binary)
- tesseract (install via installer)

### Python Dependencies

- Python 3.9+
- megaparse
- langchain-openai

### API Key (Optional)

For the highest quality conversion, you'll need an OpenAI API key. The script works without it but with limited features.

## Installation

### Quick Setup for macOS

1. **Clone or download the files:**
   ```bash
   # Download the main script and requirements
   curl -O https://raw.githubusercontent.com/your-repo/pdf_to_markdown_converter.py
   curl -O https://raw.githubusercontent.com/your-repo/requirements.txt
   curl -O https://raw.githubusercontent.com/your-repo/setup_macos.sh
   ```

2. **Run the setup script:**
   ```bash
   chmod +x setup_macos.sh
   ./setup_macos.sh
   ```

3. **Set your OpenAI API key (optional but recommended):**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

### Manual Installation

1. **Install system dependencies:**

   **macOS:**
   ```bash
   brew install poppler tesseract libmagic
   ```

   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt-get install poppler-utils tesseract-ocr libmagic1
   ```

   **Windows:**
   - Download and install [Poppler](https://poppler.freedesktop.org/)
   - Download and install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key (optional):**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

### Basic Usage

Simply run the script and follow the interactive prompts:

```bash
python3 pdf_to_markdown_converter.py
```

The script will guide you through:
1. Entering the path to your PDF file
2. Choosing where to save the output
3. Selecting the conversion mode

### Example Session

```
============================================================
PDF to Markdown Converter using MegaParse
============================================================

✓ MegaParse and dependencies are installed
✓ OpenAI API key found in environment

Enter the full path to your PDF file: /Users/username/Documents/my_document.pdf
✓ PDF file found: /Users/username/Documents/my_document.pdf

Where do you want to save the markdown file?
1. Same folder as the PDF file
2. Specify a different folder

Enter your choice (1 or 2): 2
Enter the full path to the output folder: /Users/username/Desktop/converted_docs
✓ Output folder ready: /Users/username/Desktop/converted_docs

Choose conversion mode:
1. Standard conversion (faster)
2. Vision conversion (higher quality, uses AI, costs API credits)

Enter your choice (1 or 2): 2

🔄 Converting PDF to markdown using Vision mode...
   Input:  /Users/username/Documents/my_document.pdf
   Output: /Users/username/Desktop/converted_docs/my_document_vision_converted_20240115_143022.md

   Processing with AI vision... This may take longer but provides higher quality.
✅ Vision conversion completed successfully!
   Markdown file saved to: /Users/username/Desktop/converted_docs/my_document_vision_converted_20240115_143022.md
```

### Conversion Modes

**Standard Mode (Free):**
- Fast conversion
- Good quality for most documents
- No API credits required

**Vision Mode (Premium):**
- Uses OpenAI GPT-4o for enhanced accuracy
- Better handling of complex layouts
- Superior table and figure extraction
- Requires OpenAI API key and costs credits

## Output

The converted markdown file will include:
- All text content properly formatted
- Tables preserved in markdown format
- Images extracted and referenced
- Headers and structure maintained
- Timestamps in filename to avoid conflicts

## Troubleshooting

### Common Issues

**"Missing dependency" error:**
```bash
pip install megaparse langchain-openai
```

**"No module named 'megaparse'" error:**
Make sure you're using the correct Python version and have installed in the right environment.

**PDF processing errors:**
- Ensure your PDF isn't corrupted
- Try with a smaller file first
- Check available disk space

**API key issues:**
- Verify your OpenAI API key is correct
- Check your API credit balance
- Ensure the key has appropriate permissions

### System-Specific Issues

**macOS:**
- If you get permission errors, try: `brew install libmagic`
- For M1/M2 Macs, ensure you're using the correct architecture

**Linux:**
- Install with: `sudo apt-get install tesseract-ocr-all` for all language packs
- For Ubuntu 20.04+, you may need: `sudo apt-get install python3-dev`

**Windows:**
- Ensure Poppler and Tesseract are in your PATH
- You may need to install Visual C++ Build Tools

## Configuration

### Environment Variables

```bash
# Required for Vision mode
export OPENAI_API_KEY='your-api-key-here'

# Optional: Override default torch device
export TORCH_DEVICE='cuda'  # or 'cpu', 'mps'
```

### API Key Setup

1. **Get an OpenAI API key:**
   - Go to https://platform.openai.com/api-keys
   - Create a new secret key
   - Copy the key (starts with 'sk-')

2. **Set the key permanently:**
   - Add to your shell profile (~/.zshrc, ~/.bashrc, or ~/.bash_profile):
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Tips for Best Results

1. **Use Vision mode** for complex documents with tables and figures
2. **Ensure good PDF quality** - scanned documents may need preprocessing
3. **Check output folder permissions** before running
4. **Start with small files** to test your setup
5. **Keep your API key secure** - never share it publicly

## License

This project uses MegaParse, which is licensed under Apache-2.0. Please review the [MegaParse license](https://github.com/QuivrHQ/MegaParse/blob/main/LICENSE) for commercial usage restrictions.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the [MegaParse documentation](https://github.com/QuivrHQ/MegaParse)
3. Ensure all dependencies are properly installed

## Changelog

### v1.0.0
- Initial release with MegaParse integration
- Interactive CLI interface
- Support for both standard and vision modes
- Cross-platform compatibility
- Comprehensive error handling