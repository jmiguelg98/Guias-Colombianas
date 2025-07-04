#!/bin/bash
# Setup script for macOS to install system dependencies for MegaParse

echo "🔧 Setting up MegaParse dependencies for macOS..."
echo "================================================="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew is not installed. Please install it first from https://brew.sh/"
    exit 1
fi

echo "✓ Homebrew found"

# Install system dependencies
echo "📦 Installing system dependencies..."

# Install poppler (for PDF processing)
echo "Installing poppler..."
brew install poppler

# Install tesseract (for OCR)
echo "Installing tesseract..."
brew install tesseract

# Install libmagic (required for Mac)
echo "Installing libmagic..."
brew install libmagic

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Set your OpenAI API key as an environment variable:"
echo "   export OPENAI_API_KEY='your-api-key-here'"
echo "2. Run the converter:"
echo "   python3 pdf_to_markdown_converter.py"
echo ""
echo "💡 Tip: Add the API key to your ~/.zshrc or ~/.bashrc to make it permanent"