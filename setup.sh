#!/bin/bash

# Setup script for DNA Methylation Age Prediction Project

set -e  # Exit on error

echo "=========================================="
echo "DNA Methylation Age Prediction Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Install development dependencies (optional)
read -p "Install development dependencies (testing, linting)? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
    echo "✓ Development dependencies installed"
fi
echo ""

# Run tests (optional)
if command -v pytest &> /dev/null; then
    read -p "Run tests to verify installation? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running tests..."
        pytest --tb=short
        echo "✓ Tests completed"
    fi
fi
echo ""

# Summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Train models:"
echo "     python scripts/train.py"
echo ""
echo "  3. Run applications:"
echo "     python app.py"
echo ""
echo "  4. Run tests:"
echo "     pytest"
echo ""
echo "See IMPROVEMENTS.md for detailed documentation"
echo "=========================================="
