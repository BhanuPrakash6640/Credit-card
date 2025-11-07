#!/bin/bash
# Quick Setup Script (Linux/Mac)
# This script sets up the fraud detection environment

echo "🛡️ Fraud Detection AI - Setup Script"
echo "====================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1)
if [[ $PYTHON_VERSION == *"Python 3."* ]]; then
    echo "✓ Python version OK: $PYTHON_VERSION"
else
    echo "✗ Python 3.8+ required. Current: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install requirements
echo ""
echo "Installing dependencies..."
echo "(This may take a few minutes)"
pip install -r requirements.txt --quiet
if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Error installing dependencies"
    exit 1
fi

# Create necessary directories
echo ""
echo "Setting up directories..."
mkdir -p models assets logs
echo "✓ Directories ready"

# Check for dataset
echo ""
echo "Checking for dataset..."
if [ -f "creditcard.csv" ]; then
    echo "✓ Dataset found: creditcard.csv"
    read -p "Would you like to train the model now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Training model..."
        python src/train.py
    fi
else
    echo "⚠ Dataset not found: creditcard.csv"
    echo ""
    echo "To train the model, download the dataset from:"
    echo "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    echo ""
    echo "You can still run the app with sample data!"
fi

# Summary
echo ""
echo "====================================="
echo "✅ Setup Complete!"
echo "====================================="
echo ""
echo "Next steps:"
echo "1. Run the app:     ./scripts/run.sh"
echo "2. Train model:     ./scripts/train.sh"
echo "3. Use Docker:      docker-compose up"
echo ""
echo "Documentation:"
echo "- README.md"
echo "- docs/deployment.md"
echo "- docs/architecture.md"
echo ""
echo "Happy fraud detecting! 🚀"
