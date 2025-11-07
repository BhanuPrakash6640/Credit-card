#!/bin/bash

# Run script for Streamlit application
# Usage: ./scripts/run.sh

echo "🚀 Starting Fraud Detection Dashboard..."
echo "========================================"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Check if model exists
if [ ! -f "models/rf_fraud_model.joblib" ] && [ ! -f "rf_fraud_model.joblib" ]; then
    echo "⚠️ Warning: Model not found!"
    echo "Please train the model first by running:"
    echo "./scripts/train.sh"
    echo ""
    echo "Continuing anyway (you can use sample data)..."
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt

# Run Streamlit app
echo "🎯 Launching dashboard..."
echo "📱 Open your browser at: http://localhost:8501"
echo ""
streamlit run app/streamlit_app.py
