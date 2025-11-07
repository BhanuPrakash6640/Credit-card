#!/bin/bash

# Training script for fraud detection model
# Usage: ./scripts/train.sh

echo "🚀 Starting Fraud Detection Model Training..."
echo "=============================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Check if data file exists
if [ ! -f "creditcard.csv" ]; then
    echo "❌ Error: creditcard.csv not found!"
    echo "Please download the dataset from:"
    echo "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt

# Run training
echo "🎯 Training model..."
python src/train.py

if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📊 Model saved to: models/rf_fraud_model.joblib"
    echo "📈 Metrics saved to: assets/"
else
    echo "❌ Training failed!"
    exit 1
fi
