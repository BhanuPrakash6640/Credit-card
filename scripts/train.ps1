# Training script for fraud detection model (Windows)
# Usage: .\scripts\train.ps1

Write-Host "🚀 Starting Fraud Detection Model Training..." -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green

# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    & venv\Scripts\Activate.ps1
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
}

# Check if data file exists
if (-Not (Test-Path "creditcard.csv")) {
    Write-Host "❌ Error: creditcard.csv not found!" -ForegroundColor Red
    Write-Host "Please download the dataset from:" -ForegroundColor Yellow
    Write-Host "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" -ForegroundColor Yellow
    exit 1
}

# Install dependencies if needed
Write-Host "📦 Checking dependencies..." -ForegroundColor Cyan
pip install -q -r requirements.txt

# Run training
Write-Host "🎯 Training model..." -ForegroundColor Cyan
python src\train.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Training completed successfully!" -ForegroundColor Green
    Write-Host "📊 Model saved to: models\rf_fraud_model.joblib" -ForegroundColor Green
    Write-Host "📈 Metrics saved to: assets\" -ForegroundColor Green
} else {
    Write-Host "❌ Training failed!" -ForegroundColor Red
    exit 1
}
