# Run script for Streamlit application (Windows)
# Usage: .\scripts\run.ps1

Write-Host "🚀 Starting Fraud Detection Dashboard..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    & venv\Scripts\Activate.ps1
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
}

# Check if model exists
if (-Not (Test-Path "models\rf_fraud_model.joblib") -and -Not (Test-Path "rf_fraud_model.joblib")) {
    Write-Host "⚠️ Warning: Model not found!" -ForegroundColor Yellow
    Write-Host "Please train the model first by running:" -ForegroundColor Yellow
    Write-Host ".\scripts\train.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Continuing anyway (you can use sample data)..." -ForegroundColor Yellow
}

# Install dependencies if needed
Write-Host "📦 Checking dependencies..." -ForegroundColor Cyan
pip install -q -r requirements.txt

# Run Streamlit app
Write-Host "🎯 Launching dashboard..." -ForegroundColor Cyan
Write-Host "📱 Open your browser at: http://localhost:8501" -ForegroundColor Green
Write-Host ""
streamlit run app\streamlit_app.py
