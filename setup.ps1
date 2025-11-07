# Quick Setup Script (Windows)
# This script sets up the fraud detection environment

Write-Host "🛡️ Fraud Detection AI - Setup Script" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python 3\.(8|9|10|11|12)") {
    Write-Host "✓ Python version OK: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python 3.8+ required. Current: $pythonVersion" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (-Not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Install requirements
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow
Write-Host "(This may take a few minutes)" -ForegroundColor Gray
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Error installing dependencies" -ForegroundColor Red
    exit 1
}

# Create necessary directories
Write-Host ""
Write-Host "Setting up directories..." -ForegroundColor Yellow
$directories = @("models", "assets", "logs")
foreach ($dir in $directories) {
    if (-Not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
}
Write-Host "✓ Directories ready" -ForegroundColor Green

# Check for dataset
Write-Host ""
Write-Host "Checking for dataset..." -ForegroundColor Yellow
if (Test-Path "creditcard.csv") {
    Write-Host "✓ Dataset found: creditcard.csv" -ForegroundColor Green
    $trainNow = Read-Host "Would you like to train the model now? (y/N)"
    if ($trainNow -eq "y" -or $trainNow -eq "Y") {
        Write-Host ""
        Write-Host "Training model..." -ForegroundColor Cyan
        python src\train.py
    }
} else {
    Write-Host "⚠ Dataset not found: creditcard.csv" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To train the model, download the dataset from:" -ForegroundColor Yellow
    Write-Host "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "You can still run the app with sample data!" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run the app:     .\scripts\run.ps1" -ForegroundColor White
Write-Host "2. Train model:     .\scripts\train.ps1" -ForegroundColor White
Write-Host "3. Use Docker:      docker-compose up" -ForegroundColor White
Write-Host ""
Write-Host "Documentation:" -ForegroundColor Yellow
Write-Host "- README.md" -ForegroundColor White
Write-Host "- docs/deployment.md" -ForegroundColor White
Write-Host "- docs/architecture.md" -ForegroundColor White
Write-Host ""
Write-Host "Happy fraud detecting! 🚀" -ForegroundColor Green
