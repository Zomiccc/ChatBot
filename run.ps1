# PowerShell script to run the AI Chatbot Classifier

Write-Host "AI Chatbot Classifier - Starting..." -ForegroundColor Cyan

# Check if model exists
if (-not (Test-Path "model\pytorch_model.bin")) {
    Write-Host "Model not found! Training model first..." -ForegroundColor Yellow
    python train_model.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Training failed! Please check the error messages." -ForegroundColor Red
        exit 1
    }
}

# Start the server
Write-Host "Starting FastAPI server..." -ForegroundColor Green
Write-Host "Open http://localhost:8000 in your browser" -ForegroundColor Green
python app.py

