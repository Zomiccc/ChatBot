@echo off
echo AI Chatbot Classifier - Starting...

REM Check if model exists
if not exist "model\pytorch_model.bin" (
    echo Model not found! Training model first...
    python train_model.py
    if errorlevel 1 (
        echo Training failed! Please check the error messages.
        exit /b 1
    )
)

REM Start the server
echo Starting FastAPI server...
echo Open http://localhost:8000 in your browser
python app.py

