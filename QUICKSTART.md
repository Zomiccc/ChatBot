# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train_model.py
```
This will:
- Load the dataset from `data/intents.csv`
- Train a DistilBERT model
- Save the model to `model/` directory
- Takes about 5-10 minutes depending on your hardware

### Step 3: Run the Application
```bash
python app.py
```
Or use the convenience scripts:
- Windows: `run.bat`
- PowerShell: `.\run.ps1`

Then open your browser to: **http://localhost:8000**

## 🧪 Test the API

You can test the API directly using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "What is the weather like today?"}'
```

## 📝 Example Queries

Try these in the chatbot:
- "What's the weather like today?"
- "Tell me a joke"
- "Book a flight to Paris"
- "What time is it?"
- "Thank you for your help"
- "What can you do?"

## 🔧 Troubleshooting

**Model not found error:**
- Make sure you've run `python train_model.py` first
- Check that `model/` directory exists with model files

**Port already in use:**
- Change the port in `app.py` (line 161): `uvicorn.run(app, host="0.0.0.0", port=8001)`

**Import errors:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Consider using a virtual environment

## 📊 Model Performance

The model is trained on 10 intent categories:
- weather_query
- entertainment
- booking
- time_query
- greeting
- farewell
- gratitude
- identity
- technical_support
- general_knowledge
- food_order
- media_request

You can add more training data to `data/intents.csv` and retrain for better accuracy!

