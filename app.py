"""
FastAPI Backend for AI Chatbot/Text Classifier
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pickle
import os
import re
from typing import List, Optional

app = FastAPI(title="AI Chatbot Classifier API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
tokenizer = None
label_encoder = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    keywords: List[str]
    all_predictions: Optional[dict] = None

def load_model():
    """Load the trained model, tokenizer, and label encoder."""
    global model, tokenizer, label_encoder
    
    model_path = './model'
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found! Please run 'python train_model.py' first."
        )
    
    print("Loading model...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load label encoder
    with open(os.path.join(model_path, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Model loaded successfully! Can classify {len(label_encoder.classes_)} intents.")

def extract_keywords(text: str, max_keywords: int = 3) -> List[str]:
    """Extract important keywords from text."""
    # Simple keyword extraction: remove stopwords and get meaningful words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how'}
    
    # Tokenize and filter
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Return top keywords
    return keywords[:max_keywords]

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("The API will still run, but predictions will fail until model is trained.")

@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse('static/index.html')

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict intent for given text."""
    if model is None or tokenizer is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class_id].item()
    
    # Decode prediction
    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
    
    # Get all predictions
    all_probs = probabilities[0].cpu().numpy()
    all_predictions = {
        label_encoder.inverse_transform([i])[0]: float(prob)
        for i, prob in enumerate(all_probs)
    }
    all_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
    
    # Extract keywords
    keywords = extract_keywords(text)
    
    return PredictionResponse(
        prediction=predicted_label,
        confidence=round(confidence, 4),
        keywords=keywords,
        all_predictions=all_predictions
    )

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

