# AI-Powered Chatbot / Text Classifier

A full-stack AI application that uses machine learning to classify text and predict user intents. Built with FastAPI backend and Hugging Face transformers.

## Features

- 🤖 **ML-Powered Classification**: Uses DistilBERT transformer model for text classification
- 🎯 **Intent Prediction**: Classifies user input into predefined categories
- 💬 **Interactive Chat Interface**: Modern, chat-like UI for user interaction
- 📊 **Keyword Highlighting**: Shows which words influenced the prediction
- 💾 **Conversation History**: Saves chat history locally in browser
- 🚀 **Fast Inference**: Optimized model loading and prediction

## Tech Stack

- **Backend**: FastAPI (Python)
- **ML Framework**: Hugging Face Transformers (DistilBERT)
- **Frontend**: HTML + JavaScript
- **Training**: PyTorch + scikit-learn

## Project Structure

```
ai-chatbot-classifier/
├── app.py                 # FastAPI backend server
├── train_model.py         # Model training script
├── model/                 # Saved model files
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
├── data/                  # Training dataset
│   └── intents.csv
├── static/                # Frontend assets
│   ├── index.html
│   ├── script.js
│   └── style.css
├── requirements.txt
└── README.md
```

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd ai-chatbot-classifier
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model:**
   ```bash
   python train_model.py
   ```
   This will create the model files in the `model/` directory.

## Usage

1. **Start the backend server:**
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:8000`

2. **Open the frontend:**
   - Open `static/index.html` in your browser, or
   - Navigate to `http://localhost:8000` (if serving static files)

3. **Use the chatbot:**
   - Type your message in the input box
   - Press Enter or click Send
   - See the predicted intent and confidence score

## API Endpoints

### POST `/predict`
Predicts the intent/category of user text.

**Request:**
```json
{
  "text": "What's the weather like today?"
}
```

**Response:**
```json
{
  "prediction": "weather_query",
  "confidence": 0.95,
  "keywords": ["weather", "today"]
}
```

### GET `/health`
Health check endpoint.

## Model Training

The model is trained on intent classification data. You can customize the training by:

1. Modifying `data/intents.csv` with your own dataset
2. Adjusting training parameters in `train_model.py`
3. Re-running `python train_model.py`

## Dataset Format

The training data should be in CSV format:
```csv
text,label
"What's the weather?",weather_query
"Tell me a joke",entertainment
"Book a flight",booking
```

## Optional Enhancements

- ✅ Embedding-based similarity search
- ✅ Conversation history (localStorage)
- ✅ Keyword highlighting
- ✅ Modern chat UI with bubbles

## Time Estimate

- Dataset prep + training: 1–1.5h
- Backend API: 45 min
- Frontend: 1h
- Testing + README: 30–45 min

## License

MIT License - Feel free to use this project for learning and portfolio purposes.

