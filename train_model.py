"""
Model Training Script
Trains a DistilBERT-based text classifier for intent classification.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset
import torch

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class IntentDataset(Dataset):
    """Custom dataset for intent classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} samples")
    print(f"Labels: {df['label'].unique()}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    return df

def train_model():
    """Main training function."""
    # Load data
    data_path = 'data/intents.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = load_data(data_path)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(df['label'])
    num_labels = len(label_encoder.classes_)
    
    print(f"\nNumber of unique labels: {num_labels}")
    print(f"Label classes: {label_encoder.classes_}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values,
        labels_encoded,
        test_size=0.2,
        random_state=42,
        stratify=labels_encoded
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize tokenizer and model
    model_name = 'distilbert-base-uncased'
    print(f"\nLoading tokenizer and model: {model_name}")
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    model.to(device)
    
    # Create datasets
    train_dataset = IntentDataset(X_train, y_train, tokenizer)
    test_dataset = IntentDataset(X_test, y_test, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./model',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating on test set...")
    eval_results = trainer.evaluate()
    print(f"Test accuracy: {eval_results.get('eval_loss', 'N/A')}")
    
    # Save model and tokenizer
    print("\nSaving model...")
    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')
    
    # Save label encoder
    import pickle
    with open('./model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("\n✅ Training complete! Model saved to ./model/")
    print(f"Model can classify {num_labels} different intents:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label}")

if __name__ == '__main__':
    train_model()

