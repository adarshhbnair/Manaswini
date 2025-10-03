import argparse
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }

def train_emotion_model():
    """Fine-tune BERT on GoEmotions with GPU support."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv(os.path.join("..", "..", "data", "train_goemotions.csv"))
    val_df = pd.read_csv(os.path.join("..", "..", "data", "val_goemotions.csv"))
    emotion_map = {'sadness': 0, 'joy': 1, 'anger': 2, 'fear': 3, 'love': 4, 'surprise': 5}
    train_df['label'] = train_df['emotion'].map(emotion_map).fillna(0).astype(int)
    val_df['label'] = val_df['emotion'].map(emotion_map).fillna(0).astype(int)
    
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir=os.path.join("..", "..", "models", "bert_sentiment"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join("..", "..", "models", "bert_sentiment", "logs"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.save_model(os.path.join("..", "..", "models", "bert_sentiment"))
    tokenizer.save_pretrained(os.path.join("..", "..", "models", "bert_sentiment"))
    print("Emotion model trained and saved.")

def predict_sentiment_emotion(text):
    model_path = os.path.join("..", "..", "models", "bert_sentiment")
    if os.path.exists(model_path):
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        emotion_map = {0: 'sadness', 1: 'joy', 2: 'anger', 3: 'fear', 4: 'love', 5: 'surprise'}
        emotion = emotion_map[pred]
        negative_emotions = ['sadness', 'anger', 'fear']
        sentiment = 'negative' if emotion in negative_emotions else 'positive'
    else:
        sentiment_pipe = pipeline("sentiment-analysis")
        emotion_pipe = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
        sentiment_res = sentiment_pipe(text)[0]
        emotion_res = emotion_pipe(text)[0]
        sentiment = sentiment_res['label'].lower()
        emotion = emotion_res['label'].lower()
    
    return {"sentiment": sentiment, "emotion": emotion}
