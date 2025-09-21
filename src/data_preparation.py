import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import os

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text

def anonymize_data(df, text_column):
    df[text_column] = df[text_column].apply(
        lambda x: anonymizer.anonymize(
            text=x,
            analyzer_results=analyzer.analyze(text=x, entities=["PERSON", "LOCATION", "PHONE_NUMBER", "EMAIL_ADDRESS"], language="en")
        ).text
    )
    return df
def prepare_datasets():
    os.makedirs("../data", exist_ok=True)
    
    # GoEmotions for emotion
    go_emotions = load_dataset("go_emotions", split="train")
    df_go = pd.DataFrame(go_emotions)
    df_go['text'] = df_go['text'].apply(clean_text)
    df_go = anonymize_data(df_go, 'text')
    train_go, temp_go = train_test_split(df_go, test_size=0.2, random_state=42)
    val_go, test_go = train_test_split(temp_go, test_size=0.5, random_state=42)
    train_go.to_csv("../data/train_goemotions.csv", index=False)
    val_go.to_csv("../data/val_goemotions.csv", index=False)
    test_go.to_csv("../data/test_goemotions.csv", index=False)
    
    # TweetEval for sentiment 
    tweet_eval = load_dataset("tweet_eval", "sentiment", split="train")
    tweet_eval = tweet_eval.shuffle(seed=42).select(range(50000))
    df_sent = pd.DataFrame(tweet_eval)
    df_sent['text'] = df_sent['text'].apply(clean_text)
    df_sent = anonymize_data(df_sent, 'text')
    train_sent, temp_sent = train_test_split(df_sent, test_size=0.2, random_state=42)
    val_sent, test_sent = train_test_split(temp_sent, test_size=0.5, random_state=42)
    train_sent.to_csv("../data/train_sentiment.csv", index=False)
    val_sent.to_csv("../data/val_sentiment.csv", index=False)
    test_sent.to_csv("../data/test_sentiment.csv", index=False)
    
    print("All datasets prepared and saved in ../data/")

if __name__ == "__main__":
    prepare_datasets()