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
    
    # GoEmotions
    go_emotions = load_dataset("go_emotions", split="train")
    df_go = pd.DataFrame(go_emotions)
    df_go['text'] = df_go['text'].apply(clean_text)
    df_go = anonymize_data(df_go, 'text')
    train_go, temp_go = train_test_split(df_go, test_size=0.2, random_state=42)
    val_go, test_go = train_test_split(temp_go, test_size=0.5, random_state=42)
    train_go.to_csv(os.path.join("..", "data", "train_goemotions.csv"), index=False)
    val_go.to_csv(os.path.join("..", "data", "val_goemotions.csv"), index=False)
    test_go.to_csv(os.path.join("..", "data", "test_goemotions.csv"), index=False)
    
    #Sentiment140
    sent_140 = load_dataset("Sentiment140", split="train")
    df_sent = pd.DataFrame(sent_140)
    df_sent['text'] = df_sent['text'].apply(clean_text)
    df_sent = anonymize_data(df_sent,'text')
    train_sent,temp_sent = train_test_split(df_sent, test_size=0.2, random_state=42)
    val_sent,test_sent = train_test_split(temp_sent, test_size=0.5, random_state=42)
    train_sent.to_csv(os.path.join("..","data","train_sentiment140.csv"),index=False)
    val_sent.to_csv(os.path.join("..","data","val_sentiment140.csv"),index=False)
    test_sent.to_csv(os.path.join("..","data","test_sentiment140.csv"),index=False)
    
    # CoNLL-2003
    conll = load_dataset("conll2003", split="train[:5000]")
    df_conll = pd.DataFrame(conll)
    df_conll['text'] = df_conll['tokens'].apply(lambda x: ' '.join(x))
    df_conll['text'] = df_conll['text'].apply(clean_text)
    df_conll = anonymize_data(df_conll, 'text')
    train_conll, temp_conll = train_test_split(df_conll, test_size=0.2, random_state=42)
    val_conll, test_conll = train_test_split(temp_conll, test_size=0.5, random_state=42)
    train_conll.to_csv(os.path.join("..", "data", "train_conll.csv"), index=False)
    val_conll.to_csv(os.path.join("..", "data", "val_conll.csv"), index=False)
    test_conll.to_csv(os.path.join("..", "data", "test_conll.csv"), index=False)
    
    
    # EmpatheticDialogues
    emp_dialogues = load_dataset("empathetic_dialogues", split="train[:5000]")
    df_emp = pd.DataFrame(emp_dialogues)
    df_emp['prompt'] = df_emp['prompt'].apply(clean_text)
    df_emp['utterance'] = df_emp['utterance'].apply(clean_text)
    df_emp = anonymize_data(df_emp, 'prompt')
    df_emp = anonymize_data(df_emp, 'utterance')
    train_emp, temp_emp = train_test_split(df_emp, test_size=0.2, random_state=42)
    val_emp, test_emp = train_test_split(temp_emp, test_size=0.5, random_state=42)
    train_emp.to_csv(os.path.join("..", "data", "train_empathetic.csv"), index=False)
    val_emp.to_csv(os.path.join("..", "data", "val_empathetic.csv"), index=False)
    test_emp.to_csv(os.path.join("..", "data", "test_empathetic.csv"), index=False)
if __name__ == "__main__":
    prepare_datasets()