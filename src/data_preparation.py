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
