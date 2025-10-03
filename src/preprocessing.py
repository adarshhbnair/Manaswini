import spacy
import nltk
from gensim.models import KeyedVectors
import torch
import os

nltk.download('punkt', quiet=True)
nlp = spacy.load("en_core_web_sm")

glove_path = "../../glove.6B.100d.txt"
if os.path.exists(glove_path):
    glove = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
else:
    glove = None
    print("GloVe not found. Download from https://nlp.stanford.edu/data/glove.6B.zip")
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    pos_tags = [token.pos_ for token in doc]
    filtered_tokens = [t for t in tokens if t.lower() not in nlp.Defaults.stop_words]
    
    embeddings = []
    if glove:
        for token in filtered_tokens:
            if token.lower() in glove:
                embeddings.append(glove[token.lower()])
    if embeddings:
        avg_embedding = torch.mean(torch.tensor(embeddings), dim=0)
    else:
        avg_embedding = torch.zeros(100)
    
    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "filtered_tokens": filtered_tokens,
        "embedding": avg_embedding
    }
if __name__ == "__main__":
    sample = "Hello, world! I'm testing this."
    print(preprocess_text(sample))