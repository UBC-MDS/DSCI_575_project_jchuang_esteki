import pandas as pd
import pickle
import string
from typing import List

def load_data(parquet_path: str) -> pd.DataFrame:
    """Load prepared data from parquet file."""
    return pd.read_parquet(parquet_path)

def load_corpus(pkl_path: str) -> List[str]:
    """Load corpus from pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_corpus(corpus: List[str], pkl_path: str):
    """Save corpus to pickle file."""
    with open(pkl_path, 'wb') as f:
        pickle.dump(corpus, f)

def preprocess_text(text: str) -> str:
    """Apply text preprocessing: lowercase, remove punctuation."""
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return ' '.join(text.split())

def tokenize(text: str) -> List[str]:
    """Tokenize text by whitespace. Used for BM25."""
    return preprocess_text(text).split()
