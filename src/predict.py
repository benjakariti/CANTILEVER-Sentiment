from pathlib import Path
import argparse
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "imdb_nn.h5"
TOKENIZER_PATH = BASE / "models"/ "tokenizer.json"
MAX_LEN = 200

def load_resources():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "r") as f:
        Tokenizer = tokenizer_from_json(f.read())
        return model, Tokenizer
        

def predict_texts(texts, model, Tokenizer, max_len=MAX_LEN):
    seq = Tokenizer.texts_to_sequences(texts)
    pad = pad_sequences(seq, maxlen = MAX_LEN, padding ="post")
    probs = model.predict(pad).ravel()
    labels = (probs >= 0.5).astype(int)
    return probs.tolist(), labels.tolist()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("Texts", nargs="+", help="Text(s) to analyze (wrap multi-word examples in quotes)")
    args = parser.parse_args()

    model, Tokenizer = load_resources()
    probs, labels = predict_texts(args.Texts, model, Tokenizer)
    for t,p,l in zip(args.Texts, probs, labels):
        print(f"{l} ({p: .3f}) : {t}")
    

if __name__ == "__main__":
    main()

 




    

