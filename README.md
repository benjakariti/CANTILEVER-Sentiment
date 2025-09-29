This project is a Sentiment Analysis system built on the IMDB movie reviews dataset.
It compares simple baseline machine learning models with a lightweight neural network to explore how text preprocessing and modeling choices affect performance.


Project Overview
The goal of this project is to classify IMDB movie reviews as positive or negative.
It was designed as a learning project to practice the full machine learning workflow:

Data gathering + EDA

Used TensorFlow’s built-in IMDB dataset.

Explored class balance, review length, and example samples.

Preprocessing

Tokenized text, removed stopwords, applied TF-IDF for classical models.

Used sequence padding and embeddings for neural networks.

Baseline models

Logistic Regression → Accuracy ~ 0.88

Support Vector Machine (SVM) → Accuracy ~ 0.86

Improved model

Built a simple Neural Network with embeddings → Accuracy ~ 0.90+

Saving models & reproducibility

Baselines saved with joblib.

Neural Network saved in .h5 and TensorFlow SavedModel formats.

Tokenizer saved as JSON for consistent preprocessing.

Prediction function

Added a small script (src/predict.py) to load trained models and run predictions on new text reviews.


Project Structure
CANTILEVER-Sentiment/
│
├── data/                #  raw datasets
├── notebooks/           # Jupyter notebooks (EDA, baselines, NN training)
│   ├── imdb_eda.ipynb
│   ├── imdb_baseline.ipynb
│   └── imdb_nn.ipynb
│
├── models/              # Saved models and tokenizer
│   ├── imdb_nn.h5
│   ├── tokenizer.json
│   └── sklearn/
│       ├── log_reg.joblib
│       └── svm.joblib
│
├── src/                 # Python scripts for inference
│   └── predict.py
│
├── requirements.txt     # Python dependencies
└── README.md            # This file



How to Run
Clone the repo

git clone git@github.com:benjakariti/CANTILEVER-Sentiment.git
cd CANTILEVER-Sentiment


Set up environment

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Run Jupyter notebooks

jupyter lab


Explore notebooks/ for data analysis and training code.


Run inference


python src/predict.py "This movie was absolutely fantastic!"



Results
Logistic Regression: 0.88 accuracy

SVM: 0.86 accuracy

Simple Neural Network: 0.90+ accuracy



Next Steps
Experiment with pre-trained embeddings (GloVe / Word2Vec).

Explore LSTM or Transformers for improved performance.

Wrap inference into a simple API or web app for demo purposes.



Author
Benjamin Kariti
Learning project in Machine Learning + NLP, running on Ubuntu VM with Python 3.12

Benjamin Kariti
Learning project in Machine Learning + NLP, running on Ubuntu VM with Python 3.12.
