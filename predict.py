"""
Standalone inference script for the Restaurant Review Sentiment Analysis project.

Usage:
    python predict.py "The food was absolutely amazing, will come back!"
    python predict.py "Service was slow and the pasta was cold." --model "Linear SVM"

Loads the saved TF-IDF vectorizer + trained models from models/ and returns a
sentiment prediction. No `input()` calls, so this works in scripts, tests, or
as the basis of a small API endpoint.
"""
import argparse
import json
import re
import sys

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))
PROTECTED_WORDS = {
    "not", "no", "never", "bad", "good", "great", "worst", "best",
    "n't", "nothing", "none", "nor", "without", "hardly", "barely",
    "love", "loved", "hate", "hated", "amazing", "terrible", "horrible",
    "delicious", "disgusting", "awful", "excellent", "poor", "rude",
    "friendly", "slow", "fast", "fresh", "stale", "recommend",
}
for w in PROTECTED_WORDS:
    STOP_WORDS.discard(w)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def load_artifacts(models_dir="models"):
    tfidf = joblib.load(f"{models_dir}/tfidf_vectorizer.joblib")
    models = {
        "Logistic Regression": joblib.load(f"{models_dir}/logistic_regression.joblib"),
        "Naive Bayes": joblib.load(f"{models_dir}/naive_bayes.joblib"),
        "Linear SVM": joblib.load(f"{models_dir}/linear_svm.joblib"),
    }
    return tfidf, models


def predict(text: str, model_name: str = "Logistic Regression", models_dir="models") -> dict:
    """Predict sentiment for a review using one of the classical (TF-IDF based)
    models: 'Logistic Regression', 'Naive Bayes', or 'Linear SVM'.

    For the Keras models (Dense NN / Embedding NN), load them directly with
    tensorflow.keras.models.load_model() as shown in the notebook — kept out
    of this lightweight script to avoid a hard TensorFlow dependency here.
    """
    tfidf, models = load_artifacts(models_dir)
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(models)}")

    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    prob = float(models[model_name].predict_proba(vec)[0, 1])
    label = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if label == "Positive" else 1 - prob
    return {"review": text, "model_used": model_name, "label": label, "confidence": round(confidence, 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict restaurant review sentiment.")
    parser.add_argument("review", type=str, help="Review text to classify")
    parser.add_argument("--model", type=str, default="Logistic Regression",
                         choices=["Logistic Regression", "Naive Bayes", "Linear SVM"])
    parser.add_argument("--models-dir", type=str, default="models")
    args = parser.parse_args()

    result = predict(args.review, model_name=args.model, models_dir=args.models_dir)
    print(json.dumps(result, indent=2))
