# Restaurant Review Sentiment Analysis

A sentiment classification project that predicts whether a restaurant review is
positive or negative, built with TF-IDF features and a feed-forward neural network
in Keras.

📄 [Project report (PDF)](reports/Restaurant_Review_Project_Report.pdf) ·
📓 [Notebook](Restaurant_Rev.ipynb)

## Results

| Metric | Training Set | Test Set |
|---|---|---|
| Accuracy | 97.78% | 77.00% |
| Loss (Binary Cross-Entropy) | 0.071 | 0.655 |

**Test set classification report:**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| 0 (Negative) | 0.81 | 0.80 | 0.81 |
| 1 (Positive) | 0.71 | 0.72 | 0.72 |

See the [full report](reports/Restaurant_Review_Project_Report.pdf) for the learning
curve, confusion matrix, and a discussion of the results.

## Project structure

```
.
├── Restaurant_Rev.ipynb                  # Full notebook: preprocessing, model, training, evaluation
├── Restaurant_Reviews.csv                # Dataset (1,000 labeled restaurant reviews)
├── reports/
│   └── Restaurant_Review_Project_Report.pdf
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Dataset

1,000 restaurant reviews labeled `1` (liked) or `0` (not liked), balanced 500/500.

## Methodology

1. **Text preprocessing** — lowercase, strip non-alphabetic characters, tokenize,
   remove stopwords (keeping sentiment-relevant negation words: "not", "no", "bad",
   "never")
2. **Feature extraction** — TF-IDF vectorization (scikit-learn `TfidfVectorizer`)
3. **Train/test split** — 90/10 split (`random_state=40`)
4. **Model** — Keras Sequential network: Dense(1000, relu) → Dropout(0.2) →
   Dense(500, relu) → Dropout(0.5) → Dense(1, sigmoid), compiled with Adam and
   binary cross-entropy
5. **Training** — two stages with `EarlyStopping`, monitoring validation loss
6. **Evaluation** — accuracy, classification report, and confusion matrix on the
   held-out test set

## Setup

```bash
git clone https://github.com/<your-username>/restaurant-review-sentiment-analysis.git
cd restaurant-review-sentiment-analysis
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> On Google Colab, `tensorflow`/`keras` are preinstalled — you can skip installing
> them and just run the notebook directly.

## Usage

Open and run `Restaurant_Rev.ipynb` top to bottom (Jupyter or Google Colab). The
notebook reads `Restaurant_Reviews.csv`, trains the model, evaluates it, and ends
with an interactive `predict_sentiment()` function you can use to test your own
review text.

> Note: this repo does not include a pre-trained model file. Running the notebook
> will save one locally (`SentimentModel.keras`) — add it here yourself if you'd
> like to version it, or keep it out of the repo (see `.gitignore`).

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Kapil Mali** — Senior Data Analyst
