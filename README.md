# Restaurant Review Sentiment Analysis

End-to-end NLP project that classifies restaurant reviews as positive or negative,
benchmarking **3 classical ML models** against **2 Keras deep learning models** —
with a data-leakage fix, negation-aware preprocessing, EDA-style synthetic data
augmentation, and full error analysis.

**Best result:** Logistic Regression (TF-IDF) — **78.5% accuracy**, **0.787 F1**,
**0.842 ROC-AUC** on a fully held-out, 100% real test set.

📄 [Full project report (PDF)](reports/Restaurant_Sentiment_Analysis_Report.pdf) ·
📓 [Notebook](Restaurant_Review_Sentiment_Analysis.ipynb)

---

## Why this project is more than a tutorial notebook

A first pass at this dataset (1,000 short restaurant reviews) is a common
beginner project. What makes this version worth putting on a resume:

- **Found and fixed a real data-leakage bug** — the initial version fit TF-IDF on
  the *entire* dataset before splitting into train/test, inflating results.
- **Negation-aware preprocessing** — naive stopword removal strips words like
  "not" and "never" that flip sentiment; this pipeline protects them.
- **Synthetic data augmentation done honestly** — EDA-style augmentation
  (Wei & Zou, 2019) expands the *training* set only; validation and test sets
  stay 100% real so every reported metric is trustworthy.
- **5-way model benchmark**, not just one model — 3 classical ML baselines and
  2 Keras architectures (TF-IDF dense network, learned-embedding network),
  all evaluated on the identical held-out test set.
- **Fixed severe overfitting** — the original neural net had a 23.8-point
  train/test accuracy gap; regularization + correct early-stopping closed
  most of that gap.
- **Qualitative error analysis** — not just a metric, but a breakdown of *what
  kind* of review the best model still gets wrong and why.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.785** | 0.776 | 0.797 | **0.787** | 0.842 |
| Dense NN (TF-IDF) | 0.758 | 0.716 | 0.851 | 0.778 | 0.839 |
| Linear SVM | 0.765 | 0.753 | 0.784 | 0.768 | 0.838 |
| Naive Bayes | 0.745 | 0.714 | 0.811 | 0.759 | 0.820 |
| Embedding NN (Sequence) | 0.752 | 0.768 | 0.716 | 0.741 | **0.853** |

*All models evaluated on the same 149-review, fully real, held-out test set.*

See [`reports/Restaurant_Sentiment_Analysis_Report.pdf`](reports/Restaurant_Sentiment_Analysis_Report.pdf)
for the full write-up: methodology, ROC curves, confusion matrices, and error analysis.

## Project structure

```
.
├── Restaurant_Review_Sentiment_Analysis.ipynb   # Full pipeline, executed end-to-end
├── predict.py                                   # Standalone CLI inference script
├── requirements.txt
├── LICENSE
├── data/
│   └── Restaurant_Reviews.csv                   # Original 1,000-review dataset
├── models/                                      # Saved vectorizer + trained models
│   ├── tfidf_vectorizer.joblib
│   ├── logistic_regression.joblib
│   ├── naive_bayes.joblib
│   ├── linear_svm.joblib
│   ├── dense_nn_tfidf.keras
│   ├── embedding_nn.keras
│   ├── keras_tokenizer.joblib
│   └── model_config.json
├── figures/                                     # All charts used in the report
└── reports/
    ├── Restaurant_Sentiment_Analysis_Report.pdf
    ├── model_comparison.csv
    └── misclassified_examples.csv
```

## Setup

```bash
git clone https://github.com/<your-username>/restaurant-review-sentiment-analysis.git
cd restaurant-review-sentiment-analysis
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> On Google Colab, replace `tensorflow-cpu` in `requirements.txt` with `tensorflow`
> (Colab ships a GPU build already) — or just skip installing TensorFlow, it's
> preinstalled there.

## Usage

**Run the full notebook** (reproduces every model, figure, and metric from scratch):

```bash
jupyter notebook Restaurant_Review_Sentiment_Analysis.ipynb
```

**Or use the trained models directly** without re-running training:

```bash
python predict.py "The food was absolutely amazing, will come back!"
python predict.py "Service was slow and the pasta was cold." --model "Linear SVM"
```

```json
{
  "review": "The food was absolutely amazing, will come back!",
  "model_used": "Logistic Regression",
  "label": "Positive",
  "confidence": 0.9585
}
```

## Methodology summary

1. Load data, drop duplicates, exploratory analysis (class balance, review length)
2. **Stratified 70/15/15 train/val/test split on real data**, before any augmentation
3. Negation-aware text cleaning (lowercase → strip non-alpha → tokenize → protected
   stopword removal → lemmatize)
4. **EDA-style synthetic augmentation** (synonym replacement, random swap, random
   deletion) applied only to the training split — 695 real reviews → 2,569 training rows
5. TF-IDF vectorization (1-2 grams, 5,000 features), fit only on training data
6. Train + tune (5-fold CV grid search) Logistic Regression, Naive Bayes, Linear SVM
7. Train two Keras models: a regularized dense network on TF-IDF vectors, and an
   embedding + global-average-pooling network on tokenized sequences — both with
   `EarlyStopping(monitor='val_loss', restore_best_weights=True)`
8. Evaluate all 5 models on the identical, untouched, real test set
9. Error analysis on the best model's misclassifications
10. Save all artifacts and expose a clean, non-interactive inference function

## Future work

- Fine-tune a transformer (e.g. DistilBERT) to address implicit-sentiment and
  mixed-clause review failures identified in the error analysis
- Collect more real labeled reviews to reduce reliance on synthetic augmentation
- Per-model decision-threshold tuning via validation precision-recall curves
- Aspect-based sentiment analysis (separate food quality from service quality)

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Kapil Mali** — Senior Data Analyst
