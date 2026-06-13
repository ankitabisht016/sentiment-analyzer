# 🎬 Sentiment Analyzer — IMDB Movie Reviews

**Python · NLTK · TF-IDF · Logistic Regression**

NLP-based binary text classification system to classify IMDB movie reviews as **positive** or **negative** sentiment.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **87.90%** |
| Precision | 0.8830 |
| Recall | 0.8742 |
| F1-Score | 0.8785 |
| CV Accuracy (5-fold) | 0.8802 ± 0.0055 |

---

## 🗂️ Project Structure

```
sentiment_analyzer/
├── sentiment_analyzer.py   # Full NLP pipeline
├── imdb_reviews.csv        # IMDB-style dataset (25,000 reviews)
├── eda_dashboard.png       # EDA visualizations
├── model_results.png       # Model evaluation charts
└── README.md               # This file
```

---

## 🔧 Pipeline Steps

### 1. Dataset
- **25,000 reviews** — 12,500 positive, 12,500 negative (balanced)
- Binary labels: `1 = positive`, `0 = negative`

### 2. Text Preprocessing (NLTK)
- Lowercase conversion
- HTML tag removal (`<br />` etc.)
- Special character removal (keep only letters)
- **Tokenization** — `nltk.word_tokenize`
- **Stop-word removal** — negations preserved (`not`, `never`, `don't`)

### 3. TF-IDF Vectorization
- `TfidfVectorizer` with `max_features=10,000`
- **Bigrams** included (`ngram_range=(1,2)`) — captures phrases like "not good"
- `sublinear_tf=True` — log normalization to reduce impact of frequent terms
- Result: sparse matrix of 20,000 × 2,746 features

### 4. Model — Logistic Regression
- `sklearn.linear_model.LogisticRegression` with `C=1.0`
- 5-fold cross-validation on training set
- Evaluated with Accuracy, Precision, Recall, F1-Score

### 5. Evaluation
- Confusion matrix analysis
- Classification report (per-class metrics)
- Top sentiment words by model coefficient

---

## 🚀 How to Run

```bash
# Install dependencies
pip install nltk scikit-learn pandas numpy matplotlib seaborn

# Run the pipeline
python sentiment_analyzer.py
```

---

## 💬 Sample Predictions

| Review | Prediction | Confidence |
|--------|-----------|------------|
| "Absolutely brilliant! Gripping story." | POSITIVE 😊 | 93.3% |
| "Terrible film. Complete waste of time." | NEGATIVE 😞 | 97.3% |
| "Dull and painfully slow." | NEGATIVE 😞 | 93.4% |

---

## 📈 Key Findings

- **TF-IDF bigrams** significantly improve over unigrams alone
- Preserving **negation words** (`not`, `never`) is critical for accuracy
- Top positive words: `loved`, `brilliant`, `fantastic`, `amazing`
- Top negative words: `hated`, `terrible`, `dull`, `disappointing`
- Model generalizes well — CV accuracy consistent at ~88%

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| NLTK | Tokenization & stop-word removal |
| Scikit-learn | TF-IDF, Logistic Regression, metrics |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualizations |
