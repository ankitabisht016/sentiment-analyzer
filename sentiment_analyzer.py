# ============================================================
# Sentiment Analyzer — IMDB Movie Reviews
# Tools: Python | NLTK | TF-IDF | Logistic Regression
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ────────────────────────────────────────────────
# 1. LOAD DATASET
# ────────────────────────────────────────────────
print("=" * 60)
print("  SENTIMENT ANALYZER — IMDB MOVIE REVIEWS")
print("=" * 60)

df = pd.read_csv('imdb_reviews.csv')
df['label'] = df['sentiment'].map({1: 'positive', 0: 'negative'})

print(f"\n✅ Dataset loaded: {df.shape[0]:,} reviews")
print(f"\n📊 Class Distribution:")
vc = df['label'].value_counts()
for label, count in vc.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"   {label:<12} {count:>6,}  ({pct:.1f}%)  {bar}")

# ────────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 1: TEXT PREPROCESSING (NLTK)")
print("─" * 60)

STOPWORDS = set(stopwords.words('english'))
# Keep negation words — they're important for sentiment!
KEEP_WORDS = {'not', 'no', 'never', 'neither', 'nor', 'none', "don't", "won't", "can't", "wasn't", "didn't"}
STOPWORDS -= KEEP_WORDS

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove special characters, keep letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords (keep negations)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

print("\n  Preprocessing steps applied:")
print("  ✅ Lowercase conversion")
print("  ✅ HTML tag removal")
print("  ✅ Special character removal")
print("  ✅ Tokenization (NLTK word_tokenize)")
print("  ✅ Stop-word removal (negations preserved)")

# Show before/after example
sample_raw = df['review'].iloc[0]
sample_clean = preprocess(sample_raw)
print(f"\n  Example — Before: {sample_raw[:80]}...")
print(f"  Example — After : {sample_clean[:80]}...")

print("\n  Processing 25,000 reviews...")
df['clean_review'] = df['review'].apply(preprocess)
print("  ✅ All reviews preprocessed.")

# ────────────────────────────────────────────────
# 3. TF-IDF VECTORIZATION
# ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 2: TF-IDF VECTORIZATION")
print("─" * 60)

X = df['clean_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),       # unigrams + bigrams
    min_df=2,
    max_df=0.95,
    sublinear_tf=True         # log normalization
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"\n  Vocabulary size  : {len(tfidf.vocabulary_):,} terms")
print(f"  N-gram range     : (1, 2) — unigrams + bigrams")
print(f"  Training matrix  : {X_train_tfidf.shape[0]:,} reviews × {X_train_tfidf.shape[1]:,} features")
print(f"  Test matrix      : {X_test_tfidf.shape[0]:,} reviews × {X_test_tfidf.shape[1]:,} features")
print(f"  Sparsity         : {100*(1 - X_train_tfidf.nnz/(X_train_tfidf.shape[0]*X_train_tfidf.shape[1])):.1f}%")

# ────────────────────────────────────────────────
# 4. TRAIN LOGISTIC REGRESSION
# ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 3: MODEL TRAINING — LOGISTIC REGRESSION")
print("─" * 60)

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
    random_state=42
)
model.fit(X_train_tfidf, y_train)
print("  ✅ Model trained.")

# Cross-validation
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print(f"\n  5-Fold Cross-Validation Accuracy:")
for i, s in enumerate(cv_scores, 1):
    print(f"    Fold {i}: {s:.4f}")
print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ────────────────────────────────────────────────
# 5. EVALUATION
# ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 4: MODEL EVALUATION")
print("─" * 60)

y_pred = model.predict(X_test_tfidf)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print(f"\n  {'Metric':<25} {'Value':>10}")
print("  " + "─" * 37)
print(f"  {'Accuracy':<25} {acc:>10.4f}  ({acc*100:.2f}%)")
print(f"  {'Precision':<25} {prec:>10.4f}")
print(f"  {'Recall':<25} {rec:>10.4f}")
print(f"  {'F1-Score':<25} {f1:>10.4f}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

if acc >= 0.88:
    print(f"  🎉 Target achieved! Accuracy = {acc*100:.2f}% ≥ 88%")

# Top positive / negative words
print("\n  📌 Top 10 Most Positive Words (by coefficient):")
feature_names = np.array(tfidf.get_feature_names_out())
coefs = model.coef_[0]
top_pos_idx = coefs.argsort()[-10:][::-1]
top_neg_idx = coefs.argsort()[:10]
for idx in top_pos_idx:
    print(f"     {feature_names[idx]:<25} +{coefs[idx]:.4f}")

print("\n  📌 Top 10 Most Negative Words (by coefficient):")
for idx in top_neg_idx:
    print(f"     {feature_names[idx]:<25} {coefs[idx]:.4f}")

# ────────────────────────────────────────────────
# 6. VISUALIZATIONS
# ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 5: GENERATING VISUALIZATIONS")
print("─" * 60)

plt.style.use('seaborn-v0_8-whitegrid')
BLUE  = '#2563EB'
GREEN = '#16A34A'
RED   = '#DC2626'
ORANGE= '#D97706'
GRAY  = '#6B7280'

# ── Fig 1: EDA Dashboard ──────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
fig1.suptitle('Sentiment Analyzer — EDA Dashboard', fontsize=16, fontweight='bold')

# (a) Class distribution
ax = axes[0]
counts = df['label'].value_counts()
bars = ax.bar(counts.index, counts.values, color=[GREEN, RED], edgecolor='white', width=0.5)
ax.set_title('Class Distribution', fontweight='bold')
ax.set_ylabel('Number of Reviews')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{val:,}', ha='center', fontweight='bold')
ax.set_ylim(0, max(counts.values) * 1.15)

# (b) Review length distribution
df['review_length'] = df['review'].apply(lambda x: len(x.split()))
ax = axes[1]
for label, color in [('positive', GREEN), ('negative', RED)]:
    subset = df[df['label'] == label]['review_length']
    ax.hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
ax.set_title('Review Length Distribution', fontweight='bold')
ax.set_xlabel('Word Count')
ax.set_ylabel('Frequency')
ax.legend()

# (c) Top words bar chart
ax = axes[2]
top_n = 10
pos_words = [feature_names[i] for i in top_pos_idx[:top_n]]
neg_words = [feature_names[i] for i in top_neg_idx[:top_n]]
all_words = pos_words[:5] + neg_words[:5]
all_coefs = [coefs[i] for i in top_pos_idx[:5]] + [coefs[i] for i in top_neg_idx[:5]]
colors = [GREEN]*5 + [RED]*5
y_pos = range(len(all_words))
ax.barh(list(y_pos), all_coefs, color=colors, edgecolor='white')
ax.set_yticks(list(y_pos))
ax.set_yticklabels(all_words, fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Top Sentiment Words', fontweight='bold')
ax.set_xlabel('TF-IDF Coefficient')

plt.tight_layout()
plt.savefig('/home/claude/sentiment/eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ EDA Dashboard saved.")

# ── Fig 2: Model Results ──────────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('Logistic Regression — Model Results', fontsize=16, fontweight='bold')

# (a) Confusion Matrix
ax = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            linewidths=1, annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title('Confusion Matrix', fontweight='bold')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('Actual Label')

# (b) Metrics bar chart
ax = axes[1]
metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
bars = ax.bar(metrics.keys(), metrics.values(), color=[BLUE, GREEN, ORANGE, RED], edgecolor='white', width=0.5)
ax.set_title('Evaluation Metrics', fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontweight='bold', fontsize=11)
ax.axhline(0.88, color=GRAY, linestyle='--', linewidth=1, label='88% target')
ax.legend()

# (c) CV scores
ax = axes[2]
folds = [f'Fold {i}' for i in range(1, 6)]
bars = ax.bar(folds, cv_scores, color=BLUE, edgecolor='white', width=0.5, alpha=0.85)
ax.axhline(cv_scores.mean(), color=RED, linestyle='--', linewidth=2, label=f'Mean = {cv_scores.mean():.4f}')
ax.set_title('5-Fold Cross-Validation', fontweight='bold')
ax.set_ylabel('Accuracy')
ax.set_ylim(min(cv_scores) - 0.02, 1.0)
for bar, val in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.legend()

metrics_text = f"Accuracy: {acc:.4f} ({acc*100:.2f}%)  |  Precision: {prec:.4f}  |  Recall: {rec:.4f}  |  F1: {f1:.4f}"
fig2.text(0.5, -0.02, metrics_text, ha='center', fontsize=10,
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#EFF6FF', edgecolor=BLUE),
          family='monospace')

plt.tight_layout()
plt.savefig('/home/claude/sentiment/model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Model Results Dashboard saved.")

# ────────────────────────────────────────────────
# 7. LIVE PREDICTIONS
# ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 6: SAMPLE PREDICTIONS")
print("─" * 60)

test_reviews = [
    "This movie was absolutely amazing! Brilliant acting and a gripping storyline.",
    "Terrible film. Boring plot and awful acting. Complete waste of time.",
    "A decent movie, not the best but enjoyable enough for a single watch.",
    "Masterpiece! One of the greatest films ever made. Highly recommend.",
    "Dull, predictable and painfully slow. I almost fell asleep.",
]

print(f"\n  {'Review':<55} {'Prediction':>12} {'Confidence':>12}")
print("  " + "─" * 82)
for review in test_reviews:
    cleaned = preprocess(review)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = max(prob) * 100
    label = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
    print(f"  {review[:53]:<55} {label:>12}  {confidence:>9.1f}%")

print("\n" + "=" * 60)
print(f"  ✅ PROJECT COMPLETE")
print(f"  Accuracy: {acc*100:.2f}%  |  F1: {f1:.4f}  |  Precision: {prec:.4f}  |  Recall: {rec:.4f}")
print("=" * 60)
