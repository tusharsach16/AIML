import os
import re
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)

# 1. Load Dataset
data_path = os.path.join(os.path.dirname(__file__), "..", "SMSSpamCollection")
if not os.path.exists(data_path):
    data_path = "SMSSpamCollection"

df = pd.read_csv(data_path, sep="\t", header=None)
df.columns = ['label', 'message']

# 2. Text Cleaning & Normalization (Matching exact notebook cells)
df["clean_text"] = df["message"].str.lower().str.replace(r'[^\w\s]', '', regex=True)
df["words_array"] = df["clean_text"].str.split()

stop_words = set(stopwords.words('english'))
df["words_array_without_stop"] = df["words_array"].apply(
    lambda words: [w for w in words if w.lower() not in stop_words]
)

# 3. Calculate Word Frequency & TF per document
def get_word_frequency(words):
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return freq

df["word_frequency"] = df["words_array_without_stop"].apply(get_word_frequency)
df["words_count"] = df["words_array_without_stop"].apply(len)

def cal_tf(word_frequency, words_count):
    if words_count == 0:
        return {}
    return {word: count / words_count for word, count in word_frequency.items()}

df["tf"] = df.apply(
    lambda row: cal_tf(row["word_frequency"], row["words_count"]),
    axis=1
)

# 4. Calculate Corpus IDF map
N = len(df)
doc_freq = {}
for words_list in df["words_array_without_stop"]:
    unique_words = set(words_list)
    for word in unique_words:
        doc_freq[word] = doc_freq.get(word, 0) + 1

idf_map = {word: float(np.log(N / count)) for word, count in doc_freq.items()}

# 5. Exact DictVectorizer on df["tf"] as trained in notebook
v = DictVectorizer(sparse=True)
X = v.fit_transform(df["tf"])
y = df["label"].map({'ham': 0, 'spam': 1}).values

print(f"Feature Matrix Shape: {X.shape} (N={X.shape[0]}, Features={X.shape[1]})")

# 6. Stratified Train / Test Split (Exact seed 42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Logistic Regression via Gradient Descent (Matching notebook gd function)
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def gd(X, y):
    bias = np.ones((X.shape[0], 1))
    X_with_bias = sp.hstack([bias, X])
    
    weights = np.zeros(X_with_bias.shape[1])
    lr = 0.5
    
    for i in range(2500):
        y_hat = sigmoid(X_with_bias @ weights)
        gradient = (X_with_bias.T @ (y - y_hat)) / X_with_bias.shape[0]
        weights = weights + lr * gradient
        
    return weights[1:], weights[0]

print("Executing original notebook Gradient Descent model training...")
coeff_, intercept_ = gd(X_train, y_train)

print(f"Trained coefficients shape: {coeff_.shape}")
print(f"Intercept: {intercept_}")

# Verification of known target words from notebook
print("\n--- Model Coefficients Verification ---")
vocab_map = v.vocabulary_
for word in ["free", "prize", "call", "cash", "txt"]:
    if word in vocab_map:
        idx = vocab_map[word]
        print(f"Word: '{word}' -> Index: {idx} -> Coefficient: {coeff_[idx]:.10f}")

# 8. Save Artifacts to spam_detector/model/
model_dir = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "coefficients.pkl"), "wb") as f:
    pickle.dump(coeff_, f)

with open(os.path.join(model_dir, "intercept.pkl"), "wb") as f:
    pickle.dump(intercept_, f)

with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
    pickle.dump(v, f)

with open(os.path.join(model_dir, "idf.pkl"), "wb") as f:
    pickle.dump(idf_map, f)

with open(os.path.join(model_dir, "stopwords.pkl"), "wb") as f:
    pickle.dump(list(stop_words), f)

# 9. Compute Test Set Predictions & Metrics using TF-IDF inference
# Calculate TF-IDF matrix for X_test
test_tfidf_dicts = []
vocab_feature_names = v.feature_names_

# Compute test predictions with TF-IDF vector: tfidf_val = tf_val * idf_val
z_list = []
for i in range(X_test.shape[0]):
    row = X_test[i]
    # row is TF vector
    # Multiply non-zero TF entries by idf
    indices = row.indices
    data = row.data
    row_z = intercept_
    for col_idx, tf_val in zip(indices, data):
        word = vocab_feature_names[col_idx]
        idf_val = idf_map.get(word, 0.0)
        tfidf_val = tf_val * idf_val
        row_z += tfidf_val * coeff_[col_idx]
    z_list.append(row_z)

z_arr = np.array(z_list)
y_prob = sigmoid(z_arr)
y_pred = (y_prob >= 0.157).astype(int)

from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

metrics = {
    "num_features": X.shape[1],
    "threshold": 0.157,
    "roc_auc": round(float(auc), 3),
    "f1_score": round(float(f1), 2),
    "recall": round(float(rec), 2),
    "accuracy": round(float(acc), 3)
}

with open(os.path.join(model_dir, "metrics.pkl"), "wb") as f:
    pickle.dump(metrics, f)

print("\nArtifacts saved successfully!")
print("Metrics:", metrics)
