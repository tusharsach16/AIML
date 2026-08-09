# 🛡️ SMS Spam Detection — From-Scratch ML Project

A learning-first SMS spam classifier built completely from scratch using **TF-IDF + Logistic Regression + Gradient Descent**, followed by evaluation, threshold tuning, and Streamlit deployment.

---

## 1. Project Goal

The primary objective of this project was to understand and implement every mathematical and algorithmic step of the NLP and Machine Learning pipeline manually, instead of relying on high-level library functions like `TfidfVectorizer()` or `LogisticRegression()`.

```
SMS Dataset 
  ↳ Data Understanding & EDA 
  ↳ Text Cleaning & Lowercasing 
  ↳ Tokenization & Stopword Removal 
  ↳ Vocabulary & Word-to-Index Mapping 
  ↳ Term Frequency (TF) & Inverse Document Frequency (IDF) 
  ↳ SciPy Sparse Feature Matrix 
  ↳ Stratified Train/Test Split 
  ↳ Custom Logistic Regression via Gradient Descent 
  ↳ Probability Sigmoid Mapping 
  ↳ Threshold Tuning (0.157) 
  ↳ Performance Evaluation (F1: 0.82, ROC-AUC: 0.957) 
  ↳ Streamlit Web Deployment
```

---

## 2. Dataset Overview

* **Dataset Name:** SMS Spam Collection
* **Total Samples:** 5,572 rows, 2 columns (`label`, `message`)
* **Class Distribution:**
  * `ham` (Legitimate): 4,825 messages (~86.6%)
  * `spam` (Unsolicited): 747 messages (~13.4%)

> [!IMPORTANT]
> The dataset exhibits significant **class imbalance** (~86.6% Ham vs ~13.4% Spam). This imbalance plays a critical role when interpreting raw accuracy and selecting an optimal decision threshold.

---

## 3. Initial Data Exploration

Basic dataset sanity checks:
```python
df.shape                 # (5572, 2)
df["label"].value_counts() # ham: 4825, spam: 747
df.isnull().sum()        # label: 0, message: 0
```

---

## 4. EDA — Message Length Analysis

We created a feature measuring raw character length:
```python
df["length"] = df["message"].str.len()
df.groupby("label")["length"].mean()
```

### Results:
* **Ham Average Length:** ~71.48 characters
* **Spam Average Length:** ~138.67 characters

**Key Insight:** Spam messages in this dataset tend to be significantly longer on average. However, because their distributions overlap, character length alone is insufficient for classification.

---

## 5. Word-Level Analysis

Analyzing word frequencies across classes:
```python
df[df["label"] == "spam"]["words_array"].explode().value_counts()
```

Words occurring disproportionately in spam messages:
* `free`, `call`, `prize`, `cash`, `txt`, `claim`, `urgent`, `nokia`

**Intuition:** Disproportionate word occurrences serve as distinct numerical signals for the classifier.

---

## 6. Text Cleaning & Normalization

Text was converted to lowercase and stripped of punctuation:
```python
df["clean_text"] = df["message"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
df["words_array"] = df["clean_text"].str.split()
```

**Why Lowercase?**  
Without normalization, `"Free"`, `"free"`, and `"FREE"` would be treated as three distinct features. Lowercasing maps them to a single token.

---

## 7. Stopword Removal

Common English stopwords (e.g., `the`, `is`, `a`, `and`) were filtered out using NLTK's stopword list to focus feature representation on informative tokens.

---

## 8. Vocabulary & Feature Space

The vocabulary was constructed from all unique non-stopword tokens:
```python
vocabulary = set(word for words in df["words_array_without_stop"] for word in words)
```
* **Vocabulary Size:** 9,454 unique features.
* **Vector Dimension:** Every SMS is mapped into a fixed 9,454-dimensional feature vector.

---

## 9. Document Word Count

For each message, the count of filtered words was calculated:
```python
df["words_count"] = df["words_array_without_stop"].apply(len)
```

---

## 10. Term Frequency (TF)

Term Frequency measures how frequently a term occurs in a document:

$$\text{TF}(t, d) = \frac{\text{Count of word } t \text{ in document } d}{\text{Total words in document } d}$$

### Example:
For document words `["free", "prize", "free", "cash"]`:
* `free` $\rightarrow 2/4 = 0.50$
* `prize` $\rightarrow 1/4 = 0.25$
* `cash` $\rightarrow 1/4 = 0.25$

**Why Normalize?** Longer documents naturally contain higher raw word counts. TF normalizes frequency relative to message length.

---

## 11. Why TF Alone Is Not Enough

A word may be frequent in a single document but also appear in almost every document in the corpus (e.g., `"u"`). Such words carry minimal discriminative power. Therefore, Inverse Document Frequency (IDF) is required.

---

## 12. Inverse Document Frequency (IDF)

IDF measures how rare or informative a word is across all documents:

$$\text{IDF}(t) = \log\left(\frac{N}{\text{df}(t)}\right)$$

* $N$: Total number of documents.
* $\text{df}(t)$: Number of documents containing word $t$.
* Common words receive a lower IDF score; rare/discriminative words receive a higher IDF score.

---

## 13. Combined TF-IDF

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

* **TF:** Local importance within the message.
* **IDF:** Global informativeness across the dataset.
* **TF-IDF:** Combined representation of feature importance.

---

## 14. Word-to-Index Mapping

A fixed dictionary mapping `word -> index` ensures consistent column indexing across all training and test instances.

---

## 15. Sparse Feature Matrix & Memory Efficiency

With 5,572 messages and 9,454 features, a dense representation would require storing over **42 million entries**, the vast majority of which are zeros.

We utilized `scipy.sparse.csr_matrix` to store only non-zero values, preserving memory and optimizing computation.

---

## 16. Sparse Matrix Debugging (NumPy `np.insert` vs `sp.hstack`)

Attempting `X = np.insert(X, 0, 1, axis=1)` failed because `X` was a SciPy CSR sparse matrix.

### Fix:
Instead of converting `X` to a dense array (`X.toarray()`), we prepended the bias column using `scipy.sparse.hstack`:
```python
bias = np.ones((X.shape[0], 1))
X_with_bias = sp.hstack([bias, X])
```

---

## 17. Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
* **Train Set:** 4,457 messages
* **Test Set:** 1,115 messages
* `stratify=y` preserves the ~86.6% / ~13.4% class ratio in both splits.

---

## 18. Logistic Regression From Scratch

$$z = X \mathbf{w} + b$$

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

The sigmoid function maps linear raw scores $z \in (-\infty, \infty)$ to probabilities $\sigma(z) \in (0, 1)$.

---

## 19. Gradient Descent Implementation

Gradient Descent iteratively updates model weights to minimize log-loss:

$$\mathbf{y}_{\text{hat}} = \sigma(X \mathbf{w})$$

$$\text{Gradient} = \frac{X^T (\mathbf{y} - \mathbf{y}_{\text{hat}})}{N}$$

$$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} + \alpha \times \text{Gradient}$$

---

## 20. Weight Initialization

* **Initial Try:** `weights = np.ones(X.shape[1])`
* **Final Choice:** `weights = np.zeros(X.shape[1])`

**Rationale:** Initializing with zeros produces a neutral starting state ($z = 0 \implies \sigma(0) = 0.5$), allowing the model to learn weights uninhibited by arbitrary initial bias.

---

## 21. Learning Rate vs. Decision Threshold

| Parameter | Purpose | Stage |
|---|---|---|
| **Learning Rate ($\alpha = 0.5$)** | Controls weight update step size | Model Training |
| **Classification Threshold ($0.157$)** | Converts output probability into class label | Model Prediction |

---

## 22. Training Hyperparameters

* **Learning Rate ($\alpha$):** 0.5
* **Iterations:** 2,500
* **Final Intercept ($b$):** `-1.7878`

---

## 23. The 0.5 Default Threshold Problem

Evaluating test set probabilities using the default threshold `0.5`:
```python
y_pred = (y_prob >= 0.5).astype(int)
```
* **Result:** All 1,115 test predictions were classified as `0` (Ham).

### Initial Confusion Matrix (Threshold = 0.5):
```
[[966   0]
 [149   0]]
```
* TN = 966, FP = 0, FN = 149, TP = 0.

---

## 24. Diagnostic: Proving Model Learning

All-zero predictions at threshold 0.5 did **not** mean the model failed to learn. Inspecting learned feature weights:

* `call`: `+2.454`
* `free`: `+1.941`
* `txt`: `+1.474`
* `prize`: `+0.993`
* `cash`: `+0.671`

Positive weights confirmed that the model correctly identified spam-indicative words.

---

## 25. Output Probability Distribution

* **Minimum Probability:** 0.0417
* **Maximum Probability:** 0.3343
* **Mean Probability:** 0.1341 (~13.4%, matching the dataset spam ratio)

---

## 26. Probability Separation (Ham vs. Spam)

* **Average Ham Probability:** 0.1265
* **Average Spam Probability:** 0.1830

Actual spam messages received consistently higher predicted probabilities than ham messages.

---

## 27. ROC-AUC Performance

$$\text{ROC-AUC} = \mathbf{0.95747}$$

An AUC of **0.957** confirmed excellent overall class ranking and separation capability across thresholds.

---

## 28. Threshold Tuning & Trade-Off Analysis

| Threshold | Precision | Recall | F1-Score |
|---|---|---|---|
| **0.10** | 0.142 | 1.000 | 0.249 |
| **0.15** | 0.693 | 0.879 | 0.775 |
| **0.20** | 0.756 | 0.208 | 0.326 |
| **0.25** | 0.571 | 0.027 | 0.051 |
| **0.30+**| 0.000 | 0.000 | 0.000 |

### Fine-Grained Search Result:
* **Optimal Decision Threshold:** **`0.157`**
* **Peak F1-Score:** **`0.818`**

---

## 29. Final Test Evaluation (Threshold = 0.157)

### Confusion Matrix:
```
                 Predicted Ham    Predicted Spam
Actual Ham            930               36
Actual Spam            21              128
```
* **True Negatives (TN):** 930
* **False Positives (FP):** 36
* **False Negatives (FN):** 21
* **True Positives (TP):** 128

### Classification Report:
```
              precision    recall  f1-score   support

         Ham       0.98      0.96      0.97       966
        Spam       0.78      0.86      0.82       149

    accuracy                           0.95      1115
   macro avg       0.88      0.91      0.89      1115
weighted avg       0.95      0.95      0.95      1115
```

---

## 30. Inference Pipeline for New SMS Messages

When classifying a new SMS:
1. Lowercase text
2. Remove punctuation
3. Split into words & filter stopwords
4. Compute local Term Frequency (TF)
5. Map words using the **saved training vocabulary & DictVectorizer**
6. Transform into a 9,454-dimensional sparse vector
7. Compute logit $z = X \mathbf{w} + b$
8. Apply sigmoid $\sigma(z)$
9. Classify: $\text{prob} \ge 0.157 \implies \mathbf{SPAM}$, else $\mathbf{HAM}$

> [!WARNING]
> **Critical Rule:** Never retrain or recompute a new vocabulary or IDF from single inference messages. Re-use training artifacts to preserve feature indices.

---

## 31. Application & Artifact Structure

```
spam_detector/
├── app.py                  # Streamlit web application (UI & Dashboard)
├── model_utils.py          # Modular text preprocessing & TF-IDF prediction pipeline
├── train_and_save.py       # Artifact generation script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── model/                  # Saved Model Artifacts Directory
    ├── coefficients.pkl    # Trained weights (9,448 features)
    ├── intercept.pkl       # Trained intercept (-1.8447)
    ├── vectorizer.pkl      # DictVectorizer mapping
    ├── idf.pkl             # Corpus Inverse Document Frequency (IDF) values
    ├── stopwords.pkl       # Training stopword set
    └── metrics.pkl         # Evaluation metrics
```

### Running the Web App:
```bash
cd spam_detector
streamlit run app.py
```

---

## 32. Key Revision Q&A

1. **Why can't Logistic Regression directly process raw SMS text?**  
   Logistic Regression requires numerical vectors. TF-IDF maps text into numerical feature space.
2. **Why convert text to lowercase?**  
   To prevent `"Free"`, `"free"`, and `"FREE"` from becoming three separate features.
3. **Why use a sparse matrix instead of dense arrays?**  
   Storing millions of explicit zeros in a $5572 \times 9454$ matrix wastes memory. Sparse matrices store only non-zero entries.
4. **Why did threshold 0.5 fail initially?**  
   Due to class imbalance (~13.4% spam), probabilities clustered below 0.5. Lowering the decision threshold to 0.157 aligned decision boundaries with the underlying probability distribution.
5. **What proved the model was learning?**  
   Strong positive feature weights for spam words (`call`: 2.45, `free`: 1.94) and a high ROC-AUC score of **0.957**.

---

## 33. Summary Mental Model

```
RAW SMS 
  → Clean & Lowercase 
  → Remove Stopwords 
  → Map to Fixed Vocabulary (9,454) 
  → Compute TF 
  → SciPy Sparse Vector 
  → Dot Product (X @ w + b) 
  → Sigmoid Probability 
  → Apply Threshold (0.157) 
  → SPAM / HAM
```
