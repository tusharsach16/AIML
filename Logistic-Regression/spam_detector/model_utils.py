import os
import re
import pickle
import numpy as np
import scipy.sparse as sp

def sigmoid(z):
    """Numerically stable sigmoid function."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def load_artifacts(model_dir):
    """
    Loads model artifacts from the specified directory.
    Returns (coeff, intercept, vectorizer, idf_map, stopwords, metrics, error_msg).
    """
    artifacts = {}
    required_files = {
        "coefficients": "coefficients.pkl",
        "intercept": "intercept.pkl",
        "vectorizer": "vectorizer.pkl",
        "idf_map": "idf.pkl",
        "stopwords": "stopwords.pkl",
        "metrics": "metrics.pkl"
    }
    
    for key, filename in required_files.items():
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath):
            return None, None, None, None, None, None, f"Missing required model artifact: {filename} at {filepath}"
        
        try:
            with open(filepath, "rb") as f:
                artifacts[key] = pickle.load(f)
        except Exception as e:
            return None, None, None, None, None, None, f"Error loading {filename}: {str(e)}"
            
    return (
        artifacts["coefficients"],
        artifacts["intercept"],
        artifacts["vectorizer"],
        artifacts["idf_map"],
        set(artifacts["stopwords"]),
        artifacts["metrics"],
        None
    )

def preprocess_text(text, stop_words_set):
    r"""
    Preprocesses raw text following the exact training pipeline:
    1. Lowercase
    2. Remove punctuation [^\w\s]
    3. Split into words
    4. Remove stopwords
    5. Calculate Term Frequency (TF) dict
    """
    if not text or not text.strip():
        return {}, []

    # 1. Lowercase & Punctuation removal
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    
    # 2. Split words
    words = clean_text.split()
    
    # 3. Filter stopwords
    filtered_words = [w for w in words if w not in stop_words_set]
    
    if not filtered_words:
        return {}, []
        
    # 4. Calculate word frequency
    freq = {}
    for word in filtered_words:
        freq[word] = freq.get(word, 0) + 1
        
    # 5. Calculate TF
    words_count = len(filtered_words)
    tf_dict = {word: count / words_count for word, count in freq.items()}
    
    return tf_dict, filtered_words

def predict_spam(text, coeff_, intercept_, vectorizer, idf_map, stop_words_set, threshold=0.157):
    """
    Predicts spam probability for a given raw SMS text using TF-IDF features.
    
    Pipeline:
    1. Compute TF dictionary & clean words.
    2. Multiply TF by saved training IDF values -> TF-IDF dictionary.
    3. Map TF-IDF values using saved DictVectorizer to 9454-dim feature vector X_new.
    4. Compute logit z = X_new @ coeff_ + intercept_.
    5. Compute probability = sigmoid(z).
    6. Directional Impact = tfidf_value * model_weight.
    7. Thresholding: prob >= threshold -> SPAM, else HAM.
    """
    tf_dict, filtered_words = preprocess_text(text, stop_words_set)
    
    if not tf_dict:
        z = intercept_
        prob = float(sigmoid(z))
        return {
            "text": text,
            "is_spam": False,
            "probability": prob,
            "spam_percentage": round(prob * 100, 2),
            "threshold": threshold,
            "z_score": float(z),
            "clean_words": filtered_words,
            "word_contributions": [],
            "in_vocab_words_count": 0,
            "is_empty": True
        }
    
    # Calculate TF-IDF dict using saved training IDF values
    tfidf_dict = {}
    for word, tf_val in tf_dict.items():
        if word in idf_map:
            tfidf_dict[word] = tf_val * idf_map[word]
            
    # Transform into 9454-dim sparse vector using saved DictVectorizer
    X_new = vectorizer.transform([tfidf_dict])
    
    # Calculate logit z = X_new @ coeff_ + intercept_
    z = float((X_new @ coeff_)[0] + intercept_)
    prob = float(sigmoid(z))
    is_spam = prob >= threshold
    
    # Calculate per-word contributions using TF-IDF * model_weight
    vocab_map = vectorizer.vocabulary_  # word -> col index
    word_contributions = []
    
    for word, tf_val in tf_dict.items():
        if word in vocab_map and word in idf_map:
            idx = vocab_map[word]
            idf_val = float(idf_map[word])
            tfidf_val = float(tfidf_dict[word])
            weight = float(coeff_[idx])
            directional_impact = tfidf_val * weight
            
            word_contributions.append({
                "word": word,
                "tf": round(tf_val, 4),
                "idf": round(idf_val, 4),
                "tfidf": round(tfidf_val, 4),
                "weight": round(weight, 4),
                "impact": round(directional_impact, 4),
                "is_spam_indicator": weight > 0
            })
            
    # Sort contributions by absolute directional impact
    word_contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
    
    return {
        "text": text,
        "is_spam": is_spam,
        "probability": prob,
        "spam_percentage": round(prob * 100, 2),
        "threshold": threshold,
        "z_score": round(z, 4),
        "clean_words": filtered_words,
        "word_contributions": word_contributions,
        "in_vocab_words_count": len(word_contributions),
        "is_empty": False
    }
