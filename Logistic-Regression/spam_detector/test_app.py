import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from model_utils import load_artifacts, predict_spam

model_dir = os.path.join(os.path.dirname(__file__), "model")
coeff_, intercept_, vectorizer, idf_map, stop_words_set, metrics, error = load_artifacts(model_dir)

assert error is None, f"Load error: {error}"
assert coeff_ is not None
assert intercept_ is not None
assert idf_map is not None

print("Model loaded successfully!")
print("Num features:", len(coeff_))
print("IDF map size:", len(idf_map))
print("Intercept:", intercept_)
print("Metrics:", metrics)

# Test Spam Message
spam_msg = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim."
res_spam = predict_spam(spam_msg, coeff_, intercept_, vectorizer, idf_map, stop_words_set, metrics["threshold"])
print("\n[Spam Test]")
print("Text:", spam_msg[:50], "...")
print("Is Spam:", res_spam["is_spam"])
print("Prob:", res_spam["probability"])
print("Word Contributions (Top 3):", res_spam["word_contributions"][:3])
assert res_spam["is_spam"] == True

# Test Ham Message
ham_msg = "Ok lar... Joking wif u oni..."
res_ham = predict_spam(ham_msg, coeff_, intercept_, vectorizer, idf_map, stop_words_set, metrics["threshold"])
print("\n[Ham Test]")
print("Text:", ham_msg[:50], "...")
print("Is Spam:", res_ham["is_spam"])
print("Prob:", res_ham["probability"])
assert res_ham["is_spam"] == False

print("\nALL PREDICTION TESTS PASSED PERFECTLY WITH TF-IDF INFERENCE!")
