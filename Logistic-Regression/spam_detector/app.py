import os
import streamlit as st
import numpy as np
import pandas as pd
from model_utils import load_artifacts, predict_spam

# Page Config
st.set_page_config(
    page_title="SMS Spam Detector | Logistic Regression from Scratch",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium SaaS UI Aesthetic
st.markdown("""
<style>
    /* Global Styling */
    .main {
        padding-top: 1.5rem;
    }
    
    /* Header Container */
    .header-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 800px;
    }
    .badge-custom {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.8rem;
        border: 1px solid rgba(165, 180, 252, 0.3);
    }

    /* Result Card Styling */
    .result-card-spam {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.1) 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
    }
    .result-card-ham {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(4, 120, 87, 0.1) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
    }
    .result-title-spam {
        color: #f87171;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .result-title-ham {
        color: #34d399;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .result-desc {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-top: 0.6rem;
    }

    /* Stats & Metric Box */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
    }

    /* Example Card */
    .example-btn-container {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Path to model artifacts
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))

# Load Model Artifacts (Cached with updated function name to invalidate old cache)
@st.cache_resource
def get_model_artifacts(model_path):
    return load_artifacts(model_path)

artifacts_result = get_model_artifacts(MODEL_DIR)
if len(artifacts_result) == 7:
    coeff_, intercept_, vectorizer, idf_map, stop_words_set, metrics, load_error = artifacts_result
else:
    coeff_, intercept_, vectorizer, idf_map, stop_words_set, metrics = None, None, None, None, None, None
    load_error = "Cache or artifact structure mismatch. Please restart the Streamlit server or re-run train_and_save.py."

# Header Section
st.markdown("""
<div class="header-container">
    <div class="header-title">🛡️ SMS Spam Detector</div>
    <div class="header-subtitle">
        A high-performance Machine Learning model built completely from scratch in Python.
        Features custom Gradient Descent optimization, manual TF-IDF feature extraction, and scipy sparse matrix computations.
    </div>
    <div class="badge-custom">⚙️ Pure Python & Scipy | No High-Level ML Libraries</div>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ Model Artifact Error: {load_error}")
    st.info("💡 Run `python spam_detector/train_and_save.py` to generate model artifacts.")
    st.stop()

# Sidebar: Model Architecture & Evaluation Metrics
with st.sidebar:
    st.markdown("### 📊 Model Specifications")
    st.caption("Built manually with Gradient Descent & TF-IDF")
    
    st.markdown("---")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric(label="Accuracy", value=f"{metrics['accuracy']*100:.1f}%")
        st.metric(label="ROC-AUC", value=f"{metrics['roc_auc']:.3f}")
    with col_s2:
        st.metric(label="Spam F1-Score", value=f"{metrics['f1_score']:.2f}")
        st.metric(label="Spam Recall", value=f"{metrics['recall']:.2f}")

    st.markdown("---")
    st.markdown("**🔧 Technical Details**")
    st.markdown(f"- **Algorithm:** Logistic Regression")
    st.markdown(f"- **Optimization:** Gradient Descent (2,500 iter, lr=0.5)")
    st.markdown(f"- **Feature Representation:** TF-IDF (`TF * saved IDF`) ")
    st.markdown(f"- **Vocabulary Size:** {metrics['num_features']:,} features")
    st.markdown(f"- **Decision Threshold:** `{metrics['threshold']}`")
    st.markdown(f"- **Activation:** Sigmoid function")
    
    st.markdown("---")
    
    # Debug Coefficient Verification Section
    with st.expander("🛠️ Debug: Coefficient Verification", expanded=False):
        st.caption("Saved Notebook Model Coefficients:")
        debug_words = ["free", "prize", "call", "cash", "txt"]
        vocab_map = vectorizer.vocabulary_
        debug_data = []
        for w in debug_words:
            if w in vocab_map:
                idx = vocab_map[w]
                w_coeff = coeff_[idx]
                w_idf = idf_map.get(w, 0.0)
                debug_data.append({
                    "Word": w,
                    "Index": idx,
                    "Coefficient": round(w_coeff, 6),
                    "Saved IDF": round(w_idf, 4)
                })
        st.dataframe(pd.DataFrame(debug_data), use_container_width=True, hide_index=True)
        st.caption(f"Intercept: `{intercept_:.6f}`")

    st.caption("🚀 Portfolio Project | Modern AI/ML Pipeline")

# Main Content Area
col_main, col_examples = st.columns([2, 1], gap="large")

# Predefined Examples
SPAM_EXAMPLES = [
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim.",
    "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv",
    "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010"
]

HAM_EXAMPLES = [
    "Ok lar... Joking wif u oni...",
    "Nah I don't think he goes to usf, he lives around here though",
    "Are you coming over for dinner tonight? Let me know so I can start cooking."
]

# Session state initialization for message input
if "sms_input" not in st.session_state:
    st.session_state.sms_input = ""

def set_example(text):
    st.session_state.sms_input = text

with col_examples:
    st.markdown("### 💡 Quick Test Examples")
    st.caption("Click any sample SMS below to test prediction:")
    
    st.markdown("**🚨 Sample Spam Messages**")
    for idx, sample in enumerate(SPAM_EXAMPLES, 1):
        if st.button(f"Spam #{idx}: \"{sample[:35]}...\"", key=f"spam_{idx}", use_container_width=True):
            set_example(sample)
            
    st.markdown("**💬 Sample Legitimate Messages**")
    for idx, sample in enumerate(HAM_EXAMPLES, 1):
        if st.button(f"Ham #{idx}: \"{sample[:35]}...\"", key=f"ham_{idx}", use_container_width=True):
            set_example(sample)

with col_main:
    st.markdown("### 📝 Enter SMS Message")
    
    user_input = st.text_area(
        label="Type or paste an SMS message below:",
        value=st.session_state.sms_input,
        height=140,
        placeholder="Enter your SMS message here (e.g., 'Congratulations! You've won a free ticket...')...",
        key="text_area_input"
    )
    
    btn_col1, btn_col2, _ = st.columns([1, 1, 2])
    with btn_col1:
        analyze_clicked = st.button("🔍 Analyze Message", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("🗑️ Clear Input", use_container_width=True):
            st.session_state.sms_input = ""
            st.rerun()

    # Trigger analysis on button click or if input is provided via example
    if analyze_clicked or (user_input.strip() and "prev_input" in st.session_state and st.session_state.prev_input != user_input):
        st.session_state.prev_input = user_input
        
        if not user_input.strip():
            st.warning("⚠️ Please enter an SMS message to analyze.")
        else:
            with st.spinner("Analyzing message with TF-IDF Logistic Regression..."):
                res = predict_spam(
                    user_input,
                    coeff_,
                    intercept_,
                    vectorizer,
                    idf_map,
                    stop_words_set,
                    threshold=metrics["threshold"]
                )
                
            st.markdown("---")
            st.markdown("### 🎯 Detection Result")
            
            prob_pct = res["spam_percentage"]
            prob_val = res["probability"]
            is_spam = res["is_spam"]
            
            # Result Banner
            if is_spam:
                st.markdown(f"""
                <div class="result-card-spam">
                    <div class="result-title-spam">🚨 SPAM DETECTED</div>
                    <div class="result-desc">
                        This message has a <strong>high likelihood ({prob_pct}%)</strong> of being spam, promotional, or a scam.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-ham">
                    <div class="result-title-ham">✅ LEGITIMATE MESSAGE (HAM)</div>
                    <div class="result-desc">
                        This message appears to be legitimate conversational communication (Spam probability: <strong>{prob_pct}%</strong>).
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Probability Meter & Key Indicators
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color: {'#f87171' if is_spam else '#34d399'};">{prob_pct}%</div>
                    <div class="metric-lbl">Spam Probability</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{res['threshold']}</div>
                    <div class="metric-lbl">Classification Threshold</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{res['in_vocab_words_count']} / {len(res['clean_words'])}</div>
                    <div class="metric-lbl">Matched Features</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Progress bar visualization
            st.markdown("**Probability Gauge:**")
            st.progress(float(prob_val))
            
            # Feature Insights Accordion
            with st.expander("🔍 Detailed Model Explanation & Feature Breakdown", expanded=True):
                st.markdown(f"**Logit Calculation (z-score):** `z = X_new @ weights + intercept` = `{res['z_score']}` → **Sigmoid Probability:** `{prob_pct}%`")
                
                if res["word_contributions"]:
                    st.markdown("##### Word Feature Contributions (TF-IDF × Model Weight)")
                    st.caption("Directional Impact = TF-IDF × Model Weight. Positive impact pushes prediction toward SPAM; negative impact pushes toward HAM.")
                    
                    df_contrib = pd.DataFrame(res["word_contributions"])
                    df_display = df_contrib.rename(columns={
                        "word": "Word Feature",
                        "tf": "Term Freq (TF)",
                        "idf": "IDF (Saved)",
                        "tfidf": "TF-IDF Value",
                        "weight": "Model Weight",
                        "impact": "Directional Impact (TF-IDF × Weight)",
                        "is_spam_indicator": "Spam Indicator?"
                    })
                    
                    st.dataframe(
                        df_display[["Word Feature", "Term Freq (TF)", "IDF (Saved)", "TF-IDF Value", "Model Weight", "Directional Impact (TF-IDF × Weight)", "Spam Indicator?"]],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No words in this message matched the vocabulary features from the training set.")
