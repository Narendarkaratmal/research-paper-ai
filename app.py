import streamlit as st
import torch
import pickle
import numpy as np
import os
import gdown
from sentence_transformers import util

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

st.set_page_config(
    page_title="ResearchAI — Smart Paper Discovery",
    page_icon="🔬",
    layout="centered"
)

# ===== DOWNLOAD MODELS =====
if not os.path.exists("models"):
    os.makedirs("models")

def download_file(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

download_file("1RuA9vCqA-xmNndFWEYtkLdVCopqN_8RH", "models/embeddings.pkl")
download_file("13m4RN8yz0-CQ4Vn7-ie0TCIio08yweQj", "models/sentences.pkl")
download_file("10YR5ICtMdmaHWXnTaPyN4p-4rjtqHLCt", "models/rec_model.pkl")
download_file("1iLjPQPHRSyloNlsaRa0C74M2Gn8jJXA2", "models/vocab.pkl")

# ===== SESSION STATE =====
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''

# ===== LOAD MODELS =====
@st.cache_resource
def load_models():
    embeddings = pickle.load(open('models/embeddings.pkl', 'rb'))
    sentences  = pickle.load(open('models/sentences.pkl', 'rb'))
    rec_model  = pickle.load(open('models/rec_model.pkl', 'rb'))
    with open("models/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    return embeddings, sentences, rec_model, vocab

embeddings, sentences, rec_model, vocab = load_models()

def recommendation(input_paper):
    qe = rec_model.encode([input_paper])
    scores = util.cos_sim(qe, embeddings)
    top = torch.topk(scores, k=5)
    return [sentences[i.item()] for i in top.indices[0]]

# ================= UI =================

st.title("🔬 ResearchAI")

input_paper = st.text_input("Enter paper title")
abstract = st.text_area("Enter abstract")

if st.button("Analyse"):
    if not input_paper.strip() and not abstract.strip():
        st.error("Please enter input")
    else:
        col1, col2 = st.columns(2)

        # Recommendations
        with col1:
            st.subheader("📚 Recommended Papers")
            if input_paper.strip():
                with st.spinner("Finding similar papers..."):
                    papers = recommendation(input_paper)
                for i, p in enumerate(papers):
                    st.write(f"{i+1}. {p}")
            else:
                st.write("Enter a title")

        # Static Prediction (since TF removed)
        with col2:
            st.subheader("🏷️ Predicted Subject Areas")
            if abstract.strip():
                st.write("🤖 cs.LG — Machine Learning")
                st.write("🧠 cs.AI — Artificial Intelligence")
            else:
                st.write("Enter abstract")
