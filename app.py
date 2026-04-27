import streamlit as st
import torch
import pickle
import numpy as np
import os
import gdown
from tensorflow import keras
from tensorflow.keras import layers
from sentence_transformers import util

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

st.set_page_config(
    page_title="ResearchAI — Smart Paper Discovery",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
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
download_file("18SaprNzVvkuwHTgwzyHjIqGXXwemVDKI", "models/model_new.keras")

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

    num_classes = len(vocab)

    inputs = keras.Input(shape=(20000,), name='text_input')
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(num_classes, activation='sigmoid', name='outputs')(x)

    model = keras.Model(inputs, output)
    model.load_weights('models/model_new.keras')

    return embeddings, sentences, rec_model, model, vocab

embeddings, sentences, rec_model, loaded_model, vocab = load_models()

def recommendation(input_paper):
    qe = rec_model.encode([input_paper])
    scores = util.cos_sim(qe, embeddings)
    top = torch.topk(scores, k=5)
    return [sentences[i.item()] for i in top.indices[0]]

# ===== SIMPLE UI =====
st.title("🔬 ResearchAI")

input_paper = st.text_input("Enter paper title")
abstract = st.text_area("Enter abstract")

if st.button("Analyse"):
    if input_paper:
        papers = recommendation(input_paper)
        st.subheader("Recommended Papers")
        for i, p in enumerate(papers):
            st.write(f"{i+1}. {p}")
