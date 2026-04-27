import streamlit as st
import numpy as np
import gdown
from sklearn.metrics.pairwise import cosine_similarity

# -------- DOWNLOAD FILES --------
def download(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False, fuzzy=True)
# -------- LOAD FILES --------
@st.cache_data
def load_data():
    download("1RuA9vCqA-xmNndFWEYtkLdVCopqN_8RH", "embeddings.npy")
    download("13m4RN8yz0-CQ4Vn7-ie0TCIio08yweQj", "sentences.pkl")

    embeddings = np.load("embeddings.npy")
    import pickle
    sentences = pickle.load(open("sentences.pkl", "rb"))

    return embeddings, sentences

embeddings, sentences = load_data()

# -------- UI --------
st.title("📄 Research Paper AI")

query = st.text_input("Enter your query")

if query:
    # simple fake embedding (since model removed)
    query_vec = np.random.rand(1, embeddings.shape[1])

    scores = cosine_similarity(query_vec, embeddings)[0]
    top_idx = np.argsort(scores)[-5:][::-1]

    st.subheader("Top Results:")
    for i in top_idx:
        st.write(sentences[i])
