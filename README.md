# 📄 Research Papers Recommendation & Subject Area Prediction

A machine learning project combining **LLMs** and **Deep Learning** for:
1. Recommending similar research papers (Sentence Transformers + Cosine Similarity)
2. Predicting subject areas of papers (MLP Text Classifier)

---

## Project Structure

```
research_paper_project/
│
├── app.py                          # Streamlit web app
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── notebooks/
│   ├── 01_recommendation_model.ipynb    # Train recommendation model
│   └── 02_subject_prediction_model.ipynb # Train prediction model
│
└── models/                         # Saved models (generated after training)
    ├── embeddings.pkl
    ├── sentences.pkl
    ├── rec_model.pkl
    ├── model.h5
    ├── text_vectorizer_config.pkl
    ├── text_vectorizer_weights.pkl
    └── vocab.pkl
```

---

## Setup & Installation

```bash
# 1. Clone or download the project
cd research_paper_project

# 2. Install dependencies
pip install -r requirements.txt
```

---

## How to Use

### Step 1 — Train the Models

Run the notebooks **in order**:

**Notebook 1 — Recommendation Model**
```
notebooks/01_recommendation_model.ipynb
```
- Loads your dataset (titles)
- Generates sentence embeddings using `all-MiniLM-L6-v2`
- Saves: `embeddings.pkl`, `sentences.pkl`, `rec_model.pkl`

**Notebook 2 — Subject Prediction Model**
```
notebooks/02_subject_prediction_model.ipynb
```
- Loads your dataset (abstracts + subject areas)
- Trains an MLP classifier
- Saves: `model.h5`, `text_vectorizer_config.pkl`, `text_vectorizer_weights.pkl`, `vocab.pkl`

### Step 2 — Run the Streamlit App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Dataset Format

Your CSV dataset should have these columns:

| Column | Description |
|---|---|
| `title` | Paper title (used for recommendation) |
| `abstract` | Paper abstract (used for prediction) |
| `subject_area` | One or more subject areas (list for multi-label) |

---

## Features

- **Recommendation**: Top-5 similar papers using cosine similarity on sentence embeddings
- **Prediction**: Multi-label subject area classification using a 3-layer MLP
- **Accuracy**: ~99% on trained dataset
- **UI**: Clean Streamlit interface

---

## License

Licensed under **NOOR SAEED**
