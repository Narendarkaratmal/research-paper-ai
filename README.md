# 📄 Research Paper AI  
### 🔍 Recommendation + 🧠 Subject Area Prediction using ML & NLP

🚀 Live App:  
https://research-paper-ai-buzfv5dmxfbs9jafwmayul.streamlit.app

---

## 📌 Overview

This project is an end-to-end **Machine Learning + NLP web application** that helps users:

1. 🔍 **Find similar research papers** based on a given title or text  
2. 🧠 **Predict subject areas** of research papers using deep learning  

The system is deployed as an interactive **Streamlit web app**, making it easy to use without any coding knowledge.

---

## ⚙️ Tech Stack

- **Python**
- **Streamlit** (Frontend + Deployment)
- **NumPy / Scikit-learn**
- **Sentence Transformers (MiniLM)**
- **Deep Learning (MLP Model)**
- **Pickle / NPY for model storage**

---

## 🧠 How It Works

### 🔹 1. Recommendation System
- Uses **Sentence Transformers (`all-MiniLM-L6-v2`)**
- Converts research paper titles into **dense vector embeddings**
- Computes **Cosine Similarity**
- Returns **Top-5 most similar papers**

---

### 🔹 2. Subject Area Prediction
- Uses a **Multi-Layer Perceptron (MLP)**
- Input: Paper abstract  
- Output: Predicted subject area(s)
- Uses **text vectorization + trained neural network**

---
## 📁 Project Structure

```
research-paper-ai/
│
├── app.py                  # Streamlit web app
├── requirements.txt        # Dependencies
├── README.md               # Documentation
│
├── notebooks/
│   ├── 01_recommendation_model.ipynb
│   └── 02_subject_prediction_model.ipynb
│
├── models/
│   ├── embeddings.npy
│   ├── sentences.pkl
│   ├── model.h5
│   ├── vocab.pkl
│   └── vectorizer files
```


---

## 🚀 Features

- 🔍 Top-5 similar research paper recommendations  
- 🧠 Multi-label subject prediction  
- ⚡ Fast inference using precomputed embeddings  
- 🌐 Live deployed web application  
- 🖥️ Simple and clean UI  

---

## 📊 Dataset Format

Your dataset should include:

| Column        | Description                          |
|--------------|--------------------------------------|
| title        | Paper title                          |
| abstract     | Paper abstract                       |
| subject_area | Subject category (single/multiple)   |

---

## ▶️ How to Run Locally

```bash
# Clone repo
git clone https://github.com/your-username/research-paper-ai.git
cd research-paper-ai

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
