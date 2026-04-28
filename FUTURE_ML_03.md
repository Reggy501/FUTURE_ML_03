# 📄 Resume / Candidate Screening System — FUTURE_ML_03

> **Future Interns Machine Learning Internship — Task 3**

## Overview
A fully dynamic AI-powered resume screening system. Paste any job description — the model reads it using NLP, extracts required skills automatically, then screens and ranks candidates based on semantic similarity and skill coverage. Zero hardcoding. The model makes all decisions.

---

## 🎯 Objective
Build an ML system to automatically screen and rank resumes based on a given job role, perform skill gap identification, and rank candidates by fit score.

---

## 🛠️ Tools & Libraries
| Tool | Purpose |
|------|---------|
| Python | Core language |
| spaCy (`en_core_web_sm`) | NER, noun chunks, POS tagging |
| NLTK | Stopwords, lemmatization |
| Scikit-learn | TF-IDF vectorization, cosine similarity |
| Pandas / NumPy | Data handling |
| Matplotlib / Seaborn | Visualizations |
| Streamlit | Interactive web application |
| Jupyter Notebook | Research & development |

---

## 📁 Project Structure
```
FUTURE_ML_03/
├── resume_screening.ipynb           # Research notebook
├── app.py                           # Streamlit web application
├── resume_screening_dashboard.png   # Visual dashboard
└── README.md
```

---

## 🔑 Key Features
- **Dynamic JD Parsing** — Model reads any job description and extracts requirements using spaCy NER, noun chunk extraction, POS tagging, and regex pattern matching
- **Resume Skill Extraction** — Same NLP pipeline applied to resumes — no predefined skill lists
- **Smart Matching** — Exact + substring containment matching handles partial phrases
- **Cosine Similarity Scoring** — TF-IDF semantic similarity between resume and JD
- **Composite Ranking Score** — `60% JD Similarity + 40% Skill Match Rate`
- **Skill Gap Identification** — Clearly shows what each candidate is missing
- **Streamlit App** — Full interactive web UI for real-world use
- **Visual Dashboard** — Ranking bar chart, matched vs gaps chart, skill coverage heatmap

---

## 📊 Scoring System
```
Final Score = (0.6 × JD Similarity %) + (0.4 × Skill Match Rate × 100)

🟢 ≥ 75  →  Strong fit
🟡 50–74 →  Moderate fit  
🔴 < 50  →  Weak fit
```

---

## 🚀 How to Run

### Jupyter Notebook
```bash
pip install pandas numpy scikit-learn nltk spacy matplotlib seaborn jupyter
python -m spacy download en_core_web_sm
jupyter notebook resume_screening.ipynb
```

### Streamlit App
```bash
pip install streamlit
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🌐 How the App Works
1. **Paste any job description** → model extracts requirements automatically
2. **Add candidate resumes** (up to 15) → paste raw resume text
3. **Click Screen** → model ranks all candidates instantly
4. **Review results** → ranking table, detailed reports, skill gaps, visual charts

Works for **any role** — data scientist, software engineer, product manager, finance analyst — the model adapts to the JD every time.

---

## 💡 NLP Pipeline
```
Raw Text
   ↓
spaCy Named Entity Recognition  (PRODUCT, ORG, LANGUAGE → skills)
   ↓
Noun Chunk Extraction           (multi-word technical phrases)
   ↓
POS Tagging                     (NOUN, PROPN → individual skills)
   ↓
Regex Pattern Matching          (c++, node.js, scikit-learn etc.)
   ↓
Frequency Scoring               (weighted by count + phrase length)
   ↓
Ranked Skill List
```

---

## 👤 Author
**Future Interns ML Internship**  
Track: Machine Learning | Task: 03  
GitHub Repo: `FUTURE_ML_03`