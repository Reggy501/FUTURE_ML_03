import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🤖",
    layout="wide"
)

# ─────────────────────────────────────────────
#  LOAD NLP MODELS (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nlp = spacy.load('en_core_web_sm')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return nlp, stop_words, lemmatizer

nlp, stop_words, lemmatizer = load_nlp()

# ─────────────────────────────────────────────
#  CORE NLP FUNCTIONS — fully dynamic
# ─────────────────────────────────────────────
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\+\#]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

def extract_skills_with_nlp(text):
    """
    Dynamically extracts skills from any text using:
    - spaCy Named Entity Recognition
    - Noun chunk extraction
    - POS tagging (nouns + proper nouns)
    - Regex pattern matching for tech tokens
    No hardcoded skill list whatsoever.
    """
    doc = nlp(text[:100000])
    skills = set()

    # 1. Named entities — PRODUCT, ORG, LANGUAGE often map to tech skills
    for ent in doc.ents:
        if ent.label_ in ('PRODUCT', 'ORG', 'GPE', 'LANGUAGE', 'WORK_OF_ART'):
            token = ent.text.lower().strip()
            if 2 < len(token) < 40:
                skills.add(token)

    # 2. Noun chunks — catches multi-word technical phrases
    for chunk in doc.noun_chunks:
        token = chunk.text.lower().strip()
        token = re.sub(r'[^a-z0-9\s\+\#\.]', '', token).strip()
        if 2 < len(token) < 40 and token not in stop_words:
            skills.add(token)

    # 3. Individual meaningful tokens (nouns, proper nouns)
    for token in doc:
        if token.pos_ in ('NOUN', 'PROPN') and not token.is_stop:
            t = token.lemma_.lower().strip()
            if 2 < len(t) < 30:
                skills.add(t)

    # 4. Regex — handles c++, node.js, scikit-learn, etc.
    tech_pattern = re.findall(
        r'\b([a-z][a-z0-9]*(?:[+#\.\-][a-z0-9]+)*)\b', text.lower()
    )
    for t in tech_pattern:
        if 2 < len(t) < 30 and t not in stop_words:
            skills.add(t)

    return skills

def extract_required_skills(jd_text):
    """
    Reads the job description and figures out what skills are required.
    Uses NLP extraction + frequency scoring — model decides, not us.
    """
    raw_skills = extract_skills_with_nlp(jd_text)
    jd_lower = jd_text.lower()

    scored = {}
    for skill in raw_skills:
        count = jd_lower.count(skill)
        # Multi-word phrases get a bonus — more specific = stronger signal
        weight = count * (1 + 0.3 * len(skill.split()))
        if weight > 0:
            scored[skill] = weight

    sorted_skills = sorted(scored, key=scored.get, reverse=True)

    # Filter noise
    filtered = [
        s for s in sorted_skills
        if len(s) > 2
        and s not in stop_words
        and not s.isdigit()
    ]
    return filtered[:40]

def find_skill_overlap(jd_skills, resume_skills):
    """
    Matches JD requirements against resume skills.
    Uses exact match + substring containment so partial phrases still score.
    """
    matched = []
    gaps = []
    for jd_skill in jd_skills:
        found = any(
            jd_skill == res or jd_skill in res or res in jd_skill
            for res in resume_skills
        )
        (matched if found else gaps).append(jd_skill)
    return matched, gaps

def screen_candidate(name, resume_text, jd_skills, tfidf_model, jd_vector):
    """
    Screens one candidate. Everything is driven by the NLP model —
    no rules, no hardcoded thresholds beyond the scoring weights.
    """
    resume_skills = extract_skills_with_nlp(resume_text)
    matched, gaps = find_skill_overlap(jd_skills, resume_skills)

    cleaned_resume = preprocess_text(resume_text)
    vec = tfidf_model.transform([cleaned_resume])
    similarity = float(cosine_similarity(vec, jd_vector).flatten()[0]) * 100

    match_rate = len(matched) / len(jd_skills) if jd_skills else 0
    final_score = round(min((0.6 * similarity) + (0.4 * match_rate * 100), 100), 2)

    return {
        'candidate': name,
        'final_score': final_score,
        'similarity_score': round(similarity, 2),
        'match_count': len(matched),
        'total_required': len(jd_skills),
        'skills_matched': matched,
        'skill_gaps': gaps,
        'resume_skills': sorted(resume_skills),
    }

def build_tfidf(jd_text):
    """Fits TF-IDF on the actual JD provided — dynamic per session."""
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    cleaned_jd = preprocess_text(jd_text)
    tfidf.fit([cleaned_jd])
    jd_vector = tfidf.transform([cleaned_jd])
    return tfidf, jd_vector

def fit_label(score):
    if score >= 75:
        return "🟢 Strong fit"
    elif score >= 50:
        return "🟡 Moderate fit"
    else:
        return "🔴 Weak fit"

# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────
st.title("🤖 AI Resume Screening System")
st.markdown(
    "Paste **any job description** — the model reads it and extracts "
    "requirements automatically using NLP. Then screen as many candidates "
    "as you need. **No hardcoding. No manual skill lists. Model decides.**"
)
st.divider()

# ── STEP 1: JOB DESCRIPTION ───────────────────
st.subheader("Step 1 — Paste the Job Description")
jd_text = st.text_area(
    "Job Description",
    height=220,
    placeholder=(
        "Paste the full job description here...\n\n"
        "The model will automatically read it and extract "
        "the key skills and requirements."
    )
)

if jd_text.strip():
    with st.spinner("Model is reading the job description..."):
        jd_skills = extract_required_skills(jd_text)
        tfidf_model, jd_vector = build_tfidf(jd_text)

    st.success(
        f"✅ Model extracted **{len(jd_skills)}** key requirements from the JD"
    )
    with st.expander("🔍 See what the model extracted from the JD"):
        cols = st.columns(4)
        for i, skill in enumerate(jd_skills):
            cols[i % 4].markdown(f"• `{skill}`")

    st.divider()

    # ── STEP 2: CANDIDATES ────────────────────
    st.subheader("Step 2 — Add Candidate Resumes")
    num_candidates = st.number_input(
        "How many candidates?",
        min_value=1, max_value=15, value=2, step=1
    )

    candidates_input = []
    for i in range(int(num_candidates)):
        st.markdown(f"#### Candidate {i + 1}")
        col1, col2 = st.columns([1, 3])
        with col1:
            name = st.text_input(
                "Full Name", key=f"name_{i}",
                placeholder="e.g. Alice Mwangi"
            )
        with col2:
            resume = st.text_area(
                "Resume Text", key=f"resume_{i}", height=150,
                placeholder="Paste candidate's full resume text here..."
            )
        candidates_input.append((name, resume))
        st.divider()

    # ── STEP 3: SCREEN ────────────────────────
    if st.button("🔍 Screen & Rank All Candidates",
                 type="primary", use_container_width=True):

        valid = [
            (n.strip(), r.strip())
            for n, r in candidates_input
            if n.strip() and r.strip()
        ]

        if not valid:
            st.error("Please fill in at least one candidate name and resume.")
        else:
            results = []
            progress = st.progress(0, text="Screening candidates...")
            for idx, (name, resume) in enumerate(valid):
                result = screen_candidate(
                    name, resume, jd_skills, tfidf_model, jd_vector
                )
                results.append(result)
                progress.progress(
                    (idx + 1) / len(valid),
                    text=f"Screened {idx + 1} of {len(valid)}..."
                )

            results = sorted(
                results, key=lambda x: x['final_score'], reverse=True
            )
            for i, r in enumerate(results):
                r['rank'] = i + 1

            progress.empty()
            st.success(
                f"✅ Done! {len(results)} candidate(s) screened and ranked."
            )
            st.divider()

            # ── RANKING TABLE ──────────────────
            st.subheader("🏆 Candidate Rankings")
            table_rows = [{
                'Rank': f"#{r['rank']}",
                'Candidate': r['candidate'],
                'Final Score': f"{r['final_score']}/100",
                'JD Similarity': f"{r['similarity_score']}%",
                'Requirements Met': f"{r['match_count']}/{r['total_required']}",
                'Verdict': fit_label(r['final_score']),
            } for r in results]
            st.dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True,
                hide_index=True
            )
            st.divider()

            # ── DETAILED REPORTS ───────────────
            st.subheader("📄 Candidate Reports")
            for r in results:
                with st.expander(
                    f"{fit_label(r['final_score'])}  |  "
                    f"#{r['rank']} — {r['candidate']}  |  "
                    f"Score: {r['final_score']}/100"
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Final Score", f"{r['final_score']}/100")
                    c2.metric("JD Similarity", f"{r['similarity_score']}%")
                    c3.metric(
                        "Requirements Met",
                        f"{r['match_count']}/{r['total_required']}"
                    )

                    st.markdown("**✅ Requirements matched:**")
                    st.markdown(
                        " ".join([f"`{s}`" for s in r['skills_matched']])
                        if r['skills_matched'] else "_None matched_"
                    )

                    st.markdown("**❌ Missing requirements:**")
                    st.markdown(
                        " ".join([f"`{s}`" for s in r['skill_gaps']])
                        if r['skill_gaps'] else "🎉 _No gaps — full match!_"
                    )

                    with st.expander("View all skills detected in this resume"):
                        st.markdown(
                            " ".join([f"`{s}`" for s in r['resume_skills']])
                            if r['resume_skills'] else "_None detected_"
                        )

            st.divider()

            # ── CHARTS ────────────────────────
            st.subheader("📊 Visual Analysis")
            names_short = [r['candidate'].split()[0] for r in results]
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(6, max(3, len(results))))
                scores = [r['final_score'] for r in results]
                colors = [
                    'gold' if i == 0 else
                    'steelblue' if r['final_score'] >= 50 else 'tomato'
                    for i, r in enumerate(results)
                ]
                bars = ax.barh(
                    names_short[::-1], scores[::-1], color=colors[::-1]
                )
                ax.axvline(75, color='green', linestyle='--',
                           linewidth=1, label='Strong (75)')
                ax.axvline(50, color='orange', linestyle='--',
                           linewidth=1, label='Moderate (50)')
                ax.set_title('Final Ranking Scores')
                ax.set_xlabel('Score / 100')
                ax.legend(fontsize=7)
                ax.grid(True, axis='x', alpha=0.3)
                for bar, val in zip(bars, scores[::-1]):
                    ax.text(
                        bar.get_width() + 0.3,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val}', va='center', fontsize=9
                    )
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(6, max(3, len(results))))
                x = np.arange(len(results))
                w = 0.35
                ax.bar(x - w / 2, [r['match_count'] for r in results],
                       w, label='Matched', color='steelblue')
                ax.bar(x + w / 2, [len(r['skill_gaps']) for r in results],
                       w, label='Gaps', color='tomato')
                ax.set_xticks(x)
                ax.set_xticklabels(names_short, rotation=20)
                ax.set_title('Requirements Met vs Gaps')
                ax.legend()
                ax.grid(True, axis='y', alpha=0.3)
                st.pyplot(fig)
                plt.close()

            # Heatmap
            if len(results) > 1 and jd_skills:
                st.markdown("**Skill Coverage Heatmap**")
                top_skills = jd_skills[:20]
                heatmap_data = []
                for r in results:
                    matched_set = set(r['skills_matched'])
                    heatmap_data.append([
                        1 if any(s in m or m in s for m in matched_set) else 0
                        for s in top_skills
                    ])
                heatmap_df = pd.DataFrame(
                    heatmap_data,
                    index=names_short,
                    columns=top_skills
                )
                fig, ax = plt.subplots(
                    figsize=(min(16, len(top_skills)), max(3, len(results)))
                )
                sns.heatmap(
                    heatmap_df, ax=ax, cmap='YlGn',
                    linewidths=0.5, linecolor='white',
                    cbar=False, annot=True, fmt='d'
                )
                ax.set_title('Coverage map — top 20 JD requirements')
                ax.tick_params(axis='x', rotation=40, labelsize=8)
                st.pyplot(fig)
                plt.close()

            st.divider()
            st.caption(
                "AI Resume Screener · Future Interns ML Task 3 · "
                "Powered by spaCy NER + TF-IDF · Zero hardcoding"
            )

else:
    st.info(
        "👆 Paste a job description above — "
        "the model will read it and take it from there."
    )