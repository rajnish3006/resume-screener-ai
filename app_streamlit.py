import streamlit as st
import fitz
import docx
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import base64

# ------------ Helper Functions ------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_resume(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    return ""

def highlight_keywords(text, keywords):
    highlighted = text
    for word in keywords:
        highlighted = re.sub(rf"\b({re.escape(word)})\b", r"**\1**", highlighted, flags=re.IGNORECASE)
    return highlighted

def rank_resumes(jd_text, resume_texts, top_n=None):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_embedding = model.encode([jd_text])[0]
    scores = {}

    for name, text in resume_texts.items():
        resume_embedding = model.encode([text])[0]
        score = cosine_similarity([jd_embedding], [resume_embedding])[0][0]
        scores[name] = {"score": round(score, 3), "text": text}

    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True))
    if top_n:
        sorted_scores = dict(list(sorted_scores.items())[:top_n])
    return sorted_scores

def generate_excel(results):
    data = [{"Resume": name, "Score": details["score"]} for name, details in results.items()]
    df = pd.DataFrame(data)
    towrite = pd.ExcelWriter("resume_ranking.xlsx", engine='xlsxwriter')
    df.to_excel(towrite, index=False, sheet_name='Ranking')
    towrite.save()
    with open("resume_ranking.xlsx", "rb") as file:
        b64 = base64.b64encode(file.read()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="resume_ranking.xlsx">📥 Download Ranking as Excel</a>'

# ------------ UI Layout Setup ------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
        .stButton button {
            background-color: #007BFF;
            color: white;
            font-weight: bold;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

# ------------ Logo & Title ------------
st.image("images.jpeg", width=120)
st.title("🤖 AI Resume Screener with Ranking & Highlights")

# ------------ Inputs ------------
col1, col2 = st.columns([2, 3])
with col1:
    jd_input = st.text_area("📄 Paste Job Description", height=300, placeholder="Paste the job description here...")

with col2:
    uploaded_files = st.file_uploader("📂 Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    top_n = st.number_input("🎯 Show Top N Resumes", min_value=1, step=1, value=5)

# ------------ Matching Logic ------------
if st.button("🔍 Match Resumes"):
    if not jd_input:
        st.warning("⚠️ Please paste a job description.")
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one resume.")
    else:
        with st.spinner("Processing resumes... Please wait ⏳"):
            resume_texts = {file.name: read_resume(file) for file in uploaded_files}
            results = rank_resumes(jd_input, resume_texts, top_n=top_n)
            jd_keywords = list(set(jd_input.lower().split()))

        st.success("✅ Matching Complete!")

        # Show Ranked Results
        st.subheader("🏆 Top Matching Resumes")
        for name, details in results.items():
            st.markdown(f"**📄 {name}** — Match Score: `{details['score']}`")
            with st.expander("🔎 View Resume Snippet"):
                st.markdown(highlight_keywords(details["text"][:1000], jd_keywords))

        # Download as Excel
        st.markdown("---")
        st.markdown(generate_excel(results), unsafe_allow_html=True)

        # ------------ Custom Footer ------------
st.markdown("""
    <hr style="margin-top: 50px; margin-bottom: 10px; border: 1px solid #eee;" />

    <div style="text-align: center; font-size: 18px; color: #666;">
        <span style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-style: italic;">
            🚀 Developed with ❤️ by <strong style="color: #007BFF;">Rajnish</strong>
        </span>
    </div>
""", unsafe_allow_html=True)

