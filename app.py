import os
import fitz  # PyMuPDF
import docx
import re

# ------------ Read Job Description ------------
def read_job_description(path="job_description.txt"):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

# ------------ Extract Text from PDF ------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# ------------ Extract Text from DOCX ------------
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

# ------------ Read All Resumes ------------
def read_resumes(folder="resumes"):
    resume_texts = {}
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(filepath)
        else:
            continue  # Skip unsupported file types
        resume_texts[filename] = text
    return resume_texts

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ------------ Load Sentence-BERT Model ------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------ Calculate Similarity Scores ------------
def rank_resumes(jd_text, resume_dict):
    scores = {}
    jd_embedding = model.encode([jd_text])[0]

    for name, resume_text in resume_dict.items():
        resume_embedding = model.encode([resume_text])[0]
        score = cosine_similarity([jd_embedding], [resume_embedding])[0][0]
        scores[name] = round(score, 3)  # Round for cleaner output

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# ------------ Test the Reading ------------
if __name__ == "__main__":
    jd = read_job_description()
    resumes = read_resumes()
    
    print("\n📄 Job Description Sample:\n", jd[:300])
    print("\n🧑‍💼 Matching Resumes...")

    results = rank_resumes(jd, resumes)

    print("\n🏆 Top Matching Resumes:")
    for name, score in results.items():
        print(f"{name}: {score}")


