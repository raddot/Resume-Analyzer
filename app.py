import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import re
import pdfplumber

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("AI Resume Analyzer")
st.write("Upload Resume + Paste Job Description")

# Load Model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embed_model = load_model()

# Upload Resume
uploaded_resume = st.file_uploader("Upload Resume (.pdf only)", type=["pdf"])

# Job Description
job_description = st.text_area("Paste Job Description", height=250)

# Extract text from PDF
def extract_pdf_text(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Clean text
def clean_text(text):
    text = text.lower()                    
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ATS Score
def calculate_ats_score(resume_text, jd_text):
    resume_embedding = embed_model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = embed_model.encode(jd_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(resume_embedding, jd_embedding)
    score = float(similarity[0][0]) * 100
    return round(min(score, 100), 2)

# Extract keywords
def extract_keywords(text):
    words = text.split()
    common_words = {"the", "and", "for", "with", "that", "this", "have", "from", "your", "you"}
    keywords = [word for word in words if len(word) > 4 and word not in common_words]
    return list(set(keywords))

# Analyze
if st.button("Analyze Resume"):
    if uploaded_resume and job_description:
        with st.spinner("Reading PDF..."):
            raw_text = extract_pdf_text(uploaded_resume)

        if not raw_text.strip():
            st.error("Could not extract text from PDF. It may be scanned/image-based.")
        else:
            resume_text = clean_text(raw_text)
            jd_text = clean_text(job_description)

            with st.spinner("Analyzing..."):
                ats_score = calculate_ats_score(resume_text, jd_text)

            st.subheader("ATS Match Score")
            st.metric(label="Score", value=f"{ats_score}%")

            jd_keywords = extract_keywords(jd_text)
            resume_keywords = extract_keywords(resume_text)
            matched_keywords = [w for w in jd_keywords if w in resume_keywords]
            missing_keywords = [w for w in jd_keywords if w not in resume_keywords]

            st.subheader("Matched Keywords")
            st.write(matched_keywords[:20])

            st.subheader("Missing Keywords")
            st.write(missing_keywords[:20])

            st.subheader("Suggestions")
            if ats_score >= 80:
                st.success("Excellent match for the job description.")
            elif ats_score >= 60:
                st.warning("Good match. Add more relevant keywords.")
            else:
                st.error("Low ATS score. Improve resume alignment.")
    else:
        st.warning("Please upload resume and paste job description.")