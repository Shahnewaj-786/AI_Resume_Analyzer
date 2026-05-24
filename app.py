from pdfminer.high_level import extract_text

def get_pdf_text(pdf_file):
    text = extract_text(pdf_file)
    return text


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(resume_text, job_desc):
    content = [resume_text, job_desc]
    cv = TfidfVectorizer()
    matrix = cv.fit_transform(content)
    similarity_matrix = cosine_similarity(matrix)
    # Returns percentage match
    return similarity_matrix[0][1] * 100


import streamlit as st

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer")

# Sidebar for Job Description
st.sidebar.header("Job Requirements")
job_description = st.sidebar.text_area("Paste the Job Description here...")

# File Uploader
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

if uploaded_file and job_description:
    with st.spinner('Analyzing...'):
        # 1. Extract Text
        resume_text = get_pdf_text(uploaded_file)

        # 2. Calculate Match
        score = calculate_similarity(resume_text, job_description)

        # 3. Display Results
        st.subheader("Analysis Results")
        st.metric(label="Match Score", value=f"{round(score, 2)}%")

        if score > 70:
            st.success("Great Match! Your profile aligns well with this role.")
        elif score > 40:
            st.warning("Moderate Match. Consider adding more keywords from the JD.")
        else:
            st.error("Low Match. You might need to tailor your resume significantly.")