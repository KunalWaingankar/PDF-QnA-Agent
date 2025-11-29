import streamlit as st
from agent import PDFQnAAgent
import tempfile
import os

st.set_page_config(page_title="PDF QnA Agent", layout="wide")
st.title("📄 PDF QnA Agent")
st.write("Upload a PDF and ask questions about its content.")

# Initialize agent in session
if "agent" not in st.session_state:
    st.session_state.agent = None

# PDF Upload
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Save uploaded PDF to a temp file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write(f"**Loaded PDF:** {uploaded_file.name}")

    # Process PDF when button clicked
    if st.button("Process PDF"):
        with st.spinner("Processing..."):
            agent = PDFQnAAgent()
            agent.process_pdf(temp_path)
            st.session_state.agent = agent
        st.success("PDF processed successfully!")

# Question input
if st.session_state.agent:
    question = st.text_input("Ask a question about the PDF:")
    if st.button("Get Answer"):
        if question.strip():
            with st.spinner("Searching..."):
                answer = st.session_state.agent.ask(question, top_k=5)
            st.write("### Answer:")
            st.write(answer)

