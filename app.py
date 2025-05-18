import streamlit as st
from transformers import pipeline
import requests

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_model()

def fetch_context_from_google_doc():
    # Google Docs export link (text format)
    export_url = "https://docs.google.com/document/d/10aPWxSbxGQU0awCm-yW_OjyXLsdA-Nf45yJWx3NwbmM/export?format=txt"
    response = requests.get(export_url)
    if response.status_code == 200:
        return response.text
    else:
        return "Failed to load document content."

context = fetch_context_from_google_doc()

st.title("Rice Disease Q&A")

question = st.text_input("Ask a question about rice diseases:")

if question:
    result = qa_pipeline(question=question, context=context)
    st.write("Answer:", result['answer'])
