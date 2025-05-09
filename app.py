import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# UI Title
st.set_page_config(page_title=" PDF Q&A (Offline)")
st.title("📄 Upload and Ask Questions from Your PDFs")

# Choose Ollama Model
model_choice = st.selectbox(
    "Choose local LLM model:",
    ["mistral", "phi", "llama2", "tinyllama"],
    index=0
)

uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

question = st.text_input("Ask a question based on the uploaded PDFs")

if uploaded_files and question:
    with st.spinner("Processing..."):

        # Load and split documents
        all_texts = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            os.remove(tmp_path)

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(pages)
            all_texts.extend(chunks)

        # Embed and store in Chroma
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(all_texts, embedding=embeddings, persist_directory="./chroma_db")

        # Setup Retriever
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # LLM
        llm = Ollama(model=model_choice, temperature=0.1)

        # QA Chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Get Answer
        answer = qa_chain.run(question)

    st.subheader("Answer:")
    st.success(answer)
