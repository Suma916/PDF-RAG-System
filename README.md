# PDF Q&A System (Offline)

This is a simple project I built where you can upload one or more PDF files and ask questions based on their content. It works completely offline and uses local LLMs through Ollama, you don't need an internet to get answers.

---

## Features

- Upload multiple PDFs
- Ask any question from the content
- Uses local language models like Mistral, Phi, TinyLlama, etc.
- Vector embeddings using HuggingFace and ChromaDB
- Works with Streamlit UI

---

## How to Run

1. First, install all the required packages:
pip install -r requirements.txt

2. Make sure [Ollama](https://ollama.com/download) is installed and running.

3. Pull or start any model you want to use, like this:
   ollama run mistral

4. Then, run the app:
   streamlit run app.py


5. Upload your PDF(s), ask a question, and see the answer.

---

## Models Supported

- mistral
- phi
- llama2
- tinyllama

Choose the one that works best for your system (mistral can be slower on low-end devices but tintllama can be less accurate though fast).

---

## Tech Used

- Python
- Langchain
- HuggingFace Embeddings
- ChromaDB
- Streamlit
- Ollama

---

## Notes

This was mainly built for learning purposes and to experiment with local models and vector search. Feel free to use or improve it.

---

## Author

Made by Sumasri Karuturi

   

