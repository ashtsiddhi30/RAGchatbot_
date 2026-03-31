# 📄 Smart Document Assistant

An AI-powered PDF chatbot using Retrieval-Augmented Generation (RAG).

## 🚀 Features
- Upload and analyze PDF documents
- Ask questions and get contextual answers
- AI-generated suggested questions
- Source-based answer retrieval
- Interactive UI with Streamlit

## 🧠 Tech Stack
- Python
- Streamlit
- LangChain
- FAISS (Vector Database)
- HuggingFace Transformers

## ⚙️ How it Works
1. PDF is loaded and split into chunks
2. Text is converted into embeddings
3. Stored in FAISS vector database
4. User query retrieves relevant chunks
5. LLM generates contextual answer

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py