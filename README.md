# A.R.G.U.S.🕶️
### Augmented Retrieval & Generation Unified System🤖⚡

A.R.G.U.S. is a Retrieval-Augmented Generation (RAG) system designed to deliver accurate, context-aware responses by combining vector-based search with large language models.

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