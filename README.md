# RAG PDF Chatbot 🤖📄

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload a PDF and ask questions about its content.
The system retrieves relevant information from the document and generates accurate answers using a language model.

---

## 🚀 Features

* Upload and process PDF documents
* Extract and split document text into chunks
* Create embeddings for semantic search
* Store embeddings in a FAISS vector database
* Retrieve relevant document context for user queries
* Generate answers using an LLM
* Interactive chatbot interface

---

## 🧠 How It Works

1. **Document Upload**
   The user uploads a PDF document.

2. **Text Processing**
   The document is loaded and split into smaller chunks for better retrieval.

3. **Embedding Generation**
   Each chunk is converted into vector embeddings using a HuggingFace embedding model.

4. **Vector Storage**
   The embeddings are stored in a FAISS vector database.

5. **Query Processing**
   When the user asks a question, the system:

   * Searches the vector database
   * Retrieves relevant chunks

6. **Answer Generation**
   The retrieved context is passed to the language model to generate a response.

---

## 🏗️ Project Structure

```
RAG-chatbot
│
├── app.py               # Main chatbot application
├── rag_pipeline.py      # RAG pipeline logic
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation
├── .gitignore           # Ignored files
│
├── vector_store/        # FAISS vector database (generated)
├── temp.pdf             # Temporary uploaded PDF
└── venv/                # Virtual environment
```

---

## 🛠️ Technologies Used

* Python
* LangChain
* FAISS Vector Database
* HuggingFace Embeddings
* LLM (Local or API based)
* Streamlit (for UI)

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```
git clone https://github.com/Vedika07-art/RAG-chatbot.git
cd RAG-chatbot
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate it:

Windows

```
venv\Scripts\activate
```

Mac/Linux

```
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
streamlit run app.py
```

The chatbot will open in your browser.

---

## 💡 Example Questions

* "Summarize this document"
* "What are the key points?"
* "Explain the main topic"
* "List important concepts from the PDF"

---

## 📌 Future Improvements

* Support multiple document uploads
* Improve retrieval accuracy
* Add conversation memory
* Deploy the chatbot online
* Add better UI and chat history

---

## 👩‍💻 Author

**Ashtsiddhi Kadam**

Computer Engineering Student
AI / Machine Learning Enthusiast

GitHub:
https://github.com/Vedika07-art

---

## ⭐ If you found this project useful

Give the repository a ⭐ on GitHub.
