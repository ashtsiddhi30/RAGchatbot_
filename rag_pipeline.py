import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate


# Improved Prompt
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.

Answer the question using ONLY the context provided below.

- Explain the answer clearly using bullet points.
- Rewrite the information in your own words.
- Do not copy sentences directly from the context.
- If the answer is not present in the context, say "The answer is not available in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
)


def create_vector_store(pdf_path):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load existing vector store
    if os.path.exists("vector_store"):
        print("Loading existing vector database...")
        vector_db = FAISS.load_local(
            "vector_store",
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_db

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("Documents loaded:", len(documents))

    if len(documents) == 0:
        raise ValueError("No text could be extracted from the PDF.")

    # Smaller chunks (better for FLAN-T5)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    print("Chunks created:", len(chunks))

    if len(chunks) == 0:
        raise ValueError("Chunking failed. No chunks created.")

    # Create FAISS vector store
    vector_db = FAISS.from_documents(chunks, embeddings)

    # Save vector store
    vector_db.save_local("vector_store")
    print("Vector database saved!")

    return vector_db


def create_qa_chain(vector_db):

    # ✅ FIXED: specify task explicitly
    generator = pipeline(
        "text2text-generation",   # 🔥 FIX
        model="google/flan-t5-base",
        max_new_tokens=200,
        temperature=0.2,
        do_sample=False   # 🔥 FIX repetition issue
    )

    llm = HuggingFacePipeline(pipeline=generator)

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    return qa_chain