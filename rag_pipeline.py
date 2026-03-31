import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA                          # ✅ fix #1
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate


PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.

Answer the question using ONLY the context below.
Explain the answer clearly in bullet points.

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

    if os.path.exists("vector_store"):
        print("Loading existing vector database...")
        vector_db = FAISS.load_local(
            "vector_store",
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_db

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if len(documents) == 0:
        raise ValueError("No text could be extracted from the PDF.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    if len(chunks) == 0:
        raise ValueError("Chunking failed.")

    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("vector_store")
    return vector_db


def create_qa_chain(vector_db):
    from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200,        # ✅ fix #3
        do_sample=False
    )

    llm = HuggingFacePipeline(pipeline=pipe)

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