import streamlit as st
from rag_pipeline import create_vector_store, create_qa_chain

st.set_page_config(page_title="PDF Chatbot")

st.title("📄 PDF Chatbot")
st.write("Upload a PDF and chat with it!")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    vector_db = create_vector_store("temp.pdf")
    qa_chain = create_qa_chain(vector_db)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show previous chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input bar
    question = st.chat_input("Ask a question about the PDF")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = qa_chain.run(question)

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})