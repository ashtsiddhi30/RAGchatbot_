import streamlit as st
from rag_pipeline import create_vector_store, create_qa_chain
from streamlit_pdf_viewer import pdf_viewer

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "sources" not in st.session_state:
    st.session_state.sources = []

# SESSION STATES
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

if "suggested" not in st.session_state:
    st.session_state.suggested = []

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

# PAGE SETTINGS
st.set_page_config(
    page_title="PDF Chatbot",
    layout="wide"
)

# UI STYLE
st.markdown("""
<style>

/* GLOBAL */
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: white;
}

/* HEADER */
.header {
    text-align: center;
    margin-bottom: 20px;
}

/* CHAT */
.chat-user {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    padding: 12px 18px;
    border-radius: 18px;
    margin-bottom: 10px;
    text-align: right;
    max-width: 75%;
    margin-left: auto;
}

.chat-bot {
    background: rgba(15, 23, 42, 0.7);
    padding: 12px 18px;
    border-radius: 18px;
    margin-bottom: 10px;
    max-width: 75%;
    border: 1px solid rgba(255,255,255,0.1);
}

/* CARDS */
.card {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 16px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 15px;
}

/* SOURCE */
.source-box {
    background: rgba(2,6,23,0.8);
    padding: 10px;
    border-radius: 10px;
    border: 1px solid #1e293b;
}

/* HIGHLIGHT */
.highlight {
    background: #fde68a;
    color: black;
    padding: 2px 4px;
    border-radius: 4px;
}

</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="header">
    <h1>📄 Smart Document Assistant</h1>
    <p>Upload, Analyze, and Chat with your PDF intelligently</p>
</div>
""", unsafe_allow_html=True)

# TOP BAR
col1, col2, col3 = st.columns([2,1,1])

with col1:
    st.markdown("### 📤 Upload PDF")
    uploaded_file = st.file_uploader("", type="pdf")

with col2:
    reset_chat = st.button("🔄 Reset", use_container_width=True)

with col3:
    download_chat = st.button("⬇ Download", use_container_width=True)

# 👇 THIS CREATES RIGHT-ALIGNED SUGGESTIONS (UNDER BUTTONS)
col_empty, col_suggest = st.columns([2,1])

with col_suggest:

    if st.session_state.suggested:
        st.markdown("### 💡 AI Suggested Questions")

        for i, q in enumerate(st.session_state.suggested):
            if st.button(q, key=f"suggested_{i}"):

                if st.session_state.get("qa_chain"):
                    result = st.session_state.qa_chain.invoke({"query": q})

                    answer = result["result"]

                    st.session_state.messages.append(
                        {"role":"user","content":q}
                    )

                    st.session_state.messages.append(
                        {"role":"assistant","content":answer}
                    )

                    st.session_state.sources = result["source_documents"]

                    st.rerun()
                else:
                    st.warning("QA Chain is not initialized.")

# RESET
if reset_chat:
    st.session_state.qa_chain = None
    st.session_state.vector_db = None   
    st.session_state.messages = []
    st.session_state.sources = []
    st.session_state.suggested = []
    st.session_state.pdf_bytes = None

st.divider()

# MAIN LAYOUT
left, right = st.columns([2,1])

# CHAT PANEL
with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 Chat with Document")

    st.markdown('<div style="max-height:400px; overflow-y:auto;">', unsafe_allow_html=True)

    for message in st.session_state.messages:
        ...


        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-user">👤 {message["content"]}</div>',
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f'<div class="chat-bot">🤖 {message["content"]}</div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)
    question = st.chat_input("Ask a question about the PDF")
    st.markdown('</div>', unsafe_allow_html=True)

# PROCESS PDF
if uploaded_file:

    if st.session_state.pdf_bytes is None:
        st.session_state.pdf_bytes = uploaded_file.read()
        st.session_state.vector_db = None
        st.session_state.qa_chain = None

    with open("temp.pdf", "wb") as f:
        f.write(st.session_state.pdf_bytes)

    if st.session_state.vector_db is None:
        with st.spinner("⚡ Processing PDF..."):
            st.session_state.vector_db = create_vector_store("temp.pdf")

    if st.session_state.qa_chain is None:
        st.session_state.qa_chain = create_qa_chain(st.session_state.vector_db)

    qa_chain = st.session_state.qa_chain

    if qa_chain is None:
        st.warning("QA Chain is not ready yet. Please wait...")
        st.stop()

    if st.session_state.suggested == [] and qa_chain:

        prompt = """
Generate 4 short questions a student might ask about this document.
Return them as separate lines.
"""
    
        result = qa_chain.invoke({"query": prompt})
        qs = result["result"].split("\n")
        st.session_state.suggested = qs[:4]

        st.rerun()

    if question and qa_chain:

        st.session_state.messages.append(
            {"role":"user","content":question}
        )

        with st.spinner("🤖 Generating answer..."):
            result = qa_chain.invoke({"query": question})

        answer = result["result"]
        sources = result["source_documents"]

        st.session_state.sources = sources

        st.session_state.messages.append(
            {"role":"assistant","content":answer}
        )

        st.rerun()

# RIGHT PANEL
with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📄 PDF Viewer")

    if st.session_state.pdf_bytes:
        pdf_viewer(st.session_state.pdf_bytes, height=500)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📚 Source from PDF")

    if st.session_state.sources:

        pages = list(set(
            [doc.metadata.get("page",0) for doc in st.session_state.sources]
        ))

        for page in pages:
            if st.button(f"📄 Go to Page {page+1}", key=f"page_{page}"):
                st.session_state.current_page = page + 1
                st.rerun()

        st.markdown('<div style="max-height:200px; overflow-y:auto;">', unsafe_allow_html=True)
        
        for doc in st.session_state.sources:

            text = doc.page_content

            if question:
                words = question.split()

                for w in words:
                    text = text.replace(
                        w,
                        f'<span class="highlight">{w}</span>'
                    )

            st.markdown(
                f'<div class="source-box">{text}</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.write("No sources yet.")

    st.markdown('</div>', unsafe_allow_html=True)


# DOWNLOAD CHAT
if download_chat and st.session_state.messages:

    text = ""

    for msg in st.session_state.messages:
        text += msg["role"] + ": " + msg["content"] + "\n\n"

    st.download_button(
        label="Download Chat History",
        data=text,
        file_name="chat_history.txt"
    )