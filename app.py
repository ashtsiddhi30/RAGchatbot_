import streamlit as st
from rag_pipeline import create_vector_store, create_qa_chain
from streamlit_pdf_viewer import pdf_viewer

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="PDF Chatbot",
    layout="wide"
)

# --------------------------------------------------
# MODERN UI STYLING
# --------------------------------------------------

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

.chat-user{
    background:#1f2937;
    padding:12px;
    border-radius:10px;
    margin-bottom:8px;
}

.chat-bot{
    background:#111827;
    padding:12px;
    border-radius:10px;
    border-left:4px solid #22c55e;
    margin-bottom:8px;
}

.source-box{
    background:#0f172a;
    padding:10px;
    border-radius:10px;
    border:1px solid #1e293b;
    margin-bottom:10px;
}

.highlight{
    background-color:#fde68a;
    color:black;
    padding:2px 4px;
    border-radius:4px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📄 PDF Chatbot")

# --------------------------------------------------
# TOP CONTROL BAR
# --------------------------------------------------

top1, top2, top3 = st.columns([1,1,1])

with top1:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

with top2:
    reset_chat = st.button("🔄 Reset Chat")

with top3:
    download_chat = st.button("⬇ Download Answer")

# --------------------------------------------------
# SESSION STATES
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

if "suggested" not in st.session_state:
    st.session_state.suggested = []

# Reset chat
if reset_chat:
    st.session_state.messages = []
    st.session_state.sources = []
    st.session_state.suggested = []

st.divider()

# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------

left, right = st.columns([2,1])

# --------------------------------------------------
# CHAT PANEL
# --------------------------------------------------

with left:

    st.subheader("💬 Chat with Document")

    for message in st.session_state.messages:

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

    question = st.chat_input("Ask a question about the PDF")

# --------------------------------------------------
# PROCESS PDF + QUESTION
# --------------------------------------------------

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    vector_db = create_vector_store("temp.pdf")
    qa_chain = create_qa_chain(vector_db)

    # Generate AI suggested questions
    if not st.session_state.suggested:

        prompt = """
Generate 4 short questions a student might ask about this document.
Return them as separate lines.
"""

        result = qa_chain.invoke({"query": prompt})

        qs = result["result"].split("\n")

        st.session_state.suggested = qs[:4]

    if question:

        st.session_state.messages.append(
            {"role":"user","content":question}
        )

        result = qa_chain.invoke({"query": question})

        answer = result["result"]
        sources = result["source_documents"]

        st.session_state.sources = sources

        st.session_state.messages.append(
            {"role":"assistant","content":answer}
        )

        st.rerun()

# --------------------------------------------------
# RIGHT PANEL
# --------------------------------------------------

with right:

    st.subheader("📄 PDF Viewer")

    if uploaded_file:

        # Store PDF once to prevent reload
        if "pdf_bytes" not in st.session_state:
            st.session_state.pdf_bytes = uploaded_file.read()

        pdf_viewer(st.session_state.pdf_bytes)
        
    st.divider()

    # --------------------------------------------------
    # SOURCE SECTION
    # --------------------------------------------------

    st.subheader("📚 Source from PDF")

    if st.session_state.sources:

        pages = list(set([doc.metadata["page"] for doc in st.session_state.sources]))

        for page in pages:

            if st.button(f"📄 Go to Page {page+1}"):

                st.session_state.current_page = page + 1

                st.rerun()

        st.divider()

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

    else:
        st.write("No sources yet.")

    st.divider()

    # --------------------------------------------------
    # AI SUGGESTED QUESTIONS
    # --------------------------------------------------

    st.subheader("💡 AI Suggested Questions")

    for q in st.session_state.suggested:

        if st.button(q):

            result = qa_chain.invoke({"query": q})

            answer = result["result"]

            st.session_state.messages.append(
                {"role":"user","content":q}
            )

            st.session_state.messages.append(
                {"role":"assistant","content":answer}
            )

            st.session_state.sources = result["source_documents"]

            st.rerun()

# --------------------------------------------------
# DOWNLOAD CHAT
# --------------------------------------------------

if download_chat and st.session_state.messages:

    text = ""

    for msg in st.session_state.messages:
        text += msg["role"] + ": " + msg["content"] + "\n\n"

    st.download_button(
        label="Download Chat History",
        data=text,
        file_name="chat_history.txt"
    )