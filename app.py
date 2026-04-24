import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.messages import HumanMessage

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind · RAG Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",   # we handle our own panel
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --ink:      #0f0e0d;
    --paper:    #f5f2ee;
    --cream:    #ede9e3;
    --rust:     #c95f2e;
    --rust-dim: #a04a22;
    --muted:    #7a7067;
    --border:   #ddd8d0;
    --user-bg:  #1a1815;
    --user-fg:  #f5f2ee;
    --bot-bg:   #ffffff;
    --shadow:   0 2px 12px rgba(0,0,0,0.08);
    --panel-w:  320px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--paper);
    color: var(--ink);
}

/* Hide Streamlit chrome & native sidebar */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding-top: 1rem !important; padding-bottom: 2rem; max-width: 100% !important; }

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}
.topbar h2 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.6rem;
    letter-spacing: -0.04em;
    margin: 0;
    color: var(--ink);
}
.topbar .accent { color: var(--rust); }
.topbar .sub {
    font-size: 0.8rem;
    color: var(--muted);
    margin: 0;
    font-weight: 300;
}

/* ── Toggle button ── */
div[data-testid="stButton"].toggle-btn button {
    background: var(--cream) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    width: 38px !important; height: 38px !important;
    font-size: 1.1rem !important;
    padding: 0 !important;
    color: var(--ink) !important;
    flex-shrink: 0;
}
div[data-testid="stButton"].toggle-btn button:hover {
    border-color: var(--rust) !important;
    background: var(--border) !important;
}

/* ── Panel (right column acting as sidebar) ── */
.panel-brand { margin-bottom: 1rem; }
.panel-brand h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    margin: 0 0 0.15rem;
    color: var(--ink);
}
.panel-brand p {
    font-size: 0.72rem;
    color: var(--muted);
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.73rem;
    font-weight: 500;
    padding: 4px 11px;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.status-ready   { background: rgba(80,200,120,0.12); color: #2ea85a; border: 1px solid rgba(80,200,120,0.3); }
.status-pending { background: var(--cream); color: var(--muted); border: 1px solid var(--border); }

.panel-footer {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    font-size: 0.72rem;
    color: var(--muted);
    line-height: 1.7;
}

/* ── Chat ── */
.chat-wrap { display: flex; flex-direction: column; gap: 1rem; padding: 0.5rem 0 1.5rem; }

.msg { display: flex; gap: 0.75rem; max-width: 84%; animation: fadeUp 0.25s ease; }
.msg.user  { margin-left: auto; flex-direction: row-reverse; }
.msg.bot   { margin-right: auto; }

.avatar {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; flex-shrink: 0; margin-top: 3px;
}
.avatar.user { background: var(--rust); color: black; font-weight: 700; }
.avatar.bot  { background: var(--cream); border: 1px solid var(--border); }

.bubble {
    padding: 0.8rem 1rem;
    border-radius: 16px;
    font-size: 0.88rem;
    line-height: 1.65;
    box-shadow: var(--shadow);
}
.bubble.user {
    background: var(--user-bg); color: var(--user-fg);
    border-bottom-right-radius: 4px;
}
.bubble.bot {
    background: var(--bot-bg); color: var(--ink);
    border-bottom-left-radius: 4px;
    border: 1px solid var(--border);
}

.empty-state { text-align: center; padding: 4rem 2rem; color: var(--muted); }
.empty-state .icon { font-size: 2.8rem; margin-bottom: 0.75rem; }
.empty-state h3 { font-family: 'Syne', sans-serif; font-size: 1.1rem; color: var(--ink); margin-bottom: 0.4rem; }
.empty-state p  { font-size: 0.83rem; font-weight: 300; }

/* ── Input ── */
.stTextInput input {
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    background: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1rem !important;
    color: black !important;
}
.stTextInput input:focus {
    border-color: var(--rust) !important;
    box-shadow: 0 0 0 3px rgba(201,95,46,0.1) !important;
}

.stButton button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
    transition: all 0.2s !important;
}
.stButton button[kind="primary"] { background: var(--rust) !important; color: white !important; }
.stButton button[kind="primary"]:hover { background: var(--rust-dim) !important; }

.stSpinner > div { border-top-color: var(--rust) !important; }
hr { border-color: var(--border) !important; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages":      [],
        "lc_history":    [],
        "retriever":     None,
        "db_ready":      False,
        "pdf_name":      None,
        "pending_query": "",
        "input_box":     "",
        "panel_open":    True,   # controls our custom panel
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_llm_and_prompt():
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Use the context provided to answer the question. "
         "If you cannot find the answer in the context, say you don't know."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "context: {context}\n\nQuestion: {question}"),
    ])
    return llm, prompt


def ingest_pdf(uploaded_file) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("📄 Reading & chunking PDF…"):
        docs = PyPDFLoader(tmp_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

    with st.spinner("🔢 Embedding with Mistral…"):
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        persist_dir = os.path.join(tempfile.gettempdir(), "chroma_rag_ui")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )

    st.session_state.retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
    )
    st.session_state.db_ready   = True
    st.session_state.pdf_name   = uploaded_file.name
    st.session_state.messages   = []
    st.session_state.lc_history = []
    os.unlink(tmp_path)


def ask(query: str) -> str:
    llm, prompt   = build_llm_and_prompt()
    retriever     = st.session_state.retriever
    history       = st.session_state.lc_history
    docs          = retriever.invoke(query)
    context       = "\n\n".join(d.page_content for d in docs)
    st.session_state.lc_history.append(HumanMessage(query))
    final_prompt  = prompt.invoke({"context": context, "question": query, "history": history})
    response      = llm.invoke(final_prompt)
    st.session_state.lc_history.append(response)
    return response.content


def render_message(role: str, text: str):
    cls        = "user" if role == "user" else "bot"
    avatar     = "U"   if role == "user" else "🧠"
    avatar_cls = "user" if role == "user" else "bot"
    st.markdown(f"""
    <div class="msg {cls}">
        <div class="avatar {avatar_cls}">{avatar}</div>
        <div class="bubble {cls}">{text}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Layout: top bar ───────────────────────────────────────────────────────────
panel_open = st.session_state.panel_open
icon       = "✕" if panel_open else "☰"

top_left, top_right = st.columns([1, 11])
with top_left:
    st.markdown('<div class="toggle-btn">', unsafe_allow_html=True)
    if st.button(icon, key="toggle_panel"):
        st.session_state.panel_open = not st.session_state.panel_open
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
with top_right:
    st.markdown("""
    <div style="padding-top:0.35rem">
        <span style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.5rem;letter-spacing:-0.04em;">
            Ask your <span style="color:var(--rust)">document</span>
        </span>
        <span style="font-size:0.8rem;color:var(--muted);margin-left:0.75rem;font-weight:300;">
            Upload a PDF in the panel, then ask anything.
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr style="margin:0 0 1rem">', unsafe_allow_html=True)


# ── Layout: main + panel columns ─────────────────────────────────────────────
if st.session_state.panel_open:
    chat_col, panel_col = st.columns([3, 1])
else:
    chat_col = st.container()
    panel_col = None


# ── Panel ─────────────────────────────────────────────────────────────────────
if panel_col:
    with panel_col:
        st.markdown("""
        <div class="panel-brand">
            <h3>DocMind</h3>
            <p>RAG · Mistral · Chroma</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.db_ready:
            st.markdown(
                f'<div class="status-badge status-ready">● Ready · {st.session_state.pdf_name}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-badge status-pending">○ No document loaded</div>',
                unsafe_allow_html=True,
            )

        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded and uploaded.name != st.session_state.pdf_name:
            ingest_pdf(uploaded)
            st.success("✅ Document ready!")
            st.rerun()

        if st.session_state.db_ready:
            st.markdown("---")
            if st.button("🗑 Clear conversation", use_container_width=True):
                st.session_state.messages   = []
                st.session_state.lc_history = []
                st.rerun()

        st.markdown("""
        <div class="panel-footer">
            Powered by<br>
            <strong>Mistral AI</strong> · <strong>LangChain</strong> · <strong>ChromaDB</strong>
        </div>
        """, unsafe_allow_html=True)


# ── Chat area ─────────────────────────────────────────────────────────────────
with chat_col:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📄</div>
            <h3>No conversation yet</h3>
            <p>Upload a PDF in the panel and start asking questions.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for role, text in st.session_state.messages:
            render_message(role, text)

    st.markdown('</div>', unsafe_allow_html=True)

    # Input row
    if st.session_state.db_ready:

        def submit():
            text = st.session_state.input_box.strip()
            if text:
                st.session_state.pending_query = text
                st.session_state.input_box     = ""

        col1, col2 = st.columns([6, 1])
        with col1:
            st.text_input(
                "question",
                placeholder="Ask anything about your document…",
                label_visibility="collapsed",
                key="input_box",
                on_change=submit,
            )
        with col2:
            if st.button("Send →", type="primary", use_container_width=True):
                submit()

        if st.session_state.pending_query:
            query = st.session_state.pending_query
            st.session_state.pending_query = ""
            st.session_state.messages.append(("user", query))
            with st.spinner("Thinking…"):
                answer = ask(query)
            st.session_state.messages.append(("bot", answer))
            st.rerun()

    else:
        st.info("☰ Open the panel (top-left) to upload a PDF.", icon="📂")