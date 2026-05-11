import streamlit as st
import tempfile, os, sys

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="PDF RAG Explorer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;1,9..144,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0d0d0d;
    color: #e8e0d0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #2a2a2a;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    color: #888 !important;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* Main area headings */
h1 { font-family: 'Fraunces', serif; font-size: 2.6rem; font-weight: 700;
     color: #f0e8d0; letter-spacing: -0.02em; margin-bottom: 0; }
h2 { font-family: 'Fraunces', serif; font-weight: 300; font-style: italic;
     color: #c8b89a; font-size: 1.15rem; margin-top: 0; }
h3 { font-family: 'DM Mono', monospace; font-size: 0.78rem;
     letter-spacing: 0.15em; text-transform: uppercase; color: #d4a853; }

/* Upload box */
[data-testid="stFileUploader"] {
    border: 1px dashed #3a3a3a !important;
    border-radius: 4px;
    background: #141414 !important;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover { border-color: #d4a853 !important; }

/* Buttons */
.stButton > button {
    background: #d4a853 !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.4rem !important;
    font-weight: 500 !important;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.8 !important; }
.stButton > button:disabled { background: #2a2a2a !important; color: #555 !important; }

/* Text inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 2px !important;
    color: #e8e0d0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #d4a853 !important;
    box-shadow: 0 0 0 1px #d4a85322 !important;
}

/* Slider */
.stSlider > div { color: #888 !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #d4a853 !important;
    border-color: #d4a853 !important;
}

/* Chunk cards */
.chunk-card {
    background: #141414;
    border: 1px solid #222;
    border-left: 3px solid #d4a853;
    border-radius: 2px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.82rem;
    line-height: 1.7;
    color: #c8c0b0;
}
.chunk-meta {
    display: flex; gap: 1.2rem; margin-bottom: 0.6rem;
    font-size: 0.68rem; letter-spacing: 0.1em; color: #666;
}
.score-badge {
    display: inline-block;
    background: #1e1a10;
    border: 1px solid #d4a853;
    color: #d4a853;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
}
.score-bar-bg {
    height: 3px; background: #222; border-radius: 2px; margin-top: 0.5rem;
}
.score-bar-fill {
    height: 3px; background: #d4a853; border-radius: 2px;
    transition: width 0.6s ease;
}

/* Answer box */
.answer-box {
    background: #0f0f0f;
    border: 1px solid #d4a853;
    border-radius: 2px;
    padding: 1.4rem 1.6rem;
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    font-weight: 300;
    color: #f0e8d0;
    line-height: 1.9;
    margin-top: 0.5rem;
}

/* Status pills */
.status-ok   { color: #7ec890; font-size: 0.75rem; }
.status-wait { color: #d4a853; font-size: 0.75rem; }
.status-err  { color: #e07070; font-size: 0.75rem; }

/* Dividers */
hr { border-color: #2a2a2a !important; margin: 1.5rem 0 !important; }

/* Progress / spinner colour override */
.stProgress > div > div { background-color: #d4a853 !important; }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Lazy-load heavy modules only once ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_core():
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from chunker import ImprovedUltraFastHybridChunker
    from llm import initialize_llm

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    llm_model, llm_tokenizer = initialize_llm()
    qdrant_client = QdrantClient(":memory:")
    chunker = ImprovedUltraFastHybridChunker(embedding_model=embedding_model)
    return embedding_model, llm_model, llm_tokenizer, qdrant_client, chunker


# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "collection_name": None,
    "bm25_index": None,
    "chunk_count": 0,
    "pdf_name": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>◈ PDF RAG Explorer</h1>", unsafe_allow_html=True)
st.markdown("<h2>Hybrid dense · sparse retrieval with cross-encoder reranking</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◈ Controls")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    st.markdown("### top-k chunks")
    top_k = st.slider("", min_value=1, max_value=15, value=5, label_visibility="collapsed")

    st.markdown("---")
    if st.session_state.pdf_name:
        st.markdown(f"<span class='status-ok'>◉ {st.session_state.pdf_name}</span>", unsafe_allow_html=True)
        st.markdown(f"<span class='status-ok'>{st.session_state.chunk_count} chunks indexed</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-wait'>○ No PDF loaded</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<span style='font-size:0.65rem;color:#444'>BM25 + Qdrant · RRF merge · CrossEncoder rerank</span>", unsafe_allow_html=True)


# ── PDF Processing ────────────────────────────────────────────────────────────
if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
    with st.spinner("Loading models…"):
        embedding_model, llm_model, llm_tokenizer, qdrant_client, chunker = load_rag_core()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### Processing `{uploaded_file.name}`")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        from pdf_utils import process_pdf
        from retriever import BM25Index

        progress_bar = st.progress(0, text="Extracting text…")
        collection_name, chunk_texts = process_pdf(tmp_path, embedding_model, qdrant_client, chunker)
        progress_bar.progress(70, text="Building BM25 index…")

        bm25_index = BM25Index(chunk_texts)
        progress_bar.progress(100, text="Done!")

        st.session_state.collection_name = collection_name
        st.session_state.bm25_index = bm25_index
        st.session_state.chunk_count = len(chunk_texts)
        st.session_state.pdf_name = uploaded_file.name

        progress_bar.empty()
        st.success(f"✓ Indexed **{len(chunk_texts)} chunks** from `{uploaded_file.name}`")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    finally:
        os.unlink(tmp_path)

elif st.session_state.collection_name:
    # Models already loaded silently
    with st.spinner(""):
        embedding_model, llm_model, llm_tokenizer, qdrant_client, chunker = load_rag_core()


# ── Query Interface ───────────────────────────────────────────────────────────
st.markdown("### Ask a question")

query = st.text_input(
    "", placeholder="What does the paper say about…", label_visibility="collapsed"
)

ask_disabled = not st.session_state.collection_name or not query.strip()
ask_btn = st.button("Search & Answer", disabled=ask_disabled)

if ask_btn and query.strip():
    from retriever import hybrid_search
    from llm import generate_answer

    # ── Retrieval ────────────────────────────────────────────────────────────
    with st.spinner("Searching database…"):
        results = hybrid_search(
            query=query,
            collection_name=st.session_state.collection_name,
            embedding_model=embedding_model,
            qdrant_client=qdrant_client,
            bm25_index=st.session_state.bm25_index,
            top_k=top_k,
        )

    # Normalise scores to 0-100% for display
    raw_scores = [r.get("rerank_score", r.get("rrf_score", 0.0)) for r in results]
    min_s, max_s = min(raw_scores), max(raw_scores)
    def pct(s):
        if max_s == min_s: return 100.0
        return round((s - min_s) / (max_s - min_s) * 100, 1)

    # ── Show chunks ──────────────────────────────────────────────────────────
    st.markdown(f"### Retrieved {len(results)} chunks")
    for i, result in enumerate(results, 1):
        payload  = result["payload"]
        score    = raw_scores[i - 1]
        pct_val  = pct(score)
        bar_w    = pct_val

        st.markdown(f"""
<div class="chunk-card">
  <div class="chunk-meta">
    <span>CHUNK #{payload.get('chunk_idx', '?')}</span>
    <span>{payload.get('sentence_count', '?')} sentences</span>
    <span>{payload.get('chunk_length', '?')} chars</span>
    <span class="score-badge">similarity {pct_val}%</span>
  </div>
  <div>{result['text']}</div>
  <div class="score-bar-bg">
    <div class="score-bar-fill" style="width:{bar_w}%"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Generate answer ──────────────────────────────────────────────────────
    st.markdown("### Answer")
    context = "\n\n".join([r["text"] for r in results])

    with st.spinner("Generating answer…"):
        answer = generate_answer(query, context, llm_model, llm_tokenizer)

    st.markdown(f'<div class="answer-box">{answer.strip()}</div>', unsafe_allow_html=True)

elif not st.session_state.collection_name:
    st.markdown("""
<div style='margin-top:3rem;text-align:center;color:#333;font-size:0.8rem;letter-spacing:0.15em;'>
UPLOAD A PDF TO BEGIN
</div>
""", unsafe_allow_html=True)