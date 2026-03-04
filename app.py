"""
Paracetamol Document QA System - RAG Application (OLLAMA VERSION)
100% LOCAL — No API keys. Auto-detects installed models for embeddings.
Works with only llama3 installed — no nomic-embed-text required.
"""

import streamlit as st
import os
import tempfile
import logging
import requests
import json
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.llms.base import LLM

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ============================================================================
# CONFIG
# ============================================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "llm_model": "llama3:latest",
    "embedding_model": "llama3:latest",
    "embedding_mode": "ollama",        # "ollama" | "huggingface"
    "temperature": 0.5,
    "max_tokens": 1024,
    "retrieval_k": 3,
    "ollama_url": OLLAMA_BASE_URL,
}

KNOWN_EMBED_MODELS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
    "snowflake-arctic-embed",
]

ALL_CHAT_MODELS = [
    "llama3.2","llama3.1","llama3","llama2","mistral","mistral-nemo",
    "gemma3","gemma2","gemma","phi4","phi3","qwen2.5","qwen2",
    "deepseek-r1","deepseek-r1:8b","codellama","neural-chat",
    "openchat","vicuna","orca-mini",
]

# ============================================================================
# OLLAMA HELPERS
# ============================================================================

def check_ollama(url):
    try:
        return requests.get(f"{url}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


def get_installed(url):
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def pull_stream(model, url):
    with requests.post(f"{url}/api/pull", json={"name": model},
                       stream=True, timeout=600) as r:
        for line in r.iter_lines():
            if line:
                try: yield json.loads(line)
                except Exception: pass


def best_embed(installed):
    """Pick best embedding option given what's installed."""
    # 1. Prefer a dedicated embedding model
    for m in installed:
        if any(em in m for em in KNOWN_EMBED_MODELS):
            return m, "ollama"
    # 2. Fall back to first installed chat model
    if installed:
        return installed[0], "ollama"
    # 3. Zero Ollama models — use local HuggingFace
    return "sentence-transformers/all-MiniLM-L6-v2", "huggingface"


# ============================================================================
# CUSTOM EMBEDDINGS — uses ANY Ollama model via /api/embeddings
# ============================================================================

class OllamaEmbeddings(Embeddings):
    """
    Calls Ollama's /api/embeddings endpoint.
    Every installed Ollama model supports this — no special embed model needed.
    """
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model    = model
        self.base_url = base_url

    def _embed_one(self, text: str) -> List[float]:
        r = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=120,
        )
        if r.status_code == 404:
            raise Exception(
                f"Model `{self.model}` not found.\n"
                f"Pull it with:  ollama pull {self.model}"
            )
        r.raise_for_status()
        return r.json()["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for i, t in enumerate(texts):
            results.append(self._embed_one(t))
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


# ============================================================================
# OLLAMA LLM
# ============================================================================

class OllamaLLM(LLM):
    model_name: str = "llama3:latest"
    temperature: float = 0.5
    max_tokens: int = 1024
    base_url: str = "http://localhost:11434"

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, model_name="llama3:latest", temperature=0.5,
                 max_tokens=1024, base_url="http://localhost:11434", **kwargs):
        super().__init__(model_name=model_name, temperature=temperature,
                         max_tokens=max_tokens, base_url=base_url, **kwargs)

    @property
    def _llm_type(self): return "ollama"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=300,
            )
            if r.status_code == 404:
                raise Exception(
                    f"Model `{self.model_name}` not found.\n"
                    f"Run:  ollama pull {self.model_name}"
                )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise Exception(
                f"Cannot connect to Ollama at {self.base_url}.\n"
                "Start it with:  ollama serve"
            )


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def process_document(uploaded_file) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        docs = PyPDFLoader(tmp_path).load()
        if not docs:
            raise ValueError("PDF has no readable pages.")
        return RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", " ", ""],
        ).split_documents(docs)
    except Exception as e:
        raise Exception(f"PDF processing failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def make_embeddings():
    mode  = CONFIG["embedding_mode"]
    model = CONFIG["embedding_model"]
    if mode == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return OllamaEmbeddings(model=model, base_url=CONFIG["ollama_url"])


def create_vector_db(chunks: List[Document]) -> FAISS:
    try:
        return FAISS.from_documents(chunks, make_embeddings())
    except Exception as e:
        raise Exception(f"Embedding error with `{CONFIG['embedding_model']}`: {str(e)}")


def fmt(docs): return "\n\n".join(d.page_content for d in docs)


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="Paracetamol RAG — Ollama",
                   page_icon="🦙", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.main {padding: 2rem;}
h1,h2,h3 {color: #1a1a2e;}
</style>
""", unsafe_allow_html=True)

st.title("🦙 Paracetamol Document QA — Ollama (100% Local)")
st.markdown("**No API keys · No cloud · No cost.** Upload a PDF, ask questions, get answers.")

# ---- SIDEBAR ----

with st.sidebar:

    st.header("🔌 Ollama Connection")
    ollama_url = st.text_input("Host URL:", value=OLLAMA_BASE_URL)
    CONFIG["ollama_url"] = ollama_url

    running   = check_ollama(ollama_url)
    installed = get_installed(ollama_url) if running else []

    if running:
        st.success("✅ Ollama is running")
        if installed:
            st.info("📦 Installed: " + " · ".join(f"`{m}`" for m in installed))
        else:
            st.warning("⚠️ No models pulled yet.")
    else:
        st.error("❌ Ollama not reachable")
        st.code("ollama serve")

    st.divider()

    # ---- Chat Model ----
    st.header("🤖 Chat Model")
    chat_list    = list(dict.fromkeys(installed + ALL_CHAT_MODELS))
    default_chat = chat_list.index(installed[0]) if installed else 0
    sel_chat     = st.selectbox("Model:", chat_list, index=default_chat)

    if sel_chat in installed:
        st.success(f"✅ `{sel_chat}` ready")
    else:
        st.warning(f"⚠️ Not installed")
        if st.button(f"⬇️ Pull `{sel_chat}`", use_container_width=True):
            with st.status(f"Pulling `{sel_chat}`...", expanded=True) as s:
                try:
                    for u in pull_stream(sel_chat, ollama_url):
                        if u.get("status"): st.write(u["status"])
                    s.update(label="✅ Done!", state="complete"); st.rerun()
                except Exception as e:
                    s.update(label=f"❌ {e}", state="error")

    CONFIG["llm_model"] = sel_chat

    st.divider()

    # ---- Embedding Model ----
    st.header("🔢 Embedding Model")

    auto_embed_model, auto_embed_mode = best_embed(installed)

    embed_opts   = []
    embed_labels = []

    for m in installed:
        embed_opts.append(("ollama", m))
        lbl = f"🦙 {m}"
        if any(em in m for em in KNOWN_EMBED_MODELS):
            lbl += " ⭐"
        embed_labels.append(lbl)

    # Always include HuggingFace fallback — works with zero extra pulls
    embed_opts.append(("huggingface", "sentence-transformers/all-MiniLM-L6-v2"))
    embed_labels.append("🤗 HuggingFace all-MiniLM-L6-v2 (no pull needed) ✅")

    # Default to auto-detected best
    def_embed_idx = len(embed_opts) - 1  # default to HuggingFace (always works)
    for i, (mode, model) in enumerate(embed_opts):
        if model == auto_embed_model and mode == auto_embed_mode:
            def_embed_idx = i
            break

    sel_embed_idx = st.selectbox(
        "Embedding source:",
        range(len(embed_labels)),
        format_func=lambda i: embed_labels[i],
        index=def_embed_idx,
        help=(
            "Any installed Ollama model works for embeddings (⭐ = dedicated embed model, best quality). "
            "HuggingFace option needs no extra pull and always works."
        ),
    )

    em_mode, em_model = embed_opts[sel_embed_idx]
    CONFIG["embedding_model"] = em_model
    CONFIG["embedding_mode"]  = em_mode

    if em_mode == "huggingface":
        st.info("🤗 Using local HuggingFace — no Ollama pull required.")
    else:
        st.success(f"✅ `{em_model}` ready for embeddings")

    with st.expander("⬇️ Pull a dedicated embed model (better quality)"):
        for em in KNOWN_EMBED_MODELS:
            c1, c2 = st.columns([3, 1])
            c1.write(f"`{em}`")
            mark = "✅" if any(em in m for m in installed) else "Pull"
            if c2.button(mark, key=f"pull_em_{em}"):
                with st.status(f"Pulling `{em}`...", expanded=True) as s:
                    try:
                        for u in pull_stream(em, ollama_url):
                            if u.get("status"): st.write(u["status"])
                        s.update(label="✅ Done!", state="complete"); st.rerun()
                    except Exception as e:
                        s.update(label=f"❌ {e}", state="error")

    st.divider()

    st.header("🌡️ Generation Settings")
    CONFIG["temperature"] = st.slider("Temperature", 0.0, 1.0, CONFIG["temperature"], 0.1)
    CONFIG["max_tokens"]  = st.slider("Max Tokens", 256, 4096, CONFIG["max_tokens"], 256)
    CONFIG["retrieval_k"] = st.slider("Top-K Chunks", 1, 10, CONFIG["retrieval_k"], 1)
    CONFIG["chunk_size"]  = st.slider("Chunk Size", 200, 2000, CONFIG["chunk_size"], 100)

    st.divider()
    st.info("**100% Local**\n\n• No API keys\n• No billing\n• Data stays on device")
    st.json(CONFIG)

# ---- MAIN ----

if not running:
    st.error("🚨 Ollama not running.\n```bash\nollama serve\n```")
    st.stop()

uploaded_file = st.file_uploader("📁 Upload Paracetamol PDF", type="pdf")

if uploaded_file:
    cache_key = f"{uploaded_file.name}__{em_model}__{em_mode}"

    if st.session_state.get("cache_key") != cache_key:
        embed_lbl = f"🤗 `{em_model}`" if em_mode == "huggingface" else f"🦙 `{em_model}`"
        with st.spinner(f"🔄 Chunking & embedding with {embed_lbl}..."):
            try:
                chunks = process_document(uploaded_file)
                st.session_state.vector_db   = create_vector_db(chunks)
                st.session_state.cache_key   = cache_key
                st.session_state.chunk_count = len(chunks)
                st.success(f"✅ Indexed **{len(chunks)} chunks** · Embeddings: {embed_lbl}")
            except Exception as e:
                st.error(f"❌ {e}")
                st.stop()
    else:
        st.success(
            f"✅ Already indexed **{st.session_state.chunk_count} chunks** "
            f"from `{uploaded_file.name}`"
        )

    query = st.text_input(
        "❓ Ask a question about Paracetamol:",
        placeholder="e.g., What are the side effects of Paracetamol?",
    )

    if query:
        with st.spinner(f"🦙 Running `{CONFIG['llm_model']}`..."):
            try:
                llm = OllamaLLM(
                    model_name=CONFIG["llm_model"],
                    temperature=CONFIG["temperature"],
                    max_tokens=CONFIG["max_tokens"],
                    base_url=CONFIG["ollama_url"],
                )
                retriever = st.session_state.vector_db.as_retriever(
                    search_kwargs={"k": CONFIG["retrieval_k"]}
                )
                prompt = ChatPromptTemplate.from_template("""You are an expert assistant for Paracetamol questions.

Instructions:
1. Use ONLY the provided context to answer
2. If not in context, say you don't know
3. Be accurate and concise

Context:
{context}

Question: {question}

Answer:""")
                chain = (
                    {"context": retriever | fmt, "question": RunnablePassthrough()}
                    | prompt | llm | StrOutputParser()
                )
                answer = chain.invoke(query)
                src    = retriever.invoke(query)

                st.markdown(
                    f"<span style='background:linear-gradient(90deg,#1a1a2e,#16213e);"
                    f"color:white;padding:3px 14px;border-radius:12px;"
                    f"font-size:0.8rem;font-weight:bold;'>"
                    f"🦙 Ollama — {CONFIG['llm_model']}</span>",
                    unsafe_allow_html=True,
                )
                st.subheader("📝 Answer:")
                st.write(answer)

                st.subheader(f"📚 Source Chunks (Top {CONFIG['retrieval_k']}):")
                for i, doc in enumerate(src, 1):
                    with st.expander(f"Chunk {i} — Page {doc.metadata.get('page','?')}"):
                        st.write(doc.page_content)

                st.info(
                    f"ℹ️ {len(src)} chunks · Chat: **{CONFIG['llm_model']}** · "
                    f"Embed: **{CONFIG['embedding_model']}** ({CONFIG['embedding_mode']})"
                )
            except Exception as e:
                st.error(f"❌ {e}")
                with st.expander("📋 Traceback"):
                    import traceback; st.code(traceback.format_exc(), language="python")

    with st.expander("⚙️ Architecture"):
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Chunk Size:** {CONFIG['chunk_size']} · **Overlap:** {CONFIG['chunk_overlap']}")
            st.write(f"**Embedding:** `{CONFIG['embedding_model']}` ({CONFIG['embedding_mode']})")
            st.write("**Vector Store:** FAISS")
        with c2:
            st.write(f"**Engine:** Ollama · **Model:** `{CONFIG['llm_model']}`")
            st.write(f"**Temp:** {CONFIG['temperature']} · **Max Tokens:** {CONFIG['max_tokens']}")

else:
    st.info("👆 Upload a PDF to get started")
    st.markdown("""
### 🚀 Setup

```bash
# Install & start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve

# Pull a chat model (required)
ollama pull llama3

# Optional — dedicated embedding model (better quality)
ollama pull nomic-embed-text

# Python deps
pip install streamlit langchain langchain-community faiss-cpu \\
            pypdf requests python-dotenv sentence-transformers

# Run
streamlit run paracetamol_ollama_rag.py
```

### 💡 Embedding Options
| Option | Needs Pull? | Quality |
|---|---|---|
| 🤗 HuggingFace all-MiniLM | ❌ No | Good |
| 🦙 `llama3:latest` (chat model) | Already installed | Good |
| ⭐ `nomic-embed-text` | ✅ Yes | Best |
""")

st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:gray;'>"
    f"🦙 Ollama RAG · Chat: <strong>{CONFIG['llm_model']}</strong> · "
    f"Embed: <strong>{CONFIG['embedding_model']}</strong> ({CONFIG['embedding_mode']})</div>",
    unsafe_allow_html=True,
)
