import os
from pathlib import Path

import streamlit as st
from fastembed import TextEmbedding
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODELS = {"groq": "openai/gpt-oss-120b", "gemini": "gemini-flash-latest"}
KEY_VARS = {"groq": "GROQ_API_KEY", "gemini": "GOOGLE_API_KEY"}
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60

PROMPT = ChatPromptTemplate.from_template(
    "Ти — асистент, що відповідає на питання користувача лише на основі наданого контексту.\n"
    "Правила:\n"
    "1. Використовуй тільки інформацію з контексту, нічого не вигадуй.\n"
    "2. Якщо в контексті немає потрібної інформації, відповідай: "
    "«У наданих документах немає інформації для відповіді на це питання.»\n"
    "3. Відповідай українською, коротко і точно.\n\n"
    "Контекст:\n{context}\n\n"
    "Питання: {question}\n\n"
    "Відповідь:"
)


class FastEmbedder(Embeddings):
    """Легкі багатомовні ембединги (ONNX, без torch)."""

    def __init__(self, model_name):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return [e.tolist() for e in self.model.embed(list(texts))]

    def embed_query(self, text):
        return next(iter(self.model.embed([text]))).tolist()


def get_api_key(name):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def make_llm(provider, model, api_key):
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=api_key, temperature=0)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)


def load_documents(folder):
    docs = []
    for path in sorted(Path(folder).glob("*")):
        ext = path.suffix.lower()
        if ext in {".txt", ".md"}:
            docs.append(Document(page_content=path.read_text(encoding="utf-8"),
                                 metadata={"source": path.name, "type": ext[1:]}))
        elif ext == ".pdf":
            for i, page in enumerate(PdfReader(str(path)).pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text,
                                         metadata={"source": path.name, "type": "pdf", "page": i + 1}))
    return docs


@st.cache_resource(show_spinner="Завантаження бази знань...")
def load_vectorstore():
    emb = FastEmbedder(EMBED_MODEL)
    if Path(INDEX_DIR).exists():
        return FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
    docs = load_documents(DOCS_DIR)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    store = FAISS.from_documents(chunks, emb)
    store.save_local(INDEX_DIR)
    return store


def to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


st.set_page_config(page_title="RAG-асистент", page_icon="📚")
st.title("RAG-асистент для роботи з документами")
st.caption("Відповіді формуються лише на основі завантажених документів, із зазначенням джерел.")

store = load_vectorstore()
all_sources = sorted({d.metadata["source"] for d in store.docstore._dict.values()})

with st.sidebar:
    st.header("Параметри пошуку")
    k = st.slider("Кількість фрагментів (k)", 1, 8, 4)
    search_type = st.radio("Спосіб пошуку", ["similarity", "mmr"],
                           help="mmr — різноманітніші фрагменти без повторів")
    source = st.selectbox("Фільтр за документом", ["Усі"] + all_sources)
    provider = st.selectbox("Провайдер LLM", ["groq", "gemini"])
    llm_name = st.text_input("Модель LLM", LLM_MODELS[provider])
    st.markdown("**Документи в базі:**")
    for s in all_sources:
        st.markdown(f"- {s}")

api_key = get_api_key(KEY_VARS[provider])
if not api_key:
    st.error(f"Не задано {KEY_VARS[provider]}. Додайте його в Secrets (Streamlit Cloud) "
             "або змінну середовища.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["q"])
    with st.chat_message("assistant"):
        st.write(item["a"])
        with st.expander("Джерела"):
            for s in item["sources"]:
                st.markdown(s)

question = st.chat_input("Поставте питання за документами...")

if question:
    with st.chat_message("user"):
        st.write(question)

    kwargs = {"k": k}
    if search_type == "mmr":
        kwargs["fetch_k"] = max(10, k * 3)
    if source != "Усі":
        kwargs["filter"] = {"source": source}

    retriever = store.as_retriever(search_type=search_type, search_kwargs=kwargs)
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)

    with st.chat_message("assistant"):
        try:
            llm = make_llm(provider, llm_name, api_key)
            answer = to_text(llm.invoke(PROMPT.format(context=context, question=question)).content)
        except Exception as e:
            answer = f"Помилка під час звернення до моделі: {e}"
        st.write(answer)

        sources = []
        for d in docs:
            m = d.metadata
            page = f", стор. {m['page']}" if "page" in m else ""
            sources.append(f"**{m['source']}**{page} (фрагмент {m.get('chunk_id', '?')}): "
                           f"{d.page_content[:200]}...")
        with st.expander("Джерела"):
            for s in sources:
                st.markdown(s)

    st.session_state.history.append({"q": question, "a": answer, "sources": sources})