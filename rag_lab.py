"""
Лабораторна робота № 5. RAG-система для роботи з документами.
Запуск:  python rag_lab.py
У VS Code файл можна виконувати і по блоках (# %%) через "Run Cell".
"""

# 1 Імпорт бібліотек
import os
import time
from getpass import getpass
from pathlib import Path

import pandas as pd
from fastembed import TextEmbedding
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# %% 2 Вибір LLM та API-ключ
# Провайдер: "groq" (ключ GROQ_API_KEY) або "gemini" (ключ GOOGLE_API_KEY)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")
KEY_VARS = {"groq": "GROQ_API_KEY", "gemini": "GOOGLE_API_KEY"}
GROQ_PREFERRED = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]

key_var = KEY_VARS[LLM_PROVIDER]
if not os.environ.get(key_var):
    os.environ[key_var] = getpass(f"Введіть {key_var}: ")


def pick_groq_model(preferred):
    """Обирає першу доступну модель зі списку; якщо жодної немає — показує, що доступно."""
    try:
        from groq import Groq
        ids = {m.id for m in Groq().models.list().data}
        for name in preferred:
            if name in ids:
                return name
        print("Жодна з очікуваних моделей недоступна. Доступні:", sorted(ids))
        print("Впишіть потрібну назву у змінну LLM_MODEL нижче.")
    except Exception as e:
        print("Не вдалося отримати список моделей Groq:", e)
    return preferred[0]


LLM_MODEL = pick_groq_model(GROQ_PREFERRED) if LLM_PROVIDER == "groq" else "gemini-flash-latest"
print("Модель LLM:", LLM_MODEL)

DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60

# %% 3 Підготовка документів
SAMPLE_DOCS = {
"it_security_policy.txt": '''Політика інформаційної безпеки компанії «Альфа-Тех»

1. Паролі. Мінімальна довжина пароля — 14 символів. Пароль потрібно змінювати кожні 90 днів. Забороняється використовувати один пароль для кількох сервісів. Для зберігання паролів дозволено лише корпоративний менеджер паролів «KeyVault».

2. Двофакторна автентифікація. Двофакторна автентифікація є обов'язковою для пошти, VPN та системи Jira. Як другий фактор використовується застосунок-автентифікатор; SMS-коди не допускаються.

3. Інциденти. Про підозрілі листи, а також про втрату чи крадіжку пристрою необхідно повідомити службу безпеки протягом 2 годин за адресою security@alfatech.example або за внутрішнім номером 555.

4. Пристрої. Робочі ноутбуки мають бути зашифровані за допомогою BitLocker. Підключати особисті USB-накопичувачі заборонено.''',

"remote_work.txt": '''Положення про віддалену роботу в компанії «Альфа-Тех»

Віддалена робота дозволяється до трьох днів на тиждень за погодженням з керівником. Обов'язкові офісні дні — вівторок і четвер.

Робочий час: з 9:00 до 18:00 з обідньою перервою 1 година. У віддаленому режимі працівник має бути на зв'язку в Slack протягом усього робочого часу.

Для доступу до внутрішніх ресурсів використовується корпоративний VPN «AlfaVPN». Підключення до VPN обов'язкове під час роботи з клієнтськими даними.

Компенсація інтернету: 400 грн на місяць, виплачується за наявності договору з провайдером. Обладнання: компанія видає ноутбук і монітор; мишку та клавіатуру працівник купує самостійно.''',

"leave_policy.txt": '''Положення про відпустки та лікарняні в компанії «Альфа-Тех»

Щорічна оплачувана відпустка становить 24 календарні дні. Заяву на відпустку слід подати не пізніше ніж за 14 днів до початку через систему «HR-портал». Відпустку можна розділити на частини, але одна з частин має тривати не менше 14 календарних днів.

Лікарняний: працівник має повідомити керівника в день хвороби до 10:00, а електронний лікарняний оформлює лікар.

Додаткова оплачувана відпустка 3 дні надається при одруженні або народженні дитини.

Невикористана відпустка переноситься на наступний рік не більше ніж на 10 днів.''',
}

Path(DOCS_DIR).mkdir(exist_ok=True)
if not any(Path(DOCS_DIR).iterdir()):         
    for name, text in SAMPLE_DOCS.items():
        Path(DOCS_DIR, name).write_text(text, encoding="utf-8")
print("Файли в docs/:", [p.name for p in Path(DOCS_DIR).glob("*")])


# 4 Завантаження документів із метаданими
def load_documents(folder=DOCS_DIR):
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


documents = load_documents()
print("Завантажено документів:", len(documents))
for d in documents:
    print(" ", d.metadata, "| символів:", len(d.page_content))

# %% 5 Розділення на фрагменти (метадані копіюються автоматично)
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
for i, c in enumerate(chunks):
    c.metadata["chunk_id"] = i

print("\nКількість фрагментів:", len(chunks))
print("Приклад фрагмента:", chunks[0].metadata)
print(chunks[0].page_content)


# 6 Векторні представлення та векторне сховище
class FastEmbedder(Embeddings):
    """Багатомовні ембединги (ONNX, без torch)."""

    def __init__(self, model_name):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return [e.tolist() for e in self.model.embed(list(texts))]

    def embed_query(self, text):
        return next(iter(self.model.embed([text]))).tolist()


embeddings = FastEmbedder(EMBED_MODEL)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(INDEX_DIR)
print(f"\nВекторне сховище збережено в {INDEX_DIR}/")

# 7 Завантаження сховища та retriever
vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def show_retrieval(query, retriever):
    print("ЗАПИТ:", query)
    for d in retriever.invoke(query):
        snippet = d.page_content[:110].replace("\n", " ")
        print(f"  [{d.metadata['source']} | фрагмент {d.metadata['chunk_id']}] {snippet}...")
    print()


for q in ["Яка мінімальна довжина пароля?",
          "Скільки днів на тиждень можна працювати віддалено?",
          "За скільки днів подавати заяву на відпустку?"]:
    show_retrieval(q, retriever)

# 8 Промт для мовної моделі
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

# 9 Підключення LLM і відповідь із джерелами
def get_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=LLM_MODEL, temperature=0)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


llm = get_llm()


def to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def ask(question, retriever, show=True):
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)
    answer = to_text(llm.invoke(PROMPT.format(context=context, question=question)).content)
    if show:
        print("ПИТАННЯ:", question)
        print("ВІДПОВІДЬ:", answer)
        print("ДЖЕРЕЛА:")
        for d in docs:
            m = d.metadata
            page = f", стор. {m['page']}" if "page" in m else ""
            print(f"  - {m['source']}{page} (фрагмент {m['chunk_id']})")
        print()
    return answer, docs


retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
ask("Яка мінімальна довжина пароля?", retriever)

# %% 5.2.11 Тестування на різних типах запитів
test_questions = [
    ("Один документ",     "Який менеджер паролів дозволено використовувати в компанії?"),
    ("Один документ",     "Скільки днів оплачуваної відпустки на рік?"),
    ("Кілька документів", "Що потрібно зробити при втраті ноутбука і скільки днів відпустки надається при народженні дитини?"),
    ("Кілька документів", "Чи обов'язкова двофакторна автентифікація і чи потрібен VPN при роботі з клієнтськими даними?"),
    ("Інформації немає",  "Який розмір щорічної премії для співробітників?"),
    ("Поза темою",        "Хто виграв чемпіонат світу з футболу 2018 року?"),
]

for kind, q in test_questions:
    print(f"=== Тип: {kind} ===")
    try:
        ask(q, retriever)
    except Exception as e:
        print("Помилка:", e, "\n")
    time.sleep(7)   # ліміти безкоштовного API

# %% 5.2.12 Аналіз якості: chunk_size × k × тип пошуку
SEC, REM, LEA = "it_security_policy.txt", "remote_work.txt", "leave_policy.txt"
eval_set = [
    ("Яка мінімальна довжина пароля?", {SEC}),
    ("Який менеджер паролів дозволено використовувати?", {SEC}),
    ("Скільки днів на тиждень можна працювати віддалено?", {REM}),
    ("За скільки днів потрібно подати заяву на відпустку?", {LEA}),
    ("Які дні обов'язкові для роботи в офісі і яка компенсація інтернету?", {REM}),
    ("Що робити при втраті ноутбука і скільки днів відпустки при народженні дитини?", {SEC, LEA}),
    ("Чи потрібна двофакторна автентифікація і чи потрібен VPN при роботі з клієнтськими даними?", {SEC, REM}),
]

rows = []
for cs in [200, 400, 800]:
    sp = RecursiveCharacterTextSplitter(chunk_size=cs, chunk_overlap=cs // 6)
    ch = sp.split_documents(documents)
    store = FAISS.from_documents(ch, embeddings)
    for stype in ["similarity", "mmr"]:
        for k in [2, 4]:
            kw = {"k": k}
            if stype == "mmr":
                kw["fetch_k"] = 10
            r = store.as_retriever(search_type=stype, search_kwargs=kw)
            hits = sum(exp <= {d.metadata["source"] for d in r.invoke(q)} for q, exp in eval_set)
            rows.append({"chunk_size": cs, "фрагментів": len(ch), "пошук": stype, "k": k,
                         "успішних запитів": f"{hits}/{len(eval_set)}",
                         "точність": round(hits / len(eval_set), 2)})

results_df = pd.DataFrame(rows).sort_values("точність", ascending=False)
print(results_df.to_string(index=False))

# %% Фільтрація за метаданими
filtered = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"source": "remote_work.txt"}})
show_retrieval("Скільки платять за інтернет?", filtered)