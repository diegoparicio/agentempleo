import os
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

import os
from langchain_google_genai import ChatGoogleGenerativeAI


from langchain_core.output_parsers import StrOutputParser

'''def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

def build_rag_index(docs):
    llm = get_llm()'''

# =========================
# CONFIG
# =========================

EMB_PATH = "embeddings.pkl"
DOCS_PATH = "docs.pkl"
INDEX_PATH = "index.faiss"
HASH_PATH = "hash.txt"

LLM_MODEL = "gemini-flash-lite-latest"

# =========================
# MODELOS
# =========================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

'''# LLM (solo generación final)
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0
)'''

# =========================
# EMBEDDINGS
# =========================

def get_embedding(text: str) -> np.ndarray:
    """
    Embedding local (sin API)
    """
    emb = embedding_model.encode(text)
    return np.array(emb, dtype=np.float32)

# =========================
# PERSISTENCIA
# =========================

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# =========================
# BUILD INDEX (OFFLINE)
# =========================

def build_rag_index(docs: list[str]):
    """
    Construcción del índice FAISS (ejecutar en local)
    """
    docs = [d for d in docs if d and d.strip()]

    if not docs:
        raise ValueError("No hay documentos válidos")

    print(f"🧠 Generando embeddings para {len(docs)} chunks...")

    embeddings = np.array(
        [get_embedding(t) for t in docs],
        dtype=np.float32
    )

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Guardar
    save_pickle(embeddings, EMB_PATH)
    save_pickle(docs, DOCS_PATH)
    faiss.write_index(index, INDEX_PATH)

    print("✅ Índice FAISS guardado correctamente")

# =========================
# RAG RUNTIME
# =========================

class GeminiRAG:

    def __init__(self):

        '''self.llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0
        )'''

        self.llm = ChatGoogleGenerativeAI(
            model = LLM_MODEL,
            #model="gemini-1.5-flash",
            temperature=0
        )

        """
        Solo carga índice (NO genera embeddings)
        """

        if not os.path.exists(INDEX_PATH):
            raise ValueError("❌ No existe el índice. Ejecuta build_index.py primero")

        print("📦 Cargando índice FAISS...")

        self.texts = load_pickle(DOCS_PATH)
        self.index = faiss.read_index(INDEX_PATH)

    # =========================
    # SEARCH
    # =========================

    def search(self, query: str, k: int = 5):
        q_emb = get_embedding(query)

        _, idx = self.index.search(
            np.array([q_emb], dtype=np.float32),
            k
        )

        return [self.texts[i] for i in idx[0]]

    # =========================
    # ASK (RAG CORE)
    # =========================

    def ask(self, question: str):

        context = "\n\n".join(self.search(question))

        prompt = f"""
    Usa el contexto si es relevante.
    Si no hay coincidencias exactas, sugiere ofertas similares relacionadas.
    No digas "no lo sé", intenta ser útil.
    No digas "basado en el contexto proporcionado".
    

    Contexto:
    {context}

    Pregunta:
    {question}
    """

        chain = self.llm | StrOutputParser()
        response = chain.invoke(prompt)

        return response