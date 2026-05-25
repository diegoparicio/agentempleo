import json
import os
import logging
from warnings import filters

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

import pickle
import numpy as np
import faiss

import re
import math

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
    # docs = [d for d in docs if d and d.strip()]

    docs = [
        d for d in docs
        if d and d.get("text", "").strip()
    ]

    if not docs:
        raise ValueError("No hay documentos válidos")

    print(f"🧠 Generando embeddings para {len(docs)} chunks...")

    '''    embeddings = np.array(
        [get_embedding(t) for t in docs],
        dtype=np.float32
    )'''

    embeddings = np.array(
        [get_embedding(d["text"]) for d in docs],
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


#--------------------------

def normalize(text):
    return (
        text.lower()
        .replace("á","a")
        .replace("é","e")
        .replace("í","i")
        .replace("ó","o")
        .replace("ú","u")
    )

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
 
    def search(self, query: str, k: int = 10): # k = top_k resultados a devolver
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

        filters = self.extract_filters(question)

        print(filters)

        query = filters.get("query")

        if not isinstance(query, str) or not query.strip():
            query = question

        '''if not query or not query.strip():
            query = question'''

        print("QUERY:", query)

        results = self.search(query)

        print("RESULTS BEFORE FILTER:", len(self.search(query)))

        # tools dinámicas
        if filters["city"]:
            results = self.filter_by_city(
                results,
                filters["city"]
            )

        if filters["max_experience"] is not None:
            results = self.filter_by_experience(
                results,
                filters["max_experience"]
            )

        if filters["modality"]:
            results = self.filter_by_modality(
                results,
                filters["modality"]
            )
        
        '''
        if filters["remote"]:
            results = self.filter_remote(results)
        '''

        print("RESULTS AFTER FILTER:", len(results))
     
        #context = json.dumps(
        #    results,
        #    ensure_ascii=False,
        #    indent=2
        #)
        
        context = "\n\n".join([
            f"""
        Puesto: {r["metadata"]["puesto"]}
        Empresa: {r["metadata"]["empresa"]}
        Lugar: {r["metadata"]["lugar"]}
        Experiencia: {r["metadata"]["experiencia"]}
        Modalidad: {r["metadata"]["modalidad"]}

        Resumen:
        {r["text"]}
        """
            for r in results
        ])
    
        prompt = f"""
    Responde usando las ofertas encontradas.

    No digas "basado en el contexto".

    Contexto:
    {context}

    Pregunta:
    {question}
    """

        chain = self.llm | StrOutputParser()

        return chain.invoke(prompt)

    
    def filter_by_city(self, jobs, city):

        return [
            j for j in jobs
            if city.lower() in j["metadata"]["lugar"].lower()
        ]
    
    def filter_by_modality(self, jobs, modality):

        modality = normalize(modality)

        filtered = []

        for j in jobs:

            job_modality = normalize(j["metadata"]["modalidad"])

            if modality == "remoto":

                if "remoto" in job_modality:
                    filtered.append(j)

            elif modality == "hibrido":

                if "hibrido" in job_modality:
                    filtered.append(j)

            elif modality == "presencial":

                if (
                    "presencial" in job_modality
                    or "oficina" in job_modality
                ):
                    filtered.append(j)

        return filtered
    
    def filter_by_experience(self, jobs, max_years):

        if max_years is None:
            return jobs

        filtered = []

        for j in jobs:

            raw_exp = j["metadata"].get("experiencia", None)

            # 1. limpiar NaN / None
            if raw_exp is None or (isinstance(raw_exp, float) and math.isnan(raw_exp)):
                continue

            raw_exp = str(raw_exp).lower()

            # 2. extraer número real con regex
            match = re.search(r"\d+", raw_exp)

            if not match:
                continue  # si no hay número, no filtramos

            exp_num = int(match.group())

            # 3. lógica del filtro
            if exp_num <= max_years:
                filtered.append(j)

        return filtered

    def extract_filters(self, question: str):

        '''prompt = f"""
    Extrae filtros de búsqueda de empleo.

    Devuelve SOLO JSON válido.

    Formato:

    {{
    "query": "...",
    "city": null,
    "max_experience": null,
    "modality": null
    }}

    Pregunta:
    {question}
    """
    '''
        
        prompt = f"""
    Eres un sistema de extracción de filtros de empleo.

    IMPORTANTE:
    - Si el usuario menciona una ciudad, debes detectarla.
    - Si no hay ciudad, usa null.
    - "query" debe ser un string con palabras clave de empleo (roles, tecnologías o skills).
    - NO incluyas ciudades, experiencia ni modalidad en "query".
    - Si el usuario no da keywords claras, usa una versión corta de la pregunta (quitando palabras genéricas como "ofertas", "trabajo", "empleo").
    - "query" nunca puede ser null ni vacío.

    Devuelve SOLO JSON válido.

    Formato:
    {{
    "query": "...",
    "city": null,
    "max_experience": null,
    "modality": null
    }}

    Pregunta:
    {question}
    """

        chain = self.llm | StrOutputParser()

        response = chain.invoke(prompt)

        print("========== RAW FILTER RESPONSE ==========")
        print(response)

        response = response.replace("```json", "")
        response = response.replace("```", "")
        response = response.strip()

        try:
            parsed = json.loads(response)

            print("========== PARSED FILTERS ==========")
            print(parsed)

            return parsed

        except Exception as e:

            print("========== JSON ERROR ==========")
            print(e)

            return {
                "query": question,
                "city": None,
                "max_experience": None,
                "modality": None
            }