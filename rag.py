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

        print("\nQUERY:", query)

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

        print("RESULTS AFTER FILTER 1:", len(results))
        
        if filters["zbe"] is not None:
            results = self.filter_by_zbe(
                results,
                filters["zbe"]
            )

        if filters["max_distance"] is not None:
            results = self.filter_by_distance(
                results,
                filters["max_distance"]
            )

        if filters["solicitud"] is not None:
            results = self.filter_by_solicitud(
                results,
                filters["solicitud"]
            )

        print("RESULTS AFTER FILTER 2:", len(results))

        # ===================================
        # VISUALIZACIÓN DE RESULTADOS (DEBUG)
        # ===================================

        for r in results:

            print(
                f"""
        Puesto: {r['metadata']['puesto']}
        Empresa: {r['metadata']['empresa']}
        Lugar: {r['metadata']['lugar']}
        ZBE: {r['metadata'].get('zbe')}
        Distancia: {r['metadata'].get('distancia')}
        Modalidad: {r['metadata']['modalidad']}
        Experiencia: {r['metadata']['experiencia']}
        Clasificación: {r['metadata']['clasificacion']}
        Solicitud: {r['metadata'].get('solicitud')}
        URL: {r['metadata'].get('url')}
        """
            )

        # Recurso para evitar un RESULTS AFTER FILTER = 0
        ''' Si el filtrado devuelve 0 resultados, volvemos a la búsqueda sin filtros para garantizar
        que el usuario reciba alguna respuesta relevante, aunque no cumpla todos los criterios. '''

        if not results:
            results = self.search(question)
     
        #context = json.dumps(
        #    results,
        #    ensure_ascii=False,
        #    indent=2
        #)
        
        '''        
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
        '''

        context = "\n\n".join([
            f"""
        Puesto: {r["metadata"]["puesto"]}
        Empresa: {r["metadata"]["empresa"]}
        Lugar: {r["metadata"]["lugar"]}
        ZBE: {r["metadata"].get("zbe")}
        Distancia: {r["metadata"].get("distancia")} km
        Modalidad: {r["metadata"]["modalidad"]}
        Experiencia: {r["metadata"]["experiencia"]} años
        Clasificación: {r["metadata"]["clasificacion"]}
        Solicitud: {r["metadata"].get("solicitud")}
        URL: {r["metadata"].get("url")}
        Resumen: {r["metadata"]["resumen"]}
        """
            for r in results
        ])

        # print("\n=========== CONTEXT ===========\n")
        # print(context)

        prompt = f"""
    Eres un asistente de búsqueda de empleo.

    IMPORTANTE:
    MUESTRA TODAS LAS OFERTAS EN EL CONTEXTO, incluso si no cumplen todos los filtros. El usuario quiere ver opciones variadas.
    - No digas "basado en el contexto".
    
    Para cada oferta muestra:
    - Encabezado (letra más grande): Puesto y Empresa
    - URL
    - Lugar
    - Modalidad
    - Experiencia
    - ZBE
    - Distancia
    - Clasificación
    - Estado de solicitud
    - Resumen de la oferta


    Contexto:
    {context}

    Pregunta:
    {question}
    """

        chain = self.llm | StrOutputParser()

        return chain.invoke(prompt)

    # =========================
    # FILTROS
    # =========================

    def filter_by_city(self, jobs, city):

        return [
            j for j in jobs
            if city.lower() in j["metadata"]["lugar"].lower()
        ]
    
    def match_modality(self, job_modality, modality):

        if modality == "remoto":
            return "remoto" in job_modality

        elif modality == "hibrido":
            return "hibrido" in job_modality

        elif modality == "presencial":
            return (
                "presencial" in job_modality
                or "oficina" in job_modality
            )

        return False


    def filter_by_modality(self, jobs, modalities):

        if not modalities:
            return jobs

        modalities = [normalize(m) for m in modalities]

        filtered = []

        for j in jobs:

            job_modality = normalize(j["metadata"]["modalidad"])

            if any(self.match_modality(job_modality, m) for m in modalities):
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
    
    def filter_by_zbe(self, jobs, zbe):

        if zbe is None:
            return jobs

        return [
            j for j in jobs
            if j["metadata"].get("zbe") == zbe
        ]
    
    def filter_by_distance(self, jobs, max_distance):

        if max_distance is None:
            return jobs

        filtered = []

        for j in jobs:

            dist = j["metadata"].get("distancia")

            if dist is None:
                continue

            try:
                dist = float(dist)

                if dist <= max_distance:
                    filtered.append(j)

            except:
                continue

        return filtered
    
    def filter_by_solicitud(self, jobs, solicitud):

        if solicitud is None:
            return jobs

        return [
            j for j in jobs
            if j["metadata"].get("solicitud") == solicitud
        ]
    
    # =========================
    # EXTRACT FILTERS
    # ========================= 

    def extract_filters(self, question: str):
   
        prompt = f"""
    Eres un sistema de extracción de filtros de empleo.

    IMPORTANTE:
    - Si el usuario menciona una ciudad, debes detectarla.
    - Si no hay ciudad, usa null.
    - "query" debe ser un string con palabras clave de empleo (roles, tecnologías o skills).
    - NO incluyas ciudades, experiencia ni modalidad en "query".
    - Si el usuario no da keywords claras, usa una versión corta de la pregunta (quitando palabras genéricas como "ofertas", "trabajo", "empleo").
    - "query" nunca puede ser null ni vacío.

    - "zbe" indica si el usuario quiere ofertas dentro o fuera de zonas de bajas emisiones.

        Valores:
        true → dentro de ZBE
        false → fuera de ZBE
        null → indiferente

        Reglas:
        - Si el usuario dice "evitar ZBE", "fuera de zonas de bajas emisiones" → false
        - Si dice "en ZBE", "centro ciudad con restricciones", "dentro de Madrid central" → true
        - Si no lo menciona → null

    - "distancia" indica la distancia máxima en kilómetros.

        Reglas:
        - Si el usuario dice "a menos de X km" → max_distance = X
        - Si dice "cerca", "próximo", "alrededor" → max_distance = 60 (valor por defecto razonable)
        - Si no menciona distancia → null

    - "solicitud" indica si la oferta está activa o cerrada.

        Valores:
        true → ofertas activas ("se aceptan solicitudes")
        false → ofertas cerradas ("ya no se aceptan solicitudes")

        Reglas:
        - Si el usuario dice "ofertas activas", "vacantes abiertas" → true
        - Si dice "ofertas cerradas", "caducadas" → false
        - Si no se menciona → null

    - "modalidad" debe ser una LISTA de valores o null
        Valores posibles: ["remoto", "hibrido", "presencial"]
        Reglas:
        - si el usuario dice "remoto o híbrido" → ["remoto", "hibrido"]
        - si dice "cualquiera / todas / no se especifica" → ["remoto", "hibrido", "presencial"]

    - Si un filtro reduce demasiado los resultados, es mejor devolver valores más amplios que vaciar la búsqueda.

    Devuelve SOLO JSON válido.

    Formato:
    {{
    "query": "...",
    "city": null,
    "max_experience": null,
    "zbe": null,
    "max_distance": null,
    "solicitud": null,
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
                "zbe": None,
                "max_distance": None,
                "solicitud": None,
                "modality": None
            }