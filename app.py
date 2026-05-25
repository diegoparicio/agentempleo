import os
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
from rag import GeminiRAG

import os
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Agente Empleo",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Agente IA Búsqueda de Empleo")
st.write("@diegoparicio")

# =========================
# CARGA RAG
# =========================

@st.cache_resource
def load_rag():
    return GeminiRAG()

rag = load_rag()

# =========================
# HISTORIAL
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# CHAT INPUT
# =========================

prompt = st.chat_input("Pregunta sobre las ofertas de empleo capturadas...")

if prompt:

    # usuario
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # asistente
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = rag.ask(prompt)

        if isinstance(response, list):
            response = response[0]["text"]

        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

# =========================
# RESET CHAT
# =========================

if st.button("🧹 Limpiar chat"):
    st.session_state.messages = []
    st.rerun()