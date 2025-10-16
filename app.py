# 📦 Imports
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import openai

# 🔧 Configuração da página
st.set_page_config(page_title="Amazon Kendra PDF Search", layout="wide")
st.title("🔍 Consulta Inteligente ao PDF do Amazon Kendra")

# 📁 Carregar índice FAISS e chunks
try:
    index = faiss.read_index("kendra_index.faiss")
    with open("kendra_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.stop()

# 🔍 Carregar modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# 🔐 Configurar OpenAI
openai.api_key = os.getenv("API_OPENAI")

# 🕘 Inicializar histórico
if "historico" not in st.session_state:
    st.session_state.historico = []

# 📝 Entrada de pergunta
query = st.text_input("Digite sua pergunta:")

# 🧹 Botões
col1, col2 = st.columns([1, 1])
with col1:
    limpar = st.button("🧹 Limpar histórico")
with col2:
    baixar = st.button("📥 Baixar resultados")

# 🧼 Limpar histórico
if limpar:
    st.session_state.historico = []
    st.success("Histórico limpo!")

# 🔎 Processar pergunta
if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)

    resultados = [chunks[i] for i in I[0]]
    st.session_state.historico.append((query, resultados))

    st.subheader("📚 Resultados mais relevantes:")
    for i, texto in enumerate(resultados, 1):
        with st.expander(f"Chunk {i}"):
            st.write(texto)

    # 🤖 Gerar resposta com OpenAI
    if openai.api_key:
        contexto = "\n".join(resultados)
        prompt = f"Com base no seguinte conteúdo, responda à pergunta:\n\n{contexto}\n\nPergunta: {query}"

