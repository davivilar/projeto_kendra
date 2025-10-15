import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Carregar dados
index = faiss.read_index("kendra_index.faiss")
with open("kendra_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Interface
st.set_page_config(page_title="Amazon Kendra PDF Search", layout="wide")
st.title("🔍 Consulta Inteligente ao PDF do Amazon Kendra")

query = st.text_input("Digite sua pergunta:")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)

    st.subheader("📚 Resultados mais relevantes:")
    for i in I[0]:
        st.markdown(f"**Chunk {i}:**")
        st.write(chunks[i])
