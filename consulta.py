from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Carregar índice FAISS e os chunks de texto
index = faiss.read_index("kendra_index.faiss")
with open("kendra_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Carregar modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Loop de perguntas
while True:
    query = input("\n🔍 Digite sua pergunta (ou 'sair' para encerrar): ")
    if query.lower() == "sair":
        break

    # Gerar embedding da pergunta
    query_embedding = model.encode([query])

    # Buscar os 3 chunks mais relevantes
    D, I = index.search(np.array(query_embedding), k=3)

    # Exibir resultados
    print("\n📚 Resultados mais relevantes:")
    for i in I[0]:
        print(f"\n--- Chunk {i} ---")
        print(chunks[i])