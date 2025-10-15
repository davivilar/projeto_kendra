import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# 1. Extrair texto do PDF
doc = fitz.open("Amazon-Kendra.pdf")
text = ""
for page in doc:
    text += page.get_text()
doc.close()

# 2. Dividir em chunks
def chunk_text(text, max_length=500):
    sentences = text.split('. ')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

chunks = chunk_text(text)

# 3. Gerar embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# 4. Criar índice FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# 5. Salvar índice e chunks
faiss.write_index(index, "kendra_index.faiss")
with open("kendra_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Embeddings gerados e armazenados com sucesso.")