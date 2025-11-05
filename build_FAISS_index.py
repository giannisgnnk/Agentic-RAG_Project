import numpy as np
import faiss  # type: ignore
from pymongo import MongoClient
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.docstore import InMemoryDocstore


# --- 1. Connect and Load from MongoDB ---
client = MongoClient("mongodb://localhost:27017")
collection = client["rag_db"]["medical_dataset"]
print("Connected to MongoDB and loading data...")

# Load documents with their embeddings
docs_data = list(collection.find(
    {}, 
    {"_id": 0, "source_filename": 1, "title": 1, "chunk_index": 1, "chunk_embedding": 1, "chunk_text": 1} 
))

print(f"Loaded {len(docs_data)} documents from MongoDB.")

# --- 2. Create LangChain Documents ---
langchain_docs = []
for d in docs_data:
    doc = Document(
        page_content=d["chunk_text"],
        metadata={
            "source_filename": d["source_filename"],
            "title": d.get("title", "No Title"),
            "chunk_index": d["chunk_index"]
        }
    )
    langchain_docs.append(doc)

# --- 3. Prepare Embeddings for FAISS ---
# Load the embeddings from db and create an array
embeddings = np.array([d["chunk_embedding"] for d in docs_data], dtype=np.float32)

# --- 4. Create and Normalize FAISS Index ---
dimension = embeddings.shape[1] # Find the dimension of each embedding vector 

# Normalize embeddings for cosine similarity (which IndexFlatIP uses)
faiss.normalize_L2(embeddings) # More efficient in-place normalization
faiss_index_internal = faiss.IndexFlatIP(dimension)
faiss_index_internal.add(embeddings)

print(f"Created FAISS index with {faiss_index_internal.ntotal} vectors.")

# --- 5. Create LangChain FAISS Wrapper ---
# We need to map the FAISS index (0, 1, 2...) to the LangChain documents
docstore = InMemoryDocstore({i: doc for i, doc in enumerate(langchain_docs)})
index_to_docstore_id = {i: i for i in range(len(langchain_docs))}

faiss_index = FAISS(
    embedding_function=None, # Embeddings are pre-computed
    index=faiss_index_internal,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# --- 6. Save the index locally ---
faiss_index.save_local("faiss_index")
print("\n--- Process Complete ---")
print("Saved FAISS index to folder 'faiss_index'")