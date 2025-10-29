from pymongo import MongoClient
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import numpy as np
import faiss # type: ignore
from langchain.docstore import InMemoryDocstore


client = MongoClient("mongodb://localhost:27017")
collection = client["rag_db"]["chunks_embeddings"]

#load documents with their embeddings
docs_data = list(collection.find({}, {"_id": 0, "doc_id": 1, "chunk_text": 1, "headline": 1, "embedding": 1, "chunk_id": 1}))

#create LangChain Documents
langchain_docs = []
for d in docs_data:
    doc = Document(
        page_content=d["chunk_text"],
        metadata={
            "doc_id": d["doc_id"],
            "headline": d["headline"],
            "chunk_id": d["chunk_id"]
        }
    )
    langchain_docs.append(doc)



#load the embeddings from db and create an array (num_of_chunks, emdedding_dimension)
#FAISS expects to take all the emb in an NumPy array, in order to search by the similarity 
embeddings = np.array([d["embedding"] for d in docs_data], dtype=np.float32)

#create FAISS index
dimension = embeddings.shape[1] #find the dimension of each embedding vector 
#normalize embeddings for cosine similarity
embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
faiss_index_internal = faiss.IndexFlatIP(dimension)
faiss_index_internal.add(embeddings_norm)

#faiss_index_internal = faiss.IndexFlatL2(dimension) #calculate eucleidian distance between the embeddings
#faiss_index_internal.add(embeddings) 

faiss_index = FAISS(
    index=faiss_index_internal,
    docstore=InMemoryDocstore({i: doc for i, doc in enumerate(langchain_docs)}),
    index_to_docstore_id={i: i for i in range(len(langchain_docs))},
    embedding_function=None  #embeddings already exist 
)

#save the faiss index localy
faiss_index.save_local("faiss_index")
print("Saved FAISS index to folder 'faiss_index'")
