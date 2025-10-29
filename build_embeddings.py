import pandas as pd
from pymongo import MongoClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#load thw csv
df = pd.read_csv("CNN_Articels_clean.csv")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", "!", "?",",", " "]
)

#convert columns to lists
texts = df["Article text"].tolist()
ids = df["Index"].tolist()
headlines = df["Headline"].tolist()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["rag_db"]
collection = db["chunks_embeddings"]

#create and save embeddings
total_chunks = 0

for i in range(len(texts)):
    text = texts[i]
    doc_id = ids[i]
    headline = headlines[i]

    chunks = splitter.split_text(text)

    for j, chunk_text in enumerate(chunks):
        #create embedding
        vector = embedding_model.embed_query(chunk_text)

        #to be saved in MongoDB
        doc = {
            "doc_id": int(doc_id),
            "headline": headline,
            "chunk_id": j,
            "embedding": vector,
            "chunk_text": chunk_text.strip()
        }

        #save in Mongo
        collection.insert_one(doc)
        total_chunks += 1

print(f"Finished building and storing {total_chunks} chunks with embeddings in MongoDB.")
