import json
import os
import sys
from pymongo import MongoClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
JSON_FOLDER_PATH = "C:/Users/ggian/Desktop/Agentic-RAG_Project/MedRAG/corpus/statpearls/chunk"  
MONGO_CONNECTION_STRING = "mongodb://localhost:27017"
DATABASE_NAME = "rag_db"
COLLECTION_NAME = "medical_dataset_classic_chunking"
FILES_TO_PROCESS = 100

# --- Setup Splitter ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)

# --- Setup Embedding Model ---
print("Loading embedding model...")
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model loaded.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    sys.exit()

# --- Connect to MongoDB ---
try:
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    client.server_info()
    print(f"Connected to MongoDB database '{DATABASE_NAME}'.")
except Exception as e:
    print(f"Error: Could not connect to MongoDB. {e}")
    sys.exit()

# --- Process the FOLDER of JSONL files ---
total_chunks_saved = 0
total_files_processed = 0

print(f"Processing first {FILES_TO_PROCESS} JSONL files from '{JSON_FOLDER_PATH}'...")


all_files = [f for f in os.listdir(JSON_FOLDER_PATH) if f.endswith('.jsonl')]

if not all_files:
    raise FileNotFoundError(f"No .jsonl files found in the directory: {JSON_FOLDER_PATH}")

for filename in all_files:
    if total_files_processed >= FILES_TO_PROCESS:
        print(f"\nReached file limit of {FILES_TO_PROCESS}.")
        break
    
    total_files_processed += 1
    file_path = os.path.join(JSON_FOLDER_PATH, filename)
    print(f"\nProcessing file {total_files_processed}/{FILES_TO_PROCESS}: {filename}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data_object = json.loads(line)
                
                # --- Processing 'content' field ---
                article_text = data_object.get("content")
                
                if not article_text:
                    print(f"  Skipping line in {filename}: No 'content' field found.")
                    continue

                # --- Processing 'title' and 'id' fields ---
                article_title = data_object.get("title", "No Title")
                article_id = data_object.get("id", filename) # Uses 'id' field

                # Split the article text into chunks
                chunks = splitter.split_text(article_text)
                
                for i, chunk_text in enumerate(chunks):
                    
                    vector = embedding_model.embed_query(chunk_text)
                    
                    # --- Storing all three fields ---
                    doc = {
                        "source_id": article_id,      # From 'id'
                        "source_filename": filename,
                        "title": article_title,       # From 'title'
                        "chunk_index": i,
                        "chunk_embedding": vector,
                        "chunk_text": chunk_text.strip() # From 'content'
                    }
                    
                    collection.insert_one(doc)
                    total_chunks_saved += 1
                            
            except json.JSONDecodeError:
                print(f"  Skipping bad JSON line in {filename}")
            except Exception as e:
                print(f"  Error processing a chunk or line: {e}")



# --- Final Report ---
print(f"\n--- Process Complete ---")
print(f"Processed {total_files_processed} files.")
print(f"Finished splitting and storing {total_chunks_saved} chunks in MongoDB.")
print(f"Database: '{DATABASE_NAME}', Collection: '{COLLECTION_NAME}'")

client.close()
print("MongoDB connection closed.")