import json
from pymongo import MongoClient

# --- Configuration ---
JSON_FILE_PATH = "statpearls_chunks.json"  
MONGO_CONNECTION_STRING = "mongodb://localhost:27017"
DATABASE_NAME = "rag_db"
COLLECTION_NAME = "medical_dataset"
DOCUMENTS_TO_PROCESS = 100

# --- Connect to MongoDB ---
try:
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    client.server_info() 
    print(f"Connected to MongoDB database '{DATABASE_NAME}'.")
except Exception as e:
    print(f"Error: Could not connect to MongoDB. {e}")
    exit()

# --- Process the JSON Array file ---
total_chunks_saved = 0
print(f"Loading {JSON_FILE_PATH}...")

try:
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f) #[ {obj1}, {obj2}, ... ]

    print(f"File loaded. Total objects found: {len(all_data)}. Processing the first {DOCUMENTS_TO_PROCESS}...")

    for data_object in all_data[:DOCUMENTS_TO_PROCESS]:
        
        doc = {
            "source_id": data_object.get("_id"),
            "source_filename": data_object.get("source_filename"),
            "chunk_index": data_object.get("chunk_index"),
            "chunk_embedding": data_object.get("chunk_embedding"),
            "chunk_text": data_object.get("chunk_text", "").strip()
        }

        #save in mongo
        collection.insert_one(doc)
        total_chunks_saved += 1
            
        
except FileNotFoundError:
    print(f"Error: The file '{JSON_FILE_PATH}' was not found.")

# --- Final Report ---
print(f"\n--- Process Complete ---")
print(f"Finished storing {total_chunks_saved} chunks with embeddings in MongoDB.")
print(f"Database: '{DATABASE_NAME}', Collection: '{COLLECTION_NAME}'")

# Close the MongoDB connection
client.close()
print("MongoDB connection closed.")