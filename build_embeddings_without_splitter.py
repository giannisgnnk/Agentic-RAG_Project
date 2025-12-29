import json
import os
import sys
import re
from pymongo import MongoClient
# --- REMOVED: RecursiveCharacterTextSplitter import ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
JSON_FOLDER_PATH = "C:/Users/ggian/Desktop/Agentic-RAG_Project/MedRAG/corpus/statpearls/chunk"  
FILES_TO_PROCESS = 100

# --- Setup Embedding Model ---
print("Loading embedding model...")
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model loaded.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    sys.exit()

# --- Setup LLM for Agentic Chunker ---
llm = ChatOllama(model="llama3:8b", temperature=0.0)

# --- Agentic Chunker Prompt ---
agentic_chunker_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a text splitting robot. You *must* follow these rules:
1. Your task is to split the text into a list of self-contained, semantically complete propositions.
2. A proposition is a single, complete statement or idea.
3. You *must* output *only* a valid JSON list of strings.
4. Do NOT under any circumstances output any other text, preamble, conversational reply, or explanation.
5. If you cannot process the text, output an empty JSON list: []
"""
        ),
        ("human", "Here is the text to split:\n\n{text}"),
    ]
)

# --- Create the Chain ---
agentic_chain = agentic_chunker_prompt | llm

# --- REMOVED: Splitter Setup ---
# splitter = RecursiveCharacterTextSplitter(...)

# --- Connect to MongoDB ---
try:
    client = MongoClient("mongodb://localhost:27017")
    db = client["rag_db"]
    collection = db["medical_dataset_without_splitter"]
    client.server_info()
    print(f"Connected to MongoDB")
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
                
                # Basic check to avoid sending empty text
                if not article_text or len(str(article_text)) < 50:
                    continue

                article_title = data_object.get("title", "No Title")
                article_id = data_object.get("id", filename)
                
                # --- MODIFIED LOGIC: No Pre-Splitting ---
                # We no longer loop through 'super_chunks'.
                # We send the WHOLE article_text directly to the LLM.
                
                final_chunks = [] 
                
                try:
                    # Invoke the chain with the FULL text
                    response = agentic_chain.invoke({"text": article_text})
                    
                    raw_output = response.content
                    
                    # Search for JSON list
                    match = re.search(r'\[.*\]', raw_output, re.DOTALL)
                    
                    if not match:
                        # Fallback for common error cases (empty output or just text)
                        raise ValueError(f"No JSON list found in LLM output. Length of response: {len(raw_output)}")
                    
                    json_string = match.group(0)
                    propositions = json.loads(json_string)

                    if isinstance(propositions, list):
                        final_chunks.extend(propositions)
                    else:
                        print(f"    - Parser did not return a list. Skipping file.")

                except Exception as e:
                    print(f"    - Agentic chunker failed for this article: {e}.")
                    # Optional: Print length to see if it was too big
                    # print(f"      (Article length was {len(article_text)} characters)")
            
                print(f"  > Agentic chunker created {len(final_chunks)} propositions.")

                # Save chunks to MongoDB
                for i, chunk_text in enumerate(final_chunks):
                    
                    # Ensure it's a string
                    if not isinstance(chunk_text, str):
                        continue
                    
                    vector = embedding_model.embed_query(chunk_text)
                    
                    doc = {
                        "source_id": article_id,
                        "source_filename": filename,
                        "title": article_title,
                        "chunk_index": i,
                        "chunk_embedding": vector,
                        "chunk_text": chunk_text.strip()
                    }
                    
                    collection.insert_one(doc)
                    total_chunks_saved += 1
                            
            except Exception as e:
                print(f"  Error processing line: {e}")

# --- Final Report ---
print(f"\n--- Process Complete ---")
print(f"Processed {total_files_processed} files.")
print(f"Finished splitting and storing {total_chunks_saved} chunks in MongoDB.")
client.close()
print("MongoDB connection closed.")