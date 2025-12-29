import json
import os
import sys
import re
from pymongo import MongoClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


#configuration
JSON_FOLDER_PATH = "C:/Users/ggian/Desktop/Agentic-RAG_Project/MedRAG/corpus/statpearls/chunk"  
FILES_TO_PROCESS = 100


#setup Embedding Model
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#LLM for Agentic Chunker
llm = OllamaLLM(model="llama3:8b", temperature=0.0)

########### Agentic Chunker Prompt #############
#prompt to make the llm split the text in propositions and return them as a JSON list
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


#Setup a splitter
#we need a splittter to break the very big articles in order to fit in the context window of the llm 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)

#connect to MongoDB
try:
    client = MongoClient("mongodb://localhost:27017")
    db = client["rag_db"]
    collection = db["medical_dataset"]
    client.server_info()
    print(f"Connected to MongoDB")
except Exception as e:
    print(f"Error: Could not connect to MongoDB. {e}")
    sys.exit()

#process the FOLDER of JSONL files
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
                #convert the line from json to a python dict with keys the "id", "title", "content"
                data_object = json.loads(line)
                
                #processing 'content' field 
                article_text = data_object.get("content")

                #processing 'title' and 'id' fields 
                article_title = data_object.get("title", "No Title")
                article_id = data_object.get("id", filename)
                
                #pre-split the article in pieces to fit the LLM
                super_chunks = splitter.split_text(article_text)
                
                final_chunks = [] #store the propositions
                
                #run the Agentic Chunker for every super-chunk
                for super_chunk in super_chunks:
                    try:
                        #build the prompt
                        prompt_value = agentic_chunker_prompt.invoke({"text": super_chunk})

                        #call the llm with the ready prompt 
                        raw_output = llm.invoke(prompt_value)

                        #searches for the propositions (chunks) the llm outputs in its whole raw answer
                        match = re.search(r'\[.*\]', raw_output, re.DOTALL)
                        
                        if not match:
                            raise ValueError("No JSON list found in LLM output.")
                        
                        #extract the clean string without spaces 
                        json_string = match.group(0)
                        
                        #convert to a python list 
                        propositions = json.loads(json_string)

                        if isinstance(propositions, list):
                            final_chunks.extend(propositions)
                        else:
                            print(f"    - Parser did not return a list. Skipping.")

                    except Exception as e:
                        print(f"    - Agentic chunker failed for a piece: {e}. Skipping piece.")
                
                print(f"  > Agentic chunker created {len(final_chunks)} propositions.")


                for i, chunk_text in enumerate(final_chunks):
                
                    vector = embedding_model.embed_query(chunk_text)
                    
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
                            
            except Exception as e:
                print(f"  Error processing a chunk or line: {e}")



#final report
print(f"\n--- Process Complete ---")
print(f"Processed {total_files_processed} files.")
print(f"Finished splitting and storing {total_chunks_saved} chunks in MongoDB.")
client.close()
print("MongoDB connection closed.")