# Agentic - RAG Architecture 

## 🧠Info
This project presents the implementation of an advanced Retrieval-Augmented Generation (RAG) system optimized for the medical domain, aiming to enhance the accuracy and reasoning capabilities of Large Language Models (LLMs) in answering complex clinical questions.

✅ The system introduces a specialized data processing pipeline that utilizes an "Agentic Chunker" powered by Llama-3 to decompose medical texts from the StatPearls corpus into semantically complete propositions (chunks).
      
✅ These propositions are embedded using the `all-MiniLM-L6-v2` model and indexed in a FAISS vector database for efficient retrieval.
      
✅ The retrieval architecture employs a multi-stage agentic workflow. First, a planning agent decomposes complex user queries into targeted sub-queries. Second, a retrieval module executes these queries and aggregates candidate documents. Third, an LLM-based re-ranking step filters and selects the most relevant context.
       
✅ Finally, a generative model synthesizes the answer using the refined context, strictly adhering to relevance constraints.
      
✅ To validate the system, an automated evaluation framework was developed using random questions from a benchmark.
    
✅ This framework incorporates a "Judge Agent" to quantitatively assess reasoning quality, factuality, and hallucination rates.

## 💻RUN it with that order
1. build_embeddings.py
  
   ▶️ creates the chunks from the medical articles and stores it to MongoDB
4. build_FAISS_index.py
  
   ▶️ calculates the embedding of each chunk and stores it in FAISS db
6. evaluation_rag.py
  
   ▶️ iterates through random questions from benchmark.json to evaluate the system 

## 📌Note
⚠️ Remember to change the paths in the scripts to your local path where the required files are saved.
