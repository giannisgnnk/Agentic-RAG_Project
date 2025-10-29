from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

faiss_index_path = "faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
llm = OllamaLLM(model="llama3.2:1b")
top_k = 3  

#5 queries for evaluation
queries = [
    "What is autonomous driving?",
    "How are self-driving trucks being developed?",
    "Which companies are leading in robotaxi technology?",
    "What challenges do autonomous vehicles face?",
    "How does AI improve vehicle safety?"
]

print("\n========== RAG SYSTEM EVALUATION ==========\n")

for i, query in enumerate(queries, start=1):
    print("\nQuery", i, ":", query)

    #Answer without context
    prompt_no_context = "Answer the following question briefly:\n" + query + "\nAnswer:"
    answer_no_context = llm.invoke(prompt_no_context)

    #search relevant chunks
    results = db.similarity_search(query, k=top_k)
    retrieved_docs = ""
    for rank, doc in enumerate(results, start=1):
        doc_id = doc.metadata.get("doc_id", "Unknown")
        headline = doc.metadata.get("headline", "Unknown")
        chunk_id = doc.metadata.get("chunk_id", "Unknown")
        retrieved_docs += "\n\n[Doc " + str(rank) + "] (Doc ID: " + str(doc_id) + ", Chunk ID: " + str(chunk_id) + ") " + str(headline) + "\n" + doc.page_content[:500]

    #Answer with context ---
    prompt_with_context = "You are an AI assistant. Use the provided context to answer accurately.\n\nQuestion: " + query + "\n\nContext:" + retrieved_docs + "\n\nAnswer:"
    answer_with_context = llm.invoke(prompt_with_context)

    

    print("\n--- LLM Answer WITHOUT Context ---")
    print(answer_no_context.strip())

    print("\n--- LLM Answer WITH Context (RAG) ---")
    print(answer_with_context.strip())

    print("\n--- Chunks Used ---")
    print(retrieved_docs[:1000], "...\n") 

print("\nEvaluation completed")
