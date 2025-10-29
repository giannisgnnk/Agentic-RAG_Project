from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

faiss_index_path = "faiss_index"
top_k = 5

#load the FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)

query = input("Enter your query: ").strip()

#search the top_k relevant chunks
results = db.similarity_search(query, k=top_k)

retrieved_docs = ""
for rank, doc in enumerate(results, start=1):
#    doc_id = doc.metadata.get("doc_id")
#    headline = doc.metadata.get("headline")
#    chunk_id = doc.metadata.get("chunk_id", "Unknown")
#    print(f"\n{rank}. [Doc ID: {doc_id}] [Chunk ID: {chunk_id}] {headline}")
#    print(doc.page_content[:300], "...\n")  

    #collect the chunks for the prompt
    retrieved_docs += "\n\nDocument " + str(rank) + ":\n" + doc.page_content


prompt = "You are an AI assistant. Answer the following question:\n\nQuestion: " + query + "\n\nUse the following context to answer accurately:\n" + retrieved_docs + "\n\nAnswer:"

#LLM (Ollama)
print("Generating answer with Ollama...\n")
llm = OllamaLLM(model="llama3.2:1b")

answer = llm.invoke(prompt)

print("===============================================")
print("LLM Answer:\n")
print(answer)
print("===============================================")