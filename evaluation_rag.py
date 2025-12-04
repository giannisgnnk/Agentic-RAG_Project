import json
import sys
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from unidecode import unidecode
from pydantic import BaseModel, Field
from typing import Literal, List
from langchain_core.output_parsers import PydanticOutputParser
from ollama import chat

#setup
faiss_index_path = "faiss_index"
benchmark_file_path = "benchmark.json"
QUESTIONS_TO_PROCESS = 20

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)

top_k = 3


# --- Pydantic Setup ---
# Define the strict schema for the output
class MultipleChoiceAnswer(BaseModel):
    answer: Literal["A", "B", "C", "D"] = Field(description="The single letter (A, B, C, or D) of the correct option.")

class RetrievedDoc(BaseModel):
    doc: str = Field(description="The retrieved document chunk.")

class AgenticRetrieval(BaseModel):
    retrieved_docs: List[RetrievedDoc] = Field(description="A list of the top-k most relevant document chunks.")

# Create the parser
parser = PydanticOutputParser(pydantic_object=MultipleChoiceAnswer)
agentic_retrieval_parser = PydanticOutputParser(pydantic_object=AgenticRetrieval)

# Get the automatic instructions (e.g., "You must return a JSON object...")
format_instructions = parser.get_format_instructions()
agentic_retrieval_instructions = agentic_retrieval_parser.get_format_instructions()



def clean_text(text):
    """
    Aggressively converts text to closest ASCII equivalent.
    Fixes Cyrillic homoglyphs, fancy quotes, accents, etc.
    """
    if not isinstance(text, str):
        return text
    return unidecode(text)



#load Benchmark
questions_list = []
total_loaded = 0

try:
    with open(benchmark_file_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    print(f"Loading benchmark from {benchmark_file_path}...")
    
    #iterate through all top-level sections
    stop_loading = False
    #benchmark_data.items() gives (section_name, section_questions) -> ("medqa", {"0000": {...}, ...})
    for section_name, section_questions in benchmark_data.items():
        if not isinstance(section_questions, dict):
            print(f"Skipping section '{section_name}': content is not a dictionary.")
            continue
        
        #section_questions.items() gives (q_id, qa_item) -> ("0000", {"question": "...", ...})
        for q_id, qa_item in section_questions.items():
            if len(questions_list) >= QUESTIONS_TO_PROCESS:
                stop_loading = True
                break
            
            qa_item["question"] = clean_text(qa_item["question"])
            
            raw_options = qa_item["options"]
            qa_item["options"] = {k: clean_text(v) for k, v in raw_options.items()}

            #add section and q_id to the item for tracking
            qa_item['section'] = section_name
            qa_item['q_id'] = q_id
            questions_list.append(qa_item)
        
        if stop_loading:
            break # Stop the outer loop
            
    total_questions = len(questions_list)
    
    if total_questions == 0:
        print("Error: No questions were loaded. Check file format or QUESTIONS_TO_PROCESS.")
        sys.exit()

    print(f"Loaded {total_questions} questions to process (limit was {QUESTIONS_TO_PROCESS}).\n")

except FileNotFoundError:
    print(f"Error: Benchmark file not found at {benchmark_file_path}")
    sys.exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {benchmark_file_path}")
    sys.exit()




#evaluation loop
rag_correct = 0
no_context_correct = 0

print("\n========== RAG SYSTEM EVALUATION ==========\n")

#loop over the new questions_list
for i, qa_item in enumerate(questions_list, start=1):
    
    #extract data from the item
    question = qa_item["question"]
    options = qa_item["options"]
    correct_answer = qa_item["answer"].strip().upper()
    section = qa_item["section"] # Get the section name
    q_id = qa_item["q_id"]       # Get the question ID

    print(f"\n--- Question {i}/{total_questions} (Section: {section}, ID: {q_id}) ---")
    print(f"Question: {question[:150]}...")

    #format the options into a string
    options_str = ""
    for key, value in options.items():
        options_str += f"{key}: {value}\n"
    
    # --- Test A: No Context ---
    messages_no_context = [
        {
            'role': 'system', 
            'content': f"You are a helpful medical AI. Answer the user's multiple-choice question.\n{format_instructions}"
        },
        {
            'role': 'user', 
            'content': f"Question: {question}\n\nOptions:\n{options_str}\n\nAnswer:"
        }
    ]

    try:
        response = chat(
            model='llama3:8b', 
            messages=messages_no_context, 
            format='json', # Forces valid JSON output
            options={'temperature': 0.0}
        )
        
        #parse the JSON string into our Pydantic object
        json_string = response['message']['content']
        pydantic_obj = parser.parse(json_string)
        parsed_no_context = pydantic_obj.answer

    except Exception as e:
        print(f"  > No-Context Error: {e}")
        parsed_no_context = None

    if parsed_no_context == correct_answer:
        no_context_correct += 1


    # --- Test B: With Context (RAG) ---
    # Agentic Retrieval
    # 1. Perform an initial similarity search to get a smaller set of documents
    initial_results = db.similarity_search(question, k=50)
    initial_docs_text = [doc.page_content for doc in initial_results]

    # 2. Pass the smaller set of documents to the agent for re-ranking
    retrieval_messages = [
        {
            'role': 'system',
            'content': f"You are a retriever agent. Your task is to find the {top_k} most relevant document chunks to the user's question from the provided list of documents.\n{agentic_retrieval_instructions}"
        },
        {
            'role': 'user',
            'content': f"Question: {question}\n\nDocuments:\n{initial_docs_text}"
        }
    ]

    try:
        response = chat(
            model='llama3:8b',
            messages=retrieval_messages,
            format='json',
            options={'temperature': 0.0}
        )
        
        json_string = response['message']['content']
        pydantic_obj = agentic_retrieval_parser.parse(json_string)
        results = pydantic_obj.retrieved_docs

    except Exception as e:
        print(f"  > Agentic Retrieval Error: {e}")
        results = []

    retrieved_docs_str = ""
    for rank, doc in enumerate(results, start=1):
        # We don't have metadata here, so we just show the doc
        retrieved_docs_str += f"\n[Doc {rank}]\n{doc.doc[:500]}"

    messages_with_context = [
        {
            'role': 'system', 
            'content': f"You are a helpful medical AI. Use the provided context to answer the user's multiple-choice question.\n{format_instructions}"
        },
        {
            'role': 'user', 
            'content': f"Context:\n{retrieved_docs_str}\n\nQuestion: {question}\n\nOptions:\n{options_str}\n\nAnswer:"
        }
    ]

    try:
        response = chat(
            model='llama3:8b', 
            messages=messages_with_context, 
            format='json', 
            options={'temperature': 0.0}
        )
        
        json_string = response['message']['content']
        pydantic_obj = parser.parse(json_string)
        parsed_with_context = pydantic_obj.answer

    except Exception as e:
        print(f"  > RAG Error: {e}")
        parsed_with_context = None

    if parsed_with_context == correct_answer:
        rag_correct += 1
    
    #print Result for this question
    print(f"  > RAG Answer:       {parsed_with_context} (Correct: {correct_answer}) -> {'CORRECT' if parsed_with_context == correct_answer else 'WRONG'}")
    print(f"  > No-Context Answer: {parsed_no_context} (Correct: {correct_answer}) -> {'CORRECT' if parsed_no_context == correct_answer else 'WRONG'}")


print("\n========== EVALUATION COMPLETE ==========")
print(f"Total Questions Processed: {total_questions}")

#calculate percentages
rag_accuracy = (rag_correct / total_questions) * 100 
no_context_accuracy = (no_context_correct / total_questions) * 100 

print("\n--- RAG (With Context) ---")
print(f"Correct Answers: {rag_correct}/{total_questions}")
print(f"Accuracy: {rag_accuracy:.2f}%")

print("\n--- LLM Only (No Context) ---")
print(f"Correct Answers: {no_context_correct}/{total_questions}")
print(f"Accuracy: {no_context_accuracy:.2f}%")

print("\n--- Improvement ---")
print(f"RAG system improved accuracy by: {rag_accuracy - no_context_accuracy:.2f} %.")