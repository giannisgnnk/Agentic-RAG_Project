import json
import re
import sys
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from unidecode import unidecode
from langchain_ollama import ChatOllama # Changed to ChatOllama for better instruction following
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

#setup
faiss_index_path = "faiss_index"
benchmark_file_path = "benchmark.json"
QUESTIONS_TO_PROCESS = 20

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
#llm = OllamaLLM(model="llama3:8b", temperature=0.0)
llm = ChatOllama(model="llama3:8b", temperature=0.0)
top_k = 3


# --- Pydantic Setup ---
# Define the strict schema for the output
class MultipleChoiceAnswer(BaseModel):
    answer: Literal["A", "B", "C", "D"] = Field(description="The single letter (A, B, C, or D) of the correct option.")

# Create the parser
parser = PydanticOutputParser(pydantic_object=MultipleChoiceAnswer)

# Get the automatic instructions (e.g., "You must return a JSON object...")
format_instructions = parser.get_format_instructions()


# --- NEW: The Cleaner Function ---
def extract_json_str(message):
    """
    Extracts the JSON string from the LLM output, ignoring chatty preambles.
    """
    # Get text content from the AI Message
    if hasattr(message, 'content'):
        text = message.content
    else:
        text = str(message)
        
    # Find the first '{' and the last '}' (The JSON object)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    
    # If no JSON found, return original (parser will raise the standard error)
    return text



# --- Prompt Templates ---
# Template A: No Context
prompt_no_context = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful medical AI. Answer the user's multiple-choice question.\n{format_instructions}"),
    ("human", "Question: {question}\n\nOptions:\n{options}\n\nAnswer:")
])

# Template B: With Context (RAG)
prompt_with_context = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful medical AI. Use the provided context to answer the user's multiple-choice question.\n{format_instructions}"),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nOptions:\n{options}\n\nAnswer:")
])

# Create the Chains (Prompt -> LLM -> Parser)
chain_no_context = prompt_no_context | llm | RunnableLambda(extract_json_str) | parser
chain_with_context = prompt_with_context | llm | RunnableLambda(extract_json_str) | parser


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
    try:
        # Invoke the chain with the variables
        response = chain_no_context.invoke({
            "question": question,
            "options": options_str,
            "format_instructions": format_instructions
        })
        parsed_no_context = response.answer # We get a clean object back!
    except Exception as e:
        print(f"  > No-Context Error: {e}")
        parsed_no_context = None

    if parsed_no_context == correct_answer:
        no_context_correct += 1


    # --- Test B: With Context (RAG) ---
    results = db.similarity_search(question, k=top_k)
    retrieved_docs = ""
    for rank, doc in enumerate(results, start=1):
        title = doc.metadata.get("title", "No Title")
        source_file = doc.metadata.get("source_filename", "Unknown File")
        chunk_idx = doc.metadata.get("chunk_index", "Unknown")
        
        retrieved_docs += f"\n[Doc {rank}] (Title: {title}, File: {source_file})\n"
        retrieved_docs += doc.page_content[:500]

    try:
        # Invoke the RAG chain
        response = chain_with_context.invoke({
            "context": retrieved_docs,
            "question": question,
            "options": options_str,
            "format_instructions": format_instructions
        })
        parsed_with_context = response.answer
    except Exception as e:
        print(f"  > RAG Error: {e}")
        parsed_with_context = None

    if parsed_with_context == correct_answer:
        rag_correct += 1
    
    #print Result for this question
    print(f"  > RAG Answer:       {parsed_with_context} (Correct: {correct_answer}) -> {'CORRECT' if parsed_with_context == correct_answer else 'WRONG'}")
    print(f"  > No-Context Answer: {parsed_no_context} (Correct: {correct_answer}) -> {'CORRECT' if parsed_no_context == correct_answer else 'WRONG'}")
    #if i % 10 == 0 or i == total_questions:
    #     print(f"  ... (Current RAG Score: {rag_correct}/{i}) ...")

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