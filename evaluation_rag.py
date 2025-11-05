import json
import re
import sys
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# --- 1. Setup ---
faiss_index_path = "faiss_index"
benchmark_file_path = "benchmark.json"
QUESTIONS_TO_PROCESS = 20  # The total number of questions to process from the file

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
llm = OllamaLLM(model="llama3.2:1b")
top_k = 3

# --- 2. Helper Function for Parsing LLM Output ---
def parse_llm_answer(llm_output, options_keys):
    """
    Finds the first occurrence of a valid option key (e.g., 'A', 'B')
    in the LLM's raw output.
    """
    match = re.search(f"[{''.join(options_keys)}]", llm_output.strip().upper())
    if match:
        return match.group(0)
    return None

# --- 3. Load Benchmark (MODIFIED SECTION) ---
questions_list = []
total_loaded = 0

try:
    with open(benchmark_file_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    print(f"Loading benchmark from {benchmark_file_path}...")
    
    # --- NEW: Iterate through all top-level sections ---
    stop_loading = False
    # benchmark_data.items() gives (section_name, section_content)
    # e.g., ("medqa", {"0000": {...}, ...})
    for section_name, section_questions in benchmark_data.items():
        if not isinstance(section_questions, dict):
            print(f"Skipping section '{section_name}': content is not a dictionary.")
            continue
        
        # section_questions.items() gives (q_id, qa_item)
        # e.g., ("0000", {"question": "...", ...})
        for q_id, qa_item in section_questions.items():
            if len(questions_list) >= QUESTIONS_TO_PROCESS:
                stop_loading = True
                break
            
            # Add section and q_id to the item for tracking
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

# --- 4. Evaluation Loop ---
rag_correct = 0
no_context_correct = 0

print("\n========== RAG SYSTEM EVALUATION ==========\n")

# --- MODIFIED: Loop over the new questions_list ---
for i, qa_item in enumerate(questions_list, start=1):
    
    # Extract data from the item
    question = qa_item["question"]
    options = qa_item["options"]
    correct_answer = qa_item["answer"].strip().upper()
    section = qa_item["section"] # Get the section name
    q_id = qa_item["q_id"]       # Get the question ID

    # --- MODIFIED: Print includes section info ---
    print(f"\n--- Question {i}/{total_questions} (Section: {section}, ID: {q_id}) ---")
    print(f"Question: {question[:150]}...")

    # Format the options into a string
    options_str = ""
    for key, value in options.items():
        options_str += f"{key}: {value}\n"
    
    # --- 4a. Answer WITHOUT Context ---
    prompt_no_context = f"""
Answer the following multiple-choice question by providing only the letter (A, B, C, or D) of the correct option.

Question: {question}

Options:
{options_str}
Answer:
"""
    answer_no_context_raw = llm.invoke(prompt_no_context)
    parsed_no_context = parse_llm_answer(answer_no_context_raw, options.keys())

    if parsed_no_context == correct_answer:
        no_context_correct += 1

    # --- 4b. Answer WITH Context (RAG) ---
    results = db.similarity_search(question, k=top_k)
    retrieved_docs = ""
    for rank, doc in enumerate(results, start=1):
        source_file = doc.metadata.get("source_filename", "Unknown File")
        title = doc.metadata.get("title", "No Title")
        chunk_idx = doc.metadata.get("chunk_index", "Unknown")
        
        retrieved_docs += f"\n\n[Doc {rank}] (Title: {title})\n"
        retrieved_docs += f"(File: {source_file}, Chunk: {chunk_idx})\n"        
        retrieved_docs += doc.page_content[:500]

    prompt_with_context = f"""
You are an AI assistant. Use the provided context to answer the following multiple-choice question.
Provide only the letter (A, B, C, or D) of the correct option.

Context:
{retrieved_docs}

Question: {question}

Options:
{options_str}
Answer:
"""
    answer_with_context_raw = llm.invoke(prompt_with_context)
    parsed_with_context = parse_llm_answer(answer_with_context_raw, options.keys())

    if parsed_with_context == correct_answer:
        rag_correct += 1
    
    # --- 4c. Print Result for this Question ---
    print(f"  > RAG Answer:       {parsed_with_context} (Correct: {correct_answer}) -> {'CORRECT' if parsed_with_context == correct_answer else 'WRONG'}")
    print(f"  > No-Context Answer: {parsed_no_context} (Correct: {correct_answer}) -> {'CORRECT' if parsed_no_context == correct_answer else 'WRONG'}")
    if i % 10 == 0 or i == total_questions:
         print(f"  ... (Current RAG Score: {rag_correct}/{i}) ...")

# --- 5. Final Report ---
print("\n========== EVALUATION COMPLETE ==========")
print(f"Total Questions Processed: {total_questions}")

# Calculate percentages
rag_accuracy = (rag_correct / total_questions) * 100 if total_questions > 0 else 0
no_context_accuracy = (no_context_correct / total_questions) * 100 if total_questions > 0 else 0

print("\n--- RAG (With Context) ---")
print(f"Correct Answers: {rag_correct}/{total_questions}")
print(f"Accuracy: {rag_accuracy:.2f}%")

print("\n--- LLM Only (No Context) ---")
print(f"Correct Answers: {no_context_correct}/{total_questions}")
print(f"Accuracy: {no_context_accuracy:.2f}%")

print("\n--- Improvement ---")
print(f"RAG system improved accuracy by: {rag_accuracy - no_context_accuracy:.2f} percentage points.")