import csv
import json
import sys
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from unidecode import unidecode
from pydantic import BaseModel, Field
from typing import Literal, List
from langchain_core.output_parsers import PydanticOutputParser
from ollama import chat

# --- Global Setup ---

faiss_index_path = "faiss_index"  # Path to the FAISS vector store
benchmark_file_path = "benchmark.json"  # Path to the evaluation questions
QUESTIONS_TO_PROCESS = 20  

# Initialize the embedding model from HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the FAISS vector database from the local path
db = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)

# Number of documents to retrieve in the final, re-ranked step
top_k = 3


# --- Pydantic Models for Structured Output ---
# These models define the expected JSON structure for the LLM's output, ensuring predictable and parsable results.

# Defines the structure for a multiple-choice answer
class MultipleChoiceAnswer(BaseModel):
    answer: Literal["A", "B", "C", "D"] = Field(description="The single letter (A, B, C, or D) of the correct option.")

# Defines the structure for a single retrieved document chunk
class RetrievedDoc(BaseModel):
    doc: str = Field(description="The retrieved document chunk.")

# Defines the structure for the list of top-k retrieved documents
class AgenticRetrieval(BaseModel):
    retrieved_docs: List[RetrievedDoc] = Field(description="A list of the top-k most relevant document chunks.")

# Defines the structure for the query planning step
class QueryPlan(BaseModel):
    sub_queries: List[str] = Field(description="A list of 2-3 simple, keyword-based search queries to gather necessary information.")


# --- Output Parsers ---
# These parsers use the Pydantic models to validate and parse the LLM's JSON output.
parser = PydanticOutputParser(pydantic_object=MultipleChoiceAnswer)
agentic_retrieval_parser = PydanticOutputParser(pydantic_object=AgenticRetrieval)
planning_parser = PydanticOutputParser(pydantic_object=QueryPlan)


# --- Format Instructions for LLM ---
# These instructions are automatically generated from the Pydantic models and tell the LLM how to format its response as a valid JSON object.
format_instructions = parser.get_format_instructions()
agentic_retrieval_instructions = agentic_retrieval_parser.get_format_instructions()
planning_instructions = planning_parser.get_format_instructions()

# --- Helper Functions ---

def clean_text(text):
    """
    Aggressively converts text to its closest ASCII equivalent using unidecode.
    This helps fix issues with non-standard characters like Cyrillic homoglyphs,
    fancy quotes, and various accents that might appear in the benchmark data.
    """
    if not isinstance(text, str):
        return text
    return unidecode(text)



# --- Load Benchmark Data ---

questions_list = []
total_loaded = 0

try:
    # Open and load the benchmark JSON file
    with open(benchmark_file_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    print(f"Loading benchmark from {benchmark_file_path}...")
    
    stop_loading = False
    # Iterate through the top-level sections of the benchmark (e.g., "medqa")
    for section_name, section_questions in benchmark_data.items():
        if not isinstance(section_questions, dict):
            print(f"Skipping section '{section_name}': content is not a dictionary.")
            continue
        
        # Iterate through each question-answer item in the section
        for q_id, qa_item in section_questions.items():
            # Stop loading if we have reached the desired number of questions
            if len(questions_list) >= QUESTIONS_TO_PROCESS:
                stop_loading = True
                break
            
            # Clean the text of the question and options
            qa_item["question"] = clean_text(qa_item["question"])
            raw_options = qa_item["options"]
            qa_item["options"] = {k: clean_text(v) for k, v in raw_options.items()}

            # Add metadata (section and question ID) to the item for tracking
            qa_item['section'] = section_name
            qa_item['q_id'] = q_id
            questions_list.append(qa_item)
        
        if stop_loading:
            break  # Stop the outer loop as well
            
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


# --- Evaluation Loop ---
# This loop processes each question and evaluates two scenarios:
# 1. Answering with the LLM alone (no context).
# 2. Answering with the RAG system (with context).

rag_correct = 0         # Counter for correct answers from the RAG system
no_context_correct = 0  # Counter for correct answers from the LLM alone

detailed_results = []   # List to store results for each question for later analysis
print("\n========== RAG SYSTEM EVALUATION ==========\n")

# Loop over the loaded questions
for i, qa_item in enumerate(questions_list, start=1):
    
    # Extract data for the current question
    question = qa_item["question"]
    options = qa_item["options"]
    correct_answer = qa_item["answer"].strip().upper()
    section = qa_item["section"]
    q_id = qa_item["q_id"]

    print(f"\n--- Question {i}/{total_questions} (Section: {section}, ID: {q_id}) ---")
    print(f"Question: {question[:150]}...")

    # Format the multiple-choice options into a single string
    options_str = ""
    for key, value in options.items():
        options_str += f"{key}: {value}\n"
    
    # --- Test A: LLM without Context  ---
    # This measures the model's performance on the question without any external information.
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
        # Call the LLM to get an answer
        response = chat(
            model='llama3:8b', 
            messages=messages_no_context, 
            format='json',  # Force the model to output valid JSON
            options={'temperature': 0.0} # Use low temperature for deterministic output
        )
        
        # Parse the JSON string from the response into our Pydantic object
        json_string = response['message']['content']
        pydantic_obj = parser.parse(json_string)
        parsed_no_context = pydantic_obj.answer

    except Exception as e:
        print(f"  > No-Context Error: {e}")
        parsed_no_context = None

    # Check if the baseline answer is correct
    if parsed_no_context == correct_answer:
        no_context_correct += 1


    # --- Test B: RAG with Context ---
    # This is a multi-step process involving planning, searching, and re-ranking.
    
    # PHASE 1: AGENTIC PLANNING
    # The LLM generates a set of simpler, keyword-based search queries from the original complex question.
    plan_messages = [
        {
            'role': 'system',
            'content': f"You are an expert researcher. Break down the user's complex medical question into simple keyword search queries.\n{planning_instructions}"
        },
        {
            'role': 'user',
            'content': f"Question: {question}\nOptions:\n{options_str}"
        }
    ]

    # Always include the original question as a search query
    search_queries = [question] 
    try:
        response = chat(model='llama3:8b', messages=plan_messages, format='json', options={'temperature': 0.0})
        plan = planning_parser.parse(response['message']['content'])
        search_queries.extend(plan.sub_queries)
    except Exception as e:
        # If planning fails, we fall back to using only the original question.
        print(f"  > Planning Failed (Using original query only): {e}")

    # PHASE 2: MULTI-QUERY SEARCH & GATHERING
    # We execute a similarity search for each generated query and collect unique documents.
    unique_docs = {}
    
    for query in search_queries:
        # Fetch a large number of docs for each sub-query to create a rich initial pool
        docs = db.similarity_search(query, k=15) 
        for doc in docs:
            content = doc.page_content
            
            # If the document is new, add it to our dictionary of unique docs
            if content not in unique_docs:
                # Initialize metadata to track which queries retrieved this doc
                if 'source_queries' not in doc.metadata:
                    doc.metadata['source_queries'] = []
                
                doc.metadata['source_queries'].append(query)
                unique_docs[content] = doc
            
            # If the document already exists, just add the current query to its source list
            else:
                existing_doc = unique_docs[content]
                if query not in existing_doc.metadata['source_queries']:
                    existing_doc.metadata['source_queries'].append(query)
    
    # Collect all unique docs and limit the pool to 50 to manage context size
    initial_results = list(unique_docs.values())[:50]
    initial_docs_text = [doc.page_content for doc in initial_results]

    # PHASE 3: AGENTIC RETRIEVAL (RE-RANKING)
    # The LLM acts as a "retriever agent" to select the `top_k` most relevant chunks
    # from the initial pool of documents. This refines the context.
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
        # `results` now contains the top_k most relevant document chunks
        results = pydantic_obj.retrieved_docs

    except Exception as e:
        print(f"  > Agentic Retrieval Error: {e}")
        results = []

    # Format the final, re-ranked documents into a string for the final prompt
    retrieved_docs_str = ""
    for rank, doc in enumerate(results, start=1):
        retrieved_docs_str += f"\n[Doc {rank}]\n{doc.doc[:500]}" # Truncate for context window

    # PHASE 4: ANSWERING WITH CONTEXT
    # The LLM answers the question using the refined context from the RAG process.
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

    # Check if the RAG answer is correct
    if parsed_with_context == correct_answer:
        rag_correct += 1
    
    # Print the comparison for this question
    print(f"  > RAG Answer:       {parsed_with_context} (Correct: {correct_answer}) -> {'CORRECT' if parsed_with_context == correct_answer else 'WRONG'}")
    print(f"  > No-Context Answer: {parsed_no_context} (Correct: {correct_answer}) -> {'CORRECT' if parsed_no_context == correct_answer else 'WRONG'}")

    # --- Ground Truth Comparison & Status ---
    # Determine the outcome: did RAG help, harm, or make no difference?
    status = "UNKNOWN"
    is_llm_correct = (parsed_no_context == correct_answer)
    is_rag_correct = (parsed_with_context == correct_answer)
    
    if is_llm_correct and is_rag_correct:
        status = "BOTH_CORRECT"
    elif not is_llm_correct and not is_rag_correct:
        status = "BOTH_WRONG"
    elif not is_llm_correct and is_rag_correct:
        status = "RAG_IMPROVED"  # RAG helped!
    elif is_llm_correct and not is_rag_correct:
        status = "RAG_WORSENED"  # RAG harmed!

    print(f"  > Status: {status}")
    
    # --- Store Detailed Results for CSV Export ---
    result_row = {
        "id": q_id,
        "question": question[:50] + "...", # Store a preview for reference
        "correct": correct_answer,
        "llm_ans": parsed_no_context,
        "rag_ans": parsed_with_context,
        "status": status,
    }
    detailed_results.append(result_row)

    # --- Detailed Logging for Retrieval Analysis ---
    # This section writes a detailed log for each retrieved document to a JSONL file, allowing for in-depth analysis of the retrieval process.
    try:
        json_filename = "rag_retrieval_logs_detailed1.jsonl"
        
        with open(json_filename, mode='a', encoding='utf-8') as file:
            # Iterate through the final re-ranked documents
            for rank, doc_item in enumerate(results, start=1):
                doc_content = doc_item.doc if hasattr(doc_item, 'doc') else doc_item
                # Find the original document object to access its metadata
                original_doc = unique_docs.get(doc_content)
                
                # Retrieve all the queries that found this document
                all_sources = original_doc.metadata.get('source_queries', []) if original_doc else []
                
                # --- Source Analysis (Original vs. Planner) ---
                # Check if the original question was one of the sources
                found_by_original = any(src.strip() == question.strip() for src in all_sources)
                
                # Find which sub-queries (from the planner) found the doc
                planner_matches = [src for src in all_sources if src.strip() != question.strip()]
                found_by_planner = len(planner_matches) > 0
                
                # --- Prepare Log Entry ---
                # Clean up text for a clean preview in the log
                q_preview = (question.replace('\n', ' ').replace('\r', '')[:100] + "...")
                doc_preview = (doc_content.replace('\n', ' ').replace('\r', '')[:150] + "...")

                log_entry = {
                    "question_preview": q_preview,
                    "rank": rank, # Final rank after re-ranking
                    "source_analysis": {
                        "found_by_original": found_by_original,
                        "found_by_planner": found_by_planner,
                        "planner_queries_count": len(planner_matches)
                    },
                    "match_score": len(all_sources), # Total number of queries that hit this doc
                    "all_matching_queries": all_sources, # Full list of matching queries
                    "doc_preview": doc_preview
                }
                
                # Write the log entry as a new line in the JSONL file
                file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        print(f"Detailed logs saved to {json_filename}")

    except Exception as e:
        print(f"Error saving JSON logs: {e}")


# --- Final Results Summary ---
print("\n========== EVALUATION COMPLETE ==========")
print(f"Total Questions Processed: {total_questions}")

# Calculate final accuracies as percentages
rag_accuracy = (rag_correct / total_questions) * 100 if total_questions > 0 else 0
no_context_accuracy = (no_context_correct / total_questions) * 100 if total_questions > 0 else 0

print("\n--- RAG (With Context) ---")
print(f"Correct Answers: {rag_correct}/{total_questions}")
print(f"Accuracy: {rag_accuracy:.2f}%")

print("\n--- LLM Only (No Context) ---")
print(f"Correct Answers: {no_context_correct}/{total_questions}")
print(f"Accuracy: {no_context_accuracy:.2f}%")

print("\n--- Improvement ---")
improvement = rag_accuracy - no_context_accuracy
print(f"RAG system accuracy change: {improvement:+.2f}%.")

# --- Export Summary Results to CSV ---
output_filename = "rag_evaluation_results1.csv"
if detailed_results:
    keys = detailed_results[0].keys()
    with open(output_filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(detailed_results)
    print(f"\nDetailed results saved to '{output_filename}'. Open this to analyze specific questions.")
else:
    print("\nNo results to save to CSV.")

