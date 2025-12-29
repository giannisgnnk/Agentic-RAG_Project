import csv
import json
import random
import re
import sys
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from unidecode import unidecode
from pydantic import BaseModel, Field, model_validator
from typing import Literal, List, Any
from langchain_core.output_parsers import PydanticOutputParser
from ollama import chat

# --- Global Setup ---

faiss_index_path = "faiss_index"  # Path to the FAISS vector store
benchmark_file_path = "benchmark.json"  # Path to the evaluation questions
QUESTIONS_TO_PROCESS = 150

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
    justification: str = Field(description="A brief justification for why the chosen answer is correct.")

    @model_validator(mode='before')
    @classmethod
    def preprocess_input(cls, data: Any) -> Any:
        # 1. Ensure data is a dict
        if not isinstance(data, dict):
            return data

        # 2. Unwrap 'properties' if present (Fixes the nesting issue)
        if 'properties' in data and isinstance(data['properties'], dict):
            data = data['properties']

        # 3. Handle Case Sensitivity
        if 'Justification' in data and 'justification' not in data:
            data['justification'] = data.pop('Justification')
        if 'Answer' in data and 'answer' not in data:
            data['answer'] = data.pop('Answer')
        
        # 4. SAFETY NET: Handle "Not Applicable", "None", or invalid answers
        # Αυτό είναι το κομμάτι που αποτρέπει το crash!
        valid_options = ["A", "B", "C", "D"]
        current_ans = str(data.get('answer', '')).strip().upper()
        
        # Προσπάθεια να βρούμε γράμμα αν υπάρχει (π.χ. "Option A" -> "A")
        match = re.search(r'\b([A-D])\b', current_ans)
        
        if match:
            # Αν βρήκαμε γράμμα μέσα στο κείμενο, το κρατάμε
            data['answer'] = match.group(1)
        elif current_ans not in valid_options:
            # Αν η απάντηση είναι τελείως άκυρη (π.χ. "NOT APPLICABLE"), διαλέγουμε τυχαία!
            fallback = random.choice(valid_options)
            print(f"  > [Warning] Invalid LLM Answer: '{current_ans}'. Falling back to random choice: '{fallback}'")
            
            data['answer'] = fallback
            data['justification'] = f"[SYSTEM RECOVERY] Model replied '{current_ans}'. Forced fallback to '{fallback}'. Original justification: " + str(data.get('justification', ''))

        return data

# --- JUDGE AGENT STRUCTURE ---
class JudgeVerdict(BaseModel):
    rag_logic_score: int = Field(description="Score 1-10 for the RAG model's reasoning quality.")
    llm_logic_score: int = Field(description="Score 1-10 for the No-Context LLM's reasoning quality.")
    winner: Literal["RAG", "LLM", "TIE"] = Field(description="Which model had better reasoning?")
    critique: str = Field(description="Brief explanation of why one was better than the other.")

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


def evaluate_reasoning(question, correct_opt, llm_ans, llm_just, rag_ans, rag_just):
    """
    Acts as a Judge Agent to compare the reasoning of No-Context vs RAG.
    """
    judge_prompt = f"""You are an impartial Medical Expert Evaluator (Judge).

    TASK:
    Compare the reasoning (justification) of two AI models answering a medical question.
    
    DATA:
    - Question: {question}
    - Correct Answer: {correct_opt}
    
    - Model A (No Context): Answered {llm_ans}.
      Reasoning: "{llm_just}"
      
    - Model B (RAG + Context): Answered {rag_ans}.
      Reasoning: "{rag_just}"
    
    CRITERIA:
    1. FACTUALITY: Does the reasoning align with the correct medical answer?
    2. COHERENCE: Is the logic sound and step-by-step?
    3. HALLUCINATION CHECK: Did Model B force irrelevant context into the answer?

    OUTPUT:
    Return a JSON with:
    - Scores (1-10) for both.
    - Winner ("RAG", "LLM", or "TIE").
    - A short critique explaining the verdict.
    """

    try:
        response = chat(
            model='llama3:8b',
            messages=[{'role': 'system', 'content': judge_prompt}],
            format=JudgeVerdict.model_json_schema(), # Χρήση του schema για δομημένη έξοδο
            options={'temperature': 0.0}
        )
        return JudgeVerdict.model_validate_json(response['message']['content'])
    except Exception as e:
        print(f"  > Judge Error: {e}")
        # Fallback σε περίπτωση λάθους
        return JudgeVerdict(rag_logic_score=0, llm_logic_score=0, winner="TIE", critique="Error in Judge")


# --- Load Benchmark Data ---

questions_list = []
total_loaded = 0

try:
    # Open and load the benchmark JSON file
    with open(benchmark_file_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    print(f"Loading benchmark from {benchmark_file_path}...")
    
    all_questions = []
    # Iterate through the top-level sections of the benchmark (e.g., "medqa")
    for section_name, section_questions in benchmark_data.items():
        if not isinstance(section_questions, dict):
            print(f"Skipping section '{section_name}': content is not a dictionary.")
            continue
        
        # Iterate through each question-answer item in the section
        for q_id, qa_item in section_questions.items():
            # Clean the text of the question and options
            qa_item["question"] = clean_text(qa_item["question"])
            raw_options = qa_item["options"]
            qa_item["options"] = {k: clean_text(v) for k, v in raw_options.items()}

            # Add metadata (section and question ID) to the item for tracking
            qa_item['section'] = section_name
            qa_item['q_id'] = q_id
            all_questions.append(qa_item)

    # Now, if QUESTIONS_TO_PROCESS is less than total questions, get a random sample
    if len(all_questions) > QUESTIONS_TO_PROCESS:
        questions_list = random.sample(all_questions, QUESTIONS_TO_PROCESS)
    else:
        questions_list = all_questions
            
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
            'content': f"You are a helpful medical AI. Answer the user's multiple-choice question and provide a justification for your choice.\n{format_instructions}"
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
        justification_no_context = pydantic_obj.justification

    except Exception as e:
        print(f"  > No-Context Error: {e}")
        parsed_no_context = None
        justification_no_context = None

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
        docs = db.similarity_search(query, k=7) 
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

    
    # Convert dictionary to list and get top 50 candidates
    all_candidates = list(unique_docs.values())
    initial_results = all_candidates[:50]
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
    
    system_prompt_cot = f"""You are an expert medical AI utilizing retrieval-augmented generation.

TASK:
Answer the multiple-choice question using the provided Context Documents AND your own expert medical knowledge.

CRITICAL INSTRUCTION - RELEVANCE CHECK:
Before answering, evaluate if the Context Documents are ACTUALLY related to the specific medical topic of the question.
- IF Context is Irrelevant (e.g., talks about "Physical Therapy" while the question is about "Dentistry"): **DISCARD the Context completely** and answer solely based on your internal knowledge.
- IF Context is Relevant: Use it to support your answer and clarify specific details.

INSTRUCTIONS:
1. ANALYZE: Read the context documents carefully.
2. CHECK RELEVANCE: Explicitly ask yourself: "Do these documents discuss the exact same pathology/condition as the question?"
3. SYNTHESIZE: Combine valid context facts with your internal medical expertise.
4. THINK STEP-BY-STEP: Compare the medical facts against options A, B, C, and D.
5. ELIMINATE: Rule out options that are incorrect based on the valid evidence.
6. FORCE SELECTION: You MUST select the most plausible option (A, B, C, or D).
   - Do NOT answer "Not Applicable".
   - Do NOT answer "None".
   - If unsure, pick the best educated guess based on general medical principles.

OUTPUT FORMAT:
You must return a valid JSON object strictly following this schema:
{format_instructions}
"""

    messages_with_context = [
        {
            'role': 'system', 
            'content': system_prompt_cot
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
        justification_with_context = pydantic_obj.justification

    except Exception as e:
        print(f"  > RAG Error: {e}")
        parsed_with_context = None
        justification_with_context = None

    # Check if the RAG answer is correct
    if parsed_with_context == correct_answer:
        rag_correct += 1
    
   # Print the comparison for this question
    print(f"  > RAG Answer:       {parsed_with_context} (Correct: {correct_answer}) -> {'CORRECT' if parsed_with_context == correct_answer else 'WRONG'}")
    print(f"  > No-Context Answer: {parsed_no_context} (Correct: {correct_answer}) -> {'CORRECT' if parsed_no_context == correct_answer else 'WRONG'}")

    # --- NEW: JUDGE AGENT CALL ---
    print("  > Judge Agent is deliberating...")
    verdict = evaluate_reasoning(
        question=question,
        correct_opt=correct_answer,
        llm_ans=parsed_no_context,
        llm_just=justification_no_context,
        rag_ans=parsed_with_context,
        rag_just=justification_with_context
    )
    # -----------------------------

    # --- Ground Truth Comparison & Status ---
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
        "question": question[:50] + "...",
        "correct": correct_answer,
        "llm_ans": parsed_no_context,
        "rag_ans": parsed_with_context,
        "status": status,
        "llm_justification": justification_no_context,
        "rag_justification": justification_with_context,
        
        # --- NEW: JUDGE DATA ---
        "judge_winner": verdict.winner,
        "rag_score": verdict.rag_logic_score,
        "llm_score": verdict.llm_logic_score,
        "judge_critique": verdict.critique
    }
    detailed_results.append(result_row)

    # ***JSON FILE*** --- Detailed Logging for Retrieval Analysis --- 
    try:
        json_filename = "rag_retrieval_logs_detailed1.jsonl"
        with open(json_filename, mode='a', encoding='utf-8') as file:
            for rank, doc_item in enumerate(results, start=1):
                doc_content = doc_item.doc if hasattr(doc_item, 'doc') else doc_item
                original_doc = unique_docs.get(doc_content)
                all_sources = original_doc.metadata.get('source_queries', []) if original_doc else []
                
                # --- Source Analysis ---
                found_by_original = any(src.strip() == question.strip() for src in all_sources)
                planner_matches = [src for src in all_sources if src.strip() != question.strip()]
                found_by_planner = len(planner_matches) > 0
                
                # --- FORMATTING ---
                clean_q = question.replace('\n', ' ').replace('\r', '').strip()
                if len(clean_q) > 150:
                    q_preview = clean_q[:147] + "..."
                else:
                    q_preview = clean_q 

                full_doc_content = str(doc_content).replace('\n', ' ').replace('\r', '').strip()

                log_entry = {
                    "question_id": str(q_id),
                    "question_preview": q_preview,
                    "rank": rank,
                    "source_analysis": {
                        "found_by_original": found_by_original,
                        "found_by_planner": found_by_planner,
                        "planner_queries_count": len(planner_matches)
                    },
                    "match_score": len(all_sources),
                    "all_matching_queries": all_sources,
                    "doc_content": full_doc_content 
                }
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


# ***CSV FILE*** --- Export Summary Results ---
output_filename = "rag_evaluation_results1.csv"

if detailed_results:
    # 1. Sort by Priority
    status_priority = {
        "RAG_WORSENED": 0, "RAG_IMPROVED": 1, "BOTH_WRONG": 2, "BOTH_CORRECT": 3
    }
    detailed_results.sort(key=lambda x: status_priority.get(x["status"], 99))

    # 2. Define CSV Columns (UPDATED WITH JUDGE FIELDS)
    fieldnames = [
        "id", "status", "correct", "llm_ans", "rag_ans", 
        "judge_winner", "rag_score", "llm_score",  # <--- NEW
        "question", "judge_critique",              # <--- NEW
        "llm_justification", "rag_justification"
    ]

    # 3. Calculate Max Widths
    max_w_q = 20
    max_w_critique = 20 # <--- NEW

    for row in detailed_results:
        q_text = str(row.get("question", "")).replace("\n", " ").strip()
        c_text = str(row.get("judge_critique", "")).replace("\n", " ").strip() # <--- NEW

        if len(q_text) > max_w_q: max_w_q = len(q_text)
        if len(c_text) > max_w_critique: max_w_critique = len(c_text)

    # Cap max width to avoid massive files
    max_w_q = min(max_w_q + 5, 100)
    max_w_critique = min(max_w_critique + 5, 150)

    # 4. Write to CSV
    with open(output_filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        dict_writer.writeheader()
        
        for row in detailed_results:
            clean_row = row.copy()
            
            def pad_text(text, width):
                if text is None: return "".ljust(width)
                clean = str(text).replace("\n", " ").replace("\r", "").strip()
                return clean.ljust(width)

            # Apply Padding
            clean_row["question"] = pad_text(clean_row["question"], max_w_q)
            clean_row["judge_critique"] = pad_text(clean_row.get("judge_critique", ""), max_w_critique) # <--- NEW
            
            # Align smaller fields
            clean_row["id"] = str(clean_row["id"]).strip().ljust(38)
            clean_row["status"] = str(clean_row["status"]).strip().ljust(15)
            clean_row["judge_winner"] = str(clean_row.get("judge_winner", "")).strip().center(10) # <--- NEW
            
            dict_writer.writerow(clean_row)
            
    print(f"\nResults saved to '{output_filename}'.")
else:
    print("\nNo results to save to CSV.")