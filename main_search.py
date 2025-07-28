# main_search.py
import os
import json
import argparse
from datetime import datetime

# --- Import our new modular components ---
from llm_handler import FastLanguageModel
from embedding_handler import EmbeddingHandler
from search_utils import FaissSearcher, re_rank_results

def run_search(args):
    """Main function to run the semantic search pipeline."""
    try:
        with open(args.query_json, 'r', encoding='utf-8') as f:
            query_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Error: --query_json '{args.query_json}' is not a valid file path.")
        exit(1)

    # Handle both old and new query.json formats
    if "documents" in query_data:
        # New format with documents array
        documents = query_data.get("documents", [])
        document_titles = [doc.get("title", "") for doc in documents]
        document_filenames = [doc.get("filename", "") for doc in documents]
        # Create mapping from title to filename for file operations
        title_to_filename = {doc.get("title", ""): doc.get("filename", "") for doc in documents}
    else:
        # Old format with document_name array
        document_names = query_data.get("document_name", [])
        document_titles = document_names
        document_filenames = document_names
        title_to_filename = {name: name for name in document_names}
    
    persona = query_data.get("persona", {})
    job_to_be_done = query_data.get("job_to_be_done", {})

    if not all([document_titles, persona, job_to_be_done]):
        print("Error: Query JSON must contain documents/document_name, persona, and job_to_be_done.")
        exit(1)
        
    # --- Step 1: Instantiate AI Handlers ---
    # Using TinyLlama GGUF model for fast, offline LLM generation
    gguf_repo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    gguf_filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    llm = FastLanguageModel(model_repo_id=gguf_repo, model_filename=gguf_filename)
    embedder = EmbeddingHandler(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # --- Step 2: Use LLM to Generate Enhanced Query Content ---
    persona_role = persona.get('role', 'user')
    job_task = job_to_be_done.get('task', 'find relevant information')
    
    # Create messages for LLM to generate enhanced query content
    system_prompt = (
    "You are a precise assistant helping HR professionals design better digital workflows. "
    "Given a persona and task, write 7 to 10 short, direct suggestions (1 line each) describing creative ways to fulfill that task. "
    "Each suggestion must be a single line (max 15 words), no bullet points, no numbering, no extra explanation. "
    "Only output the 7–10 lines. Do not repeat persona or task. Do not give headings. "
    "Do not write more than 10 lines. Keep it focused and practical."
    # "Strictly follow all constraints or preferences stated in the task, including dietary, ethical, or functional limits."
)

    # system_prompt = (
    #     "You are an expert semantic search query enhancer. Your task is to expand a user's request "
    #     "into a detailed paragraph that will be used as a query to find the most relevant "
    #     "document sections. Focus on the key goals and priorities implied by the user's profile."
    # )
    
    # user_content = (
    #     f"Expand the following user profile into a detailed paragraph:\n"
    #     f"- Role: {persona_role}\n"
    #     f"- Task: {job_task}\n\n"
    #     f"Expanded Goal Description:"
    # )
    user_content = (
    
    # f"Deconstruct the following user request into its core semantic components based on the provided context. "
    # f"Your output must be a concise, bulleted list under the specified headings.\n\n"
    # f"### Context:\n"
    f"- Persona: {persona_role}\n"
    f"- Available Documents: {', '.join(document_titles)}\n\n"
    f"### User Task:\n"
    f"\"{job_task}\"\n\n"
    f"### Deconstructed Analysis:\n"
    f"- *Primary Goal*: [A 7-8 lined, concise sentence describing the main objective , predict yourself]\n"
    )    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # Generate enhanced query content using LLM
    enhanced_query_text = llm.generate(messages=messages, max_tokens=200)
    
    print("\n" + "="*80)
    print("LLM-Generated Enhanced Query:")
    print(f"   '{enhanced_query_text}'")
    print("="*80 + "\n")

    # --- Step 3: Embed the LLM-Generated Query ---
    query_vector = embedder.encode_and_normalize([enhanced_query_text])
    
    # --- Step 4: Retrieve and Rank Candidates ---
    all_candidates = []
    for doc_title in document_titles:
        print(f"--- Retrieving candidates from: {doc_title} ---")
        try:
            # Get the filename for this title
            filename = title_to_filename.get(doc_title, doc_title)
            base_filename = os.path.splitext(filename)[0]
            index_path = os.path.join(args.data_dir, f"{base_filename}.index")
            map_path = os.path.join(args.data_dir, f"{base_filename}_map.json")
            
            searcher = FaissSearcher(index_path=index_path, map_path=map_path)
            candidates = searcher.retrieve_candidates(query_vector, top_k=30)
            
            for cand in candidates:
                cand['document_name'] = doc_title  # Use title for display
                cand['document_filename'] = filename  # Keep filename for reference
            all_candidates.extend(candidates)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Could not process '{doc_title}'. Skipping. Details: {e}")

    print(f"\n--- Re-ranking {len(all_candidates)} candidates for final importance score ---")
    final_ranked_results = re_rank_results(all_candidates)

    # --- Step 5: Format and Save Final Output ---
    final_output = {
        "metadata": {
            "input_documents": document_filenames,  # Use filenames in metadata
            "persona": persona_role,  # Direct text, not nested object
            "job_to_be_done": job_task,  # Direct text, not nested object
            "processing_timestamp": datetime.now().isoformat(),
            "enhanced_query_used": enhanced_query_text
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }
    
    seen_sections = set()
    final_top_n = []
    for res in final_ranked_results:
        section_key = (res.get("document_name"), res.get("section_title"))
        if section_key not in seen_sections:
            final_top_n.append(res)
            seen_sections.add(section_key)
        # Limit to top 10 results
        if len(final_top_n) >= 10:
            break
            
    for i, res in enumerate(final_top_n):
        # Ensure document name has .pdf extension for display
        doc_name = res.get("document_name", "")
        if doc_name and not doc_name.lower().endswith('.pdf'):
            doc_name = f"{doc_name}.pdf"
        
        final_output["extracted_sections"].append({
            "document": doc_name, 
            "section_title": res.get("section_title"),
            "importance_rank": i + 1, 
            "page_number": res.get("page_number")
        })
        final_output["subsection_analysis"].append({
            "document": doc_name, 
            "refined_text": res.get("text"),
            "page_number": res.get("page_number")
        })

    print(f"\nSaving top {len(final_output['extracted_sections'])} results to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    
    print("Ranking complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Perform modular semantic search using a generative LLM and an embedding model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, required=True, help="Directory with .index and _map.json files.")
    parser.add_argument('--query_json', type=str, required=True, help="Path to the JSON query file.")
    parser.add_argument('--top_n', type=int, default=5, help="Final number of top results to save.")
    parser.add_argument('--output_file', type=str, default="ranked_output.json", help="Path to save the final output.")

    args = parser.parse_args()
    run_search(args)