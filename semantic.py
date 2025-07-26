import os
import json
import argparse
import numpy as np

# --- Dependencies ---
# You will need to install faiss and sentence-transformers:
# pip install faiss-cpu sentence-transformers

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: Dependencies not found. Please install them before running:")
    print("pip install faiss-cpu sentence-transformers")
    exit()


class SemanticSearcher:
    """
    A class to perform semantic search on a pre-built FAISS index.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the Searcher with a sentence transformer model.
        This assumes the model is cached locally (e.g., in a Docker image).
        """
        print(f"Loading sentence transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.index_to_chunk_map = None
        print("Model loaded successfully.")

    def load_index_and_map(self, index_path, map_path):
        """
        Loads the FAISS index and its corresponding metadata map from files.

        Args:
            index_path (str): Path to the .index file.
            map_path (str): Path to the _map.json file.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at: {index_path}")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map file not found at: {map_path}")

        print(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)

        print(f"Loading metadata map from {map_path}")
        with open(map_path, 'r', encoding='utf-8') as f:
            self.index_to_chunk_map = json.load(f)

    def search(self, persona, job, document_name, top_k=5):
        """
        Performs a semantic search based on a persona and job description.

        Args:
            persona (dict): A dictionary describing the user's persona (e.g., {"role": "Project Manager"}).
            job (dict): A dictionary describing the task (e.g., {"task": "Create a project timeline"}).
            document_name (str): The name of the source document for context.
            top_k (int): The number of top results to return.

        Returns:
            list: A list of formatted result dictionaries.
        """
        if self.index is None or self.index_to_chunk_map is None:
            raise RuntimeError("Index and map are not loaded. Please call load_index_and_map() first.")

        # 1. Convert the JSON input to a semantic query
        query_text = f"You are a {persona.get('role', 'user')}. Your task is to {job.get('task', '')}."
        print(f"Formatted Query: {query_text}")

        # 2. Embed the query using the same model
        query_vector = self.model.encode([query_text], convert_to_tensor=False)
        query_vector_np = np.array(query_vector, dtype='float32')

        # 3. Search the FAISS Index
        distances, indices = self.index.search(query_vector_np, top_k)

        # 4. Process and format the results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]

            if idx == -1:
                continue

            # 5. Map Index Back to Chunk
            chunk = self.index_to_chunk_map[idx]

            # 6. Assign Importance Rank
            importance_rank = max(0, 100 - dist)

            # 7. Format the Final Output
            results.append({
                "document": document_name,
                "section_title": chunk.get("section_title"),
                "page_number": chunk.get("page_number"),
                "text_preview": chunk.get("text", "")[:150] + "...",
                "importance_rank": float(round(importance_rank, 2))
            })

        return results


# --- Main execution block for command-line usage ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Perform semantic search on a FAISS index using a JSON query.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the .index and _map.json files.")
    parser.add_argument('--query_json', type=str, required=True, help="Path to the JSON file containing the query, or a raw JSON string.")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top results to retrieve per document.")
    parser.add_argument('--output_file', type=str, default="output.json", help="Path to save the final output JSON file.")

    args = parser.parse_args()
    
    # Load the query from a file or parse it as a string
    query_data = None
    if os.path.exists(args.query_json):
        with open(args.query_json, 'r', encoding='utf-8') as f:
            query_data = json.load(f)
    else:
        try:
            query_data = json.loads(args.query_json)
        except json.JSONDecodeError:
            print(f"Error: --query_json is not a valid file path or a valid JSON string.")
            exit(1)

    # --- UPDATED LOGIC TO HANDLE A LIST OF DOCUMENTS ---
    document_names = query_data.get("document_name")
    if not document_names:
        print("Error: The query JSON must contain a 'document_name' field with a single name or a list of names.")
        exit(1)

    # If it's a single document, convert it to a list to handle it uniformly
    if isinstance(document_names, str):
        document_names = [document_names]

    all_results = []
    searcher = SemanticSearcher()

    # Loop through each document name provided
    for doc_name in document_names:
        print(f"\n--- Searching in document: {doc_name} ---")
        try:
            # Construct the specific file paths for the current document
            base_filename = os.path.splitext(doc_name)[0]
            index_path = os.path.join(args.data_dir, f"{base_filename}.index")
            map_path = os.path.join(args.data_dir, f"{base_filename}_map.json")

            # Load the specific index and map for the requested document
            searcher.load_index_and_map(index_path, map_path)

            search_results = searcher.search(
                persona=query_data.get("persona", {}),
                job=query_data.get("job", {}),
                document_name=doc_name,
                top_k=args.top_k
            )
            all_results.extend(search_results)

        except FileNotFoundError as e:
            print(f"Warning: Could not find index/map for '{doc_name}'. Skipping. Details: {e}")
        except RuntimeError as e:
            print(f"A runtime error occurred for '{doc_name}'. Skipping. Details: {e}")

    # Sort the aggregated results from all documents by importance
    sorted_results = sorted(all_results, key=lambda x: x['importance_rank'], reverse=True)

    # Save the final, sorted JSON object to a file
    print(f"\n--- Aggregated and Sorted Results ---")
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_results, f, indent=2, ensure_ascii=False)
    
    print("Results saved successfully.")



# I've updated the main execution block to implement the new logic.

# **Key Changes:**

# 1.  The command-line arguments `--index_path` and `--map_path` have been replaced with a single `--data_dir`.
# 2.  The script now requires a `document_name` field in your `query.json` file.
# 3.  It uses this `document_name` to build the correct paths to the `.index` and `_map.json` files inside the `--data_dir`.

# **How to Run It Now:**

# Your command would look like this:

# ```bash
# python semantic.py --data_dir embedded --query_json query.json
# ```

# And your `query.json` must contain the target document name:

# ```json
# {
#   "document_name": "your_source_file.pdf",
#   "persona": {
#     "role": "Data Analyst"
#   },
#   "job": {
#     "task": "Find the section on quarterly earnings"
#   }
# }
# ```

# This will correctly load `embedded/your_source_file.index` and `embedded/your_source_file_map.json` to perform the sear