import os
import json
import argparse
import glob
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


class EmbeddingIndexer:
    """
    A class to create a FAISS index from hierarchical text chunks.
    It handles model loading and the indexing process.
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initializes the Indexer with a sentence transformer model.
        """
        print(f"Loading sentence transformer model: {model_name}...")
        
        # Set environment variables to ensure offline mode
        import os
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        # Load the model (should use cached version)
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.")

    def create_index_from_chunks(self, chunks: list):
        """
        Creates a FAISS index and a metadata mapping from the given chunks.
        """
        if not chunks:
            return None, []

        # 1. Create unified text for each chunk for better embedding context
        unified_texts = [f"{chunk.get('section_title', '')}: {chunk.get('text', '')}" for chunk in chunks]

        # 2. Generate embeddings
        print(f"Generating embeddings for {len(unified_texts)} chunks...")
        embeddings = self.model.encode(unified_texts, convert_to_tensor=False, show_progress_bar=True)
        
        if not hasattr(embeddings, 'size') or embeddings.size == 0:
            print("Warning: No embeddings were generated. Cannot create index.")
            return None, []
            
        # --- IMPROVEMENT: L2-normalize the vectors ---
        # This is a critical step for using IndexFlatL2 to perform cosine similarity searches.
        # The semantic search script normalizes the query, so the index must be normalized too.
        faiss.normalize_L2(embeddings)
        
        embedding_dimension = embeddings.shape[1]

        # 3. Build the FAISS index
        print("Building FAISS index...")
        # IndexFlatL2 is perfect for exact search with cosine similarity on normalized vectors.
        faiss_index = faiss.IndexFlatL2(embedding_dimension)
        
        # Add the normalized vectors to the index
        faiss_index.add(np.array(embeddings, dtype='float32'))
        
        print(f"FAISS index built successfully with {faiss_index.ntotal} vectors.")

        # 4. The map is the list of chunks, as their list index corresponds to the FAISS ID.
        index_to_chunk_map = chunks

        return faiss_index, index_to_chunk_map


def process_chunked_directory(input_dir: str, output_dir: str):
    """
    Processes all chunked JSON files, creates a FAISS index and metadata
    map for each, and saves them to an output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    
    if not json_files:
        print(f"Error: No JSON files found in the input directory: {input_dir}")
        return

    print(f"Found {len(json_files)} chunked JSON files to process.")
    
    # Initialize the indexer once to avoid reloading the heavy model
    indexer = EmbeddingIndexer()

    for file_path in json_files:
        base_filename = os.path.basename(file_path)
        filename_no_ext = os.path.splitext(base_filename)[0]
        print(f"\n--- Processing {base_filename} ---")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                hierarchical_chunks = json.load(f)

            if not hierarchical_chunks:
                print("JSON file is empty. Skipping.")
                continue

            # --- Create Index and Metadata Map ---
            faiss_index, index_to_chunk_map = indexer.create_index_from_chunks(hierarchical_chunks)
            
            if faiss_index:
                # Define paths for the output files
                index_path = os.path.join(output_dir, f"{filename_no_ext}.index")
                metadata_path = os.path.join(output_dir, f"{filename_no_ext}_map.json")
                
                # Save the FAISS index
                faiss.write_index(faiss_index, index_path)
                
                # Save the metadata map
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(index_to_chunk_map, f, indent=2, ensure_ascii=False)
                
                print(f"Successfully saved FAISS index to: {index_path}")
                print(f"Successfully saved metadata map to: {metadata_path}")

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a searchable FAISS index from pre-chunked JSON files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_dir', type=str, required=True, 
                        help="Directory containing the pre-chunked JSON files.")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Directory where the final FAISS index and metadata map will be saved.")
    args = parser.parse_args()

    # --- How to Run ---
    # This script assumes you have already run a chunking/parsing script.
    # python embedding.py --input_dir chunked --output_dir embedded
    process_chunked_directory(args.input_dir, args.output_dir)

# import os
# import json
# import argparse
# import glob
# import numpy as np

# # --- Dependencies ---
# # You will need to install faiss and sentence-transformers:
# # pip install faiss-cpu sentence-transformers

# try:
#     import faiss
#     from sentence_transformers import SentenceTransformer
# except ImportError:
#     print("Error: Dependencies not found. Please install them before running:")
#     print("pip install faiss-cpu sentence-transformers")
#     exit()


# class EmbeddingIndexer:
#     """
#     A class to create a FAISS index from hierarchical text chunks.
#     It takes pre-chunked data as input.
#     """
#     def __init__(self, model_name='all-MiniLM-L6-v2'):
#         """
#         Initializes the Indexer with a sentence transformer model.

#         Args:
#             model_name (str): The name of the sentence-transformer model to use.
#         """
#         print(f"Loading sentence transformer model: {model_name}...")
#         self.model = SentenceTransformer(model_name)
#         print("Model loaded successfully.")

#     def create_index_from_chunks(self, chunks):
#         """
#         Creates a FAISS index and a metadata mapping from the given chunks.

#         Args:
#             chunks (list): A list of hierarchical chunk dictionaries.

#         Returns:
#             tuple: A tuple containing:
#                 - faiss_index: The created FAISS index object.
#                 - index_to_chunk_map (list): A list mapping the index ID to the original chunk.
#         """
#         if not chunks:
#             return None, []

#         # 1. Create unified text for each chunk for better embedding context
#         unified_texts = [f"{chunk.get('section_title', '')}: {chunk.get('text', '')}" for chunk in chunks]

#         # 2. Generate embeddings for the unified texts
#         print(f"Generating embeddings for {len(unified_texts)} chunks...")
#         embeddings = self.model.encode(unified_texts, convert_to_tensor=False, show_progress_bar=True)
        
#         if not hasattr(embeddings, 'size') or embeddings.size == 0:
#             print("Warning: No embeddings were generated. Cannot create index.")
#             return None, []
            
#         embedding_dimension = embeddings.shape[1]

#         # 3. Build the FAISS index
#         print("Building FAISS index...")
#         # Using IndexFlatL2 for exact, brute-force L2 distance search.
#         # It's fast for up to a million vectors. For larger datasets, a more 
#         # complex index like 'IndexIVFPQ' would provide better performance.
#         faiss_index = faiss.IndexFlatL2(embedding_dimension)
        
#         # Add the generated vectors to the index
#         faiss_index.add(np.array(embeddings, dtype='float32'))
        
#         print(f"FAISS index built successfully with {faiss_index.ntotal} vectors.")

#         # 4. Create the map to link index IDs back to chunk data
#         # The map is simply the list of chunks, as their index in the list
#         # corresponds to their ID in the FAISS index (for IndexFlatL2).
#         index_to_chunk_map = chunks

#         return faiss_index, index_to_chunk_map


# def process_chunked_directory(input_dir, output_dir):
#     """
#     Processes all chunked JSON files in an input directory, creates a FAISS
#     index and metadata map for each, and saves them to an output directory.
#     """
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Find all chunked json files in the input directory
#     json_files = glob.glob(os.path.join(input_dir, '*.json'))
    
#     if not json_files:
#         print(f"Error: No JSON files found in the input directory: {input_dir}")
#         return

#     print(f"Found {len(json_files)} chunked JSON files to process.")
    
#     # Initialize the indexer once to avoid reloading the model for each file
#     indexer = EmbeddingIndexer()

#     for file_path in json_files:
#         base_filename = os.path.basename(file_path)
#         filename_no_ext = os.path.splitext(base_filename)[0]
#         print(f"\n--- Processing {base_filename} ---")

#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 # Load the already-chunked data
#                 hierarchical_chunks = json.load(f)

#             if not hierarchical_chunks:
#                 print("JSON file is empty. Skipping.")
#                 continue

#             # --- Create Index and Metadata Map ---
#             faiss_index, index_to_chunk_map = indexer.create_index_from_chunks(hierarchical_chunks)
            
#             if faiss_index:
#                 # Define paths for the output files
#                 index_path = os.path.join(output_dir, f"{filename_no_ext}.index")
#                 metadata_path = os.path.join(output_dir, f"{filename_no_ext}_map.json")
                
#                 # Save the FAISS index to disk
#                 faiss.write_index(faiss_index, index_path)
                
#                 # Save the metadata map to disk
#                 with open(metadata_path, 'w', encoding='utf-8') as f:
#                     json.dump(index_to_chunk_map, f, indent=2, ensure_ascii=False)
                
#                 print(f"Successfully saved FAISS index to: {index_path}")
#                 print(f"Successfully saved metadata map to: {metadata_path}")

#         except json.JSONDecodeError:
#             print(f"Error: Could not decode JSON from {file_path}. It might be corrupted. Skipping.")
#         except Exception as e:
#             print(f"An unexpected error occurred while processing {file_path}: {e}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description="Create a searchable FAISS index from pre-chunked JSON files.",
#         formatter_class=argparse.RawTextHelpFormatter
#     )
#     parser.add_argument('--input_dir', type=str, required=True, 
#                         help="Directory containing the pre-chunked JSON files.")
#     parser.add_argument('--output_dir', type=str, required=True, 
#                         help="Directory where the final FAISS index and metadata map will be saved.")
#     args = parser.parse_args()

#     # --- How to Run ---
#     # This script assumes you have already run the chunking script.
#     # python embedding.py --input_dir chunked --output_dir embedded
#     process_chunked_directory(args.input_dir, args.output_dir)
