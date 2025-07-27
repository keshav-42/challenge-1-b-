# search_utils.py
import os
import json
import numpy as np
import faiss
from collections import defaultdict

class FaissSearcher:
    def __init__(self, index_path: str, map_path: str):
        if not os.path.exists(index_path): raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(map_path): raise FileNotFoundError(f"Map file not found: {map_path}")
        print(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        with open(map_path, 'r', encoding='utf-8') as f:
            self.index_to_chunk_map = json.load(f)

    def retrieve_candidates(self, query_vector: np.ndarray, top_k: int = 30) -> list:
        distances, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        candidates = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx != -1 and 0 <= idx < len(self.index_to_chunk_map):
                chunk = self.index_to_chunk_map[idx]
                # Attach the raw similarity score for re-ranking
                # For L2 normalized vectors, similarity = 1 - (L2_distance^2 / 2)
                # Since faiss returns squared L2 distance, this is correct.
                chunk['base_similarity'] = 1 - (distances[0][i] / 2)
                candidates.append(chunk)
        return candidates

def re_rank_results(candidates: list) -> list:
    doc_counts = defaultdict(int)
    for cand in candidates:
        score = cand.get('base_similarity', 0)
        
        # Apply a diversity boost
        doc_name = cand.get("document_name")
        penalty = doc_counts[doc_name] * 0.02
        score -= penalty
        doc_counts[doc_name] += 1
        
        cand['importance_score'] = float(round(score, 4))
        
    return sorted(candidates, key=lambda x: x['importance_score'], reverse=True)