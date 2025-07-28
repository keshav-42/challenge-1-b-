import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class EmbeddingHandler:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}...")
        
        import os
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully.")

    def encode(self, texts: list, batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)

    def encode_and_normalize(self, texts: list) -> np.ndarray:
        embeddings = self.encode(texts)
        faiss.normalize_L2(embeddings)
        return embeddings