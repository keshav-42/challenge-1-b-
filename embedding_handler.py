# embedding_handler.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class EmbeddingHandler:
    """
    Handles loading an embedding model and encoding text into vectors.
    Uses a dedicated sentence-transformer model for high-quality embeddings.
    """
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initializes the sentence-transformer model.

        Args:
            model_name (str): The name of the sentence-transformer model.
            cache_dir (str, optional): A directory to cache the downloaded model.
        """
        print(f"🔍 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("   Embedding model loaded successfully.")

    def encode(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of texts into semantic embeddings.
        """
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)

    def encode_and_normalize(self, texts: list) -> np.ndarray:
        """
        Encodes texts and normalizes the resulting vectors for FAISS (L2 norm).
        """
        embeddings = self.encode(texts)
        faiss.normalize_L2(embeddings)
        return embeddings