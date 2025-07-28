# llm_handler.py

from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import sys

class FastLanguageModel:
    """
    This class downloads and runs a GGUF-quantized model using llama-cpp-python.
    """
    def __init__(self, model_repo_id: str, model_filename: str):
        print("Downloading GGUF model (if not cached)...")
        
        # Set offline mode environment variables
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        try:
            self.model_path = hf_hub_download(
                repo_id=model_repo_id,
                filename=model_filename,
                local_files_only=True  # Force offline mode
            )
            print("GGUF Model located.")
        except Exception as e:
            print(f"Error loading cached model: {e}")
            # Try to find the model in cache manually
            from huggingface_hub import snapshot_download
            try:
                cache_dir = snapshot_download(
                    repo_id=model_repo_id,
                    local_files_only=True,
                    allow_patterns=[model_filename]
                )
                self.model_path = os.path.join(cache_dir, model_filename)
                print(f"GGUF Model found in cache: {self.model_path}")
            except Exception as e2:
                print(f"Error finding model in cache: {e2}")
                sys.exit(1)
        
        print("Loading model with llama-cpp...")
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_gpu_layers=0,
            verbose=False
        )
        print("Model loaded successfully on CPU.")

    def generate(self, messages: list, max_tokens: int = 150) -> str:
        """
        Generates a response using the model's chat completion method.
        """
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens
        )
        return response['choices'][0]['message']['content']

