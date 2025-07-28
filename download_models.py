#!/usr/bin/env python3
"""
Pre-download all models and data needed for the pipeline to work offline.
This script should be run during Docker build to cache all required models.
"""

import os
import sys
from pathlib import Path

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    import nltk
    try:
        # Download punkt tokenizer (both old and new versions)
        nltk.download('punkt', quiet=False)
        print("✓ NLTK punkt tokenizer downloaded")
        
        # Download the newer punkt_tab resource
        try:
            nltk.download('punkt_tab', quiet=False)
            print("✓ NLTK punkt_tab tokenizer downloaded")
        except Exception as e:
            print(f"⚠ Could not download punkt_tab (might not be available): {e}")
            
    except Exception as e:
        print(f"✗ Failed to download NLTK data: {e}")
        return False
    return True

def download_sentence_transformer():
    """Download the sentence transformer model."""
    print("Downloading sentence transformer model...")
    try:
        from sentence_transformers import SentenceTransformer
        
        # This is the model used in the pipeline
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        print(f"Loading {model_name}...")
        
        # This will download and cache the model
        model = SentenceTransformer(model_name)
        print("✓ Sentence transformer model downloaded and cached")
        
        # Test that it works
        test_embedding = model.encode(["test sentence"])
        print(f"✓ Model test successful, embedding shape: {test_embedding.shape}")
        
    except Exception as e:
        print(f"✗ Failed to download sentence transformer: {e}")
        return False
    return True

def download_llm_model():
    """Download the LLM model."""
    print("Downloading LLM model...")
    try:
        from huggingface_hub import hf_hub_download
        
        # These are the model parameters used in main_search.py
        gguf_repo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
        gguf_filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        
        print(f"Downloading {gguf_repo}/{gguf_filename}...")
        
        # Download the model file
        model_path = hf_hub_download(
            repo_id=gguf_repo,
            filename=gguf_filename
        )
        
        print(f"✓ LLM model downloaded to: {model_path}")
        
        # Test loading with llama-cpp-python
        from llama_cpp import Llama
        print("Testing LLM model loading...")
        
        # Load model with minimal settings for testing
        llm = Llama(
            model_path=model_path,
            n_ctx=512,  # Smaller context for testing
            n_gpu_layers=0,
            verbose=False
        )
        
        print("✓ LLM model loaded successfully")
        
        # Clean up to save memory
        del llm
        
    except Exception as e:
        print(f"✗ Failed to download LLM model: {e}")
        return False
    return True

def verify_models():
    """Verify that all models are available and working."""
    print("\n=== Verifying Models ===")
    
    success = True
    
    # Test NLTK
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        print("✓ NLTK punkt tokenizer available")
    except Exception as e:
        print(f"✗ NLTK punkt tokenizer not available: {e}")
        success = False
    
    # Test Sentence Transformer
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        test_embedding = model.encode(["verification test"])
        print("✓ Sentence transformer working")
        del model
    except Exception as e:
        print(f"✗ Sentence transformer not working: {e}")
        success = False
    
    # Test LLM
    try:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        
        gguf_repo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
        gguf_filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        
        model_path = hf_hub_download(repo_id=gguf_repo, filename=gguf_filename)
        
        # Quick load test
        llm = Llama(
            model_path=model_path,
            n_ctx=256,
            n_gpu_layers=0,
            verbose=False
        )
        print("✓ LLM model working")
        del llm
        
    except Exception as e:
        print(f"✗ LLM model not working: {e}")
        success = False
    
    return success

def main():
    """Download all required models and data."""
    print("=== Pre-downloading Models for Offline Usage ===\n")
    
    downloads = [
        ("NLTK Data", download_nltk_data),
        ("Sentence Transformer", download_sentence_transformer),
        ("LLM Model", download_llm_model)
    ]
    
    success = True
    for name, download_func in downloads:
        print(f"\n--- {name} ---")
        try:
            if not download_func():
                success = False
        except Exception as e:
            print(f"✗ Unexpected error downloading {name}: {e}")
            success = False
    
    # Verify everything works
    if success:
        success = verify_models()
    
    if success:
        print("\n🎉 All models downloaded and verified successfully!")
        print("The container can now run offline without internet access.")
    else:
        print("\n❌ Some downloads failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
