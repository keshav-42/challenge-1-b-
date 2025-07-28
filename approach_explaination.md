# Document Processing Pipeline - Approach Explanation

## Overview

This document processing pipeline implements a **RAG (Retrieval-Augmented Generation) architecture** that transforms PDF documents into semantically searchable content. Building upon the preprocessing modules from Challenge 1A, we developed a sequential pipeline that processes documents through six distinct stages, culminating in an intelligent search system capable of answering complex queries about document collections.

## RAG Pipeline Architecture

Our implementation follows the classic **RAG pattern** with three core phases:

**1. Retrieval Phase**: Extract and index document content

- Document parsing with structure preservation
- Vector embedding generation using sentence transformers
- FAISS indexing for efficient similarity search

**2. Augmentation Phase**: Enhance retrieved content

- Hierarchical clustering and labeling of text blocks
- Context-aware chunking respecting document structure
- Query enhancement using LLM-powered expansion

**3. Generation Phase**: Produce intelligent responses

- TinyLlama integration for offline text generation
- Contextual answer synthesis with source attribution
- Structured JSON output with confidence scoring

## Pipeline Stages

**Stage 1 - Parsing**: Convert PDFs to structured JSON with positional metadata using PyMuPDF
**Stage 2 - Vector Building**: Generate feature vectors from font sizes, positioning, and formatting patterns
**Stage 3 - Clustering & Labeling**: Adaptively assign hierarchical labels (Title, H1, H2, H3, Body) using either MiniBatchKMeans clustering or rule-based heuristics
**Stage 4 - Chunking**: Create semantically coherent segments while preserving document hierarchy
**Stage 5 - Embedding**: Generate 384-dimensional vectors using `sentence-transformers/all-MiniLM-L6-v2`
**Stage 6 - Search**: Execute LLM-enhanced queries against the FAISS index

## Model Architecture

- **LLM**: TinyLlama-1.1B-Chat-v1.0 (GGUF Q4_K_M quantized) for query enhancement and response generation
- **Embeddings**: SentenceTransformer all-MiniLM-L6-v2 for semantic similarity
- **Vector DB**: FAISS with L2 distance metric for efficient similarity search
- **Clustering**: MiniBatchKMeans for scalable document structure analysis

## Docker Implementation

The system operates as a fully containerized, offline-capable solution:

```bash
docker run --rm -v //$(pwd)/input/Collection_3/PDFs:/app/input -v //$(pwd)/output:/app/output --network none document-processor:v1.0
```

**Key Features**:

- **`--network none`**: Complete offline operation after initial setup
- **`--rm`**: Automatic container cleanup
- **Pre-cached Models**: All AI models downloaded during Docker build
- **Volume Mounting**: Direct access to input PDFs and output results

## Data Flow

Each stage produces intermediate outputs stored in dedicated directories (`output_parsed`, `output_vectors`, `output_labeled`, `chunked`, `embedded`), enabling debugging, resumability, and modular testing. The final `output.json` contains ranked search results with metadata, confidence scores, and LLM-generated responses.

This architecture delivers a robust, scalable solution for intelligent document processing that operates completely offline while maintaining high-quality semantic search capabilities.
