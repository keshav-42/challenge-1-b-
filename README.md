# Automated Document Processing Pipeline

This repository contains a containerized, automated pipeline for processing PDF documents through semantic search and AI-powered analysis.

## Approach

This solution implements a multi-stage document processing pipeline that:

1. **Extracts structured text** from PDFs with semantic and stylistic features
2. **Creates feature vectors** from text blocks for machine learning analysis
3. **Applies clustering and labeling** to identify document structure (headings, body text, etc.)
4. **Chunks content semantically** into searchable units with hierarchical section IDs
5. **Generates embeddings** using sentence transformers and creates FAISS search indexes
6. **Performs intelligent search** using LLM-enhanced queries and semantic similarity

The pipeline is designed to be:

- **Fully automated** - One command execution from input to output
- **Containerized** - Runs consistently across different environments
- **Modular** - Each step can be run independently for debugging
- **Robust** - Handles encoding issues and provides detailed logging

## Models and Libraries Used

### Core ML Models

- **Sentence Transformers**: `sentence-transformers/all-MiniLM-L6-v2` for text embeddings
- **TinyLlama**: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` for query enhancement via LLM
- **FAISS**: Facebook AI Similarity Search for efficient vector similarity search
- **MiniBatchKMeans**: Scikit-learn clustering for document structure analysis

### Key Libraries

- **pdfminer.six**: Advanced PDF text extraction with layout analysis
- **llama-cpp-python**: CPU-optimized LLM inference
- **transformers & torch**: Hugging Face ecosystem for ML models
- **nltk**: Natural language processing utilities
- **scikit-learn**: Machine learning algorithms for clustering and feature processing

## Docker Build and Run Instructions

### Building the Docker Image

```bash
docker build --platform linux/amd64 -t document-processor:v1.0 .
```

### Running the Solution

The container runs completely offline with all models pre-cached:

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none document-processor:v1.0
```

**Important Notes:**

- Ensure your `input/` directory contains PDF files before running
- The `query.json` file should be present in the container (included in build)
- Output will be written to `output.json` in the mounted output directory
- The container runs with `--network none` for security (offline processing)
- All models are pre-downloaded, so no internet access is needed at runtime

## Local Development (Non-Docker)

### Prerequisites

1. Python 3.8+ with required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Place your PDF files in the `input/` directory

3. Configure your search query in `query.json`:
   ```json
   {
     "document_name": ["Document Name 1", "Document Name 2"],
     "persona": { "role": "Your Role" },
     "job": { "task": "Your search task description" }
   }
   ```

### Running the Pipeline Locally

Execute the complete pipeline with a single command:

```bash
python pipeline.py
```

### Pipeline Options

```bash
# Run full pipeline with cleanup (default)
python pipeline.py

# Run pipeline without cleaning intermediate directories
python pipeline.py --no-clean

# Check current pipeline status
python pipeline.py --status

# Use custom workspace directory
python pipeline.py --workspace /path/to/workspace
```

## Directory Structure

```
workspace/
├── input/              # Place PDF files here
├── output_parsed/      # Parsed text blocks (intermediate)
├── output_vectors/     # Feature vectors (intermediate)
├── output_labeled/     # Labeled blocks (intermediate)
├── chunked/           # Semantic chunks (intermediate)
├── embedded/          # FAISS embeddings (intermediate)
├── query.json         # Search configuration
├── output.json        # Final search results
└── pipeline.py        # Main pipeline script
```

## Pipeline Steps Explained

1. **Parsing** (`parsing.py`): Extracts text from PDFs with layout and font analysis
2. **Build Vectors** (`build_vectors.py`): Creates numerical feature vectors from text properties
3. **Cluster & Label** (`cluster_and_label.py`): Uses ML to identify headings, body text, etc.
4. **Chunking** (`chunking.py`): Groups related content into searchable semantic units
5. **Embedding** (`embedding.py`): Creates vector embeddings and FAISS search index
6. **Main Search** (`main_search.py`): Uses LLM and embeddings to find relevant content

## Manual Step Execution

You can also run individual steps manually for debugging:

```bash
# Step 1: Parse PDFs
python parsing.py --input_dir input --output_dir output

# Step 2: Build feature vectors
python build_vectors.py --input_dir output --output_dir output_vectors

# Step 3: Cluster and label
python cluster_and_label.py --input_dir output_vectors --output_dir output_labeled

# Step 4: Create chunks
python chunking.py --input_dir output_labeled --output_dir chunked

# Step 5: Create embeddings
python embedding.py --input_dir chunked --output_dir embedded

# Step 6: Perform search
python main_search.py --data_dir embedded --query_json query.json --output_file output.json
```

## Output Format

The final `output.json` contains:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": { "role": "Food Contractor" },
    "job_to_be_done": { "task": "Find vegetarian recipes" },
    "enhanced_query_used": "LLM-generated expanded query"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "section_title": "Vegetarian Main Dishes",
      "importance_rank": 1,
      "page_number": 5
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc1.pdf",
      "refined_text": "Detailed recipe text...",
      "page_number": 5
    }
  ]
}
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: `pip install -r requirements.txt`
2. **No PDF Files**: Place PDFs in `input/` directory
3. **Invalid query.json**: Check JSON format and required fields
4. **Encoding Errors**: The pipeline handles UTF-8 automatically
5. **Memory Issues**: Large PDFs may require more RAM for processing

### Pipeline Status Check

```bash
python pipeline.py --status
```

Shows current state of all intermediate directories and final output.

## Technical Architecture

The solution uses a microservices-style architecture where each processing step is independent:

- **Modularity**: Each step can be developed and tested separately
- **Fault Tolerance**: Pipeline stops on errors with clear diagnostics
- **Scalability**: Steps can be parallelized or distributed in future versions
- **Maintainability**: Clear separation of concerns with well-defined interfaces

## License

[Specify your license here]
pdfminer.six
numpy
scikit-learn
faiss-cpu
sentence-transformers
nltk

```

```
