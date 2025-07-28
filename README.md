# Automated Document Processing Pipeline

This repository contains an automated pipeline for processing PDF documents through semantic search and analysis.

## Overview

The pipeline executes the following sequence:

1. **Parsing** - Extract text blocks from PDF files with style and semantic features
2. **Build Vectors** - Create feature vectors from text blocks
3. **Cluster & Label** - Cluster blocks and assign hierarchical labels
4. **Chunking** - Convert blocks into semantic chunks with section IDs
5. **Embedding** - Create FAISS embeddings and search index
6. **Main Search** - Perform semantic search and generate final output

## Quick Start

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

### Running the Pipeline

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
├── output/             # Parsed text blocks (intermediate)
├── output_vectors/     # Feature vectors (intermediate)
├── output_labeled/     # Labeled blocks (intermediate)
├── chunked/           # Semantic chunks (intermediate)
├── embedded/          # FAISS embeddings (intermediate)
├── query.json         # Search configuration
├── output.json        # Final search results
└── pipeline.py        # Main pipeline script
```

## What Happens When You Run the Pipeline

1. **Automatic Cleanup**: All intermediate directories are cleared to ensure a fresh start
2. **Sequential Processing**: Each step runs only after the previous step completes successfully
3. **Error Handling**: Pipeline stops if any step fails, with detailed error messages
4. **Progress Tracking**: Real-time logging shows which step is currently running
5. **Final Validation**: Checks that `output.json` was created successfully

## Manual Step Execution

You can also run individual steps manually:

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

## Pipeline Status

Check the current state of your pipeline:

```bash
python pipeline.py --status
```

This shows:

- Number of input PDF files
- Status of intermediate directories
- Whether final output exists

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install required packages with `pip install -r requirements.txt`
2. **No PDF Files**: Ensure PDF files are in the `input/` directory
3. **Invalid query.json**: Check that `query.json` has proper format and required fields
4. **Permission Errors**: Ensure write permissions for the workspace directory

### Debug Mode

For detailed output, check the logs during pipeline execution. Each step provides:

- Start/completion timestamps
- File counts and paths
- Error messages with details

### Clean Start

To completely reset the pipeline:

```bash
python pipeline.py  # (automatically cleans intermediate directories)
```

## Output Format

The final `output.json` contains:

- **metadata**: Input documents, persona, and enhanced query used
- **extracted_sections**: Ranked sections with importance scores
- **subsection_analysis**: Detailed text content for each section

## Requirements

Create a `requirements.txt` file with the necessary dependencies:

```txt
pdfminer.six
numpy
scikit-learn
faiss-cpu
sentence-transformers
nltk
```
