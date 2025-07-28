# Document Processing Pipeline

This project provides a document processing pipeline that can be run using Docker for easy deployment and consistent results.

## Important: Query Configuration Required

**Before running the pipeline, you must modify the `query.json` file** to specify:
- Your **persona** (role you want the AI to assume)
- Your **job to be done** (specific task or question you want answered)
- Your **input query** (the actual question or search request)
- The **document filenames** that should be processed

## Available Document Collections

The following document collections are available in the `input/` directory:

- **Collection_1/PDFs/** - South of France Travel Documents
- **Collection_2/PDFs/** - Acrobat Learning Documents  
- **Collection_3/PDFs/** - Recipe Documents
- **your-collection/PDFs/** - Your custom document collection

## Docker Usage Instructions

Follow these steps to use the Docker container for processing your PDF documents:

### Step 1: Configure Your Query

First, modify the `query.json` file to specify your requirements:

```json
{
  "challenge_info": {
    "challenge_id": "your_challenge_id",
    "test_case_name": "your_test_case",
    "description": "Your description"
  },
  "documents": [
    {
      "filename": "document1.pdf",
      "title": "Document 1 Title"
    },
    {
      "filename": "document2.pdf",
      "title": "Document 2 Title"
    }
  ],
  "persona": {
    "role": "Your Role (e.g., Travel Expert, Chef, Technical Writer)"
  },
  "job_to_be_done": {
    "task": "Your specific task description and query"
  }
}
```

### Step 2: Prepare Your Input Directory

Create or modify the input directory structure to include your PDF collection. The PDFs mentioned in `query.json` should be placed in the input directory. For example:

```
input/
├── Collection_1/
│   └── PDFs/
│       ├── document1.pdf
│       ├── document2.pdf
│       └── ...
├── Collection_2/
│   └── PDFs/
│       ├── other_document1.pdf
│       └── other_document2.pdf
└── ...
```

### Step 3: Build the Docker Image

Build the Docker image using the following command:

```bash
docker build -t document-processor .
```

### Step 4: Run the Docker Container

Run the container while mounting your local input directory and output directory:

```bash
docker run -v /path/to/your/local/input:/app/input -v /path/to/your/local/output:/app/output document-processor
```

### Step 5: Retrieve Results

After the container finishes processing, you'll find the results in your local output directory under `output.json`.

## Docker Command Examples

### Example 1: Processing Collection_1 (South of France Travel Documents)

```bash
# Build the image
docker build -t document-processor:v1.0 .

# Run with Collection_1 (Git Bash on Windows)
docker run --rm -v //$(pwd)/input/Collection_1/PDFs:/app/input -v //$(pwd)/output:/app/output --network none document-processor:v1.0

# Run with Collection_1 (Linux/Mac)
docker run --rm -v $(pwd)/input/Collection_1/PDFs:/app/input -v $(pwd)/output:/app/output --network none document-processor:v1.0
```

### Example 2: Processing Collection_2 (Acrobat Learning Documents)

```bash
# Run with Collection_2 (Git Bash on Windows)
docker run --rm -v //$(pwd)/input/Collection_2/PDFs:/app/input -v //$(pwd)/output:/app/output --network none document-processor:v1.0

# Run with Collection_2 (Linux/Mac)
docker run --rm -v $(pwd)/input/Collection_2/PDFs:/app/input -v $(pwd)/output:/app/output --network none document-processor:v1.0
```

### Example 3: Processing Collection_3 (Recipe Documents)

```bash
# Run with Collection_3 (Git Bash on Windows)
docker run --rm -v //$(pwd)/input/Collection_3/PDFs:/app/input -v //$(pwd)/output:/app/output --network none document-processor:v1.0

# Run with Collection_3 (Linux/Mac)
docker run --rm -v $(pwd)/input/Collection_3/PDFs:/app/input -v $(pwd)/output:/app/output --network none document-processor:v1.0
```

### Example 4: Processing Custom Collection

```bash
# For your own custom collection (Git Bash on Windows)
docker run --rm -v //$(pwd)/input/your-collection/PDFs:/app/input -v //$(pwd)/output:/app/output --network none document-processor:v1.0

# For your own custom collection (Linux/Mac)
docker run --rm -v $(pwd)/input/your-collection/PDFs:/app/input -v $(pwd)/output:/app/output --network none document-processor:v1.0
```

## Important Notes

- **Recommended Command**: Use the command with `--rm` (auto-cleanup), `--network none` (offline mode), and versioned tag (`document-processor:v1.0`) for best practices.
- **Different Collections**: For each different collection, you need to mount the specific collection directory during the `docker run` command, as shown in the examples above.
- **Path Format**: Use `//$(pwd)` for Git Bash on Windows and `$(pwd)` for Linux/Mac to provide relative path mounting from your current directory.
- **Query Configuration**: Make sure your `query.json` file references the correct PDF filenames that exist in your mounted input directory.
- **Output Location**: The processed results will always appear in the mounted output directory as `output.json`.
- **Container Cleanup**: The `--rm` flag automatically removes the container after processing is complete.
- **Offline Mode**: The `--network none` flag ensures the container runs completely offline after the initial build.

## Troubleshooting

- Ensure that the PDF files specified in `query.json` exist in the mounted input directory
- Check that you have proper read/write permissions for the mounted directories
- Verify that Docker has sufficient resources allocated for processing large PDF collections
