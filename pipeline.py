#!/usr/bin/env python3
"""
Automated Pipeline for Document Processing

This script creates a clean, automated pipeline that:
1. Places a new input file/folder
2. Executes all intermediate steps in order
3. Stores intermediate results in directories
4. Wipes old results if re-run with new input
5. Produces final output.json

Sequence: parsing → output_parsed → build_vectors → output_vectors → cluster_and_label → output_labeled → chunking → chunked → embedding → embedded → main_search → output.json
"""

import os
import sys
import shutil
import subprocess
import json
import argparse
from pathlib import Path
from typing import List, Optional
import time

class DocumentProcessingPipeline:
    """Automated pipeline for document processing workflow."""
    
    def __init__(self, workspace_dir: str = None):
        """Initialize the pipeline with workspace directory."""
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path(__file__).parent
        
        # Auto-detect input directory
        self.input_dir = self._detect_input_directory()
        
        self.output_parsed_dir = self.workspace_dir / "output_parsed"
        self.chunked_dir = self.workspace_dir / "chunked"
        self.output_labeled_dir = self.workspace_dir / "output_labeled"
        self.embedded_dir = self.workspace_dir / "embedded"
        self.output_vectors_dir = self.workspace_dir / "output_vectors"
        # self.waste_dir = self.workspace_dir / "waste"
        
        # Pipeline step modules
        self.modules = {
            'parsing': 'parsing.py',
            'build_vectors': 'build_vectors.py', 
            'cluster_and_label': 'cluster_and_label.py',
            'chunking': 'chunking.py',
            'embedding': 'embedding.py',
            'main_search': 'main_search.py'
        }
        
        # Intermediate directories that get cleaned
        self.intermediate_dirs = [
            self.output_parsed_dir,
            self.chunked_dir, 
            self.output_labeled_dir,
            self.embedded_dir,
            self.output_vectors_dir
        ]
        
    def _detect_input_directory(self) -> Path:
        """Auto-detect the input directory with PDF files."""
        base_input_dir = self.workspace_dir / "input"
        
        # If input directory contains PDFs directly, use it
        if base_input_dir.exists():
            pdf_files = list(base_input_dir.glob("*.pdf"))
            if pdf_files:
                return base_input_dir
            
            # Also check one level deep for any subdirectory with PDFs
            for subdir in base_input_dir.iterdir():
                if subdir.is_dir():
                    pdf_files = list(subdir.glob("*.pdf"))
                    if pdf_files:
                        return subdir
        
        # Fallback to the input directory itself
        return base_input_dir
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def clean_intermediate_directories(self, force=False):
        """Remove and recreate all intermediate directories."""
        self.log("Cleaning intermediate directories...")
        
        for dir_path in self.intermediate_dirs:
            if dir_path.exists():
                try:
                    # Count files before cleaning
                    existing_files = list(dir_path.rglob("*"))
                    file_count = len([f for f in existing_files if f.is_file()])
                    
                    if file_count > 0:
                        self.log(f"Found {file_count} existing files in {dir_path}")
                    
                    # Try to remove contents instead of the directory itself
                    # This handles mounted volumes that can't be removed
                    for item in dir_path.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    
                    # Verify cleaning was successful
                    remaining_files = list(dir_path.rglob("*"))
                    remaining_count = len([f for f in remaining_files if f.is_file()])
                    
                    if remaining_count == 0:
                        self.log(f"✓ Successfully cleaned {dir_path}")
                    else:
                        self.log(f"⚠ {remaining_count} files remain in {dir_path}", "WARNING")
                        
                except OSError as e:
                    if "Device or resource busy" in str(e) or force:
                        self.log(f"Directory {dir_path} is mounted or busy, forcing cleanup")
                        # More aggressive cleaning for Docker scenarios
                        for item in dir_path.iterdir():
                            try:
                                if item.is_file():
                                    item.unlink()
                                elif item.is_dir():
                                    shutil.rmtree(item)
                            except OSError:
                                # Try to remove individual files
                                if item.is_dir():
                                    for subitem in item.rglob("*"):
                                        try:
                                            if subitem.is_file():
                                                subitem.unlink()
                                        except OSError:
                                            pass
                                    try:
                                        item.rmdir()
                                    except OSError:
                                        pass
                    else:
                        self.log(f"Could not clean {dir_path}: {e}", "WARNING")
            else:
                self.log(f"Creating {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
        
    def validate_input(self) -> bool:
        """Validate that input directory exists and contains files."""
        if not self.input_dir.exists():
            self.log(f"Input directory {self.input_dir} does not exist", "ERROR")
            return False
            
        # Check for PDF files or subdirectories with PDFs
        pdf_files = list(self.input_dir.rglob("*.pdf"))
        if not pdf_files:
            self.log(f"No PDF files found in {self.input_dir}", "ERROR")
            return False
            
        self.log(f"Found {len(pdf_files)} PDF file(s) to process")
        return True
        
    def validate_modules(self) -> bool:
        """Validate that all required Python modules exist."""
        missing_modules = []
        
        for step, module_file in self.modules.items():
            module_path = self.workspace_dir / module_file
            if not module_path.exists():
                missing_modules.append(module_file)
                
        if missing_modules:
            self.log(f"Missing required modules: {', '.join(missing_modules)}", "ERROR")
            return False
            
        return True
        
    def run_module(self, step: str, args: List[str]) -> bool:
        """Run a pipeline module with given arguments."""
        module_file = self.modules[step]
        module_path = self.workspace_dir / module_file
        
        cmd = [sys.executable, str(module_path)] + args
        self.log(f"Running {step}: {' '.join(cmd)}")
        
        try:
            # Set environment variables for UTF-8 encoding and offline mode
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            # Ensure offline mode for transformers and huggingface
            env['TRANSFORMERS_OFFLINE'] = '1'
            env['HF_HUB_OFFLINE'] = '1'
            
            result = subprocess.run(
                cmd, 
                cwd=self.workspace_dir,
                capture_output=True, 
                text=True, 
                check=True,
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            self.log(f"+ {step} completed successfully")
            if result.stdout.strip():
                self.log(f"Output: {result.stdout.strip()}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"x {step} failed with exit code {e.returncode}", "ERROR")
            if e.stdout:
                self.log(f"STDOUT: {e.stdout}", "ERROR")
            if e.stderr:
                self.log(f"STDERR: {e.stderr}", "ERROR")
            return False
            
    def step_parsing(self) -> bool:
        """Step 1: Parse PDF files into structured text blocks."""
        self.log("=== Step 1: Parsing ===")
        
        args = [
            "--input_dir", str(self.input_dir),
            "--output_dir", str(self.output_parsed_dir)
        ]
        
        return self.run_module('parsing', args)
        
    def step_build_vectors(self) -> bool:
        """Step 2: Build feature vectors from parsed text blocks."""
        self.log("=== Step 2: Building Vectors ===")
        
        args = [
            "--input_dir", str(self.output_parsed_dir),
            "--output_dir", str(self.output_vectors_dir)
        ]
        
        return self.run_module('build_vectors', args)
        
    def step_cluster_and_label(self) -> bool:
        """Step 3: Cluster and label text blocks."""
        self.log("=== Step 3: Clustering and Labeling ===")
        
        args = [
            "--input_dir", str(self.output_vectors_dir),
            "--output_dir", str(self.output_labeled_dir)
        ]
        
        return self.run_module('cluster_and_label', args)
        
    def step_chunking(self) -> bool:
        """Step 4: Chunk labeled text into semantic units."""
        self.log("=== Step 4: Chunking ===")
        
        args = [
            "--input_dir", str(self.output_labeled_dir),
            "--output_dir", str(self.chunked_dir)
        ]
        
        return self.run_module('chunking', args)
        
    def step_embedding(self) -> bool:
        """Step 5: Create embeddings and FAISS index from chunked data."""
        self.log("=== Step 5: Creating Embeddings ===")
        
        args = [
            "--input_dir", str(self.chunked_dir),
            "--output_dir", str(self.embedded_dir)
        ]
        
        return self.run_module('embedding', args)
        
    def step_main_search(self) -> bool:
        """Step 6: Perform search to create final output."""
        self.log("=== Step 6: Main Search ===")
        
        # Check if query.json exists
        query_file = self.workspace_dir / "query.json"
        if not query_file.exists():
            self.log(f"Query file {query_file} not found", "ERROR")
            return False
        
        # Determine output file location (for Docker compatibility)
        output_file = self.workspace_dir / "output.json"
        
        args = [
            "--data_dir", str(self.embedded_dir),
            "--query_json", str(query_file),
            "--output_file", str(output_file)
        ]
        
        return self.run_module('main_search', args)
        
    def validate_final_output(self) -> bool:
        """Validate that the final output.json was created successfully."""
        output_file = self.workspace_dir / "output.json"
        
        if not output_file.exists():
            self.log("Final output.json was not created", "ERROR")
            return False
            
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.log(f"+ Final output.json created successfully with {len(data)} items")
            return True
        except json.JSONDecodeError as e:
            self.log(f"Final output.json is not valid JSON: {e}", "ERROR")
            return False
            
    def run_pipeline(self, clean: bool = True, force_clean: bool = False) -> bool:
        """Run the complete pipeline."""
        self.log("Starting Document Processing Pipeline")
        
        # Check if we're running in Docker (common environment variable)
        in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)
        if in_docker:
            self.log("Detected Docker environment - forcing clean start")
            force_clean = True
        
        # Validation steps
        if not self.validate_modules():
            return False
            
        if not self.validate_input():
            return False
            
        # Clean intermediate directories
        if clean:
            self.clean_intermediate_directories(force=force_clean)
            
        # Execute pipeline steps in sequence
        pipeline_steps = [
            self.step_parsing,
            self.step_build_vectors, 
            self.step_cluster_and_label,
            self.step_chunking,
            self.step_embedding,
            self.step_main_search
        ]
        
        for i, step_func in enumerate(pipeline_steps, 1):
            if not step_func():
                self.log(f"Pipeline failed at step {i}", "ERROR")
                return False
                
        # Validate final output
        if not self.validate_final_output():
            return False
            
        self.log("Pipeline completed successfully!")
        return True
        
    def get_status(self) -> dict:
        """Get the current status of intermediate directories and files."""
        status = {
            'input_files': len(list(self.input_dir.rglob("*.pdf"))) if self.input_dir.exists() else 0,
            'intermediate_dirs': {},
            'final_output': (self.workspace_dir / "output.json").exists()
        }
        
        for dir_path in self.intermediate_dirs:
            if dir_path.exists():
                json_files = list(dir_path.glob("*.json"))
                index_files = list(dir_path.glob("*.index"))
                status['intermediate_dirs'][dir_path.name] = {
                    'exists': True,
                    'json_files': len(json_files),
                    'index_files': len(index_files),
                    'total_files': len(list(dir_path.iterdir()))
                }
            else:
                status['intermediate_dirs'][dir_path.name] = {'exists': False}
                
        return status


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Automated Document Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                    # Run full pipeline with cleanup
  python pipeline.py --no-clean        # Run pipeline without cleaning intermediate dirs
  python pipeline.py --force-clean     # Force aggressive cleanup (useful for Docker)
  python pipeline.py --status          # Show current pipeline status
  python pipeline.py --workspace /path # Run with custom workspace directory
        """
    )
    
    parser.add_argument(
        "--workspace", 
        type=str,
        default=None,
        help="Workspace directory (default: script directory)"
    )
    
    parser.add_argument(
        "--no-clean",
        action="store_true", 
        help="Skip cleaning intermediate directories"
    )
    
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Force aggressive cleaning of intermediate directories (useful for Docker)"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show pipeline status and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DocumentProcessingPipeline(args.workspace)
    
    if args.status:
        # Show status
        status = pipeline.get_status()
        print("\n=== Pipeline Status ===")
        print(f"Input PDF files: {status['input_files']}")
        print(f"Final output exists: {status['final_output']}")
        print("\nIntermediate directories:")
        for dir_name, dir_status in status['intermediate_dirs'].items():
            if dir_status['exists']:
                print(f"  {dir_name}: {dir_status['total_files']} files "
                      f"({dir_status['json_files']} JSON, {dir_status['index_files']} index)")
            else:
                print(f"  {dir_name}: does not exist")
        return
        
    # Run pipeline
    clean = not args.no_clean
    force_clean = args.force_clean
    success = pipeline.run_pipeline(clean=clean, force_clean=force_clean)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
