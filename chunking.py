import re
import os
import glob
import json
import argparse
import nltk

def setup_nltk():
    """
    Downloads the required tokenizer models if not already present.
    This is required for sentence tokenization.
    """
    try:
        # Try the newer punkt_tab first
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            # Fall back to older punkt
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("NLTK tokenizer models not found. Downloading...")
            try:
                # Try downloading punkt_tab first (newer version)
                nltk.download('punkt_tab')
                print("Downloaded punkt_tab tokenizer.")
            except:
                # Fall back to punkt if punkt_tab fails
                nltk.download('punkt')
                print("Downloaded punkt tokenizer.")
            print("Download complete.")

class HierarchicalChunker:
    """
    A class to convert a flat list of text blocks from a document
    into a hierarchical structure, splitting long sections into
    token-aware sub-chunks with unique section identifiers.
    """

    def __init__(self, blocks, max_tokens=200):
        """
        Initializes the Chunker.

        Args:
            blocks (list): A list of dictionaries representing text blocks.
            max_tokens (int): The maximum number of tokens for a single chunk.
        """
        self.blocks = sorted(blocks, key=lambda b: (b.get('page_number', 0), -b.get('features', {}).get('y_coordinate', 0)))
        self.heading_levels = {'Title': 0, 'H1': 1, 'H2': 2, 'H3': 3}
        self.max_tokens = max_tokens

    def _count_tokens(self, text):
        """Estimates the number of tokens by splitting on whitespace."""
        return len(text.split())

    def _is_heading(self, block):
        """Checks if a block is a heading."""
        return block.get('predicted_label') in self.heading_levels

    def _get_heading_level(self, block):
        """Gets the numerical level of a heading."""
        return self.heading_levels.get(block.get('predicted_label'), 99)

    def _identify_headings(self):
        """Identifies all heading blocks in the document."""
        headings = []
        for i, block in enumerate(self.blocks):
            if self._is_heading(block):
                level = self._get_heading_level(block)
                headings.append({'index': i, 'level': level, 'block': block})
        return headings

    def _clean_text(self, text):
        """Cleans and normalizes a string of text."""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _create_section_id(self, text):
        """Creates a URL-friendly slug from heading text to use as an ID."""
        text = text.lower().strip()
        # Replace non-alphanumeric characters with a hyphen
        text = re.sub(r'[\s\W]+', '-', text)
        # Remove any leading or trailing hyphens
        text = re.sub(r'^-+|-+$', '', text)
        return text if text else "untitled-section"


    def chunk_document(self):
        """
        Processes flat blocks to create a flat list of logical chunks.
        Long sections are split into smaller sub-chunks.

        Returns:
            list: A flat list of chunk dictionaries, each with a `section_id`.
        """
        headings = self._identify_headings()
        if not headings:
            full_text = ' '.join(self._clean_text(block['text']) for block in self.blocks)
            if not full_text: return []
            
            # Create a base block and ID for documents without headings
            doc_heading_block = {
                "document_name": self.blocks[0].get('document_name', 'Unknown') if self.blocks else 'Unknown',
                "text": "Document",
                "predicted_label": "H1",
                "page_number": self.blocks[0]['page_number'] if self.blocks else 1
            }
            section_id = "document-main-content"
            return self._split_long_section(full_text, doc_heading_block, section_id)

        final_chunks = []
        for i, current_heading_info in enumerate(headings):
            current_heading_block = current_heading_info['block']
            section_title = self._clean_text(current_heading_block['text'])
            section_id = self._create_section_id(section_title) # **Generate section_id**
            
            # Determine the end of the content for this heading
            content_end_index = len(self.blocks)
            for j in range(i + 1, len(headings)):
                if headings[j]['level'] <= current_heading_info['level']:
                    content_end_index = headings[j]['index']
                    break
            
            # Gather and clean body text
            body_blocks = self.blocks[current_heading_info['index'] + 1 : content_end_index]
            body_text = ' '.join(block['text'] for block in body_blocks if not self._is_heading(block))
            cleaned_body_text = self._clean_text(body_text)
            
            # Skip sections with no meaningful content
            if len(cleaned_body_text) < 15:
                continue
            
            # Decide whether to split the section
            if self._count_tokens(cleaned_body_text) <= self.max_tokens:
                # Keep as a single chunk
                final_chunks.append({
                    "section_id": section_id,
                    "document": current_heading_block.get('document_name', 'Unknown'),
                    "section_title": section_title,
                    "level": current_heading_block['predicted_label'],
                    "text": cleaned_body_text,
                    "page_number": current_heading_block['page_number'],
                    "part": "1 of 1"
                })
            else:
                # Split into multiple sub-chunks
                sub_chunks = self._split_long_section(cleaned_body_text, current_heading_block, section_id)
                final_chunks.extend(sub_chunks)
        
        return final_chunks

    def _split_long_section(self, text, heading_block, section_id):
        """
        Splits a long text into sentence-aware sub-chunks and tags them
        with the provided section_id.
        """
        sentences = nltk.sent_tokenize(text)
        sub_chunks_text = []
        current_chunk_sentences = []
        current_token_count = 0
        
        for sentence in sentences:
            sentence_token_count = self._count_tokens(sentence)
            if current_token_count + sentence_token_count > self.max_tokens and current_chunk_sentences:
                sub_chunks_text.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = [sentence]
                current_token_count = sentence_token_count
            else:
                current_chunk_sentences.append(sentence)
                current_token_count += sentence_token_count
        
        if current_chunk_sentences:
            sub_chunks_text.append(' '.join(current_chunk_sentences))

        # Format the sub-chunks with consistent metadata
        formatted_chunks = []
        total_parts = len(sub_chunks_text)
        section_title = self._clean_text(heading_block['text'])

        for i, chunk_text in enumerate(sub_chunks_text):
            formatted_chunks.append({
                "section_id": section_id, # **Use the passed-in section_id**
                "document": heading_block.get('document_name', 'Unknown'),
                "section_title": section_title,
                "level": heading_block['predicted_label'],
                "text": chunk_text,
                "page_number": heading_block['page_number'],
                "part": f"{i + 1} of {total_parts}"
            })
            
        return formatted_chunks

# --- Example Usage & Main Block ---
# (The main execution block remains the same as the previous version)

def process_directory(input_dir, output_dir, max_tokens):
    """
    Processes all JSON files in an input directory and saves the
    hierarchically chunked output to an output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    
    if not json_files:
        print(f"No JSON files found in directory: {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files to process.")
    
    setup_nltk() # Setup NLTK once

    for file_path in json_files:
        print(f"Processing {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                flat_blocks = json.load(f)

            chunker = HierarchicalChunker(flat_blocks, max_tokens=max_tokens)
            final_chunks = chunker.chunk_document()

            base_filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, base_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_chunks, f, indent=2, ensure_ascii=False)
            
            print(f"Successfully saved {len(final_chunks)} chunks to {output_path}")

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert flat text block JSONs into a flat list of logical chunks with section IDs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing JSON files with flat text blocks.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where the final chunked JSON files will be saved.")
    parser.add_argument('--max_tokens', type=int, default=200, help="Maximum number of tokens per chunk. Default is 200.")
    args = parser.parse_args()

    # --- How to Run ---
    # 1. Install NLTK: pip install nltk
    # 2. Run the script:
    # python chunking.py --input_dir output_labeled --output_dir chunked --max_tokens 200
    process_directory(args.input_dir, args.output_dir, args.max_tokens)