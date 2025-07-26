import re
import os
import glob
import json 
import argparse

class HierarchicalChunker:
    """
    A class to convert a flat list of text blocks from a document
    into a hierarchical structure based on predicted labels and positions.
    """

    def __init__(self, blocks):
        """
        Initializes the Chunker with a list of text blocks.

        Args:
            blocks (list): A list of dictionaries, where each dictionary
                           represents a text block.
        """
        self.blocks = sorted(blocks, key=lambda b: (b['page_number'], -b['features']['y_coordinate']))
        self.heading_levels = {'Title': 0, 'H1': 1, 'H2': 2, 'H3': 3}

    def _is_heading(self, block):
        """Checks if a block is a heading."""
        return block.get('predicted_label') in self.heading_levels

    def _get_heading_level(self, block):
        """Gets the numerical level of a heading."""
        return self.heading_levels.get(block.get('predicted_label'), 99) # Default to a high number for non-headings

    def _identify_headings(self):
        """
        Identifies all heading blocks in the document.

        Returns:
            list: A list of tuples, where each tuple contains the index,
                  level, and the block itself for each heading.
        """
        headings = []
        for i, block in enumerate(self.blocks):
            if self._is_heading(block):
                level = self._get_heading_level(block)
                headings.append({'index': i, 'level': level, 'block': block})
        return headings

    def _clean_text(self, text):
        """
        Cleans and normalizes a string of text.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        # Strip leading/trailing whitespace
        text = text.strip()
        # Optional: remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text

    def chunk_document(self):
        """
        Processes the flat blocks to create hierarchical chunks.

        Returns:
            list: A list of dictionaries, where each dictionary is a
                  hierarchical chunk containing a heading and its associated text.
        """
        headings = self._identify_headings()
        if not headings:
            # If no headings, treat the whole document as one chunk
            full_text = ' '.join(block['text'] for block in self.blocks)
            if not full_text.strip():
                return []
            return [{
                "heading": "Document",
                "level": "H1",
                "text": self._clean_text(full_text),
                "page_number": self.blocks[0]['page_number'] if self.blocks else 1,
                "document_name": self.blocks[0].get('document_name', 'Unknown') if self.blocks else 'Unknown'
            }]

        chunks = []
        for i, current_heading_info in enumerate(headings):
            current_heading_index = current_heading_info['index']
            current_heading_level = current_heading_info['level']
            current_heading_block = current_heading_info['block']

            # Determine the start and end of the content for this heading
            content_start_index = current_heading_index + 1
            content_end_index = len(self.blocks)

            # Find the next heading of the same or higher level
            for j in range(i + 1, len(headings)):
                next_heading_info = headings[j]
                if next_heading_info['level'] <= current_heading_level:
                    content_end_index = next_heading_info['index']
                    break
            
            # Gather all body text blocks between the current heading and the next
            body_blocks = self.blocks[content_start_index:content_end_index]
            
            # Filter out any nested headings from the body text
            body_text_blocks = [
                block['text'] for block in body_blocks 
                if not self._is_heading(block)
            ]
            
            body_text = ' '.join(body_text_blocks)
            cleaned_body_text = self._clean_text(body_text)

            # --- Step 3: Clean & Normalize Chunks ---
            
            # 1. Remove chunks with mostly punctuation or very short text
            # This regex checks if the string is primarily non-alphanumeric characters
            if re.match(r'^[\W_]+$', cleaned_body_text) and len(cleaned_body_text) < 10:
                continue
            
            # 2. Filter short body texts
            if len(cleaned_body_text) < 15:
                continue

            chunks.append({
                "document": current_heading_block.get('document_name', 'Unknown'),
                "section_title": self._clean_text(current_heading_block['text']),
                "level": current_heading_block['predicted_label'],
                "text": cleaned_body_text,
                "page_number": current_heading_block['page_number']
            })

        return chunks

# --- Example Usage ---
def process_directory(input_dir, output_dir):
    """
    Processes all JSON files in an input directory and saves the
    hierarchically chunked output to an output directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all json files in the input directory
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    
    if not json_files:
        print(f"No JSON files found in directory: {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files to process.")

    for file_path in json_files:
        print(f"Processing {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                flat_blocks = json.load(f)

            # Create a chunker instance and process the blocks
            chunker = HierarchicalChunker(flat_blocks)
            hierarchical_chunks = chunker.chunk_document()

            # Define the output path
            base_filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, base_filename)

            # Save the result
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(hierarchical_chunks, f, indent=2, ensure_ascii=False)
            
            print(f"Successfully saved chunked file to {output_path}")

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path}: {e}")


# --- Main execution block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert flat text block JSONs into hierarchical chunks.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing JSON files with flat text blocks.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where the final chunked JSON files will be saved.")
    args = parser.parse_args()

    # --- How to Run ---
    # python chunking.py --input_dir output_labeled --output_dir chunked
    process_directory(args.input_dir, args.output_dir)
