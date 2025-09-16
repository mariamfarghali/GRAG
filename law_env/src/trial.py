import re
import tiktoken
import os

def count_tokens(text):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def split_text_by_tokens(text, max_tokens=500):
    """Split text into chunks of maximum token size"""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks

def chunk_text(text, max_tokens=500, min_tokens=50):
    """
    Chunk text based on token count rules:
    - If tokens <=500 and >50: make it as one chunk
    - If tokens <=50: append it to the next text
    - If tokens >500: split into chunks of max_tokens
    """
    # Split text into paragraphs while preserving structure
    paragraphs = re.split(r'(\n\s*\n)', text)
    chunks = []
    current_chunk = ""
    current_token_count = 0

    for i, paragraph in enumerate(paragraphs):
        # Skip empty paragraphs
        if not paragraph.strip():
            continue

        para_token_count = count_tokens(paragraph)

        # If paragraph is too large, split it
        if para_token_count > max_tokens:
            # Finalize current chunk if exists
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
                current_token_count = 0

            # Split large paragraph
            split_chunks = split_text_by_tokens(paragraph, max_tokens)

            # Add all but the last chunk
            for chunk in split_chunks[:-1]:
                chunks.append(chunk)

            # Handle the last chunk
            last_chunk = split_chunks[-1]
            last_token_count = count_tokens(last_chunk)

            if last_token_count <= min_tokens and i < len(paragraphs) - 1:
                # If last chunk is small and there's a next paragraph, don't finalize yet
                current_chunk = last_chunk
                current_token_count = last_token_count
            else:
                chunks.append(last_chunk)

        # If paragraph is too small, add to current chunk
        elif para_token_count <= min_tokens:
            # If current chunk + paragraph would exceed max, finalize current chunk
            if current_token_count + para_token_count > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
                current_token_count = para_token_count
            else:
                current_chunk += paragraph
                current_token_count += para_token_count

        # If paragraph is within acceptable range
        else:
            # If we have a current chunk, check if we can merge
            if current_chunk and current_token_count + para_token_count <= max_tokens:
                current_chunk += paragraph
                current_token_count += para_token_count
            else:
                # Finalize current chunk if exists
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
                current_token_count = para_token_count

    # Add the final chunk if it exists
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def read_file(filename):
    """Read file content with proper path handling"""
    # Use os.path to handle file paths correctly
    full_path = os.path.normpath(filename)
    with open(full_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_chunks_to_file(chunks, output_filename):
    """Write chunks to a file for inspection"""
    full_path = os.path.normpath(output_filename)
    with open(full_path, 'w', encoding='utf-8') as file:
        for i, chunk in enumerate(chunks):
            file.write(f"=== Chunk {i+1} === (Tokens: {count_tokens(chunk)})\n")
            file.write(chunk)
            file.write("\n\n")