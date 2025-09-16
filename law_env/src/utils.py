import os
from typing import List 
from src.chunker import ITokenizer

def read_file(filename):
    path = os.path.normpath(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_chunks_to_file(chunks: List[str], output_filename: str, tokenizer: ITokenizer):
    with open(os.path.normpath(output_filename), 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"=== Chunk {i+1} === (Tokens: {tokenizer.count_tokens(chunk)})\n")
            f.write(chunk + "\n\n")

