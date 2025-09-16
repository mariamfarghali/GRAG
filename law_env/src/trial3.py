import re
import os
from typing import List, Union
import torch
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from abc import ABC, abstractmethod

class ITokenizer(ABC):
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

class Tokenizer(ITokenizer):
    def __init__(self, model_name:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text)) if text else 0
 
    
class TextChunker:
    """Responsible for applying chunking rules."""

    def __init__(self, tokenizer: ITokenizer, max_tokens: int = 400, min_tokens: int = 50):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
    
    async def chunk_text(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        chunks, buffer = [], ""

        def split_text_at_word_boundary(paragraph: str, max_tokens: int) -> List[str]:
            splits = paragraph.split()
            chunks, current_chunk, current_tokens_count = [], "", 0

            #No need for case of single split exceed the max_tokens as it can't happen 
            for split in splits:
                split_token_count = self.tokenizer.count_tokens(split)
                if current_tokens_count + split_token_count > max_tokens:#In case current split will make the current chunk excced the token limit
                    if current_chunk:#If there is a current running chunk
                        chunks.append(current_chunk) #append the current_chunk to chunks list
                    current_chunk = split #make the new split the new current_chunk
                    current_tokens_count = split_token_count #make the split_token_count the new current_token_count
                else:
                    current_chunk = current_chunk + " " + split #In case the surrent split wiil not make the current_chunk exceed the token limit append the split to the current_chunk
                    current_tokens_count += split_token_count

            if current_chunk:
                chunks.append(current_chunk)
            return chunks
        
        # Iterate paragraphs
        for paragraph in paragraphs:
            p_toks = self.tokenizer.count_tokens(paragraph)

            # CASE A: paragraph too large
            if p_toks > self.max_tokens:
                pieces = split_text_at_word_boundary(paragraph, self.max_tokens)
                for piece in pieces:
                    pt = self.tokenizer.count_tokens(piece)
                    if pt <= self.min_tokens:
                        buffer = (buffer + " " + piece).strip() if buffer else piece
                    else:
                        if buffer and self.tokenizer.count_tokens(buffer) + pt <= self.max_tokens:
                            chunks.append((buffer + " " + piece).strip())
                            buffer = ""
                        else:
                            if buffer:
                                merged = (buffer + " " + piece).strip()
                                chunks.extend(split_text_at_word_boundary(merged, self.max_tokens))
                                buffer = ""
                            else:
                                chunks.append(piece)

            # CASE B: too small
            elif p_toks <= self.min_tokens:
                buffer = (buffer + " " + paragraph).strip() if buffer else paragraph
                
            # CASE C: normal paragraph
            else:
                if buffer and self.tokenizer.count_tokens(buffer) + p_toks <= self.max_tokens:
                    chunks.append((buffer + " " + paragraph).strip())
                    buffer = ""
                elif buffer:
                    merged = (buffer + " " + paragraph).strip()
                    chunks.extend(split_text_at_word_boundary(merged, self.max_tokens))
                    buffer = ""
                else:
                    chunks.append(paragraph)

        # Finalize buffer
        if buffer:
            if chunks and self.tokenizer.count_tokens(chunks[-1]) + self.tokenizer.count_tokens(buffer) <= self.max_tokens:
                chunks[-1] = chunks[-1] + " " + buffer
            else:
                chunks.extend(split_text_at_word_boundary(buffer, self.max_tokens))

        # Ensure no chunk < min_tokens
        # i = 0
        # while i < len(chunks):
        #     t = self.tokenizer.count_tokens(chunks[i])
        #     if t <= self.min_tokens:
        #         if i > 0 and self.tokenizer.count_tokens(chunks[i - 1]) + t <= self.max_tokens:
        #             chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
        #             del chunks[i]
        #             continue
        #         elif i + 1 < len(chunks):
        #             chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
        #             del chunks[i]
        #             continue
        #     i += 1

        return chunks

class Embedder:
    def __init__(
        self,
        embedder_model: Union[str, SentenceTransformer] = "LLukas22/paraphrase-multilingual-mpnet-base-v2-embedding-all",
        batch_size: int = 32,
        to_numpy: bool = True,
    ):
        """
        Embedder class for encoding text chunks.
        Args:
            embedder_model (Union[str, SentenceTransformer]): Hugging Face model name or preloaded model.
            batch_size (int): Batch size for embeddings.
            to_numpy (bool): Return numpy arrays if True, else torch tensors.
        """
        self.batch_size = batch_size
        self.to_numpy = to_numpy

        # Allow injection of preloaded SentenceTransformer model OR model name
        if isinstance(embedder_model, str):
            self.model = SentenceTransformer(embedder_model)
            self.model_name = embedder_model
        elif isinstance(embedder_model, SentenceTransformer):
            self.model = embedder_model
            self.model_name = embedder_model.__class__.__name__
        else:
            raise ValueError("embedder_model must be a str or SentenceTransformer instance")

    async def embed(self, chunks: List[str]) -> Union[np.ndarray, torch.Tensor]:
        """
        Embed a list of text chunks.
        Args:
            chunks (List[str]): List of text chunks.
        Returns:
            Union[np.ndarray, torch.Tensor]: Embeddings of shape (num_chunks, embedding_dim).
        """
        if not chunks:
            return np.array([]) if self.to_numpy else torch.empty(0)

        embeddings = self.model.encode(
            chunks,
            batch_size=self.batch_size,
            convert_to_tensor=not self.to_numpy,
            show_progress_bar=True,
        )

        print(f"Embedded {len(chunks)} chunks. Shape: {embeddings.shape}")
        return embeddings

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
