from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from typing import List, Union
import re

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
    
    #splitting paragraphs 
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

        return chunks