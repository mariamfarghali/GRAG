import os
from typing import List, Union
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

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