from typing import Literal
import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import pipeline
from typing import List, Union
from pydantic import BaseModel

from ragc.simple_rag.dataset import CodeRepoDataset
from ragc.llm.embedding import BaseEmbedder
from ragc.llm.types import EmbedderConfig, GeneratorConfig
from ragc.llm.generator import BaseGenerator

try:
    import faiss  # faiss-cpu or faiss-gpu must be installed
except ImportError:
    import faiss_cpu as faiss

class SimpleRAGPipeline:
    def __init__(self, dataset: CodeRepoDataset, embedder: BaseEmbedder, generator: BaseGenerator, topk: int = 5):
        self.dataset = dataset
        self.embedder = embedder
        self.llm = generator
        self.index = self._build_faiss_index()

    def _build_faiss_index(self):
        embeddings = torch.stack([self.dataset[i]["embedding"] for i in range(len(self.dataset))]).numpy()
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def retrieve(self, query: str, top_k=5):
        query_embedding = self.embedder.embed([query])[0]
        D, I = self.index.search(np.array([query_embedding]), top_k)
        return [self.dataset[i] for i in I[0]]

    def _format_prompt(self, query: str, context_chunks):
        context = "\n\n".join(chunk["chunk"] for chunk in context_chunks)
        return f"""You are a helpful assistant that understands code.

Context:
{context}

Question: {query}
Answer:"""

    def query(self, question: str, top_k=5) -> str:
        chunks = self.retrieve(question, top_k)
        prompt = self._format_prompt(question, chunks)
        print(prompt)
        output = self.llm.generate(prompt=prompt)
        return output


class SimpleRAGPipelineConfig(BaseModel):
    type: Literal["simple_rag_pipeline"]
    embedder: EmbedderConfig
    generator: GeneratorConfig
    topk: int = 5

    def create(self, dataset: CodeRepoDataset) -> SimpleRAGPipeline:
        _embedder = self.embedder.create()
        _generator = self.generator.create()
        return SimpleRAGPipeline(
            dataset=dataset,
            embedder=_embedder,
            generator=_generator,
            topk=self.topk,
        )
