from typing import Literal

import numpy as np
import ollama
from pydantic import HttpUrl

from ragc.llm.embedding import BaseEmbedder, BaseEmbederConfig
from ragc.llm.generator import BaseGenerator, BaseGeneratorConfig


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, ollama_url: str, emb_model: str):
        self.emb_model = emb_model
        self.client = ollama.Client(host=ollama_url)

    def embed(self, inputs: list[str] | str) -> np.ndarray:
        response = self.client.embed(model=self.emb_model, input=inputs)
        embeddings = np.array(response.embeddings, dtype=np.float32)
        return embeddings


class OllamaEmbedderConfig(BaseEmbederConfig):
    type: Literal["ollama_embedding"] = "ollama_embedding"
    ollama_url: HttpUrl
    emb_model: str

    def create(self) -> OllamaEmbedder:
        return OllamaEmbedder(
            ollama_url=str(self.ollama_url),
            emb_model=self.emb_model,
        )


class OllamaGenerator(BaseGenerator):
    type: Literal["ollama_embedding"] = "ollama_generator"

    def __init__(self, ollama_url: str, model: str):
        self.client = ollama.Client(ollama_url)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.generate(model=self.model, prompt=prompt)
        return response.response


class OllamaGeneratorConfig(BaseGeneratorConfig):
    type: Literal["ollama_generator"] = "ollama_generator"
    ollama_url: HttpUrl = "localhost:11434"
    model: str

    def create(self) -> OllamaGenerator:
        return OllamaGenerator(
            ollama_url=str(self.ollama_url),
            model=self.model,
        )
