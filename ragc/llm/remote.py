"""OpenAI-compatible remote API clients for embeddings and chat (OpenAI or OpenRouter)."""

import os
from typing import Any, Literal
from dotenv import load_dotenv

import numpy as np
from openai import OpenAI
from pydantic import Field

from ragc.graphs.common import Node
from ragc.llm.embedding import BaseEmbedder, BaseEmbederConfig
from ragc.llm.generator import AugmentedGenerator, AugmentedGeneratorConfig

load_dotenv()

# Default base URLs
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _build_prompt_from_query_and_nodes(
    query: dict[str, Any],
    relevant_nodes: list[Node],
    local_context_lines: int = 0,
) -> str:
    """Build full prompt from query dict and relevant nodes (same shape as CompletionGenerator)."""
    prompt = query.get("prompt", "")
    local_context = query.get("local_context", "")
    completion_path = query.get("completion_path", "")

    relevant_nodes = sorted(relevant_nodes, key=lambda x: -len(x.code))
    docs = [f"#{node.file_path}\n{node.code}" for node in relevant_nodes]

    if local_context_lines > 0 and local_context:
        local_lines = local_context.split("\n")
        trimmed = "\n".join(
            local_lines[max(0, len(local_lines) - local_context_lines) :],
        )
        local_block = f"#{completion_path}\n{trimmed}"
    else:
        local_block = ""

    parts = docs + ([local_block] if local_block else []) + [prompt]
    return "\n\n".join(parts)


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI (or OpenRouter) embeddings API."""

    def __init__(self, client: OpenAI, model: str, dimensions: int | None = None):
        self._client = client
        self._model = model
        self._dimensions = dimensions

    def embed(self, inputs: list[str] | str) -> np.ndarray:
        """Return embedding vectors for the given text(s)."""
        if isinstance(inputs, str):
            inputs = [inputs]
        kwargs = {"model": self._model, "input": inputs}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        response = self._client.embeddings.create(**kwargs)
        order = sorted(range(len(response.data)), key=lambda i: response.data[i].index)
        vectors = [response.data[i].embedding for i in order]
        return np.array(vectors, dtype=np.float32)


class OpenAIEmbedderConfig(BaseEmbederConfig):
    """Config for OpenAI/OpenRouter embeddings. Set base_url for OpenRouter.
    API key is read from config or from .env via api_key_env (default OPENAI_API_KEY).
    """

    type: Literal["openai"] = "openai"
    model: str = Field(default="text-embedding-3-small", description="Embedding model name")
    base_url: str | None = Field(
        default=None,
        description="Base URL. None = OpenAI, or use https://openrouter.ai/api/v1 for OpenRouter",
    )
    dimensions: int | None = Field(default=None, description="Optional embedding dimension")

    def create(self) -> OpenAIEmbedder:
        """Build OpenAI client and embedder instance."""
        key = os.environ.get("API_KEY")
        client = OpenAI(
            api_key=key,
            base_url=self.base_url or OPENAI_BASE_URL,
        )
        return OpenAIEmbedder(client=client, model=self.model, dimensions=self.dimensions)


class OpenAIChatGenerator(AugmentedGenerator):
    """Chat completion generator using OpenAI/OpenRouter chat API."""

    def __init__(  # noqa: PLR0913
        self,
        client: OpenAI,
        model: str,
        local_context_lines: int = 0,
        system_message: str | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        self._client = client
        self._model = model
        self._local_context_lines = local_context_lines
        self._system_message = system_message
        self._extra_kwargs = extra_kwargs or {}

    def generate(self, query: str | dict[str, Any], relevant_nodes: list[Node]) -> str:
        """Build prompt from query and context, call chat API, return assistant content."""
        if isinstance(query, str):
            prompt = query
        else:
            prompt = _build_prompt_from_query_and_nodes(
                query, relevant_nodes, local_context_lines=self._local_context_lines,
            )
        print("--------------------------------")
        print(prompt)
        print("--------------------------------")
        messages = []
        if self._system_message:
            messages.append({"role": "system", "content": self._system_message})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self._model,
            "messages": messages,
            **self._extra_kwargs,
        }
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        return (choice.message.content or "").strip()


class OpenAIChatGeneratorConfig(AugmentedGeneratorConfig):
    """Config for OpenAI/OpenRouter chat. Set base_url for OpenRouter.
    API key is read from config or from .env via api_key_env (default OPENAI_API_KEY).
    """

    type: Literal["openai_chat"] = "openai_chat"
    model: str = Field(description="Chat model, e.g. gpt-4o or openai/gpt-4o-mini")
    base_url: str | None = Field(
        default=None,
        description="Base URL. None = OpenAI, or https://openrouter.ai/api/v1 for OpenRouter",
    )
    local_context_lines: int = Field(default=0, description="Lines of local context in prompt")
    system_message: str | None = Field(default=None, description="Optional system message")
    extra_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs for chat.completions.create (e.g. temperature)",
    )

    def create(self) -> OpenAIChatGenerator:
        """Build OpenAI client and chat generator instance."""
        key = os.environ.get("API_KEY")
        client = OpenAI(
            api_key=key,
            base_url=self.base_url or OPENAI_BASE_URL,
        )
        return OpenAIChatGenerator(
            client=client,
            model=self.model,
            local_context_lines=self.local_context_lines,
            system_message=self.system_message,
            extra_kwargs=self.extra_kwargs,
        )
