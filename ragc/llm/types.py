from typing import Annotated, Union, Literal
from pydantic import BaseModel, Field
from ragc.llm.huggingface import HuggingFaceEmbedderConfig, CompletionGeneratorConfig
from ragc.llm.ollama import OllamaEmbedderConfig, OllamaGeneratorConfig

EmbedderConfig = OllamaEmbedderConfig | HuggingFaceEmbedderConfig
GeneratorConfig = Annotated[Union[CompletionGeneratorConfig, OllamaGeneratorConfig], Field(discriminator="type")]
