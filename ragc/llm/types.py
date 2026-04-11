from ragc.llm.huggingface import HuggingFaceEmbedderConfig, CompletionGeneratorConfig
from ragc.llm.ollama import OllamaEmbedderConfig
from ragc.llm.remote import OpenAIEmbedderConfig, OpenAIChatGeneratorConfig

EmbedderConfig = OllamaEmbedderConfig | HuggingFaceEmbedderConfig | OpenAIEmbedderConfig
GeneratorConfig = CompletionGeneratorConfig | OpenAIChatGeneratorConfig
