from ragc.llm.huggingface import HuggingFaceEmbedderConfig, CompletionGeneratorConfig
from ragc.llm.ollama import OllamaEmbedderConfig, OllamaGeneratorConfig

EmbedderConfig = OllamaEmbedderConfig | HuggingFaceEmbedderConfig
GeneratorConfig = CompletionGeneratorConfig | OllamaGeneratorConfig
