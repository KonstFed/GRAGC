"""Модуль отвечает за RAG где представляем его просто как текст.

Тут жесктий вайб кодинг TODO: удалить потом
"""
from pathlib import Path


from ragc.llm.embedding import BaseEmbederConfig, BaseEmbedder

def load_code_files(repo_path, extensions=[".py"]):
    code_chunks = []
    for path in Path(repo_path).rglob("*"):
        if path.suffix in extensions:
            with open(path, encoding='utf-8', errors='ignore') as f:
                code_chunks.append({"path": str(path), "content": f.read()})
    return code_chunks

def simple_chunking(code, max_lines=20):
    lines = code.split("\n")
    return ["\n".join(lines[i:i+max_lines]) for i in range(0, len(lines), max_lines)]

def embed_chunks(chunks):
    return embed_model.encode(chunks, convert_to_tensor=False, show_progress_bar=True)



class FaissInference:
    """Класс для инференса простого RAG."""

    embedder: BaseEmbedder


    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder


    def __call__(self, prompt: str) -> str:
        prompt