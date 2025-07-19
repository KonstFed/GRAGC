from pathlib import Path

import torch
from torch.utils.data import Dataset
from pydantic import BaseModel

from ragc.llm.embedding import BaseEmbedder
from ragc.llm.types import EmbedderConfig


class CodeRepoDataset(Dataset):
    extensions = [".py"]

    def __init__(
        self,
        repo_paths: list[str] | list,
        cache_dir: Path,
        embedder: BaseEmbedder,
        max_lines=20,
        use_cache=True,
    ):
        self.repo_paths = repo_paths if isinstance(repo_paths, list) else [repo_paths]
        self.cache_dir = Path(cache_dir)
        self.max_lines = max_lines
        self.use_cache = use_cache
        self.embedder = embedder

        self.data = []
        self.repo_to_indices = {}

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_all_repos()

    def _repo_name(self, path: str | Path):
        return Path(path).name

    def _get_cache_file(self, repo_path):
        repo_name = self._repo_name(repo_path)
        return self.cache_dir / f"{repo_name}.pt"

    def _chunk_code(self, code):
        lines = code.splitlines()
        return ["\n".join(lines[i : i + self.max_lines]) for i in range(0, len(lines), self.max_lines)]

    def _load_all_repos(self):
        for repo_path in self.repo_paths:
            cache_file = self._get_cache_file(repo_path)
            if self.use_cache and cache_file.exists():
                repo_data = torch.load(cache_file)
            else:
                repo_data = self._process_repo(repo_path)
                if self.use_cache:
                    torch.save(repo_data, cache_file)

            start_index = len(self.data)
            self.data.extend(repo_data)
            self.repo_to_indices[Path(repo_path).name] = list(range(start_index, start_index + len(repo_data)))

    def _process_repo(self, repo_path):
        repo_data = []
        cnt = 0
        for path in Path(repo_path).rglob("*"):
            if path.suffix in self.extensions:
                cnt += 1
                if cnt == 10:
                    break
                try:
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                        chunks = self._chunk_code(code)
                        for i, chunk in enumerate(chunks):
                            repo_data.append({
                                "repo": str(repo_path),
                                "path": str(path),
                                "chunk": chunk,
                                "chunk_id": i,
                            })
                except Exception as e:
                    print(f"Error reading {path}: {e}")

        texts = [item["chunk"] for item in repo_data]
        embeddings = self.embedder.embed(texts)

        for i, emb in enumerate(embeddings):
            repo_data[i]["embedding"] = emb

        return repo_data

    def get_indices_by_repo(self, repo_path: str):
        return self.repo_to_indices.get(str(repo_path), [])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            repo_path, local_idx = index
            repo_path = str(repo_path)
            repo_indices = self.repo_to_indices.get(repo_path)
            if repo_indices is None:
                raise KeyError(f"Repo path not found: {repo_path}")
            if local_idx >= len(repo_indices):
                raise IndexError(f"Index {local_idx} out of range for repo {repo_path}")
            global_idx = repo_indices[local_idx]
        else:
            global_idx = index

        item = self.data[global_idx]
        return {
            "embedding": torch.tensor(item["embedding"], dtype=torch.float32),
            "chunk": item["chunk"],
            "path": item["path"],
            "chunk_id": item["chunk_id"],
            "repo": item["repo"],
        }


class CodeRepoDatasetConfig(BaseModel):
    cache_path: Path
    embedder: EmbedderConfig

    def create(self, repo_name: str) -> CodeRepoDataset:
        _embedder = self.embedder.create()
        return CodeRepoDataset(
            repo_paths=repo_name,
            cache_dir=self.cache_path,
            embedder=_embedder,
        )
