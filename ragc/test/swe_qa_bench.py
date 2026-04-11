"""SWE-QA-Bench evaluation pipeline: repo-level QA with GNN retrieval."""

import json
import os
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel
from torch_geometric.data import Data
from tqdm import tqdm

from ragc.datasets.train_dataset import TorchGraphDataset, TorchGraphDatasetConfig
from ragc.graphs import Node
from ragc.inference import InferenceConfig


class SWEQABenchInference:
    """Run GNN-based retrieval + LLM generation on SWE-QA-Bench questions."""

    def __init__(
        self,
        dataset: TorchGraphDataset,
        inference_cfg: InferenceConfig,
        questions_dir: Path,
        repos: list[str] | None = None,
    ):
        self.dataset = dataset
        self.inference_cfg = inference_cfg
        self.questions_dir = Path(questions_dir)

        available_repos = set(dataset.get_repos_names())

        # Determine which repos to evaluate
        if repos is not None:
            self.repos = [r for r in repos if r in available_repos]
            missing = set(repos) - available_repos
            if missing:
                print(f"Warning: repos not in dataset, skipping: {missing}")
        else:
            # Use all repos that have both a graph and a question file
            self.repos = [
                r for r in available_repos
                if (self.questions_dir / f"{r}.jsonl").exists()
            ]

        self.repos.sort()
        print(f"Will evaluate {len(self.repos)} repos: {self.repos}")

    def _load_questions(self, repo: str) -> list[dict]:
        """Load questions for a given repo from its JSONL file."""
        path = self.questions_dir / f"{repo}.jsonl"
        questions = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        return questions

    def generate_answers(
        self,
        progress_bar: bool = True,
    ) -> Iterator[tuple[str, dict]]:
        """Yield (repo_name, answer_record) for every question across all repos.

        Each answer_record has keys: question, final_answer, retrieved_context.
        """
        for repo in self.repos:
            questions = self._load_questions(repo)
            graph: Data = self.dataset.get_by_name(repo)
            if graph is None:
                print(f"Warning: no graph for {repo}, skipping")
                continue

            inference = self.inference_cfg.create(graph=graph)

            bar = tqdm(questions, desc=repo) if progress_bar else questions
            for q in bar:
                question_text = q["question"]

                # Query dict compatible with OpenAIChatGenerator / _build_prompt_from_query_and_nodes
                query = {"prompt": question_text}

                try:
                    relevant_nodes = inference.retrieve(question_text)
                    result = inference.fusion.fuse_and_generate(
                        query=query, relevant_nodes=relevant_nodes,
                    )
                    answer = result.pop("answer")
                except Exception as e:
                    print(f"Error on question '{question_text[:60]}...': {e}")
                    answer = ""
                    relevant_nodes = []

                record = {
                    "question": question_text,
                    "final_answer": answer,
                    "ground_truth": q.get("ground_truth", ""),
                    "retrieved_context": [
                        {"file_path": str(n.file_path), "name": n.name, "code": n.code}
                        for n in relevant_nodes
                    ],
                }
                yield repo, record

    def generate_answers_per_repo(
        self,
        output_dir: Path,
        progress_bar: bool = True,
    ) -> None:
        """Write answer JSONL files per repo into output_dir."""
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Open file handles per repo
        handles: dict[str, object] = {}
        try:
            for repo, record in self.generate_answers(progress_bar=progress_bar):
                if repo not in handles:
                    handles[repo] = open(output_dir / f"{repo}.jsonl", "w", encoding="utf-8")
                handles[repo].write(json.dumps(record, ensure_ascii=False) + "\n")
                handles[repo].flush()
        finally:
            for fh in handles.values():
                fh.close()


class SWEQABenchConfig(BaseModel):
    """Config for SWE-QA-Bench evaluation."""

    inference: InferenceConfig
    dataset: TorchGraphDatasetConfig

    questions_dir: Path
    repos: list[str] | None = None

    def create(self) -> SWEQABenchInference:
        return SWEQABenchInference(
            dataset=self.dataset.create(),
            inference_cfg=self.inference,
            questions_dir=self.questions_dir,
            repos=self.repos,
        )
