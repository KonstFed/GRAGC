import warnings
import os
from pathlib import Path
from typing import Iterator, Literal, Dict, Union, Any

import pandas as pd
from pydantic import BaseModel
from torch_geometric.data import Data
from tqdm import tqdm

from ragc.datasets.train_dataset import TorchGraphDataset, TorchGraphDatasetConfig
from ragc.simple_rag.dataset import CodeRepoDatasetConfig
from ragc.simple_rag.pipeline import SimpleRAGPipeline, SimpleRAGPipelineConfig
from ragc.graphs import Node
from ragc.graphs.transforms import MaskNodes
from ragc.graphs.utils import pyg_extract_node
from ragc.inference import InferenceConfig
from ragc.test.utils import extract_lines, map_cross_file_dependency


def completion_prompt(
    completion_path: str,
    signature: str,
    requirement: Dict[str, str],
) -> str:
    req_args = requirement['Arguments'].replace('\n', '\n    ')
    prompt = f"""#{completion_path}
{signature}
    \"\"\"{requirement['Functionality']}

    {req_args}
    \"\"\""""

    return prompt


def gpt_prompt(
    completion_path: str,
    namespace: str,
    signature: str,
    requirement: dict[str, str],
    completion_type: Literal["function", "method"],
) -> str:
    requirement_str = "\n".join([f"- **{key}**: {value}" for key, value in requirement.items()])
    prompt = f"""# TASK

You are to generate a Python {completion_type} implementation that strictly follows the instructions below. Return ONLY the code inside a ```python ``` block, no explanations. Do not forget about imports.

## Context Usage

Any supplementary context provided is retrieved automatically.
You should use it only if it is directly relevant and improves the implementation.
Disregard irrelevant or conflicting information.

## File Location

- **Path**: `{completion_path}`
- **Namespace**: `{namespace}`

## Requirements

{requirement_str}

## Signature to Implement

{signature}

Your code here:\n"""
    return prompt


def build_prompt(task: Dict[str, Any], repos_path: Union[str, os.PathLike]) -> Dict[str, str]:
    signature = extract_lines(repos_path / task["completion_path"], task["signature_position"])
    rel_path = task['completion_path'].replace(task['project_path'], '', 1).strip('/')

    prompt = completion_prompt(rel_path, signature, task['requirement'])
    local_context = extract_lines(repos_path / task["completion_path"], (1, task["signature_position"][0] - 1))
    return {'prompt': prompt, 'local_context': local_context, 'completion_path': rel_path}


def _get_correct_namespace(completion_path: str, project_path: str, namespace: str) -> dict:
    """Transform namespace of evocodebench into graph namespace."""
    file_path = Path(completion_path).relative_to(project_path)

    parts = list(file_path.parts)
    # remove python file
    parts[-1] = parts[-1].removesuffix(".py")
    if file_path.name == "__init__.py":
        parts = parts[:-1]

    namespace_parts = namespace.split(".")
    match_idx = 0
    for i in range(len(parts)):
        if parts[i] != namespace_parts[0]:
            continue

        is_match = True

        for j in range(i, len(parts)):
            if parts[j] != namespace_parts[j - i]:
                is_match = False
                break

        if is_match:
            match_idx = i
            break

    namespace = ".".join(parts[:match_idx] + namespace.split("."))

    if file_path.name == "__init__.py":
        _parts = list(file_path.parts)
        _parts[-1] = _parts[-1].removesuffix(".py")
        namespace = namespace.replace(".".join(file_path.parts[:-1]), ".".join(_parts))

    return namespace


class TestInference:
    def __init__(
        self,
        dataset: TorchGraphDataset,
        inference_cfg: InferenceConfig,
        task_path: Path,
        repos_path: Path,
        only_with_cross_file: bool = True,
        use_gold_context: bool = False,
    ):
        self.dataset = dataset
        self.inference_cfg = inference_cfg

        self.repos_path = repos_path
        self.task_path = task_path
        self.only_with_cross_file = only_with_cross_file
        self.use_gold_context = use_gold_context

        _task = pd.read_json(task_path, lines=True)
        _task = _task[_task["project_path"].apply(lambda x: x.split("/")[-1]).isin(dataset.get_repos_names())]

        if only_with_cross_file:
            _task = self._clean_dependecies(_task)

        self.tasks = _task

    def _clean_dependecies(self, tasks: pd.DataFrame) -> pd.DataFrame:
        def _present_in_graph(task) -> bool:
            repo_name = Path(task["project_path"]).parts[-1]
            namespace = _get_correct_namespace(task["completion_path"], task["project_path"], task["namespace"])

            graph = self._prepare_graph(repo_name=repo_name, node_namespace=namespace)

            gold_nodes = self._get_gold_snippets(task_row=task, graph=graph)
            return gold_nodes is not None

        tasks["cross_file"] = tasks["dependency"].apply(lambda x: x["cross_file"])
        tasks = tasks[tasks["cross_file"].apply(len) > 0]
        return tasks[tasks.apply(_present_in_graph, axis=1)]

    def _prepare_graph(self, repo_name: str, node_namespace: str) -> Data:
        """Get graph from node_namespace."""
        graph = self.dataset.get_by_name(repo_name)
        t = MaskNodes([node_namespace], mask_callee=True)
        graph = t(graph)
        return graph

    def retrieve(self, repo_name: str, prompt: str, node_namespace: str) -> list[Node]:
        graph = self._prepare_graph(repo_name, node_namespace)
        inference = self.inference_cfg.create(graph=graph)
        return inference.retrieve(prompt)

    def __call__(self, repo_name: str, prompt: str, node_namespace: str) -> tuple[str, dict]:
        graph = self._prepare_graph(repo_name, node_namespace)
        inference = self.inference_cfg.create(graph=graph)
        return inference(query=prompt)

    def _generate_with_golden(
        self, repo_name: str, prompt: str, node_namespace: str, golden_nodes: list[Node]
    ) -> tuple[str, dict]:
        graph = self._prepare_graph(repo_name, node_namespace)
        inference = self.inference_cfg.create(graph=graph)
        return inference.generate_with_context(query=prompt, nodes=golden_nodes)

    def generate_completion(self, progress_bar: bool = True) -> Iterator[dict[str, str]]:
        """Pipeline for evocodebench generation."""
        bar = tqdm(self.tasks.iterrows(), total=len(self.tasks)) if progress_bar else self.tasks.iterrows()
        for _, task in bar:
            namespace = _get_correct_namespace(task["completion_path"], task["project_path"], task["namespace"])
            repo_name = Path(task["project_path"]).parts[-1]

            prompt = build_prompt(
                task=task,
                repos_path=self.repos_path,
            )
            if not self.use_gold_context:
                generation, _meta = self(
                    repo_name=repo_name,
                    prompt=prompt,
                    node_namespace=namespace,
                )
            else:
                graph = self._prepare_graph(repo_name=repo_name, node_namespace=namespace)
                gold_nodes = self._get_gold_snippets(task_row=task, graph=graph)
                if gold_nodes is None:
                    # gold dependencies are not in our graph
                    warnings.warn(f"{task['namespace']} got cross file reference not found in graph.")
                    continue
                generation, _meta = self._generate_with_golden(
                    repo_name=repo_name,
                    prompt=prompt,
                    node_namespace=namespace,
                    golden_nodes=gold_nodes,
                )

            result = {
                "namespace": task["namespace"],
                "completion": generation,
            }
            yield result

    def _get_gold_snippets(self, task_row, graph: Data) -> list[Node] | None:
        indices = []
        for dependency in task_row["dependency"]["cross_file"]:
            cur_idx = map_cross_file_dependency(dependency, task_row["project_path"], graph)

            if cur_idx is None:
                return None

            indices.append(cur_idx)

        return pyg_extract_node(graph, indices)

    def generate_retrieval_pairs(self, progress_bar: bool = True) -> Iterator[tuple[list[str], list[str]]]:
        """Pipeline for evocodebench generation."""
        bar = tqdm(self.tasks.iterrows(), total=len(self.tasks)) if progress_bar else self.tasks.iterrows()
        for _, task in bar:
            repo_name = Path(task["project_path"]).parts[-1]
            namespace = _get_correct_namespace(task["completion_path"], task["project_path"], task["namespace"])

            graph = self._prepare_graph(repo_name=repo_name, node_namespace=namespace)

            gold_nodes = self._get_gold_snippets(task_row=task, graph=graph)
            if gold_nodes is None:
                # gold dependencies are not in our graph
                warnings.warn(f"{task['namespace']} got cross file reference not found in graph.")
                continue

            prompt = build_prompt(
                task=task,
                repos_path=self.repos_path,
            )
            retrieved_nodes = self.retrieve(
                repo_name=Path(task["project_path"]).parts[-1],
                prompt=prompt,
                node_namespace=namespace,
            )

            yield gold_nodes, retrieved_nodes


class TestInferenceConfig(BaseModel):
    """Cached inference for test metrics."""

    inference: InferenceConfig
    dataset: TorchGraphDatasetConfig

    task_path: Path
    repos_path: Path
    only_with_cross_file: bool = True
    use_gold_context: bool = False

    def create(self) -> TestInference:
        return TestInference(
            dataset=self.dataset.create(),
            inference_cfg=self.inference,
            task_path=self.task_path,
            repos_path=self.repos_path,
            only_with_cross_file=self.only_with_cross_file,
            use_gold_context=self.use_gold_context,
        )


class SimpleRAGInference:
    def __init__(
        self,
        dataset_cfg: CodeRepoDatasetConfig,
        pipeline_cfg: SimpleRAGPipelineConfig,
        task_path: Path,
        repos_path: Path,
    ):
        self.dataset_cfg = dataset_cfg
        self.inference_cfg = pipeline_cfg

        self.repos_path = repos_path
        self.task_path = task_path

        _task = pd.read_json(task_path, lines=True)

        self.tasks = _task

    def __call__(self, repo_name: str, prompt: str) -> tuple[str, dict]:
        ds = self.dataset_cfg.create(repo_name=repo_name)
        inference = self.inference_cfg.create(ds)
        return inference.query(question=prompt)

    def generate_completion(self, progress_bar: bool = True) -> Iterator[dict[str, str]]:
        """Pipeline for evocodebench generation."""
        bar = tqdm(self.tasks.iterrows(), total=len(self.tasks)) if progress_bar else self.tasks.iterrows()
        for _, task in bar:
            # namespace = _get_correct_namespace(task["completion_path"], task["project_path"], task["namespace"])
            repo_name = Path(task["project_path"]).parts[-1]
            # перенесли внутри SimpleRAGPipeline
            prompt = build_prompt(
                task=task,
                repos_path=self.repos_path,
            )["prompt"]
            print("AAAA", type(prompt))
            print(prompt)

            generation = self(repo_name=repo_name, prompt=prompt)
            result = {
                "namespace": task["namespace"],
                "completion": generation,
            }
            yield result


class SimpleRAGInferenceConfig(BaseModel):
    type: Literal["simple_rag_inference"] = "simple_rag_inference"
    dataset: CodeRepoDatasetConfig
    pipeline: SimpleRAGPipelineConfig

    task_path: Path
    repos_path: Path

    def create(self) -> SimpleRAGInference:
        return SimpleRAGInference(
            dataset_cfg=self.dataset,
            pipeline_cfg=self.pipeline,
            task_path=self.task_path,
            repos_path=self.repos_path,
        )

EvoCodeBenchInferenceConfig = TestInferenceConfig | SimpleRAGPipeline
