import warnings
import os
import re
from pathlib import Path
from typing import Iterator, Literal, Dict, Union, Any

import pandas as pd
from pydantic import BaseModel
from torch_geometric.data import Data
from tqdm import tqdm

from ragc.datasets.train_dataset import TorchGraphDataset, TorchGraphDatasetConfig
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
    requirement_str = "\n".join([f"{k}: {v}" for k, v in requirement.items()])
    func_name = signature.split("(")[0].strip().split()[-1]  # crude but ok if signature is "def name(...):"

    prompt = f"""
You are a code generator. Output must follow the contract exactly.

CONTRACT
- Output exactly one fenced code block starting with ```python and ending with ```.
- Inside the code block, output ONLY the Python {completion_type} named {func_name}.
- Do NOT include imports, tests, helper functions, comments, docstrings, or any text outside the function.
- The signature must match exactly:
{signature}

CONTEXT
Path: {completion_path}
Namespace: {namespace}

REQUIREMENTS
{requirement_str}
""".strip()
    return prompt


def _debug_check_leak(
    task: Dict[str, Any],
    repos_path: Union[str, os.PathLike],
    signature: str,
    local_context: str,
) -> None:
    """Emit debug info to sanity-check that `local_context` does not leak the target body.

    Activated by setting the env var `RAGC_DEBUG_LEAK=1`. Checks:
      1. The target function name does not already appear inside `local_context`
         (if it does, the model can copy the definition directly).
      2. None of the lines from the ground-truth body (`body_position`) appear in
         `local_context`.
      3. Prints totals: file size, where the signature is, how many lines of
         local_context were produced, and the tail of local_context.
    """
    if not os.environ.get("RAGC_DEBUG_LEAK"):
        return

    namespace = task.get("namespace", "?")
    completion_path = task.get("completion_path", "?")
    signature_position = task.get("signature_position")
    body_position = task.get("body_position")

    try:
        with open(Path(repos_path) / completion_path, "r") as f:
            total_lines = len(f.read().split("\n"))
    except Exception:
        total_lines = -1

    local_lines = local_context.split("\n")

    print(f"\n[LEAK DEBUG] namespace={namespace}")
    print(f"  file={completion_path}  total_lines={total_lines}")
    print(f"  signature_position={signature_position}  body_position={body_position}")
    print(f"  signature={signature.strip()!r}")
    print(f"  local_context: {len(local_lines)} lines")

    # 1) function name leak
    m = re.search(r"def\s+(\w+)\s*\(", signature)
    if m:
        fname = m.group(1)
        hits = [i for i, line in enumerate(local_lines, start=1) if re.search(rf"\b{re.escape(fname)}\b", line)]
        if hits:
            print(f"  WARNING: target name {fname!r} appears in local_context at lines {hits[:10]}")
            for h in hits[:5]:
                print(f"    L{h}: {local_lines[h - 1].rstrip()}")

    # 2) ground-truth body leak
    if body_position and isinstance(body_position, (list, tuple)) and len(body_position) == 2:
        try:
            body = extract_lines(Path(repos_path) / completion_path, tuple(body_position))
            body_lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
            leaked = [bl for bl in body_lines if bl in local_context]
            if leaked:
                print(f"  WARNING: {len(leaked)}/{len(body_lines)} body lines also present in local_context")
                for bl in leaked[:5]:
                    print(f"    BODY: {bl}")
        except Exception as e:
            print(f"  (could not extract body for leak check: {e})")

    # 3) tail of local_context
    tail = local_lines[-20:]
    print(f"  --- last {len(tail)} lines of local_context ---")
    for line in tail:
        print(f"    {line}")
    print(f"  --- end ---\n")


def build_prompt(task: Dict[str, Any], repos_path: Union[str, os.PathLike]) -> Dict[str, str]:
    signature = extract_lines(repos_path / task["completion_path"], task["signature_position"])
    rel_path = task['completion_path'].replace(task['project_path'], '', 1).strip('/')

    prompt = completion_prompt(rel_path, signature, task['requirement'])
    local_context = extract_lines(repos_path / task["completion_path"], (1, task["signature_position"][0] - 1))

    _debug_check_leak(task, repos_path, signature, local_context)

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
