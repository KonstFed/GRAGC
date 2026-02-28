import warnings
from pathlib import Path

import networkx as nx
import torch
from pydantic import BaseModel, PrivateAttr, model_validator
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.transforms import Compose
from tqdm import tqdm

from ragc.graphs import GraphParserConfig
from ragc.graphs.types import Transform


class TorchGraphDataset(InMemoryDataset):
    """Dataset for training GNN."""

    def __init__(
        self,
        root: str,
        graphs: list[nx.MultiDiGraph] | None = None,
        graph_names: list[str] | None = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self._graphs = graphs
        self._graph_names = graph_names
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        self._graph_names = torch.load(Path(self.processed_dir) / "graph_names.pt")

    @property
    def processed_file_names(self):
        return ["original_graph.pt", "graph_names.pt"]

    def get_by_name(self, repo_name: str) -> Data | HeteroData:
        """Get graph by its name."""
        try:
            idx = self._graph_names.index(repo_name)
            return self[idx]
        except ValueError:
            return None

    def get_repos_names(self) -> list[str]:
        """Get all repos names."""
        return self._graph_names

    def process(self) -> None:
        data_list = self._graphs

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in tqdm(data_list)]

        if self.pre_filter is not None:
            data_list = list(filter(self.pre_filter, data_list))

        # Save the processed data
        self.save(data_list, self.processed_paths[0])
        torch.save(self._graph_names, Path(self.processed_dir) / self.processed_file_names[1])


class TorchGraphDatasetConfig(BaseModel):
    """Config for torch_geometric dataset."""

    root_path: Path
    repos_path: Path | None = None
    graphs_path: Path | None = None

    parser: GraphParserConfig | None = None

    pre_transform: list[Transform] | None = None
    transform: list[Transform] | None = None

    _repos: list[Path] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def _validate_general(self):
        if self.repos_path is not None and self.graphs_path is not None:
            raise ValueError("If you want to init graphs you should provide either repos_path or graphs_path not both")

        if self.repos_path is not None and self.parser is None:
            raise ValueError("parser should be provided if repos_path is provided")

        return self

    def _parse_graphs(self, repos2parse: list[Path]) -> tuple[list[nx.MultiDiGraph], list[str]]:
        """Parse graphs from repo. Logs all failed repos at the end."""
        parser = self.parser.create()
        graphs = []
        repo_names = []
        failed: list[tuple[Path, BaseException]] = []
        for repo_p in repos2parse:
            try:
                graph = parser.parse(repo_path=repo_p)
                graphs.append(graph)
                repo_names.append(repo_p.name)
            except Exception as e:
                failed.append((repo_p, e))
                warnings.warn(
                    f"Failed to parse {repo_p.absolute()} with error:\n{e}",
                    stacklevel=1,
                )
        if failed:
            print("\nRepos that failed to parse:", flush=True)  # noqa: T201
            for repo_p, err in failed:
                print(f"  {repo_p.absolute()}", flush=True)  # noqa: T201
                print(f"    -> {err!r}", flush=True)  # noqa: T201
            print(f"Total failed: {len(failed)} / {len(repos2parse)}\n", flush=True)  # noqa: T201
        return graphs, repo_names

    def _load_graphs(self) -> list[nx.MultiDiGraph]:
        """Read .gml graphs."""
        graphs = []
        for graph_p in self.graphs_path.iterdir():
            graph = nx.read_gml(graph_p)
            graphs.append(graph)
        return graphs

    def add_repo(self, repo_path: Path) -> None:
        """Add repo for processing."""
        self._repos.append(repo_path)

    def create(self) -> TorchGraphDataset:
        graphs = None
        graph_names = None
        if len(self._repos) != 0:
            graphs, graph_names = self._parse_graphs(self._repos)
        elif self.repos_path is not None:
            graphs, graph_names = self._parse_graphs(self.repos_path.iterdir())
        elif self.graphs_path is not None:
            graph_names = [p.stem for p in self.graphs_path.iterdir()]
            graphs = self._load_graphs()

        pre_transform = None
        if self.pre_transform is not None:
            pre_transform = Compose([t.create() for t in self.pre_transform])

        transform = None
        if self.transform is not None:
            transform = Compose([t.create() for t in self.transform])

        return TorchGraphDataset(
            root=self.root_path,
            graphs=graphs,
            graph_names=graph_names,
            transform=transform,
            pre_transform=pre_transform,
        )
