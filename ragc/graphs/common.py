from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Annotated

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseGraphParser(ABC):
    """Base class for parsing repo."""

    cache_path: Path | None = None

    def __init__(self, cache_path: Path | None = None):
        self.cache_path = cache_path

    def parse(self, repo_path: Path | None = None) -> nx.MultiDiGraph:
        """Parse repository into graph.

        Structure of graph is described using `Node` and `Edge` in the same file.

        Args:
            repo_path (Path): path to root of the repo

        Returns:
            nx.MultiDiGraph

        """
        if self.cache_path is not None and self.cache_path.exists():
            return read_graph(self.cache_path)

        if repo_path is not None:
            graph = self.parse_raw(repo_path=repo_path)
            graph = self.to_standart(graph=graph, repo_path=repo_path)

            if self.cache_path is not None:
                save_graph(graph=graph, save_path=self.cache_path)

            return graph

        raise ValueError("Should provide either repo_path or cache_path")

    @abstractmethod
    def parse_raw(self, repo_path: Path) -> nx.MultiDiGraph:
        """Get raw original graph from parser.

        Args:
            repo_path (Path): path to root of the repo

        Returns:
            nx.MultiDiGraph: raw original semantic graph

        """
        raise NotImplementedError

    @abstractmethod
    def to_standart(self, graph: nx.MultiDiGraph, repo_path: Path) -> nx.MultiDiGraph:
        """Transform original semantic graph into united format.

        Args:
            graph (nx.MultiDiGraph): original graph
            repo_path (Path): path to root of the repo

        Returns:
            nx.MultiDiGraph

        """
        raise NotImplementedError

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )


class BaseGraphParserConfig(BaseModel, ABC):
    cache_path: Path | None = None

    @abstractmethod
    def create(self) -> BaseGraphParser:
        """Create graph parser."""


class NodeType(Enum):
    CLASS = "class"
    FUNCTION = "function"
    FILE = "file"
    RESIDUAL = "residual"


    def __str__(self):
        return self.value


class NodeTypeNumeric(Enum):
    FUNCTION = 0
    CLASS = 1
    FILE = 2
    RESIDUAL = 3

    def __str__(self):
        return self.value

class EdgeType(Enum):
    IMPORT = "import"
    OWNER = "owner"
    CALL = "call"
    INHERITED = "inherited"

    def __str__(self):
        return self.value


class EdgeTypeNumeric(Enum):
    CALL = 0
    OWNER = 1
    IMPORT = 2
    INHERITED = 3

    def __str__(self):
        return self.value


class Node(BaseModel):
    name: Annotated[str, Field(pattern=r"^[^\s]+$")]
    type: NodeType
    docstring: str
    signature: str
    code: str
    file_path: Path

    @field_validator("file_path", mode="before")
    @classmethod
    def _normalize_file_path(cls, file_p: str) -> Path:
        return Path(file_p)

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )


class Edge(BaseModel):
    type: EdgeType

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )


def save_graph(graph: nx.Graph, save_path: Path) -> None:
    nx.write_gml(graph, save_path, stringizer=str)


def read_graph(read_path: Path) -> nx.Graph:
    graph = nx.read_gml(read_path)
    for _, attr in graph.nodes(data=True):
        attr["type"] = NodeType(attr["type"])
        attr["file_path"] = Path(attr["file_path"])
    return graph
