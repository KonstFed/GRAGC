import ast
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import networkx as nx
import black
from semantic_parser import SemanticGraphBuilder

# --- monkey-patch code2flow to handle all AST call-target types gracefully ---
import code2flow.python as _c2f_py

_c2f_orig_get_call = _c2f_py.get_call_from_func_element


def _patched_get_call_from_func_element(func):
    if type(func) not in (ast.Attribute, ast.Name, ast.Subscript, ast.Call):
        return None
    return _c2f_orig_get_call(func)


_c2f_py.get_call_from_func_element = _patched_get_call_from_func_element
# --- end monkey-patch ---

from ragc.graphs.common import (
    BaseGraphParser,
    BaseGraphParserConfig,
    Edge,
    EdgeType,
    Node,
    NodeType,
)
from ragc.graphs.ast_tools import extract_function_info, extract_class_info, to_zero_ident

COLOR2CLASS: dict[str, str] = {
    "green": "file",
    "blue": "class",
    "orange": "function",
}

NODE2TYPE: dict[str, str] = {
    "file": NodeType.FILE,
    "class": NodeType.CLASS,
    "function": NodeType.FUNCTION,
}

EDGE2TYPE: dict[str, str] = {
    "Encapsulation": EdgeType.OWNER,
    "Invoke": EdgeType.CALL,
    "Import": EdgeType.IMPORT,
    "Ownership": EdgeType.OWNER,
    "Class Hierarchy": EdgeType.INHERITED,
}


class SemanticParser(BaseGraphParser):
    def _clean(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Remove incorrect nodes."""
        bad_nodes = [n for n in graph.nodes() if " " in n.strip()]
        for node in bad_nodes:
            graph.remove_node(node)

        return graph

    def _process(self, graph: nx.MultiDiGraph, repo_path: Path) -> nx.MultiDiGraph:
        """Change `color` to type and assign `body` to file nodes."""
        node2delete = [node for node, attr in graph.nodes(data=True) if "color" not in attr]
        for node in node2delete:
            graph.remove_node(node)

        new_types = {node: COLOR2CLASS[attr["color"]] for node, attr in graph.nodes(data=True)}

        nx.set_node_attributes(graph, new_types, "type")

        file_nodes: list[str] = [node for node, attr in graph.nodes(data=True) if attr["type"] == "file"]

        new_values = {}
        for node in file_nodes:
            attr = graph.nodes(data=True)[node]
            with (repo_path / attr["file_path"]).open("r", errors="replace") as f:
                file_code = f.read()

            if "body" in graph.nodes(data=True)[node]:
                raise ValueError(graph.nodes(data=True)[node])

            new_values[node] = file_code

        nx.set_node_attributes(graph, new_values, "body")

        return graph

    def _relabel(self, graph: nx.MultiDiGraph, repo_path: Path) -> nx.MultiDiGraph:
        """Relabel node id by remobing .py and assign file_path to each node."""
        mapping = {}
        file_path_map = {}
        for node in graph.nodes:
            if node[:2] != "C.":
                _err = f"All nodes should start form 'C.'. Got instead {node}"
                raise ValueError(_err)
            new_node = node.removeprefix("C.")
            new_node = Path(new_node).relative_to(repo_path)

            # remove .py and extract file_path
            parts = list(new_node.parts)
            filename = list(filter(lambda p: p[1].endswith(".py"), enumerate(parts)))
            if len(filename) != 1:
                raise ValueError(f"incorrect path {new_node}")

            idx, filename = filename[0]
            file_path = Path(*parts[: idx + 1])

            parts[idx] = parts[idx].removesuffix(".py")

            mapping[node] = ".".join(parts)
            file_path_map[node] = file_path

        nx.set_node_attributes(graph, file_path_map, "file_path")
        nx.relabel_nodes(graph, mapping, copy=False)
        return graph

    def _fix_code_ident(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Add correct ident for first line of code snippets."""
        for _, attr in graph.nodes(data=True):
            if attr["type"] == "file":
                continue

            # fix that first line has different ident
            code = attr["body"]
            ident_pos = attr["start_point"][1]
            code = " " * ident_pos + code
            code = to_zero_ident(code)

            try:
                formatted_code = black.format_str(code, mode=black.Mode())
            except Exception as e:
                # if fails we just use unformatted
                formatted_code = code

            attr["body"] = formatted_code

        return graph

    def to_standart(self, graph: nx.MultiDiGraph, repo_path: Path) -> nx.MultiDiGraph:
        """Transform SemanticParser graph into standart."""
        graph = self._clean(graph=graph)
        graph = self._relabel(graph=graph, repo_path=repo_path)
        graph = self._process(graph=graph, repo_path=repo_path)
        graph = self._fix_code_ident(graph=graph)

        norm_graph = nx.MultiDiGraph()
        for node, attr in graph.nodes(data=True):
            code = attr["body"]

            signature = ""
            docstring = ""

            node_type = NODE2TYPE[attr["type"]]

            if node_type == NodeType.FUNCTION:
                try:
                    ext_signature, ext_docstring = extract_function_info(code)
                    signature = signature if ext_signature is None else ext_signature
                    docstring = docstring if ext_docstring is None else ext_docstring
                except Exception as e:
                    warnings.warn(f"Failed to extract function info {node}", stacklevel=1)
            elif node_type == NodeType.CLASS:
                try:
                    ext_docstring = extract_class_info(code)
                    docstring = docstring if ext_docstring is None else ext_docstring
                except Exception as e:
                    warnings.warn(f"Failed to extract class info {node}", stacklevel=1)

            new_node = Node(
                name=node,
                type=NODE2TYPE[attr["type"]],
                code=code,
                file_path=attr["file_path"],
                docstring=docstring,
                signature=signature,
            )
            norm_graph.add_node(node, **new_node.model_dump())

        for n_from, n_to, edge_data in graph.edges(data=True):
            new_edge = Edge(type=EDGE2TYPE[edge_data["type"]])
            norm_graph.add_edge(n_from, n_to, **new_edge.model_dump())
        return norm_graph

    def parse_raw(self, repo_path: Path) -> nx.MultiDiGraph:
        repo_path = repo_path.absolute()

        with TemporaryDirectory() as t:
            _builder = SemanticGraphBuilder()
            _builder.build_from_one(str(repo_path), t, gsave=True, gprint=False)

            tmp_dir = Path(t)
            graph_file = next(tmp_dir.iterdir())
            graph = nx.read_gml(graph_file)

        return graph


class SemanticParserConfig(BaseGraphParserConfig):
    type: Literal["python_parser"] = "python_parser"

    def create(self) -> SemanticParser:
        """Create instance of SemantcParser."""
        return SemanticParser(cache_path=self.cache_path)
