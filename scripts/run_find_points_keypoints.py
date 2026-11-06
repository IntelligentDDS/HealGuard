from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from loguru import logger
from tree_sitter_language_pack import get_parser

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agenticheal import load_dataset_records
from util import HEALBENCH_INSTANCE_IDS
from heal_repository_scope import (
    REPOSITORY_EXCLUDED_DIRECTORIES,
    is_repository_path_excluded,
    iter_repository_python_files,
)
import run_find_points_sinks as sink_finder
from util import load_instance_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = {
    "cache_dir": PROJECT_ROOT / "datasets" / "raw_repo_source_code",
    "output_dir": PROJECT_ROOT / "results" / "find_points",
    "graph_dir": PROJECT_ROOT / "results" / "find_points",
    "excluded_directories": REPOSITORY_EXCLUDED_DIRECTORIES,
    "node_types": {
        "identifier": "identifier",
        "function": "function_definition",
        "class": "class_definition",
        "assignment": {"assignment", "augmented_assignment", "named_expression"},
        "call": "call",
        "branch": {"if_statement", "elif_clause"},
        "if": {"if_statement", "elif_clause"},
        "loop": {"for_statement", "while_statement"},
        "for": "for_statement",
        "return": "return_statement",
        "raise": "raise_statement",
        "break": "break_statement",
        "continue": "continue_statement",
        "exit": {"return_statement", "raise_statement", "break_statement", "continue_statement"},
        "try": "try_statement",
        "with": "with_statement",
        "with_item": "with_item",
        "match": "match_statement",
        "compound": {"if_statement", "for_statement", "while_statement", "try_statement", "with_statement", "match_statement"},
        "clause": {"else_clause", "finally_clause", "except_clause", "case_clause"},
        "error": "ERROR",
        "ignored_block_children": {"comment", "function_definition", "class_definition"},
    },
    "cfg_edges": {
        "entry", "next", "true", "false", "loop_body", "loop_exit", "break", "continue",
        "try", "exception", "finally", "match_case", "body", "return", "raise", "call", "return_to_call",
    },
    "dfg_edges": {"def_use"},
    "key_point_kinds": {
        "function_entry", "function_normal_exit", "function_exception_exit", "variable_definition",
        "data_transform", "data_merge", "branch", "loop", "branch_entry", "branch_exit",
        "loop_entry", "loop_iteration", "loop_exit", "control_exit", "exception_region",
        "exception_handler_entry", "call_before", "call_after",
    },
}


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_excluded(relative_path: Path) -> bool:
    return is_repository_path_excluded(relative_path.parts)


def node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def walk_tree(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.named_children))


def identifiers(root, source: bytes) -> set[str]:
    return {node_text(node, source) for node in walk_tree(root) if node.type == CONFIG["node_types"]["identifier"]}


def decision_references(root, source: bytes) -> set[str]:
    references = set()
    for node in walk_tree(root):
        if node.type == "attribute":
            parent = node.parent
            if parent is not None and parent.type == "attribute" and parent.child_by_field_name("object") == node:
                continue
            if parent is not None and parent.type == "call" and parent.child_by_field_name("function") == node:
                continue
            references.add(node_text(node, source))
            continue
        if node.type != CONFIG["node_types"]["identifier"]:
            continue
        parent = node.parent
        if parent is not None and parent.type == "attribute":
            continue
        if parent is not None and parent.type == "call" and parent.child_by_field_name("function") == node:
            continue
        references.add(node_text(node, source))
    return references


def assigned_identifiers(node, source: bytes) -> set[str]:
    if node.type in CONFIG["node_types"]["assignment"]:
        left = node.child_by_field_name("left") or node.child_by_field_name("name")
        return identifiers(left, source) if left else set()
    if node.type == CONFIG["node_types"]["for"]:
        left = node.child_by_field_name("left")
        return identifiers(left, source) if left else set()
    if node.type == CONFIG["node_types"]["with"]:
        definitions = set()
        for child in walk_tree(node):
            if child.type == CONFIG["node_types"]["with_item"]:
                alias = child.child_by_field_name("alias")
                if alias:
                    definitions.update(identifiers(alias, source))
        return definitions
    return set()


def used_identifiers(node, source: bytes) -> set[str]:
    excluded_ranges = []
    if node.type in CONFIG["node_types"]["assignment"]:
        left = node.child_by_field_name("left") or node.child_by_field_name("name")
        if left and node.type != "augmented_assignment":
            excluded_ranges.append((left.start_byte, left.end_byte))
    elif node.type == CONFIG["node_types"]["for"]:
        left = node.child_by_field_name("left")
        body = node.child_by_field_name("body")
        if left:
            excluded_ranges.append((left.start_byte, left.end_byte))
        if body:
            excluded_ranges.append((body.start_byte, body.end_byte))
    elif node.type in CONFIG["node_types"]["compound"]:
        for field in ("body", "consequence", "alternative"):
            child = node.child_by_field_name(field)
            if child:
                excluded_ranges.append((child.start_byte, child.end_byte))
    return {
        node_text(child, source)
        for child in walk_tree(node)
        if child.type == CONFIG["node_types"]["identifier"]
        and not any(start <= child.start_byte < end for start, end in excluded_ranges)
    }


def call_references(node, source: bytes) -> list[dict]:
    calls = []
    for child in walk_tree(node):
        if child.type != "call":
            continue
        function = child.child_by_field_name("function")
        if function is None:
            continue
        names = [node_text(item, source) for item in walk_tree(function) if item.type in {"identifier", "attribute"}]
        terminal = next((node_text(item, source) for item in reversed(list(walk_tree(function))) if item.type == "identifier"), "")
        if terminal:
            calls.append({
                "name": terminal,
                "expression": node_text(function, source),
                "line": child.start_point.row + 1,
                "column": child.start_point.column,
            })
    return calls


def direct_statements(block) -> list:
    if block is None:
        return []
    if block.type != "block":
        return [block] if block.type not in CONFIG["node_types"]["ignored_block_children"] else []
    statements = [
        child for child in block.named_children
        if child.type not in CONFIG["node_types"]["ignored_block_children"]
    ]
    if statements:
        first = statements[0]
        expression = first.named_children[0] if first.type == "expression_statement" and first.named_children else first
        if expression.type in {"string", "concatenated_string"}:
            statements = statements[1:]
    return statements


def child_block(node, field: str) -> list:
    child = node.child_by_field_name(field)
    if child is None:
        return []
    if child.type == "elif_clause":
        return [child]
    if child.type in CONFIG["node_types"]["clause"]:
        child = child.child_by_field_name("body") or next((item for item in child.named_children if item.type == "block"), None)
    return direct_statements(child)


def named_child_blocks(node, child_type: str) -> list[list]:
    blocks = []
    for child in node.named_children:
        if child.type != child_type:
            continue
        block = child.child_by_field_name("body") or next((item for item in child.named_children if item.type == "block"), None)
        blocks.append(direct_statements(block))
    return blocks


class CFGBuilder:
    def __init__(self, relative_file: str, qualified_name: str, function, source: bytes):
        self.relative_file = relative_file
        self.qualified_name = qualified_name
        self.function = function
        self.source = source
        self.statements = []
        self.statement_ids = {}
        self.cfg_edges = []
        prefix = f"{relative_file}::{qualified_name}"
        self.entry_id = f"{prefix}::<entry>"
        self.normal_exit_id = f"{prefix}::<normal_exit>"
        self.exception_exit_id = f"{prefix}::<exception_exit>"
        self.collect_statements(direct_statements(function.child_by_field_name("body")))

    def collect_statements(self, statements: list) -> None:
        for statement in statements:
            if statement.id in self.statement_ids:
                continue
            statement_id = self.make_statement_id(statement)
            self.statement_ids[statement.id] = statement_id
            self.statements.append(statement)
            for block in self.nested_blocks(statement):
                self.collect_statements(block)

    def nested_blocks(self, statement) -> list[list]:
        blocks = []
        for field in ("body", "consequence", "alternative"):
            block = child_block(statement, field)
            if block:
                blocks.append(block)
        for child_type in ("except_clause", "finally_clause", "case_clause"):
            blocks.extend(block for block in named_child_blocks(statement, child_type) if block)
        return blocks

    def make_statement_id(self, statement) -> str:
        return (
            f"{self.relative_file}::{self.qualified_name}::"
            f"{statement.start_point.row + 1}:{statement.start_point.column}:{statement.type}"
        )

    def ids(self, statements: list) -> list[str]:
        return [self.statement_ids[statement.id] for statement in statements]

    def add_edges(self, source_id: str, target_ids: list[str], edge_type: str, detail: str = "") -> None:
        for target_id in target_ids:
            self.cfg_edges.append({"source_id": source_id, "target_id": target_id, "edge_type": edge_type, "detail": detail})

    def build_block(self, statements: list, follow: list, break_targets=None, continue_targets=None) -> list:
        entries = follow
        for statement in reversed(statements):
            entries = self.build_statement(statement, entries, break_targets or [], continue_targets or [])
        return entries

    def build_statement(self, statement, follow: list, break_targets: list, continue_targets: list) -> list:
        source_id = self.statement_ids[statement.id]
        if statement.type == CONFIG["node_types"]["return"]:
            self.add_edges(source_id, [self.normal_exit_id], "cfg_return")
            return [statement]
        if statement.type == CONFIG["node_types"]["raise"]:
            self.add_edges(source_id, [self.exception_exit_id], "cfg_raise")
            return [statement]
        if statement.type == CONFIG["node_types"]["break"]:
            self.add_edges(source_id, self.ids(break_targets), "cfg_break")
            return [statement]
        if statement.type == CONFIG["node_types"]["continue"]:
            self.add_edges(source_id, self.ids(continue_targets), "cfg_continue")
            return [statement]
        if statement.type in CONFIG["node_types"]["if"]:
            body = self.build_block(child_block(statement, "consequence"), follow, break_targets, continue_targets)
            alternative = self.build_block(child_block(statement, "alternative"), follow, break_targets, continue_targets)
            condition = statement.child_by_field_name("condition")
            detail = node_text(condition, self.source) if condition else ""
            self.add_edges(source_id, self.ids(body) or [self.normal_exit_id], "cfg_true", detail)
            self.add_edges(source_id, self.ids(alternative or follow) or [self.normal_exit_id], "cfg_false", detail)
            return [statement]
        if statement.type in CONFIG["node_types"]["loop"]:
            body = self.build_block(child_block(statement, "body"), [statement], follow, [statement])
            alternative = self.build_block(child_block(statement, "alternative"), follow, break_targets, continue_targets)
            condition = statement.child_by_field_name("condition") or statement.child_by_field_name("right")
            detail = node_text(condition, self.source) if condition else ""
            self.add_edges(source_id, self.ids(body), "cfg_loop_body", detail)
            self.add_edges(source_id, self.ids(alternative or follow) or [self.normal_exit_id], "cfg_loop_exit", detail)
            return [statement]
        if statement.type == CONFIG["node_types"]["try"]:
            final_blocks = named_child_blocks(statement, "finally_clause")
            final_entries = self.build_block(final_blocks[0], follow, break_targets, continue_targets) if final_blocks else follow
            body = self.build_block(child_block(statement, "body"), final_entries, break_targets, continue_targets)
            self.add_edges(source_id, self.ids(body) or self.ids(final_entries) or [self.normal_exit_id], "cfg_try")
            for handler in named_child_blocks(statement, "except_clause"):
                entries = self.build_block(handler, final_entries, break_targets, continue_targets)
                self.add_edges(source_id, self.ids(entries) or self.ids(final_entries) or [self.normal_exit_id], "cfg_exception")
            if final_blocks:
                self.add_edges(source_id, self.ids(final_entries), "cfg_finally")
            return [statement]
        if statement.type == CONFIG["node_types"]["match"]:
            cases = named_child_blocks(statement, "case_clause")
            for index, case in enumerate(cases):
                entries = self.build_block(case, follow, break_targets, continue_targets)
                self.add_edges(source_id, self.ids(entries or follow) or [self.normal_exit_id], "cfg_match_case", str(index))
            return [statement]
        if statement.type == CONFIG["node_types"]["with"]:
            body = self.build_block(child_block(statement, "body"), follow, break_targets, continue_targets)
            self.add_edges(source_id, self.ids(body or follow) or [self.normal_exit_id], "cfg_body")
            return [statement]
        self.add_edges(source_id, self.ids(follow) or [self.normal_exit_id], "cfg_next")
        return [statement]

    def parameters(self) -> set[str]:
        parameters = self.function.child_by_field_name("parameters")
        return identifiers(parameters, self.source) if parameters else set()

    def graph_nodes(self) -> list[dict]:
        function_line = self.function.start_point.row + 1
        end_line = self.function.end_point.row + 1
        nodes = [
            {"node_id": self.entry_id, "function": self.qualified_name, "node_type": "function_entry", "line": function_line, "column": 0, "end_line": function_line, "end_column": 0},
            {"node_id": self.normal_exit_id, "function": self.qualified_name, "node_type": "function_normal_exit", "line": end_line, "column": 0, "end_line": end_line, "end_column": 0},
            {"node_id": self.exception_exit_id, "function": self.qualified_name, "node_type": "function_exception_exit", "line": end_line, "column": 0, "end_line": end_line, "end_column": 0},
        ]
        nodes.extend({
            "node_id": self.statement_ids[statement.id], "function": self.qualified_name,
            "node_type": statement.type, "line": statement.start_point.row + 1,
            "column": statement.start_point.column, "end_line": statement.end_point.row + 1,
            "end_column": statement.end_point.column,
        } for statement in self.statements)
        return nodes

    def build(self) -> tuple[list[dict], dict]:
        entries = self.build_block(direct_statements(self.function.child_by_field_name("body")), [])
        self.add_edges(self.entry_id, self.ids(entries) or [self.normal_exit_id], "cfg_entry")
        context = {
            "relative_file": self.relative_file, "qualified_name": self.qualified_name,
            "function": self.function, "source": self.source, "statements": self.statements,
            "statement_ids": self.statement_ids, "entry_id": self.entry_id,
            "normal_exit_id": self.normal_exit_id, "exception_exit_id": self.exception_exit_id,
            "graph_nodes": self.graph_nodes(),
        }
        return self.cfg_edges, context


class DFGBuilder:
    def __init__(self, context: dict, cfg_edges: list[dict]):
        self.context = context
        self.cfg_edges = cfg_edges

    def parameters(self) -> set[str]:
        parameters = self.context["function"].child_by_field_name("parameters")
        return identifiers(parameters, self.context["source"]) if parameters else set()

    def build(self) -> list[dict]:
        statements_by_id = {self.context["statement_ids"][node.id]: node for node in self.context["statements"]}
        predecessors = {statement_id: set() for statement_id in statements_by_id}
        for edge in self.cfg_edges:
            if edge["source_id"] in statements_by_id and edge["target_id"] in statements_by_id:
                predecessors[edge["target_id"]].add(edge["source_id"])
        reaching_out = {statement_id: {} for statement_id in statements_by_id}
        changed = True
        while changed:
            changed = False
            for statement_id, statement in statements_by_id.items():
                incoming = {name: {self.context["entry_id"]} for name in self.parameters()}
                for predecessor in predecessors[statement_id]:
                    for name, definitions in reaching_out[predecessor].items():
                        incoming.setdefault(name, set()).update(definitions)
                outgoing = {name: set(definitions) for name, definitions in incoming.items()}
                for name in assigned_identifiers(statement, self.context["source"]):
                    outgoing[name] = {statement_id}
                if outgoing != reaching_out[statement_id]:
                    reaching_out[statement_id] = outgoing
                    changed = True
        edges = []
        for statement_id, statement in statements_by_id.items():
            incoming = {name: {self.context["entry_id"]} for name in self.parameters()}
            for predecessor in predecessors[statement_id]:
                for name, definitions in reaching_out[predecessor].items():
                    incoming.setdefault(name, set()).update(definitions)
            for name in sorted(used_identifiers(statement, self.context["source"])):
                for definition_id in sorted(incoming.get(name, set())):
                    edges.append({"source_id": definition_id, "target_id": statement_id, "edge_type": "dfg_def_use", "detail": name})
        return edges



class KeyPointBuilder:
    def __init__(self, context: dict, cfg_edges: list[dict], dfg_edges: list[dict]):
        self.context = context
        self.cfg_edges = cfg_edges
        self.dfg_edges = dfg_edges

    @staticmethod
    def kinds(statement, definitions: set[str], uses: set[str], calls: list[dict]) -> list[str]:
        kinds = []
        if definitions:
            kinds.append("variable_definition")
            if uses:
                kinds.append("data_transform")
            if len(uses) > 1:
                kinds.append("data_merge")
        if statement.type in CONFIG["node_types"]["branch"]:
            kinds.append("branch")
        if statement.type in CONFIG["node_types"]["loop"]:
            kinds.append("loop")
        if statement.type in CONFIG["node_types"]["exit"]:
            kinds.append("control_exit")
        if statement.type == CONFIG["node_types"]["try"]:
            kinds.append("exception_region")
        if calls:
            kinds.append("call_boundary")
        return kinds

    def synthetic_point(self, point_id: str, kind: str, line: int) -> dict:
        return {
            "point_id": point_id, "function": self.context["qualified_name"], "line": line, "column": 0,
            "end_line": line, "end_column": 0, "node_type": "synthetic", "definitions": [],
            "uses": [], "calls": [], "key_point_kinds": [kind],
        }

    def build(self) -> tuple[list[dict], list[dict]]:
        function = self.context["function"]
        source = self.context["source"]
        start_line = function.start_point.row + 1
        end_line = function.end_point.row + 1
        points = [
            self.synthetic_point(self.context["entry_id"], "function_entry", start_line),
            self.synthetic_point(self.context["normal_exit_id"], "function_normal_exit", end_line),
            self.synthetic_point(self.context["exception_exit_id"], "function_exception_exit", end_line),
        ]
        call_sites = []
        for statement in self.context["statements"]:
            definitions = assigned_identifiers(statement, source)
            uses = used_identifiers(statement, source)
            calls = call_references(statement, source)
            kinds = self.kinds(statement, definitions, uses, calls)
            if not kinds:
                continue
            statement_id = self.context["statement_ids"][statement.id]
            points.append({
                "point_id": statement_id, "function": self.context["qualified_name"],
                "line": statement.start_point.row + 1, "column": statement.start_point.column,
                "end_line": statement.end_point.row + 1, "end_column": statement.end_point.column,
                "node_type": statement.type, "definitions": sorted(definitions), "uses": sorted(uses),
                "calls": calls, "key_point_kinds": kinds,
            })
            for call in calls:
                call_sites.append({"caller_id": self.context["qualified_name"], "statement_id": statement_id, **call})
        points.sort(key=lambda item: (item["line"], item["column"], item["point_id"]))
        return points, call_sites


def parse_repository(source_root: Path) -> tuple[dict[str, bytes], dict[str, Any], list[dict]]:
    parser = get_parser("python")
    sources = {}
    trees = {}
    gaps = []
    for path in iter_repository_python_files(source_root):
        relative_file = path.relative_to(source_root).as_posix()
        try:
            source = path.read_bytes()
            tree = parser.parse(source)
        except (OSError, RuntimeError) as error:
            logger.warning(f"Tree-sitter parse failed for {relative_file}: {error}")
            gaps.append({"file": relative_file, "error_type": type(error).__name__, "error": str(error)})
            continue
        error_nodes = [node for node in walk_tree(tree.root_node) if node.type == CONFIG["node_types"]["error"] or node.is_missing]
        if error_nodes:
            try:
                compile(source, relative_file, "exec")
            except (SyntaxError, ValueError, TypeError) as error:
                logger.warning(f"Python compilation failed for parsed source {relative_file}: {type(error).__name__}: {error}")
                gaps.append({
                    "file": relative_file, "error_type": "TreeSitterParseError",
                    "error": f"{type(error).__name__}: {error}",
                    "error_ranges": [
                        {"line": node.start_point.row + 1, "column": node.start_point.column,
                         "end_line": node.end_point.row + 1, "end_column": node.end_point.column}
                        for node in error_nodes
                    ],
                })
                continue
            coordinate_source = source.replace(b":=}", b"!r}")
            fallback_tree = parser.parse(coordinate_source)
            fallback_errors = [
                node for node in walk_tree(fallback_tree.root_node)
                if node.type == CONFIG["node_types"]["error"] or node.is_missing
            ]
            if fallback_errors:
                gaps.append({
                    "file": relative_file,
                    "error_type": "TreeSitterCoordinateFallbackFailed",
                    "error_ranges": [
                        {"line": node.start_point.row + 1, "column": node.start_point.column,
                         "end_line": node.end_point.row + 1, "end_column": node.end_point.column}
                        for node in fallback_errors
                    ],
                })
                continue
            tree = fallback_tree
            logger.warning(
                f"Used equal-length debug-f-string coordinate fallback for {relative_file}; "
                f"first 2 original errors: {[(node.start_point.row + 1, node.start_point.column) for node in error_nodes[:2]]}"
            )
        sources[relative_file] = source
        trees[relative_file] = tree
    return sources, trees, gaps


def function_index(trees: dict[str, Any], sources: dict[str, bytes]) -> dict[str, tuple[str, str, str, Any]]:
    functions = {}
    for relative_file, tree in trees.items():
        module = relative_file[:-3].replace("/", ".") if relative_file.endswith(".py") else relative_file.replace("/", ".")

        def visit(node, parents: list[str]) -> None:
            for child in node.named_children:
                definition = child
                if child.type == "decorated_definition":
                    definition = next((
                        item for item in child.named_children
                        if item.type in {CONFIG["node_types"]["class"], CONFIG["node_types"]["function"]}
                    ), child)
                if definition.type == CONFIG["node_types"]["class"]:
                    name_node = definition.child_by_field_name("name")
                    name = node_text(name_node, sources[relative_file]) if name_node else "<anonymous>"
                    visit(definition.child_by_field_name("body") or definition, [*parents, name])
                elif definition.type == CONFIG["node_types"]["function"]:
                    name_node = definition.child_by_field_name("name")
                    name = node_text(name_node, sources[relative_file]) if name_node else "<anonymous>"
                    qualified_name = ".".join([*parents, name])
                    location = f"{definition.start_point.row + 1}:{definition.start_point.column}"
                    identity = f"{qualified_name}@{location}"
                    functions[f"{relative_file}::{identity}"] = (relative_file, qualified_name, identity, definition)
                    body = definition.child_by_field_name("body")
                    if body:
                        visit(body, [*parents, name])
                else:
                    visit(child, parents)

        visit(tree.root_node, [module])
    return functions


def add_interprocedural_edges(files: dict[str, dict], functions: dict, call_sites: list[dict]) -> list[dict]:
    by_name = defaultdict(list)
    entries = {}
    exits = {}
    for key, (relative_file, qualified_name, _) in functions.items():
        by_name[qualified_name.rsplit(".", 1)[-1]].append(key)
        entries[key] = f"{relative_file}::{identity}::<entry>"
        exits[key] = f"{relative_file}::{identity}::<normal_exit>"
    gaps = []
    for call in call_sites:
        candidates = by_name.get(call["name"], [])
        if not candidates:
            continue
        caller_file = call["statement_id"].split("::", 1)[0]
        caller_parent = call["caller_id"].rsplit(".", 1)[0]
        same_parent = [key for key in candidates if functions[key][0] == caller_file and functions[key][1] == f"{caller_parent}.{call['name']}"]
        same_file = [key for key in candidates if functions[key][0] == caller_file]
        receiver = call["expression"].rsplit(".", 1)[0] if "." in call["expression"] else ""
        receiver_matches = [key for key in same_file if receiver and receiver not in {"self", "cls"} and receiver.lower() in functions[key][1].lower()]
        resolved = same_parent or receiver_matches or (same_file if len(same_file) == 1 else []) or (candidates if len(candidates) == 1 else [])
        if len(resolved) != 1:
            gaps.append({
                "reason": "ambiguous_call", "caller": call["caller_id"], "statement_id": call["statement_id"],
                "expression": call["expression"], "candidate_count": len(candidates),
            })
            continue
        callee = resolved[0]
        callee_file = functions[callee][0]
        files[caller_file]["cfg_edges"].append({
            "source_id": call["statement_id"], "target_id": entries[callee], "edge_type": "cfg_call",
            "detail": call["expression"], "confidence": 1.0, "ambiguity": 1,
            "resolution": "exact", "provenance": "tree_sitter",
        })
        files[callee_file]["cfg_edges"].append({
            "source_id": exits[callee], "target_id": call["statement_id"], "edge_type": "cfg_return_to_call",
            "detail": call["expression"], "confidence": 1.0, "ambiguity": 1,
            "resolution": "exact", "provenance": "tree_sitter",
        })
    return gaps


def _statement_index(tree) -> tuple[dict[tuple[int, int], list], dict[int, list]]:
    excluded = {"module", "block", "function_definition", "class_definition", "parameters"}
    by_start: dict[tuple[int, int], list] = defaultdict(list)
    by_line: dict[int, list] = defaultdict(list)
    for node in walk_tree(tree.root_node):
        if node.type in excluded:
            continue
        line = node.start_point.row + 1
        by_start[(line, node.start_point.column)].append(node)
        by_line[line].append(node)
    return by_start, by_line


def _statement_node_for_location(index: tuple[dict, dict], line: int, column: int):
    by_start, by_line = index
    candidates = by_start.get((line, column), []) or by_line.get(line, [])
    if not candidates:
        return None
    node = min(candidates, key=lambda item: (item.end_byte - item.start_byte, item.start_byte))
    while node is not None and not (node.type.endswith("_statement") or node.type.endswith("_clause")):
        node = node.parent
    return node


def _decision_node(statement):
    fields = {
        "if_statement": "condition", "elif_clause": "condition", "while_statement": "condition",
        "for_statement": "right", "except_clause": "value",
    }
    field = fields.get(statement.type)
    return statement.child_by_field_name(field) if field else None


def _decision_mode(statement) -> str:
    if statement.type == "except_clause":
        return "exception"
    if statement.type == "for_statement":
        return "iteration"
    if statement.type == "while_statement":
        return "while"
    return "truth"


def enclosing_function_for_node(node, functions: list[tuple[str, str, Any]], relative_file: str) -> tuple[str, str]:
    parent = node.parent
    functions_by_node = {
        function.id: (qualified_name, identity)
        for qualified_name, identity, function in functions
    }
    while parent is not None:
        if parent.type == CONFIG["node_types"]["function"]:
            return functions_by_node.get(parent.id, ("", ""))
        parent = parent.parent
    module_name = f"{relative_file}::<module>"
    return module_name, module_name


def _codeql_key_point(node: dict, statement_index: tuple[dict, dict], source: bytes) -> dict | None:
    statement = _statement_node_for_location(statement_index, int(node["line"]), int(node["column"]))
    if statement is None:
        return None
    line_start = source.rfind(b"\n", 0, statement.start_byte) + 1
    if source[line_start:statement.start_byte].strip():
        return None
    kinds = []
    if statement.type in CONFIG["node_types"]["branch"]:
        kinds.append("branch")
    if statement.type in CONFIG["node_types"]["loop"]:
        kinds.append("loop")
    if statement.type in CONFIG["node_types"]["exit"]:
        kinds.append("control_exit")
    if statement.type == "except_clause":
        kinds.append("exception_handler")
    if not kinds:
        return None
    decision = _decision_node(statement)
    decision_names = sorted(decision_references(decision, source)) if decision is not None else []
    return {
        "point_id": node["node_id"], "function": node.get("function", ""),
        "line": statement.start_point.row + 1, "column": statement.start_point.column,
        "end_line": statement.end_point.row + 1,
        "end_column": statement.end_point.column, "node_type": statement.type,
        "key_point_level": 2 if decision is not None else 1,
        **({"decision_expression": node_text(decision, source), "decision_mode": _decision_mode(statement)} if decision is not None else {}),
        **({"decision_names": decision_names} if decision_names else {}),
        "key_point_kinds": sorted(set(kinds)),
    }


def validate_file_key_points(relative_file: str, tree, points: list[dict]) -> list[dict]:
    gaps = []
    nodes = list(walk_tree(tree.root_node))
    point_ids = [point["point_id"] for point in points]
    duplicate_ids = sorted(point_id for point_id, count in Counter(point_ids).items() if count > 1)
    for point_id in duplicate_ids:
        gaps.append({"reason": "duplicate_point_id", "file": relative_file, "point_id": point_id})

    expected_functions = {
        (node.start_point.row + 1, node.start_point.column)
        for node in nodes if node.type == CONFIG["node_types"]["function"]
    }
    function_entries = {
        (point["line"], point["column"])
        for point in points if "function_entry" in point.get("key_point_kinds", [])
    }
    for location in sorted(expected_functions - function_entries):
        gaps.append({"reason": "function_entry_missing", "file": relative_file, "location": location})
    for location in sorted(function_entries - expected_functions):
        gaps.append({"reason": "function_entry_without_source_function", "file": relative_file, "location": location})

    expected_decisions = {
        (node.start_point.row + 1, node.start_point.column, node.type)
        for node in nodes if _decision_node(node) is not None or node.type == "except_clause"
    }
    decision_counts = Counter(
        (point["line"], point["column"], point["node_type"])
        for point in points if point.get("key_point_level") == 2
    )
    for location in sorted(expected_decisions - set(decision_counts)):
        gaps.append({"reason": "decision_point_missing", "file": relative_file, "location": location})
    for location in sorted(set(decision_counts) - expected_decisions):
        gaps.append({"reason": "decision_point_without_source_decision", "file": relative_file, "location": location})
    for location, count in sorted(decision_counts.items()):
        if count != 1:
            gaps.append({
                "reason": "decision_point_not_unique", "file": relative_file,
                "location": location, "count": count,
            })

    return gaps


def build_path_key_points(source_root: Path, codeql_graph: dict) -> dict:
    sources, trees, parse_gaps = parse_repository(source_root)
    codeql_graph["files"] = {
        relative_file: graph_file
        for relative_file, graph_file in codeql_graph.get("files", {}).items()
        if not is_repository_path_excluded(Path(relative_file).parts)
    }
    functions_by_file: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for relative_file, qualified_name, identity, function in function_index(trees, sources).values():
        functions_by_file[relative_file].append((qualified_name, identity, function))
    files = {}
    mapping_gaps = []
    for relative_file, graph_file in codeql_graph.get("files", {}).items():
        if relative_file not in trees or relative_file not in sources:
            mapping_gaps.append({"reason": "codeql_file_not_parsed", "file": relative_file})
            continue
        points_by_statement = {}
        statement_index = _statement_index(trees[relative_file])
        for node in graph_file.get("nodes", []):
            point = _codeql_key_point(node, statement_index, sources[relative_file])
            if point is None:
                continue
            key = (point["line"], point["column"], point["node_type"])
            existing = points_by_statement.get(key)
            if existing is None:
                point.pop("point_id")
                point["point_id"] = (
                    f"{relative_file}::{point['function']}::"
                    f"{point['line']}:{point['column']}:{point['node_type']}"
                )
                points_by_statement[key] = point
                continue
            existing["key_point_kinds"] = sorted(set(existing["key_point_kinds"]) | set(point["key_point_kinds"]))
        points = list(points_by_statement.values())
        existing_decisions = {
            (point["line"], point["column"], point["node_type"])
            for point in points if point.get("key_point_level") == 2
        }
        for statement in walk_tree(trees[relative_file].root_node):
            if statement.type != "except_clause" or _decision_node(statement) is not None:
                continue
            key = (statement.start_point.row + 1, statement.start_point.column, statement.type)
            if key in existing_decisions:
                continue
            function, _ = enclosing_function_for_node(
                statement, functions_by_file.get(relative_file, []), relative_file,
            )
            points.append({
                "point_id": f"{relative_file}::{function}::{key[0]}:{key[1]}:{statement.type}",
                "function": function, "line": key[0], "column": key[1],
                "end_line": statement.end_point.row + 1, "end_column": statement.end_point.column,
                "node_type": statement.type, "key_point_level": 2,
                "decision_mode": "exception", "key_point_kinds": ["exception_handler"],
            })
            existing_decisions.add(key)
        for statement in walk_tree(trees[relative_file].root_node):
            decision = _decision_node(statement)
            if decision is None:
                continue
            key = (statement.start_point.row + 1, statement.start_point.column, statement.type)
            if key in existing_decisions:
                continue
            function, _ = enclosing_function_for_node(
                statement, functions_by_file.get(relative_file, []), relative_file,
            )
            decision_names = sorted(decision_references(decision, sources[relative_file]))
            point = {
                "point_id": (
                    f"{relative_file}::{function}::{key[0]}:{key[1]}:{statement.type}"
                ),
                "function": function, "line": key[0], "column": key[1],
                "end_line": statement.end_point.row + 1, "end_column": statement.end_point.column,
                "node_type": statement.type, "key_point_level": 2,
                "decision_expression": node_text(decision, sources[relative_file]),
                "decision_mode": _decision_mode(statement),
                "key_point_kinds": [
                    "exception_handler" if statement.type == "except_clause"
                    else "loop" if statement.type in CONFIG["node_types"]["loop"]
                    else "branch"
                ],
            }
            if decision_names:
                point["decision_names"] = decision_names
            points.append(point)
        existing_control_exits = {
            (point["line"], point["column"], point["node_type"])
            for point in points if "control_exit" in point.get("key_point_kinds", [])
        }
        for statement in walk_tree(trees[relative_file].root_node):
            if statement.type not in CONFIG["node_types"]["exit"]:
                continue
            key = (statement.start_point.row + 1, statement.start_point.column, statement.type)
            if key in existing_control_exits:
                continue
            function, _ = enclosing_function_for_node(
                statement, functions_by_file.get(relative_file, []), relative_file,
            )
            points.append({
                "point_id": f"{relative_file}::{function}::{key[0]}:{key[1]}:{statement.type}",
                "function": function, "line": key[0], "column": key[1],
                "end_line": statement.end_point.row + 1, "end_column": statement.end_point.column,
                "node_type": statement.type, "key_point_level": 1,
                "key_point_kinds": ["control_exit"],
            })
            existing_control_exits.add(key)
        for qualified_name, identity, function in functions_by_file.get(relative_file, []):
            entry_line = function.start_point.row + 1
            points.append({
                "point_id": f"{relative_file}::{identity}::<entry>",
                "function": qualified_name, "line": entry_line, "column": function.start_point.column,
                "end_line": entry_line, "end_column": function.start_point.column, "node_type": "synthetic",
                "key_point_level": 1, "key_point_kinds": ["function_entry"],
            })
        unique_points = {}
        for point in points:
            unique_points.setdefault(point["point_id"], point)
        points = list(unique_points.values())
        mapping_gaps.extend(validate_file_key_points(relative_file, trees[relative_file], points))
        file_record = {
            "source_sha256": hashlib.sha256(sources[relative_file]).hexdigest(),
            "key_points_by_line": {}, "graph_nodes": graph_file.get("nodes", []),
            "cfg_edges": graph_file.get("cfg_edges", []), "dfg_edges": graph_file.get("dfg_edges", []),
        }
        for point in sorted(points, key=lambda item: (item["line"], item["column"], item["point_id"])):
            file_record["key_points_by_line"].setdefault(str(point["line"]), []).append(point)
        files[relative_file] = file_record
    for relative_file in sorted(set(sources) - set(codeql_graph.get("files", {}))):
        if not sources[relative_file].strip():
            continue
        tree = trees[relative_file]
        meaningful_nodes = [
            node for node in tree.root_node.named_children
            if node.type not in {"comment", "string", "concatenated_string"}
            and not (
                node.type == "expression_statement"
                and node.named_children
                and node.named_children[0].type in {"string", "concatenated_string"}
            )
        ]
        if meaningful_nodes:
            mapping_gaps.append({"reason": "codeql_file_missing", "file": relative_file})
    return {
        "parser": "codeql", "coordinate_validator": "tree_sitter", "scope": "full_repository",
        "files": files, "repository_parse_gaps": parse_gaps, "analysis_gaps": mapping_gaps,
    }


def write_path_key_points(source_root: Path, _legacy_files: list[str], output_path: Path, _legacy_functions: set[str] | None = None) -> dict:
    payload = build_path_key_points(source_root)
    json_dump(output_path, payload)
    return payload


def cached_records(cache_root: Path, source: str | None, per_dataset_limit: int, instance_file: str | None) -> list[dict]:
    _, records = load_dataset_records(source)
    before = len(records)
    records = [record for record in records if record["instance_id"] in HEALBENCH_INSTANCE_IDS]
    logger.info(f"Excluded {before - len(records)} records not present in the benchmark manifest")
    selected_ids = set(load_instance_file(instance_file) or [])
    if selected_ids:
        records = [record for record in records if record["instance_id"] in selected_ids]
    found = []
    for rank, record in enumerate(records):
        repository = re.sub(r"[^A-Za-z0-9_.-]+", "_", record["repo"].replace("/", "__"))
        prefix = f"{repository}___base-{record['base_commit'][:12]}___src-"
        matches = sorted(cache_root.glob(f"{prefix}*/metadata.json"))
        if len(matches) != 1:
            logger.warning(f"Skipping source with {len(matches)} matching directories for {record['instance_id']}: {prefix}")
            continue
        metadata = json_load(matches[0])
        repo = matches[0].parent / metadata.get("source_path", "repo")
        if metadata.get("repository") != record["repo"] or metadata.get("expected_commit") != record["base_commit"] or not repo.is_dir():
            logger.warning(f"Skipping mismatched repository source metadata: {matches[0]}")
            continue
        found.append({"record": record, "metadata": metadata, "repo": repo, "rank": rank})
    if not per_dataset_limit:
        return found
    selected = []
    counts: dict[str, int] = defaultdict(int)
    for item in found:
        dataset = item["record"].get("source_dataset", "unknown")
        if counts[dataset] >= per_dataset_limit:
            continue
        counts[dataset] += 1
        selected.append(item)
    return selected


def export_codeql_csv(item: dict, output_root: Path, timeout: int, workers: int, ram_mb: int, force_codeql: bool) -> dict:
    instance_id = item["record"]["instance_id"]
    case_dir = output_root / instance_id
    csv_path = case_dir / "codeql_graph.csv.gz"
    case_dir.mkdir(parents=True, exist_ok=True)
    source_view, source_scope = sink_finder.prepare_source_view(item["repo"], case_dir)
    binary = sink_finder.tool_executables()["codeql"]
    if not binary:
        raise RuntimeError("CodeQL executable is unavailable")
    database, database_stage = sink_finder.create_codeql_database(
        source_view,
        case_dir / "cache" / "codeql_database",
        binary["main"],
        timeout,
        workers,
        source_scope["source_fingerprint"],
        force_codeql,
    )
    if database_stage["status"] != "completed":
        raise RuntimeError(f"CodeQL database creation failed for {instance_id}")
    result = sink_finder.run_codeql_graph_query(
        source_view,
        case_dir,
        binary["main"],
        database,
        timeout,
        workers,
        ram_mb,
        source_scope["source_fingerprint"],
        force_codeql,
    )
    temporary = case_dir / ".tmp"
    if temporary.exists():
        import shutil
        shutil.rmtree(temporary)
    if result["status"] != "completed" or not csv_path.is_file():
        raise RuntimeError(f"CodeQL CSV export failed for {instance_id}")
    return {
        "instance_id": instance_id,
        "status": "completed",
        "reused": bool(database_stage.get("reused") and result.get("reused")),
    }


def export_and_process_instance(
    item: dict,
    output_root: Path,
    timeout: int,
    workers: int,
    ram_mb: int,
    force: bool,
    force_codeql: bool,
) -> dict:
    instance_id = item["record"]["instance_id"]
    case_dir = output_root / instance_id
    case_dir.mkdir(parents=True, exist_ok=True)
    lock_path = case_dir / ".keypoints.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another key-point worker is writing this case: {case_dir}") from error
        export_result = export_codeql_csv(item, output_root, timeout, workers, ram_mb, force_codeql)
        return process_instance(item, output_root, force or not export_result["reused"])


def key_point_builder_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def process_instance(item: dict, output_root: Path, force: bool) -> dict:
    record = item["record"]
    instance_id = record["instance_id"]
    case_dir = output_root / instance_id
    key_points_path = case_dir / "combined_key_points.json"
    builder_fingerprint = key_point_builder_fingerprint()
    if key_points_path.is_file() and not force:
        try:
            existing = json_load(key_points_path)
            graph_status = json_load(case_dir / "codeql_graph_status.json")
            if (
                existing.get("status") == "completed"
                and existing.get("parser") == "codeql"
                and existing.get("builder_fingerprint") == builder_fingerprint
                and existing.get("source_fingerprint") == graph_status.get("source_fingerprint")
                and existing.get("query_fingerprint") == graph_status.get("query_fingerprint")
            ):
                logger.info(f"Skipped completed Guard path analysis: {instance_id}")
                return {"instance_id": instance_id, "status": existing["status"], "reused": True}
        except (OSError, json.JSONDecodeError, RuntimeError) as error:
            logger.warning(f"Invalid resumable output for {instance_id}: {error}")
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        graph_path = output_root / instance_id / "codeql_graph.csv.gz"
        if not graph_path.is_file():
            raise FileNotFoundError(f"CodeQL graph CSV is missing: {graph_path}; rerun run_find_points_keypoints.py")
        analysis = build_path_key_points(item["repo"], sink_finder.parse_codeql_graph(item["repo"], graph_path))
        status = "completed_with_gaps" if analysis["repository_parse_gaps"] or analysis["analysis_gaps"] else "completed"
    except Exception as error:
        logger.error(f"Guard path analysis failed for {instance_id}: {type(error).__name__}: {error}")
        analysis = {"parser": "codeql", "coordinate_validator": "tree_sitter", "scope": "full_repository", "files": {}, "repository_parse_gaps": [], "analysis_gaps": [], "error": f"{type(error).__name__}: {error}"}
        status = "error"
    compact_files = {}
    for path, data in analysis["files"].items():
        compact_files[path] = {"key_points_by_line": {
            line: [
                {key: value for key, value in point.items() if key != "line"}
                for point in points
            ]
            for line, points in data["key_points_by_line"].items()
        }}
    graph_status = json_load(case_dir / "codeql_graph_status.json")
    key_points = {
        "builder_fingerprint": builder_fingerprint,
        "source_fingerprint": graph_status.get("source_fingerprint"),
        "query_fingerprint": graph_status.get("query_fingerprint"),
        "status": status, "parser": "codeql", "instance_id": instance_id,
        "repository": record["repo"], "base_commit": record["base_commit"], "files": compact_files,
        "repository_parse_gaps": analysis["repository_parse_gaps"], "analysis_gaps": analysis["analysis_gaps"],
    }
    if analysis.get("error"):
        key_points["error"] = analysis["error"]
    json_dump(key_points_path, key_points)
    stats = {
        "instance_id": instance_id, "status": status, "duration_s": round(time.time() - started, 3),
        "files": len(analysis["files"]),
        "key_points": sum(len(points) for data in analysis["files"].values() for points in data["key_points_by_line"].values()),
        "cfg_edges": sum(len(data["cfg_edges"]) for data in analysis["files"].values()),
        "dfg_edges": sum(len(data["dfg_edges"]) for data in analysis["files"].values()),
        "parse_gaps": len(analysis["repository_parse_gaps"]),
    }
    logger.info(f"Guard path analysis {instance_id}: {stats}")
    return {"reused": False, **stats}


def count_failed_results(results: list[dict]) -> int:
    return sum(item.get("status") != "completed" for item in results)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export CodeQL CFG/DFG CSV, then extract key points with Tree-sitter coordinate validation")
    parser.add_argument("--cache-dir", type=Path, default=CONFIG["cache_dir"])
    parser.add_argument("--output-dir", type=Path, default=CONFIG["output_dir"])
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--codeql-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--codeql-ram-mb", type=int, default=8192)
    parser.add_argument("--source", choices=["SWE-Bench-Pro", "SWE-Bench-Verified", "SWE-Bench-Full-Test", "R2E-Gym-Lite"])
    parser.add_argument("--instance-file")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-dataset-limit", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--force", action="store_true", help="Re-extract key points while reusing valid CodeQL artifacts")
    parser.add_argument("--force-codeql", action="store_true", help="Rebuild the CodeQL database and graph CSV")
    args = parser.parse_args()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    items = cached_records(args.cache_dir.resolve(), args.source, args.per_dataset_limit, args.instance_file)
    if args.limit:
        items = items[:args.limit]
    logger.info(f"Selected {len(items)} cached repositories; first 2: {[item['record']['instance_id'] for item in items[:2]]}")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                export_and_process_instance,
                item,
                output_root,
                args.timeout,
                args.codeql_workers,
                args.codeql_ram_mb,
                args.force,
                args.force_codeql,
            ): item
            for item in items
        }
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                instance_id = item["record"]["instance_id"]
                logger.error(f"Guard path worker failed for {instance_id}: {type(error).__name__}: {error}")
                results.append({"instance_id": instance_id, "status": "error", "error": f"{type(error).__name__}: {error}"})
    failed = count_failed_results(results)
    logger.info(f"Guard path summary: instances={len(results)}, failed={failed}, first 2={results[:2]}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
