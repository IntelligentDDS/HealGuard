from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import fcntl
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import stat as stat_module
import subprocess
import sys
import threading
import time
import tokenize
import warnings
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from loguru import logger
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agenticheal import load_dataset_records
from util import HEALBENCH_INSTANCE_IDS
from heal_repository_scope import (
    REPOSITORY_EXCLUDED_DIRECTORIES,
    REPOSITORY_EXCLUDED_PATTERNS,
    is_repository_path_excluded,
)
from run_build_sink_rules import build_sink_rules
from util import init_logger, load_instance_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "find_points"
DEFAULT_REPO_CACHE = PROJECT_ROOT / "datasets" / "raw_repo_source_code"
PYSA_MODELS = PROJECT_ROOT / "refs" / "pyre-check" / "stubs" / "taint"
RULES_DIR = PROJECT_ROOT / "scripts" / "sink_rules"
GENERATED_RULES = Path(os.environ.get("AGENTICHEAL_GENERATED_SINK_RULES", RULES_DIR / "generated"))
OFFICIAL_RULES_PATH = RULES_DIR / "official_rules.yaml"
FINAL_RULES_PATH = Path(os.environ.get("AGENTICHEAL_SINK_RULES", RULES_DIR / "final_rules.yaml"))
CATALOGS_DIR = GENERATED_RULES / "catalogs"
NATIVE_RULES_DIR = GENERATED_RULES / "native"
TOOL_RULE_PATHS = {tool: CATALOGS_DIR / f"{tool}_rules.json" for tool in ("bandit", "semgrep", "pysa", "codeql")}
SEMGREP_RULES = NATIVE_RULES_DIR / "semgrep.yaml"
PYSA_RUNTIME_MODELS = NATIVE_RULES_DIR / "pysa"
CODEQL_MODEL_PACK = NATIVE_RULES_DIR / "codeql"
CODEQL_SINK_QUERY = PROJECT_ROOT / "scripts" / "codeql" / "SensitiveSinks.ql"
CODEQL_IMPACT_QUERY = PROJECT_ROOT / "scripts" / "codeql" / "SensitiveModels.ql"
CODEQL_GRAPH_QUERY = PROJECT_ROOT / "scripts" / "codeql" / "HealingCapabilities.ql"
CODEQL_SEARCH_PATH = PROJECT_ROOT / "refs" / "codeql"
SUPPORTED_TOOLS = {"bandit", "semgrep", "codeql"}
BANDIT_SINK_IDS = {
    "B102", "B202", "B301", "B302", "B303", "B304", "B305", "B306", "B307", "B310",
    "B311", "B312", "B313", "B314", "B315", "B316", "B317", "B318", "B319", "B320",
    "B321", "B401", "B402", "B506", "B601", "B602", "B603", "B604", "B605", "B606",
    "B607", "B608", "B609", "B610", "B611", "B612", "B615", "B702", "B703", "B704",
}
SEMGREP_NON_SINK_MARKERS = {
    "compatibility", "app-run-security-config", "missing-integrity", "mutable-action-tag",
    "dependabot", "plaintext-http-link", "no-set-ciphers", "weak-ssl-version",
}
UNKNOWN_CALL_MARKERS = {
    "eval", "exec", "__import__", "import_module", "ctypes", "cffi", "dlopen", "load_library",
    "setattr", "getattr", "globals", "locals", "monkeypatch", "patch", "reload",
}



def _node_id(file: str, node: ast.stmt) -> str:
    location = (
        file, node.lineno, node.col_offset + 1,
        node.end_lineno or node.lineno, (node.end_col_offset or node.col_offset) + 1,
        type(node).__name__,
    )
    payload = json.dumps(location, separators=(",", ":"))
    return "node-" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def _edge_id(source_id: str, target_id: str, edge_type: str, detail: str = "") -> str:
    payload = json.dumps((source_id, target_id, edge_type, detail), separators=(",", ":"))
    return "edge-" + hashlib.sha256(payload.encode()).hexdigest()[:20]


class _CfgBuilder:
    def __init__(self, file: str, tree: ast.Module):
        self.file = file
        self.tree = tree
        self.nodes: dict[str, dict] = {}
        self.edges: dict[str, dict] = {}
        self.scope_ids: dict[int, str] = {}
        self._index_scope(tree.body, "module")

    def _index_scope(self, statements: list[ast.stmt], scope_id: str) -> None:
        for node in statements:
            self.scope_ids[id(node)] = scope_id
            node_id = _node_id(self.file, node)
            decorators = getattr(node, "decorator_list", [])
            self.nodes[node_id] = {
                "node_id": node_id,
                "file": self.file,
                "line": node.lineno,
                "column": node.col_offset + 1,
                "end_line": node.end_lineno or node.lineno,
                "end_column": (node.end_col_offset or node.col_offset) + 1,
                "decorator_lines": sorted({line for decorator in decorators for line in range(decorator.lineno, (decorator.end_lineno or decorator.lineno) + 1)}),
                "ast_type": type(node).__name__,
                "scope_id": scope_id,
                "sources": ["ast"],
            }
            child_scope = node_id if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) else scope_id
            for field in ("body", "orelse", "finalbody"):
                children = getattr(node, field, None)
                if children:
                    self._index_scope(children, child_scope)
            for handler in getattr(node, "handlers", []):
                self._index_scope(handler.body, child_scope)
            for case in getattr(node, "cases", []):
                self._index_scope(case.body, child_scope)

    def add_edge(self, source: ast.stmt, target: ast.stmt, edge_type: str, detail: str = "") -> None:
        source_id = _node_id(self.file, source)
        target_id = _node_id(self.file, target)
        edge_id = _edge_id(source_id, target_id, edge_type, detail)
        self.edges[edge_id] = {
            "edge_id": edge_id,
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
            "detail": detail,
            "scope_id": self.scope_ids.get(id(source), "module"),
            "source": "ast",
        }

    def build_block(
        self,
        statements: list[ast.stmt],
        follow: list[ast.stmt],
        break_targets: list[ast.stmt] | None = None,
        continue_targets: list[ast.stmt] | None = None,
    ) -> list[ast.stmt]:
        next_entries = follow
        for statement in reversed(statements):
            next_entries = self.build_statement(
                statement, next_entries, break_targets or [], continue_targets or [],
            )
        return next_entries

    def connect(self, source: ast.stmt, targets: Iterable[ast.stmt], edge_type: str, detail: str = "") -> None:
        for target in targets:
            self.add_edge(source, target, edge_type, detail)

    def build_statement(
        self,
        statement: ast.stmt,
        follow: list[ast.stmt],
        break_targets: list[ast.stmt],
        continue_targets: list[ast.stmt],
    ) -> list[ast.stmt]:
        if isinstance(statement, (ast.Return, ast.Raise)):
            return [statement]
        if isinstance(statement, ast.Break):
            self.connect(statement, break_targets, "cfg_break")
            return [statement]
        if isinstance(statement, ast.Continue):
            self.connect(statement, continue_targets, "cfg_continue")
            return [statement]
        if isinstance(statement, ast.If):
            body = self.build_block(statement.body, follow, break_targets, continue_targets)
            orelse = self.build_block(statement.orelse, follow, break_targets, continue_targets) if statement.orelse else follow
            condition = ast.unparse(statement.test)
            self.connect(statement, body, "cfg_true", condition)
            self.connect(statement, orelse, "cfg_false", condition)
            return [statement]
        if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            body = self.build_block(statement.body, [statement], follow, [statement])
            orelse = self.build_block(statement.orelse, follow, break_targets, continue_targets) if statement.orelse else follow
            condition = ast.unparse(statement.test) if isinstance(statement, ast.While) else ast.unparse(statement.iter)
            self.connect(statement, body, "cfg_loop_body", condition)
            self.connect(statement, orelse, "cfg_loop_exit", condition)
            return [statement]
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            body = self.build_block(statement.body, follow, break_targets, continue_targets)
            self.connect(statement, body or follow, "cfg_body")
            return [statement]
        if isinstance(statement, ast.Try):
            final_entries = self.build_block(statement.finalbody, follow, break_targets, continue_targets) if statement.finalbody else follow
            else_entries = self.build_block(statement.orelse, final_entries, break_targets, continue_targets) if statement.orelse else final_entries
            body = self.build_block(statement.body, else_entries, break_targets, continue_targets)
            self.connect(statement, body or else_entries, "cfg_try")
            for handler in statement.handlers:
                handler_entries = self.build_block(handler.body, final_entries, break_targets, continue_targets)
                self.connect(statement, handler_entries or final_entries, "cfg_exception")
            return [statement]
        if hasattr(ast, "TryStar") and isinstance(statement, ast.TryStar):
            final_entries = self.build_block(statement.finalbody, follow, break_targets, continue_targets) if statement.finalbody else follow
            else_entries = self.build_block(statement.orelse, final_entries, break_targets, continue_targets) if statement.orelse else final_entries
            body = self.build_block(statement.body, else_entries, break_targets, continue_targets)
            self.connect(statement, body or else_entries, "cfg_try_star")
            for handler in statement.handlers:
                handler_entries = self.build_block(handler.body, final_entries, break_targets, continue_targets)
                self.connect(statement, handler_entries or final_entries, "cfg_exception_group")
            return [statement]
        if isinstance(statement, ast.Match):
            for case_index, case in enumerate(statement.cases):
                entries = self.build_block(case.body, follow, break_targets, continue_targets)
                self.connect(statement, entries or follow, "cfg_match_case", str(case_index))
            return [statement]
        self.connect(statement, follow, "cfg_next")
        return [statement]

    def build(self) -> tuple[list[dict], list[dict]]:
        self.build_block(self.tree.body, [])
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.build_block(node.body, [])
        return list(self.nodes.values()), list(self.edges.values())


def build_ast_graph(source_root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}
    failures = []
    for path in sorted(source_root.rglob("*.py")):
        if is_repository_path_excluded(path.relative_to(source_root).parts):
            continue
        file = path.relative_to(source_root).as_posix()
        try:
            with tokenize.open(path) as source:
                tree = ast.parse(source.read(), filename=file)
            file_nodes, file_edges = _CfgBuilder(file, tree).build()
            nodes.update((item["node_id"], item) for item in file_nodes)
            edges.update((item["edge_id"], item) for item in file_edges)
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            logger.warning(f"AST graph parse failed for {file}: {type(error).__name__}: {error}")
            failures.append({
                "file": file,
                "error_type": type(error).__name__,
                "error": str(error),
            })
    return (
        sorted(nodes.values(), key=lambda item: item["node_id"]),
        sorted(edges.values(), key=lambda item: item["edge_id"]),
        failures,
    )


def containing_node(nodes_by_file: dict[str, dict[int, list[dict]]], location: dict) -> dict | None:
    file = str(location.get("file", ""))
    line = int(location.get("line", 0) or 0)
    column = int(location.get("column", 1) or 1)
    end_line = int(location.get("end_line", line) or line)
    end_column = int(location.get("end_column", column) or column)
    line_nodes = nodes_by_file.get(file, {}).get(line, [])
    candidates = [
        node for node in line_nodes
        if (node["line"], node["column"]) <= (line, column)
        and (end_line, end_column) <= (node["end_line"], node["end_column"])
    ]
    if not candidates:
        candidates = [
            node for node in line_nodes
            if node["line"] <= line <= node["end_line"]
            or line in node.get("decorator_lines", [])
        ]
    return min(
        candidates,
        key=lambda node: (
            node["end_line"] - node["line"],
            node["end_column"] - node["column"],
        ),
        default=None,
    )


def index_nodes(nodes: list[dict]) -> dict[str, dict[int, list[dict]]]:
    result: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for node in nodes:
        indexed_lines = set(range(node["line"], node["end_line"] + 1))
        indexed_lines.update(node.get("decorator_lines", []))
        for line in indexed_lines:
            result[node["file"]][line].append(node)
    return result


def merge_semantic_records(
    nodes: list[dict],
    edges: list[dict],
    records: Iterable[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    nodes_by_file = index_nodes(nodes)
    merged_edges = {edge["edge_id"]: edge for edge in edges}
    sinks = []
    failures = []
    for index, record in enumerate(records):
        source = containing_node(nodes_by_file, record)
        if record.get("record_type") == "sink":
            if source is None:
                failures.append({"reason": "sink_node_missing", "result_index": index, "record": record})
                continue
            sink_id = "sink-" + hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:20]
            sinks.append({
                "sink_id": sink_id,
                "node_id": source["node_id"],
                "file": record["file"],
                "line": record["line"],
                "column": record["column"],
                "end_line": record["end_line"],
                "end_column": record["end_column"],
                "kind": record.get("kind", ""),
                "detail": record.get("detail", ""),
                "source": "codeql",
            })
            continue
        target_location = {
            "file": record.get("sink_file", ""),
            "line": record.get("sink_line", 0),
            "column": record.get("sink_column", 1),
            "end_line": record.get("sink_end_line", 0),
            "end_column": record.get("sink_end_column", 1),
        }
        target = containing_node(nodes_by_file, target_location)
        if not target_location["line"]:
            failures.append({
                "reason": "semantic_edge_target_location_unavailable", "result_index": index,
                "source_found": source is not None, "target_found": False, "record": record,
            })
            continue
        if source is None or target is None:
            failures.append({
                "reason": "semantic_edge_node_missing", "result_index": index,
                "source_found": source is not None, "target_found": target is not None,
                "record": record,
            })
            continue
        edge_type = "semantic_" + str(record.get("record_type", "unknown"))
        detail = str(record.get("detail", ""))
        edge_id = _edge_id(source["node_id"], target["node_id"], edge_type, detail)
        merged_edges[edge_id] = {
            "edge_id": edge_id,
            "source_id": source["node_id"],
            "target_id": target["node_id"],
            "edge_type": edge_type,
            "detail": detail,
            "kind": record.get("kind", ""),
            "source": "codeql",
        }
    unique_sinks = {sink["sink_id"]: sink for sink in sinks}
    return nodes, sorted(merged_edges.values(), key=lambda item: item["edge_id"]), sorted(unique_sinks.values(), key=lambda item: item["sink_id"]), failures


def reverse_sensitive_subgraph(
    nodes: list[dict], edges: list[dict], sinks: list[dict],
) -> tuple[list[dict], list[dict]]:
    incoming: dict[str, list[dict]] = defaultdict(list)
    for edge in edges:
        incoming[edge["target_id"]].append(edge)
    retained = {sink["node_id"] for sink in sinks}
    scope_by_node = {node["node_id"]: node.get("scope_id") for node in nodes}
    retained.update(
        scope_id for node_id in list(retained)
        if (scope_id := scope_by_node.get(node_id)) and scope_id != "module"
    )
    queue = deque(retained)
    while queue:
        node_id = queue.popleft()
        for edge in incoming.get(node_id, []):
            if edge["source_id"] not in retained:
                retained.add(edge["source_id"])
                queue.append(edge["source_id"])

    retained_edges = [
        edge for edge in edges
        if edge["source_id"] in retained and edge["target_id"] in retained
    ]
    node_ids = set(scope_by_node)
    scope_queue = deque(
        scope_id for scope_id in {
            *(edge.get("scope_id") for edge in retained_edges),
            *(scope_by_node.get(node_id) for node_id in retained),
        }
        if scope_id in node_ids
    )
    while scope_queue:
        scope_id = scope_queue.popleft()
        if scope_id not in retained:
            retained.add(scope_id)
        parent_scope = scope_by_node.get(scope_id)
        if parent_scope in node_ids and parent_scope not in retained:
            scope_queue.append(parent_scope)
    return (
        [node for node in nodes if node["node_id"] in retained],
        retained_edges,
    )


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSONL {path}:{line_number}: {error}") from error
    return records

def ensure_rule_artifacts() -> None:
    required = [
        OFFICIAL_RULES_PATH,
        GENERATED_RULES / "internal" / "official_catalog.json",
        GENERATED_RULES / "internal" / "normalized_catalog.json",
        *TOOL_RULE_PATHS.values(),
    ]
    stale = any(path.exists() and path.stat().st_mtime < FINAL_RULES_PATH.stat().st_mtime for path in required)
    if not all(path.is_file() for path in required) or stale:
        build_sink_rules(FINAL_RULES_PATH, RULES_DIR / "sources", GENERATED_RULES)


ensure_rule_artifacts()
FINAL_RULES = yaml.safe_load(FINAL_RULES_PATH.read_text(encoding="utf-8"))["rules"]
RUNTIME_RULES = json.loads((NATIVE_RULES_DIR / "bandit_plugin/runtime_healing_bandit/rules.json").read_text(encoding="utf-8"))
OFFICIAL_RULES = json.loads((GENERATED_RULES / "internal" / "official_catalog.json").read_text(encoding="utf-8"))["records"]
TOOL_RULE_DATA = {tool: json.loads(path.read_text(encoding="utf-8")) for tool, path in TOOL_RULE_PATHS.items()}
TOOL_RULE_ALIASES = {tool: data["aliases"] for tool, data in TOOL_RULE_DATA.items()}
OFFICIAL_RULES_BY_TOOL: dict[str, dict[str, list[dict]]] = {}
for official_rule in OFFICIAL_RULES:
    tool_rules = OFFICIAL_RULES_BY_TOOL.setdefault(official_rule["tool"], {})
    tool_rules.setdefault(official_rule["native_rule_id"], []).append(official_rule)


def official_rules(tool: str, native_rule_id: str) -> list[dict]:
    rules = OFFICIAL_RULES_BY_TOOL.get(tool, {})
    if native_rule_id in rules:
        return rules[native_rule_id]
    matches = [record for rule_id, records in rules.items() if native_rule_id.endswith("." + rule_id) or native_rule_id.endswith("/" + rule_id) for record in records]
    if tool != "semgrep":
        return matches
    path_matches = []
    for record in matches:
        relative_path = record["source_path"].removeprefix("refs/semgrep-rules/python/")
        rule_namespace = str(Path(relative_path).with_suffix("")).replace("/", ".")
        if native_rule_id.endswith(f"{rule_namespace}.{record['native_rule_id']}"):
            path_matches.append(record)
    return path_matches or matches


def official_rule_references(tool: str, native_rule_id: str, message: str | None = None, metadata: dict | None = None) -> list[str]:
    records = official_rules(tool, native_rule_id)
    if tool == "semgrep" and len(records) > 1:
        exact = [record for record in records if record["raw_rule"].get("message") == message and record["raw_rule"].get("metadata", {}) == (metadata or {})]
        records = exact or records
    return [record["catalog_key"] for record in records]


def with_official_references(record: dict, references: list[str]) -> dict:
    return {**record, "official_rule_references": references}


def runtime_rule_matches(rule: dict, call: str) -> bool:
    normalized = call.removeprefix("builtins.")
    target = str(rule.get("callee", "")).removeprefix("builtins.")
    if target:
        return normalized == target
    method_name = normalized.rsplit(".", 1)[-1]
    method_pattern = rule.get("method_regex")
    if rule.get("method") == method_name or (method_pattern and re.fullmatch(method_pattern, method_name)):
        receiver = rule.get("receiver")
        return receiver not in {"socket", "sys.path"} or receiver in normalized
    pattern = rule.get("callee_regex") or rule.get("target_regex")
    return bool(pattern and re.fullmatch(pattern, normalized))


def rules_for_calls(calls: list[str]) -> list[dict]:
    return [rule for rule in RUNTIME_RULES if rule["sink_kind"] != "UNKNOWN_EFFECT" and any(runtime_rule_matches(rule, call) for call in calls)]


def candidate_records(base: dict, calls: list[str], explicit_kind: str | None = None, explicit_rule_id: str | None = None) -> list[dict]:
    aliases = TOOL_RULE_ALIASES[base["tool"]]
    if explicit_kind and explicit_rule_id:
        rule = next((item for item in RUNTIME_RULES if item["rule_id"] == explicit_rule_id), None)
        if not rule or explicit_kind != rule["sink_kind"] or explicit_rule_id not in aliases:
            return []
        if rule["template"] != "assignment" and not any(runtime_rule_matches(rule, call) for call in calls):
            return []
        return [{**base, "calls": calls, "sink_kind": explicit_kind, "runtime_rule_id": explicit_rule_id, "runtime_healing_rule_references": [aliases[explicit_rule_id]]}]
    return [{**base, "calls": calls, "sink_kind": rule["sink_kind"], "runtime_rule_id": rule["rule_id"], "runtime_healing_rule_references": [aliases[rule["rule_id"]]]} for rule in rules_for_calls(calls) if rule["rule_id"] in aliases]

def json_dump(path: Path, value: Any, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        dump_options = {"separators": (",", ":")} if compact else {"indent": 2}
        temporary.write_text(json.dumps(value, ensure_ascii=False, **dump_options) + "\n", encoding="utf-8")
        temporary.replace(path)
    except Exception as error:
        logger.error(f"Failed to write JSON artifact {path}: {type(error).__name__}: {error}")
        temporary.unlink(missing_ok=True)
        raise


def json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Failed to read required artifact {path}: {error}") from error


def directory_stats(path: Path) -> tuple[int, int, list[dict]]:
    file_count = 0
    total_bytes = 0
    errors = []

    def record_error(error: OSError) -> None:
        logger.warning(f"Excluded directory inspection failed for {error.filename}: {error}")
        errors.append({"path": str(error.filename), "error_type": type(error).__name__, "error": str(error)})

    for current, _, files in os.walk(path, followlinks=False, onerror=record_error):
        for name in files:
            item = Path(current) / name
            try:
                stat = item.lstat()
            except OSError as error:
                record_error(error)
                continue
            file_count += 1
            total_bytes += stat.st_size
    return file_count, total_bytes, errors


def prepare_source_view(repo: Path, output_dir: Path) -> tuple[Path, dict]:
    repo = repo.resolve()
    output_dir = output_dir.resolve()
    if output_dir == repo or repo in output_dir.parents:
        raise RuntimeError(f"Output directory must be outside the repository: repo={repo}, output={output_dir}")
    source_view = output_dir / ".tmp" / "source_view"
    shutil.rmtree(source_view, ignore_errors=True)
    included_file_count = 0
    included_bytes = 0
    inspection_gaps = []
    excluded = []
    fingerprint = hashlib.sha256()

    def record_error(error: OSError) -> None:
        logger.warning(f"Repository source inspection failed for {error.filename}: {error}")
        inspection_gaps.append({"path": str(error.filename), "error_type": type(error).__name__, "error": str(error)})

    for current, directories, files in os.walk(repo, topdown=True, followlinks=False, onerror=record_error):
        current_path = Path(current)
        kept_directories = []
        for name in sorted(directories):
            source = current_path / name
            relative = source.relative_to(repo)
            if is_repository_path_excluded(relative.parts, is_directory=True):
                file_count, total_bytes, errors = directory_stats(source)
                excluded.append({"path": relative.as_posix(), "reason": "excluded_directory", "name": name, "file_count": file_count, "bytes": total_bytes, "inspection_errors": errors})
            else:
                kept_directories.append(name)
        directories[:] = kept_directories
        for name in sorted(files):
            source = current_path / name
            relative = source.relative_to(repo)
            if is_repository_path_excluded(relative.parts):
                try:
                    stat = source.lstat()
                except OSError as error:
                    record_error(error)
                    continue
                excluded.append({"path": relative.as_posix(), "reason": "excluded_file", "name": name, "file_count": 1, "bytes": stat.st_size, "inspection_errors": []})
                continue
            try:
                stat = source.lstat()
            except OSError as error:
                record_error(error)
                continue
            try:
                source_hash = hashlib.sha256()
                with source.open("rb") as source_file:
                    while chunk := source_file.read(1024 * 1024):
                        source_hash.update(chunk)
            except OSError as error:
                record_error(error)
                continue
            included_file_count += 1
            included_bytes += stat.st_size
            fingerprint.update(relative.as_posix().encode())
            fingerprint.update(b"\0")
            fingerprint.update(str(stat_module.S_IMODE(stat.st_mode)).encode())
            fingerprint.update(b"\0")
            fingerprint.update(source_hash.digest())
            fingerprint.update(b"\n")

    def ignore_excluded(directory: str, names: list[str]) -> set[str]:
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(repo)
        return {
            name for name in names
            if is_repository_path_excluded(
                (*relative_directory.parts, name),
                is_directory=(directory_path / name).is_dir(),
            )
        }

    shutil.copytree(repo, source_view, symlinks=True, ignore=ignore_excluded)
    scope = {
        "source_fingerprint": fingerprint.hexdigest(),
        "source_repo": str(repo),
        "source_view": str(source_view),
        "source_storage": "filtered_copy_view",
        "source_copied_to_results": False,
        "excluded_directory_names": sorted(REPOSITORY_EXCLUDED_DIRECTORIES),
        "excluded_patterns": sorted(REPOSITORY_EXCLUDED_PATTERNS),
        "included_file_count": included_file_count,
        "included_bytes": included_bytes,
        "excluded_directory_count": sum(item["reason"] == "excluded_directory" for item in excluded),
        "excluded_file_count": sum(item["file_count"] for item in excluded),
        "excluded_bytes": sum(item["bytes"] for item in excluded),
        "copy_gap_count": len(inspection_gaps),
        "excluded_paths": excluded,
        "copy_gaps": inspection_gaps,
    }
    return source_view, scope


def executable(candidates: list[str | Path | None]) -> str | None:
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def tool_executables() -> dict[str, dict[str, str] | None]:
    return {
        "bandit": {"main": path} if (path := executable([shutil.which("bandit")])) else None,
        "semgrep": {"main": path} if (path := executable([shutil.which("semgrep")])) else None,
        "codeql": {"main": path} if (path := executable([shutil.which("codeql"), PROJECT_ROOT / ".tools/codeql/codeql"])) else None,
    }


def run_stage(tool: str, stage: str, command: list[str], cwd: Path, tool_dir: Path, timeout: int) -> dict:
    started = time.time()
    stdout_path = tool_dir / f"{stage}.stdout"
    stderr_path = tool_dir / f"{stage}.stderr"
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr, text=True, timeout=timeout)
        status = "completed" if result.returncode == 0 else "error"
        return {
            "tool": tool,
            "stage": stage,
            "status": status,
            "command": command,
            "cwd": str(cwd),
            "returncode": result.returncode,
            "duration_s": round(time.time() - started, 3),
            "stdout_path": stdout_path.name,
            "stderr_path": stderr_path.name,
        }
    except subprocess.TimeoutExpired as error:
        logger.error(f"{tool} {stage} timed out: {error}")
        return {
            "tool": tool,
            "stage": stage,
            "status": "timeout",
            "command": command,
            "cwd": str(cwd),
            "returncode": None,
            "duration_s": round(time.time() - started, 3),
            "stdout_path": stdout_path.name,
            "stderr_path": stderr_path.name,
            "error": f"TimeoutExpired: {error}",
        }
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        logger.error(f"{tool} {stage} failed: {message}")
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(message + "\n", encoding="utf-8")
        return {
            "tool": tool,
            "stage": stage,
            "status": "error",
            "command": command,
            "cwd": str(cwd),
            "returncode": None,
            "duration_s": round(time.time() - started, 3),
            "stdout_path": stdout_path.name,
            "stderr_path": stderr_path.name,
            "error": message,
        }


def unavailable(tool: str, tool_dir: Path, reason: str) -> dict:
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "unavailable.stdout").write_text("", encoding="utf-8")
    (tool_dir / "unavailable.stderr").write_text(reason + "\n", encoding="utf-8")
    return {
        "tool": tool,
        "status": "unavailable",
        "reason": reason,
        "stages": [{
            "tool": tool,
            "stage": "unavailable",
            "status": "unavailable",
            "command": None,
            "cwd": None,
            "returncode": None,
            "duration_s": 0.0,
            "stdout_path": "unavailable.stdout",
            "stderr_path": "unavailable.stderr",
        }],
        "artifacts": [],
    }


def single_stage_result(tool: str, source: str, stage: dict, artifacts: list[str]) -> dict:
    return {"tool": tool, "status": stage["status"], "official_source": source, "stages": [stage], "artifacts": artifacts}


def complete_tool_evidence(tool_dir: Path, status: dict, sinks: list[dict], scan_input: dict, parse_gaps: list[dict]) -> dict:
    stages = []
    for stage in status.get("stages", []):
        stdout_path = Path(stage.get("stdout_path", ""))
        stderr_path = Path(stage.get("stderr_path", ""))
        if not stdout_path.is_absolute():
            stdout_path = tool_dir / stdout_path
        if not stderr_path.is_absolute():
            stderr_path = tool_dir / stderr_path
        embedded_stage = {key: value for key, value in stage.items() if key not in {"stdout_path", "stderr_path"}}
        stages.append({
            **embedded_stage,
            "stdout": stage.get("stdout", stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""),
            "stderr": stage.get("stderr", stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""),
        })
    artifacts = []
    for relative in status.get("artifacts", []):
        path = tool_dir / relative
        if not path.is_file():
            artifacts.append({"path": relative, "status": "missing"})
            continue
        try:
            if path.suffix == ".json":
                content = json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix in {".csv", ".stdout", ".stderr", ".yaml", ".yml"}:
                content = path.read_text(encoding="utf-8", errors="replace")
            else:
                content = {"binary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(f"Failed to embed tool artifact {path}: {error}")
            content = {"read_error": f"{type(error).__name__}: {error}"}
        artifacts.append({"path": relative, "content": content})
    return {"phase": "find", "tool": status["tool"], "status": status["status"], "input": scan_input, "stages": stages, "artifacts": artifacts, "sinks": sinks, "parse_gaps": parse_gaps}


BANDIT_PLUGIN_LOCK = threading.Lock()
BANDIT_PLUGIN_READY = False


def ensure_bandit_plugin(binary: str, timeout: int) -> dict | None:
    global BANDIT_PLUGIN_READY
    with BANDIT_PLUGIN_LOCK:
        if BANDIT_PLUGIN_READY:
            return None
        probe = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=timeout)
        if probe.returncode == 0 and "RH001" in probe.stdout and "RH002" in probe.stdout:
            BANDIT_PLUGIN_READY = True
            return None
        first_line = Path(binary).read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        interpreter = first_line.removeprefix("#!").strip() if first_line.startswith("#!") else sys.executable
        command = [interpreter, "-m", "pip", "install", "--user", "-e", str(NATIVE_RULES_DIR / "bandit_plugin"), "--no-build-isolation"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Bandit plugin installation failed: {result.stderr}")
        BANDIT_PLUGIN_READY = True
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def run_bandit(repo: Path, tool_dir: Path, binary: str, timeout: int, phase: str) -> dict:
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "results.json").unlink(missing_ok=True)
    installation = ensure_bandit_plugin(binary, timeout)
    if installation:
        json_dump(tool_dir / "plugin_install.json", installation)
    command = [binary, "-r", str(repo), "-f", "json", "-o", str(tool_dir / "results.json"), "--exit-zero", "-t", "RH001,RH002"]
    artifacts = ["results.json", "plugin_install.json"] if installation else ["results.json"]
    return single_stage_result("bandit", "refs/bandit", run_stage("bandit", "scan", command, repo, tool_dir, timeout), artifacts)


def run_semgrep(repo: Path, tool_dir: Path, binary: str, timeout: int, phase: str) -> dict:
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "results.json").unlink(missing_ok=True)
    command = [binary, "scan", "--config", str(SEMGREP_RULES), "--json", "--json-output", str(tool_dir / "results.json"), str(repo)]
    return single_stage_result("semgrep", "refs/semgrep", run_stage("semgrep", "scan", command, repo, tool_dir, timeout), ["results.json"])


def write_pysa_configuration(repo: Path, tool_dir: Path, phase: str) -> Path:
    path = tool_dir / ".pyre_configuration"
    json_dump(path, {
        "source_directories": [str(repo)],
        "taint_models_path": [str(PYSA_MODELS)] if phase == "official" else [str(PYSA_RUNTIME_MODELS)],
        "search_path": [str(PROJECT_ROOT / "refs/pyre-check/stubs")],
        "site_package_search_strategy": "pep561",
    })
    return path


def run_pysa(repo: Path, tool_dir: Path, binary: str, timeout: int, workers: int, phase: str, work_dir: Path | None = None) -> dict:
    tool_dir.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir or tool_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_pysa_configuration(repo, work_dir, phase)
    model_paths = [PYSA_MODELS] if phase == "official" else [PYSA_RUNTIME_MODELS]
    if phase == "analysis":
        model_paths = [PYSA_MODELS, PYSA_RUNTIME_MODELS]
    command = [binary, "--number-of-workers", str(workers), "analyze"]
    for model_path in model_paths:
        command.extend(["--taint-models-path", str(model_path)])
    command.extend(["--save-results-to", str(tool_dir / "results"), "--output-format", "json", "--no-verify", "--use-cache"])
    stage = run_stage("pysa", "analyze", command, work_dir, tool_dir, timeout)
    artifacts = [path.relative_to(tool_dir).as_posix() for path in tool_dir.rglob("*") if path.is_file() and not path.name.endswith((".stdout", ".stderr"))]
    return {"tool": "pysa", "status": stage["status"], "official_source": "refs/pyre-check", "configuration": str(config_path), "work_dir": str(work_dir), "stages": [stage], "artifacts": sorted(artifacts)}


def run_joern(repo: Path, tool_dir: Path, binaries: dict[str, str], timeout: int) -> dict:
    tool_dir.mkdir(parents=True, exist_ok=True)
    command = [binaries["cpg"], str(repo), "--output", str(tool_dir / "cpg.bin")]
    stages = [run_stage("joern", "pysrc2cpg", command, repo, tool_dir, timeout)]
    if stages[-1]["status"] == "completed":
        command = [binaries["main"], str(tool_dir / "cpg.bin"), "--script", str(PROJECT_ROOT / "scripts/joern_prepare.sc")]
        stages.append(run_stage("joern", "prepare_dataflow", command, tool_dir, tool_dir, timeout))
        workspace_cpgs = sorted((tool_dir / "workspace").glob("*/cpg.bin"))
        if stages[-1]["status"] == "completed" and workspace_cpgs:
            shutil.copy2(workspace_cpgs[-1], tool_dir / "cpg.bin")
    status = "completed" if all(stage["status"] == "completed" for stage in stages) else "error"
    return {"tool": "joern", "status": status, "official_source": "refs/joern", "stages": stages, "artifacts": ["cpg.bin"]}


def create_codeql_database(
    repo: Path,
    database: Path,
    binary: str,
    timeout: int,
    workers: int,
    source_fingerprint: str | None = None,
    force: bool = False,
) -> tuple[Path, dict]:
    cache_dir = database.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    status_path = cache_dir / "codeql_database_status.json"
    if not force and database.is_dir() and status_path.is_file():
        existing = json_load(status_path)
        source_root = next((item.split("=", 1)[1] for item in existing.get("command", []) if isinstance(item, str) and item.startswith("--source-root=")), None)
        if (
            existing.get("status") == "completed"
            and source_root == str(repo)
            and existing.get("source_fingerprint") == source_fingerprint
        ):
            return database, {**existing, "reused": True}
    if database.exists():
        shutil.rmtree(database)
    raw_stage = run_stage("codeql", "database_create", [binary, "database", "create", str(database), "--language=python", f"--source-root={repo}", f"--threads={workers}"], repo, cache_dir, timeout)
    stdout_path = cache_dir / raw_stage["stdout_path"]
    stderr_path = cache_dir / raw_stage["stderr_path"]
    stage = {
        **raw_stage,
        "source_fingerprint": source_fingerprint,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout": stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else "",
        "stderr": stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else "",
    }
    json_dump(status_path, stage)
    stdout_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)
    return database, stage


def codeql_graph_query_fingerprint(binary: str) -> str:
    fingerprint = hashlib.sha256()
    try:
        fingerprint.update(subprocess.check_output([binary, "version", "--format=json"], text=False))
    except (OSError, subprocess.CalledProcessError) as error:
        logger.error(f"Failed to fingerprint CodeQL version from {binary}: {error}")
        raise RuntimeError(f"Failed to fingerprint CodeQL version from {binary}: {error}") from error
    fingerprint.update(b"\n")
    for query_path in sorted(CODEQL_GRAPH_QUERY.parent.glob("*")):
        if query_path.is_file():
            fingerprint.update(query_path.name.encode())
            fingerprint.update(b"\0")
            fingerprint.update(query_path.read_bytes())
            fingerprint.update(b"\n")
    return fingerprint.hexdigest()


def run_codeql_graph_query(
    repo: Path,
    output_dir: Path,
    binary: str,
    database: Path,
    timeout: int,
    workers: int,
    ram_mb: int,
    source_fingerprint: str | None = None,
    force: bool = False,
) -> dict:
    graph_dir = output_dir / ".tmp" / "codeql_graph"
    bqrs_path = graph_dir / "repository-graph.bqrs"
    csv_path = graph_dir / "codeql_graph.csv"
    gzip_path = output_dir / "codeql_graph.csv.gz"
    status_path = output_dir / "codeql_graph_status.json"
    query_fingerprint = codeql_graph_query_fingerprint(binary)
    if not force and gzip_path.is_file() and status_path.is_file():
        existing = json_load(status_path)
        if (
            existing.get("status") == "completed"
            and existing.get("source_fingerprint") == source_fingerprint
            and existing.get("query_fingerprint") == query_fingerprint
        ):
            return {**existing, "reused": True}
    if graph_dir.exists():
        shutil.rmtree(graph_dir)
    graph_dir.mkdir(parents=True)
    stages = []
    query_command = [
        binary, "query", "run", f"--database={database}", f"--output={bqrs_path}",
        f"--search-path={CODEQL_SEARCH_PATH}", f"--threads={workers}", f"--ram={ram_mb}",
        str(CODEQL_GRAPH_QUERY),
    ]
    stages.append(run_stage("codeql", "repository_graph", query_command, repo, graph_dir, timeout))
    if stages[-1]["status"] == "completed":
        decode_command = [binary, "bqrs", "decode", "--format=csv", f"--output={csv_path}", str(bqrs_path)]
        stages.append(run_stage("codeql", "repository_graph_decode", decode_command, repo, graph_dir, timeout))
    status = "completed" if stages and all(stage["status"] == "completed" for stage in stages) else "error"
    if status == "completed":
        temporary = gzip_path.with_name(f".{gzip_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with csv_path.open("rb") as source, gzip.open(temporary, "wb", compresslevel=6) as target:
                shutil.copyfileobj(source, target)
            os.replace(temporary, gzip_path)
            (output_dir / "codeql_graph.csv").unlink(missing_ok=True)
        except Exception as error:
            logger.error(f"Failed to compress CodeQL graph CSV {csv_path}: {type(error).__name__}: {error}")
            temporary.unlink(missing_ok=True)
            status = "error"
    result = {
        "status": status,
        "stages": stages,
        "graph_path": str(gzip_path),
        "query": str(CODEQL_GRAPH_QUERY),
        "source_fingerprint": source_fingerprint,
        "query_fingerprint": query_fingerprint,
        "reused": False,
    }
    json_dump(status_path, result)
    return result


def run_codeql_query(repo: Path, tool_dir: Path, binary: str, database: Path, database_stage: dict, timeout: int, workers: int, ram_mb: int, phase: str, query: Path = CODEQL_SINK_QUERY, sinks: list[dict] | None = None, calls: dict[tuple[str, int], list[dict]] | None = None) -> dict:
    if tool_dir.exists():
        shutil.rmtree(tool_dir)
    tool_dir.mkdir(parents=True)
    stages = []
    if database_stage["status"] == "completed":
        command = [
            binary,
            "query",
            "run",
            f"--database={database}",
            f"--output={tool_dir / 'sensitive-models.bqrs'}",
            f"--search-path={CODEQL_SEARCH_PATH}",
            f"--threads={workers}",
            f"--ram={ram_mb}",
        ]
        if sinks is not None:
            candidate_path = tool_dir / "candidate-sinks.csv"
            rows = set()
            source_calls = calls if calls is not None else calls_by_line(repo)[0]
            missing_sites = []
            for record in sinks:
                names = set(record.get("calls", []))
                sites = [
                    site for site in source_calls.get((record["file"], record["line"]), [])
                    if site["name"] in names
                ]
                if not sites:
                    missing_sites.append({"file": record["file"], "line": record["line"], "calls": sorted(names)})
                for site in sites:
                    rows.add((
                        str((repo / record["file"]).resolve()), site["line"], site["column"] + 1,
                        site["end_line"], site["end_column"] + 1,
                    ))
            if missing_sites:
                raise RuntimeError(f"Exact call sites missing for filtered sinks: {missing_sites}")
            with candidate_path.open("w", encoding="utf-8", newline="") as output:
                csv.writer(output).writerows(sorted(rows))
            command.append(f"--external=candidateSink={candidate_path}")
        if phase == "shared":
            command.extend([f"--additional-packs={NATIVE_RULES_DIR}", "--model-packs=agenticheal/runtime-healing-models@*"])
        command.append(str(query))
        stages.append(run_stage("codeql", "sensitive_models", command, repo, tool_dir, timeout))
    if stages and stages[-1]["status"] == "completed":
        decode_format = "csv" if query == CODEQL_IMPACT_QUERY else "json"
        decoded_name = f"sensitive-models.{decode_format}"
        command = [binary, "bqrs", "decode", f"--format={decode_format}", f"--output={tool_dir / decoded_name}", str(tool_dir / "sensitive-models.bqrs")]
        stages.append(run_stage("codeql", "decode", command, repo, tool_dir, timeout))
    status = "completed" if database_stage["status"] == "completed" and stages and all(stage["status"] == "completed" for stage in stages) else "error"
    artifacts = ["sensitive-models.bqrs", "sensitive-models.csv" if query == CODEQL_IMPACT_QUERY else "sensitive-models.json"]
    if sinks is not None:
        artifacts.append("candidate-sinks.csv")
    return {"tool": "codeql", "status": status, "official_source": "refs/codeql", "query": str(query), "database": str(database), "database_status": database_stage, "stages": stages, "artifacts": artifacts}


def run_custom(tool: str, repo: Path, tool_dir: Path, command_text: str, timeout: int) -> dict:
    tool_dir.mkdir(parents=True, exist_ok=True)
    stage = run_stage(tool, "custom", shlex.split(command_text) + [str(repo)], repo, tool_dir, timeout)
    return single_stage_result(tool, "user-supplied official command", stage, [])


def run_tool(tool: str, repo: Path, root: Path, binaries: dict[str, dict[str, str] | None], custom: dict[str, str], timeout: int, workers: int, candidates: list[dict], phase: str, shared: dict[str, Any]) -> dict:
    tool_dir = root / tool
    if tool in custom:
        return run_custom(tool, repo, tool_dir, custom[tool], timeout)
    found = binaries.get(tool)
    if not found:
        return unavailable(tool, tool_dir, f"Official {tool} executable is unavailable")
    if tool == "bandit":
        return run_bandit(repo, tool_dir, found["main"], timeout, phase)
    if tool == "semgrep":
        return run_semgrep(repo, tool_dir, found["main"], timeout, phase)
    if tool == "pysa":
        return run_pysa(repo, tool_dir, found["main"], timeout, workers, phase, shared.get("pysa_work_dir"))
    if tool == "joern":
        return run_joern(repo, tool_dir, found, timeout)
    return run_codeql_query(
        repo,
        tool_dir,
        found["main"],
        shared["codeql_database"],
        shared["codeql_database_stage"],
        timeout,
        workers,
        shared["codeql_ram_mb"],
        phase,
    )


def run_tools(repo: Path, output_dir: Path, tools: list[str], custom: dict[str, str], timeout: int, workers: int, candidates: list[dict], phase: str, shared: dict[str, Any] | None = None) -> list[dict]:
    binaries = tool_executables()
    output_dir.mkdir(parents=True, exist_ok=True)
    shared = shared or {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(tools))) as pool:
        futures = [pool.submit(run_tool, tool, repo, output_dir, binaries, custom, timeout, workers, candidates, phase, shared) for tool in tools]
        return sorted((future.result() for future in concurrent.futures.as_completed(futures)), key=lambda item: item["tool"])


def relative_path(repo: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as error:
        logger.warning(f"Path is outside repository root: value={value}, repo={repo}, error={error}")
        normalized = path.as_posix()
        return normalized.rsplit("/repo/", 1)[-1] if "/repo/" in normalized else normalized


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return call_name(node.func)
    return ""


def calls_by_line(repo: Path) -> tuple[dict[tuple[str, int], list[dict]], list[dict]]:
    calls: dict[tuple[str, int], list[dict]] = {}
    parse_gaps = []
    for path in repo.rglob("*.py"):
        relative = path.relative_to(repo)
        if is_repository_path_excluded(relative.parts):
            continue
        file = relative.as_posix()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", SyntaxWarning)
                with tokenize.open(path) as source:
                    tree = ast.parse(source.read(), filename=str(path))
            warning_messages = sorted({str(warning.message) for warning in caught})
            for message in warning_messages:
                logger.warning(f"AST warning for {path}: {message}")
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            logger.warning(f"AST call extraction failed for {path}: {error}")
            parse_gaps.append({"file": file, "error_type": type(error).__name__, "error": str(error)})
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = call_name(node.func)
            if name:
                calls.setdefault((file, node.lineno), []).append({
                    "name": name,
                    "line": node.lineno,
                    "column": node.col_offset,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "end_column": getattr(node, "end_col_offset", node.col_offset),
                })
    return calls, parse_gaps


def matched_call_sites(calls: dict[tuple[str, int], list[dict]], file: str, line: int, start_column: int, end_column: int, one_based: bool = False, all_enclosing: bool = False) -> list[dict]:
    if one_based:
        start_column = max(0, start_column - 1)
        end_column = max(0, end_column - 1)
    enclosing = [item for item in calls.get((file, line), []) if item["column"] <= start_column and item["end_column"] >= end_column]
    if not enclosing:
        enclosing = [item for item in calls.get((file, line), []) if item["column"] < end_column and item["end_column"] > start_column]
    if not enclosing:
        return []
    selected = enclosing
    if not all_enclosing:
        shortest = min(item["end_column"] - item["column"] for item in enclosing)
        selected = [item for item in enclosing if item["end_column"] - item["column"] == shortest]
    return sorted(selected, key=lambda item: (item["column"], item["end_column"], item["name"]))


def matched_calls(calls: dict[tuple[str, int], list[dict]], file: str, line: int, start_column: int, end_column: int, one_based: bool = False) -> list[str]:
    return sorted({item["name"] for item in matched_call_sites(calls, file, line, start_column, end_column, one_based)})


def is_unknown_call(name: str) -> bool:
    parts = set(name.lower().split("."))
    return bool(parts & UNKNOWN_CALL_MARKERS)


def parse_bandit_candidates(repo: Path, tool_dir: Path, calls: dict[tuple[str, int], list[dict]]) -> list[dict]:
    path = tool_dir / "bandit/results.json"
    if not path.is_file():
        return []
    candidates = []
    for index, item in enumerate(json.loads(path.read_text(encoding="utf-8")).get("results", [])):
        rule_id = item.get("test_id", "")
        file = relative_path(repo, item.get("filename", ""))
        line = item.get("line_number", 0)
        call_sites = matched_call_sites(calls, file, line, item.get("col_offset", 0), item.get("end_col_offset", 0))
        names = sorted({site["name"] for site in call_sites})
        base = {"tool": "bandit", "artifact": "bandit/results.json", "result_index": index, "rule_id": rule_id, "file": file, "line": line, "column": item.get("col_offset", 0), "end_line": max(item.get("line_range") or [line]), "end_column": item.get("end_col_offset", 0), "call_sites": call_sites}
        if rule_id in {"RH001", "RH002"}:
            parts = item.get("issue_text", "").split("|", 2)
            if len(parts) >= 2:
                runtime_rule = next((rule for rule in RUNTIME_RULES if rule["rule_id"] == parts[1]), None)
                evidence_names = names or ([runtime_rule["fully_qualified_callee_or_event"]] if runtime_rule else [])
                candidates.extend(candidate_records(base, evidence_names, parts[0], parts[1]))
        elif rule_id in BANDIT_SINK_IDS:
            references = official_rule_references("bandit", rule_id)
            mapped = candidate_records(base, names)
            candidates.extend([with_official_references(record, references) for record in mapped])
    return candidates


def parse_semgrep_candidates(repo: Path, tool_dir: Path, calls: dict[tuple[str, int], list[dict]]) -> list[dict]:
    path = tool_dir / "semgrep/results.json"
    if not path.is_file():
        return []
    candidates = []
    for index, item in enumerate(json.loads(path.read_text(encoding="utf-8")).get("results", [])):
        rule_id = item.get("check_id", "")
        metadata = item.get("extra", {}).get("metadata", {})
        file = relative_path(repo, item.get("path", ""))
        line = item.get("start", {}).get("line", 0)
        call_sites = matched_call_sites(calls, file, line, item.get("start", {}).get("col", 1), item.get("end", {}).get("col", 1), one_based=True)
        names = sorted({site["name"] for site in call_sites})
        if metadata.get("category") != "security" or any(marker in rule_id for marker in SEMGREP_NON_SINK_MARKERS):
            continue
        base = {"tool": "semgrep", "artifact": "semgrep/results.json", "result_index": index, "rule_id": rule_id, "file": file, "line": line, "column": max(0, item.get("start", {}).get("col", 1) - 1), "end_line": item.get("end", {}).get("line", line), "end_column": max(0, item.get("end", {}).get("col", 1) - 1), "call_sites": call_sites}
        explicit_kind = metadata.get("sink_kind")
        explicit_rule_id = metadata.get("runtime_healing_rule_id")
        runtime_rule = next((rule for rule in RUNTIME_RULES if rule["rule_id"] == explicit_rule_id), None)
        evidence_names = names or ([runtime_rule["fully_qualified_callee_or_event"]] if runtime_rule else [])
        mapped = candidate_records(base, evidence_names, explicit_kind, explicit_rule_id)
        if explicit_kind and explicit_rule_id:
            candidates.extend(mapped)
        elif mapped:
            references = official_rule_references("semgrep", rule_id, item.get("extra", {}).get("message"), metadata)
            candidates.extend([with_official_references(record, references) for record in mapped])
    return candidates


def _graph_node_id(function: str, file: str, line: int, column: int, end_line: int, end_column: int) -> str:
    identity = json.dumps(
        [function, file, line, column, end_line, end_column],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "codeql-" + hashlib.sha256(identity.encode()).hexdigest()[:20]


def parse_codeql_graph(repo: Path, csv_path: Path) -> dict:
    files: dict[str, dict] = {}
    summaries = []
    sinks = []
    open_csv = gzip.open if csv_path.suffix == ".gz" else open
    with open_csv(csv_path, mode="rt", encoding="utf-8", newline="") as source:
        rows = csv.reader(source)
        header = next(rows, None)
        if header is None:
            raise RuntimeError(f"CodeQL graph output is empty: {csv_path}")
        for index, row in enumerate(rows, 1):
            if len(row) != 17:
                logger.warning(f"Unexpected CodeQL graph row {index}: {row}")
                continue
            (
                record_type, edge_type, source_function, source_variable, source_file,
                source_line, source_column, source_end_line, source_end_column,
                target_function, target_variable, target_file,
                target_line, target_column, target_end_line, target_end_column, detail,
            ) = row
            try:
                source_location = list(map(int, (source_line, source_column, source_end_line, source_end_column)))
                target_location = list(map(int, (target_line, target_column, target_end_line, target_end_column)))
            except ValueError as error:
                logger.warning(f"Invalid CodeQL graph location at row {index}: {error}")
                continue
            def graph_relative(value: str) -> str:
                normalized = Path(value).as_posix()
                if "/.tmp/source_view/" in normalized:
                    return normalized.split("/.tmp/source_view/", 1)[1]
                return relative_path(repo, value)

            source_relative = graph_relative(source_file)
            target_relative = graph_relative(target_file)
            if (
                is_repository_path_excluded(Path(source_relative).parts)
                or is_repository_path_excluded(Path(target_relative).parts)
            ):
                continue
            source_id = _graph_node_id(source_function, source_relative, *source_location)
            target_id = _graph_node_id(target_function, target_relative, *target_location)
            source_node = {
                "node_id": source_id, "function": source_function,
                **({"variable": source_variable} if source_variable else {}),
                "line": source_location[0], "column": max(0, source_location[1] - 1),
                "end_line": source_location[2], "end_column": max(0, source_location[3] - 1),
            }
            target_node = {
                "node_id": target_id, "function": target_function,
                **({"variable": target_variable} if target_variable else {}),
                "line": target_location[0], "column": max(0, target_location[1] - 1),
                "end_line": target_location[2], "end_column": max(0, target_location[3] - 1),
            }
            for relative_file, node in ((source_relative, source_node), (target_relative, target_node)):
                file_record = files.setdefault(relative_file, {"nodes": {}, "cfg_edges": {}, "dfg_edges": {}})
                existing = file_record["nodes"].setdefault(node["node_id"], node)
                variables = set(existing.get("variables", []))
                if existing.get("variable"):
                    variables.add(existing["variable"])
                if node.get("variable"):
                    variables.add(node["variable"])
                    existing.setdefault("variable", node["variable"])
                if variables:
                    existing["variables"] = sorted(variables)
            edge = {"source_id": source_id, "target_id": target_id, "edge_type": edge_type}
            edge_key = json.dumps(edge, ensure_ascii=False, sort_keys=True)
            if record_type == "cfg_edge":
                files[source_relative]["cfg_edges"].setdefault(edge_key, edge)
            elif record_type == "dfg_edge":
                files[source_relative]["dfg_edges"].setdefault(edge_key, edge)
            elif record_type == "function_summary":
                summaries.append({**edge, "function": source_function, "sink_kind": detail})
            elif record_type == "sink":
                sinks.append({**target_node, "file": target_relative, "sink_kind": detail})
    normalized_files = {
        file: {
            "nodes": sorted(record["nodes"].values(), key=lambda item: item["node_id"]),
            "cfg_edges": list(record["cfg_edges"].values()),
            "dfg_edges": list(record["dfg_edges"].values()),
        }
        for file, record in sorted(files.items())
    }
    return {
        "schema_version": 1,
        "producer": "codeql",
        "query": str(CODEQL_GRAPH_QUERY),
        "files": normalized_files,
        "function_summaries": deduplicate(summaries),
        "sinks": deduplicate(sinks),
    }


def _codeql_record(
    repo: Path,
    calls: dict[tuple[str, int], list[dict]],
    row: list,
    index: int,
    artifact: str,
    failures: list[dict] | None = None,
) -> dict | None:
    if len(row) != 13:
        failure = {"reason": "unexpected_codeql_row", "result_index": index, "artifact": artifact, "row": row}
        if failures is not None:
            failures.append(failure)
        logger.warning(f"Unexpected CodeQL row {index}: {row}")
        return None
    record_type, kind, file, start_line, start_column, end_line, end_column, detail, sink_file, sink_line, sink_column, sink_end_line, sink_end_column = row
    try:
        start_line, start_column, end_line, end_column = map(int, (start_line, start_column, end_line, end_column))
        sink_line, sink_column, sink_end_line, sink_end_column = map(int, (sink_line, sink_column, sink_end_line, sink_end_column))
    except ValueError as error:
        failure = {
            "reason": "invalid_codeql_location", "result_index": index,
            "artifact": artifact, "row": row, "error": str(error),
        }
        if failures is not None:
            failures.append(failure)
        logger.warning(f"Invalid CodeQL location at row {index}: {error}")
        return None
    relative_file = relative_path(repo, file) if start_line else file
    relative_sink_file = relative_path(repo, sink_file) if sink_line else sink_file
    sink_sites = matched_call_sites(calls, relative_sink_file, sink_line, sink_column, sink_end_column, one_based=True, all_enclosing=True) if record_type == "sink" and sink_line else []
    sink_calls = sorted({site["name"] for site in sink_sites})
    matched_kinds = {rule["sink_kind"] for rule in rules_for_calls(sink_calls)}
    mapped_kind = matched_kinds.pop() if record_type == "sink" and len(matched_kinds) == 1 else "UNKNOWN_EFFECT"
    record = {"tool": "codeql", "artifact": artifact, "result_index": index, "record_type": record_type, "kind": kind, "mapped_sink_kind": mapped_kind, "file": relative_file, "line": start_line, "column": start_column, "end_line": end_line, "end_column": end_column, "detail": detail, "sink_file": relative_sink_file, "sink_line": sink_line, "sink_column": sink_column, "sink_end_line": sink_end_line, "sink_end_column": sink_end_column, "sink_calls": sink_calls, "sink_call_sites": sink_sites}
    references = official_rule_references("codeql", kind) if record_type == "sink" else []
    return with_official_references(record, references)


def iter_codeql_records(
    repo: Path,
    tool_dir: Path,
    calls: dict[tuple[str, int], list[dict]],
    failures: list[dict] | None = None,
):
    csv_path = tool_dir / "codeql/sensitive-models.csv"
    json_path = tool_dir / "codeql/sensitive-models.json"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as source:
            rows = csv.reader(source)
            header = next(rows, None)
            if header and header[0] != "recordType":
                record = _codeql_record(repo, calls, header, 0, "codeql/sensitive-models.csv", failures)
                if record:
                    yield record
            for index, row in enumerate(rows, 1):
                record = _codeql_record(repo, calls, row, index, "codeql/sensitive-models.csv", failures)
                if record:
                    yield record
        return
    if not json_path.is_file():
        return
    rows = json.loads(json_path.read_text(encoding="utf-8")).get("#select", {}).get("tuples", [])
    for index, row in enumerate(rows):
        record = _codeql_record(repo, calls, row, index, "codeql/sensitive-models.json", failures)
        if record:
            yield record


def parse_codeql_records(repo: Path, tool_dir: Path, calls: dict[tuple[str, int], list[dict]]) -> list[dict]:
    return list(iter_codeql_records(repo, tool_dir, calls))


def parse_pysa_records(repo: Path, tool_dir: Path) -> list[dict]:
    path = tool_dir / "pysa/results/taint-output.json"
    if not path.is_file():
        return []
    records = []
    for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            logger.error(f"Pysa JSONL parse failed for {path} line {line_index + 1}: {error}")
            continue
        if item.get("kind") != "model":
            continue
        data = item.get("data", {})
        model_path = str(data.get("path", ""))
        if not model_path.startswith(str(repo)):
            continue
        file = relative_path(repo, model_path)
        callable_line = data.get("callable_line", 0)
        for sink_index, sink in enumerate(data.get("sinks", [])):
            for taint_index, taint in enumerate(sink.get("taint", [])):
                origin = taint.get("origin", {})
                for kind_index, kind in enumerate(taint.get("kinds", [])):
                    kind_name = str(kind.get("kind", ""))
                    if kind_name.startswith(("PartialSink[", "Triggered[")) or kind_name == "AddFeatureToArgument":
                        continue
                    matched_kinds = {rule["sink_kind"] for rule in rules_for_calls([str(data.get("callable", ""))])}
                    mapped_kind = matched_kinds.pop() if len(matched_kinds) == 1 else "UNKNOWN_EFFECT"
                    references = official_rule_references("pysa", kind_name)
                    record = {"tool": "pysa", "artifact": "pysa/results/taint-output.json", "result_index": line_index, "sink_index": sink_index, "taint_index": taint_index, "kind_index": kind_index, "record_type": "sink", "kind": kind_name, "mapped_sink_kind": mapped_kind, "file": file, "line": origin.get("line", callable_line), "end_line": origin.get("line", callable_line), "callable": data.get("callable", "")}
                    records.append(with_official_references(record, references))
        for tito_index, tito in enumerate(data.get("tito", [])):
            for taint_index, taint in enumerate(tito.get("taint", [])):
                for position_index, position in enumerate(taint.get("tito_positions", [])):
                    records.append({"tool": "pysa", "artifact": "pysa/results/taint-output.json", "result_index": line_index, "tito_index": tito_index, "taint_index": taint_index, "position_index": position_index, "record_type": "summary", "kind": "tito", "file": file, "line": position.get("line", callable_line), "end_line": position.get("line", callable_line), "callable": data.get("callable", "")})
        if data.get("sinks") or data.get("tito"):
            records.append({"tool": "pysa", "artifact": "pysa/results/taint-output.json", "result_index": line_index, "record_type": "wrapper", "kind": "callable-model", "file": file, "line": callable_line, "end_line": callable_line, "callable": data.get("callable", "")})
    return records


def coordinate(record: dict) -> tuple[str, int]:
    return record.get("file", ""), int(record.get("line", 0) or 0)


def build_raw_sinks(candidates: list[dict], codeql: list[dict], pysa: list[dict]) -> list[dict]:
    records = [{**record, "sink_source": record["tool"]} for record in candidates]
    for record in codeql:
        if record.get("record_type") == "sink" and record.get("line") and record.get("mapped_sink_kind") != "UNKNOWN_EFFECT":
            records.append({
                "tool": "codeql",
                "sink_source": "codeql",
                "file": record["file"],
                "line": record["line"],
                "end_line": record["end_line"],
                "calls": record.get("sink_calls", []),
                "sink_kind": record["mapped_sink_kind"],
                "rule_id": record["kind"],
                "runtime_rule_id": f"CODEQL.{record['kind']}",
                "artifact": record["artifact"],
                "result_index": record["result_index"],
            })
    for record in pysa:
        if record.get("record_type") == "sink" and record.get("line") and record.get("mapped_sink_kind") != "UNKNOWN_EFFECT":
            records.append({
                "tool": "pysa",
                "sink_source": "pysa",
                "file": record["file"],
                "line": record["line"],
                "end_line": record["end_line"],
                "calls": [record["callable"]] if record.get("callable") else [],
                "sink_kind": record["mapped_sink_kind"],
                "rule_id": record["kind"],
                "runtime_rule_id": f"PYSA.{record['kind']}",
                "artifact": record["artifact"],
                "result_index": record["result_index"],
            })
    return deduplicate(records)


def merge_confirmed_sinks(confirmed: list[dict], pysa: list[dict]) -> list[dict]:
    accepted = list(confirmed)
    for record in pysa:
        if record.get("record_type") != "sink" or not record.get("line") or record.get("mapped_sink_kind") == "UNKNOWN_EFFECT":
            continue
        accepted.append({
            "tool": "pysa",
            "file": record["file"],
            "line": record["line"],
            "end_line": record["end_line"],
            "calls": [record["callable"]] if record.get("callable") else [],
            "sink_kind": record["mapped_sink_kind"],
            "runtime_rule_id": f"PYSA.{record['kind']}",
            "rule_id": record["kind"],
            "status": "CONFIRMED",
            "confirmed_by": [{"tool": "pysa", "artifact": record["artifact"], "result_index": record["result_index"], "scan_phase": record.get("scan_phase"), "kind": record["kind"]}],
        })
    grouped: dict[tuple[str, int, int, int, int, str], dict] = {}
    for record in accepted:
        key = (
            record["file"], record["line"], record.get("column", 0),
            record["end_line"], record.get("end_column", 0), record["sink_kind"],
        )
        group = grouped.setdefault(key, {
            "file": record["file"], "line": record["line"], "column": record.get("column", 0),
            "end_line": record["end_line"], "end_column": record.get("end_column", 0),
            "sink_kind": record["sink_kind"], "sink_kinds": [], "kind_conflict": False,
            "status": "CONFIRMED", "calls": [], "source_tools": [], "rule_ids": [],
            "candidate_evidence": [], "confirmed_by": [],
        })
        group["column"] = min(group["column"], record.get("column", group["column"]))
        group["end_line"] = max(group["end_line"], record["end_line"])
        group["end_column"] = max(group["end_column"], record.get("end_column", group["end_column"]))
        group["calls"] = sorted(set(group["calls"] + record.get("calls", [])))
        group["sink_kinds"] = sorted(set(group["sink_kinds"] + record.get("sink_kinds", [record["sink_kind"]])))
        group["kind_conflict"] = group["kind_conflict"] or record.get("kind_conflict", False)
        group["source_tools"] = sorted(set(group["source_tools"] + [record["tool"]]))
        group["rule_ids"] = sorted(set(group["rule_ids"] + [record["runtime_rule_id"]]))
        group["candidate_evidence"].append(record)
        group["confirmed_by"].extend(record.get("confirmed_by", []))
    for group in grouped.values():
        group["candidate_evidence"] = deduplicate(group["candidate_evidence"])
        group["confirmed_by"] = deduplicate(group["confirmed_by"])
    return sorted(grouped.values(), key=lambda record: (record["file"], record["line"], record["sink_kind"]))


def site_overlaps(candidate: dict, record: dict) -> bool:
    if candidate.get("file") != record.get("sink_file"):
        return False
    candidate_start = (int(candidate.get("line", 0)), int(candidate.get("column", 0)))
    candidate_end = (int(candidate.get("end_line", candidate_start[0])), int(candidate.get("end_column", candidate_start[1])))
    record_start = (int(record.get("sink_line", 0)), max(0, int(record.get("sink_column", 1)) - 1))
    record_end = (int(record.get("sink_end_line", record_start[0])), max(0, int(record.get("sink_end_column", 1)) - 1))
    return candidate_start <= record_end and record_start <= candidate_end


def build_confirmed_sinks(
    candidates: list[dict],
    codeql: list[dict],
    parse_gap_files: set[str],
    codeql_available: bool,
) -> tuple[list[dict], list[dict]]:
    codeql_sinks = [record for record in codeql if record.get("record_type") == "sink" and record.get("sink_line")]
    codeql_by_file: dict[str, list[dict]] = {}
    for record in codeql_sinks:
        codeql_by_file.setdefault(record["sink_file"], []).append(record)
    confirmed = []
    unknown = []
    for candidate in candidates:
        if not codeql_available:
            unknown.append({**candidate, "status": "UNKNOWN", "reason": "UNKNOWN_CODEQL_UNAVAILABLE"})
            continue
        if candidate["file"] in parse_gap_files or not candidate.get("call_sites"):
            unknown.append({**candidate, "status": "UNKNOWN", "reason": "UNKNOWN_PARSE_GAP" if candidate["file"] in parse_gap_files else "UNKNOWN_CALL_LOCATION"})
            continue
        if candidate.get("sink_kind") == "UNKNOWN_EFFECT":
            unknown.append({**candidate, "status": "UNKNOWN", "reason": "UNKNOWN_EFFECT"})
            continue
        position_matches = [record for record in codeql_by_file.get(candidate["file"], []) if site_overlaps(candidate, record)]
        candidate_calls = set(candidate.get("calls", []))
        call_matches = [record for record in position_matches if candidate_calls & set(record.get("sink_calls", []))]
        if not position_matches:
            reason = "UNKNOWN_DYNAMIC_CALL" if any(is_unknown_call(call) for call in candidate_calls) else "UNKNOWN_NOT_CONFIRMED_BY_CODEQL"
            unknown.append({**candidate, "status": "UNKNOWN", "reason": reason})
            continue
        if not call_matches:
            unknown.append({**candidate, "status": "UNKNOWN", "reason": "UNKNOWN_CODEQL_CALL_MISMATCH", "codeql_position_evidence": position_matches})
            continue
        kind_matches = [record for record in call_matches if record.get("mapped_sink_kind") == candidate["sink_kind"]]
        if not kind_matches:
            unknown.append({**candidate, "status": "UNKNOWN", "reason": "UNKNOWN_CODEQL_KIND_MISMATCH", "codeql_call_evidence": call_matches})
            continue
        codeql_kinds = sorted({record["mapped_sink_kind"] for record in kind_matches})
        confirmed_by = deduplicate([{"tool": record["tool"], "artifact": record["artifact"], "result_index": record["result_index"], "scan_phase": record.get("scan_phase"), "kind": record["kind"], "mapped_sink_kind": record.get("mapped_sink_kind")} for record in kind_matches])
        confirmed.append({**candidate, "status": "CONFIRMED", "sink_kinds": codeql_kinds, "kind_conflict": False, "confirmed_by": confirmed_by})
    return deduplicate(confirmed), deduplicate(unknown)


def deduplicate(records: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for record in records:
        key = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result


def compact_sensitive_region(records: list[dict]) -> tuple[list[dict], list[dict]]:
    sink_keys = sorted({
        (
            record.get("sink_file", ""), int(record.get("sink_line", 0)),
            int(record.get("sink_column", 0)), int(record.get("sink_end_line", 0)),
            int(record.get("sink_end_column", 0)), record.get("kind", ""),
        )
        for record in records
        if record.get("record_type") in {"sink", "data", "control", "exception", "reachable", "caller"}
        and record.get("sink_file") and record.get("sink_line")
    })
    sink_ids = {key: f"sink-{index:06d}" for index, key in enumerate(sink_keys, 1)}
    sinks = [
        {
            "sink_id": sink_ids[key], "file": key[0], "line": key[1], "column": key[2],
            "end_line": key[3], "end_column": key[4], "kind": key[5],
        }
        for key in sink_keys
    ]
    grouped: dict[tuple, dict] = {}
    for record in records:
        if record.get("record_type") not in {"sink", "data", "control", "exception", "reachable", "caller"}:
            continue
        node_key = (
            record.get("file", ""), int(record.get("line", 0)), int(record.get("column", 0)),
            int(record.get("end_line", 0)), int(record.get("end_column", 0)),
        )
        if not node_key[0] or not node_key[1]:
            continue
        node = grouped.setdefault(node_key, {
            "file": node_key[0], "line": node_key[1], "column": node_key[2],
            "end_line": node_key[3], "end_column": node_key[4],
            "relation_types": set(), "sink_ids": set(),
        })
        node["relation_types"].add(record["record_type"])
        sink_key = (
            record.get("sink_file", ""), int(record.get("sink_line", 0)),
            int(record.get("sink_column", 0)), int(record.get("sink_end_line", 0)),
            int(record.get("sink_end_column", 0)), record.get("kind", ""),
        )
        if sink_key in sink_ids:
            node["sink_ids"].add(sink_ids[sink_key])
    nodes = [
        {
            **{key: value for key, value in node.items() if key not in {"relation_types", "sink_ids"}},
            "relation_types": sorted(node["relation_types"]), "sink_ids": sorted(node["sink_ids"]),
        }
        for _, node in sorted(grouped.items())
    ]
    return nodes, sinks


def run_joern_slices(tool_dir: Path, confirmed: list[dict], timeout: int, workers: int) -> list[dict]:
    binaries = tool_executables().get("joern")
    cpg = tool_dir / "joern/cpg.bin"
    if not binaries or not cpg.is_file():
        return []
    sink_calls = sorted({call for record in confirmed for call in record.get("calls", [])})
    if not sink_calls:
        logger.warning("Joern slicing skipped because confirmed sinks have no call evidence")
        return []
    pattern = ".*(?:" + "|".join(re.escape(name) for name in sink_calls) + r")\s*\(.*"
    output = tool_dir / "joern" / "slices.json"
    command = [binaries["slice"], "data-flow", "--slice-depth", "5", "--sink-filter", pattern, "--out", str(output), "--parallelism", str(workers), str(cpg)]
    return [run_stage("joern", "slice", command, tool_dir / "joern", tool_dir / "joern", timeout)]


def parse_joern_slices(repo: Path, tool_dir: Path) -> list[dict]:
    records = []
    for path in sorted((tool_dir / "joern").glob("slices.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        slices = data.get("dataFlowSlices", data.get("slices", []))
        if data.get("$type") == "DataFlowSlice":
            slices = [data]
        for slice_index, data_slice in enumerate(slices):
            for node_index, node in enumerate(data_slice.get("nodes", [])):
                line = int(node.get("lineNumber", 0) or 0)
                file = node.get("fileName") or node.get("parentFile") or ""
                if line and file:
                    records.append({"tool": "joern", "artifact": path.relative_to(tool_dir).as_posix(), "slice_index": slice_index, "node_index": node_index, "record_type": "data", "kind": "reverse-slice", "file": relative_path(repo, file), "line": line, "end_line": line})
    return records


def build_statement_results(candidates: list[dict], confirmed: list[dict], analysis_gaps: list[dict]) -> list[dict]:
    confirmed_keys = {(record["file"], record["line"], record["sink_kind"], record["runtime_rule_id"]) for record in confirmed}
    gap_keys = {
        (record.get("file", ""), record.get("line", 0), record.get("sink_kind", "UNKNOWN_EFFECT"), record.get("runtime_rule_id", record.get("rule_id", "")))
        for record in analysis_gaps
    }
    groups: dict[tuple[str, int, str, str], dict[str, set]] = {}
    for record in candidates:
        key = (record["file"], record["line"], record["sink_kind"], record["runtime_rule_id"])
        group = groups.setdefault(key, {"tools": set(), "tool_rule_ids": set(), "calls": set()})
        group["tools"].add(record["tool"])
        group["tool_rule_ids"].add(record["rule_id"])
        group["calls"].update(record["calls"])
    results = []
    for (file, line, sink_kind, rule_id), evidence in sorted(groups.items()):
        key = (file, line, sink_kind, rule_id)
        result = "CONFIRMED_SINK" if key in confirmed_keys else "ANALYSIS_GAP"
        if key not in confirmed_keys and key not in gap_keys:
            logger.warning(f"Candidate has neither confirmation nor analysis gap: {file}:{line} {rule_id}")
        results.append({"file": file, "line": line, "result": result, "sink_kind": sink_kind, "rule_id": rule_id, "calls": sorted(evidence["calls"]), "tools": sorted(evidence["tools"]), "tool_rule_ids": sorted(evidence["tool_rule_ids"])})
    return results


def stable_sink_id(record: dict) -> str:
    identity = {
        "file": record["file"], "line": record["line"], "column": record["column"],
        "end_line": record["end_line"], "end_column": record["end_column"],
        "sink_kind": record["sink_kind"], "rule_ids": record["rule_ids"], "calls": record["calls"],
    }
    return "sink-" + hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def build_sink_statistics(confirmed: list[dict], analysis_gaps: list[dict]) -> dict:
    def counts(records: list[dict], field: str) -> dict[str, int]:
        values: dict[str, int] = {}
        for record in records:
            for value in record.get(field, []):
                values[value] = values.get(value, 0) + 1
        return dict(sorted(values.items()))

    gap_reasons: dict[str, int] = {}
    for record in analysis_gaps:
        reason = record.get("reason", "UNSPECIFIED_ANALYSIS_GAP")
        gap_reasons[reason] = gap_reasons.get(reason, 0) + 1
    return {
        "confirmed_sink_count": len(confirmed),
        "confirmed_sink_counts_by_kind": counts(confirmed, "sink_kinds"),
        "confirmed_sink_counts_by_tool": counts(confirmed, "source_tools"),
        "confirmed_sink_counts_by_rule": counts(confirmed, "rule_ids"),
        "analysis_gap_count": len(analysis_gaps),
        "analysis_gap_counts_by_reason": dict(sorted(gap_reasons.items())),
    }


def build_impact(confirmed: list[dict], codeql: list[dict], pysa: list[dict], joern: list[dict]) -> list[dict]:
    facts = []
    confirmed_coordinates = {coordinate(record) for record in confirmed}
    codeql_by_evidence = {(record["artifact"], record["result_index"], record.get("scan_phase")): record for record in codeql if record.get("record_type") == "sink"}
    sink_metadata: dict[tuple, dict[str, set[str]]] = {}
    for item in confirmed:
        for confirmation in item.get("confirmed_by", []):
            evidence_key = (confirmation["artifact"], confirmation["result_index"], confirmation.get("scan_phase"))
            if confirmation["tool"] != "codeql" or not (record := codeql_by_evidence.get(evidence_key)):
                continue
            key = (record["file"], record["line"], record["column"], record["end_line"], record["end_column"], record["kind"])
            values = sink_metadata.setdefault(key, {"sink_kinds": set(), "rule_ids": set()})
            values["sink_kinds"].add(item["sink_kind"])
            values["rule_ids"].update(item["rule_ids"])
    callable_sink_kinds: dict[str, set[str]] = {}
    for record in pysa:
        if record.get("record_type") == "sink" and coordinate(record) in confirmed_coordinates and record.get("callable"):
            callable_sink_kinds.setdefault(record["callable"], set()).add(record["mapped_sink_kind"])
    all_sink_kinds = {record["sink_kind"] for record in confirmed}
    for record in confirmed:
        for source in record["source_tools"]:
            for rule_id in record["rule_ids"]:
                facts.append((record["file"], record["line"], record["end_line"], "sink", source, rule_id, record["sink_kind"], rule_id, "CONFIRMED"))
    for record in codeql:
        sink_key = (record.get("sink_file", ""), record.get("sink_line", 0), record.get("sink_column", 0), record.get("sink_end_line", 0), record.get("sink_end_column", 0), record.get("kind", ""))
        if record["record_type"] not in {"data", "control", "exception", "reachable", "caller"} or not record["line"] or sink_key not in sink_metadata:
            continue
        role = "reachable" if record["record_type"] in {"reachable", "caller"} else record["record_type"]
        for sink_kind in sink_metadata[sink_key]["sink_kinds"]:
            for rule_id in sink_metadata[sink_key]["rule_ids"]:
                facts.append((record["file"], record["line"], record["end_line"], role, "codeql", record["kind"], sink_kind, rule_id, "CONFIRMED"))
    for record in pysa:
        if record["record_type"] in {"summary", "wrapper"} and record["line"] and record.get("callable", "") in callable_sink_kinds:
            for sink_kind in callable_sink_kinds[record["callable"]]:
                facts.append((record["file"], record["line"], record["end_line"], record["record_type"], "pysa", record["kind"], sink_kind, "PYSA_SUMMARY", "CONFIRMED"))
    for record in joern:
        for sink_kind in all_sink_kinds:
            facts.append((record["file"], record["line"], record["end_line"], "data", "joern", "reverse-slice", sink_kind, "JOERN_SLICE", "CONFIRMED"))
    grouped: dict[tuple[str, int, int, str, str], dict[str, set[str]]] = {}
    for file, line, end_line, role, source, tool_kind, sink_kind, rule_id, status in facts:
        group = grouped.setdefault((file, line, end_line, role, status), {"sources": set(), "tool_kinds": set(), "sink_kinds": set(), "rule_ids": set()})
        group["sources"].add(source)
        group["tool_kinds"].add(tool_kind)
        group["sink_kinds"].add(sink_kind)
        group["rule_ids"].add(rule_id)
    return [{"file": file, "line": line, "end_line": end_line, "role": role, "sources": sorted(values["sources"]), "tool_kinds": sorted(values["tool_kinds"]), "sink_kinds": sorted(values["sink_kinds"]), "rule_ids": sorted(values["rule_ids"]), "status": status} for (file, line, end_line, role, status), values in sorted(grouped.items())]


def run_scan_phase(repo: Path, root: Path, tools: list[str], custom: dict[str, str], timeout: int, workers: int, calls: dict, phase: str, shared: dict[str, Any]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    statuses = run_tools(repo, root, tools, custom, timeout, workers, [], phase, shared)
    candidates = parse_bandit_candidates(repo, root, calls) + parse_semgrep_candidates(repo, root, calls)
    statuses.sort(key=lambda item: item["tool"])
    candidates = [{**record, "scan_phase": phase} for record in candidates]
    codeql = [{**record, "scan_phase": phase} for record in parse_codeql_records(repo, root, calls)]
    return statuses, candidates, codeql, []


FIND_RESULT_FILES = {
    "bandit": "find_bandit_sinks.json",
    "semgrep": "find_semgrep_sinks.json",
    "codeql": "find_codeql_sinks.json",
}


def clean_phase_outputs(output_dir: Path, phase: str) -> None:
    files = {
        "find": [*FIND_RESULT_FILES.values()],
        "combine": ["combined_sinks.json"],
    }
    ordered = ["find", "combine"]
    for current in ordered[ordered.index(phase):]:
        for name in files[current]:
            (output_dir / name).unlink(missing_ok=True)
    if phase == "find":
        (output_dir / "codeql_graph.csv").unlink(missing_ok=True)
        (output_dir / "codeql_graph.json").unlink(missing_ok=True)
        temporary = output_dir / ".tmp"
        if temporary.exists():
            try:
                shutil.rmtree(temporary)
            except OSError as error:
                logger.error(f"Failed to clean temporary directory {temporary}: {error}")
                raise


def load_find_outputs(output_dir: Path) -> dict[str, dict]:
    return {
        tool: json_load(output_dir / filename)
        for tool, filename in FIND_RESULT_FILES.items()
    }


def find_result_summary(find_outputs: dict[str, dict]) -> dict:
    statuses = {tool: evidence.get("status") for tool, evidence in find_outputs.items()}
    parse_gaps = deduplicate([
        record
        for evidence in find_outputs.values()
        for record in evidence.get("parse_gaps", [])
    ])
    return {
        "phase": "find",
        "status": "completed" if all(status == "completed" for status in statuses.values()) else "completed_with_gaps",
        "tool_statuses": statuses,
        "candidate_count": sum(len(find_outputs[tool].get("sinks", [])) for tool in ("bandit", "semgrep")),
        "parse_gap_count": len(parse_gaps),
    }


def reusable_find_outputs(output_dir: Path, expected_input: dict) -> dict[str, dict] | None:
    if not all((output_dir / filename).is_file() for filename in FIND_RESULT_FILES.values()):
        return None
    find_outputs = load_find_outputs(output_dir)
    if any(evidence.get("input") != expected_input for evidence in find_outputs.values()):
        return None
    if any(evidence.get("status") != "completed" for evidence in find_outputs.values()):
        return None
    return find_outputs


def find_sinks(repo: Path, output_dir: Path, tools: list[str], custom: dict[str, str], timeout: int, workers: int, codeql_ram_mb: int, metadata: dict) -> dict:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_phase_outputs(output_dir, "find")
    source_view, source_scope = prepare_source_view(repo, output_dir)
    input_record = {"phase": "find", "repo": str(repo), "source_view": str(source_view), "metadata": metadata, "tools": tools, "tool_commands": custom, "timeout": timeout, "workers": workers, "codeql_ram_mb": codeql_ram_mb}
    calls, parse_gaps = calls_by_line(source_view)
    source_copy_gaps = [
        {"file": record["path"], "error_type": record["error_type"], "error": record["error"], "reason": "SOURCE_COPY_GAP"}
        for record in source_scope["copy_gaps"]
        if record["path"].endswith(".py")
    ]
    parse_gaps = deduplicate([*parse_gaps, *source_copy_gaps])
    scanner_tools = [tool for tool in tools if tool in {"bandit", "semgrep", "codeql"}]
    find_root = output_dir / ".tmp" / "find"
    native_codeql = "codeql" in scanner_tools and "codeql" not in custom
    phase_tools = [tool for tool in scanner_tools if tool != "codeql"]
    if "codeql" in custom:
        phase_tools.append("codeql")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        scan_future = pool.submit(run_scan_phase, source_view, find_root, phase_tools, custom, timeout, workers, calls, "find", {})
        database_future = None
        if native_codeql and (binary := tool_executables()["codeql"]):
            database_future = pool.submit(
                create_codeql_database,
                source_view,
                output_dir / "cache" / "codeql_database",
                binary["main"],
                timeout,
                workers,
                source_scope["source_fingerprint"],
            )
        statuses, candidates, codeql, pysa = scan_future.result()
        codeql_database = database_future.result() if database_future else None
    if native_codeql:
        if codeql_database:
            database, database_stage = codeql_database
            resources = {
                "codeql_database": database,
                "codeql_database_stage": database_stage,
                "codeql_ram_mb": codeql_ram_mb,
            }
            statuses.extend(run_tools(source_view, find_root, ["codeql"], custom, timeout, workers, [], "shared", resources))
            codeql = [{**record, "scan_phase": "find"} for record in parse_codeql_records(source_view, find_root, calls)]
            if database_stage["status"] == "completed":
                graph_result = run_codeql_graph_query(
                    source_view, output_dir, binary["main"], database,
                    timeout, workers, codeql_ram_mb,
                    source_scope["source_fingerprint"],
                )
                codeql_status = next(status for status in statuses if status["tool"] == "codeql")
                codeql_status["graph"] = graph_result
                if graph_result["status"] != "completed":
                    codeql_status["status"] = "error"
        else:
            statuses.append(unavailable("codeql", find_root / "codeql", "CodeQL executable is unavailable"))
    candidates = deduplicate(candidates)
    codeql = deduplicate(codeql)
    pysa = deduplicate(pysa)
    raw_sinks = build_raw_sinks(candidates, codeql, pysa)
    statuses = sorted(({**status, "phase": "find"} for status in statuses), key=lambda item: item["tool"])
    candidates_by_tool = {tool: [record for record in candidates if record["tool"] == tool] for tool in ("bandit", "semgrep")}
    statuses_by_tool = {status["tool"]: status for status in statuses}
    for tool in FIND_RESULT_FILES:
        if tool not in statuses_by_tool:
            statuses_by_tool[tool] = unavailable(tool, find_root / tool, f"{tool.title()} status is missing")
    find_outputs = {
        "bandit": complete_tool_evidence(find_root / "bandit", statuses_by_tool["bandit"], candidates_by_tool["bandit"], input_record, parse_gaps),
        "semgrep": complete_tool_evidence(find_root / "semgrep", statuses_by_tool["semgrep"], candidates_by_tool["semgrep"], input_record, parse_gaps),
        "codeql": complete_tool_evidence(find_root / "codeql", statuses_by_tool["codeql"], codeql, input_record, parse_gaps),
    }
    for tool, filename in FIND_RESULT_FILES.items():
        json_dump(output_dir / filename, find_outputs[tool])
    temporary = output_dir / ".tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    complete = bool(statuses) and all(item["status"] == "completed" for item in statuses) and not source_scope["copy_gaps"]
    return {
        "phase": "find",
        "status": "completed" if complete else "completed_with_gaps",
        "duration_s": round(time.time() - started, 3),
        "candidate_count": len(candidates),
        "raw_sink_count": len(raw_sinks),
        "parse_gap_count": len(parse_gaps),
        "source_scope": {key: value for key, value in source_scope.items() if key not in {"excluded_paths", "copy_gaps"}},
    }


def combine_sinks(output_dir: Path) -> dict:
    started = time.time()
    find_outputs = load_find_outputs(output_dir)
    inputs = [evidence.get("input") for evidence in find_outputs.values()]
    if not inputs[0] or any(value != inputs[0] for value in inputs[1:]):
        raise RuntimeError(f"Find tool inputs are missing or inconsistent; rerun --phase find for {output_dir}")
    expected = {tool for tool in inputs[0]["tools"] if tool in find_outputs}
    completed = {tool for tool, evidence in find_outputs.items() if evidence.get("status") == "completed"}
    required = {"bandit", "semgrep"}
    if not required.issubset(expected) or not required.issubset(completed):
        raise RuntimeError(f"Required Find tool evidence is incomplete; rerun --phase find for {output_dir}")
    codeql_available = "codeql" in expected and "codeql" in completed
    if not codeql_available:
        logger.warning(f"Combining without completed CodeQL evidence: {output_dir}")
    clean_phase_outputs(output_dir, "combine")
    parse_gaps = deduplicate(find_outputs["bandit"].get("parse_gaps", []) + find_outputs["semgrep"].get("parse_gaps", []) + find_outputs["codeql"].get("parse_gaps", []))
    candidates = find_outputs["bandit"]["sinks"] + find_outputs["semgrep"]["sinks"]
    codeql = find_outputs["codeql"]["sinks"]
    candidate_confirmed, unconfirmed = build_confirmed_sinks(
        candidates,
        codeql,
        {record["file"] for record in parse_gaps},
        codeql_available,
    )
    confirmed_sinks = merge_confirmed_sinks(candidate_confirmed, [])
    confirmed_sinks = [{"sink_id": stable_sink_id(record), **record} for record in confirmed_sinks]
    parse_analysis_gaps = [
        {**record, "status": "UNKNOWN", "reason": record.get("reason", "UNKNOWN_PARSE_GAP")}
        for record in parse_gaps
    ]
    analysis_gaps = deduplicate([*unconfirmed, *parse_analysis_gaps])
    statement_results = build_statement_results(candidates, candidate_confirmed, analysis_gaps)
    statistics = build_sink_statistics(confirmed_sinks, analysis_gaps)
    combined = {
        "phase": "combine",
        "status": "completed" if not analysis_gaps else "completed_with_gaps",
        "confirmed_sinks": confirmed_sinks,
        "analysis_gaps": analysis_gaps,
    }
    json_dump(output_dir / "combined_sinks.json", combined)
    return {
        "phase": "combine",
        "status": "completed" if not analysis_gaps else "completed_with_gaps",
        "duration_s": round(time.time() - started, 3),
        "candidate_count": len(candidates),
        "candidate_confirmed_count": len(candidate_confirmed),
        **statistics,
        "statement_result_count": len(statement_results),
    }


def run_phases(repo: Path, output_dir: Path, phase: str, tools: list[str], custom: dict[str, str], timeout: int, workers: int, codeql_ram_mb: int, metadata: dict, force: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".scan.lock"
    try:
        with lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                logger.error(f"Another scan is already writing this case: {output_dir}")
                raise RuntimeError(f"Another scan is already writing this case: {output_dir}") from error
            selected = ["find", "combine"] if phase == "all" else [phase]
            expected_input = {
                "phase": "find",
                "repo": str(repo),
                "source_view": str(output_dir / ".tmp" / "source_view"),
                "metadata": metadata,
                "tools": tools,
                "tool_commands": custom,
                "timeout": timeout,
                "workers": workers,
                "codeql_ram_mb": codeql_ram_mb,
            }
            result = {}
            rerun_downstream = False
            for current in selected:
                if current == "find" and not force and not rerun_downstream:
                    existing_outputs = reusable_find_outputs(output_dir, expected_input)
                    if existing_outputs:
                        result = find_result_summary(existing_outputs)
                        logger.info(f"skipped reusable phase=find, status={result['status']}: {output_dir}")
                        continue
                if current == "combine" and not force and not rerun_downstream:
                    combined_path = output_dir / "combined_sinks.json"
                    if combined_path.is_file():
                        existing = json_load(combined_path)
                        if existing.get("status") in {"completed", "completed_with_gaps"}:
                            result = existing
                            logger.info(f"skipped reusable phase=combine, status={existing['status']}: {output_dir}")
                            continue
                if current == "find":
                    result = find_sinks(repo, output_dir, tools, custom, timeout, workers, codeql_ram_mb, metadata)
                    rerun_downstream = True
                else:
                    result = combine_sinks(output_dir)
                    rerun_downstream = True
            return result
    finally:
        lock_path.unlink(missing_ok=True)


def parse_tool_commands(values: list[str]) -> dict[str, str]:
    commands = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--tool-command must be TOOL=COMMAND: {value}")
        tool, command = value.split("=", 1)
        commands[tool.strip()] = command.strip()
    return commands


def cached_repositories(
    cache_root: Path,
    source: str | None,
    per_dataset_limit: int,
    instance_file: str | None,
    repository_overrides: Path | None = None,
) -> list[dict]:
    _, dataset_records = load_dataset_records(source)
    before = len(dataset_records)
    dataset_records = [record for record in dataset_records if record["instance_id"] in HEALBENCH_INSTANCE_IDS]
    logger.info(f"Excluded {before - len(dataset_records)} records not present in the benchmark manifest")
    selected_ids = set(load_instance_file(instance_file)) if instance_file else set()
    if selected_ids:
        dataset_records = [record for record in dataset_records if record["instance_id"] in selected_ids]
    overrides = {}
    if repository_overrides:
        payload = yaml.safe_load(repository_overrides.read_text(encoding="utf-8"))
        overrides = {
            str(item["instance_id"]): Path(item["repository"])
            for item in payload.get("injections", [])
        }
        logger.info(f"Loaded {len(overrides)} repository overrides; first 2: {list(overrides.items())[:2]}")
    records = []
    for rank, record in enumerate(dataset_records):
        directory_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", record["repo"].replace("/", "__")) + f"___base-{record['base_commit'][:12]}___src-"
        matches = sorted(cache_root.glob(f"{directory_prefix}*/metadata.json"))
        if len(matches) != 1:
            logger.warning(f"Skipping source with {len(matches)} matching directories for {record['instance_id']}: {directory_prefix}")
            continue
        metadata_path = matches[0]
        metadata = json_load(metadata_path)
        repo = overrides.get(record["instance_id"], metadata_path.parent / metadata.get("source_path", "repo"))
        if metadata.get("repository") != record["repo"] or metadata.get("expected_commit") != record["base_commit"] or not repo.is_dir():
            logger.warning(f"Skipping mismatched repository source metadata: {metadata_path}")
            continue
        records.append({"repo": repo, "metadata": metadata, "dataset": record["source_dataset"], "rank": rank, "instance_id": record["instance_id"], "directory_name": metadata_path.parent.name})
    records.sort(key=lambda item: item["rank"])
    if not per_dataset_limit:
        return records
    selected = []
    dataset_counts: dict[str, int] = {}
    for item in records:
        dataset = item["dataset"]
        if dataset_counts.get(dataset, 0) >= per_dataset_limit:
            continue
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        selected.append(item)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze pre-downloaded repository source and generate confirmed sinks")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_REPO_CACHE)
    parser.add_argument("--source", choices=["SWE-Bench-Pro", "SWE-Bench-Verified", "SWE-Bench-Full-Test", "R2E-Gym-Lite"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-dataset-limit", type=int, default=0)
    parser.add_argument("--instance-file")
    parser.add_argument("--repository-overrides", type=Path)
    parser.add_argument("--phase", choices=["find", "combine", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tools", default="bandit,semgrep,codeql", help="Comma-separated tools; Pysa and Joern are disabled because repository-wide analysis is too slow")
    parser.add_argument("--tool-command", action="append", default=[])
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--tool-workers", type=int, default=max(2, min(4, os.cpu_count() or 2)))
    parser.add_argument("--codeql-ram", type=int, default=32768, help="Maximum RAM in MB per CodeQL query")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    tools = [item.strip() for item in args.tools.split(",") if item.strip()]
    invalid = sorted(set(tools) - SUPPORTED_TOOLS)
    if invalid:
        parser.error(f"unsupported tools: {invalid}")
    custom = parse_tool_commands(args.tool_command)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    init_logger(str(output_root / "run.log"))
    if args.repo:
        if args.phase in {"find", "all"} and not args.repo.resolve().is_dir():
            parser.error(f"repository directory does not exist: {args.repo}")
        result = run_phases(args.repo.resolve(), output_root, args.phase, tools, custom, args.timeout, args.tool_workers, args.codeql_ram, {"mode": "repo"}, args.force)
        logger.info(f"completed repo phase={args.phase}: status={result['status']}")
        return 0 if result["status"] in {"completed", "completed_with_gaps"} else 1

    cache_records = cached_repositories(
        args.cache_dir.resolve(),
        args.source,
        args.per_dataset_limit,
        args.instance_file,
        args.repository_overrides.resolve() if args.repository_overrides else None,
    )
    if args.limit:
        cache_records = cache_records[:args.limit]
    logger.info(f"selected {len(cache_records)} cached repositories; first 2: {[item['directory_name'] for item in cache_records[:2]]}")

    def process_cache(item: dict) -> dict:
        metadata = item["metadata"]
        case_dir = output_root / item["instance_id"]
        scan_metadata = {
            "mode": "cache",
            "source_directory": item["directory_name"],
            "repository": metadata["repository"],
            "expected_commit": metadata["expected_commit"],
            "verified_commit": metadata["verified_commit"],
            "source_status": metadata["source_status"],
            "source_id": metadata["source_id"],
            "image": metadata["docker_image"],
            "workdir": metadata["docker_workdir"],
            "instance_id": item["instance_id"],
            "source_dataset": item["dataset"],
        }
        return run_phases(item["repo"], case_dir, args.phase, tools, custom, args.timeout, args.tool_workers, args.codeql_ram, scan_metadata, args.force)

    work_items = cache_records
    processor = process_cache
    item_name = lambda item: item["instance_id"]

    results = []
    cpu_count = max(1, os.cpu_count() or 1)
    try:
        memory_mb = max(1, os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024))
    except (AttributeError, OSError, ValueError) as error:
        logger.warning(f"Physical memory detection failed; using one concurrent CodeQL instance: {error}")
        memory_mb = args.codeql_ram
    memory_parallel = max(1, int(memory_mb * 0.8) // max(1, args.codeql_ram))
    effective_parallel = max(
        1,
        min(args.parallel, cpu_count // max(1, args.tool_workers), memory_parallel),
    )
    if effective_parallel != args.parallel:
        logger.warning(
            f"Reduced outer parallelism from {args.parallel} to {effective_parallel} "
            f"for {args.tool_workers} tool workers and {args.codeql_ram} MB per instance"
        )
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=effective_parallel)
    try:
        future_to_item = {pool.submit(processor, item): item for item in work_items}
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
            except Exception as error:
                logger.error(f"{item_name(item)}: {type(error).__name__}: {error}")
                result = {"phase": args.phase, "status": "error", "metadata": {"item": item_name(item)}, "error": f"{type(error).__name__}: {error}"}
            result = {"item": item_name(item), **result}
            results.append(result)
            logger.info(f"{item_name(item)}: phase={args.phase}, status={result['status']}")
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True, cancel_futures=True)
    summary = {
        "phase": args.phase,
        "instances": len(results),
        "completed": sum(item["status"] == "completed" for item in results),
        "completed_with_gaps": sum(item["status"] == "completed_with_gaps" for item in results),
        "failed": sum(item["status"] not in {"completed", "completed_with_gaps"} for item in results),
        "results": results,
    }
    json_dump(output_root / f"summary_{args.phase}.json", summary)
    logger.info(f"summary phase={args.phase}: completed={summary['completed']}, gaps={summary['completed_with_gaps']}, failed={summary['failed']}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
