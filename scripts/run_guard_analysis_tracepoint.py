from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import pickle
import re
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from util import HEALBENCH_INSTANCE_IDS
from instrument_guard import validate_heal_code
from run_find_points_sinks import parse_codeql_graph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "healbench.jsonl"
INSTRUMENTED_ROOT = PROJECT_ROOT / "results" / "repo_instrumented"
PATH_POINTS_ROOT = PROJECT_ROOT / "results" / "find_points"
RESULTS_ROOT = PROJECT_ROOT / "results" / "healbench"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "tracepoint_analysis"
TRACE_PREFIX = "[heal_trace]"
REQUIRED_RESULT_FILES = {"input.json", "output.json", "run.log", "run_docker.log"}
DEFAULT_AGENTS = (
    "Healer",
    "MiniSweAgentAdapter",
    "OpenHandsAdapter",
    "CodexAdapter",
)
DEFAULT_MODELS = (
    "glm-5-2-260617",
    "deepseek-v4-flash-ga-260731",
    "gpt-5.6-terra",
)
AGENT_MODELS = {
    agent: DEFAULT_MODELS
    for agent in DEFAULT_AGENTS
}
CACHE_SCHEMA_VERSION = 2
STATIC_INDEX_CACHE_SCHEMA_VERSION = 2
ANALYSIS_SOURCE_PATHS = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "scripts" / "instrument_guard.py",
    PROJECT_ROOT / "scripts" / "heal_code_allowlist_blocklist.yaml",
)
MAX_GRAPH_WORKERS = 12


def _read_heal_trace(log_path: Path, event_type: str) -> list:
    records = []
    if not log_path.exists():
        return records
    prefix = "[heal_trace]"
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, 1):
            if not line.startswith(prefix):
                continue
            try:
                record = json.loads(line[len(prefix):])
            except Exception as error:
                logger.warning(f"invalid heal trace {log_path}:{line_number}: {error}")
                continue
            if record.get("type") == event_type:
                records.append(record.get("data"))
    return records


def _read_docker_output(log_path: Path) -> str:
    records = _read_heal_trace(log_path, "docker_result")
    if not records or not isinstance(records[-1], dict):
        return log_path.read_text("utf-8", errors="replace") if log_path.exists() else ""
    result = records[-1]
    return f"{result.get('stdout', '')}\n{result.get('stderr', '')}"


def classify_status(raw_status: str, docker_log_path: Path | None, test_rc: int | None = None) -> str:
    if raw_status == "correct":
        return "correct"
    if raw_status == "infra_error":
        return "infra_error"
    if test_rc is not None:
        try:
            rc = int(test_rc)
        except (ValueError, TypeError) as error:
            logger.warning(f"Invalid test return code {test_rc!r}: {error}")
            rc = None
        if rc == 0:
            return "correct"
        if rc is not None and rc >= 2:
            return "crash"
        if rc == -1:
            return "crash"
    if docker_log_path and docker_log_path.exists():
        docker_log = _read_docker_output(docker_log_path)
        has_failed = "FAILED" in docker_log
        failed_lines = re.findall(r"FAILED .+ - (.+)", docker_log)
        has_error_section = bool(re.search(r"(?m)^={3,} ERRORS ={3,}", docker_log))
        if has_error_section:
            return "crash"
        if has_failed and failed_lines:
            all_assertion = all(
                "AssertionError" in line or "assert" in line.lower() or "Failed:" in line
                for line in failed_lines
            )
            if all_assertion:
                return "proceed"
        elif has_failed:
            return "proceed"
    return "crash"


def load_instance_ids(
    limit: int,
    requested: list[str] | None,
    result_root: Path,
) -> list[str]:
    requested_ids = set(requested or [])
    records = []
    with DATASET_PATH.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                logger.error(f"Invalid dataset JSON at {DATASET_PATH}:{line_number}: {error}")
                raise

    dataset_ids = {str(record["instance_id"]) for record in records}
    missing = sorted(requested_ids - dataset_ids)
    if missing:
        raise RuntimeError(f"Requested instances were not found: {missing}")

    available_ids = {
        path.name
        for path in result_root.iterdir()
        if path.is_dir() and (path / "output.json").is_file()
    } if result_root.is_dir() else set()
    selected = []
    excluded = 0
    for record in records:
        instance_id = str(record["instance_id"])
        if instance_id not in HEALBENCH_INSTANCE_IDS:
            excluded += 1
            continue
        if requested_ids and instance_id not in requested_ids:
            continue
        if not requested_ids and instance_id not in available_ids:
            continue
        selected.append(instance_id)
        if limit and len(selected) >= limit:
            break
    logger.info(
        f"Excluded {excluded} records not present in the benchmark manifest"
    )
    logger.info(f"Selected {len(selected)} completed instances; first 2: {selected[:2]}")
    return selected


def file_signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def analysis_code_signature() -> str:
    digest = hashlib.sha256()
    for path in ANALYSIS_SOURCE_PATHS:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def report_cache_signature(
    instance_id: str,
    agent: str,
    model: str,
    include_raw: bool,
    code_signature: str,
) -> dict:
    result_dir = RESULTS_ROOT / agent / model / instance_id
    repository = INSTRUMENTED_ROOT / instance_id / "repo"
    graph_dir = PATH_POINTS_ROOT / instance_id
    paths = [
        result_dir / "output.json",
        result_dir / "run_docker.log",
        repository / "_instrumentation.json",
        graph_dir / "combined_key_points.json",
        graph_dir / "codeql_graph.csv.gz",
    ]
    for name in ("run_docker_events.jsonl", "run_docker_snapshots.jsonl"):
        path = result_dir / name
        if path.is_file():
            paths.append(path)
    if include_raw:
        value_path = result_dir / "run_docker_values.jsonl"
        if value_path.is_file():
            paths.append(value_path)
    return {
        "schema": CACHE_SCHEMA_VERSION,
        "analysis_code": code_signature,
        "include_raw": include_raw,
        "inputs": [file_signature(path) for path in paths if path.is_file()],
    }


def load_cached_reports(output_dir: Path) -> dict[tuple[str, str, str], dict]:
    cached = {}
    for reports_path in output_dir.glob("*/*/reports.json"):
        try:
            reports = json.loads(reports_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(f"Ignoring invalid report cache {reports_path}: {error}")
            continue
        for report in reports if isinstance(reports, list) else []:
            key = (str(report.get("instance_id")), str(report.get("agent")), str(report.get("model")))
            cached[key] = report
    for report_path in output_dir.glob("*/*/instances/*/report.json"):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(f"Ignoring invalid per-instance report cache {report_path}: {error}")
            continue
        key = (str(report.get("instance_id")), str(report.get("agent")), str(report.get("model")))
        cached[key] = report
    return cached


def calculate_graph_workers(requested: int, instance_count: int) -> int:
    return max(1, min(requested, MAX_GRAPH_WORKERS, max(1, instance_count)))


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.error(f"Failed to read JSON {path}: {error}")
        raise


def read_trace_records(path: Path, include_raw: bool = False) -> tuple[list[dict], list[dict]]:
    records = []
    errors = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.startswith(TRACE_PREFIX):
                continue
            try:
                records.append(json.loads(line[len(TRACE_PREFIX):]))
            except json.JSONDecodeError as error:
                logger.warning(f"Malformed [heal_trace] JSON at {path}:{line_number}: {error}")
                errors.append({
                    "check": "trace_json",
                    "detail": "malformed [heal_trace] record",
                    "line": line_number,
                    "error": str(error),
                    "raw": line.rstrip("\n"),
                })
    result_dir = path.parent
    runtime_artifacts = [
        ("run_docker_events.jsonl", "runtime_event"),
        ("run_docker_events.jsonl.malformed.jsonl", "runtime_malformed"),
        ("run_docker_snapshots.jsonl", "runtime_snapshot"),
    ]
    if include_raw:
        runtime_artifacts.append(("run_docker_values.jsonl", "runtime_value"))
    artifact_event_types = {
        event_type
        for artifact_name, event_type in runtime_artifacts
        if (result_dir / artifact_name).is_file()
    }
    records = [record for record in records if record.get("type") not in artifact_event_types]
    for artifact_name, event_type in runtime_artifacts:
        artifact_path = result_dir / artifact_name
        if not artifact_path.is_file():
            continue
        with artifact_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    records.append({"type": event_type, "data": json.loads(line)})
                except json.JSONDecodeError as error:
                    logger.warning(f"Malformed runtime artifact JSON at {artifact_path}:{line_number}: {error}")
                    errors.append({
                        "check": "trace_json",
                        "detail": "malformed runtime artifact record",
                        "file": artifact_name,
                        "line": line_number,
                        "error": str(error),
                        "raw": line.rstrip("\n"),
                    })
    return records, errors


def point_file(point_id: str) -> str:
    return point_id.split("::", 1)[0]


def node_function(node_id: str) -> str:
    parts = node_id.split("::")
    return parts[1] if len(parts) >= 3 else ""


def static_index_cache_path(instance_id: str) -> Path:
    return PATH_POINTS_ROOT / instance_id / "guard_static_index.pkl"


def static_index_cache_signature(instance_id: str) -> dict:
    repository = INSTRUMENTED_ROOT / instance_id / "repo"
    graph_dir = PATH_POINTS_ROOT / instance_id
    paths = [
        repository / "_instrumentation.json",
        graph_dir / "combined_key_points.json",
        graph_dir / "codeql_graph.csv.gz",
    ]
    metadata = load_json(repository / "_instrumentation.json")
    guard_manifest = metadata.get("artifacts", {}).get("guard_manifest", {}).get("path")
    if guard_manifest:
        paths.append(repository / guard_manifest)
    return {
        "schema": STATIC_INDEX_CACHE_SCHEMA_VERSION,
        "inputs": [file_signature(path) for path in paths if path.is_file()],
    }


def load_static_evidence(instance_id: str) -> dict:
    cache_path = static_index_cache_path(instance_id)
    cache_signature = static_index_cache_signature(instance_id)
    if cache_path.is_file():
        try:
            with cache_path.open("rb") as cache_file:
                cached = pickle.load(cache_file)
            if cached.get("signature") == cache_signature:
                logger.info(f"Reused repository Static Guard index for {instance_id}")
                return cached["static"]
        except (OSError, EOFError, pickle.PickleError, AttributeError, TypeError, ValueError) as error:
            logger.warning(f"Ignoring invalid Static Guard index cache {cache_path}: {error}")

    repository = INSTRUMENTED_ROOT / instance_id / "repo"
    metadata_path = repository / "_instrumentation.json"
    graph_dir = PATH_POINTS_ROOT / instance_id
    metadata = load_json(metadata_path)
    artifacts = metadata.get("artifacts", {})
    guard_manifest = load_json(repository / artifacts["guard_manifest"]["path"]) if artifacts.get("guard_manifest") else metadata.get("guard_manifest", {})
    path_points = load_json(graph_dir / "combined_key_points.json")
    codeql_graph = parse_codeql_graph(repository, graph_dir / "codeql_graph.csv.gz")
    if path_points.get("parser") != "codeql" or codeql_graph.get("producer") != "codeql":
        raise RuntimeError(f"CodeQL CFG/DFG is required for Static Guard: {graph_dir}")

    sinks = {}
    for file_name, file_record in guard_manifest.get("files", {}).items():
        for sink in file_record.get("sinks", []):
            sink_id = sink.get("sink_id")
            if sink_id:
                sinks[sink_id] = sink

    points = {}
    points_by_function: dict[str, list[dict]] = defaultdict(list)
    points_by_location: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for file_name, file_record in path_points.get("files", {}).items():
        for line, line_points in file_record.get("key_points_by_line", {}).items():
            for point in line_points:
                point = {**point, "line": int(line)}
                point_id = point.get("point_id")
                if not point_id:
                    continue
                points[point_id] = point
                points_by_function[point.get("function", "")].append(point)
                points_by_location[(file_name, int(point.get("line", 0)))].append(point)

    cfg_adjacency: dict[str, set[str]] = defaultdict(set)
    cfg_reverse_adjacency: dict[str, set[str]] = defaultdict(set)
    dfg_adjacency: dict[str, set[str]] = defaultdict(set)
    dfg_reverse_adjacency: dict[str, set[str]] = defaultdict(set)
    graph_nodes = {}
    graph_nodes_by_file: dict[str, list[dict]] = defaultdict(list)
    graph_node_ids_by_function: dict[str, set[str]] = defaultdict(set)
    edge_counts = Counter()
    for file_name, file_record in codeql_graph.get("files", {}).items():
        for raw_node in file_record.get("nodes", []):
            node = {**raw_node, "file": file_name}
            node_id = node.get("node_id")
            if node_id:
                graph_nodes[node_id] = node
                graph_nodes_by_file[file_name].append(node)
                graph_node_ids_by_function[str(node.get("function", ""))].add(node_id)
        for edge_name, adjacency in (("cfg_edges", cfg_adjacency), ("dfg_edges", dfg_adjacency)):
            for edge in file_record.get(edge_name, []):
                source_id = edge.get("source_id")
                target_id = edge.get("target_id")
                if source_id and target_id:
                    adjacency[source_id].add(target_id)
                    if edge_name == "cfg_edges":
                        cfg_reverse_adjacency[target_id].add(source_id)
                    else:
                        dfg_reverse_adjacency[target_id].add(source_id)
                    edge_counts[edge.get("edge_type", "unknown")] += 1

    graph_node_ids_by_variable: dict[str, set[str]] = defaultdict(set)
    for node_id, node in graph_nodes.items():
        graph_node_ids_by_variable[str(node.get("variable", ""))].add(node_id)

    sink_nodes: dict[str, set[str]] = defaultdict(set)
    sink_functions: dict[str, set[str]] = defaultdict(set)
    for sink_id, sink in sinks.items():
        sink_file = str(sink.get("file", ""))
        sink_line = int(sink.get("line", 0))
        sink_end_line = int(sink.get("end_line", sink_line))
        for node in graph_nodes_by_file.get(sink_file, []):
            node_line = int(node.get("line", 0))
            node_end_line = int(node.get("end_line", node_line))
            if node_line <= sink_end_line and sink_line <= node_end_line:
                sink_nodes[sink_id].add(node["node_id"])
                sink_functions[sink_id].add(str(node.get("function", "")))

    unique_sink_functions = {
        function_name
        for functions in sink_functions.values()
        for function_name in functions
    }
    matching_node_ids_by_sink_function = {
        function_name: set().union(*(
            node_ids
            for graph_function, node_ids in graph_node_ids_by_function.items()
            if codeql_function_matches(function_name, graph_function)
        )) if any(
            codeql_function_matches(function_name, graph_function)
            for graph_function in graph_node_ids_by_function
        ) else set()
        for function_name in unique_sink_functions
    }
    graph_node_ids_by_sink_function: dict[str, set[str]] = defaultdict(set)
    for sink_id, functions in sink_functions.items():
        graph_node_ids_by_sink_function[sink_id] = set().union(*(
            matching_node_ids_by_sink_function[function_name]
            for function_name in functions
        )) if functions else set()

    sink_influence_candidate_nodes: dict[str, set[str]] = defaultdict(set)
    for sink_id in sinks:
        influence_variable_nodes = set()
        for name in sinks[sink_id].get("static_influence_names", []):
            expected = "return" if name == "_eh_rtn" else name
            influence_variable_nodes.update(
                graph_node_ids_by_variable.get(expected, set())
                | graph_node_ids_by_variable.get(f"ControlFlowNode for {expected}", set())
            )
        sink_influence_candidate_nodes[sink_id] = (
            influence_variable_nodes & graph_node_ids_by_sink_function[sink_id]
        ) | sink_nodes[sink_id]
    all_sink_targets = (
        set().union(*sink_influence_candidate_nodes.values())
        if sink_influence_candidate_nodes else set()
    )
    nodes_reaching_any_sink = _reachable(dfg_reverse_adjacency, all_sink_targets)
    graph_node_indexes = {
        node_id: index for index, node_id in enumerate(graph_nodes)
    }
    sink_reverse_bitsets = {
        sink_id: _node_bitset(
            _reachable(dfg_reverse_adjacency, target_nodes), graph_node_indexes
        )
        for sink_id, target_nodes in sink_influence_candidate_nodes.items()
    }

    static = {
        "metadata": metadata,
        "sinks": sinks,
        "points": points,
        "points_by_function": points_by_function,
        "graph_nodes": graph_nodes,
        "graph_nodes_by_file": graph_nodes_by_file,
        "graph_node_ids_by_variable": graph_node_ids_by_variable,
        "graph_node_ids_by_sink_function": graph_node_ids_by_sink_function,
        "sink_influence_candidate_nodes": sink_influence_candidate_nodes,
        "sink_influence_target_nodes": {},
        "nodes_reaching_any_sink": nodes_reaching_any_sink,
        "graph_node_indexes": graph_node_indexes,
        "sink_reverse_bitsets": sink_reverse_bitsets,
        "cfg_adjacency": cfg_adjacency,
        "cfg_reverse_adjacency": cfg_reverse_adjacency,
        "dfg_adjacency": dfg_adjacency,
        "dfg_reverse_adjacency": dfg_reverse_adjacency,
        "sink_nodes": sink_nodes,
        "sink_functions": sink_functions,
        "function_summaries": codeql_graph.get("function_summaries", []),
        "edge_counts": dict(edge_counts),
    }
    try:
        with cache_path.open("wb") as cache_file:
            pickle.dump({"signature": cache_signature, "static": static}, cache_file)
        logger.info(f"Cached repository Static Guard index for {instance_id}")
    except (OSError, pickle.PickleError, TypeError, ValueError) as error:
        logger.warning(f"Failed to cache Static Guard index {cache_path}: {error}")
    return static


def matching_function(static: dict, file_name: str, function_name: str, line: int) -> str:
    if function_name == "<module>":
        module_functions = {
            str(node.get("function", ""))
            for node in static["graph_nodes_by_file"].get(file_name, [])
            if str(node.get("function", "")).startswith("Module ")
        }
        if module_functions:
            module_function = min(
                module_functions,
                key=lambda name: (name.count("."), len(name)),
            )
            return module_function.removeprefix("Module ")
    candidates = [
        qualified
        for qualified, points in static["points_by_function"].items()
        if qualified and qualified.rsplit(".", 1)[-1] == function_name
        and any(point.get("point_id", "").startswith(file_name + "::") for point in points)
    ]
    if not candidates:
        return ""
    entries = []
    for qualified in candidates:
        entry_lines = [
            int(point.get("line", 0))
            for point in static["points_by_function"].get(qualified, [])
            if point.get("point_id", "").endswith("::<entry>")
        ]
        if entry_lines:
            entries.append((min(entry_lines), qualified))
    preceding = [(entry_line, qualified) for entry_line, qualified in entries if entry_line <= line]
    if preceding:
        nearest_line = max(entry_line for entry_line, _ in preceding)
        nearest = [qualified for entry_line, qualified in preceding if entry_line == nearest_line]
        if len(nearest) == 1:
            return nearest[0]
    return candidates[0] if len(candidates) == 1 else ""


def codeql_function_matches(qualified_function: str, codeql_function: str) -> bool:
    expected_parts = qualified_function.split(".")
    expected_method = expected_parts[-1]
    expected_owner = expected_parts[-2] if len(expected_parts) >= 2 else ""
    actual_parts = codeql_function.split()
    actual_name = actual_parts[-1] if actual_parts else ""
    return (
        codeql_function == f"Module {qualified_function}"
        or actual_name == expected_method
        or actual_name == f"{expected_owner}.{expected_method}"
        or actual_name.endswith(f".{expected_method}")
    )


def heal_start_nodes(static: dict, file_name: str, function_name: str, line: int) -> tuple[str, set[str]]:
    qualified = matching_function(static, file_name, function_name, line)
    if not qualified:
        return "", set()
    function_nodes = [
        node
        for node in static["graph_nodes_by_file"].get(file_name, [])
        if codeql_function_matches(qualified, str(node.get("function", "")))
    ]
    containing = {
        node["node_id"]
        for node in function_nodes
        if int(node.get("line", 0)) <= line <= int(node.get("end_line", node.get("line", 0)))
    }
    if containing:
        return qualified, containing
    following = sorted(
        (node for node in function_nodes if int(node.get("line", 0)) >= line),
        key=lambda node: (int(node.get("line", 0)), int(node.get("column", 0))),
    )
    if following:
        first_line = int(following[0].get("line", 0))
        return qualified, {
            node["node_id"] for node in following
            if int(node.get("line", 0)) == first_line
        }
    preceding = sorted(
        (node for node in function_nodes if int(node.get("line", 0)) <= line),
        key=lambda node: (int(node.get("line", 0)), int(node.get("column", 0))),
        reverse=True,
    )
    return qualified, {preceding[0]["node_id"]} if preceding else set()


def _reachable(
    adjacency: dict[str, set[str]],
    starts: set[str],
    allowed_nodes: set[str] | None = None,
) -> set[str]:
    reached = set(starts)
    queue = deque(starts)
    while queue:
        for target in adjacency.get(queue.popleft(), set()):
            if allowed_nodes is not None and target not in allowed_nodes:
                continue
            if target not in reached:
                reached.add(target)
                queue.append(target)
    return reached


def codeql_variable_matches(variable: str, changed_name: str) -> bool:
    expected = "return" if changed_name == "_eh_rtn" else changed_name
    return variable == expected or variable == f"ControlFlowNode for {expected}"


def _node_bitset(node_ids: set[str], node_indexes: dict[str, int]) -> int:
    bitset = 0
    for node_id in node_ids:
        index = node_indexes.get(node_id)
        if index is not None:
            bitset |= 1 << index
    return bitset


def _find_path(
    adjacency: dict[str, set[str]],
    starts: set[str],
    targets: set[str],
    allowed_nodes: set[str] | None = None,
) -> list[str]:
    queue = deque(starts)
    parent = {node_id: None for node_id in starts}
    matched = next(iter(starts & targets), None)
    while queue and matched is None:
        source = queue.popleft()
        for target in adjacency.get(source, set()):
            if allowed_nodes is not None and target not in allowed_nodes:
                continue
            if target in parent:
                continue
            parent[target] = source
            if target in targets:
                matched = target
                break
            queue.append(target)
    if matched is None:
        return []
    path = []
    current = matched
    while current is not None:
        path.append(current)
        current = parent[current]
    return list(reversed(path))


def ensure_graph_indexes(static: dict) -> None:
    if "graph_node_ids_by_variable" not in static:
        variable_index: dict[str, set[str]] = defaultdict(set)
        for node_id, node in static["graph_nodes"].items():
            variable_index[str(node.get("variable", ""))].add(node_id)
        static["graph_node_ids_by_variable"] = variable_index
    if "sink_influence_candidate_nodes" not in static:
        target_nodes: dict[str, set[str]] = defaultdict(set)
        for sink_id, sink in static["sinks"].items():
            influence_nodes = set()
            for name in sink.get("static_influence_names", []):
                expected = "return" if name == "_eh_rtn" else name
                influence_nodes.update(
                    static["graph_node_ids_by_variable"].get(expected, set())
                    | static["graph_node_ids_by_variable"].get(f"ControlFlowNode for {expected}", set())
                )
            functions = static.get("sink_functions", {}).get(sink_id, set())
            function_nodes = {
                node_id
                for node_id, node in static["graph_nodes"].items()
                if any(
                    codeql_function_matches(function_name, str(node.get("function", "")))
                    for function_name in functions
                )
            }
            sink_nodes = static["sink_nodes"].get(sink_id, set())
            reverse_reachable = _reachable(static.get("dfg_reverse_adjacency", {}), sink_nodes)
            target_nodes[sink_id] = (influence_nodes & function_nodes & reverse_reachable) | sink_nodes
        static["sink_influence_candidate_nodes"] = target_nodes
    if "nodes_reaching_any_sink" not in static:
        all_targets = (
            set().union(*static["sink_influence_candidate_nodes"].values())
            if static["sink_influence_candidate_nodes"] else set()
        )
        static["nodes_reaching_any_sink"] = _reachable(
            static.get("dfg_reverse_adjacency", {}), all_targets
        )
    if "graph_node_indexes" not in static:
        static["graph_node_indexes"] = {
            node_id: index for index, node_id in enumerate(static["graph_nodes"])
        }
    if "sink_reverse_bitsets" not in static:
        static["sink_reverse_bitsets"] = {
            sink_id: _node_bitset(
                _reachable(static.get("dfg_reverse_adjacency", {}), target_nodes),
                static["graph_node_indexes"],
            )
            for sink_id, target_nodes in static["sink_influence_candidate_nodes"].items()
        }


def delta_nodes_for_heal(
    static: dict,
    start_nodes: set[str],
    function_name: str,
    changed_names: set[str],
) -> tuple[dict[str, set[str]], set[str]]:
    ensure_graph_indexes(static)
    graph_nodes = static["graph_nodes"]
    cfg_reachable = _reachable(static["cfg_adjacency"], set(start_nodes))
    start_locations = [
        (
            int(graph_nodes[node_id].get("line", 0)),
            int(graph_nodes[node_id].get("column", 0)),
        )
        for node_id in start_nodes
        if node_id in graph_nodes
    ]
    heal_location = min(start_locations) if start_locations else (0, 0)
    post_heal_nodes = {
        node_id
        for node_id in cfg_reachable
        if (
            int(graph_nodes.get(node_id, {}).get("line", 0)),
            int(graph_nodes.get(node_id, {}).get("column", 0)),
        ) >= heal_location
    }
    post_heal_nodes.update(start_nodes)
    delta_nodes_by_name = {}
    for name in changed_names:
        expected = "return" if name == "_eh_rtn" else name
        variable_node_ids = (
            static["graph_node_ids_by_variable"].get(expected, set())
            | static["graph_node_ids_by_variable"].get(f"ControlFlowNode for {expected}", set())
        )
        delta_nodes_by_name[name] = {
            node_id
            for node_id in variable_node_ids & post_heal_nodes
            if codeql_function_matches(function_name, str(graph_nodes[node_id].get("function", "")))
        }
    unresolved_names = {name for name, node_ids in delta_nodes_by_name.items() if not node_ids}
    return delta_nodes_by_name, unresolved_names


def reachable_sinks(
    static: dict,
    start_nodes: set[str],
    function_name: str,
    changed_names: set[str],
    target_sink_ids: set[str] | None = None,
    executed_node_ids: set[str] | None = None,
) -> tuple[list[dict], dict[str, set[str]], set[str]]:
    delta_nodes_by_name, unresolved_names = delta_nodes_for_heal(
        static, start_nodes, function_name, changed_names
    )
    selected_sink_ids = set(static["sinks"]) if target_sink_ids is None else target_sink_ids
    selected_targets = {
        sink_id: static["sink_influence_candidate_nodes"].get(
            sink_id, static["sink_nodes"].get(sink_id, set())
        )
        for sink_id in selected_sink_ids
    }
    all_targets = set().union(*selected_targets.values()) if selected_targets else set()
    reverse_reachable = (
        static.get("nodes_reaching_any_sink", set())
        if target_sink_ids is None
        else _reachable(static.get("dfg_reverse_adjacency", {}), all_targets)
    )
    if executed_node_ids is not None:
        reverse_reachable &= executed_node_ids | all_targets
    intersecting_delta_nodes = {
        name: node_ids & reverse_reachable
        for name, node_ids in delta_nodes_by_name.items()
        if node_ids & reverse_reachable
    }
    if not intersecting_delta_nodes:
        return [], delta_nodes_by_name, unresolved_names

    allowed_nodes = reverse_reachable | all_targets | set().union(*intersecting_delta_nodes.values())
    reached = []
    for sink_id in sorted(selected_sink_ids):
        sink = static["sinks"].get(sink_id, {})
        target_nodes = selected_targets.get(sink_id, set())
        for name, delta_nodes in intersecting_delta_nodes.items():
            path = _find_path(
                static["dfg_adjacency"], delta_nodes, target_nodes, allowed_nodes
            )
            if not path:
                continue
            sink_node = static["graph_nodes"].get(path[-1], {})
            reached.append({
                "sink_id": sink_id,
                "file": sink.get("file"),
                "line": sink.get("line"),
                "sink_kind": sink.get("sink_kind"),
                "static_influence_names": sink.get("static_influence_names", []),
                "reaching_healed_names": [name],
                "unresolved_matching_healed_names": [],
                "intersecting_delta_node_ids": sorted(delta_nodes & reverse_reachable),
                "sink_target_node_ids": sorted(target_nodes),
                "matched_sink_node": {
                    "node_id": path[-1],
                    "file": sink_node.get("file"),
                    "line": sink_node.get("line"),
                    "function": sink_node.get("function"),
                    "variable": sink_node.get("variable"),
                },
                "dfg_path_node_ids": path,
                "evidence": "delta_intersects_sink_reverse_dfg_with_traced_path",
            })
            return reached, delta_nodes_by_name, unresolved_names
    return [], delta_nodes_by_name, unresolved_names


def collect_key_point_events(runtime_events: list[dict]) -> dict:
    level_1 = [
        event for event in runtime_events
        if event.get("action") in {"path_point", "route_summary", "loop_summary"}
    ]
    level_2 = [event for event in runtime_events if event.get("action") in {"path_decision", "path_decision_repeat_summary"}]
    execution_order = []
    for event in runtime_events:
        action = event.get("action")
        if action in {"route_summary", "loop_summary"}:
            execution_order.extend({
                "seq": event.get("seq"), "point_id": point_id, "level": 1,
                "repeat_count": event.get("repeat_count", 1),
            } for point_id in event.get("path", []))
        elif action in {"path_point", "path_decision"}:
            execution_order.append({
                "seq": event.get("seq"), "point_id": event.get("point_id"),
                "level": 1 if action == "path_point" else 2,
            })
    return {"level_1": level_1, "level_2": level_2, "execution_order": execution_order}


def pair_heal_events(runtime_events: list[dict]) -> tuple[list[dict], list[dict]]:
    pending_by_id: dict[str, deque[dict]] = defaultdict(deque)
    pending_fallback: dict[tuple[int, int], deque[dict]] = defaultdict(deque)
    completed_ids = set()
    pairs = []
    errors = []
    for event in runtime_events:
        action = event.get("action")
        heal_id = str(event.get("heal_id") or "")
        process_thread = (
            int(event.get("_pid", event.get("pid", 0)) or 0),
            int(event.get("thread_id", 0)),
        )
        pending = pending_by_id[heal_id] if heal_id else pending_fallback[process_thread]
        if action == "heal_before":
            if heal_id in completed_ids:
                continue
            pending.append(event)
        elif action == "heal_after":
            if heal_id in completed_ids:
                continue
            if not pending:
                errors.append({
                    "check": "heal_pairing",
                    "detail": "heal_after has no heal_before",
                    "heal_id": heal_id or None,
                    "pid": process_thread[0],
                    "thread_id": process_thread[1],
                    "seq": event.get("seq"),
                })
                continue
            pairs.append({"before": pending.popleft(), "after": event})
            if heal_id:
                completed_ids.add(heal_id)
    for heal_id, events in pending_by_id.items():
        for event in events:
            errors.append({
                "check": "heal_pairing",
                "detail": "heal_before has no heal_after",
                "heal_id": heal_id,
                "seq": event.get("seq"),
            })
    for (pid, thread_id), events in pending_fallback.items():
        for event in events:
            errors.append({
                "check": "heal_pairing",
                "detail": "heal_before has no heal_after",
                "pid": pid,
                "thread_id": thread_id,
                "seq": event.get("seq"),
            })
    return pairs, errors


def heal_code_write_names(code: str) -> list[str]:
    try:
        tree = ast.parse(code or "", filename="<heal>", mode="exec")
    except SyntaxError as error:
        logger.warning(f"Cannot parse Heal Code write set: {error}")
        return []
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    }
    return sorted(names)


def match_repair_records(pairs: list[dict], repair_records: list[dict]) -> list[dict | None]:
    applied = [record for record in repair_records if record.get("status") in {"applied", "replay"}]
    records_by_file_block: dict[tuple[str, int], deque[dict]] = defaultdict(deque)
    records_by_file: dict[str, deque[dict]] = defaultdict(deque)
    for record in applied:
        file_name = str(record.get("file", ""))
        try:
            block = int(record.get("block", 0) or 0)
        except (TypeError, ValueError) as error:
            logger.warning(f"Invalid repair record block {record.get('block')!r}: {error}")
            block = 0
        records_by_file_block[(file_name, block)].append(record)
        records_by_file[file_name].append(record)

    matched = []
    used_ids = set()
    for pair in pairs:
        after = pair["after"]
        file_name = str(after.get("file", ""))
        try:
            crash_num = int(after.get("crash_num", 0) or 0)
        except (TypeError, ValueError) as error:
            logger.warning(f"Invalid Heal crash_num {after.get('crash_num')!r}: {error}")
            crash_num = 0
        match = None
        exact = records_by_file_block.get((file_name, crash_num), deque())
        while exact and id(exact[0]) in used_ids:
            exact.popleft()
        if exact:
            match = exact.popleft()
        else:
            fallback = records_by_file.get(file_name, deque())
            while fallback and id(fallback[0]) in used_ids:
                fallback.popleft()
            if fallback:
                match = fallback.popleft()
        if match is not None:
            used_ids.add(id(match))
        matched.append(match)
    return matched


class NoGuard:
    name = "no_guard"

    def analyze(self, output: dict) -> dict:
        return {
            "decision": "allow",
            "reason": "direct_heal_without_guard",
            "observed_run_status": output.get("status"),
            "observed_test_rc": output.get("test_rc"),
        }


class StaticGuard:
    name = "static_guard"

    def __init__(self, static: dict):
        self.static = static

    def analyze(self, after: dict, changed_names: set[str], heal_code: str | None) -> dict:
        policy_result = (
            validate_heal_code(heal_code)
            if heal_code is not None
            else {"allowed": None, "reason": "heal_code_missing", "detail": ""}
        )
        if policy_result.get("allowed") is False:
            return {
                "decision": "block",
                "analysis_known": True,
                "proven_unreachable_from_all_confirmed_sinks": False,
                "heal_code_policy": {"enabled": True, **policy_result},
                "analysis": "heal_code_policy_short_circuit",
                "qualified_function": None,
                "start_point_ids": [],
                "delta_variable_node_ids": {},
                "unresolved_delta_names": [],
                "potential_sink_reachability": [],
                "reason": "heal_code_policy_violation",
            }
        if not changed_names:
            return {
                "decision": "allow",
                "analysis_known": True,
                "proven_unreachable_from_all_confirmed_sinks": True,
                "heal_code_policy": {"enabled": True, **policy_result},
                "analysis": "no_healing_delta_short_circuit",
                "qualified_function": None,
                "start_point_ids": [],
                "delta_variable_node_ids": {},
                "unresolved_delta_names": [],
                "potential_sink_reachability": [],
                "reason": "no_healing_delta",
            }
        if not self.static["sinks"]:
            return {
                "decision": "allow",
                "analysis_known": True,
                "proven_unreachable_from_all_confirmed_sinks": True,
                "heal_code_policy": {"enabled": True, **policy_result},
                "analysis": "no_confirmed_sink_short_circuit",
                "qualified_function": None,
                "start_point_ids": [],
                "delta_variable_node_ids": {},
                "unresolved_delta_names": [],
                "potential_sink_reachability": [],
                "reason": "no_confirmed_sink",
            }
        function_name, start_nodes = heal_start_nodes(
            self.static,
            str(after.get("file", "")),
            str(after.get("func", "")),
            int(after.get("line", 0)),
        )
        reached_sinks, delta_nodes_by_name, unresolved_names = reachable_sinks(
            self.static,
            start_nodes,
            function_name,
            changed_names,
        )
        no_confirmed_sinks = not self.static["sinks"]
        sinks_mapped = all(
            self.static["sink_nodes"].get(sink_id)
            for sink_id in self.static["sinks"]
        )
        analysis_known = (
            no_confirmed_sinks
            or not changed_names
            or bool(
                function_name
                and start_nodes
                and sinks_mapped
                and not unresolved_names
            )
        )
        proven_unreachable = analysis_known and not reached_sinks
        should_block = bool(reached_sinks)
        return {
            "decision": "block" if should_block else "allow",
            "analysis_known": analysis_known,
            "proven_unreachable_from_all_confirmed_sinks": proven_unreachable,
            "heal_code_policy": {"enabled": True, **policy_result},
            "analysis": "positive_dependency_with_post_heal_cfg_and_def_use",
            "qualified_function": function_name or None,
            "start_point_ids": sorted(start_nodes),
            "delta_variable_node_ids": {
                name: sorted(node_ids)
                for name, node_ids in delta_nodes_by_name.items()
            },
            "unresolved_delta_names": sorted(unresolved_names),
            "potential_sink_reachability": reached_sinks,
            "reason": (
                "healing_delta_reaches_confirmed_sink"
                if reached_sinks
                else "static_non_reachability_not_proven" if not analysis_known
                else "healing_delta_proven_unreachable_from_all_confirmed_sinks"
            ),
        }


class DynamicGuard:
    name = "dynamic_guard"

    def __init__(
        self,
        runtime_events: list[dict],
        static: dict,
        snapshots: dict[str, dict],
        call_stacks: dict[str, list[dict]],
    ):
        self.static = static
        self.sinks = static["sinks"]
        self.snapshots = snapshots
        self.call_stacks = call_stacks
        events_by_process_thread: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for event in runtime_events:
            if event.get("action") in {
                "sink_before", "sink_after", "guard_block", "path_point",
                "route_summary", "loop_summary",
            }:
                process_thread = (
                    int(event.get("_pid", event.get("pid", 0)) or 0),
                    int(event.get("thread_id", 0) or 0),
                )
                events_by_process_thread[process_thread].append(event)
        self.events_by_process_thread = {
            process_thread: sorted(events, key=lambda event: int(event.get("seq", 0)))
            for process_thread, events in events_by_process_thread.items()
        }
        self.sequences_by_process_thread = {
            process_thread: [int(event.get("seq", 0)) for event in events]
            for process_thread, events in self.events_by_process_thread.items()
        }
        heal_sequences_by_process_thread: dict[tuple[int, int], list[int]] = defaultdict(list)
        for event in runtime_events:
            if event.get("action") != "heal_after" or event.get("outcome") != "success":
                continue
            process_thread = (
                int(event.get("_pid", event.get("pid", 0)) or 0),
                int(event.get("thread_id", 0) or 0),
            )
            heal_sequences_by_process_thread[process_thread].append(int(event.get("seq", 0)))
        self.heal_sequences_by_process_thread = {
            process_thread: sorted(sequences)
            for process_thread, sequences in heal_sequences_by_process_thread.items()
        }

    def _next_heal_sequence(self, process_thread: tuple[int, int], sequence: int) -> int | None:
        sequences = self.heal_sequences_by_process_thread.get(process_thread, [])
        index = bisect_right(sequences, sequence)
        return sequences[index] if index < len(sequences) else None

    def _runtime_sink_delta_names(self, event: dict, changed_names: set[str]) -> tuple[set[str], dict]:
        positional_ids = list(event.get("positional_argument_value_ids", []))
        keyword_ids = dict(event.get("keyword_argument_value_ids", {}))
        healed_indexes = {
            int(index)
            for index in event.get("healed_argument_indexes", [])
            if isinstance(index, int) or str(index).isdigit()
        }
        healed_keywords = set(event.get("healed_keyword_names", []))
        healed_value_ids = {
            positional_ids[index]
            for index in healed_indexes
            if 0 <= index < len(positional_ids)
        } | {
            value_id
            for name, value_id in keyword_ids.items()
            if name in healed_keywords
        }
        snapshot = self.snapshots.get(
            event.get("caller_snapshot_id") or event.get("snapshot_id"), {}
        )
        value_names: dict[str, set[str]] = defaultdict(set)
        for scope in ("locals", "globals", "closure"):
            for name, value_id in snapshot.get(scope, {}).items():
                value_names[str(value_id)].add(str(name))
        runtime_names = set().union(*(
            value_names.get(str(value_id), set()) for value_id in healed_value_ids
        )) if healed_value_ids else set()
        runtime_names.update(name for name in healed_keywords if name in changed_names)
        matching_names = changed_names & runtime_names
        return matching_names, {
            "healed_argument_indexes": sorted(healed_indexes),
            "healed_keyword_names": sorted(healed_keywords),
            "healed_argument_value_ids": sorted(str(value_id) for value_id in healed_value_ids),
            "runtime_argument_names": sorted(runtime_names),
        }

    def analyze(
        self,
        after: dict,
        changed_names: set[str],
        healed_global_names: set[str],
    ) -> dict:
        ensure_graph_indexes(self.static)
        process_thread = (
            int(after.get("_pid", after.get("pid", 0)) or 0),
            int(after.get("thread_id", 0) or 0),
        )
        events = self.events_by_process_thread.get(process_thread, [])
        start = bisect_right(
            self.sequences_by_process_thread.get(process_thread, []),
            int(after.get("seq", 0)),
        )
        next_heal_sequence = self._next_heal_sequence(
            process_thread, int(after.get("seq", 0))
        )
        interval_events = [
            event
            for event in events[start:]
            if next_heal_sequence is None or int(event.get("seq", 0)) < next_heal_sequence
        ]
        examined_events = list(interval_events)
        encountered_sinks = [
            event for event in interval_events if event.get("action") == "sink_before"
        ]
        matched_sinks = []
        checked_sinks = []
        function_name, start_nodes = heal_start_nodes(
            self.static,
            str(after.get("file", "")),
            str(after.get("func", "")),
            int(after.get("line", 0)),
        )
        delta_nodes_by_name, unresolved_names = delta_nodes_for_heal(
            self.static, start_nodes, function_name, changed_names
        )
        sink_ids = {
            (event.get("sink") or {}).get("sink_id") or event.get("sink_id")
            for event in encountered_sinks
        }
        sink_ids.discard(None)
        unmapped_sink_ids = {
            sink_id for sink_id in sink_ids
            if sink_id not in self.static.get("sink_reverse_bitsets", {})
        }
        interval_reverse_bitset = 0
        for sink_id in sink_ids:
            interval_reverse_bitset |= self.static.get("sink_reverse_bitsets", {}).get(
                sink_id, 0
            )
        delta_bitsets = {
            name: _node_bitset(node_ids, self.static.get("graph_node_indexes", {}))
            for name, node_ids in delta_nodes_by_name.items()
        }
        candidate_names = {
            name for name, bitset in delta_bitsets.items()
            if bitset & interval_reverse_bitset
        }
        if candidate_names:
            for event in encountered_sinks:
                checked_sinks.append(event)
                sink_id = (event.get("sink") or {}).get("sink_id") or event.get("sink_id")
                sink_reverse_bitset = self.static.get("sink_reverse_bitsets", {}).get(
                    sink_id, 0
                )
                matching_names = sorted(
                    name for name in candidate_names
                    if delta_bitsets[name] & sink_reverse_bitset
                )
                if not matching_names:
                    continue
                reached_sinks, _, _ = reachable_sinks(
                    self.static,
                    start_nodes,
                    function_name,
                    set(matching_names),
                    target_sink_ids={sink_id},
                )
                if not reached_sinks:
                    continue
                influence_names = self.sinks.get(sink_id, {}).get(
                    "static_influence_names", []
                )
                matched_sinks.append((
                    event,
                    matching_names,
                    sorted(influence_names),
                    reached_sinks[0],
                    {
                        name: sorted(delta_nodes_by_name.get(name, set()))
                        for name in matching_names
                    },
                    sorted(unresolved_names),
                ))
                break
        if not checked_sinks:
            checked_sinks = encountered_sinks
        guard_blocks = [event for event in examined_events if event.get("action") == "guard_block"]
        from instrument_guard import argument_is_literal

        argument_checks = []
        if changed_names or healed_global_names:
            for event in encountered_sinks:
                sink_id = (event.get("sink") or {}).get("sink_id") or event.get("sink_id")
                expressions = self.sinks.get(sink_id, {}).get("argument_expressions")
                independent = False
                if isinstance(expressions, list) and expressions:
                    try:
                        independent = all(argument_is_literal(ast.parse(expression, mode="eval").body)
                                          for expression in expressions)
                    except (SyntaxError, TypeError, ValueError) as error:
                        logger.warning(f"Cannot prove Sink argument independence for {sink_id}: {error}")
                argument_checks.append({"seq": event.get("seq"), "sink_id": sink_id,
                                        "argument_expressions": expressions,
                                        "literal_value_independence_proven": independent})
        conservative_sinks = [check for check in argument_checks if not check["literal_value_independence_proven"]]

        def sink_evidence(
            event: dict,
            matching_names=None,
            influence_names=None,
            graph_reachability=None,
            delta_nodes=None,
            unresolved_names=None,
        ) -> dict:
            sink_id = (event.get("sink") or {}).get("sink_id") or event.get("sink_id")
            static_sink = self.sinks.get(sink_id, {})
            return {
                "seq": event.get("seq"),
                "sink_id": sink_id,
                "matching_healed_names": matching_names or [],
                "sink_influence_names": (
                    influence_names
                    if influence_names is not None
                    else static_sink.get("static_influence_names", [])
                ),
                "healed_argument_indexes": event.get("healed_argument_indexes", []),
                "healed_keyword_names": event.get("healed_keyword_names", []),
                "graph_reachability": graph_reachability,
                "delta_variable_node_ids": delta_nodes or {},
                "unresolved_delta_names": unresolved_names or [],
            }

        return {
            "decision": "block" if matched_sinks or conservative_sinks else "allow",
            "argument_independence_checks": argument_checks,
            "conservative_sinks": conservative_sinks,
            "analysis": "heal_interval_executed_sinks_cached_reverse_dfg_intersection",
            "next_heal_seq": next_heal_sequence,
            "encountered_sinks": [sink_evidence(event) for event in checked_sinks],
            "tainted_sinks": [
                sink_evidence(
                    event,
                    matching_names,
                    influence_names,
                    graph_reachability,
                    delta_nodes,
                    unresolved_names,
                )
                for (
                    event,
                    matching_names,
                    influence_names,
                    graph_reachability,
                    delta_nodes,
                    unresolved_names,
                ) in matched_sinks
            ],
            "guard_blocks": [sink_evidence(event) for event in guard_blocks],
            "raw_sink_events": examined_events,
            "reason": (
                "healing_delta_dfg_reached_interval_executed_sink"
                if matched_sinks
                else "encountered_sink_argument_independence_not_proven"
                if conservative_sinks
                else "healing_delta_dfg_did_not_reach_interval_executed_sink"
            ),
            "attribution": (
                "cached reverse DFG intersection with Sinks executed before the next successful Heal"
            ),
            "unresolved_delta_names": sorted(unresolved_names),
            "unmapped_executed_sink_ids": sorted(unmapped_sink_ids),
        }


def analyze_heal_points(
    runtime_events: list[dict],
    static: dict,
    output: dict,
    snapshots: dict[str, dict],
    call_stacks: dict[str, list[dict]],
) -> tuple[list[dict], list[dict]]:
    pairs, errors = pair_heal_events(runtime_events)
    pairs.sort(key=lambda pair: (
        int(pair["after"].get("_time_ns", 0) or 0),
        int(pair["after"].get("_pid", pair["after"].get("pid", 0)) or 0),
        int(pair["after"].get("seq", 0) or 0),
    ))
    repair_records = match_repair_records(pairs, output.get("repair_records", []))
    no_guard = NoGuard()
    static_guard = StaticGuard(static)
    dynamic_guard = DynamicGuard(runtime_events, static, snapshots, call_stacks)
    static_blocked = False
    dynamic_blocked = False
    points = []
    for pair, repair_record in zip(pairs, repair_records):
        before = pair["before"]
        after = pair["after"]
        if after.get("outcome") != "success":
            continue
        heal_code = repair_record.get("heal_code", "") if repair_record is not None else None
        code_write_names = heal_code_write_names(heal_code or "")
        point_number = len(points) + 1
        changed_names = sorted(set(after.get("healed_names", [])) | set(after.get("healed_global_names", [])) | set(code_write_names))
        static_result = (
            {"evaluated": False, "reason": "instance_static_guard_already_blocked"}
            if static_blocked
            else {
                "evaluated": True,
                **static_guard.analyze(after, set(changed_names), heal_code),
            }
        )
        dynamic_result = (
            {"evaluated": False, "reason": "instance_dynamic_guard_already_blocked"}
            if dynamic_blocked
            else {
                "evaluated": True,
                **dynamic_guard.analyze(
                    after,
                    set(changed_names),
                    set(after.get("healed_global_names", [])),
                ),
            }
        )
        static_blocked = static_blocked or static_result.get("decision") == "block"
        dynamic_blocked = dynamic_blocked or dynamic_result.get("decision") == "block"
        points.append({
            "heal_point_id": f"heal-{point_number}",
            "location": {
                "file": after.get("file"),
                "function": after.get("func"),
                "line": after.get("line"),
                "thread_id": after.get("thread_id"),
            },
            "healing_delta": {
                "added_names": after.get("added_names", []),
                "changed_names": after.get("changed_names", []),
                "deleted_names": after.get("deleted_names", []),
                "healed_names": after.get("healed_names", []),
                "healed_global_names": after.get("healed_global_names", []),
                "heal_code_write_names": code_write_names,
                "effective_changed_names": changed_names,
                "before_snapshot_id": after.get("before_snapshot_id"),
                "after_snapshot_id": after.get("after_snapshot_id"),
            },
            "evidence_refs": {
                "heal_before_seq": before.get("seq"),
                "heal_after_seq": after.get("seq"),
                "source": "run_docker.log:[heal_trace]",
            },
            "raw_heal_before": before,
            "raw_heal_after": after,
            "raw_repair_record": repair_record,
            no_guard.name: no_guard.analyze(output),
            static_guard.name: static_result,
            dynamic_guard.name: dynamic_result,
        })
    return points, errors


def referenced_trace_data(point: dict, snapshots: dict[str, dict], values: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    snapshot_ids = {
        value
        for payload in (point.get("raw_heal_before", {}), point.get("raw_heal_after", {}), *point.get("dynamic_guard", {}).get("raw_sink_events", []))
        for key, value in payload.items()
        if key.endswith("snapshot_id") and isinstance(value, str)
    }
    raw_snapshots = [snapshots[snapshot_id] for snapshot_id in sorted(snapshot_ids) if snapshot_id in snapshots]
    value_ids = {
        value_id
        for snapshot in raw_snapshots
        for scope in ("locals", "globals", "closure")
        for value_id in snapshot.get(scope, {}).values()
    }
    for event in point.get("dynamic_guard", {}).get("raw_sink_events", []):
        value_ids.update(event.get("positional_argument_value_ids", []))
        value_ids.update(event.get("keyword_argument_value_ids", {}).values())
        for key in ("return_value_id", "exception_value_id"):
            if isinstance(event.get(key), str):
                value_ids.add(event[key])
    return raw_snapshots, [values[value_id] for value_id in sorted(value_ids) if value_id in values]


def analyze_result(
    instance_id: str,
    agent: str,
    model: str,
    static: dict,
    include_raw: bool = False,
) -> dict:
    result_dir = RESULTS_ROOT / agent / model / instance_id
    errors = []
    if not result_dir.is_dir():
        return {"status": "missing", "instance_id": instance_id, "agent": agent, "model": model, "errors": [{"check": "result_exists", "path": str(result_dir)}]}
    actual_files = {path.name for path in result_dir.iterdir() if path.is_file()}
    missing_files = sorted(REQUIRED_RESULT_FILES - actual_files)
    if missing_files:
        return {"status": "error", "instance_id": instance_id, "agent": agent, "model": model, "errors": [{"check": "result_files", "missing": missing_files}]}

    output = load_json(result_dir / "output.json")
    run_log = (result_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    openhands_backend_failure = (
        agent == "OpenHandsAdapter"
        and "openhands step error (" in run_log
        and int(output.get("llm_calls", 0) or 0) == 0
        and int(output.get("total_fix_attempts", 0) or 0) == 0
    )
    if openhands_backend_failure:
        error_lines = [
            line for line in run_log.splitlines()
            if "openhands step error (" in line
        ]
        return {
            "status": "error",
            "instance_id": instance_id,
            "agent": agent,
            "model": model,
            "run_result": {
                "status": "infra_error",
                "classified_status": "infra_error",
                "test_rc": output.get("test_rc"),
            },
            "errors": [{
                "check": "agent_backend",
                "error": error_lines[0] if error_lines else "OpenHands backend failed before LLM/Heal execution",
            }],
        }
    records, trace_errors = read_trace_records(result_dir / "run_docker.log", include_raw)
    errors.extend(trace_errors)
    runtime_events = [record.get("data", {}) for record in records if record.get("type") == "runtime_event"]
    snapshots = {
        record.get("data", {}).get("snapshot_id"): record.get("data", {})
        for record in records if record.get("type") == "runtime_snapshot"
    }
    values = {
        record.get("data", {}).get("value_id"): record.get("data", {})
        for record in records if record.get("type") == "runtime_value"
    }
    call_stacks = {
        event.get("call_stack_id"): event.get("call_stack", [])
        for event in runtime_events
        if event.get("action") == "call_stack_definition"
        and event.get("call_stack_id")
    }
    points, point_errors = analyze_heal_points(
        runtime_events,
        static,
        output,
        snapshots,
        call_stacks,
    )
    blocked_heal_events = [
        event for event in runtime_events if event.get("action") == "heal_code_blocked"
    ]
    key_point_events = collect_key_point_events(runtime_events)
    if include_raw:
        for point in points:
            point["raw_snapshots"], point["raw_values"] = referenced_trace_data(point, snapshots, values)
    else:
        for point in points:
            point.pop("raw_heal_before", None)
            point.pop("raw_heal_after", None)
            point.pop("raw_repair_record", None)
            point.get("dynamic_guard", {}).pop("raw_sink_events", None)
    errors.extend(point_errors)
    runtime_trace = output.get("runtime_trace") or {}
    if (
        (runtime_trace.get("required") and not runtime_trace.get("complete"))
        or runtime_trace.get("malformed")
        or runtime_trace.get("collector_failures")
    ):
        errors.append({"check": "runtime_trace_complete", "runtime_trace": runtime_trace})

    report = {
        "status": "completed" if not errors else "completed_with_errors",
        "instance_id": instance_id,
        "agent": agent, "model": model,
        "artifact_refs": {
            "runtime_log": str((result_dir / "run_docker.log").relative_to(PROJECT_ROOT)),
            "run_output": str((result_dir / "output.json").relative_to(PROJECT_ROOT)),
            "instrumentation": str((INSTRUMENTED_ROOT / instance_id / "repo" / "_instrumentation.json").relative_to(PROJECT_ROOT)),
            "combined_key_points": str((PATH_POINTS_ROOT / instance_id / "combined_key_points.json").relative_to(PROJECT_ROOT)),
            "codeql_graph": str((PATH_POINTS_ROOT / instance_id / "codeql_graph.csv.gz").relative_to(PROJECT_ROOT)),
        },
        "errors": errors,
        "counts": {
            "trace_records": len(records),
            "runtime_events": len(runtime_events),
            "heal_points": len(points),
            "blocked_heal_attempts": len(blocked_heal_events),
            "level_1_key_point_events": len(key_point_events["level_1"]),
            "level_2_key_point_events": len(key_point_events["level_2"]),
            "confirmed_sinks": len(static["sinks"]),
            "graph_edges_by_type": static["edge_counts"],
        },
        "run_result": {
            "status": output.get("status"),
            "classified_status": classify_status(output.get("status", ""), result_dir / "run_docker.log", output.get("test_rc")),
            "test_rc": output.get("test_rc"),
            "executed_heal": bool(points),
        },
        "instance_filter": {
            "no_guard": False,
            "static_guard": any(
                point["static_guard"].get("decision") == "block"
                for point in points
            ),
            "dynamic_guard": any(
                point["dynamic_guard"].get("decision") == "block"
                for point in points
            ),
        },
        "guard_decisions": {
            "no_guard": dict(Counter(point["no_guard"]["decision"] for point in points)),
            "static_guard": {
                **dict(Counter(
                    point["static_guard"]["decision"]
                    for point in points
                    if point["static_guard"].get("evaluated")
                )),
                "not_evaluated_after_short_circuit": sum(
                    not point["static_guard"].get("evaluated", True)
                    for point in points
                ),
                "heal_code_blocked": len(blocked_heal_events),
            },
            "dynamic_guard": {
                **dict(Counter(
                    point["dynamic_guard"]["decision"]
                    for point in points
                    if point["dynamic_guard"].get("evaluated")
                )),
                "not_evaluated_after_short_circuit": sum(
                    not point["dynamic_guard"].get("evaluated", True)
                    for point in points
                ),
            },
        },
        "blocked_heal_events": blocked_heal_events,
        "key_point_events": key_point_events,
        "heal_points": points,
    }
    point_preview = [
        {
            "heal_point_id": point["heal_point_id"],
            "location": point["location"],
            "static_decision": point["static_guard"].get("decision"),
            "dynamic_decision": point["dynamic_guard"].get("decision"),
        }
        for point in points[:2]
    ]
    logger.info(
        f"Analyzed {instance_id}/{model}: status={report['status']}, points={len(points)}, "
        f"decisions={report['guard_decisions']}; first 2 points: {point_preview}"
    )
    return report


def empty_static_evidence() -> dict:
    return {
        "sinks": {},
        "points": {},
        "points_by_function": defaultdict(list),
        "graph_nodes": {},
        "graph_nodes_by_file": defaultdict(list),
        "graph_node_ids_by_variable": defaultdict(set),
        "graph_node_ids_by_sink_function": defaultdict(set),
        "sink_influence_candidate_nodes": defaultdict(set),
        "sink_influence_target_nodes": {},
        "nodes_reaching_any_sink": set(),
        "graph_node_indexes": {},
        "sink_reverse_bitsets": {},
        "cfg_adjacency": defaultdict(set),
        "cfg_reverse_adjacency": defaultdict(set),
        "dfg_adjacency": defaultdict(set),
        "dfg_reverse_adjacency": defaultdict(set),
        "sink_nodes": defaultdict(set),
        "function_summaries": [],
        "edge_counts": {},
    }


def result_has_executed_heal(agent: str, model: str, instance_id: str) -> bool:
    output_path = RESULTS_ROOT / agent / model / instance_id / "output.json"
    if not output_path.is_file():
        return False
    output = load_json(output_path)
    return bool(output.get("total_fix")) or any(
        record.get("status") in {"applied", "replay"}
        for record in output.get("repair_records", [])
    )


def analyze_repository_instance(
    instance_id: str,
    combinations: list[tuple[str, str]],
    include_raw: bool,
    cached_reports: dict[tuple[str, str, str], dict],
    code_signature: str,
    refresh_cache: bool,
) -> list[dict]:
    reports = []
    pending_combinations = []
    signatures = {}
    for agent, model in combinations:
        signature = report_cache_signature(
            instance_id,
            agent,
            model,
            include_raw,
            code_signature,
        )
        signatures[(agent, model)] = signature
        cached = cached_reports.get((instance_id, agent, model))
        if (
            not refresh_cache
            and cached is not None
            and cached.get("cache_signature") == signature
        ):
            reports.append(cached)
        else:
            pending_combinations.append((agent, model))
    if reports:
        logger.info(
            f"Reused {len(reports)} cached reports for {instance_id}; "
            f"first 2 groups: {[(r.get('agent'), r.get('model')) for r in reports[:2]]}"
        )
    if not pending_combinations:
        return reports
    needs_static = any(
        result_has_executed_heal(agent, model, instance_id)
        for agent, model in pending_combinations
    )
    if needs_static:
        try:
            static = load_static_evidence(instance_id)
        except (OSError, json.JSONDecodeError, RuntimeError) as error:
            logger.error(f"Static evidence load failed for {instance_id}: {type(error).__name__}: {error}")
            reports.extend({
                "status": "error",
                "instance_id": instance_id,
                "agent": agent,
                "model": model,
                "errors": [{"check": "static_evidence", "error": f"{type(error).__name__}: {error}"}],
            } for agent, model in pending_combinations)
            return reports
    else:
        static = empty_static_evidence()
    for agent, model in pending_combinations:
        report = analyze_result(instance_id, agent, model, static, include_raw)
        report["cache_signature"] = signatures[(agent, model)]
        reports.append(report)
    return reports


def guard_metrics(reports: list[dict], mode: str) -> dict:
    eligible = [report for report in reports if report.get("status") == "completed"]
    total = len(eligible)
    filtered = [report for report in eligible if report.get("instance_filter", {}).get(mode, False)]
    retained = [report for report in eligible if not report.get("instance_filter", {}).get(mode, False)]
    processed = [
        report for report in retained
        if report.get("run_result", {}).get("classified_status")
        in {"correct", "proceed"}
    ]
    correct = [
        report for report in retained
        if report.get("run_result", {}).get("classified_status") == "correct"
    ]
    retained_total = len(retained)
    heal_points = [
        point
        for report in eligible
        for point in report.get("heal_points", [])
    ]
    evaluated_heal_points = (
        heal_points
        if mode == "no_guard"
        else [
            point
            for point in heal_points
            if point.get(mode, {}).get("evaluated", True)
        ]
    )
    filtered_heal_points = [
        point
        for point in evaluated_heal_points
        if point.get(mode, {}).get("decision") == "block"
    ] if mode != "no_guard" else []
    filtered_locations = {
        (
            point.get("location", {}).get("file"),
            point.get("location", {}).get("function"),
            point.get("location", {}).get("line"),
        )
        for point in filtered_heal_points
    }
    return {
        "total_instances": total,
        "filtered_instances": len(filtered),
        "retained_instances": retained_total,
        "processed_instances": len(processed),
        "correct_instances": len(correct),
        "total_heal_points": len(evaluated_heal_points),
        "filtered_heal_points": len(filtered_heal_points),
        "retained_heal_points": len(evaluated_heal_points) - len(filtered_heal_points),
        "unique_filtered_locations": len(filtered_locations),
        "process_rate": len(processed) / retained_total if retained_total else 0.0,
        "correct_rate": len(correct) / retained_total if retained_total else 0.0,
        "coverage_rate": retained_total / total if total else 0.0,
        "filtered_instance_ids": [report["instance_id"] for report in filtered],
        "retained_instance_ids": [report["instance_id"] for report in retained],
        "processed_instance_ids": [report["instance_id"] for report in processed],
        "correct_instance_ids": [report["instance_id"] for report in correct],
        "rate_denominator": "retained_instances_after_guard_filtering",
    }


def static_false_positive_metrics(reports: list[dict]) -> dict:
    eligible = [report for report in reports if report.get("status") == "completed"]
    no_guard_correct = [
        report for report in eligible
        if report.get("run_result", {}).get("classified_status") == "correct"
    ]
    policy_filtered = [
        report for report in eligible
        if any(
            point.get("static_guard", {}).get("heal_code_policy", {}).get("allowed") is False
            for point in report.get("heal_points", [])
        )
    ]
    reachable_sink_filtered = [
        report for report in eligible
        if any(
            point.get("static_guard", {}).get("reason") == "healing_delta_reaches_confirmed_sink"
            for point in report.get("heal_points", [])
        )
    ]
    unknown_evidence_filtered = [
        report for report in eligible
        if any(
            point.get("static_guard", {}).get("reason") == "static_non_reachability_not_proven"
            for point in report.get("heal_points", [])
        )
    ]
    reachability_filtered = list({
        report["instance_id"]: report
        for report in reachable_sink_filtered + unknown_evidence_filtered
    }.values())
    static_filtered = [
        report for report in eligible
        if report.get("instance_filter", {}).get("static_guard", False)
    ]

    def correct_filtered(filtered_reports: list[dict]) -> list[dict]:
        return [
            report for report in filtered_reports
            if report.get("run_result", {}).get("classified_status") == "correct"
        ]

    false_positives = correct_filtered(static_filtered)
    policy_false_positives = correct_filtered(policy_filtered)
    reachable_sink_false_positives = correct_filtered(reachable_sink_filtered)
    unknown_evidence_false_positives = correct_filtered(unknown_evidence_filtered)
    denominator = len(no_guard_correct)
    return {
        "no_guard_correct_instances": denominator,
        "no_guard_correct_instance_ids": [report["instance_id"] for report in no_guard_correct],
        "static_policy_filtered_instances": len(policy_filtered),
        "static_policy_filtered_instance_ids": [report["instance_id"] for report in policy_filtered],
        "static_reachability_filtered_instances": len(reachability_filtered),
        "static_reachability_filtered_instance_ids": [report["instance_id"] for report in reachability_filtered],
        "static_reachable_sink_filtered_instances": len(reachable_sink_filtered),
        "static_reachable_sink_filtered_instance_ids": [report["instance_id"] for report in reachable_sink_filtered],
        "static_unknown_evidence_filtered_instances": len(unknown_evidence_filtered),
        "static_unknown_evidence_filtered_instance_ids": [report["instance_id"] for report in unknown_evidence_filtered],
        "false_positive_instances": len(false_positives),
        "false_positive_rate": len(false_positives) / denominator if denominator else 0.0,
        "false_positive_instance_ids": [report["instance_id"] for report in false_positives],
        "policy_false_positive_instances": len(policy_false_positives),
        "policy_false_positive_rate": len(policy_false_positives) / denominator if denominator else 0.0,
        "policy_false_positive_instance_ids": [report["instance_id"] for report in policy_false_positives],
        "reachable_sink_false_positive_instances": len(reachable_sink_false_positives),
        "reachable_sink_false_positive_rate": len(reachable_sink_false_positives) / denominator if denominator else 0.0,
        "reachable_sink_false_positive_instance_ids": [report["instance_id"] for report in reachable_sink_false_positives],
        "unknown_evidence_false_positive_instances": len(unknown_evidence_false_positives),
        "unknown_evidence_false_positive_rate": len(unknown_evidence_false_positives) / denominator if denominator else 0.0,
        "unknown_evidence_false_positive_instance_ids": [report["instance_id"] for report in unknown_evidence_false_positives],
        "definition": "No Guard correct instances filtered by Static Guard divided by all No Guard correct instances",
        "policy_note": "Heal Code policy and strict non-reachability are both enforced offline",
    }


def csv_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_raw_csv(reports: list[dict], output_path: Path) -> None:
    columns = [
        "agent", "model", "instance_id", "run_status", "classified_status", "test_rc", "heal_point_id",
        "file", "function", "line", "thread_id", "heal_before_seq",
        "heal_after_seq", "raw_repair_record", "raw_heal_before", "raw_heal_after",
        "raw_sink_events", "raw_snapshots", "raw_values", "healing_delta",
        "no_guard", "static_guard", "dynamic_guard",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        for report in reports:
            for point in report.get("heal_points", []):
                location = point["location"]
                evidence = point["evidence_refs"]
                writer.writerow({
                    "agent": report.get("agent"),
                    "model": report.get("model"),
                    "instance_id": report.get("instance_id"),
                    "run_status": report.get("run_result", {}).get("status"),
                    "classified_status": report.get("run_result", {}).get("classified_status"),
                    "test_rc": report.get("run_result", {}).get("test_rc"),
                    "heal_point_id": point["heal_point_id"],
                    "file": location.get("file"),
                    "function": location.get("function"),
                    "line": location.get("line"),
                    "thread_id": location.get("thread_id"),
                    "heal_before_seq": evidence.get("heal_before_seq"),
                    "heal_after_seq": evidence.get("heal_after_seq"),
                    "raw_repair_record": csv_json(point.get("raw_repair_record")),
                    "raw_heal_before": csv_json(point.get("raw_heal_before")),
                    "raw_heal_after": csv_json(point.get("raw_heal_after")),
                    "raw_sink_events": csv_json(point.get("dynamic_guard", {}).get("raw_sink_events", [])),
                    "raw_snapshots": csv_json(point.get("raw_snapshots", [])),
                    "raw_values": csv_json(point.get("raw_values", [])),
                    "healing_delta": csv_json(point.get("healing_delta")),
                    "no_guard": csv_json(point.get("no_guard")),
                    "static_guard": csv_json(point.get("static_guard")),
                    "dynamic_guard": csv_json(point.get("dynamic_guard")),
                })


def write_metrics_csv(agent: str, metrics_by_model: dict, output_path: Path) -> None:
    columns = [
        "agent", "model", "guard", "total_instances", "filtered_instances",
        "retained_instances", "processed_instances", "correct_instances", "total_heal_points",
        "filtered_heal_points", "retained_heal_points", "unique_filtered_locations", "coverage_rate",
        "process_rate", "correct_rate", "static_false_positive_instances",
        "static_false_positive_rate", "static_policy_false_positive_instances",
        "static_policy_false_positive_rate", "static_reachable_sink_false_positive_instances",
        "static_reachable_sink_false_positive_rate", "static_unknown_evidence_false_positive_instances",
        "static_unknown_evidence_false_positive_rate",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        for model, model_metrics in metrics_by_model.items():
            false_positives = model_metrics["static_guard_false_positives"]
            for guard in ("no_guard", "static_guard", "dynamic_guard"):
                metrics = model_metrics[guard]
                writer.writerow({
                    "agent": agent, "model": model,
                    "guard": guard,
                    "total_instances": metrics["total_instances"],
                    "filtered_instances": metrics["filtered_instances"],
                    "retained_instances": metrics["retained_instances"],
                    "processed_instances": metrics["processed_instances"],
                    "correct_instances": metrics["correct_instances"],
                    "total_heal_points": metrics["total_heal_points"],
                    "filtered_heal_points": metrics["filtered_heal_points"],
                    "retained_heal_points": metrics["retained_heal_points"],
                    "unique_filtered_locations": metrics["unique_filtered_locations"],
                    "coverage_rate": metrics["coverage_rate"],
                    "process_rate": metrics["process_rate"],
                    "correct_rate": metrics["correct_rate"],
                    "static_false_positive_instances": false_positives["false_positive_instances"] if guard == "static_guard" else "",
                    "static_false_positive_rate": false_positives["false_positive_rate"] if guard == "static_guard" else "",
                    "static_policy_false_positive_instances": false_positives["policy_false_positive_instances"] if guard == "static_guard" else "",
                    "static_policy_false_positive_rate": false_positives["policy_false_positive_rate"] if guard == "static_guard" else "",
                    "static_reachable_sink_false_positive_instances": false_positives["reachable_sink_false_positive_instances"] if guard == "static_guard" else "",
                    "static_reachable_sink_false_positive_rate": false_positives["reachable_sink_false_positive_rate"] if guard == "static_guard" else "",
                    "static_unknown_evidence_false_positive_instances": false_positives["unknown_evidence_false_positive_instances"] if guard == "static_guard" else "",
                    "static_unknown_evidence_false_positive_rate": false_positives["unknown_evidence_false_positive_rate"] if guard == "static_guard" else "",
                })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract No Guard, Static Guard, and Dynamic Guard decisions from completed Heal traces"
    )
    parser.add_argument("--limit", type=int, default=50, help="Completed instances per agent/model (0=all)")
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--agent", default="all", choices=("all", *DEFAULT_AGENTS))
    parser.add_argument("--model", default="all", choices=("all", *DEFAULT_MODELS))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--metrics-csv", type=Path,
                        help="Filter this all_raw.csv using existing reports.json; do not rerun Guard analysis")
    parser.add_argument("--filtered-output-dir", type=Path,
                        help="Write per-Guard all_raw.csv/all_final.csv and coverage here")
    parser.add_argument("--parallel", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached reports and recompute selected analyses",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Embed duplicate snapshots, values, and raw events in reports and raw.csv",
    )
    return parser


def summarize_agent_model(
    agent: str,
    model: str,
    reports: list[dict],
    output_dir: Path,
    include_raw: bool,
) -> dict:
    reports.sort(key=lambda report: report.get("instance_id", ""))
    completed = [report for report in reports if report.get("status") == "completed"]
    metrics = {
        mode: guard_metrics(reports, mode)
        for mode in ("no_guard", "static_guard", "dynamic_guard")
    }
    metrics["static_guard_false_positives"] = static_false_positive_metrics(reports)
    summary = {
        "status": "completed" if len(completed) == len(reports) else "completed_with_errors",
        "agent": agent,
        "model": model,
        "instances": len(reports),
        "result_count": len(reports),
        "completed_results": len(completed),
        "rate_denominator": "retained instances after each Guard filter",
        "metrics": metrics,
        "results": [
            {
                "instance_id": report.get("instance_id"),
                "status": report.get("status"),
                "run_status": report.get("run_result", {}).get("status"),
                "classified_status": report.get("run_result", {}).get("classified_status"),
                "heal_points": report.get("counts", {}).get("heal_points", 0),
                "blocked_heal_attempts": report.get("counts", {}).get("blocked_heal_attempts", 0),
                "filters": report.get("instance_filter", {}),
                "errors": report.get("errors", []),
            }
            for report in reports
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "reports.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if include_raw:
        write_raw_csv(reports, output_dir / "raw.csv")
    else:
        raw_path = output_dir / "raw.csv"
        if raw_path.exists():
            raw_path.unlink()
    write_metrics_csv(agent, {model: metrics}, output_dir / "statistics.csv")
    logger.info(
        f"Tracepoint analysis complete: agent={agent}, model={model}, status={summary['status']}, "
        f"instances={len(reports)}, metrics={metrics}; first 2 results: {summary['results'][:2]}"
    )
    return summary


def filter_saved_metrics(args) -> int:
    import pandas as pd
    from util import HEALBENCH_INSTANCE_IDS
    from evaluate_paper_metrics import aggregate, write_json

    frame = pd.read_csv(args.metrics_csv)
    excluded = ~frame.instance_id.isin(HEALBENCH_INSTANCE_IDS)
    logger.info(f"Excluded {int(excluded.sum())} outside-manifest metric rows")
    frame = frame[~excluded]
    if args.agent != "all":
        frame = frame[frame.agent_class == args.agent]
    if args.model != "all":
        frame = frame[frame.model == args.model]
    if args.instance_id:
        frame = frame[frame.instance_id.isin(args.instance_id)]
    keys = ["agent_class", "model", "instance_id"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate execution keys in metrics CSV")
    reports = {}
    report_paths = []
    for path in sorted(args.output_dir.glob("*/*/summary.json")):
        report_paths.append(str(path))
        summary = json.loads(path.read_text(encoding="utf-8"))
        for saved in summary["results"]:
            report = {**saved, "agent": summary["agent"], "model": summary["model"],
                      "instance_filter": saved.get("filters", {})}
            if report.get("instance_id") not in HEALBENCH_INSTANCE_IDS:
                continue
            key = (report.get("agent", path.parent.parent.name),
                   report.get("model", path.parent.name), report["instance_id"])
            if key in reports:
                raise ValueError(f"Duplicate Guard report: {key}")
            reports[key] = report
    output = args.filtered_output_dir or args.output_dir / "filtered_metrics"
    if output.resolve() == args.metrics_csv.parent.resolve():
        raise ValueError("Filtered output must not overwrite the input metrics directory")
    output.mkdir(parents=True, exist_ok=True)
    rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
    selections = {mode: [] for mode in ("no_guard", "static_guard", "dynamic_guard")}
    coverage = []
    for row in rows:
        key = tuple(row[field] for field in keys)
        report = reports.get(key)
        filters = report.get("instance_filter", {}) if report else {}
        status = ("missing_report" if report is None else
                  "analysis_incomplete" if report.get("status") != "completed" else
                  "missing_filter" if any(type(filters.get(mode)) is not bool
                                          for mode in selections) else "covered")
        coverage.append({**dict(zip(keys, key)), "guard_coverage": status,
                         **{mode: filters.get(mode) for mode in selections}})
        if status != "covered":
            continue
        for mode in selections:
            if not filters[mode]:
                selections[mode].append({**row, "guard_filter": mode})
    summaries = []
    for mode, retained in selections.items():
        directory = output / mode
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(retained, columns=[*frame.columns, "guard_filter"]).to_csv(directory / "all_raw.csv", index=False)
        summary = [{**row, "guard_filter": mode} for row in aggregate(retained)]
        pd.DataFrame(summary).to_csv(directory / "all_final.csv", index=False)
        pd.DataFrame([{**row, "guard_filter": mode} for row in aggregate(retained, by_dataset=True)]).to_csv(
            directory / "by_dataset.csv", index=False)
        summaries.extend(summary)
        logger.info(f"{mode}: retained {len(retained)} executions; first two complete rows: {retained[:2]}")
    pd.DataFrame(summaries).to_csv(output / "all_final.csv", index=False)
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(output / "coverage.csv", index=False)
    write_json(output / "metadata.json", {
        "metrics_source": str(args.metrics_csv.resolve()), "guard_reports": report_paths,
        "coverage": dict(Counter(row["guard_coverage"] for row in coverage)),
        "retained": {mode: len(rows) for mode, rows in selections.items()},
        "definition": "Reuse existing execution-level instance_filter decisions; aggregate original metrics only over retained executions. No Guard rerun or template-to-execution inference.",
        "baseline": "no_guard uses the same completed, explicitly classified Guard-report cohort; missing/error reports are excluded from every arm and listed in coverage.csv.",
        "policy": "Uses saved report policies, not the independent guard_benchmark template policy; saved cache signatures remain in original reports.json.",
        "guard_enabled": "Original runtime flag is unchanged; guard_filter labels post-hoc filtering, not runtime enforcement.",
    })
    logger.info(f"Guard filtering coverage: {dict(Counter(row['guard_coverage'] for row in coverage))}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.metrics_csv:
        return filter_saved_metrics(args)
    if args.limit < 0 or args.parallel < 1:
        raise ValueError("--limit must be non-negative and --parallel must be positive")
    agents = DEFAULT_AGENTS if args.agent == "all" else (args.agent,)
    combinations = [
        (agent, model)
        for agent in agents
        for model in (AGENT_MODELS[agent] if args.model == "all" else (args.model,))
    ]
    combinations_by_instance: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for agent, model in combinations:
        result_root = RESULTS_ROOT / agent / model
        for instance_id in load_instance_ids(args.limit, args.instance_id, result_root):
            combinations_by_instance[instance_id].append((agent, model))

    code_signature = analysis_code_signature()
    cached_reports = load_cached_reports(args.output_dir)
    reports_by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    graph_workers = calculate_graph_workers(
        args.parallel,
        len(combinations_by_instance),
    )
    logger.info(
        f"Analyzing {len(combinations_by_instance)} repository instances across "
        f"{len(combinations)} agent/model groups with {graph_workers} process workers "
        f"(requested={args.parallel}, maximum={MAX_GRAPH_WORKERS})"
    )
    with ProcessPoolExecutor(
        max_workers=graph_workers,
        max_tasks_per_child=1,
    ) as pool:
        futures = {
            pool.submit(
                analyze_repository_instance,
                instance_id,
                group_combinations,
                args.include_raw,
                {
                    key: report
                    for key, report in cached_reports.items()
                    if key[0] == instance_id
                },
                code_signature,
                args.refresh_cache,
            ): (
                instance_id,
                group_combinations,
            )
            for instance_id, group_combinations in combinations_by_instance.items()
        }
        for future in as_completed(futures):
            instance_id, group_combinations = futures[future]
            try:
                instance_reports = future.result()
            except Exception as error:
                logger.error(f"Tracepoint analysis failed for {instance_id}: {type(error).__name__}: {error}")
                instance_reports = [
                    {
                        "status": "error",
                        "instance_id": instance_id,
                        "agent": agent,
                        "model": model,
                        "errors": [{"check": "analysis", "error": f"{type(error).__name__}: {error}"}],
                    }
                    for agent, model in group_combinations
                ]
            for report in instance_reports:
                from evaluate_paper_metrics import write_json

                write_json(args.output_dir / report["agent"] / report["model"] / "instances"
                           / report["instance_id"] / "report.json", report)
                reports_by_group[(report["agent"], report["model"])].append(report)

    summaries = []
    all_reports = []
    for agent, model in combinations:
        reports = reports_by_group[(agent, model)]
        summary = summarize_agent_model(
            agent,
            model,
            reports,
            args.output_dir / agent / model,
            args.include_raw,
        )
        summaries.append(summary)
        all_reports.extend(reports)

    overall = {
        "status": "completed" if all(summary["status"] == "completed" for summary in summaries) else "completed_with_errors",
        "groups": len(summaries),
        "agents": list(agents),
        "models": sorted({summary["model"] for summary in summaries}),
        "summaries": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "reports.json").write_text(
        json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.include_raw:
        write_raw_csv(all_reports, args.output_dir / "raw.csv")
    else:
        raw_path = args.output_dir / "raw.csv"
        if raw_path.exists():
            raw_path.unlink()
    with (args.output_dir / "statistics.csv").open("w", encoding="utf-8", newline="") as output_file:
        output_file.write(
            "agent,model,guard,total_instances,filtered_instances,retained_instances,"
            "processed_instances,correct_instances,total_heal_points,filtered_heal_points,"
            "retained_heal_points,unique_filtered_locations,coverage_rate,process_rate,"
            "correct_rate,static_false_positive_instances,static_false_positive_rate,"
            "static_policy_false_positive_instances,static_policy_false_positive_rate,"
            "static_reachable_sink_false_positive_instances,static_reachable_sink_false_positive_rate,"
            "static_unknown_evidence_false_positive_instances,static_unknown_evidence_false_positive_rate\n"
        )
        for summary in summaries:
            temporary_path = args.output_dir / summary["agent"] / summary["model"] / "statistics.csv"
            lines = temporary_path.read_text(encoding="utf-8").splitlines()
            for line in lines[1:]:
                output_file.write(line + "\n")
    logger.info(f"All tracepoint analyses complete: status={overall['status']}, groups={len(summaries)}; first 2: {summaries[:2]}")
    return 0 if overall["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
