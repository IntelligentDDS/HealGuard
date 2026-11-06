from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from tree_sitter_language_pack import get_parser

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util import HEALBENCH_INSTANCE_IDS
from heal_repository_scope import (
    REPOSITORY_EXCLUDED_DIRECTORIES,
    is_repository_path_excluded,
    iter_repository_python_files,
)

warnings.filterwarnings("ignore", category=SyntaxWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = {
    "source_root": PROJECT_ROOT / "datasets" / "raw_repo_source_code",
    "sink_root": PROJECT_ROOT / "results" / "find_points",
    "key_point_root": PROJECT_ROOT / "results" / "find_points",
    "output_root": PROJECT_ROOT / "results" / "repo_instrumented",
    "excluded_directories": REPOSITORY_EXCLUDED_DIRECTORIES,
}


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def syntax_error_record(error: SyntaxError) -> dict:
    return {
        "type": type(error).__name__,
        "message": error.msg,
        "line": error.lineno,
        "column": error.offset,
        "end_line": error.end_lineno,
        "end_column": error.end_offset,
        "text": error.text,
    }


def node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def call_name(call, source: bytes) -> str:
    function = call.child_by_field_name("function")
    return node_text(function, source) if function else ""


def walk_tree(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.named_children))


def normalize_call(value: str) -> str:
    return value.replace(" ", "")


def _select_sink_calls(source: bytes, sinks: list[dict], relative_file: str) -> tuple[list[dict], list[dict]]:
    tree = get_parser("python").parse(source)
    calls = [node for node in walk_tree(tree.root_node) if node.type == "call"]
    selected = []
    gaps = []
    used_ranges = set()
    for sink in sinks:
        expected_names = {normalize_call(name) for name in sink.get("calls", [])}
        matching_calls = [
            call for call in calls
            if not expected_names or normalize_call(call_name(call, source)) in expected_names
        ]
        sink_range = (
            int(sink.get("line", 0)), int(sink.get("column", 0)),
            int(sink.get("end_line", sink.get("line", 0))), int(sink.get("end_column", 0)),
        )
        exact_candidates = [
            call for call in matching_calls
            if (
                call.start_point.row + 1, call.start_point.column,
                call.end_point.row + 1, call.end_point.column,
            ) == sink_range
        ]
        candidates = exact_candidates or [
            call for call in matching_calls
            if call.start_point.row + 1 >= sink_range[0]
            and call.end_point.row + 1 <= sink_range[2]
        ]
        if len(candidates) != 1:
            gaps.append({
                "reason": "sink_call_not_unique", "sink_id": sink.get("sink_id"), "file": relative_file,
                "candidate_count": len(candidates), "calls": sink.get("calls", []),
            })
            continue
        call = candidates[0]
        byte_range = (call.start_byte, call.end_byte)
        if byte_range in used_ranges:
            gaps.append({"reason": "duplicate_sink_call_range", "sink_id": sink.get("sink_id"), "file": relative_file})
            continue
        used_ranges.add(byte_range)
        selected.append({"sink": sink, "call": call, "actual_call_target": call_name(call, source)})
    return selected, gaps


def _prepare_sink_points(source_text: str, sinks: list[dict], relative_file: str) -> tuple[list[dict], list[dict], list[dict]]:
    source = source_text.encode("utf-8")
    selected, gaps = _select_sink_calls(source, sinks, relative_file)
    locations = []
    valid = []
    for item in selected:
        call = item["call"]
        sink = item["sink"]
        function = call.child_by_field_name("function")
        arguments = call.child_by_field_name("arguments")
        if function is None or arguments is None:
            gaps.append({"reason": "sink_call_fields_missing", "sink_id": sink.get("sink_id"), "file": relative_file})
            continue
        metadata = {
            "sink_id": sink.get("sink_id"), "file": relative_file,
            "line": call.start_point.row + 1, "column": call.start_point.column,
            "end_line": call.end_point.row + 1, "end_column": call.end_point.column,
            "sink_kind": sink.get("sink_kind"), "actual_call_target": item["actual_call_target"],
        }
        valid.append({**item, "function": function, "arguments": arguments, "metadata": metadata})
        locations.append(metadata)
    return valid, locations, gaps


def _instrument_sink_points(source_text: str, sinks: list[dict], relative_file: str) -> tuple[str, list[dict], list[dict]]:
    source = source_text.encode("utf-8")
    valid, locations, gaps = _prepare_sink_points(source_text, sinks, relative_file)

    def render_region(start: int, end: int, excluded=None) -> bytes:
        contained = [
            item for item in valid
            if item is not excluded
            and start <= item["call"].start_byte
            and item["call"].end_byte <= end
        ]
        outermost = [
            item for item in contained
            if not any(
                other["call"].start_byte <= item["call"].start_byte
                and item["call"].end_byte <= other["call"].end_byte
                and other is not item
                for other in contained
            )
        ]
        rendered = bytearray()
        cursor = start
        for item in sorted(outermost, key=lambda value: value["call"].start_byte):
            call = item["call"]
            rendered.extend(source[cursor:call.start_byte])
            rendered.extend(render_call(item))
            cursor = call.end_byte
        rendered.extend(source[cursor:end])
        return bytes(rendered)

    def render_call(item: dict) -> bytes:
        function = item["function"]
        arguments = item["arguments"]
        function_text = render_region(function.start_byte, function.end_byte, item).decode("utf-8", errors="replace")
        arguments_text = render_region(arguments.start_byte + 1, arguments.end_byte - 1, item).decode("utf-8", errors="replace")
        replacement = (
            f"__import__('builtins')._eh_guard_sink_call("
            f"{json.dumps(item['metadata'], ensure_ascii=False)}, {function_text}, locals(), {arguments_text})"
        )
        return replacement.encode("utf-8")

    instrumented = render_region(0, len(source))
    return instrumented.decode("utf-8", errors="replace"), locations, gaps


class SinkPointInstrumenter:
    """Insert exact wrappers around confirmed Sink call expressions."""

    def prepare(
        self, source_text: str, sinks: list[dict], relative_file: str
    ) -> tuple[list[dict], list[dict], list[dict]]:
        return _prepare_sink_points(source_text, sinks, relative_file)

    def instrument(
        self, source_text: str, sinks: list[dict], relative_file: str
    ) -> tuple[str, list[dict], list[dict]]:
        return _instrument_sink_points(source_text, sinks, relative_file)


def build_source_map(instrumented: str, relative_file: str, sink_locations: list[dict], path_locations: list[dict]) -> dict[str, int]:
    line_map = {}
    original_line = 1
    sink_lines = {location["sink_id"]: int(location["line"]) for location in sink_locations}
    point_lines = {location["point_id"]: int(location["line"]) for location in path_locations}
    sink_pattern = re.compile(r'"sink_id":\s*("(?:\\.|[^"\\])*")')
    point_pattern = re.compile(r"_eh_path_hit\((\"(?:\\.|[^\"\\])*\")\)")
    for instrumented_line, text in enumerate(instrumented.splitlines(), start=1):
        mapped_line = original_line
        if "_eh_guard_sink_call" in text:
            match = sink_pattern.search(text)
            if match:
                mapped_line = sink_lines.get(json.loads(match.group(1)), original_line)
        if "_eh_path_hit" in text:
            match = point_pattern.search(text)
            if match:
                mapped_line = point_lines.get(json.loads(match.group(1)), original_line)
        line_map[str(instrumented_line)] = mapped_line
        stripped = text.lstrip()
        is_probe_line = stripped.startswith("__import__('builtins')._eh_path_hit(") or stripped.startswith("__import__('builtins')._eh_path_exception(")
        if not is_probe_line:
            original_line += 1
    return line_map


def locate_source(instance_id: str, key_points: dict, source_root: Path) -> Path:
    repository = str(key_points["repository"]).replace("/", "__")
    prefix = f"{repository}___base-{key_points['base_commit'][:12]}___src-"
    matches = sorted(source_root.glob(f"{prefix}*/repo"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one source repository for {instance_id}, found {len(matches)}: {prefix}")
    return matches[0]


def instrument_instance(
    instance_id: str,
    source_root: Path,
    sink_root: Path,
    key_point_root: Path,
    output_root: Path,
    force: bool,
    sink_instrumenter: SinkPointInstrumenter | None = None,
    key_point_instrumenter: PathKeyPointInstrumenter | None = None,
    file_transform: Callable[[str, str, dict[str, int]], tuple[str, int, dict[int, int]]] | None = None,
) -> dict:
    if instance_id not in HEALBENCH_INSTANCE_IDS:
        return {"instance_id": instance_id, "status": "excluded_gold_trace_unusable"}
    key_points_path = key_point_root / instance_id / "combined_key_points.json"
    sinks_path = sink_root / instance_id / "combined_sinks.json"
    if not key_points_path.is_file() or not sinks_path.is_file():
        return {"instance_id": instance_id, "status": "missing_input"}
    key_points = json_load(key_points_path)
    combined_sinks = json_load(sinks_path)
    if key_points.get("instance_id") != instance_id:
        raise RuntimeError(
            f"Path-point instance mismatch: expected {instance_id}, got {key_points.get('instance_id')}"
        )
    if key_points.get("status") not in {"completed", "completed_with_gaps"}:
        raise RuntimeError(
            f"Path-point analysis is incomplete for {instance_id}: {key_points.get('status')}"
        )
    if combined_sinks.get("status") not in {"completed", "completed_with_gaps"}:
        raise RuntimeError(
            f"Sink analysis is incomplete for {instance_id}: {combined_sinks.get('status')}"
        )
    case_dir = output_root / instance_id
    manifest_path = case_dir / "instrumentation_manifest.json"
    if manifest_path.is_file() and not force:
        existing = json_load(manifest_path)
        if existing.get("status") == "completed":
            return {"instance_id": instance_id, "status": "completed", "reused": True}
    repository = locate_source(instance_id, key_points, source_root)
    output_repo = case_dir / "repo"
    if output_repo.exists():
        shutil.rmtree(output_repo)
    shutil.copytree(repository, output_repo, symlinks=True)
    sinks_by_file: dict[str, list[dict]] = {}
    for sink in combined_sinks.get("confirmed_sinks", []):
        sinks_by_file.setdefault(sink["file"], []).append(sink)
    sink_instrumenter = sink_instrumenter or SinkPointInstrumenter()
    if key_point_instrumenter is None:
        raise ValueError("key_point_instrumenter is required")
    sink_locations = []
    path_locations = []
    source_maps = {}
    sink_gaps = []
    path_gaps = []
    source_syntax_errors = []
    transformed_files = 0
    transformed_functions = 0
    copied = 0
    for source_path in iter_repository_python_files(repository):
        relative = source_path.relative_to(repository)
        if source_path.is_symlink() or is_repository_path_excluded(relative.parts):
            continue
        relative_file = relative.as_posix()
        original_source = source_path.read_text(encoding="utf-8", errors="replace")
        try:
            compile(original_source, relative_file, "exec")
        except SyntaxError as error:
            source_syntax_errors.append({"file": relative_file, **syntax_error_record(error)})
            continue
        source_text = original_source
        file_sink_locations = []
        file_path_locations = []
        file_key_points = key_points.get("files", {}).get(relative_file, {}).get("key_points_by_line", {})
        file_sinks = sinks_by_file.get(relative_file, [])
        if file_key_points:
            source_text, file_path_locations, file_gaps = key_point_instrumenter.instrument(
                source_text, file_key_points, relative_file, file_sinks,
            )
            path_locations.extend(
                location for location in file_path_locations
                if not location.get("sink_id")
            )
            sink_gaps.extend(
                gap for gap in file_gaps
                if gap.get("instrumentation_kind") == "sink"
            )
            path_gaps.extend(
                gap for gap in file_gaps
                if gap.get("instrumentation_kind") != "sink"
            )
        if file_sinks:
            if file_key_points:
                file_sink_locations = [
                    location for location in file_path_locations
                    if location.get("sink_id")
                ]
            else:
                source_text, file_sink_locations, file_gaps = sink_instrumenter.instrument(
                    source_text, file_sinks, relative_file,
                )
                sink_gaps.extend(file_gaps)
            sink_locations.extend(file_sink_locations)
        static_map = build_source_map(
            source_text,
            relative_file,
            file_sink_locations,
            [location for location in file_path_locations if location.get("point_id")],
        )
        if file_transform is not None:
            source_text, function_count, transformed_map = file_transform(
                source_text, relative_file, static_map,
            )
            if function_count:
                source_maps[relative_file] = {
                    str(line): int(static_map[str(static_line)])
                    for line, static_line in transformed_map.items()
                    if str(static_line) in static_map
                }
                transformed_files += 1
                transformed_functions += function_count
            else:
                source_maps[relative_file] = static_map
        else:
            source_maps[relative_file] = static_map
        compile(source_text, relative_file, "exec")
        destination = output_repo / relative
        destination.write_text(source_text, encoding="utf-8")
        copied += 1
    json_dump(output_repo / "_instrumentation_source_map.json", source_maps)
    status = "completed" if not sink_gaps and not path_gaps else "error"
    analysis_gaps = [*sink_gaps, *path_gaps]
    manifest = {
        "status": status, "instance_id": instance_id, "repository": key_points["repository"],
        "base_commit": key_points["base_commit"], "source_repository": str(repository),
        "instrumented_repository": str(output_repo), "python_files": copied,
        "confirmed_sink_count": len(combined_sinks.get("confirmed_sinks", [])),
        "instrumented_sinks": sink_locations, "instrumented_path_key_points": path_locations,
        "source_map": "repo/_instrumentation_source_map.json",
        "analysis_gaps": analysis_gaps,
        "sink_gaps": sink_gaps,
        "path_point_gaps": path_gaps,
        "source_syntax_errors": source_syntax_errors,
        "runtime_strategy": {
            "heal": "logical-region hooks before, after, and exception",
            "sink": "Tree-sitter exact call-expression wrapper",
            "path_key_points": "static statement probes selected by combined_key_points.json",
        },
    }
    json_dump(manifest_path, manifest)
    if path_gaps:
        logger.warning(
            f"Path-point instrumentation recorded {len(path_gaps)} blocking gaps for "
            f"{instance_id}; first 2: {path_gaps[:2]}"
        )
    logger.info(f"Repository instrumentation {instance_id}: status={status}, files={copied}, sinks={len(sink_locations)}, path_points={len(path_locations)}, sink_gaps={len(sink_gaps)}, path_gaps={len(path_gaps)}; first 2 sinks: {sink_locations[:2]}; first 2 path points: {path_locations[:2]}")
    return {
        "instance_id": instance_id,
        "status": status,
        "instrumented_sinks": len(sink_locations),
        "instrumented_path_key_points": len(path_locations),
        "transformed_files": transformed_files,
        "transformed_functions": transformed_functions,
        "sink_gaps": len(sink_gaps),
        "path_gaps": len(path_gaps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Instrument confirmed Sinks and repository path points")
    parser.add_argument("--source-root", type=Path, default=CONFIG["source_root"])
    parser.add_argument("--sink-root", type=Path, default=CONFIG["sink_root"])
    parser.add_argument("--key-point-root", type=Path, default=CONFIG["key_point_root"])
    parser.add_argument("--output-root", type=Path, default=CONFIG["output_root"])
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    instance_ids = args.instance_id or sorted(path.name for path in args.key_point_root.iterdir() if path.is_dir())
    instance_ids = [instance_id for instance_id in instance_ids if instance_id in HEALBENCH_INSTANCE_IDS]
    if args.limit:
        instance_ids = instance_ids[:args.limit]
    logger.info(f"Selected {len(instance_ids)} instances after excluding outside-manifest; first 2: {instance_ids[:2]}")
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        results = list(pool.map(
            lambda instance_id: instrument_instance(instance_id, args.source_root, args.sink_root, args.key_point_root, args.output_root, args.force),
            instance_ids,
        ))
    failures = [result for result in results if result["status"] not in {"completed", "excluded_gold_trace_unusable"}]
    logger.info(f"Sink and path instrumentation finished: total={len(results)}, failures={len(failures)}; first 2 results: {results[:2]}")
    return 1 if failures else 0


# Runtime hooks loaded by the instrumented repository bootstrap.
import ast
import hashlib
import heapq
import json
import sqlite3
from pathlib import Path

import yaml
from loguru import logger


HEAL_POLICY_PATH = Path(__file__).resolve().with_name("heal_code_allowlist_blocklist.yaml")
BRANCH_NODES = (ast.If, ast.Match)
LOOP_NODES = (ast.For, ast.AsyncFor, ast.While)
CALL_BOUNDARY_NODES = (ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return)
EXIT_NODES = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def _call_name(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def load_heal_policy(path: Path = HEAL_POLICY_PATH) -> dict:
    try:
        policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"Heal Policy load failed for {path}: {error}") from error
    if not isinstance(policy, dict):
        raise RuntimeError(f"Heal Policy must be a mapping: {path}")
    if policy.get("default") != "allow_normal_python" or policy.get("forbidden_precedence") is not True:
        raise RuntimeError("Heal Policy must allow normal Python with forbidden precedence")
    for section in ("allowed", "forbidden"):
        if not isinstance(policy.get(section), dict):
            raise RuntimeError(f"Heal Policy section is missing: {section}")
    return policy


class _HealPolicyValidator:
    """Static part 1: allow or reject the complete Heal code region."""

    def __init__(self, policy: dict):
        self.policy = policy
        allowed = policy["allowed"]
        forbidden = policy["forbidden"]
        self.allowed_statements = set(allowed.get("statements", []))
        self.allowed_expressions = set(allowed.get("expressions", []))
        self.allowed_builtin_calls = set(allowed.get("builtin_calls", []))
        self.allowed_import_roots = set(allowed.get("import_roots", []))
        self.allow_project_calls = allowed.get("project_calls") == "allow"
        self.allow_method_calls = allowed.get("method_calls") == "allow"
        self.forbidden_nodes = set(forbidden.get("nodes", []))
        self.forbidden_calls = set(forbidden.get("calls", []))
        self.forbidden_module_roots = set(forbidden.get("module_roots", []))
        self.forbidden_attributes = set(forbidden.get("attributes", []))
        self.forbidden_attribute_calls = set(forbidden.get("attribute_calls", []))

    @classmethod
    def from_file(cls, path: Path = HEAL_POLICY_PATH) -> "_HealPolicyValidator":
        return cls(load_heal_policy(path))

    def validate(self, code: str) -> dict:
        try:
            tree = ast.parse(code, filename="<heal>", mode="exec")
        except SyntaxError as error:
            return {"allowed": False, "reason": "heal_syntax_error", "detail": str(error)}
        imports: dict[str, str] = {}
        for node in ast.walk(tree):
            node_name = type(node).__name__
            if node_name in self.forbidden_nodes:
                return {"allowed": False, "reason": "heal_forbidden_node", "detail": node_name}
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    reason = self._validate_module(module)
                    if reason:
                        return reason
                    imports[alias.asname or module.split(".")[0]] = module
                continue
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                reason = self._validate_module(module)
                if reason:
                    return reason
                for alias in node.names:
                    imports[alias.asname or alias.name] = f"{module}.{alias.name}"
                continue
            if isinstance(node, ast.stmt) and node_name not in self.allowed_statements:
                return {"allowed": False, "reason": "heal_statement_not_allowed", "detail": node_name}
            if isinstance(node, ast.expr) and not isinstance(node, (ast.Call, ast.operator, ast.boolop, ast.cmpop, ast.unaryop, ast.expr_context, ast.keyword)):
                if node_name not in self.allowed_expressions:
                    return {"allowed": False, "reason": "heal_expression_not_allowed", "detail": node_name}
            if isinstance(node, ast.Attribute) and node.attr in self.forbidden_attributes:
                return {"allowed": False, "reason": "heal_forbidden_attribute", "detail": node.attr}
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node.func)
            leaf_name = call_name.rsplit(".", 1)[-1]
            root_name = call_name.split(".", 1)[0]
            resolved_call = call_name.replace(root_name, imports.get(root_name, root_name), 1)
            if call_name in self.forbidden_calls or leaf_name in self.forbidden_calls:
                return {"allowed": False, "reason": "heal_forbidden_call", "detail": call_name}
            if leaf_name in self.forbidden_attribute_calls:
                return {"allowed": False, "reason": "heal_forbidden_attribute_call", "detail": call_name}
            if any(resolved_call == module or resolved_call.startswith(module + ".") for module in self.forbidden_module_roots):
                return {"allowed": False, "reason": "heal_forbidden_module_call", "detail": resolved_call}
            if isinstance(node.func, ast.Attribute):
                if not self.allow_method_calls:
                    return {"allowed": False, "reason": "heal_method_not_allowed", "detail": call_name}
            elif call_name not in self.allowed_builtin_calls and not self.allow_project_calls:
                return {"allowed": False, "reason": "heal_project_call_not_allowed", "detail": call_name or "<dynamic>"}
        return {"allowed": True, "reason": "", "detail": ""}

    def _validate_module(self, module: str) -> dict | None:
        if any(module == forbidden or module.startswith(forbidden + ".") for forbidden in self.forbidden_module_roots):
            return {"allowed": False, "reason": "heal_forbidden_module", "detail": module}
        root = module.split(".", 1)[0]
        if root not in self.allowed_import_roots:
            return {"allowed": False, "reason": "heal_import_not_allowed", "detail": module}
        return None


def validate_heal_code(code: str, policy: dict | None = None) -> dict:
    return _HealPolicyValidator(policy or load_heal_policy()).validate(code)


def _loaded_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}


def _stored_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del))}


def _statement_definitions(statement: ast.stmt) -> set[str]:
    if isinstance(statement, ast.Assign):
        return {name for target in statement.targets for name in _stored_names(target)}
    if isinstance(statement, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        return _stored_names(statement.target)
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        return _stored_names(statement.target)
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return {
            name
            for item in statement.items
            for name in _stored_names(item.optional_vars)
        }
    return set()


def _statement_uses(statement: ast.stmt) -> set[str]:
    if isinstance(statement, ast.Assign):
        return _loaded_names(statement.value)
    if isinstance(statement, ast.AnnAssign):
        return _loaded_names(statement.value)
    if isinstance(statement, ast.AugAssign):
        return _loaded_names(statement.target) | _loaded_names(statement.value)
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        return _loaded_names(statement.iter)
    return _loaded_names(statement)


def _key_point_kinds(statement: ast.stmt, definitions: set[str], uses: set[str]) -> list[str]:
    kinds = []
    if definitions:
        kinds.append("variable_definition")
        if uses:
            kinds.append("data_transform")
        if len(uses) > 1:
            kinds.append("data_merge")
    if isinstance(statement, BRANCH_NODES):
        kinds.append("branch")
    if isinstance(statement, LOOP_NODES):
        kinds.append("loop")
    if isinstance(statement, EXIT_NODES):
        kinds.append("function_or_control_exit")
    if isinstance(statement, (ast.Try, getattr(ast, "TryStar", ast.Try))):
        kinds.append("exception_region")
    if isinstance(statement, CALL_BOUNDARY_NODES) and any(isinstance(node, ast.Call) for node in ast.walk(statement)):
        kinds.append("call_boundary")
    return kinds


def _statement_summary(statement: ast.stmt, function_name: str) -> dict:
    definitions = _statement_definitions(statement)
    uses = _statement_uses(statement)
    calls = sorted({_call_name(node.func) for node in ast.walk(statement) if isinstance(node, ast.Call) and _call_name(node.func)})
    return {
        "statement_id": f"{statement.lineno}:{statement.col_offset}:{type(statement).__name__}",
        "function": function_name,
        "line": statement.lineno,
        "column": statement.col_offset,
        "end_line": statement.end_lineno or statement.lineno,
        "end_column": statement.end_col_offset or statement.col_offset,
        "node_type": type(statement).__name__,
        "definitions": sorted(definitions),
        "uses": sorted(uses),
        "calls": calls,
        "key_point_kinds": _key_point_kinds(statement, definitions, uses),
    }


def _owned_statements(node: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    statements = []

    def visit(statement: ast.stmt) -> None:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        statements.append(statement)
        for child in ast.iter_child_nodes(statement):
            if isinstance(child, ast.stmt):
                visit(child)

    for statement in node.body:
        visit(statement)
    return statements


class _FunctionGraphBuilder:
    def __init__(self, function_node: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef):
        self.function_node = function_node
        self.function_name = function_node.name if isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "<module>"
        self.statements = _owned_statements(function_node)
        self.statements_by_id = {self._statement_id(statement): statement for statement in self.statements}
        self.cfg_edges: list[dict] = []

    @staticmethod
    def _statement_id(statement: ast.stmt) -> str:
        return f"{statement.lineno}:{statement.col_offset}:{type(statement).__name__}"

    def _add_cfg_edges(self, source: ast.stmt, targets: list[ast.stmt], edge_type: str, detail: str = "") -> None:
        for target in targets:
            self.cfg_edges.append({
                "source_id": self._statement_id(source),
                "target_id": self._statement_id(target),
                "edge_type": edge_type,
                "detail": detail,
            })

    def _build_block(
        self,
        statements: list[ast.stmt],
        follow: list[ast.stmt],
        break_targets: list[ast.stmt] | None = None,
        continue_targets: list[ast.stmt] | None = None,
    ) -> list[ast.stmt]:
        next_entries = follow
        for statement in reversed(statements):
            next_entries = self._build_statement(
                statement, next_entries, break_targets or [], continue_targets or [],
            )
        return next_entries

    def _build_statement(
        self,
        statement: ast.stmt,
        follow: list[ast.stmt],
        break_targets: list[ast.stmt],
        continue_targets: list[ast.stmt],
    ) -> list[ast.stmt]:
        if isinstance(statement, (ast.Return, ast.Raise)):
            return [statement]
        if isinstance(statement, ast.Break):
            self._add_cfg_edges(statement, break_targets, "cfg_break")
            return [statement]
        if isinstance(statement, ast.Continue):
            self._add_cfg_edges(statement, continue_targets, "cfg_continue")
            return [statement]
        if isinstance(statement, ast.If):
            body = self._build_block(statement.body, follow, break_targets, continue_targets)
            orelse = self._build_block(statement.orelse, follow, break_targets, continue_targets) if statement.orelse else follow
            condition = ast.unparse(statement.test)
            self._add_cfg_edges(statement, body, "cfg_true", condition)
            self._add_cfg_edges(statement, orelse, "cfg_false", condition)
            return [statement]
        if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            body = self._build_block(statement.body, [statement], follow, [statement])
            orelse = self._build_block(statement.orelse, follow, break_targets, continue_targets) if statement.orelse else follow
            condition = ast.unparse(statement.test) if isinstance(statement, ast.While) else ast.unparse(statement.iter)
            self._add_cfg_edges(statement, body, "cfg_loop_body", condition)
            self._add_cfg_edges(statement, orelse, "cfg_loop_exit", condition)
            return [statement]
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            body = self._build_block(statement.body, follow, break_targets, continue_targets)
            self._add_cfg_edges(statement, body or follow, "cfg_body")
            return [statement]
        if isinstance(statement, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            final_entries = self._build_block(statement.finalbody, follow, break_targets, continue_targets) if statement.finalbody else follow
            else_entries = self._build_block(statement.orelse, final_entries, break_targets, continue_targets) if statement.orelse else final_entries
            body = self._build_block(statement.body, else_entries, break_targets, continue_targets)
            self._add_cfg_edges(statement, body or else_entries, "cfg_try")
            for handler in statement.handlers:
                handler_entries = self._build_block(handler.body, final_entries, break_targets, continue_targets)
                self._add_cfg_edges(statement, handler_entries or final_entries, "cfg_exception")
            return [statement]
        if isinstance(statement, ast.Match):
            for case_index, case in enumerate(statement.cases):
                entries = self._build_block(case.body, follow, break_targets, continue_targets)
                self._add_cfg_edges(statement, entries or follow, "cfg_match_case", str(case_index))
            return [statement]
        self._add_cfg_edges(statement, follow, "cfg_next")
        return [statement]

    def _build_dfg_edges(self) -> list[dict]:
        predecessors: dict[str, set[str]] = {statement_id: set() for statement_id in self.statements_by_id}
        for edge in self.cfg_edges:
            predecessors.setdefault(edge["target_id"], set()).add(edge["source_id"])
        reaching_out: dict[str, dict[str, set[str]]] = {statement_id: {} for statement_id in self.statements_by_id}
        changed = True
        while changed:
            changed = False
            for statement_id, statement in self.statements_by_id.items():
                incoming: dict[str, set[str]] = {}
                for predecessor in predecessors.get(statement_id, set()):
                    for name, definitions in reaching_out.get(predecessor, {}).items():
                        incoming.setdefault(name, set()).update(definitions)
                outgoing = {name: set(definitions) for name, definitions in incoming.items()}
                for name in _statement_definitions(statement):
                    outgoing[name] = {statement_id}
                if outgoing != reaching_out[statement_id]:
                    reaching_out[statement_id] = outgoing
                    changed = True
        dfg_edges = []
        for statement_id, statement in self.statements_by_id.items():
            incoming: dict[str, set[str]] = {}
            for predecessor in predecessors.get(statement_id, set()):
                for name, definitions in reaching_out.get(predecessor, {}).items():
                    incoming.setdefault(name, set()).update(definitions)
            for name in sorted(_statement_uses(statement)):
                for definition_id in sorted(incoming.get(name, set())):
                    dfg_edges.append({
                        "source_id": definition_id,
                        "target_id": statement_id,
                        "edge_type": "dfg_def_use",
                        "detail": name,
                    })
        return dfg_edges

    def build(self) -> tuple[list[dict], list[dict], list[dict]]:
        self._build_block(self.function_node.body, [])
        points = [_statement_summary(statement, self.function_name) for statement in self.statements]
        dfg_edges = self._build_dfg_edges()
        edge_types_by_node: dict[str, set[str]] = {}
        for edge in [*self.cfg_edges, *dfg_edges]:
            edge_types_by_node.setdefault(edge["source_id"], set()).add(edge["edge_type"])
            edge_types_by_node.setdefault(edge["target_id"], set()).add(edge["edge_type"])
        for point in points:
            point["graph_edge_types"] = sorted(edge_types_by_node.get(point["statement_id"], set()))
        key_points = [
            point for point in points
            if point["key_point_kinds"] or point["graph_edge_types"]
        ]
        key_points.sort(key=lambda item: (item["line"], item["column"], item["end_line"], item["node_type"]))
        return key_points, self.cfg_edges, dfg_edges


def _containing_function(tree: ast.Module, line: int) -> ast.Module | ast.FunctionDef | ast.AsyncFunctionDef:
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    return min(candidates, key=lambda node: (node.end_lineno or node.lineno) - node.lineno) if candidates else tree


def _sink_influence_names(
    function_node: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
    sink: dict,
) -> list[str]:
    sink_start = int(sink.get("line", 0))
    sink_end = int(sink.get("end_line", sink_start))
    sink_calls = set(sink.get("calls", []))
    influence = set()
    statements = _owned_statements(function_node)
    for statement in statements:
        if (statement.end_lineno or statement.lineno) < sink_start or statement.lineno > sink_end:
            continue
        for call in (node for node in ast.walk(statement) if isinstance(node, ast.Call)):
            if sink_calls and _call_name(call.func) not in sink_calls:
                continue
            influence.update(
                name
                for argument in [*call.args, *(keyword.value for keyword in call.keywords)]
                for name in _loaded_names(argument)
            )
    changed = True
    while changed:
        changed = False
        for statement in reversed(statements):
            definitions = _statement_definitions(statement)
            if not definitions & influence:
                continue
            expanded = influence | _statement_uses(statement)
            if expanded != influence:
                influence = expanded
                changed = True
        for statement in statements:
            condition = None
            controlled_body = []
            if isinstance(statement, ast.If):
                condition = statement.test
                controlled_body = [*statement.body, *statement.orelse]
            elif isinstance(statement, ast.While):
                condition = statement.test
                controlled_body = [*statement.body, *statement.orelse]
            elif isinstance(statement, (ast.For, ast.AsyncFor)):
                condition = statement.iter
                controlled_body = [*statement.body, *statement.orelse]
            if condition is None:
                continue
            controlled_definitions = {
                name
                for controlled_statement in controlled_body
                for name in _statement_definitions(controlled_statement)
            }
            if not controlled_definitions & influence:
                continue
            expanded = influence | _loaded_names(condition)
            if expanded != influence:
                influence = expanded
                changed = True
    return sorted(influence)


def argument_is_literal(expression: ast.AST) -> bool:
    """Prove value independence only for syntax without runtime name lookups."""
    if isinstance(expression, ast.Constant):
        return True
    if isinstance(expression, (ast.Tuple, ast.List, ast.Set)):
        return all(argument_is_literal(item) for item in expression.elts)
    if isinstance(expression, ast.Dict):
        return all(key is not None and argument_is_literal(key) and argument_is_literal(value)
                   for key, value in zip(expression.keys, expression.values))
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, (ast.UAdd, ast.USub, ast.Not, ast.Invert)):
        return argument_is_literal(expression.operand)
    return False


def sink_argument_expressions(function_node: ast.AST, sink: dict) -> list[str] | None:
    """Return complete arguments only when the confirmed call is unambiguous."""
    start = int(sink.get("line", 0))
    end = int(sink.get("end_line", start))
    names = set(sink.get("calls", []))
    calls = [node for node in ast.walk(function_node) if isinstance(node, ast.Call)
             and start <= node.lineno <= end and (not names or _call_name(node.func) in names)]
    if len(calls) != 1 or any(keyword.arg is None for keyword in calls[0].keywords):
        return None
    return [ast.unparse(argument) for argument in
            [*calls[0].args, *(keyword.value for keyword in calls[0].keywords)]]


class _StaticSinkReachabilityAnalyzer:
    """Static part 2: derive Sink influence names and CFG/DFG recording points."""

    def analyze_file(self, tree: ast.Module, confirmed_sinks: list[dict]) -> tuple[dict[str, list[dict]], list[dict], dict]:
        function_nodes = [tree]
        function_nodes.extend(
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        key_points_by_line: dict[str, list[dict]] = {}
        cfg_edges = []
        dfg_edges = []
        for function_node in function_nodes:
            points, function_cfg_edges, function_dfg_edges = _FunctionGraphBuilder(function_node).build()
            for point in points:
                key_points_by_line.setdefault(str(point["line"]), []).append(point)
            cfg_edges.extend(function_cfg_edges)
            dfg_edges.extend(function_dfg_edges)
        analyzed_sinks = []
        for sink in confirmed_sinks:
            sink_copy = dict(sink)
            function_node = _containing_function(tree, int(sink.get("line", 0)))
            sink_copy["static_influence_names"] = _sink_influence_names(function_node, sink)
            sink_copy["argument_expressions"] = sink_argument_expressions(function_node, sink)
            analyzed_sinks.append(sink_copy)
        graph = {"cfg_edges": cfg_edges, "dfg_edges": dfg_edges}
        return key_points_by_line, analyzed_sinks, graph


class StaticGuard:
    """Pre-run Guard: validate Heal code and analyze paths to confirmed Sinks."""

    def __init__(self, source_root: Path | None = None, heal_policy: dict | None = None):
        self.source_root = source_root
        self.heal_policy = heal_policy or load_heal_policy()
        self.heal_validator = _HealPolicyValidator(self.heal_policy)
        self.reachability = _StaticSinkReachabilityAnalyzer()

    def validate_heal(self, code: str) -> dict:
        return self.heal_validator.validate(code)

    def build_manifest(
        self,
        confirmed_sinks: list[dict],
        traced_files: list[str] | None = None,
        path_key_points: dict | None = None,
    ) -> dict:
        if self.source_root is None:
            raise RuntimeError("StaticGuard source_root is required to build a manifest")
        return _build_guard_manifest(
            self.source_root,
            confirmed_sinks,
            traced_files,
            self.heal_policy,
            self.reachability,
            path_key_points,
        )


def _build_guard_manifest(
    source_root: Path,
    confirmed_sinks: list[dict],
    traced_files: list[str] | None,
    heal_policy: dict,
    reachability: _StaticSinkReachabilityAnalyzer,
    path_key_points: dict | None,
) -> dict:
    del traced_files, path_key_points
    sink_files = {str(sink.get("file", "")) for sink in confirmed_sinks if sink.get("file")}
    files = {}
    gaps = []
    for relative_path in sorted(sink_files):
        source_path = source_root / relative_path
        try:
            source = source_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=relative_path)
        except (OSError, SyntaxError) as error:
            logger.warning(f"Guard manifest parse failed for {relative_path}: {error}")
            gaps.append({"reason": "guard_source_parse_failed", "file": relative_path, "error": str(error)})
            continue
        file_sinks = [sink for sink in confirmed_sinks if sink.get("file") == relative_path]
        _, file_sinks, _ = reachability.analyze_file(tree, file_sinks)
        files[relative_path] = {
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "sinks": file_sinks,
        }
    return {"files": files, "analysis_gaps": gaps, "heal_policy": heal_policy}


def build_guard_manifest(
    source_root: Path,
    confirmed_sinks: list[dict],
    traced_files: list[str] | None = None,
    path_key_points: dict | None = None,
) -> dict:
    return StaticGuard(source_root).build_manifest(confirmed_sinks, traced_files, path_key_points)


def write_guard_manifest(
    source_root: Path,
    confirmed_sinks_path: Path,
    output_path: Path,
) -> dict:
    confirmed_sinks = json.loads(confirmed_sinks_path.read_text(encoding="utf-8"))
    manifest = StaticGuard(source_root).build_manifest(confirmed_sinks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


TRACE_MANIFEST_NAME = "_trace_manifest.json"


class _AstTraceSelector:
    """Recording preparation: select AST locations for Heal, Sink, and path events."""

    def build(self, source_root: str, relative_paths: list[str]) -> dict:
        return _build_ast_trace_manifest(source_root, relative_paths)


def _build_ast_trace_manifest(source_root: str, relative_paths: list[str]) -> dict:
    root = Path(source_root)
    files = {}
    for relative_path in sorted(set(relative_paths)):
        source_path = root / relative_path
        source = source_path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=relative_path)
        except SyntaxError as error:
            logger.warning(
                f"Skipping Python-incompatible trace manifest file {relative_path}: {error}"
            )
            continue
        statements = []
        line_entries: dict[str, list[dict]] = {}

        def visit_body(body: list[ast.stmt], regions: tuple[dict, ...]) -> None:
            for statement in body:
                start = statement.lineno
                end = statement.end_lineno or start
                statement_id = f"{relative_path}:{start}:{getattr(statement, 'col_offset', 0)}:{type(statement).__name__}"
                entry = {
                    "statement_id": statement_id,
                    "node_type": type(statement).__name__,
                    "line": start,
                    "end_line": end,
                    "column": getattr(statement, "col_offset", 0),
                    "end_column": getattr(statement, "end_col_offset", 0),
                    "regions": list(regions),
                }
                statements.append(entry)
                for line_number in range(start, end + 1):
                    line_entries.setdefault(str(line_number), []).append(entry)

                child_regions = regions
                if isinstance(statement, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                    visit_body(statement.body, (*regions, {"kind": "try", "line": statement.lineno}))
                    for handler_index, handler in enumerate(statement.handlers):
                        visit_body(handler.body, (*regions, {
                            "kind": "except",
                            "line": handler.lineno,
                            "handler_index": handler_index,
                            "exception_type": ast.unparse(handler.type) if handler.type else None,
                        }))
                    visit_body(statement.orelse, (*regions, {"kind": "try_else", "line": statement.lineno}))
                    visit_body(statement.finalbody, (*regions, {"kind": "finally", "line": statement.lineno}))
                    continue
                for field in ("body", "orelse", "finalbody"):
                    child = getattr(statement, field, None)
                    if isinstance(child, list):
                        visit_body(child, child_regions)
                if isinstance(statement, ast.Match):
                    for case_index, case in enumerate(statement.cases):
                        visit_body(case.body, (*regions, {"kind": "case", "line": case.pattern.lineno, "case_index": case_index}))

        visit_body(tree.body, ())
        files[relative_path] = {
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "statements": statements,
            "line_entries": line_entries,
        }
        executable_lines = set(line_entries)
        files[relative_path]["source_lines"] = [
            {
                "line": line_number,
                "source": content,
                "classification": "executable_ast" if str(line_number) in executable_lines else "non_executable",
            }
            for line_number, content in enumerate(source.splitlines(), 1)
        ]
    payload = {"schema_version": "ast-runtime-trace-manifest-v1", "files": files}
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def build_ast_trace_manifest(source_root: str, relative_paths: list[str]) -> dict:
    return _AstTraceSelector().build(source_root, relative_paths)


def write_ast_trace_manifest(source_root: str, relative_paths: list[str], output_path: str) -> dict:
    manifest = _AstTraceSelector().build(source_root, relative_paths)
    Path(output_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


class _TraceArtifactManager:
    """Post-run stage: prepare, identify, merge, and finalize complete Trace artifacts."""

    @staticmethod
    def prepare(path: str) -> None:
        prepare_runtime_trace(path)

    @staticmethod
    def finalize(path: str) -> dict:
        return finalize_runtime_trace(path)

    @staticmethod
    def write_metadata(path: str, source_root: str, source_map_path: str, manifest_path: str) -> dict:
        return write_runtime_trace_metadata(path, source_root, source_map_path, manifest_path)

    @staticmethod
    def consolidate(trace_paths: list[str], output_dir: str) -> dict:
        return consolidate_runtime_trace_tables(trace_paths, output_dir)


def _data_path(trace_path: Path, kind: str) -> Path:
    return trace_path.with_name(f"run_docker_{kind}.jsonl")


def write_runtime_trace_metadata(path: str, source_root: str, source_map_path: str, manifest_path: str) -> dict:
    source_map_bytes = Path(source_map_path).read_bytes()
    manifest_bytes = Path(manifest_path).read_bytes()
    source_map = json.loads(source_map_bytes)
    manifest = json.loads(manifest_bytes)
    source_hashes = {
        relative_path: file_manifest["source_sha256"]
        for relative_path, file_manifest in manifest.get("files", {}).items()
    }
    if set(source_map) != set(source_hashes):
        raise RuntimeError("Runtime source map and AST manifest file coverage differ")
    trace_path = Path(path)
    source_map_artifact = trace_path.with_name(trace_path.name + ".source_map.json")
    manifest_artifact = trace_path.with_name(trace_path.name + ".manifest.json")
    source_map_artifact.write_bytes(source_map_bytes)
    manifest_artifact.write_bytes(manifest_bytes)
    metadata = {
        "schema_version": "runtime-trace-source-identity-v1",
        "source_map_path": source_map_artifact.name,
        "source_map_sha256": hashlib.sha256(source_map_bytes).hexdigest(),
        "ast_manifest_path": manifest_artifact.name,
        "ast_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_mapped_files": sorted(source_map),
        "original_source_sha256": source_hashes,
    }
    metadata_path = trace_path.with_name(trace_path.name + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def prepare_runtime_trace(path: str) -> None:
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = [
        trace_path,
        _data_path(trace_path, "snapshots"),
        _data_path(trace_path, "values"),
        trace_path.with_name(trace_path.name + ".metadata.json"),
        trace_path.with_name(trace_path.name + ".source_map.json"),
        trace_path.with_name(trace_path.name + ".manifest.json"),
        trace_path.with_name(trace_path.name + ".collector_failures.json"),
    ]
    candidates.append(trace_path.with_name(trace_path.name + ".malformed.jsonl"))
    for suffix in ("events", "snapshots", "values"):
        candidates.extend(trace_path.parent.glob(trace_path.name + f".*.{suffix}.jsonl"))
    for candidate in candidates:
        candidate.unlink(missing_ok=True)


def _read_record(trace_file, source_path: Path, line_number: int, malformed_file):
    line = trace_file.readline()
    if not line:
        return None
    try:
        return json.loads(line), line_number + 1
    except Exception as error:
        malformed_file.write(json.dumps({
            "file": source_path.name,
            "line": line_number,
            "raw": line,
            "error": f"{type(error).__name__}: {error}",
        }, ensure_ascii=False) + "\n")
        return _read_record(trace_file, source_path, line_number + 1, malformed_file)


def _merge_events(trace_path: Path, source_paths: list[Path], malformed_file) -> int:
    nonempty_paths = [path for path in source_paths if path.stat().st_size > 0]
    for empty_path in set(source_paths) - set(nonempty_paths):
        empty_path.unlink()
    source_paths = nonempty_paths
    if not source_paths:
        trace_path.write_text("", encoding="utf-8")
        return 0
    if len(source_paths) == 1:
        source_path = source_paths[0]
        count = 0
        with source_path.open("r", encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, 1):
                try:
                    json.loads(line)
                    count += 1
                except Exception as error:
                    malformed_file.write(json.dumps({
                        "file": source_path.name,
                        "line": line_number,
                        "raw": line,
                        "error": f"{type(error).__name__}: {error}",
                    }, ensure_ascii=False) + "\n")
        trace_path.unlink(missing_ok=True)
        source_path.replace(trace_path)
        return count
    files = [path.open("r", encoding="utf-8") for path in source_paths]
    heap = []
    try:
        for index, (source_path, trace_file) in enumerate(zip(source_paths, files)):
            item = _read_record(trace_file, source_path, 1, malformed_file)
            if item:
                record, next_line = item
                heapq.heappush(heap, ((record.get("_time_ns", 0), record.get("_pid", 0), index), record, next_line))
        count = 0
        with trace_path.open("w", encoding="utf-8") as output_file:
            while heap:
                (_, _, index), record, next_line = heapq.heappop(heap)
                count += 1
                record["source_seq"] = record.get("seq")
                record["seq"] = count
                record.pop("_time_ns", None)
                record.pop("_pid", None)
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                item = _read_record(files[index], source_paths[index], next_line, malformed_file)
                if item:
                    next_record, following_line = item
                    key = (next_record.get("_time_ns", 0), next_record.get("_pid", 0), index)
                    heapq.heappush(heap, (key, next_record, following_line))
    finally:
        for trace_file in files:
            trace_file.close()
    for source_path in source_paths:
        source_path.unlink()
    return count


def _merge_unique(output_path: Path, source_paths: list[Path], id_field: str, malformed_file) -> int:
    index_path = output_path.with_suffix(output_path.suffix + ".index.sqlite3")
    index_path.unlink(missing_ok=True)
    count = 0
    with sqlite3.connect(index_path) as database, output_path.open("w", encoding="utf-8") as output_file:
        database.execute("CREATE TABLE seen (record_id TEXT PRIMARY KEY)")
        for source_path in source_paths:
            with source_path.open("r", encoding="utf-8") as source_file:
                for line_number, line in enumerate(source_file, 1):
                    try:
                        record = json.loads(line)
                    except Exception as error:
                        malformed_file.write(json.dumps({
                            "file": source_path.name,
                            "line": line_number,
                            "raw": line,
                            "error": f"{type(error).__name__}: {error}",
                        }, ensure_ascii=False) + "\n")
                        continue
                    cursor = database.execute(
                        "INSERT OR IGNORE INTO seen(record_id) VALUES (?)",
                        (record[id_field],),
                    )
                    if cursor.rowcount == 0:
                        continue
                    count += 1
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            database.commit()
            source_path.unlink()
    index_path.unlink()
    return count


def consolidate_runtime_trace_tables(trace_paths: list[str], output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    snapshot_paths = [_data_path(Path(path), "snapshots") for path in trace_paths]
    value_paths = [_data_path(Path(path), "values") for path in trace_paths]
    malformed_path = output_path / "malformed.jsonl"
    with malformed_path.open("w", encoding="utf-8") as malformed_file:
        snapshots = _merge_unique(output_path / "snapshots.jsonl", snapshot_paths, "snapshot_id", malformed_file)
        values = _merge_unique(output_path / "values.jsonl", value_paths, "value_id", malformed_file)
    with malformed_path.open("r", encoding="utf-8") as malformed_file:
        malformed = sum(1 for _ in malformed_file)
    if malformed == 0:
        malformed_path.unlink()
    return {
        "snapshots": snapshots,
        "values": values,
        "snapshots_path": "snapshots.jsonl",
        "values_path": "values.jsonl",
        "malformed": malformed,
    }


def _close_unfinished_heals(trace_path: Path) -> int:
    pending = {}
    maximum_seq = 0
    with trace_path.open("r", encoding="utf-8") as trace_file:
        for line in trace_file:
            record = json.loads(line)
            maximum_seq = max(maximum_seq, int(record.get("seq", 0)))
            heal_id = record.get("heal_id")
            if not heal_id:
                continue
            if record.get("action") == "heal_before":
                pending[heal_id] = record
            elif record.get("action") == "heal_after":
                pending.pop(heal_id, None)
    if not pending:
        return 0
    replacement_path = trace_path.with_name(f".{trace_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(trace_path, replacement_path)
        with replacement_path.open("a", encoding="utf-8") as trace_file:
            for offset, (heal_id, before) in enumerate(pending.items(), 1):
                snapshot_id = before.get("snapshot_id")
                record = {
                    "run_id": before.get("run_id"),
                    "seq": maximum_seq + offset,
                    "thread_id": before.get("thread_id"),
                    "file": before.get("file"),
                    "func": before.get("func"),
                    "line": before.get("line"),
                    "call_stack_id": before.get("call_stack_id"),
                    "action": "heal_after",
                    "ast_statements": before.get("ast_statements", []),
                    "try_regions": before.get("try_regions", []),
                    "snapshot_id": None,
                    "snapshot_scope": before.get("snapshot_scope", "locals"),
                    "heal_id": heal_id,
                    "crash_num": before.get("crash_num"),
                    "outcome": "interrupted",
                    "interruption_reason": "runtime_terminated_without_after",
                    "before_snapshot_id": before.get("captured_snapshot_id", snapshot_id),
                    "after_snapshot_id": None,
                    "last_available_snapshot_id": snapshot_id,
                    "added_names": None,
                    "deleted_names": None,
                    "changed_names": None,
                }
                trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        replacement_path.replace(trace_path)
    finally:
        replacement_path.unlink(missing_ok=True)
    return len(pending)


def finalize_runtime_trace(path: str) -> dict:
    trace_path = Path(path)
    event_paths = sorted(trace_path.parent.glob(trace_path.name + ".*.events.jsonl"))
    snapshot_paths = sorted(trace_path.parent.glob(trace_path.name + ".*.snapshots.jsonl"))
    value_paths = sorted(trace_path.parent.glob(trace_path.name + ".*.values.jsonl"))
    malformed_path = trace_path.with_name(trace_path.name + ".malformed.jsonl")
    with malformed_path.open("w", encoding="utf-8") as malformed_file:
        records = _merge_events(trace_path, event_paths, malformed_file)
        interrupted_heals = _close_unfinished_heals(trace_path)
        records += interrupted_heals
        snapshots = _merge_unique(_data_path(trace_path, "snapshots"), snapshot_paths, "snapshot_id", malformed_file)
        values = _merge_unique(_data_path(trace_path, "values"), value_paths, "value_id", malformed_file)
    with malformed_path.open("r", encoding="utf-8") as malformed_file:
        malformed = sum(1 for _ in malformed_file)
    if malformed == 0:
        malformed_path.unlink()
    collector_failures_path = trace_path.with_name(trace_path.name + ".collector_failures.json")
    collector_failures = []
    if collector_failures_path.is_file():
        try:
            collector_failures = json.loads(collector_failures_path.read_text(encoding="utf-8")).get("failure_stages", [])
        except Exception as error:
            collector_failures = [f"failure_report_unreadable:{type(error).__name__}:{error}"]
    return {
        "records": records,
        "processes": len(event_paths),
        "snapshots": snapshots,
        "values": values,
        "snapshots_path": _data_path(trace_path, "snapshots").name,
        "values_path": _data_path(trace_path, "values").name,
        "malformed": malformed,
        "collector_failures": collector_failures,
        "interrupted_heals": interrupted_heals,
        "complete": records > 0 and malformed == 0 and not collector_failures,
        "value_fallbacks_present": "REPR" in collector_failures,
    }


if __name__ == "__main__":
    raise SystemExit(main())
