from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import re
import shutil
import warnings
from pathlib import Path

from loguru import logger
from tree_sitter_language_pack import get_parser

from util import HEALBENCH_INSTANCE_IDS
from heal_repository_scope import (
    HEAL_EXCLUDED_DIRECTORIES,
    is_repository_path_excluded,
    iter_repository_python_files,
)
from instrument_guard import (
    CONFIG,
    TRACE_MANIFEST_NAME,
    SinkPointInstrumenter,
    instrument_instance,
    write_ast_trace_manifest,
    write_guard_manifest,
)

warnings.filterwarnings("ignore", category=SyntaxWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPAIR_RECORD_PATH_MARKER = "__EH_RECORD_PATH__"
MARKER_PORT = "__EH_PORT__"
HEAL_HOOK_NAMES = {
    "_eh_heal_begin",
    "_eh_heal_failed",
    "_eh_heal_succeeded",
    "_eh_path_hit",
    "_eh_path_decision_before",
    "_eh_path_decision",
    "_eh_path_control",
    "_eh_path_exception",
    "_eh_guard_sink_call",
}


def walk_tree(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def _indent_unit(indent: str) -> str:
    return "\t" if "\t" in indent else "    "


def _build_except_block(
    file_path: str,
    source_line: int,
    block_idx: int,
    indent: str,
    stmt_source: str,
    host_port: int | str = 19876,
) -> str:
    metadata = {
        "file": file_path,
        "line": source_line,
        "block_index": block_idx,
        "statement": stmt_source,
        "port": host_port,
        "token": "__EH_TOKEN__",
        "record_path": REPAIR_RECORD_PATH_MARKER,
    }
    unit = _indent_unit(indent)
    inner = indent + unit
    return (
        f"{indent}except Exception as _eh_err:\n"
        f"{inner}_eh_has_return, _eh_return = "
        f"__import__('builtins')._eh_handle_exception("
        f"_eh_err, {json.dumps(metadata, ensure_ascii=False)}, "
        f"__import__('sys')._getframe(0))\n"
        f"{inner}if _eh_has_return:\n"
        f"{inner}{unit}return _eh_return\n"
    )


def _split_return_stmts(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def containing_function(node: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        parent = parents.get(node)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return parent
            parent = parents.get(parent)
        return None

    def is_generator(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        stack = list(function.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, (ast.Yield, ast.YieldFrom)):
                return True
            stack.extend(ast.iter_child_nodes(node))
        return False

    returns = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        function = containing_function(node)
        if function is None or not function.body or function.body[0].lineno == function.lineno:
            continue
        if is_generator(function) or isinstance(node.value, (ast.Name, ast.Constant)):
            continue
        returns.append(node)

    if not returns:
        return source

    lines = source.splitlines(keepends=True)
    returns.sort(key=lambda n: n.lineno, reverse=True)

    for ret_node in returns:
        s_line = ret_node.lineno - 1
        e_line = getattr(ret_node, "end_lineno", ret_node.lineno)
        if s_line >= len(lines):
            continue

        orig_line = lines[s_line]
        indent = ""
        for ch in orig_line:
            if ch in " \t":
                indent += ch
            else:
                break

        ret_text = "".join(lines[s_line:e_line]).rstrip()
        match = re.match(r"^(\s*return\s+)", ret_text)
        if not match:
            continue
        expr_text = ret_text[len(match.group(1)) :]
        if expr_text.rstrip().endswith("\\"):
            expr_text = expr_text.rstrip()[:-1].rstrip()
        if not expr_text.strip():
            continue

        assign_line = f"{indent}_eh_rtn = {expr_text}\n"
        return_line = f"{indent}return _eh_rtn\n"
        lines[s_line:e_line] = [assign_line, return_line]

    return "".join(lines)


def _is_control_flow_try(stmt: ast.Try) -> bool:
    if len(stmt.handlers) != 1:
        return False
    handler = stmt.handlers[0]
    if handler.type is None:
        return False
    if len(handler.body) > 2:
        return False
    for h_stmt in handler.body:
        if not isinstance(h_stmt, (ast.Return, ast.Assign, ast.Pass, ast.Expr)):
            return False
    if len(stmt.body) > 2:
        return False
    for b_stmt in stmt.body:
        if not isinstance(b_stmt, (ast.Expr, ast.Assign, ast.Return)):
            return False
    if stmt.orelse or stmt.finalbody:
        return False
    return True


def _is_instrumentation_hook_statement(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    return any(
        isinstance(node, ast.Attribute) and node.attr in HEAL_HOOK_NAMES
        for node in ast.walk(stmt.value.func)
    )


def _iter_instrumentable_stmts(body: list):
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if _is_instrumentation_hook_statement(stmt):
            continue
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            yield stmt
        elif isinstance(stmt, ast.If):
            yield stmt
        elif isinstance(stmt, ast.Try):
            if stmt.handlers:
                continue
            if stmt.body and stmt.body[0].lineno == stmt.lineno:
                yield stmt
            else:
                yield from _iter_instrumentable_stmts(stmt.body)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            yield stmt
        elif isinstance(stmt, ast.Return) and isinstance(getattr(stmt, "value", None), ast.Name) and stmt.value.id == "_eh_rtn":
            continue
        elif isinstance(stmt, ast.Raise):
            continue
        else:
            yield stmt


def _build_import_helper(file_path: str, host_port: int | str = 19876) -> str:
    metadata = {
        "file": file_path,
        "line": 0,
        "block_index": -1,
        "statement": "",
        "port": host_port,
        "token": "__EH_TOKEN__",
        "record_path": REPAIR_RECORD_PATH_MARKER,
    }
    return f'''def _eh_import_heal(_eh_err, _eh_stmt):
    _eh_metadata = {json.dumps(metadata, ensure_ascii=False)}
    _eh_metadata["statement"] = _eh_stmt
    __import__('builtins')._eh_handle_exception(
        _eh_err, _eh_metadata, __import__('sys')._getframe(1)
    )
'''


def _instrument_heal_points(
    source: str, file_path: str, host_port: int | str = 19876,
    input_source_map: dict[int, int] | None = None,
) -> tuple[str, int, dict[int, int]]:
    pre_split_lines = source.splitlines()
    n_pre_split = len(pre_split_lines)

    source = _split_return_stmts(source)

    post_split_lines = source.splitlines()
    n_post_split = len(post_split_lines)
    rtn_map: dict[int, int] = {}
    if n_post_split != n_pre_split:
        pi = 0
        for qi in range(n_post_split):
            if pi >= n_pre_split:
                rtn_map[qi + 1] = n_pre_split
                continue
            rtn_map[qi + 1] = pi + 1
            post_content = post_split_lines[qi].strip()
            if post_content == "return _eh_rtn":
                pass
            elif post_content.startswith("_eh_rtn ="):
                pi += 1
            else:
                pi += 1
    else:
        for i in range(n_post_split):
            rtn_map[i + 1] = i + 1

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, 0, {}

    insertions = []
    func_count = 0
    source_lines = source.splitlines()

    def _is_generator(func_node):
        def _has_yield(stmts):
            for stmt in stmts:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(stmt, (ast.Expr,)) and isinstance(stmt.value, (ast.Yield, ast.YieldFrom)):
                    return True
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, (ast.Yield, ast.YieldFrom)):
                    return True
                for child in ast.iter_child_nodes(stmt):
                    if isinstance(child, (ast.Yield, ast.YieldFrom)):
                        return True
                for attr in ("body", "orelse", "finalbody", "handlers"):
                    sub = getattr(stmt, attr, None)
                    if isinstance(sub, list) and sub:
                        if _has_yield(sub):
                            return True
            return False
        return _has_yield(func_node.body)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not hasattr(node, "end_lineno") or not node.body:
            continue
        if node.body[0].lineno == node.lineno:
            continue
        if node.name.startswith("__") and node.name.endswith("__"):
            continue
        if _is_generator(node):
            continue

        block_idx = 0
        has_instrumented = False
        for stmt in _iter_instrumentable_stmts(node.body):
            if not hasattr(stmt, "lineno") or not hasattr(stmt, "end_lineno"):
                continue
            s_start = stmt.lineno - 1
            s_end = stmt.end_lineno
            if s_start >= len(source_lines):
                continue
            statement_prefix = source_lines[s_start][:stmt.col_offset]
            if statement_prefix.strip():
                continue
            insertions.append((s_start, s_end, file_path, block_idx))
            block_idx += 1
            has_instrumented = True
        if has_instrumented:
            func_count += 1

    if func_count == 0:
        return source, 0, {}

    _seen_starts: dict[int, int] = {}
    _deduped: list[tuple] = []
    for idx, ins in enumerate(insertions):
        s_start = ins[0]
        if s_start in _seen_starts:
            prev_idx = _seen_starts[s_start]
            if ins[1] > _deduped[prev_idx][1]:
                _deduped[prev_idx] = ins
        else:
            _seen_starts[s_start] = len(_deduped)
            _deduped.append(ins)
    insertions = _deduped

    insertions.sort(key=lambda x: x[0], reverse=True)
    mut_insertions = [[s, e, fp, bi] for s, e, fp, bi in insertions]
    lines = source.splitlines(keepends=True)

    splice_records: list[tuple[int, int, int]] = []

    for i, ins in enumerate(mut_insertions):
        s_start, s_end, fpath, bidx = ins
        if s_start >= len(lines):
            continue
        first_line = lines[s_start]
        first_stripped = first_line.lstrip()
        if first_stripped.startswith(("else:", "else ", "elif ", "except ", "except:", "finally:")):
            continue
        indent = ""
        for ch in first_line:
            if ch in " \t":
                indent += ch
            else:
                break

        stmt_source = "".join(lines[s_start:s_end]).rstrip()

        unit = _indent_unit(indent)
        try_line = f"{indent}try:\n"
        orig_lines = lines[s_start:s_end]
        indented_orig = []
        for ol in orig_lines:
            sub_lines = ol.split("\n")
            indented_subs = []
            for sl in sub_lines:
                if sl.strip():
                    indented_subs.append(unit + sl)
                else:
                    indented_subs.append(sl)
            indented_orig.append("\n".join(indented_subs))
        if indented_orig and not indented_orig[-1].endswith("\n"):
            indented_orig[-1] += "\n"
        static_line = rtn_map.get(s_start + 1, s_start + 1)
        source_line = input_source_map.get(static_line, static_line) if input_source_map else static_line
        except_block = _build_except_block(fpath, source_line, bidx, indent, stmt_source, host_port)

        replacement = [try_line] + indented_orig + [except_block]
        old_len = s_end - s_start
        new_len = len(replacement)

        except_n_lines = except_block.count("\n")
        splice_records.append((s_start, old_len, except_n_lines))

        lines[s_start:s_end] = replacement

        delta = new_len - old_len
        if delta != 0:
            for j in range(i + 1, len(mut_insertions)):
                other = mut_insertions[j]
                if other[1] > s_start:
                    other[1] += delta

    splice_records.reverse()
    source_map: dict[int, int] = {}
    n_orig_lines = len(source.splitlines())
    output_line = 1
    cursor = 0
    for splice_start, old_len, except_line_count in splice_records:
        for original_index in range(cursor, splice_start):
            source_map[output_line] = original_index + 1
            output_line += 1
        source_map[output_line] = splice_start + 1
        output_line += 1
        for offset in range(old_len):
            source_map[output_line] = splice_start + offset + 1
            output_line += 1
        for _ in range(except_line_count):
            source_map[output_line] = splice_start + 1
            output_line += 1
        cursor = splice_start + old_len
    for original_index in range(cursor, n_orig_lines):
        source_map[output_line] = original_index + 1
        output_line += 1

    if n_post_split != n_pre_split:
        source_map = {line: rtn_map.get(original, original) for line, original in source_map.items()}

    import_nodes = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if import_nodes:
        helper_text = _build_import_helper(file_path, host_port)
        helper_lines = helper_text.splitlines(keepends=True)
        if helper_lines and not helper_lines[-1].endswith("\n"):
            helper_lines[-1] += "\n"
        helper_n = len(helper_lines)

        current_source = "".join(lines)
        lines = current_source.splitlines(keepends=True)
        try:
            current_tree = ast.parse(current_source)
        except SyntaxError:
            current_tree = None

        if current_tree is not None:
            current_imports = [
                node for node in current_tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
                and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
            ]
            _seen_import_lines: dict[int, object] = {}
            for node in current_imports:
                ln = node.lineno
                end_ln = getattr(node, "end_lineno", node.lineno)
                prev = _seen_import_lines.get(ln)
                if prev is None or end_ln > getattr(prev, "end_lineno", prev.lineno):
                    _seen_import_lines[ln] = node
            current_imports = list(_seen_import_lines.values())
            for node in sorted(current_imports, key=lambda n: n.lineno, reverse=True):
                s_start = node.lineno - 1
                s_end = getattr(node, "end_lineno", node.lineno)
                import_stmt_text = "".join(lines[s_start:s_end]).strip()
                escaped_stmt = json.dumps(import_stmt_text)

                orig_lines = lines[s_start:s_end]
                indented_orig = ["    " + ol if ol.strip() else ol for ol in orig_lines]
                wrapper = (
                    ["try:\n"]
                    + indented_orig
                    + ["except (ImportError, ModuleNotFoundError) as _eh_err:\n"]
                    + [f"    _eh_import_heal(_eh_err, {escaped_stmt})\n"]
                )
                lines[s_start:s_end] = wrapper

            insert_pos = 0
            for idx, line in enumerate(lines):
                if line.strip().startswith("from __future__"):
                    insert_pos = idx + 1
            lines = lines[:insert_pos] + helper_lines + ["\n"] + lines[insert_pos:]

            import_ranges = sorted(
                [(node.lineno, getattr(node, "end_lineno", node.lineno))
                 for node in current_imports]
            )
            new_map = {}
            for k, v in source_map.items():
                shift = 0
                for imp_start, imp_end in import_ranges:
                    if k > imp_end:
                        shift += 3
                    elif k >= imp_start:
                        shift += 1
                        break
                    else:
                        break
                new_map[k + shift] = v
            source_map = new_map
            prepend_n = helper_n + 1
            source_map = {
                (k + prepend_n if k > insert_pos else k): v
                for k, v in source_map.items()
            }

    instrumented = "".join(lines)
    line_count = len(instrumented.splitlines())
    if source_map:
        next_original = None
        next_by_line = {}
        for line_number in range(line_count, 0, -1):
            if line_number in source_map:
                next_original = source_map[line_number]
            next_by_line[line_number] = next_original
        previous_original = None
        for line_number in range(1, line_count + 1):
            if line_number in source_map:
                previous_original = source_map[line_number]
                continue
            source_map[line_number] = previous_original or next_by_line[line_number] or 1
    return instrumented, func_count, source_map


def _instrument_key_points(
    source_text: str,
    key_points_by_line: dict,
    relative_file: str,
    sinks: list[dict] | None = None,
) -> tuple[str, list[dict], list[dict]]:
    source = source_text.encode("utf-8")
    tree = get_parser("python").parse(source)
    statement_nodes = {
        (node.start_point.row + 1, node.start_point.column, node.type): node
        for node in walk_tree(tree.root_node)
    }
    call_nodes = {
        (
            node.start_point.row + 1, node.start_point.column,
            node.end_point.row + 1, node.end_point.column,
        ): node
        for node in walk_tree(tree.root_node)
        if node.type == "call"
    }
    insertions: dict[int, list[dict]] = {}
    replacements: dict[tuple[int, int], list[dict]] = {}
    locations = []
    gaps = []
    if sinks:
        prepared_sinks, sink_locations, sink_gaps = SinkPointInstrumenter().prepare(
            source_text, sinks, relative_file,
        )
        gaps.extend({**gap, "instrumentation_kind": "sink"} for gap in sink_gaps)
        locations.extend(sink_locations)
        for item in prepared_sinks:
            replacements.setdefault(
                (item["call"].start_byte, item["call"].end_byte), []
            ).append({
                **item["metadata"],
                "sink_boundary": True,
                "function_range": (item["function"].start_byte, item["function"].end_byte),
                "arguments_range": (item["arguments"].start_byte + 1, item["arguments"].end_byte - 1),
            })
    function_nodes = {
        (node.start_point.row + 1, node.start_point.column): node
        for node in walk_tree(tree.root_node)
        if node.type == "function_definition"
    }
    for line, line_points in key_points_by_line.items():
        for point in line_points:
            point_line = int(line)
            inline_clause_body = False
            if point.get("node_type") == "synthetic":
                if "function_entry" not in point.get("key_point_kinds", []):
                    continue
                function = function_nodes.get((point_line, int(point["column"])))
                body = function.child_by_field_name("body") if function is not None else None
                body_children = list(body.named_children) if body is not None else []
                docstring = None
                if body_children and body_children[0].type in {"expression_statement", "string"}:
                    first_expression = body_children[0].named_children[0] if body_children[0].type == "expression_statement" and body_children[0].named_children else body_children[0]
                    if first_expression.type in {"string", "concatenated_string"}:
                        docstring = body_children[0]
                        body_children = body_children[1:]
                body_children = [child for child in body_children if child.type != "comment"]
                node = body_children[0] if body_children else None
                if node is None and docstring is not None:
                    line_start = source.rfind(b"\n", 0, docstring.start_byte) + 1
                    inline_docstring = bool(source[line_start:docstring.start_byte].strip())
                    if inline_docstring:
                        offset = docstring.end_byte
                        indentation = b"; "
                    else:
                        offset = source.find(b"\n", docstring.end_byte)
                        offset = len(source) if offset < 0 else offset + 1
                        indentation = b" " * (function.start_point.column + 4)
                    metadata = {
                        "point_id": point["point_id"], "file": relative_file,
                        "function": point["function"], "line": point_line,
                        "column": point["column"], "node_type": point["node_type"],
                        "key_point_level": 1, "key_point_kinds": point.get("key_point_kinds", []),
                    }
                    prefix = b"" if inline_docstring or offset == 0 or source[offset - 1:offset] == b"\n" else b"\n"
                    insertions.setdefault(offset, []).append({
                        "indentation": prefix + indentation,
                        "metadata": metadata,
                        "inline": inline_docstring,
                    })
                    locations.append(metadata)
                    continue
                if node is None:
                    gaps.append({"reason": "function_entry_not_found", "point_id": point.get("point_id"), "file": relative_file})
                    continue
                inline_clause_body = bool(source[source.rfind(b"\n", 0, node.start_byte) + 1:node.start_byte].strip())
            else:
                key = (point_line, int(point["column"]), str(point["node_type"]))
                node = call_nodes.get((
                    point_line, int(point["column"]),
                    int(point["end_line"]), int(point["end_column"]),
                )) if point.get("node_type") == "call" else statement_nodes.get(key)
            if node is None:
                gaps.append({"reason": "path_key_point_not_found", "point_id": point.get("point_id"), "file": relative_file})
                continue
            if point.get("node_type") == "call":
                if "call_before" in point.get("key_point_kinds", []):
                    call_start = node.start_byte
                    while source[call_start:call_start + 1] == b"*":
                        call_start += 1
                    call_key = (call_start, node.end_byte)
                    ancestor = node.parent
                    inside_f_string = False
                    while ancestor is not None:
                        if ancestor.type == "interpolation":
                            inside_f_string = True
                            break
                        ancestor = ancestor.parent
                    replacements.setdefault(call_key, []).extend([
                        {
                            "point_id": point["point_id"],
                            "call_boundary": "before",
                            "encode_point_id": inside_f_string,
                        },
                        {
                            "point_id": point["point_id"].removesuffix(":before") + ":after",
                            "call_boundary": "after",
                            "encode_point_id": inside_f_string,
                        },
                    ])
                locations.append({
                    "point_id": point["point_id"], "file": relative_file,
                    "function": point["function"], "line": point_line,
                    "column": point["column"], "node_type": "call",
                    "key_point_level": 1, "key_point_kinds": point.get("key_point_kinds", []),
                })
                continue
            is_except_decision = node.type == "except_clause" and point.get("key_point_level", 1) == 2
            if point.get("control_position") == "handler_entry":
                body = node.child_by_field_name("body") or next((child for child in node.named_children if child.type == "block"), None)
                body_children = [child for child in body.named_children if child.type != "comment"] if body is not None else []
                node = body_children[0] if body_children else None
                if node is None:
                    gaps.append({"reason": "except_body_not_found", "point_id": point.get("point_id"), "file": relative_file})
                    continue
                inline_clause_body = bool(source[source.rfind(b"\n", 0, node.start_byte) + 1:node.start_byte].strip())
            if node.type in {"elif_clause", "else_clause", "except_clause", "finally_clause", "case_clause"} and point.get("key_point_level", 1) == 1 and not point.get("control_position"):
                body = node.child_by_field_name("consequence") or node.child_by_field_name("body")
                if body is None:
                    body = next((child for child in node.named_children if child.type == "block"), None)
                body_children = [child for child in body.named_children if child.type != "comment"] if body is not None else []
                node = body_children[0] if body_children else None
                if node is None:
                    gaps.append({"reason": "clause_body_not_found", "point_id": point.get("point_id"), "file": relative_file})
                    continue
                inline_clause_body = bool(source[source.rfind(b"\n", 0, node.start_byte) + 1:node.start_byte].strip())
            if is_except_decision:
                body = node.child_by_field_name("body") or next((child for child in node.named_children if child.type == "block"), None)
                body_children = [child for child in body.named_children if child.type != "comment"] if body is not None else []
                node = body_children[0] if body_children else None
                if node is None:
                    gaps.append({"reason": "except_body_not_found", "point_id": point.get("point_id"), "file": relative_file})
                    continue
                inline_clause_body = bool(source[source.rfind(b"\n", 0, node.start_byte) + 1:node.start_byte].strip())
            if point.get("control_position") in {"entry", "exit"}:
                decision = node.child_by_field_name({
                    "if_statement": "condition", "elif_clause": "condition", "while_statement": "condition",
                    "for_statement": "right", "match_statement": "subject",
                }.get(node.type, ""))
                if decision is None:
                    line_start = source.rfind(b"\n", 0, node.start_byte) + 1
                    body = node.child_by_field_name("body") or node.child_by_field_name("consequence")
                    if body is None:
                        body = next((child for child in node.named_children if child.type == "block"), None)
                    body_children = [child for child in body.named_children if child.type != "comment"] if body is not None else []
                    body_node = body_children[0] if body_children else None
                    if point["control_position"] == "entry" and body_node is not None:
                        line_start = source.rfind(b"\n", 0, body_node.start_byte) + 1
                        prefix = source[line_start:body_node.start_byte]
                        inline = bool(prefix.strip())
                        offset = body_node.start_byte if inline else line_start
                        indentation = b"" if inline else prefix
                        metadata = {
                            "point_id": point["point_id"], "file": relative_file,
                            "function": point["function"], "line": point_line,
                            "column": point["column"], "node_type": point["node_type"],
                            "key_point_level": 1, "key_point_kinds": point.get("key_point_kinds", []),
                        }
                        insertions.setdefault(offset, []).append({
                            "indentation": indentation, "metadata": metadata, "inline": inline,
                        })
                        locations.append(metadata)
                    continue
                metadata = {
                    "point_id": point["point_id"], "file": relative_file, "function": point["function"],
                    "line": point_line, "column": point["column"], "node_type": point["node_type"],
                    "key_point_level": 1, "key_point_kinds": point.get("key_point_kinds", []),
                    "control_boundary": point["control_position"],
                    "control_start_line": point_line,
                    "control_end_line": int(point.get("end_line", point_line)),
                    "control_has_alternative": bool(point.get("has_alternative")),
                    "control_is_loop": point["node_type"] in {"while_statement", "for_statement"},
                }
                replacements.setdefault((decision.start_byte, decision.end_byte), []).append(metadata)
                locations.append(metadata)
                continue
            line_start = source.rfind(b"\n", 0, node.start_byte) + 1
            prefix = source[line_start:node.start_byte]
            inline_iteration = point.get("control_position") == "iteration" and bool(prefix.strip())
            inline_control_exit = "control_exit" in point.get("key_point_kinds", []) and bool(prefix.strip())
            if prefix.strip() and not inline_clause_body and not inline_iteration and not inline_control_exit:
                gaps.append({"reason": "path_key_point_not_statement_aligned", "point_id": point.get("point_id"), "file": relative_file})
                continue
            metadata = {
                "point_id": point["point_id"], "file": relative_file, "function": point["function"],
                "line": point_line, "column": point["column"], "node_type": point["node_type"],
                "key_point_level": point.get("key_point_level", 1),
                "control_position": point.get("control_position"),
                "decision_expression": point.get("decision_expression"),
                "decision_mode": point.get("decision_mode"),
                "decision_names": point.get("decision_names", []),
                "definitions": point.get("definitions", []), "uses": point.get("uses", []),
                "key_point_kinds": point.get("key_point_kinds", []),
            }
            indentation = source[line_start:node.start_byte]
            if metadata["key_point_level"] == 2 and (metadata.get("decision_expression") or is_except_decision):
                if is_except_decision:
                    offset = node.start_byte if inline_clause_body else line_start
                    insertions.setdefault(offset, []).append({
                        "indentation": b"" if inline_clause_body else indentation,
                        "metadata": metadata,
                        "inline": inline_clause_body,
                    })
                else:
                    decision = node.child_by_field_name({
                        "if_statement": "condition", "elif_clause": "condition", "while_statement": "condition",
                        "for_statement": "right",
                    }.get(node.type, ""))
                    if decision is None:
                        gaps.append({"reason": "decision_expression_not_found", "point_id": point.get("point_id"), "file": relative_file})
                        continue
                    replacements.setdefault((decision.start_byte, decision.end_byte), []).append(metadata)
            else:
                inline_probe = inline_clause_body or inline_iteration or inline_control_exit
                offset = node.start_byte if inline_probe else line_start
                insertions.setdefault(offset, []).append({
                    "indentation": b"" if inline_probe else indentation,
                    "metadata": metadata,
                    "inline": inline_probe,
                })
            locations.append(metadata)
    edits = []

    def string_expression(value: str) -> str:
        encoded = tuple(value.encode("utf-8"))
        return f"bytes({encoded}).decode()"

    def path_hit_expression(point_id: str, encode: bool) -> str:
        if not encode:
            return (
                f"__import__('builtins')._eh_path_hit("
                f"{json.dumps(point_id, ensure_ascii=False)})"
            )
        return (
            f"__import__({string_expression('builtins')})._eh_path_hit("
            f"{string_expression(point_id)})"
        )

    def wrap_expression(expression: str, metadata_items: list[dict]) -> str:
        metadata_items.sort(key=lambda item: item.get("point_id", item.get("sink_id", "")))
        sink_items = [item for item in metadata_items if item.get("sink_boundary")]
        call_items = [item for item in metadata_items if item.get("call_boundary")]
        control_items = [item for item in metadata_items if item.get("control_boundary")]
        decision_items = [
            item for item in metadata_items
            if not item.get("sink_boundary")
            and not item.get("call_boundary")
            and not item.get("control_boundary")
        ]
        wrapped = expression
        if sink_items:
            if len(sink_items) != 1:
                gaps.append({
                    "reason": "sink_boundary_not_unique", "file": relative_file,
                    "sink_ids": [item.get("sink_id") for item in sink_items],
                })
            else:
                sink = sink_items[0]
                function_start, function_end = sink["function_range"]
                arguments_start, arguments_end = sink["arguments_range"]
                function_text = render_range(function_start, function_end)
                arguments_text = render_range(arguments_start, arguments_end)
                metadata = {
                    key: value for key, value in sink.items()
                    if key not in {"sink_boundary", "function_range", "arguments_range"}
                }
                wrapped = (
                    f"__import__('builtins')._eh_guard_sink_call("
                    f"{json.dumps(metadata, ensure_ascii=False)}, {function_text}, locals(), {arguments_text})"
                )
        if call_items:
            before = next((item for item in call_items if item["call_boundary"] == "before"), None)
            after = next((item for item in call_items if item["call_boundary"] == "after"), None)
            if before is None or after is None:
                gaps.append({
                    "reason": "call_boundary_pair_missing", "file": relative_file,
                    "point_ids": [item["point_id"] for item in call_items],
                })
            else:
                wrapped = (
                    f"({path_hit_expression(before['point_id'], before['encode_point_id'])}, "
                    f"({wrapped}), "
                    f"{path_hit_expression(after['point_id'], after['encode_point_id'])})[1]"
                )
        if control_items:
            entry = next((item for item in control_items if item["control_boundary"] == "entry"), None)
            exit_item = next((item for item in control_items if item["control_boundary"] == "exit"), None)
            if entry is None or exit_item is None:
                gaps.append({
                    "reason": "control_boundary_pair_missing", "file": relative_file,
                    "point_ids": [item["point_id"] for item in control_items],
                })
            else:
                wrapped = (
                    f"__import__('builtins')._eh_path_control("
                    f"{json.dumps(entry['point_id'], ensure_ascii=False)}, "
                    f"{json.dumps(exit_item['point_id'], ensure_ascii=False)}, "
                    f"{entry['control_start_line']}, {entry['control_end_line']}, "
                    f"{entry['control_has_alternative']!r}, ({wrapped}), {entry['control_is_loop']!r})"
                )
        for metadata in reversed(decision_items):
            wrapped = (
                f"__import__('builtins')._eh_path_decision("
                f"{json.dumps(metadata['point_id'], ensure_ascii=False)}, "
                f"None, ({wrapped}), {json.dumps(metadata['decision_mode'], ensure_ascii=False)}, [])"
            )
        return wrapped

    def render_range(start: int, end: int) -> str:
        children = [
            (child_start, child_end) for child_start, child_end in replacements
            if start <= child_start and child_end <= end and (child_start, child_end) != (start, end)
            and not any(
                start <= outer_start <= child_start and child_end <= outer_end <= end
                and (outer_start, outer_end) not in {(start, end), (child_start, child_end)}
                for outer_start, outer_end in replacements
            )
        ]
        rendered = bytearray()
        cursor = start
        for child_start, child_end in sorted(children):
            rendered.extend(source[cursor:child_start])
            rendered.extend(render_range(child_start, child_end).encode("utf-8"))
            cursor = child_end
        rendered.extend(source[cursor:end])
        original = rendered.decode("utf-8", errors="replace")
        wrapped = wrap_expression(original, replacements.get((start, end), []))
        if wrapped != original and original.lstrip().startswith("(") and original.rstrip().endswith(")"):
            wrapped = f"({wrapped})"
        if (
            wrapped != original
            and start > 0
            and chr(source[start - 1]).isalnum()
            and wrapped[:1] in "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ):
            wrapped = " " + wrapped
        return wrapped

    outer_ranges = [
        (start, end) for start, end in replacements
        if not any(
            outer_start <= start and end <= outer_end and (outer_start, outer_end) != (start, end)
            for outer_start, outer_end in replacements
        )
    ]
    for start, end in outer_ranges:
        edits.append((start, end, render_range(start, end).encode("utf-8")))
    for offset, probes in insertions.items():
        probes.sort(key=lambda probe: ("function_entry" not in probe["metadata"].get("key_point_kinds", []), probe["metadata"]["point_id"]))
        before = b"".join(
            probe["indentation"] + (
                (
                    f"__import__('builtins')._eh_path_exception("
                    f"{json.dumps(probe['metadata']['point_id'], ensure_ascii=False)}, "
                    f"{json.dumps(probe['metadata']['decision_names'], ensure_ascii=False)})"
                )
                if probe["metadata"]["key_point_level"] == 2
                else f"__import__('builtins')._eh_path_hit({json.dumps(probe['metadata']['point_id'], ensure_ascii=False)})"
            ).encode("utf-8") + (b"; " if probe.get("inline") else b"\n")
            for probe in probes
        )
        edits.append((offset, offset, before))
    instrumented = source
    edit_ranges = {(start, end) for start, end, _ in edits if start != end}
    filtered_edits = []
    for start, end, replacement in edits:
        if start != end and any(
            outer_start <= start and end <= outer_end and (outer_start, outer_end) != (start, end)
            for outer_start, outer_end in edit_ranges
        ):
            continue
        filtered_edits.append((start, end, replacement))
    chunks = []
    cursor = 0
    for start, end, replacement in sorted(filtered_edits, key=lambda item: (item[0], item[1])):
        if start < cursor:
            continue
        chunks.extend((source[cursor:start], replacement))
        cursor = end
    chunks.append(source[cursor:])
    instrumented = b"".join(chunks)
    return instrumented.decode("utf-8", errors="replace"), locations, gaps

class PathKeyPointInstrumenter:
    """Insert lightweight probes at precomputed repository path key points."""

    def instrument(
        self,
        source_text: str,
        key_points_by_line: dict,
        relative_file: str,
        sinks: list[dict] | None = None,
    ) -> tuple[str, list[dict], list[dict]]:
        return _instrument_key_points(source_text, key_points_by_line, relative_file, sinks)


class HealPointInstrumenter:
    """Insert Heal try/except regions only in project production code."""

    def should_instrument(self, relative_path: Path) -> bool:
        return not is_repository_path_excluded(relative_path.parts)

    def instrument(
        self,
        source: str,
        file_path: str,
        host_port: int | str = 19876,
        input_source_map: dict[int, int] | None = None,
    ) -> tuple[str, int, dict[int, int]]:
        return _instrument_heal_points(
            source,
            file_path,
            host_port=host_port,
            input_source_map=input_source_map,
        )



def compose_source_map(
    heal_map: dict[int, int],
    static_map: dict[str, int],
) -> dict[str, int]:
    return {
        str(instrumented_line): int(static_map[str(static_line)])
        for instrumented_line, static_line in heal_map.items()
        if str(static_line) in static_map
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
    }


def _build_fingerprint(instance_id: str) -> str:
    inputs = [
        CONFIG["sink_root"] / instance_id / "combined_sinks.json",
        CONFIG["key_point_root"] / instance_id / "combined_key_points.json",
        Path(__file__),
        Path(__file__).resolve().with_name("instrument_guard.py"),
        Path(__file__).resolve().with_name("heal_builtins.py"),
        Path(__file__).resolve().with_name("heal_code_allowlist_blocklist.yaml"),
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Instrumentation fingerprint inputs missing: {missing}")
    digest = hashlib.sha256()
    for path in inputs:
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _instrument_all_unlocked(instance_id: str, force: bool) -> dict:
    case_dir = CONFIG["output_root"] / instance_id
    instrumentation_path = case_dir / "repo" / "_instrumentation.json"
    build_fingerprint = _build_fingerprint(instance_id)
    if instrumentation_path.is_file() and not force:
        try:
            instrumentation = json.loads(instrumentation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(f"Invalid resumable instrumentation for {instance_id}: {error}")
        else:
            build = instrumentation.get("build", {})
            if build.get("status") == "completed" and build.get("fingerprint") == build_fingerprint:
                return {
                    "instance_id": instance_id,
                    "status": "completed",
                    "reused": True,
                    "instrumented_files": build.get("heal_instrumented_files", 0),
                    "instrumented_functions": build.get("heal_instrumented_functions", 0),
                }
            logger.info(f"Rebuilding stale instrumentation for {instance_id}")

    sink_instrumenter = SinkPointInstrumenter()
    key_point_instrumenter = PathKeyPointInstrumenter()
    heal_instrumenter = HealPointInstrumenter()
    def instrument_heal(source: str, relative_file: str, static_map: dict[str, int]):
        return heal_instrumenter.instrument(
            source,
            relative_file,
            host_port=MARKER_PORT,
            input_source_map={int(line): original for line, original in static_map.items()},
        )

    instrumentation_result = instrument_instance(
        instance_id,
        CONFIG["source_root"],
        CONFIG["sink_root"],
        CONFIG["key_point_root"],
        CONFIG["output_root"],
        True,
        sink_instrumenter=sink_instrumenter,
        key_point_instrumenter=key_point_instrumenter,
        file_transform=instrument_heal,
    )
    if instrumentation_result["status"] != "completed":
        return instrumentation_result

    manifest_path = case_dir / "instrumentation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repository = Path(manifest["source_repository"])
    output_repo = Path(manifest["instrumented_repository"])
    instrumentation_path = output_repo / "_instrumentation.json"
    final_maps = json.loads((output_repo / "_instrumentation_source_map.json").read_text(encoding="utf-8"))
    instrumented_files = instrumentation_result["transformed_files"]
    instrumented_functions = instrumentation_result["transformed_functions"]
    heal_skipped_files = 0

    trace_manifest_path = output_repo / TRACE_MANIFEST_NAME
    trace_manifest = write_ast_trace_manifest(
        repository,
        sorted(final_maps),
        str(trace_manifest_path),
    )

    sinks_path = CONFIG["sink_root"] / instance_id / "combined_sinks.json"
    key_points_path = CONFIG["key_point_root"] / instance_id / "combined_key_points.json"
    combined_sinks = json.loads(sinks_path.read_text(encoding="utf-8"))
    key_points = json.loads(key_points_path.read_text(encoding="utf-8"))
    confirmed_sinks_path = output_repo / "_confirmed_sinks.json"
    key_points_copy_path = output_repo / "_path_key_points.json"
    guard_manifest_path = output_repo / "_guard_manifest.json"
    instrumentation_manifest_path = output_repo / "_instrumentation_manifest.json"
    confirmed_sinks_path.write_text(
        json.dumps(combined_sinks.get("confirmed_sinks", []), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    key_points_copy_path.write_text(json.dumps(key_points, ensure_ascii=False) + "\n", encoding="utf-8")
    instrumentation_manifest_path.write_bytes(manifest_path.read_bytes())
    guard_manifest = write_guard_manifest(
        repository,
        confirmed_sinks_path,
        guard_manifest_path,
    )
    if guard_manifest.get("analysis_gaps"):
        logger.warning(
            f"Guard manifest recorded {len(guard_manifest['analysis_gaps'])} analysis gaps for "
            f"{instance_id}; first 2: {guard_manifest['analysis_gaps'][:2]}"
        )

    instrumentation = {
        "instance_id": instance_id,
        "repository": manifest["repository"],
        "base_commit": manifest["base_commit"],
        "artifacts": {
            "source_map": _artifact(output_repo / "_instrumentation_source_map.json", output_repo),
            "trace_manifest": _artifact(trace_manifest_path, output_repo),
            "path_points": _artifact(key_points_copy_path, output_repo),
            "confirmed_sinks": _artifact(confirmed_sinks_path, output_repo),
            "guard_manifest": _artifact(guard_manifest_path, output_repo),
            "instrumentation_manifest": _artifact(instrumentation_manifest_path, output_repo),
        },
        "source_syntax_errors": manifest.get("source_syntax_errors", []),
        "build": {
            "status": "completed",
            "fingerprint": build_fingerprint,
            "instrumentation_order": ["confirmed_sinks", "path_key_points", "heal_try_catch"],
            "heal_instrumented_files": instrumented_files,
            "heal_instrumented_functions": instrumented_functions,
            "heal_skipped_files": heal_skipped_files,
            "heal_excluded_directories": sorted(HEAL_EXCLUDED_DIRECTORIES),
            "instrumented_sink_count": len(manifest.get("instrumented_sinks", [])),
            "instrumented_path_point_count": len(manifest.get("instrumented_path_key_points", [])),
        },
    }
    instrumentation_path.write_text(
        json.dumps(instrumentation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path.unlink(missing_ok=True)
    logger.info(
        f"All instrumentation {instance_id}: heal_files={instrumented_files}, "
        f"heal_functions={instrumented_functions}, heal_skipped_files={heal_skipped_files}, "
        f"sinks={len(manifest.get('instrumented_sinks', []))}, "
        f"path_points={len(manifest.get('instrumented_path_key_points', []))}"
    )
    return {
        "instance_id": instance_id,
        "status": "completed",
        "instrumented_files": instrumented_files,
        "instrumented_functions": instrumented_functions,
    }


def instrument_all(instance_id: str, force: bool) -> dict:
    lock_dir = CONFIG["output_root"] / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{instance_id}.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Instrumentation already running for {instance_id}") from error
        return _instrument_all_unlocked(instance_id, force)


def _run_instance(arguments: tuple[str, bool]) -> dict:
    instance_id, force = arguments
    try:
        return instrument_all(instance_id, force)
    except Exception as error:
        logger.error(
            f"Instrumentation failed for {instance_id}: {type(error).__name__}: {error}"
        )
        return {
            "instance_id": instance_id,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-build repositories with Heal, Sink, and path instrumentation")
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    instance_ids = args.instance_id or sorted(
        path.name for path in CONFIG["key_point_root"].iterdir() if path.is_dir()
    )
    instance_ids = [item for item in instance_ids if item in HEALBENCH_INSTANCE_IDS]
    if args.limit:
        instance_ids = instance_ids[:args.limit]
    logger.info(f"Selected {len(instance_ids)} instances; first 2: {instance_ids[:2]}")

    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        results = list(pool.map(_run_instance, ((instance_id, args.force) for instance_id in instance_ids)))
    failures = [result for result in results if result.get("status") != "completed"]
    logger.info(f"All instrumentation finished: total={len(results)}, failures={len(failures)}; first 2: {results[:2]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
