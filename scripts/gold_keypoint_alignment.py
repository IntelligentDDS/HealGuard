from __future__ import annotations

import ast
import re

from loguru import logger

from evaluate_results import parse_unified_diff


def align_saved_point_identities(relative, rebuilt, saved, record, line_count):
    """Retain saved identities only at unchanged, uniquely mapped AST coordinates."""
    mappings = [parse_unified_diff(record.get(name, ""), align_replacements=False)
                for name in ("test_patch", "patch")]
    original_lines, patched_lines = {}, {}
    for line in range(1, line_count + 1):
        original = line
        for mapping in mappings:
            if original is not None and relative in mapping:
                original = mapping[relative].get(original)
        original_lines[line] = original
        if original is not None:
            if original in patched_lines:
                raise ValueError(f"Ambiguous unchanged point mapping: {relative}:{original}")
            patched_lines[original] = line
    def point_key(line, point):
        return (int(line), point["column"], point["node_type"],
                point.get("key_point_level"), point.get("decision_mode"),
                tuple(sorted(point.get("key_point_kinds", []))))

    old_points = {}
    for line, points in saved.get("key_points_by_line", {}).items():
        for point in points:
            key = point_key(line, point)
            if key in old_points:
                raise ValueError(f"Duplicate saved point coordinates: {relative}:{key}")
            old_points[key] = point
    aligned = []
    for line, points in rebuilt["key_points_by_line"].items():
        original = original_lines[int(line)]
        if original is None:
            continue
        for point in points:
            old = old_points.get(point_key(original, point))
            if old is None:
                continue
            def translate(match):
                coordinate = int(match.group(2))
                if coordinate not in patched_lines:
                    raise ValueError(f"Unmapped saved point coordinate: {relative}:{coordinate}")
                return match.group(1) + str(patched_lines[coordinate]) + match.group(3)
            identity = re.sub(r"(::|@)(\d+)(:)", translate, old["point_id"])
            aligned.append({"saved_point_id": old["point_id"], "rebuilt_point_id": point["point_id"],
                            "aligned_point_id": identity})
            point["point_id"] = identity
            point["function"] = old["function"]
    ids = [point["point_id"] for points in rebuilt["key_points_by_line"].values() for point in points]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate aligned point identities: {relative}")
    return aligned


def restore_gold_docstrings(source, original, relative):
    """Restore docstrings only in the private Golden copy without shifting line numbers."""
    def scopes(tree):
        result = {}
        def visit(node, path):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if not isinstance(node, ast.Module):
                    path = (*path, node.name)
                result[path] = node
            for child in ast.iter_child_nodes(node):
                visit(child, path)
        visit(tree, ())
        return result
    try:
        original_scopes = scopes(ast.parse(original))
        current_scopes = scopes(ast.parse(source))
    except SyntaxError as error:
        if "_eh_handle_exception" not in source and "_eh_path_hit" not in source:
            logger.warning(f"Preserving uninstrumented Python-version-specific file {relative}: {error}")
            return source, []
        logger.error(f"Golden docstring check cannot parse instrumented file {relative}: {error}")
        raise
    lines = source.splitlines(keepends=True)
    edits, changes = [], []
    for path, original_node in original_scopes.items():
        doc = ast.get_docstring(original_node, clean=False)
        node = current_scopes.get(path)
        if doc is None or node is None or ast.get_docstring(node, clean=False) == doc:
            continue
        first = node.body[0]
        wrapped = next((stmt for stmt in node.body if isinstance(stmt, ast.Try) and stmt.body
                        and isinstance(stmt.body[0], ast.Expr) and isinstance(stmt.body[0].value, ast.Constant)
                        and isinstance(stmt.body[0].value.value, str)
                        and stmt.body[0].value.value.split() == doc.split()), None)
        if isinstance(node, ast.Module) and isinstance(first, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            # Prefix a literal on the same physical line; semicolon cannot precede a compound statement.
            literal = next((stmt for stmt in node.body if isinstance(stmt, ast.Expr)
                            and isinstance(stmt.value, ast.Constant) and stmt.value.value == doc), None)
            if literal is None:
                raise ValueError(f"Missing module docstring literal: {relative}")
            # Preserve physical line count by moving the injected prelude into the literal's old span.
            before = ''.join(lines[:literal.lineno - 1])
            replacement = repr(doc) + '\n' + before
            replacement += '\n' * (literal.end_lineno - len(replacement.splitlines()))
            if len(replacement.splitlines()) != literal.end_lineno:
                raise ValueError(f"Insufficient lines to preserve module coordinates: {relative}")
            edits.append((0, literal.end_lineno, 0, replacement.rstrip('\n')))
        elif wrapped is not None and first is wrapped:
            edits.append((wrapped.lineno - 1, wrapped.end_lineno, wrapped.col_offset, repr(doc)))
        else:
            raise ValueError(f"Unexpected Golden docstring layout: {relative}:{path}")
        changes.append({"scope": list(path), "line": first.lineno, "reason": "restore_original_docstring"})
    for start, end, column, text in sorted(edits, reverse=True):
        replacement = (" " * column + text + "\n").splitlines(keepends=True)
        lines[start:end] = replacement + ["\n"] * (end - start - len(replacement))
    restored = "".join(lines)
    verified = scopes(ast.parse(restored))
    for path, original_node in original_scopes.items():
        doc = ast.get_docstring(original_node, clean=False)
        if doc is not None and path in verified and ast.get_docstring(verified[path], clean=False) != doc:
            raise ValueError(f"Golden docstring restoration failed: {relative}:{path}")
    return restored, changes


def fix_gold_value_comparison(runtime):
    old = '''        try:
            differs = before_value != after_value
        except BaseException:
            differs = before_value is not after_value
        if differs:
'''
    new = '''        try:
            differs = bool(before_value != after_value)
        except BaseException as error:
            sys.stderr.write("__EH_VALUE_COMPARISON_WARNING__ %s: %s\\n" % (type(error).__name__, error))
            differs = before_value is not after_value
        if differs:
'''
    if runtime.count(old) != 1:
        raise ValueError("Golden comparison fix no longer matches the shared runtime")
    return runtime.replace(old, new, 1)
