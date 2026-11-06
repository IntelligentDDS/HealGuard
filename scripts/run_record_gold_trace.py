import argparse
import hashlib
import re
import fcntl
import ast
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agenticheal import (
    load_dataset_records, _resolve_docker_image, _ensure_docker_images,
)
from util import load_instance_file, build_instr_prep_cmd
from util import HEALBENCH_INSTANCE_IDS
from heal_repository_scope import iter_repository_python_files
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "gold_traces"
DOCKER_TIMEOUT = 1500


_EH_REC_SRC = (
    "def _eh_rec(kind, key, vals=''):\n"
    "    import builtins as _b\n"
    "    if getattr(_b, '_eh_active', False): return\n"
    "    _b._eh_active = True\n"
    "    try:\n"
    "        import sys as _s\n"
    "        if not hasattr(_b, '_eh_seq'): _b._eh_seq = [0]\n"
    "        n = _b._eh_seq[0]; _b._eh_seq[0] = n + 1\n"
    "        v = vals() if callable(vals) else vals\n"
    "        v = str(v).replace('\\t',' ').replace('\\n',' ')\n"
    "        _s.stderr.write('__EH__\\t' + kind + '\\t' + str(n) + '\\t' + key + '\\t' + v + '\\n')\n"
    "    except BaseException as _e:\n"
    "        _s.stderr.write('__EH_REC_FAIL__ ' + type(_e).__name__ + ': ' + str(_e) + '\\n')\n"
    "    finally:\n"
    "        _b._eh_active = False\n"
    "def _eh_v(x):\n"
    "    try:\n"
    "        return repr(x)\n"
    "    except BaseException as _e:\n"
    "        return '<repr-error ' + type(_e).__name__ + ': ' + str(_e) + '>'\n"
    "def _eh_stack(co_name=''):\n"
    "    try:\n"
    "        import sys as _s2\n"
    "        f = _s2._getframe(1)\n"
    "        depth = 0; rec = 0\n"
    "        while f is not None and depth < 1000:\n"
    "            depth += 1\n"
    "            try:\n"
    "                if co_name and f.f_code.co_name == co_name: rec += 1\n"
    "            except Exception as _e:\n"
    "                _s2.stderr.write('__EH_STACK_FRAME_FAIL__ ' + type(_e).__name__ + ': ' + str(_e) + '\\n')\n"
    "            f = f.f_back\n"
    "        return 'depth=' + str(depth) + ' rec=' + str(rec)\n"
    "    except Exception as _e:\n"
    "        _s2.stderr.write('__EH_STACK_FAIL__ ' + type(_e).__name__ + ': ' + str(_e) + '\\n')\n"
    "        return ''\n"
    "import builtins as _bb\n"
    "_bb._eh_rec = _eh_rec\n"
    "_bb._eh_v = _eh_v\n"
    "_bb._eh_stack = _eh_stack\n"
    "_bb._eh_seen = set()\n"
)


def build_eh_setup_cmd() -> str:
    import base64 as _b64
    rec_b64 = _b64.b64encode(_EH_REC_SRC.encode("utf-8")).decode("ascii")
    pth_line = f"import base64,builtins; exec(base64.b64decode('{rec_b64}').decode('utf-8'))\n"
    pth_b64 = _b64.b64encode(pth_line.encode("utf-8")).decode("ascii")
    writer = (
        "import site,os,base64,sys\n"
        "_d=base64.b64decode('" + pth_b64 + "')\n"
        "cands=[]\n"
        "try:\n"
        "    cands+=list(site.getsitepackages())\n"
        "except Exception as error:\n"
        "    sys.stderr.write('__EH_SETUP_SITE_FAIL__ %s: %s\\n' % (type(error).__name__, error))\n"
        "try:\n"
        "    u=site.getusersitepackages()\n"
        "    if u: cands.append(u)\n"
        "except Exception as error:\n"
        "    sys.stderr.write('__EH_SETUP_USER_SITE_FAIL__ %s: %s\\n' % (type(error).__name__, error))\n"
        "cands+=[p for p in sys.path if 'site-packages' in p]\n"
        "wrote=False\n"
        "for c in cands:\n"
        "    try:\n"
        "        os.makedirs(c, exist_ok=True)\n"
        "        open(os.path.join(c,'_eh_seen.pth'),'wb').write(_d)\n"
        "        wrote=True; break\n"
        "    except Exception as error:\n"
        "        sys.stderr.write('__EH_SETUP_WRITE_FAIL__ %s: %s\\n' % (type(error).__name__, error))\n"
        "if not wrote: sys.stderr.write('__EH_SETUP_FAIL__ no writable site dir\\n')\n"
    )
    writer_b64 = _b64.b64encode(writer.encode("utf-8")).decode("ascii")
    from heal_builtins import build_builtins_setup_cmd
    marker_setup = (
        "python -c 'import base64; "
        f'exec(base64.b64decode("{writer_b64}").decode("utf-8"))\' && '
    )
    return marker_setup + build_builtins_setup_cmd()




_SKIP_INSTRUMENT_METHODS = {
    "__getattr__", "__getattribute__", "__setattr__", "__delattr__",
    "__repr__", "__str__", "__hash__",
    "__getstate__", "__setstate__", "__reduce__", "__reduce_ex__",
    "__getnewargs__", "__getnewargs_ex__", "__deepcopy__", "__copy__",
}


def _marker_key(file_path: str, func_name: str, branch_type: str, decision_line: int, source_text: str) -> str:
    src = source_text.strip()
    return f"{file_path}\t{func_name}\t{branch_type}\t{decision_line}\t{src}"


def build_insert_only_source_map(original_source: str, instrumented_source: str) -> dict[int, int]:
    original_line = 0
    source_map = {}
    for instrumented_line, content in enumerate(instrumented_source.splitlines(), 1):
        if "__import__('builtins')._eh_rec(" in content:
            continue
        original_line += 1
        source_map[instrumented_line] = original_line
    if original_line != len(original_source.splitlines()):
        raise ValueError(
            f"source map mismatch: mapped={original_line}, original={len(original_source.splitlines())}"
        )
    return source_map


def instrument_markers(source: str, file_path: str) -> tuple[str, list[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as _e:
        logger.warning(f"instrument_markers: source has SyntaxError, skipping: {_e}")
        return source, []

    source_lines = source.splitlines()
    markers: list[str] = []
    insertions: list[tuple[int, str]] = []

    def _walk_with_class(node, class_name=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                nested = f"{class_name}.{child.name}" if class_name else child.name
                _walk_with_class(child, class_name=nested)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fname = f"{class_name}.{child.name}" if class_name else child.name
                if child.name not in _SKIP_INSTRUMENT_METHODS:
                    _mk_add_fn_enter(child, file_path, fname, source_lines, markers, insertions)
                    _mk_visit_body(child.body, file_path, fname, source_lines, markers, insertions)
                _walk_with_class(child, class_name="")
            else:
                _walk_with_class(child, class_name=class_name)

    _walk_with_class(tree)

    decorator_anchors = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.decorator_list:
            decorator_anchors[node.lineno - 1] = min(decorator.lineno for decorator in node.decorator_list) - 1
    insertions = [
        (decorator_anchors.get(lineno_0, lineno_0), statement)
        for lineno_0, statement in insertions
    ]

    def _inline_suite(lineno_0):
        if not 0 <= lineno_0 < len(source_lines):
            return False
        stripped = source_lines[lineno_0].strip()
        if ":" not in stripped:
            return False
        head, tail = stripped.split(":", 1)
        compound = (
            "if ", "elif ", "else", "for ", "async for ", "while ", "try", "except",
            "finally", "with ", "async with ", "def ", "async def ", "class ", "match ", "case ",
        )
        return tail.strip() != "" and head.startswith(compound)

    safe_pairs = [
        (marker, insertion)
        for marker, insertion in zip(markers, insertions)
        if not _inline_suite(insertion[0])
    ]
    markers = [marker for marker, _ in safe_pairs]
    insertions = [insertion for _, insertion in safe_pairs]

    if not insertions:
        return source, []

    indexed = list(enumerate(insertions))
    indexed.sort(key=lambda t: (t[1][0], t[0]), reverse=True)
    lines = source.splitlines(keepends=True)
    for _idx, (lineno_0, call_str) in indexed:
        if lineno_0 >= len(lines):
            continue
        target_line = lines[lineno_0]
        indent = ""
        for ch in target_line:
            if ch in (" ", "\t"):
                indent += ch
            else:
                break
        lines.insert(lineno_0, f"{indent}{call_str}\n")

    result = "".join(lines)
    try:
        ast.parse(result)
        return result, markers
    except SyntaxError as _e:
        logger.warning(
            f"instrument_markers: full insert had SyntaxError ({_e}); re-applying markers individually"
        )

    base = source.splitlines(keepends=True)

    def _apply(subset):
        out = list(base)
        for _ai, (lineno_0, call_str) in subset:
            if lineno_0 >= len(out):
                continue
            tl = out[lineno_0]
            ind = ""
            for ch in tl:
                if ch in (" ", "\t"):
                    ind += ch
                else:
                    break
            out.insert(lineno_0, f"{ind}{call_str}\n")
        return "".join(out)

    kept_idx: set = set()
    stack = [list(indexed)]
    while stack:
        group = stack.pop()
        if not group:
            continue
        try:
            ast.parse(_apply(group))
            kept_idx.update(ai for ai, _ in group)
            continue
        except SyntaxError as error:
            logger.warning(f"instrument_markers: splitting invalid marker group: {error}")
        if len(group) == 1:
            continue
        mid = len(group) // 2
        stack.append(group[:mid])
        stack.append(group[mid:])

    final_subset = [item for item in indexed if item[0] in kept_idx]
    result = _apply(final_subset)
    try:
        ast.parse(result)
    except SyntaxError as error:
        logger.warning(f"instrument_markers: combined marker groups invalid: {error}")
        cur = list(base)
        kept_idx = set()
        for append_idx, (lineno_0, call_str) in final_subset:
            if lineno_0 >= len(cur):
                continue
            tl = cur[lineno_0]
            ind = ""
            for ch in tl:
                if ch in (" ", "\t"):
                    ind += ch
                else:
                    break
            cur.insert(lineno_0, f"{ind}{call_str}\n")
            try:
                ast.parse("".join(cur))
                kept_idx.add(append_idx)
            except SyntaxError as error:
                logger.warning(f"instrument_markers: rejected marker at {lineno_0 + 1}: {error}")
                cur.pop(lineno_0)
        result = "".join(cur)
        try:
            ast.parse(result)
        except SyntaxError as _e3:
            logger.warning(f"instrument_markers: per-marker fallback still invalid ({_e3}), reverting file")
            return source, []
    dropped_idxs = {ai for ai, _ in indexed} - kept_idx
    kept_markers = [k for i, k in enumerate(markers) if i not in dropped_idxs]
    dropped_markers = [k for i, k in enumerate(markers) if i in dropped_idxs]
    logger.warning(
        f"instrument_markers: kept {len(kept_markers)}/{len(markers)} markers after re-apply; "
        f"dropped={dropped_markers}"
    )
    return result, kept_markers


_TRYSTAR = getattr(ast, "TryStar", None)


def _mk_visit_body(body, file_path, func_name, source_lines, markers, insertions, in_try=False, suite_line=None):

    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, ast.If):
            _mk_handle_if(stmt, file_path, func_name, source_lines, markers, insertions, in_try)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            _mk_handle_for(stmt, file_path, func_name, source_lines, markers, insertions, in_try)
        elif isinstance(stmt, ast.While):
            _mk_handle_while(stmt, file_path, func_name, source_lines, markers, insertions, in_try)
        elif isinstance(stmt, ast.Try) or (_TRYSTAR and isinstance(stmt, _TRYSTAR)):
            _mk_handle_try(stmt, file_path, func_name, source_lines, markers, insertions, in_try)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            _mk_handle_with(stmt, file_path, func_name, source_lines, markers, insertions, in_try)
        elif isinstance(stmt, ast.Raise):
            if stmt.lineno == suite_line:
                continue
            _mk_handle_raise(stmt, file_path, func_name, source_lines, markers, insertions)
        elif isinstance(stmt, ast.Match):
            _mk_handle_match(stmt, file_path, func_name, source_lines, markers, insertions, in_try)
        elif isinstance(stmt, ast.Assert):
            if stmt.lineno == suite_line:
                continue
            _mk_add(stmt.lineno, file_path, func_name, "assert", stmt.lineno,
                    _mk_src(source_lines, stmt.lineno), markers, insertions,
                    value_names=_cond_var_names(stmt.test))
        elif isinstance(stmt, ast.Return):
            if stmt.lineno == suite_line:
                continue
            _mk_add(stmt.lineno, file_path, func_name, "return", stmt.lineno,
                    _mk_src(source_lines, stmt.lineno), markers, insertions)
        elif isinstance(stmt, ast.Break):
            if stmt.lineno == suite_line:
                continue
            _mk_add(stmt.lineno, file_path, func_name, "break", stmt.lineno,
                    _mk_src(source_lines, stmt.lineno), markers, insertions)
        elif isinstance(stmt, ast.Continue):
            if stmt.lineno == suite_line:
                continue
            _mk_add(stmt.lineno, file_path, func_name, "continue", stmt.lineno,
                    _mk_src(source_lines, stmt.lineno), markers, insertions)


def _mk_src(source_lines, lineno):
    return source_lines[lineno - 1].strip() if 1 <= lineno <= len(source_lines) else ""


def _safe_str_literal(s: str) -> str:
    return (s.replace("\\", "\\\\").replace("'", "\\'")
             .replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r"))


_RECURSION_RISK_DUNDERS = {
    "__getattr__", "__getattribute__", "__repr__", "__str__", "__hash__",
    "__eq__", "__ne__", "__getitem__", "__contains__", "__len__", "__iter__",
}


def _mk_add(lineno_1based, file_path, func_name, branch_type, decision_line, source_text,
            markers, insertions, is_raise=False, value_names=None):
    key = _marker_key(file_path, func_name, branch_type, decision_line, source_text)
    markers.append(key)
    kind = "RS" if is_raise else "BR"
    safe_key = _safe_str_literal(key)
    insertions.append((
        lineno_1based - 1,
        f"__import__('builtins')._eh_rec('{kind}', '{safe_key}')",
    ))


def _fn_param_names(fn_node) -> list[str]:
    a = fn_node.args
    names: list[str] = []
    for arg in list(getattr(a, "posonlyargs", [])) + list(a.args) + list(a.kwonlyargs):
        names.append(arg.arg)
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    out: list[str] = []
    for n in names:
        if not n.startswith("__") and n not in out:
            out.append(n)
    return out


def _mk_add_fn_enter(fn_node, file_path, func_name, source_lines, markers, insertions):
    if not fn_node.body:
        return
    body = fn_node.body
    first = body[0]
    is_doc = (
        isinstance(first, ast.Expr)
        and isinstance(getattr(first, "value", None), ast.Constant)
        and isinstance(first.value.value, str)
    )
    if is_doc:
        if len(body) == 1:
            return
        anchor = body[1]
    else:
        anchor = first
    key = _marker_key(file_path, func_name, "fn_enter", fn_node.lineno, _mk_src(source_lines, fn_node.lineno))
    markers.append(key)
    safe_key = _safe_str_literal(key)
    insertions.append((
        anchor.lineno - 1,
        f"__import__('builtins')._eh_rec('FN_ENTER', '{safe_key}')",
    ))


def _deco_name(node) -> str:
    try:
        if isinstance(node, ast.Call):
            node = node.func
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts = []
            cur = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            return ".".join(reversed(parts))
    except Exception as _e:
        logger.warning(f"_deco_name failed: {_e}")
    return ""


_PY_KEYWORDS = {"True", "False", "None", "and", "or", "not", "is", "in", "if", "else", "lambda"}


def _cond_var_names(test_node) -> list[str]:
    if test_node is None:
        return []
    names: list[str] = []
    try:
        for nd in ast.walk(test_node):
            if isinstance(nd, ast.Name) and isinstance(nd.ctx, ast.Load):
                nm = nd.id
                if nm not in _PY_KEYWORDS and not nm.startswith("__") and nm not in names:
                    names.append(nm)
    except Exception as _e:
        logger.warning(f"_cond_var_names failed: {_e}")
        return []
    return names


def _target_names(target_node) -> list[str]:
    if target_node is None:
        return []
    names: list[str] = []
    try:
        for nd in ast.walk(target_node):
            if isinstance(nd, ast.Name):
                if nd.id not in names and not nd.id.startswith("__"):
                    names.append(nd.id)
    except Exception as _e:
        logger.warning(f"_target_names failed: {_e}")
        return []
    return names


def _mk_handle_if(stmt, fp, fn, sl, m, ins, in_try=False):
    if stmt.body:
        if stmt.body[0].lineno != stmt.lineno:
            _mk_add(stmt.body[0].lineno, fp, fn, "if_body", stmt.lineno, _mk_src(sl, stmt.lineno), m, ins,
                    value_names=_cond_var_names(stmt.test))
        _mk_visit_body(stmt.body, fp, fn, sl, m, ins, in_try, stmt.lineno)
    current = stmt
    while current.orelse:
        fe = current.orelse[0]
        if isinstance(fe, ast.If):
            if fe.body:
                if fe.body[0].lineno != fe.lineno:
                    _mk_add(fe.body[0].lineno, fp, fn, "elif_body", fe.lineno, _mk_src(sl, fe.lineno), m, ins,
                            value_names=_cond_var_names(fe.test))
                _mk_visit_body(fe.body, fp, fn, sl, m, ins, in_try, fe.lineno)
            current = fe
        else:
            if fe.lineno != current.lineno:
                _mk_add(fe.lineno, fp, fn, "if_else", current.lineno, _mk_src(sl, current.lineno), m, ins)
            _mk_visit_body(current.orelse, fp, fn, sl, m, ins, in_try, current.lineno)
            break


def _mk_handle_for(stmt, fp, fn, sl, m, ins, in_try=False):
    if stmt.body:
        if stmt.body[0].lineno != stmt.lineno:
            _mk_add(stmt.body[0].lineno, fp, fn, "for_body", stmt.lineno, _mk_src(sl, stmt.lineno), m, ins,
                    value_names=_target_names(stmt.target))
        _mk_visit_body(stmt.body, fp, fn, sl, m, ins, in_try, stmt.lineno)
    if stmt.orelse:
        fe = stmt.orelse[0]
        if fe.lineno != stmt.lineno:
            _mk_add(fe.lineno, fp, fn, "for_else", stmt.lineno, _mk_src(sl, stmt.lineno), m, ins)
        _mk_visit_body(stmt.orelse, fp, fn, sl, m, ins, in_try, stmt.lineno)


def _mk_handle_while(stmt, fp, fn, sl, m, ins, in_try=False):
    if stmt.body:
        if stmt.body[0].lineno != stmt.lineno:
            _mk_add(stmt.body[0].lineno, fp, fn, "while_body", stmt.lineno, _mk_src(sl, stmt.lineno), m, ins,
                    value_names=_cond_var_names(stmt.test))
        _mk_visit_body(stmt.body, fp, fn, sl, m, ins, in_try, stmt.lineno)
    if stmt.orelse:
        fe = stmt.orelse[0]
        if fe.lineno != stmt.lineno:
            _mk_add(fe.lineno, fp, fn, "while_else", stmt.lineno, _mk_src(sl, stmt.lineno), m, ins)
        _mk_visit_body(stmt.orelse, fp, fn, sl, m, ins, in_try, stmt.lineno)


def _mk_handle_try(stmt, fp, fn, sl, m, ins, in_try=False):
    if stmt.body:
        if stmt.body[0].lineno != stmt.lineno:
            _mk_add(stmt.body[0].lineno, fp, fn, "try_body", stmt.lineno, _mk_src(sl, stmt.lineno), m, ins)
        _mk_visit_body(stmt.body, fp, fn, sl, m, ins, True, stmt.lineno)
    for handler in getattr(stmt, "handlers", []):
        if handler.body:
            exc_names = [handler.name] if getattr(handler, "name", None) else None
            if handler.body[0].lineno != handler.lineno:
                _mk_add(handler.body[0].lineno, fp, fn, "except", handler.lineno, _mk_src(sl, handler.lineno), m, ins,
                        value_names=exc_names)
            _mk_visit_body(handler.body, fp, fn, sl, m, ins, True, handler.lineno)
    if stmt.orelse:
        fe = stmt.orelse[0]
        if fe.lineno != stmt.lineno:
            _mk_add(fe.lineno, fp, fn, "try_else", stmt.lineno, _mk_src(sl, stmt.lineno), m, ins)
        _mk_visit_body(stmt.orelse, fp, fn, sl, m, ins, True, stmt.lineno)
    fb = getattr(stmt, "finalbody", None)
    if fb:
        if fb[0].lineno != stmt.lineno:
            _mk_add(fb[0].lineno, fp, fn, "finally", stmt.lineno, _mk_src(sl, stmt.lineno), m, ins)
        _mk_visit_body(fb, fp, fn, sl, m, ins, True, stmt.lineno)


def _mk_handle_with(stmt, fp, fn, sl, m, ins, in_try=False):
    if stmt.body:
        if stmt.body[0].lineno != stmt.lineno:
            _mk_add(stmt.body[0].lineno, fp, fn, "with_body", stmt.lineno, _mk_src(sl, stmt.lineno), m, ins)
        _mk_visit_body(stmt.body, fp, fn, sl, m, ins, in_try, stmt.lineno)


def _mk_handle_raise(stmt, fp, fn, sl, m, ins):
    _mk_add(stmt.lineno, fp, fn, "raise", stmt.lineno, _mk_src(sl, stmt.lineno), m, ins, is_raise=True)


def _mk_handle_match(stmt, fp, fn, sl, m, ins, in_try=False):
    subj_names = _cond_var_names(stmt.subject)
    for case in stmt.cases:
        if case.body:
            if case.body[0].lineno != case.pattern.lineno:
                _mk_add(case.body[0].lineno, fp, fn, "case_body", case.pattern.lineno,
                        _mk_src(sl, case.pattern.lineno), m, ins, value_names=subj_names)
            _mk_visit_body(case.body, fp, fn, sl, m, ins, in_try, case.pattern.lineno)


def instrument_repo_markers(repo_dir: str, output_dir: str) -> list[str]:
    all_markers: list[str] = []
    source_maps: dict[str, dict[int, int]] = {}
    for py_file in iter_repository_python_files(Path(repo_dir)):
        rel_path = str(py_file.relative_to(repo_dir))
        try:
            source = py_file.read_text(errors="replace")
        except Exception as _e:
            logger.error(f"cannot read {py_file}: {_e}")
            continue
        instrumented, file_markers = instrument_markers(source, rel_path)
        if not file_markers:
            continue
        try:
            ast.parse(instrumented)
        except SyntaxError as _e:
            logger.warning(f"instrumented {rel_path} has SyntaxError: {_e}")
            continue
        host_file = Path(output_dir) / rel_path
        host_file.parent.mkdir(parents=True, exist_ok=True)
        host_file.write_text(instrumented, "utf-8")
        source_maps[rel_path] = build_insert_only_source_map(source, instrumented)
        all_markers.extend(file_markers)
    if source_maps:
        serialized = {
            file_path: {str(line): original for line, original in source_map.items()}
            for file_path, source_map in source_maps.items()
        }
        output_path = Path(output_dir)
        (output_path / "_source_map.json").write_text(json.dumps(serialized), "utf-8")
        from instrument_guard import TRACE_MANIFEST_NAME, write_ast_trace_manifest
        write_ast_trace_manifest(repo_dir, list(source_maps), str(output_path / TRACE_MANIFEST_NAME))
    return all_markers


EH_PYTEST_WRAPPER = '''"""Wrapper: injects dedup set into builtins, adds cwd to sys.path, runs pytest."""
import sys
import os
import builtins

builtins._eh_seen = set()

_cwd = os.getcwd()
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)

import pytest
sys.exit(pytest.main(sys.argv[1:] + ["--tb=short", "-q", "--capture=no", "-o", "addopts=", "-p", "no:forked", "-p", "no:cacheprovider"]))
'''




def instrument_patched_repository(repository, overlay, record, preparation):
    from run_instrument_all import PathKeyPointInstrumenter, CONFIG
    from instrument_guard import build_source_map
    from evaluate_results import parse_unified_diff
    from run_instrument_all import get_parser, walk_tree

    repository, overlay = Path(repository), Path(overlay)
    # Disabled: patched collection reuses saved probes instead of running CodeQL.
    # export_and_process_instance({"record": record, "repo": repository}, analysis_root,
    #                             timeout=1800, workers=2, ram_mb=4096, force=True, force_codeql=False)
    points_file = CONFIG["key_point_root"] / record["instance_id"] / "combined_key_points.json"
    points = json.loads(points_file.read_text(encoding="utf-8"))
    if (points.get("status") != "completed" or points.get("instance_id") != record["instance_id"]
            or points.get("base_commit") != record["base_commit"]
            or points.get("repository") != record["repo"]):
        raise RuntimeError(f"Saved key points unavailable or incompatible: {points_file}")
    mapping = parse_unified_diff(record.get("patch", ""))
    unchanged = parse_unified_diff(record.get("patch", ""), align_replacements=False)
    source_hashes = {}
    removed_points = []
    for relative, data in points["files"].items():
        source = (repository / relative).read_text(encoding="utf-8")
        source_hashes[relative] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if relative not in mapping:
            continue
        inverse = {}
        for patched_line in range(1, len(source.splitlines()) + 1):
            original_line = mapping[relative].get(patched_line)
            if original_line is not None:
                if original_line in inverse:
                    raise RuntimeError(f"Ambiguous patched line mapping: {relative}:{original_line}")
                inverse[original_line] = patched_line
        tree = get_parser("python").parse(source.encode("utf-8"))
        nodes = {(node.start_point.row + 1, node.start_point.column, node.type): node
                 for node in walk_tree(tree.root_node)}
        remapped = {}
        def translate(match):
            original_line = int(match.group(2))
            if original_line not in inverse:
                raise RuntimeError(f"Unresolved surviving point identity: {relative}:{original_line}")
            return match.group(1) + str(inverse[original_line]) + match.group(3)
        for original_line, line_points in data["key_points_by_line"].items():
            patched_line = inverse.get(int(original_line))
            for point in line_points:
                node_type = "function_definition" if point["node_type"] == "synthetic" else point["node_type"]
                node = nodes.get((patched_line, int(point["column"]), node_type))
                if node is None:
                    if patched_line is not None and unchanged[relative].get(patched_line) is not None:
                        raise RuntimeError(f"Unchanged saved point not found: {relative}:{original_line}")
                    removed_points.append({"file": relative, "line": int(original_line), "point": point,
                                           "reason": "deleted_or_replaced_by_patch"})
                    continue
                point["point_id"] = re.sub(r"(::|@)(\d+)(:)", translate, point["point_id"])
                if "line" in point:
                    point["line"] = patched_line
                point["end_line"] = patched_line if point["node_type"] == "synthetic" else node.end_point.row + 1
                point["end_column"] = point["column"] if point["node_type"] == "synthetic" else node.end_point.column
                remapped.setdefault(str(patched_line), []).append(point)
        data["key_points_by_line"] = remapped
        data["source_sha256"] = source_hashes[relative]
    points["source_fingerprint"] = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    points["collection_mode"] = "reused_saved_keypoints"
    if removed_points:
        logger.warning(f"{record['instance_id']}: patch deleted or replaced saved probes: {removed_points}")
    with Path(preparation).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"stage": "patched_reused_keypoints", "source": str(points_file),
                                 "source_sha256": hashlib.sha256(points_file.read_bytes()).hexdigest(),
                                 "removed_points": removed_points, "points": points}, ensure_ascii=False) + "\n")
    instrumenter = PathKeyPointInstrumenter()
    source_maps = {}
    for relative, data in points["files"].items():
        source = (repository / relative).read_text(encoding="utf-8")
        instrumented, locations, gaps = instrumenter.instrument(source, data["key_points_by_line"], relative)
        if gaps:
            raise RuntimeError(f"Patched instrumentation gaps in {relative}: {gaps}")
        compile(instrumented, relative, "exec")
        destination = overlay / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(instrumented, encoding="utf-8")
        source_maps[relative] = build_source_map(instrumented, relative, [], locations)
    metadata = {"source_map": source_maps, "path_points": points, "trace_manifest": {}, "guard_manifest": {"files": {}}}
    (overlay / "_instrumentation.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    with Path(preparation).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"stage": "patched_source_maps", "source_map": source_maps}, ensure_ascii=False) + "\n")
    logger.info(f"{record['instance_id']}: reused saved key-point instrumentation (CodeQL disabled): {len(source_maps)} files")
    return {"path_schema": "epmr-keypoints-v1", "keypoints_source_fingerprint": points.get("source_fingerprint"),
            "keypoints_builder_fingerprint": points.get("builder_fingerprint"),
            "keypoints_collection_mode": "reused_saved_keypoints",
            "saved_keypoints_sha256": hashlib.sha256(points_file.read_bytes()).hexdigest()}


def prepare_cached_gold(record, output_dir):
    from run_instrument_all import PathKeyPointInstrumenter
    from run_find_points_keypoints import parse_repository, build_path_key_points, walk_tree
    from instrument_guard import build_source_map
    from gold_keypoint_alignment import align_saved_point_identities

    iid = record["instance_id"]
    prebuilt = PROJECT_ROOT / "results/repo_instrumented" / iid / "repo"
    manifest_file = prebuilt / "_instrumentation.json"
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("build", {}).get("status") != "completed" or manifest.get("base_commit") != record["base_commit"]:
        raise RuntimeError(f"Incomplete or incompatible offline repository: {prebuilt}")
    def artifact(name):
        spec = manifest["artifacts"][name]
        raw = (prebuilt / spec["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise RuntimeError(f"Offline artifact hash mismatch: {name}")
        return json.loads(raw)
    points, source_maps = artifact("path_points"), artifact("source_map")
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", record["repo"].replace("/", "__"))
    matches = list((PROJECT_ROOT / "datasets/raw_repo_source_code").glob(
        f"{prefix}___base-{record['base_commit'][:12]}___src-*/metadata.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one offline source cache for {iid}, found {len(matches)}")
    source_metadata = json.loads(matches[0].read_text())
    source_repo = matches[0].parent / source_metadata.get("source_path", "repo")
    patches = {name: record.get(name, "") for name in ("patch", "test_patch")}
    touched = sorted(set(re.findall(r"^(?:--- a/|\+\+\+ b/)(.+)$", "\n".join(patches.values()), re.M)))
    if any(Path(path).is_absolute() or ".." in Path(path).parts or path.startswith(".git/") for path in touched):
        raise ValueError("Unsafe patch path")
    dependencies = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("run_record_gold_trace.py", "run_instrument_all.py", "run_find_points_keypoints.py", "instrument_guard.py", "gold_keypoint_alignment.py", "evaluate_results.py")}
    signature = hashlib.sha256(json.dumps({"record": record, "manifest": manifest,
        "source": source_metadata, "dependencies": dependencies,
        "touched_sources": {path: hashlib.sha256((source_repo / path).read_bytes()).hexdigest()
                            if (source_repo / path).is_file() else None for path in touched}}, sort_keys=True).encode()).hexdigest()
    cache = output_dir / ".prepared" / iid / signature
    cache.parent.mkdir(parents=True, exist_ok=True)
    with (cache.parent / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (cache / "output.json").is_file():
            saved = json.loads((cache / "output.json").read_text())
            if all((cache / "repo" / path).is_file() and hashlib.sha256((cache / "repo" / path).read_bytes()).hexdigest() == digest
                   for path, digest in saved["files_sha256"].items()):
                logger.info(f"{iid}: reusing prepared patched repository: {cache}")
                return cache, saved
            raise RuntimeError(f"Prepared repository integrity mismatch: {cache}")
        with tempfile.TemporaryDirectory(prefix=".build-", dir=cache.parent) as temporary:
            build = Path(temporary)
            changed = build / "changed"
            overlay = build / "repo"
            changed.mkdir()
            overlay.mkdir()
            evidence = build / "preparation.jsonl"
            evidence.touch()
            # Copies are independent: neither patching nor container writes touch original artifacts.
            for relative in sorted(set(source_maps) | set(points["files"])):
                destination = overlay / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(prebuilt / relative, destination)
            for path in touched:
                if (source_repo / path).is_file():
                    destination = changed / path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_repo / path, destination)
            for name, text in patches.items():
                if not text:
                    continue
                patch_file = build / f"{name}.diff"
                patch_file.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
                result = _run_logged(["git", "apply", "--check", str(patch_file)], build / f"{name}_check_raw.json", 120, changed)
                if result["returncode"] != 0:
                    reverse = _run_logged(["git", "apply", "--reverse", "--check", str(patch_file)], build / f"{name}_reverse_raw.json", 120, changed)
                    if reverse["returncode"] == 0:
                        continue
                    raise RuntimeError(f"Offline {name} does not apply: {result['stderr']}")
                result = _run_logged(["git", "apply", str(patch_file)], build / f"{name}_apply_raw.json", 120, changed)
                if result["returncode"] != 0:
                    raise RuntimeError(f"Offline {name} failed: {result['stderr']}")
            # Reuse the existing AST coordinate/point builder only on changed source files.
            sources, trees, gaps = parse_repository(changed)
            if gaps:
                raise RuntimeError(f"Changed source parse gaps: {gaps}")
            graph = {"files": {}}
            for relative, tree in trees.items():
                nodes = []
                for node in walk_tree(tree.root_node):
                    parent = node.parent
                    while parent is not None and parent.type != "function_definition":
                        parent = parent.parent
                    function = ("Function " + sources[relative][parent.child_by_field_name("name").start_byte:parent.child_by_field_name("name").end_byte].decode()
                                if parent is not None else "Module " + relative.removesuffix(".py").replace("/", "."))
                    nodes.append({"node_id": f"{relative}:{node.start_byte}", "function": function,
                                  "line": node.start_point.row + 1, "column": node.start_point.column})
                graph["files"][relative] = {"nodes": nodes}
            rebuilt = build_path_key_points(changed, graph)
            if rebuilt.get("analysis_gaps") or rebuilt.get("repository_parse_gaps"):
                raise RuntimeError(f"Changed point analysis gaps: {rebuilt.get('analysis_gaps')}")
            deleted = []
            instrumenter = PathKeyPointInstrumenter()
            for relative in touched:
                source = changed / relative
                destination = overlay / relative
                if not source.is_file():
                    deleted.append(relative)
                    if destination.is_file():
                        destination.unlink()
                    points["files"].pop(relative, None)
                    source_maps.pop(relative, None)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if relative not in rebuilt["files"]:
                    shutil.copy2(source, destination)
                    points["files"].pop(relative, None)
                    source_maps.pop(relative, None)
                    continue
                data = rebuilt["files"][relative]
                source_text = source.read_text()
                aligned = align_saved_point_identities(
                    relative, data, points["files"].get(relative, {}), record, len(source_text.splitlines()))
                with evidence.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"stage": "saved_point_identity_alignment", "file": relative,
                                             "points": aligned}, ensure_ascii=False) + "\n")
                instrumented, locations, gaps = instrumenter.instrument(source_text, data["key_points_by_line"], relative)
                if gaps:
                    raise RuntimeError(f"Changed instrumentation gaps: {relative}: {gaps}")
                compile(instrumented, relative, "exec")
                destination.write_text(instrumented, encoding="utf-8")
                points["files"][relative] = data
                source_maps[relative] = build_source_map(instrumented, relative, [], locations)
            from gold_keypoint_alignment import restore_gold_docstrings
            for destination in sorted(overlay.rglob("*.py")):
                relative = destination.relative_to(overlay).as_posix()
                original = changed / relative if relative in touched else source_repo / relative
                if not original.is_file():
                    continue
                restored, repairs = restore_gold_docstrings(destination.read_text(), original.read_text(), relative)
                if repairs:
                    destination.write_text(restored)
                    with evidence.open("a") as stream:
                        stream.write(json.dumps({"stage": "gold_only_semantic_repairs", "file": relative,
                                                 "repairs": repairs}, ensure_ascii=False) + "\n")
            points["source_fingerprint"] = signature
            points["collection_mode"] = "offline_incremental_patched"
            runtime_metadata = {"source_map": source_maps, "path_points": points,
                                "trace_manifest": {}, "guard_manifest": {"files": {}}}
            (overlay / "_instrumentation.json").write_text(json.dumps(runtime_metadata), encoding="utf-8")
            metadata = {"path_schema": "epmr-keypoints-v1", "keypoints_source_fingerprint": signature,
                        "keypoints_builder_fingerprint": points["builder_fingerprint"],
                        "keypoints_collection_mode": "offline_incremental_patched", "deleted_files": deleted,
                        "changed_files": touched, "signature": signature}
            with evidence.open("a") as stream:
                stream.write(json.dumps({"stage": "offline_incremental_preparation", "metadata": metadata,
                                         "source_metadata": source_metadata, "rebuilt_points": rebuilt}) + "\n")
            metadata["files_sha256"] = {path.relative_to(overlay).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in overlay.rglob("*") if path.is_file()}
            _write_json(build / "output.json", metadata)
            build.rename(cache)
        logger.info(f"{iid}: prepared changed files={touched}; unchanged offline instrumentation reused")
        return cache, metadata


def build_gold_builtins_setup_cmd():
    import types
    import heal_builtins

    from gold_keypoint_alignment import fix_gold_value_comparison

    runtime = fix_gold_value_comparison(heal_builtins.RUNTIME_BOOTSTRAP)
    start = runtime.index("def _epmr_event(")
    end = runtime.index("\ndef _path_hit_record(", start)
    guarded_emitter = '''import base64 as _epmr_base64
import zlib as _epmr_zlib
_EPMR_BATCH = []
_EPMR_BATCH_PID = os.getpid()

def _ensure_epmr_process():
    global _EPMR_BATCH_PID, _EPMR_SEQ, _EPMR_DIGEST
    if _EPMR_BATCH_PID == os.getpid():
        return
    _EPMR_BATCH_PID = os.getpid()
    _EPMR_BATCH.clear()
    _EPMR_SEQ = 0
    _EPMR_DIGEST = hashlib.sha256()
    sys.stderr.write("__EPMR_START__\\t" + json.dumps({
        "schema": "epmr-keypoints-v1", "pid": os.getpid(),
        "runtime_fingerprint": _EPMR_RUNTIME_FINGERPRINT,
        "builder_fingerprint": _PATH_POINTS.get("builder_fingerprint"),
        "source_fingerprint": _PATH_POINTS.get("source_fingerprint"),
        "query_fingerprint": _PATH_POINTS.get("query_fingerprint"),
        "instance_id": _PATH_POINTS.get("instance_id"),
    }, sort_keys=True) + "\\n")

def _flush_epmr_batch():
    _ensure_epmr_process()
    if not _EPMR_BATCH:
        return
    payload = json.dumps(_EPMR_BATCH, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    record = {"schema": "epmr-keypoints-v1", "pid": os.getpid(),
              "seq": _EPMR_SEQ - len(_EPMR_BATCH), "count": len(_EPMR_BATCH),
              "encoding": "zlib-json-tuples-v1",
              "data": _epmr_base64.b64encode(_epmr_zlib.compress(payload, 1)).decode("ascii")}
    raw = json.dumps(record, ensure_ascii=False)
    sys.stderr.write("__EPMR_BATCH__\\t" + raw + "\\n")
    _EPMR_DIGEST.update((raw + "\\n").encode("utf-8"))
    _EPMR_BATCH.clear()

def _epmr_event(point_id, decision=None):
    global _EPMR_SEQ, _EPMR_BATCH_PID, _EPMR_DIGEST
    if not _EPMR_ENABLED or getattr(_TRACE_LOCAL, "active", False):
        return
    # Reuse the collector's thread-local guard for instrumented stream wrappers.
    _TRACE_LOCAL.active = True
    try:
        with _TRACE_LOCK:
            _ensure_epmr_process()
            _EPMR_BATCH.append((threading.get_ident(), point_id, decision))
            _EPMR_SEQ += 1
            if len(_EPMR_BATCH) >= 2048:
                _flush_epmr_batch()
    except BaseException as error:
        _report_failure("EPMR", error)
    finally:
        _TRACE_LOCAL.active = False

'''
    # Keep shared runtime globals untouched while reusing its setup and fingerprinting.
    setup_globals = dict(vars(heal_builtins))
    runtime = runtime[:start] + guarded_emitter + runtime[end:]
    for required in ('    if _EPMR_ENABLED:\n        sys.stderr.write("__EPMR_END__',
                     '    _epmr_event(point_id)\n    with _TRACE_LOCK:',
                     '    _epmr_event(point_id, {"result": decision_result})\n    thread_id'):
        if runtime.count(required) != 1:
            raise ValueError(f"Golden batch optimization no longer matches shared runtime: {required}")
    runtime = runtime.replace('    if _EPMR_ENABLED:\n        sys.stderr.write("__EPMR_END__',
                              '    if _EPMR_ENABLED:\n        _flush_epmr_batch()\n        sys.stderr.write("__EPMR_END__', 1)
    # EPMR already retains these hits; do not also construct duplicate route summaries.
    runtime = runtime.replace('    _epmr_event(point_id)\n    with _TRACE_LOCK:',
                              '    _epmr_event(point_id)\n    if _EPMR_ENABLED:\n        return\n    with _TRACE_LOCK:', 1)
    runtime = runtime.replace('    _epmr_event(point_id, {"result": decision_result})\n    thread_id',
                              '    _epmr_event(point_id, {"result": decision_result})\n    if _EPMR_ENABLED:\n        return decision_result if mode == "truth" else value\n    thread_id', 1)
    old_call = '        result = function(*args, **kwargs)'
    if runtime.count(old_call) != 1:
        raise ValueError("Golden exec scope fix no longer matches shared runtime")
    runtime = runtime.replace(old_call, '''        if function is exec and len(args) == 1 and not kwargs:
            result = exec(args[0], frame.f_globals, frame.f_locals)
        elif function is eval and len(args) == 1 and not kwargs:
            result = eval(args[0], frame.f_globals, frame.f_locals)
        else:
            result = function(*args, **kwargs)''', 1)
    setup_globals["RUNTIME_BOOTSTRAP"] = runtime
    return types.FunctionType(heal_builtins.build_builtins_setup_cmd.__code__, setup_globals)()


def _docker_run_with_markers(
    image: str,
    before_cmd: str,
    test_args: str,
    workdir: str,
    patch_file: str = "",
    test_patch_file: str = "",
    want_gold: bool = True,
    workspace_dir: str = "",
    instr_dir: str = "",
    runtime_trace_path: str = "",
    runtime_trace_run_id: str = "",
    epmr: bool = False,
    deleted_files: list[str] | None = None,
    timeout_seconds: int = DOCKER_TIMEOUT,
) -> dict:
    vol_flags = [
        "-v", f"{workspace_dir}:/workspace:rw",
    ]
    trace_env = []
    if runtime_trace_path:
        trace_dir = str(Path(runtime_trace_path).parent)
        vol_flags += ["-v", f"{trace_dir}:/records:rw"]
        trace_env = [
            "-e", f"EH_RUNTIME_TRACE_FILE=/records/{Path(runtime_trace_path).name}",
            "-e", f"EH_RUNTIME_TRACE_RUN_ID={runtime_trace_run_id}",
            "-e", f"EH_RUNTIME_TRACE_ROOT={workdir}",
        ]

    patch_cmd = ""
    if patch_file and os.path.exists(patch_file):
        vol_flags += ["-v", f"{patch_file}:/tmp/_gold.patch:ro"]
        if want_gold:
            patch_cmd = f"(cd {shlex.quote(workdir)} && git apply --check /tmp/_gold.patch && git apply /tmp/_gold.patch) || exit $?; "

    test_patch_cmd = ""
    if test_patch_file and os.path.exists(test_patch_file):
        vol_flags += ["-v", f"{test_patch_file}:/tmp/_test.patch:ro"]
        test_patch_cmd = (
            f"cd {shlex.quote(workdir)} || exit $?; "
            "if git apply --check /tmp/_test.patch; then git apply /tmp/_test.patch || exit $?; "
            "elif git apply --reverse --check /tmp/_test.patch; then printf 'Test patch already applied\\n'; "
            "else exit 1; fi; "
        )

    overlaying = bool(instr_dir and os.path.isdir(instr_dir))
    if overlaying:
        vol_flags += ["-v", f"{instr_dir}:/opt/eh-marker-overlay:ro"]

    setup = build_gold_builtins_setup_cmd() if epmr else build_eh_setup_cmd()
    prep_cmd, _cap = build_instr_prep_cmd(
        workdir, do_overlay=False, use_pythonpath=overlaying, eh_setup_cmd=setup
    )
    epmr_env = (f"export EH_EPMR_ENABLED=1 EH_PATH_POINTS_ENABLED=1 EH_RUNTIME_TRACE_ROOT={shlex.quote(workdir)} "
                "EH_RUNTIME_TRACE_FILE=/workspace/epmr-runtime/events.jsonl EH_RUNTIME_TRACE_RUN_ID=correct; ") if epmr else ""
    overlay_cmd = f"cp -r /opt/eh-marker-overlay/. {shlex.quote(workdir)}/ || exit $?; " if overlaying else ""
    for relative in deleted_files or []:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe deleted file: {relative}")
        overlay_cmd += f"rm -f -- {shlex.quote(workdir + '/' + relative)}; "
    if epmr:
        overlay_cmd += "unset _EH_ACTIVE; "

    inner_cmd = (
        "set -e; "
        f"{before_cmd}; "
        f"{test_patch_cmd}"
        f"{patch_cmd}"
        f"{overlay_cmd}"
        f"{prep_cmd}"
        f"export COVERAGE_PROCESS_START='' && "
        f"unset PYTEST_ADDOPTS && "
        f"cd {workdir} && "
        f"{epmr_env}"
        f"python -m pytest {test_args} --tb=short -v --capture=no; "
        f"_PYTEST_RC=$?; "
        f"exit $_PYTEST_RC"
    )

    container_name = f"gt_{uuid.uuid4().hex[:12]}"
    try:
        proc = subprocess.run(
            [
                "docker", "run", "--rm",
                "--name", container_name,
                *trace_env,
                *vol_flags,
                "--entrypoint", "/bin/bash",
                image, "-c", inner_cmd,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}
    except subprocess.TimeoutExpired as error:
        logger.warning(f"Container {container_name} timed out after {timeout_seconds}s: {error}")
        try:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=60)
        except Exception as e:
            logger.warning(f"docker rm -f {container_name} after timeout failed: {e}")
        def decode(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
        return {"stdout": decode(error.stdout), "stderr": decode(error.stderr),
                "returncode": -1, "error": "timeout", "timeout_seconds": timeout_seconds}
    except Exception as e:
        logger.error(f"Container {container_name} failed: {e}")
        return {"stdout": "", "stderr": str(e), "returncode": -1, "error": type(e).__name__}




def _parse_markers_from_stderr(stderr: str) -> dict:
    branch_markers: list[str] = []
    raise_markers: list[str] = []
    events: list[dict] = []
    truncated: list[str] = []
    errors: list[str] = []
    for line in stderr.splitlines():
        if line.startswith(("__EH_REC_FAIL__", "__EH_SETUP_FAIL__", "__EH_STACK_FAIL__", "__EH_STACK_FRAME_FAIL__")):
            errors.append(line)
        if line.startswith("__EH_TRUNC__\t"):
            tp = line.split("\t", 2)
            if len(tp) == 3:
                truncated.append(f"{tp[1]}\t{tp[2]}")
            continue
        if not line.startswith("__EH__\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            errors.append(line)
            continue
        kind = parts[1]
        seq = None
        if parts[2].lstrip("-").isdigit():
            seq = int(parts[2])
            rest = parts[3:]
        else:
            rest = parts[2:]
        if seq is not None and len(rest) >= 6:
            key = "\t".join(rest[:5])
            vals = "\t".join(rest[5:])
        elif seq is not None and len(rest) == 5:
            key = "\t".join(rest)
            vals = ""
        else:
            key = "\t".join(rest)
            vals = ""
        events.append({"seq": seq, "kind": kind, "key": key, "vals": vals})
        if kind == "BR":
            branch_markers.append(key)
        elif kind == "RS":
            raise_markers.append(key)
    return {
        "branch_markers": list(dict.fromkeys(branch_markers)),
        "raise_markers": list(dict.fromkeys(raise_markers)),
        "events": events,
        "truncated": truncated,
        "errors": errors,
    }


def _basic_trace_stats(gold_trace: dict, buggy_trace: dict) -> dict:
    gold_b = set(gold_trace.get("branch_markers", []))
    buggy_b = set(buggy_trace.get("branch_markers", []))
    gold_r = set(gold_trace.get("raise_markers", []))
    buggy_r = set(buggy_trace.get("raise_markers", []))

    return {
        "gold_only_branches": len(gold_b - buggy_b),
        "buggy_only_branches": len(buggy_b - gold_b),
        "shared_branches": len(gold_b & buggy_b),
        "gold_only_raises": len(gold_r - buggy_r),
        "buggy_only_raises": len(buggy_r - gold_r),
        "shared_raises": len(gold_r & buggy_r),
    }




def _fuzz_test_args(rec: dict, limit: int) -> tuple[str, list[str]]:
    if limit <= 0:
        return "", []
    ptp_raw = rec.get("pass_to_pass")
    if isinstance(ptp_raw, str):
        try:
            ptp = json.loads(ptp_raw)
        except json.JSONDecodeError as e:
            logger.warning(f"_fuzz_test_args: pass_to_pass not valid JSON ({e}); fuzz flow skipped")
            return "", []
    elif isinstance(ptp_raw, list):
        ptp = ptp_raw
    else:
        return "", []
    seen: set[str] = set()
    nodeids: list[str] = []
    for n in ptp:
        if not isinstance(n, str) or not n.strip() or n in seen:
            continue
        seen.add(n)
        nodeids.append(n.strip())
    if len(nodeids) > limit:
        logger.warning(f"{rec['instance_id']}: requested pass_to_pass sample limit={limit}; selecting {limit} of {len(nodeids)} tests")
        nodeids = nodeids[:limit]
    if not nodeids:
        return "", []
    args = " ".join(shlex.quote(n) for n in nodeids)
    return args, nodeids




def _save_log(path: Path, result: dict):
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"=== rc={result['returncode']} ===\n=== STDOUT ===\n")
        stream.write(result.get("stdout", ""))
        stream.write("\n=== STDERR ===\n")
        stream.write(result.get("stderr", ""))


def _run_logged(arguments, path, timeout, cwd=None):
    try:
        proc = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        result = {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}
    except subprocess.TimeoutExpired as error:
        logger.error(f"{path.name}: command timed out: {error}")
        result = {"returncode": -1, "error": "timeout", "timeout_seconds": timeout}
        for key in ("stdout", "stderr"):
            value = getattr(error, key)
            result[key] = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
    except OSError as error:
        logger.error(f"{path.name}: command failed: {error}")
        result = {"stdout": "", "stderr": str(error), "returncode": -1, "error": type(error).__name__}
    with (path.parent / "preparation.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"stage": path.stem.removesuffix("_raw"), "command": [str(value) for value in arguments],
                                 "cwd": str(cwd) if cwd is not None else None, **result}, ensure_ascii=False) + "\n")
    return result


def _write_json(path: Path, data):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        temporary = stream.name
    os.replace(temporary, path)


def _valid_marker_trace(trace):
    events = trace.get("events", [])
    seqs = [event.get("seq") for event in events]
    return bool(events) and not trace.get("truncated") and not trace.get("errors") and all(
        type(seq) is int and seq >= 0 and (index == 0 or seq > seqs[index - 1])
        for index, seq in enumerate(seqs))


def _collection_complete(result, instance_dir, run_buggy, fuzz_nodeids):
    if not all((instance_dir / name).is_file() for name in ("input_raw.json", "preparation.jsonl")):
        return False
    for label, key, allowed in (("correct", "gold", (0,)), ("buggy", "buggy", (0, 1))):
        if label == "buggy" and not run_buggy:
            continue
        if result.get(f"{key}_test_rc") not in allowed or not (instance_dir / f"{key}.log").is_file():
            return False
        if key == "gold" and result.get("path_schema") == "epmr-keypoints-v1":
            integrity = result.get("gold_stream_integrity")
            if integrity is None:
                from evaluate_direct_keypoints import stream_points
                _, integrity = stream_points(instance_dir / "gold.log", [])
            if not integrity["complete"]:
                logger.warning(f"{instance_dir.name}: EPMR trace incomplete: {integrity}")
                return False
            continue
        trace = result.get(f"{key}_trace")
        if trace is None:
            from evaluate_results import read_gold_marker_log
            trace = read_gold_marker_log(instance_dir / f"{key}.log")
        if not _valid_marker_trace(trace):
            return False
    if fuzz_nodeids:
        runs = result.get("fuzz_inputs", [])
        if len(runs) != len(fuzz_nodeids) or any(not run.get("normal_baseline_included") for run in runs):
            return False
    return True


def process_instance(rec: dict, output_dir: Path, force: bool = False,
                     fuzz_limit: int = 10, run_fuzz: bool = True, run_buggy: bool = True,
                     timeout_seconds: int = DOCKER_TIMEOUT) -> dict:
    # Keep imported callers' legacy defaults; the CLI explicitly selects patched-only.
    iid = rec["instance_id"]
    if iid not in HEALBENCH_INSTANCE_IDS:
        logger.info(f"Excluded 1 instance by exact HEALBENCH_INSTANCE_IDS match: {iid}")
        return {"instance_id": iid, "error": "excluded_gold_trace_unusable", "collection_complete": False}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = output_dir / ".locks"
    lock_dir.mkdir(exist_ok=True)
    dependencies = ("run_record_gold_trace.py", "heal_repository_scope.py", "instrument_guard.py",
                    "heal_builtins.py", "util.py", "agenticheal.py", "evaluate_paper_metrics.py",
                    "run_find_points_keypoints.py", "run_find_points_sinks.py", "run_instrument_all.py",
                    "gold_keypoint_alignment.py", "evaluate_results.py", "evaluate_direct_keypoints.py")
    from run_instrument_all import CONFIG
    points_file = CONFIG["key_point_root"] / iid / "combined_key_points.json"
    points_hash = hashlib.sha256(points_file.read_bytes()).hexdigest() if points_file.is_file() and not (run_buggy or run_fuzz) else None
    offline_manifest = PROJECT_ROOT / "results/repo_instrumented" / iid / "repo/_instrumentation.json"
    offline_hash = hashlib.sha256(offline_manifest.read_bytes()).hexdigest() if offline_manifest.is_file() else None
    signature = hashlib.sha256(json.dumps({"record": rec, "buggy": run_buggy,
        "offline_manifest_sha256": offline_hash, "saved_keypoints_sha256": points_hash,
        "fuzz": run_fuzz, "fuzz_limit": fuzz_limit if run_fuzz else 0, "timeout_seconds": timeout_seconds,
        "scripts": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in dependencies}}, sort_keys=True).encode()).hexdigest()
    fuzz_nodeids = _fuzz_test_args(rec, fuzz_limit)[1] if run_fuzz and fuzz_limit > 0 else []
    out_file = output_dir / iid / "output.json"
    with (lock_dir / f"{iid}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if out_file.is_file() and not force:
            try:
                saved = json.loads(out_file.read_text())
                if (saved.get("collection_signature") == signature and saved.get("collection_complete")
                        and _collection_complete({**saved, "gold_stream_integrity": None}, output_dir / iid, run_buggy, fuzz_nodeids)):
                    logger.info(f"{iid}: valid matching collection exists, skipping")
                    if run_buggy or run_fuzz:
                        from evaluate_results import load_gold_trace
                        restored = load_gold_trace(iid, output_dir)
                        if run_buggy:
                            restored["metrics"] = _basic_trace_stats(restored.get("gold_trace", {}), restored.get("buggy_trace", {}))
                        return restored
                    return saved
            except (OSError, ValueError) as error:
                logger.warning(f"{iid}: invalid saved result: {error}")
        old_paths = [path for path in (output_dir / f"{iid}.json", output_dir / iid) if path.exists()]
        if old_paths:
            archive = output_dir / ".history" / iid / uuid.uuid4().hex
            archive.mkdir(parents=True)
            logger.warning(f"{iid}: preserving previous collection in {archive}")
            for path in old_paths:
                shutil.move(str(path), str(archive / path.name))
        instance_dir = output_dir / iid
        instance_dir.mkdir()
        _write_json(instance_dir / "input_raw.json", rec)
        (instance_dir / "preparation.jsonl").touch()
        def save_diagnostic(message):
            entry = message.record
            with (instance_dir / "preparation.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"stage": "diagnostic", "level": entry["level"].name,
                                         "message": entry["message"]}, ensure_ascii=False) + "\n")
        sink = logger.add(save_diagnostic, filter=lambda entry: entry["extra"].get("gold_instance") == iid)
        try:
            with logger.contextualize(gold_instance=iid):
                result = _collect_instance(rec, output_dir, fuzz_limit, run_fuzz, run_buggy, timeout_seconds)
        except Exception as error:
            with logger.contextualize(gold_instance=iid):
                logger.error(f"{iid}: collection failed: {type(error).__name__}: {error}")
            result = {"instance_id": iid, "error": str(error)}
        finally:
            logger.remove(sink)
        complete = _collection_complete(result, instance_dir, run_buggy, fuzz_nodeids)
        result.update(collection_signature=signature, collection_complete=bool(complete))
        saved = {key: value for key, value in result.items()
                 if key not in ("gold_trace", "buggy_trace", "fuzz_trace", "metrics")}
        _write_json(out_file, saved)
        return result if run_buggy or run_fuzz else saved


def _collect_instance(rec, output_dir, fuzz_limit, run_fuzz, run_buggy, timeout_seconds):
    iid = rec["instance_id"]
    if not rec.get("patch") or not rec.get("test_name"):
        return {"instance_id": iid, "error": "missing_patch_or_target_test"}

    image, workdir = _resolve_docker_image(rec)
    if not image:
        logger.error(f"  {iid}: no image, skipping")
        return {"instance_id": iid, "error": "no_image"}

    tag = rec.get("dockerhub_tag", "")
    is_r2egym = bool(tag and "/" in tag)
    before_cmd = rec.get("before_repo_set_cmd", "") or "true"
    test_name = rec.get("test_name", "")
    if is_r2egym and test_name:
        test_name = test_name.replace("../r2e_tests/", "/r2e_tests/")
    test_args = shlex.quote(test_name) if test_name else ""

    with tempfile.TemporaryDirectory(prefix="gold_trace_") as tmpdir:
        patch_file = ""
        if rec.get("patch"):
            patch_file = os.path.join(tmpdir, "gold.patch")
            Path(patch_file).write_text(rec["patch"], "utf-8")

        test_patch_file = ""
        if rec.get("test_patch"):
            test_patch_file = os.path.join(tmpdir, "test.patch")
            Path(test_patch_file).write_text(rec["test_patch"], "utf-8")

        epmr = not (run_buggy or run_fuzz)
        inst_out_dir = output_dir / iid
        deleted_files = []
        if epmr:
            prepared, cached = prepare_cached_gold(rec, output_dir)
            gold_instr_dir = str(prepared / "repo")
            deleted_files = cached["deleted_files"]
            epmr_metadata = {key: cached[key] for key in (
                "path_schema", "keypoints_source_fingerprint", "keypoints_builder_fingerprint", "keypoints_collection_mode")}
            with (inst_out_dir / "preparation.jsonl").open("a") as destination:
                with (prepared / "preparation.jsonl").open() as source:
                    shutil.copyfileobj(source, destination)
            # The overlay already includes both patches; never apply them to instrumented text.
            patch_file = test_patch_file = ""
        else:
            extract_dir = os.path.join(tmpdir, "repo_extract")
            os.makedirs(extract_dir)
            container_name = f"extract_{uuid.uuid4().hex[:8]}"
            inst_out_dir = output_dir / iid
            try:
                steps = (
                    ("docker_create", ["docker", "create", "--name", container_name,
                     "--entrypoint", "/bin/bash", image, "-c", f"set -e; {before_cmd}"], 120),
                    ("setup", ["docker", "start", "-a", container_name], DOCKER_TIMEOUT),
                    ("docker_cp", ["docker", "cp", f"{container_name}:{workdir}/.", extract_dir], 600),
                )
                for stage, arguments, timeout in steps:
                    result = _run_logged(arguments, inst_out_dir / f"{stage}_raw.json", timeout)
                    if result["returncode"] != 0:
                        logger.error(f"{iid}: {stage} failed: {result['stderr']}")
                        return {"instance_id": iid, "error": f"{stage}_failed"}
            finally:
                cleanup = _run_logged(["docker", "rm", "-f", container_name], inst_out_dir / "cleanup_raw.json", 30)
                if cleanup["returncode"] != 0:
                    logger.warning(f"{iid}: extraction container cleanup failed: {cleanup['stderr']}")

            if test_patch_file:
                for operation in ("check", "apply"):
                    arguments = ["git", "apply", *(["--check"] if operation == "check" else []), test_patch_file]
                    result = _run_logged(arguments, inst_out_dir / f"test_patch_{operation}_raw.json", 120, extract_dir)
                    if result["returncode"] != 0:
                        if operation == "check":
                            reverse = _run_logged(["git", "apply", "--reverse", "--check", test_patch_file],
                                                  inst_out_dir / "test_patch_reverse_check_raw.json", 120, extract_dir)
                            if reverse["returncode"] == 0:
                                logger.info(f"{iid}: test patch already applied by repository setup")
                                break
                        logger.error(f"{iid}: test patch {operation} failed: {result['stderr']}")
                        return {"instance_id": iid, "error": f"test_patch_{operation}_failed"}

            buggy_instr_dir = os.path.join(tmpdir, "buggy_instr")
            os.makedirs(buggy_instr_dir)
            gold_instr_dir = os.path.join(tmpdir, "gold_instr")
            os.makedirs(gold_instr_dir)

            if run_buggy or run_fuzz:
                buggy_markers = instrument_repo_markers(extract_dir, buggy_instr_dir)
                logger.info(f"{iid}: buggy instrumented {len(buggy_markers)} markers")
            if patch_file:
                try:
                    _ptxt = Path(patch_file).read_text("utf-8", errors="replace")
                    _fixed = "\n".join(
                        ("\\ No newline at end of file" if ln.strip() == "No newline at end of file" else ln)
                        for ln in _ptxt.splitlines()
                    )
                    if not _fixed.endswith("\n"):
                        _fixed += "\n"
                    norm_patch = os.path.join(tmpdir, "gold_norm.patch")
                    Path(norm_patch).write_text(_fixed, "utf-8")
                except Exception as _pe:
                    logger.warning(f"  {iid}: patch normalize failed ({_pe}); using raw patch")
                    norm_patch = patch_file

                patch_file = norm_patch
                for operation, arguments in (("check", ["git", "apply", "--check", patch_file]),
                                             ("apply", ["git", "apply", patch_file])):
                    patch_result = _run_logged(arguments, inst_out_dir / f"patch_{operation}_raw.json", 120, extract_dir)
                    if patch_result["returncode"] != 0:
                        logger.error(f"{iid}: patch {operation} failed: {patch_result['stderr']}")
                        return {"instance_id": iid, "error": f"patch_{operation}_failed"}
            epmr = not (run_buggy or run_fuzz)
            epmr_metadata = {}
            if epmr:
                epmr_metadata = instrument_patched_repository(extract_dir, gold_instr_dir, rec, inst_out_dir / "preparation.jsonl")
            else:
                logger.warning(f"{iid}: optional buggy/fuzz comparison uses legacy markers, not latest-keypoint EPMR")
                instrument_repo_markers(extract_dir, gold_instr_dir)

        gold_workspace = os.path.join(tmpdir, "gold_ws")
        os.makedirs(gold_workspace)
        if epmr:
            runtime_directory = Path(gold_workspace, "epmr-runtime")
            runtime_directory.mkdir()
            runtime_directory.chmod(0o777)
        Path(gold_workspace, "eh_run_pytest.py").write_text(EH_PYTEST_WRAPPER, "utf-8")
        buggy_workspace = os.path.join(tmpdir, "buggy_ws")
        if run_buggy:
            os.makedirs(buggy_workspace)
            Path(buggy_workspace, "eh_run_pytest.py").write_text(EH_PYTEST_WRAPPER, "utf-8")

        _, fuzz_nodeids = _fuzz_test_args(rec, fuzz_limit) if run_fuzz else ("", [])
        fuzz_root = inst_out_dir / "fuzz"
        fuzz_specs = []
        for index, nodeid in enumerate(fuzz_nodeids):
            run_dir = fuzz_root / f"{index:03d}"
            run_dir.mkdir(parents=True)
            workspace = os.path.join(tmpdir, f"fuzz_ws_{index:03d}")
            os.makedirs(workspace)
            Path(workspace, "eh_run_pytest.py").write_text(EH_PYTEST_WRAPPER, "utf-8")
            fuzz_specs.append({"index": index, "nodeid": nodeid, "run_dir": run_dir, "workspace": workspace})

        logger.info(f"{iid}: running GOLD" + (" + BUGGY" if run_buggy else "")
                    + (f" + normal tests ({len(fuzz_specs)})" if fuzz_specs else ""))

        def _run_gold():
            return _docker_run_with_markers(
                image=image, before_cmd=before_cmd, test_args=test_args, workdir=workdir,
                patch_file=patch_file, test_patch_file=test_patch_file,
                want_gold=True, workspace_dir=gold_workspace, instr_dir=gold_instr_dir, epmr=epmr,
                deleted_files=deleted_files, timeout_seconds=timeout_seconds,
            )

        def _run_buggy():
            return _docker_run_with_markers(
                image=image, before_cmd=before_cmd, test_args=test_args, workdir=workdir,
                patch_file=patch_file, test_patch_file=test_patch_file,
                want_gold=False, workspace_dir=buggy_workspace, instr_dir=buggy_instr_dir, timeout_seconds=timeout_seconds,
            )

        def _run_fuzz(spec):
            return _docker_run_with_markers(
                image=image, before_cmd=before_cmd, test_args=shlex.quote(spec["nodeid"]), workdir=workdir,
                patch_file=patch_file, test_patch_file=test_patch_file,
                want_gold=False, workspace_dir=spec["workspace"], instr_dir=buggy_instr_dir, timeout_seconds=timeout_seconds,
            )

        from concurrent.futures import ThreadPoolExecutor as _TPE
        fuzz_workers = min(4, len(fuzz_specs))
        def run_and_save(function, path, *arguments):
            with logger.contextualize(gold_instance=iid):
                result = function(*arguments)
                _save_log(path, result)
                if epmr and path.name == "gold.log":
                    with (inst_out_dir / "preparation.jsonl").open("a", encoding="utf-8") as stream:
                        for raw_path in sorted(Path(gold_workspace, "epmr-runtime").rglob("*")):
                            if raw_path.is_file():
                                stream.write(json.dumps({"stage": "gold_runtime", "file": str(raw_path.relative_to(gold_workspace)),
                                                         "content": raw_path.read_text(encoding="utf-8")}, ensure_ascii=False) + "\n")
                return result

        with _TPE(max_workers=1 + int(run_buggy) + fuzz_workers) as pool:
            gold_future = pool.submit(run_and_save, _run_gold, inst_out_dir / "gold.log")
            buggy_future = pool.submit(run_and_save, _run_buggy, inst_out_dir / "buggy.log") if run_buggy else None
            fuzz_futures = [(spec, pool.submit(run_and_save, _run_fuzz, spec["run_dir"] / "run.log", spec)) for spec in fuzz_specs]
            gold_result = gold_future.result()
            buggy_result = buggy_future.result() if buggy_future else {"returncode": None}
            fuzz_runs = [{**spec, "result": future.result()} for spec, future in fuzz_futures]

        gold_rc = gold_result["returncode"]
        buggy_rc = buggy_result["returncode"]
        gold_trace = _parse_markers_from_stderr(gold_result.get("stderr", ""))
        buggy_trace = _parse_markers_from_stderr(buggy_result.get("stderr", ""))
        normal_fuzz_results = []
        for run in fuzz_runs:
            run["trace"] = _parse_markers_from_stderr(run["result"].get("stderr", ""))
            run["included"] = run["result"]["returncode"] == 0 and _valid_marker_trace(run["trace"])
            if run["included"]:
                normal_fuzz_results.append(run)
        fuzz_traces = [run["trace"] for run in normal_fuzz_results]
        fuzz_trace = {
            "normal_run_count": len(normal_fuzz_results),
            "excluded_run_count": len(fuzz_runs) - len(normal_fuzz_results),
            "branch_markers": list(dict.fromkeys(
                marker for trace in fuzz_traces for marker in trace.get("branch_markers", [])
            )),
            "raise_markers": list(dict.fromkeys(
                marker for trace in fuzz_traces for marker in trace.get("raise_markers", [])
            )),
            "events": [event for trace in fuzz_traces for event in trace.get("events", [])],
            "truncated": [item for trace in fuzz_traces for item in trace.get("truncated", [])],
        }

        metrics = _basic_trace_stats(gold_trace, buggy_trace) if run_buggy else {}

        result = {
            "instance_id": iid,
            **epmr_metadata,
            "gold_test_rc": gold_rc,
            **{key: value for key, value in gold_result.items() if key not in ("stdout", "stderr", "returncode")},
            "buggy_test_rc": buggy_rc,
            "gold_trace": gold_trace,
            "buggy_trace": buggy_trace,
            "metrics": metrics,
        }
        if fuzz_runs:
            result["fuzz_test_rc"] = 0 if all(run["result"]["returncode"] == 0 for run in fuzz_runs) else 1
            result["fuzz_trace"] = fuzz_trace
            result["fuzz_inputs"] = [
                {"index": run["index"], "nodeid": run["nodeid"], "returncode": run["result"]["returncode"], "normal_baseline_included": run["included"]}
                for run in fuzz_runs
            ]

        if epmr:
            from evaluate_direct_keypoints import stream_points
            from itertools import islice
            points, integrity = stream_points(inst_out_dir / "gold.log", [])
            result["gold_stream_integrity"] = integrity
            logger.info(f"{iid}: EPMR events={integrity['raw_events']}; first 2 point counts: {list(islice(points.items(), 2))}")
            if not integrity["complete"]:
                result["gold_trace_warning"] = "incomplete_epmr_stream"
        elif not gold_trace["branch_markers"] and gold_rc == 0:
            result["gold_trace_warning"] = "no markers captured despite rc=0"
        if not buggy_trace["branch_markers"] and buggy_rc == 1:
            result["buggy_trace_warning"] = "no markers captured despite rc=1"
        if fuzz_runs and not fuzz_trace.get("events") and result["fuzz_test_rc"] == 0:
            result["fuzz_trace_warning"] = "no markers captured despite all fuzz inputs passing"

        if not run_buggy:
            for key in ("buggy_test_rc", "buggy_trace", "metrics"):
                result.pop(key, None)
        logger.info(
            f"{iid}: gold_rc={gold_rc}, events={len(gold_trace['events'])}; first 2 events: {gold_trace['events'][:2]}"
            + (f"; buggy_rc={buggy_rc}, comparison={metrics}" if run_buggy else "")
            + (f"; normal_tests={len(normal_fuzz_results)}/{len(fuzz_runs)}" if fuzz_runs else "")
        )
        return result


def main():
    ap = argparse.ArgumentParser(description="Collect patched paths using cached offline instrumentation; rebuild only patch-touched files, without CodeQL")
    ap.add_argument("--instance-file", type=str, default=None, help="Text file with instance IDs")
    ap.add_argument("--limit", type=int, default=0, help="Max instances to process (0=all)")
    ap.add_argument("--source", type=str, default=None, help="Filter by source_dataset")
    ap.add_argument("--force", action="store_true", help="Re-run even if output exists")
    ap.add_argument("--parallel", type=int, default=1, help="Number of parallel workers (instances in parallel)")
    ap.add_argument("--fuzz-limit", type=int, default=10,
                    help="Fuzzing flow: number of pass_to_pass nodeids to run (0 or --no-fuzz to skip)")
    ap.add_argument("--with-buggy", action="store_true", help="Legacy marker mode: also collect unpatched comparison; not latest-keypoint EPMR")
    ap.add_argument("--with-fuzz", action="store_true", help="Legacy marker mode: also run selected pass_to_pass tests; not latest-keypoint EPMR")
    ap.add_argument("--no-fuzz", action="store_true", help="Compatibility flag: disable normal-input runs")
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Collection output directory")
    ap.add_argument("--timeout", type=int, default=DOCKER_TIMEOUT, help="Per-test Docker timeout in seconds (default: 1500)")
    args = ap.parse_args()

    if args.timeout < 1:
        ap.error("timeout must be positive")
    if args.limit < 0 or args.parallel < 1 or args.fuzz_limit < 0:
        ap.error("limit/fuzz-limit must be nonnegative and parallel must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr)
    logger.add(output_dir / "run.log", filter=lambda entry: not entry["extra"].get("gold_instance"))
    run_fuzz = args.with_fuzz and not args.no_fuzz and args.fuzz_limit > 0

    _, all_records = load_dataset_records(args.source)
    records = all_records
    excluded_count = sum(record["instance_id"] not in HEALBENCH_INSTANCE_IDS for record in records)
    records = [record for record in records if record["instance_id"] in HEALBENCH_INSTANCE_IDS]
    logger.info(f"Excluded {excluded_count} instances not present in the benchmark manifest")

    instance_ids = load_instance_file(args.instance_file)
    if instance_ids:
        id_set = set(instance_ids)
        records = [r for r in records if r["instance_id"] in id_set]
        logger.info(f"instance-file filter: {len(id_set)} requested → {len(records)} matched")

    if args.limit:
        records = records[:args.limit]

    logger.info(f"Processing {len(records)} instances → {output_dir} (parallel={args.parallel}, force={args.force}, buggy={args.with_buggy}, normal_tests={run_fuzz})")

    failed_ids = _ensure_docker_images(records)
    if failed_ids:
        logger.warning(f"Skipping {len(failed_ids)} instances with missing Docker images: {sorted(failed_ids)}")
        records = [r for r in records if r["instance_id"] not in failed_ids]
    if not records:
        logger.info("No instances with available Docker images.")
        return

    if args.parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(records)
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            fut_to_idx = {
                pool.submit(
                    process_instance, rec, output_dir, args.force,
                    args.fuzz_limit, run_fuzz, args.with_buggy, args.timeout,
                ): i
                for i, rec in enumerate(records)
            }
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    logger.error(f"[{idx+1}/{len(records)}] {records[idx]['instance_id']}: {e}")
                    results[idx] = {"instance_id": records[idx]["instance_id"], "error": str(e)}
                else:
                    logger.info(f"[{idx+1}/{len(records)}] {records[idx]['instance_id']}: done")
    else:
        results = []
        for i, rec in enumerate(records):
            logger.info(f"[{i+1}/{len(records)}] {rec['instance_id']}")
            r = process_instance(rec, output_dir, args.force, args.fuzz_limit, run_fuzz, args.with_buggy, args.timeout)
            results.append(r)

    n_gold_pass = sum(1 for r in results if r and r.get("gold_test_rc") == 0)
    n_buggy_fail = sum(1 for r in results if r and r.get("buggy_test_rc") not in (None, 0))
    n_trace_ok = sum(1 for r in results if r and r.get("collection_complete"))
    n_fuzz_run = sum(1 for r in results if r and r.get("fuzz_trace") is not None)
    n_fuzz_ok = sum(1 for r in results if r and r.get("fuzz_test_rc") == 0)
    logger.info(
        f"Done: {len(results)} instances, gold_pass={n_gold_pass}, "
        f"buggy_fail={n_buggy_fail}, trace_ok={n_trace_ok}, "
        f"fuzz_run={n_fuzz_run}, fuzz_pass={n_fuzz_ok}"
    )

    if results:
        gold_only_b = [r.get("metrics", {}).get("gold_only_branches", 0) for r in results if r and r.get("metrics")]
        buggy_only_b = [r.get("metrics", {}).get("buggy_only_branches", 0) for r in results if r and r.get("metrics")]
        if gold_only_b:
            logger.info(
                f"Divergent branches: gold_only avg={sum(gold_only_b)/len(gold_only_b):.1f}, "
                f"buggy_only avg={sum(buggy_only_b)/len(buggy_only_b):.1f}"
            )


if __name__ == "__main__":
    main()
