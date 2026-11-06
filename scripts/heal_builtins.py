from __future__ import annotations

import base64
import textwrap
import zlib


RUNTIME_BOOTSTRAP = r'''
import ast, atexit, dis, hashlib, json, os, sys, threading, time

_TRACE_PATH = os.environ.get("EH_RUNTIME_TRACE_FILE")
_TRACE_ROOT = os.environ.get("EH_RUNTIME_TRACE_ROOT", "").rstrip("/")
_TRACE_RUN_ID = os.environ.get("EH_RUNTIME_TRACE_RUN_ID", "")
_TRACE_SEQ = 0
_TRACE_LOCK = threading.RLock()
_TRACE_FILES = {}
_TRACE_OWNER_PID = 0
_TRACE_LOCAL = threading.local()
_HEAL_RUNTIME_LOCAL = threading.local()
_TRACE_FAILURE_STAGES = set()
_TRACE_LINE_NAMES = {}
_TRACE_SOURCE_MAPS = {}
_TRACE_MANIFEST = {}
_TRACE_VALUES = {}
_TRACE_SNAPSHOTS = {}
_TRACE_CALL_STACKS = {}
_TRACE_PATH_BUFFER = []
_TRACE_PATH_STATES = {}
_TRACE_ROUTE_ACTIVE = {}
_TRACE_ROUTE_STATES = {}
_TRACE_DECISION_STATES = {}
_TRACE_LOOP_STATES = {}
_TRACE_CONTROL_STACKS = {}
_TRACE_EXCEPTION_STATES = {}
_TRACE_PATH_BUFFER_SIZE = 1000
_PATH_POINTS_ENABLED = os.environ.get("EH_PATH_POINTS_ENABLED") == "1"
_PATH_POINTS = {"files": {}}
_PATH_FUNCTION_POINTS = {}
_GUARD_MANIFEST = {"files": {}}
_HEAL_POLICY = {}
_GUARD_AST = {}
_GUARD_FRAME_STATE = {}
_GUARD_GLOBAL_LABELS = {}
_GUARD_RETURN_LABELS = {}
_GUARD_HEAL_ACTIVE = False
_GUARD_PENDING_HEALS = {}
_GUARD_HEAL_SEQUENCE = 0
_EPMR_ENABLED = os.environ.get("EH_EPMR_ENABLED") == "1"
_EPMR_SEQ = 0
_EPMR_DIGEST = hashlib.sha256()
_PATH_RECORDING = _EPMR_ENABLED
_GUARD_HEALED_OBJECTS = {}


class HealedSinkBlocked(BaseException):
    pass


class ForbiddenHealCode(BaseException):
    pass


def _ignored_name(name):
    return name.startswith("_eh_") or (name.startswith("__") and name.endswith("__"))


def _monotonic_ns():
    monotonic_ns = getattr(time, "monotonic_ns", None)
    if monotonic_ns is not None:
        return monotonic_ns()
    return int(time.monotonic() * 1000000000)


def _report_failure(stage, error):
    if stage in _TRACE_FAILURE_STAGES:
        return
    _TRACE_FAILURE_STAGES.add(stage)
    sys.stderr.write("__EH_RUNTIME_TRACE_%s_FAIL__ %s: %s\n" % (stage, type(error).__name__, error))


def _trace_fallback_repr(value, seen=None):
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return "<cycle %s>" % type(value).__name__
    if isinstance(value, str):
        return str.__repr__(value)
    if isinstance(value, bytes):
        return bytes.__repr__(value)
    if value is None:
        return "None"
    if type(value) is bool:
        return bool.__repr__(value)
    if type(value) is int:
        return int.__repr__(value)
    if type(value) is float:
        return float.__repr__(value)
    if type(value) is complex:
        return complex.__repr__(value)
    seen.add(value_id)
    try:
        if isinstance(value, dict):
            items = list(dict.copy(value).items())
            return "{" + ", ".join(
                "%s: %s" % (_trace_fallback_repr(key, seen), _trace_fallback_repr(item, seen))
                for key, item in items
            ) + "}"
        if isinstance(value, (list, tuple)):
            items = list(value)
            content = ", ".join(_trace_fallback_repr(item, seen) for item in items)
            return "%s([%s])" % (type(value).__name__, content)
        if isinstance(value, set):
            items = list(set.copy(value))
            content = ", ".join(_trace_fallback_repr(item, seen) for item in items)
            return "%s([%s])" % (type(value).__name__, content)
        if isinstance(value, frozenset):
            items = list(frozenset.copy(value))
            content = ", ".join(_trace_fallback_repr(item, seen) for item in items)
            return "%s([%s])" % (type(value).__name__, content)
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            attributes = dict.copy(attributes)
            return "<%s.%s %s>" % (
                type(value).__module__,
                type(value).__qualname__,
                _trace_fallback_repr(attributes, seen),
            )
        return object.__repr__(value)
    finally:
        seen.remove(value_id)


def _trace_repr(value):
    try:
        return repr(value)
    except BaseException as repr_error:
        try:
            return _trace_fallback_repr(value)
        except BaseException as fallback_error:
            _report_failure(
                "REPR",
                RuntimeError("repr=%s; fallback=%s" % (repr_error, fallback_error)),
            )
            return "<unrepr %s: %s>" % (type(value).__name__, repr_error)


def _inside_root(filename):
    return bool(_TRACE_ROOT and filename.startswith(_TRACE_ROOT + "/"))


def _relative_path(filename):
    return filename[len(_TRACE_ROOT) + 1:]


def _load_instrumentation():
    global _TRACE_MANIFEST, _PATH_POINTS, _PATH_FUNCTION_POINTS, _GUARD_MANIFEST, _HEAL_POLICY
    instrumentation_path = os.path.join(_TRACE_ROOT, "_instrumentation.json")
    if not os.path.exists(instrumentation_path):
        _report_failure("INSTRUMENTATION", RuntimeError("_instrumentation.json missing"))
        return
    try:
        with open(instrumentation_path, "r", encoding="utf-8") as instrumentation_file:
            instrumentation = json.load(instrumentation_file)
        artifacts = instrumentation.get("artifacts", {})

        def load_artifact(name, fallback):
            artifact = artifacts.get(name)
            if not artifact:
                return instrumentation.get(name, fallback)
            artifact_path = os.path.join(_TRACE_ROOT, artifact["path"])
            with open(artifact_path, "rb") as artifact_file:
                raw = artifact_file.read()
            actual_sha256 = hashlib.sha256(raw).hexdigest()
            if actual_sha256 != artifact["sha256"]:
                raise RuntimeError("instrumentation artifact hash mismatch: %s" % name)
            return json.loads(raw)

        for file_path, line_map in load_artifact("source_map", {}).items():
            _TRACE_SOURCE_MAPS[file_path] = {int(line): original for line, original in line_map.items()}
        _TRACE_MANIFEST = load_artifact("trace_manifest", {})
        _PATH_POINTS = load_artifact("path_points", {"files": {}})
        _PATH_FUNCTION_POINTS = {}
        for file_path, file_record in _PATH_POINTS.get("files", {}).items():
            for points in file_record.get("key_points_by_line", {}).values():
                for point in points:
                    for kind in ("function_normal_exit", "function_exception_exit"):
                        if kind in point.get("key_point_kinds", []):
                            function = point.get("function", "")
                            short_qualname = function
                            for marker in ("Function ", "Class ", "Module "):
                                short_qualname = short_qualname.replace(marker, "")
                            module_name = file_path[:-3].replace("/", ".") if file_path.endswith(".py") else file_path.replace("/", ".")
                            if short_qualname.startswith(module_name + "."):
                                short_qualname = short_qualname[len(module_name) + 1:]
                            _PATH_FUNCTION_POINTS[(file_path, short_qualname, kind)] = point
        _GUARD_MANIFEST = load_artifact("guard_manifest", {"files": {}})
        _HEAL_POLICY = _GUARD_MANIFEST.get("heal_policy", instrumentation.get("heal_policy", {}))
        if _GUARD_MANIFEST.get("analysis_gaps"):
            raise RuntimeError("guard manifest contains analysis gaps")
    except BaseException as error:
        _report_failure("INSTRUMENTATION", error)


def _original_line(frame):
    relative_path = _relative_path(frame.f_code.co_filename)
    line_map = _TRACE_SOURCE_MAPS.get(relative_path)
    if line_map is None:
        return None
    return line_map.get(frame.f_lineno)


def _hash_json(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _after_fork_child():
    global _TRACE_FILES, _TRACE_OWNER_PID, _TRACE_SEQ, _TRACE_LOCK, _TRACE_LOCAL, _HEAL_RUNTIME_LOCAL, _TRACE_VALUES, _TRACE_SNAPSHOTS, _TRACE_CALL_STACKS, _TRACE_PATH_BUFFER, _TRACE_PATH_STATES, _TRACE_ROUTE_ACTIVE, _TRACE_ROUTE_STATES, _TRACE_DECISION_STATES, _TRACE_LOOP_STATES, _TRACE_CONTROL_STACKS, _TRACE_EXCEPTION_STATES, _GUARD_PENDING_HEALS, _GUARD_HEAL_ACTIVE
    _TRACE_LOCAL.active = True
    _TRACE_PATH_BUFFER = []
    for trace_file in _TRACE_FILES.values():
        trace_file.close()
    _TRACE_FILES = {}
    _TRACE_OWNER_PID = 0
    _TRACE_SEQ = 0
    _TRACE_LOCK = threading.RLock()
    _TRACE_LOCAL = threading.local()
    _HEAL_RUNTIME_LOCAL = threading.local()
    _GUARD_PENDING_HEALS = {}
    _GUARD_HEAL_ACTIVE = False
    _TRACE_VALUES = {}
    _TRACE_SNAPSHOTS = {}
    _TRACE_CALL_STACKS = {}
    _TRACE_PATH_BUFFER = []
    _TRACE_PATH_STATES = {}
    _TRACE_ROUTE_ACTIVE = {}
    _TRACE_ROUTE_STATES = {}
    _TRACE_DECISION_STATES = {}
    _TRACE_LOOP_STATES = {}
    _TRACE_CONTROL_STACKS = {}
    _TRACE_EXCEPTION_STATES = {}


def _ensure_process_files():
    global _TRACE_FILES, _TRACE_OWNER_PID, _TRACE_SEQ, _TRACE_VALUES, _TRACE_SNAPSHOTS, _TRACE_CALL_STACKS, _TRACE_PATH_STATES, _TRACE_ROUTE_ACTIVE, _TRACE_ROUTE_STATES, _TRACE_DECISION_STATES, _TRACE_LOOP_STATES
    process_id = os.getpid()
    if _TRACE_OWNER_PID == process_id:
        return
    for trace_file in _TRACE_FILES.values():
        trace_file.close()
    prefix = _TRACE_PATH + ".%d" % process_id
    _TRACE_FILES = {
        "events": open(prefix + ".events.jsonl", "a", encoding="utf-8"),
        "snapshots": open(prefix + ".snapshots.jsonl", "a", encoding="utf-8"),
        "values": open(prefix + ".values.jsonl", "a", encoding="utf-8"),
    }
    _TRACE_OWNER_PID = process_id
    _TRACE_SEQ = 0
    _TRACE_VALUES = {}
    _TRACE_SNAPSHOTS = {}
    _TRACE_CALL_STACKS = {}
    _TRACE_PATH_STATES = {}
    _TRACE_ROUTE_ACTIVE = {}
    _TRACE_ROUTE_STATES = {}
    _TRACE_DECISION_STATES = {}
    _TRACE_LOOP_STATES = {}


def _flush_path_buffer(flush=False):
    if not _TRACE_PATH_BUFFER:
        return
    trace_file = _TRACE_FILES["events"]
    trace_file.write("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in _TRACE_PATH_BUFFER))
    _TRACE_PATH_BUFFER.clear()
    if flush:
        trace_file.flush()


def _flush_path_state(thread_id):
    payload = _TRACE_PATH_STATES.pop(thread_id, None)
    if payload is None:
        return
    _TRACE_PATH_BUFFER.append({
        "run_id": _TRACE_RUN_ID,
        "seq": payload["first_seq"],
        "_time_ns": payload["first_time_ns"],
        "_pid": os.getpid(),
        "thread_id": payload["thread_id"],
        "action": "path_point",
        "point_id": payload["point_id"],
        "repeat_count": payload["repeat_count"],
        "last_repeat_seq": payload["last_repeat_seq"],
        "last_repeat_time_ns": payload["last_repeat_time_ns"],
    })


def _flush_path_repeats():
    for thread_id in list(_TRACE_PATH_STATES):
        _flush_path_state(thread_id)


def _route_scope(point_id):
    return point_id.rsplit("::", 1)[0]


def _nearest_loop_state(frame, thread_id, include_current=True):
    states = _TRACE_LOOP_STATES.get(thread_id, [])
    current = frame if include_current else frame.f_back
    while current is not None:
        frame_id = id(current)
        state = next(
            (item for item in reversed(states) if item["frame_id"] == frame_id),
            None,
        )
        if state is not None:
            return state
        current = current.f_back
    return None


def _frame_can_suspend(frame):
    return bool(frame.f_code.co_flags & (0x20 | 0x80 | 0x200))


def _emit_route_summary(thread_id, state):
    if not state or not state["path"]:
        return
    _TRACE_PATH_BUFFER.append({
        "run_id": _TRACE_RUN_ID,
        "seq": state["first_seq"],
        "_time_ns": state["first_time_ns"],
        "_pid": os.getpid(),
        "thread_id": thread_id,
        "action": "route_summary",
        "scope": state["scope"],
        "depth": state["depth"],
        "path": state["path"],
        "first_decisions": _materialize_loop_decisions(state.get("first_decisions", [])),
        "last_decisions": _materialize_loop_decisions(state.get("last_decisions", [])),
        "repeat_count": state["repeat_count"],
        "last_repeat_seq": state["last_repeat_seq"],
        "last_repeat_time_ns": state["last_repeat_time_ns"],
    })


def _finish_function_route(thread_id, frame):
    active = _TRACE_ROUTE_ACTIVE.pop(id(frame), None)
    if active is None or active.get("embedded_in_loop"):
        return
    route_key = (thread_id, active["depth"])
    state = _TRACE_ROUTE_STATES.get(route_key)
    same_route = (
        state is not None
        and state["scope"] == active["scope"]
        and state["path"] == active["path"]
        and state["decision_signature"] == active["decision_signature"]
    )
    if same_route:
        state["repeat_count"] += 1
        state["last_decisions"] = active["first_decisions"]
        state["last_repeat_seq"] = _TRACE_SEQ
        state["last_repeat_time_ns"] = _monotonic_ns()
        return
    if state is not None:
        _emit_route_summary(thread_id, state)
    active["last_decisions"] = active["first_decisions"]
    active["repeat_count"] = 1
    active["last_repeat_seq"] = _TRACE_SEQ
    active["last_repeat_time_ns"] = _monotonic_ns()
    _TRACE_ROUTE_STATES[route_key] = active


def _flush_thread_route_states(thread_id):
    for route_key, state in list(_TRACE_ROUTE_STATES.items()):
        if route_key[0] != thread_id:
            continue
        _emit_route_summary(thread_id, state)
        _TRACE_ROUTE_STATES.pop(route_key, None)


def _flush_route_states(include_active=False):
    for (thread_id, _), state in _TRACE_ROUTE_STATES.items():
        _emit_route_summary(thread_id, state)
    _TRACE_ROUTE_STATES.clear()
    if not include_active:
        return
    for active in _TRACE_ROUTE_ACTIVE.values():
        active["repeat_count"] = 1
        active["last_repeat_seq"] = _TRACE_SEQ
        active["last_repeat_time_ns"] = _monotonic_ns()
        _emit_route_summary(active["thread_id"], active)
    _TRACE_ROUTE_ACTIVE.clear()


def _emit_decision_repeats(payload):
    global _TRACE_SEQ
    if payload is None or payload["repeat_count"] == 0:
        return
    _TRACE_SEQ += 1
    record = {name: value for name, value in payload.items() if name != "signature"}
    _TRACE_PATH_BUFFER.append({
        "run_id": _TRACE_RUN_ID,
        "seq": _TRACE_SEQ,
        "_time_ns": _monotonic_ns(),
        "_pid": os.getpid(),
        "thread_id": payload["thread_id"],
        "action": "path_decision_repeat_summary",
        **record,
    })


def _flush_decision_repeats():
    for payload in _TRACE_DECISION_STATES.values():
        _emit_decision_repeats(payload)
    _TRACE_DECISION_STATES.clear()


def _emit_path_decision(thread_id, payload):
    global _TRACE_SEQ
    _TRACE_SEQ += 1
    _TRACE_PATH_BUFFER.append({
        "run_id": _TRACE_RUN_ID,
        "seq": _TRACE_SEQ,
        "_time_ns": _monotonic_ns(),
        "_pid": os.getpid(),
        "thread_id": thread_id,
        "action": "path_decision",
        **payload,
    })


def _materialize_loop_decisions(decisions):
    fields = ("decision_result", "selection", "outcome", "exception_type")
    return [
        {
            "point_id": decision["point_id"],
            **{name: decision[name] for name in fields if decision.get(name) is not None},
        }
        for decision in decisions
    ]


def _new_loop_state(frame_id, point_id, now_ns):
    return {
        "frame_id": frame_id,
        "loop_point_id": point_id,
        "path": [],
        "current_path": [],
        "decision_signature": [],
        "first_decisions": [],
        "last_decisions": [],
        "current_decisions": [],
        "current_decision_signature": [],
        "repeat_count": 0,
        "iteration_seq": _TRACE_SEQ,
        "iteration_time_ns": now_ns,
        "first_seq": _TRACE_SEQ,
        "first_time_ns": now_ns,
        "last_repeat_seq": _TRACE_SEQ,
        "last_repeat_time_ns": now_ns,
    }


def _start_loop_iteration(frame, point_id, decision_result, selection=None):
    global _TRACE_SEQ
    _epmr_event(point_id, {"result": decision_result, "selection": selection})
    with _TRACE_LOCK:
        _ensure_process_files()
        _TRACE_SEQ += 1
        thread_id = threading.get_ident()
        states = _TRACE_LOOP_STATES.setdefault(thread_id, [])
        state = next(
            (
                item for item in reversed(states)
                if item["frame_id"] == id(frame) and item["loop_point_id"] == point_id
            ),
            None,
        )
        if state is None:
            _flush_thread_route_states(thread_id)
            parent_loop = _nearest_loop_state(frame, thread_id)
            if parent_loop is not None and parent_loop["current_path"]:
                parent_loop["current_path"].append(point_id)
            state = _new_loop_state(id(frame), point_id, _monotonic_ns())
            states.append(state)
        else:
            _finish_loop_iteration(thread_id, state)
        state["iteration_seq"] = _TRACE_SEQ
        state["iteration_time_ns"] = _monotonic_ns()
        decision = {"point_id": point_id, "decision_result": decision_result}
        if selection is not None:
            decision["selection"] = selection
        state["current_path"] = [point_id]
        state["current_decisions"] = [decision]
        state["current_decision_signature"] = [(point_id, decision_result, selection)]
        return state


def _end_loop(frame, point_id, decision_result=False, selection=None):
    thread_id = threading.get_ident()
    state = _start_loop_iteration(frame, point_id, decision_result, selection)
    with _TRACE_LOCK:
        _finish_loop_iteration(thread_id, state)
        _emit_loop_summary(thread_id, state)
        states = _TRACE_LOOP_STATES.get(thread_id, [])
        if state in states:
            states.remove(state)
        if not states:
            _TRACE_LOOP_STATES.pop(thread_id, None)


class _PathIteration:
    def __init__(self, iterable, frame, point_id):
        self.iterable = iterable
        self.frame = frame
        self.point_id = point_id

    def __iter__(self):
        iterator = iter(self.iterable)
        completed = False
        try:
            while True:
                try:
                    item = next(iterator)
                except StopIteration:
                    completed = True
                    states = _TRACE_LOOP_STATES.get(threading.get_ident(), [])
                    active = any(
                        state["frame_id"] == id(self.frame) and state["loop_point_id"] == self.point_id
                        for state in states
                    )
                    if active:
                        _end_loop(self.frame, self.point_id, False, "exhausted")
                    else:
                        _start_loop_iteration(self.frame, self.point_id, False, "exhausted")
                        _flush_loop_state(threading.get_ident(), id(self.frame), self.point_id)
                    return
                _start_loop_iteration(self.frame, self.point_id, True, "entered")
                yield item
        finally:
            if not completed:
                _flush_loop_state(threading.get_ident(), id(self.frame), self.point_id)


def _emit_loop_summary(thread_id, state):
    if not state["path"]:
        return
    _TRACE_PATH_BUFFER.append({
        "run_id": _TRACE_RUN_ID,
        "seq": state["first_seq"],
        "_time_ns": state["first_time_ns"],
        "_pid": os.getpid(),
        "thread_id": thread_id,
        "action": "loop_summary",
        "loop_point_id": state["loop_point_id"],
        "path": state["path"],
        "first_decisions": _materialize_loop_decisions(state["first_decisions"]),
        "last_decisions": _materialize_loop_decisions(state["last_decisions"]),
        "repeat_count": state["repeat_count"],
        "last_repeat_seq": state["last_repeat_seq"],
        "last_repeat_time_ns": state["last_repeat_time_ns"],
    })


def _finish_loop_iteration(thread_id, state):
    current_path = state["current_path"]
    if not current_path:
        return
    now_ns = _monotonic_ns()
    current_decisions = state["current_decisions"]
    current_decision_signature = state["current_decision_signature"]
    if not state["path"]:
        state["path"] = current_path
        state["decision_signature"] = current_decision_signature
        state["first_decisions"] = current_decisions
        state["last_decisions"] = current_decisions
        state["repeat_count"] = 1
        state["first_seq"] = state["iteration_seq"]
        state["first_time_ns"] = state["iteration_time_ns"]
    elif current_path == state["path"] and current_decision_signature == state["decision_signature"]:
        state["repeat_count"] += 1
        state["last_decisions"] = current_decisions
    else:
        _flush_thread_route_states(thread_id)
        _emit_loop_summary(thread_id, state)
        state["path"] = current_path
        state["decision_signature"] = current_decision_signature
        state["first_decisions"] = current_decisions
        state["last_decisions"] = current_decisions
        state["repeat_count"] = 1
        state["first_seq"] = state["iteration_seq"]
        state["first_time_ns"] = state["iteration_time_ns"]
    state["last_repeat_seq"] = _TRACE_SEQ
    state["last_repeat_time_ns"] = now_ns
    state["current_path"] = []
    state["current_decisions"] = []
    state["current_decision_signature"] = []


def _flush_loop_state(thread_id, frame_id=None, loop_point_id=None):
    _flush_thread_route_states(thread_id)
    states = _TRACE_LOOP_STATES.get(thread_id, [])
    for index in range(len(states) - 1, -1, -1):
        state = states[index]
        if frame_id is not None and state["frame_id"] != frame_id:
            continue
        if loop_point_id is not None and state["loop_point_id"] != loop_point_id:
            continue
        _finish_loop_iteration(thread_id, state)
        _emit_loop_summary(thread_id, state)
        states.pop(index)
        break
    if not states:
        _TRACE_LOOP_STATES.pop(thread_id, None)


def _flush_loop_states():
    for thread_id, states in list(_TRACE_LOOP_STATES.items()):
        for state in states:
            _finish_loop_iteration(thread_id, state)
            _emit_loop_summary(thread_id, state)
    _TRACE_LOOP_STATES.clear()


def _write(kind, record):
    if kind == "events":
        _flush_path_buffer()
    trace_file = _TRACE_FILES[kind]
    trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    trace_file.flush()


def _value_repr_id(value_repr):
    value_id = _hash_json(value_repr)
    existing = _TRACE_VALUES.get(value_id)
    if existing is not None and existing != value_repr:
        raise RuntimeError("value hash collision")
    if existing is None:
        _TRACE_VALUES[value_id] = value_repr
        _write("values", {"value_id": value_id, "repr": value_repr})
    return value_id


def _value_id(value):
    return _value_repr_id(_trace_repr(value))


def _snapshot_repr_id(value_reprs):
    snapshot = {
        "locals": {name: _value_repr_id(value_repr) for name, value_repr in value_reprs.items()},
        "globals": {},
        "closure": {},
    }
    snapshot_id = _hash_json(snapshot)
    existing = _TRACE_SNAPSHOTS.get(snapshot_id)
    if existing is not None and existing != snapshot:
        raise RuntimeError("snapshot hash collision")
    if existing is None:
        _TRACE_SNAPSHOTS[snapshot_id] = snapshot
        _write("snapshots", {"snapshot_id": snapshot_id, **snapshot})
    return snapshot_id


def _line_names(filename, lineno):
    line_map = _TRACE_LINE_NAMES.get(filename)
    if line_map is None:
        line_map = {}
        try:
            with open(filename, "r", encoding="utf-8", errors="replace") as source_file:
                tree = ast.parse(source_file.read(), filename=filename)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    line_map.setdefault(node.lineno, set()).add(node.id)
        except BaseException as error:
            _report_failure("AST", error)
        _TRACE_LINE_NAMES[filename] = line_map
    return line_map.get(lineno, ())


def _guard_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _guard_nodes(filename):
    nodes = _GUARD_AST.get(filename)
    if nodes is not None:
        return nodes
    nodes = {}
    if not filename.endswith(".py"):
        _GUARD_AST[filename] = nodes
        return nodes
    try:
        with open(filename, "r", encoding="utf-8", errors="replace") as source_file:
            tree = ast.parse(source_file.read(), filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.stmt):
                nodes.setdefault(node.lineno, []).append(node)
    except BaseException as error:
        _report_failure("GUARD_AST", error)
    _GUARD_AST[filename] = nodes
    return nodes


def _guard_file(frame):
    relative_path = _relative_path(frame.f_code.co_filename)
    return _GUARD_MANIFEST.get("files", {}).get(relative_path, {})


def _guard_sink(frame, line):
    sinks = _guard_file(frame).get("sinks", [])
    return [sink for sink in sinks if int(sink.get("line", 0)) <= line <= int(sink.get("end_line", sink.get("line", 0)))]


def _path_file(frame):
    relative_path = _relative_path(frame.f_code.co_filename)
    return _PATH_POINTS.get("files", {}).get(relative_path, {})


def _path_function_point(frame, kind):
    relative_path = _relative_path(frame.f_code.co_filename)
    qualname = getattr(frame.f_code, "co_qualname", frame.f_code.co_name).replace(".<locals>.", ".")
    return _PATH_FUNCTION_POINTS.get((relative_path, qualname, kind))


def _path_emit_function_exit(frame, kind, outcome, value=None, exception=None):
    point = _path_function_point(frame, kind)
    if point is not None:
        _path_hit_record(frame, point["point_id"])


def _frame_is_suspended(frame):
    try:
        instruction = next(
            item for item in dis.get_instructions(frame.f_code)
            if item.offset == frame.f_lasti
        )
    except (StopIteration, ValueError, TypeError):
        return False
    return instruction.opname in {"YIELD_VALUE", "YIELD_FROM", "SEND"}


class _GuardCapturedValue:
    def __init__(self, fingerprint):
        self.fingerprint = fingerprint


def _guard_value_fingerprint(value):
    try:
        return _hash_json(_trace_repr(value))
    except BaseException:
        return "identity:%d" % id(value)


def _guard_capture_values(values):
    return {
        name: _GuardCapturedValue(_guard_value_fingerprint(value))
        for name, value in values.items()
        if not _ignored_name(name)
    }


def _guard_trackable_identity(value):
    return value is not Ellipsis and value is not NotImplemented and not isinstance(
        value,
        (type(None), bool, int, float, complex, str, bytes, tuple, frozenset),
    )


def _guard_remember_healed(value):
    if _guard_trackable_identity(value):
        _GUARD_HEALED_OBJECTS[id(value)] = value


def _guard_is_healed(value):
    return (
        _guard_trackable_identity(value)
        and _GUARD_HEALED_OBJECTS.get(id(value)) is value
    )


def _guard_changed_names(before, after):
    changed = []
    for name in sorted(set(before) | set(after)):
        if _ignored_name(name):
            continue
        if name not in before or name not in after:
            changed.append(name)
            continue
        before_value = before[name]
        after_value = after[name]
        if isinstance(before_value, _GuardCapturedValue):
            if before_value.fingerprint != _guard_value_fingerprint(after_value):
                changed.append(name)
            continue
        try:
            differs = before_value != after_value
        except BaseException:
            differs = before_value is not after_value
        if differs:
            changed.append(name)
    return changed


def _guard_local_changes(before, after):
    before_names = {name for name in before if not _ignored_name(name)}
    after_names = {name for name in after if not _ignored_name(name)}
    return {
        "added_names": sorted(after_names - before_names),
        "deleted_names": sorted(before_names - after_names),
        "changed_names": _guard_changed_names(before, after),
        "before_snapshot_id": _snapshot_mapping_id(before),
        "after_snapshot_id": _snapshot_mapping_id(after),
    }


def _guard_resolve(frame, name):
    parts = name.split(".")
    value = frame.f_locals.get(parts[0], frame.f_globals.get(parts[0]))
    for part in parts[1:]:
        try:
            value = getattr(value, part)
        except BaseException:
            return None
    return value


def _guard_frame_needs_lines(frame):
    controls = _TRACE_CONTROL_STACKS.get(id(frame), {})
    state = _GUARD_FRAME_STATE.get(id(frame), {})
    return (
        any(control.get("active") for control in controls.values())
        or bool(state.get("labels"))
        or bool(_GUARD_GLOBAL_LABELS.get(id(frame.f_globals)))
    )


def _update_frame_line_tracing(frame):
    if hasattr(frame, "f_trace_lines"):
        frame.f_trace_lines = _guard_frame_needs_lines(frame)


def _guard_expression_label(frame, node, labels):
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return (
            node.id in labels
            or node.id in _GUARD_GLOBAL_LABELS.get(id(frame.f_globals), set())
        )
    if isinstance(node, ast.Call):
        if any(_guard_expression_label(frame, item, labels) for item in [*node.args, *(keyword.value for keyword in node.keywords)]):
            return True
        callee = _guard_resolve(frame, _guard_name(node.func))
        summary = _GUARD_RETURN_LABELS.get(id(getattr(callee, "__code__", None)))
        return bool(summary and summary.get("tainted"))
    return any(_guard_expression_label(frame, child, labels) for child in ast.iter_child_nodes(node))


def _guard_targets(node):
    targets = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) or (
        hasattr(ast, "NamedExpr") and isinstance(node, ast.NamedExpr)
    ):
        targets = [node.target]
    return {item.id for target in targets for item in ast.walk(target) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)}


def _heal_validate_code(code):
    try:
        tree = ast.parse(code, filename="<heal>", mode="exec")
    except SyntaxError as error:
        raise ForbiddenHealCode("heal_syntax_error: %s" % error)
    policy = _HEAL_POLICY
    if not isinstance(policy, dict) or policy.get("default") != "allow_normal_python" or policy.get("forbidden_precedence") is not True:
        raise ForbiddenHealCode("heal_policy_missing_or_invalid")
    allowed = policy.get("allowed", {})
    forbidden = policy.get("forbidden", {})
    allowed_statements = set(allowed.get("statements", []))
    allowed_expressions = set(allowed.get("expressions", []))
    allowed_builtin_calls = set(allowed.get("builtin_calls", []))
    allowed_import_roots = set(allowed.get("import_roots", []))
    allow_project_calls = allowed.get("project_calls") == "allow"
    allow_method_calls = allowed.get("method_calls") == "allow"
    forbidden_nodes = set(forbidden.get("nodes", []))
    forbidden_calls = set(forbidden.get("calls", []))
    forbidden_module_roots = set(forbidden.get("module_roots", []))
    forbidden_attributes = set(forbidden.get("attributes", []))
    forbidden_attribute_calls = set(forbidden.get("attribute_calls", []))
    imports = {}
    for node in ast.walk(tree):
        node_name = type(node).__name__
        if node_name in forbidden_nodes:
            raise ForbiddenHealCode("heal_forbidden_node: %s" % node_name)
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                root = module.split(".", 1)[0]
                if any(module == item or module.startswith(item + ".") for item in forbidden_module_roots):
                    raise ForbiddenHealCode("heal_forbidden_module: %s" % module)
                if root not in allowed_import_roots:
                    raise ForbiddenHealCode("heal_import_not_allowed: %s" % module)
                imports[alias.asname or root] = module
            continue
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if any(module == item or module.startswith(item + ".") for item in forbidden_module_roots):
                raise ForbiddenHealCode("heal_forbidden_module: %s" % module)
            if root not in allowed_import_roots:
                raise ForbiddenHealCode("heal_import_not_allowed: %s" % module)
            for alias in node.names:
                imports[alias.asname or alias.name] = module + "." + alias.name
            continue
        if isinstance(node, ast.stmt) and node_name not in allowed_statements:
            raise ForbiddenHealCode("heal_statement_not_allowed: %s" % node_name)
        if isinstance(node, ast.expr) and not isinstance(node, (ast.Call, ast.operator, ast.boolop, ast.cmpop, ast.unaryop, ast.expr_context, ast.keyword)):
            if node_name not in allowed_expressions:
                raise ForbiddenHealCode("heal_expression_not_allowed: %s" % node_name)
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes:
            raise ForbiddenHealCode("heal_forbidden_attribute: %s" % node.attr)
        if not isinstance(node, ast.Call):
            continue
        call_name = _guard_name(node.func)
        leaf_name = call_name.rsplit(".", 1)[-1]
        root_name = call_name.split(".", 1)[0]
        resolved_call = call_name.replace(root_name, imports.get(root_name, root_name), 1)
        if call_name in forbidden_calls or leaf_name in forbidden_calls:
            raise ForbiddenHealCode("heal_forbidden_call: %s" % call_name)
        if leaf_name in forbidden_attribute_calls:
            raise ForbiddenHealCode("heal_forbidden_attribute_call: %s" % call_name)
        if any(resolved_call == item or resolved_call.startswith(item + ".") for item in forbidden_module_roots):
            raise ForbiddenHealCode("heal_forbidden_module_call: %s" % resolved_call)
        if isinstance(node.func, ast.Attribute):
            if not allow_method_calls:
                raise ForbiddenHealCode("heal_method_not_allowed: %s" % call_name)
        elif call_name not in allowed_builtin_calls and not allow_project_calls:
            raise ForbiddenHealCode("heal_project_call_not_allowed: %s" % (call_name or "<dynamic>"))
    return True


def _heal_begin_record(frame, before, code=None, crash_num=None):
    global _GUARD_HEAL_ACTIVE, _GUARD_HEAL_SEQUENCE, _PATH_RECORDING
    _GUARD_HEAL_ACTIVE = True
    _PATH_RECORDING = True
    if hasattr(frame, "f_trace_lines"):
        frame.f_trace_lines = True
    _GUARD_HEAL_SEQUENCE += 1
    heal_id = "%d:%d:%d" % (os.getpid(), threading.get_ident(), _GUARD_HEAL_SEQUENCE)
    _GUARD_PENDING_HEALS[heal_id] = {
        "frame": frame,
        "before": before,
        "crash_num": crash_num,
    }
    _HEAL_RUNTIME_LOCAL.heal_id = heal_id
    _emit(
        frame,
        "heal_before",
        include_snapshot=True,
        heal_id=heal_id,
        crash_num=crash_num,
        static_verdict="not_enforced_at_runtime",
        captured_snapshot_id=_snapshot_mapping_id(before),
    )
    return heal_id


def _take_pending_heal():
    heal_id = getattr(_HEAL_RUNTIME_LOCAL, "heal_id", None)
    if heal_id is not None:
        _HEAL_RUNTIME_LOCAL.heal_id = None
    return heal_id, _GUARD_PENDING_HEALS.pop(heal_id, None)


def _heal_failure_record(frame, before, error, crash_num=None):
    global _GUARD_HEAL_ACTIVE
    _GUARD_HEAL_ACTIVE = False
    heal_id, pending = _take_pending_heal()
    if pending is not None:
        crash_num = pending["crash_num"]
    _emit(
        frame,
        "heal_after",
        include_snapshot=True,
        heal_id=heal_id,
        crash_num=crash_num,
        outcome="exception",
        **_guard_local_changes(before, frame.f_locals),
        exception_type=error.__class__.__module__ + "." + error.__class__.__qualname__,
        exception_value_id=_value_id(error),
    )


def _heal_success_record(
    frame,
    before,
    after,
    crash_num=None,
    before_local_fingerprints=None,
    before_global_fingerprints=None,
):
    global _GUARD_HEAL_ACTIVE
    _GUARD_HEAL_ACTIVE = False
    heal_id, pending = _take_pending_heal()
    if pending is not None:
        crash_num = pending["crash_num"]
    local_before = before_local_fingerprints or before
    labels = set(_guard_changed_names(local_before, after))
    if "_eh_rtn" in after:
        _guard_remember_healed(after["_eh_rtn"])
    state = _GUARD_FRAME_STATE.setdefault(id(frame), {"labels": set(), "previous": None})
    state["labels"].update(labels)
    for name in labels:
        if name in after:
            _guard_remember_healed(after[name])

    global_labels = set()
    if before_global_fingerprints is not None:
        global_labels.update(_guard_changed_names(before_global_fingerprints, frame.f_globals))
        _GUARD_GLOBAL_LABELS.setdefault(id(frame.f_globals), set()).update(global_labels)
        for name in global_labels:
            if name in frame.f_globals:
                _guard_remember_healed(frame.f_globals[name])

    local_changes = _guard_local_changes(before, after)
    local_changes["changed_names"] = sorted(labels)
    _emit(
        frame,
        "heal_after",
        include_snapshot=True,
        heal_id=heal_id,
        healed_names=sorted(labels),
        healed_global_names=sorted(global_labels),
        crash_num=crash_num,
        outcome="success",
        **local_changes,
    )
    return sorted(labels)


def _guard_update_labels(frame, runtime_line):
    state = _GUARD_FRAME_STATE.setdefault(id(frame), {"labels": set(), "previous": None})
    labels = state["labels"]
    previous = state.get("previous")
    if previous is not None:
        for node in _guard_nodes(frame.f_code.co_filename).get(previous, []):
            targets = _guard_targets(node)
            if not targets:
                continue
            value = getattr(node, "value", None)
            if _guard_expression_label(frame, value, labels):
                labels.update(targets)
            else:
                labels.difference_update(targets)
    state["previous"] = runtime_line


def _epmr_event(point_id, decision=None):
    global _EPMR_SEQ
    if not _EPMR_ENABLED:
        return
    with _TRACE_LOCK:
        try:
            record = {"schema": "epmr-keypoints-v1", "seq": _EPMR_SEQ, "pid": os.getpid(),
                      "thread": threading.get_ident(), "point_id": point_id}
            if decision is not None:
                record["decision"] = decision
            raw = json.dumps(record, ensure_ascii=False)
            sys.stderr.write("__EPMR__\t" + raw + "\n")
            _EPMR_DIGEST.update((raw + "\n").encode("utf-8"))
            _EPMR_SEQ += 1
        except BaseException as error:
            _report_failure("EPMR", error)


def _path_hit_record(frame, point_id):
    global _TRACE_SEQ
    if not _PATH_POINTS_ENABLED or not _TRACE_PATH or not _PATH_RECORDING or getattr(_TRACE_LOCAL, "active", False):
        return
    _epmr_event(point_id)
    with _TRACE_LOCK:
        _ensure_process_files()
        _TRACE_SEQ += 1
        thread_id = threading.get_ident()
        loop_state = _nearest_loop_state(frame, thread_id)
        frame_id = id(frame)
        active_route = _TRACE_ROUTE_ACTIVE.get(frame_id)
        own_loop = loop_state is not None and loop_state["frame_id"] == frame_id
        embedded_in_loop = loop_state is not None and not own_loop and not _frame_can_suspend(frame)
        if own_loop:
            loop_point_id = loop_state["loop_point_id"]
            iteration_point_id = (
                loop_point_id[:-len(":entry")]
                if loop_point_id.endswith(":entry")
                else loop_point_id
            ) + ":iteration"
            if point_id == iteration_point_id:
                _finish_loop_iteration(thread_id, loop_state)
                loop_state["current_path"] = [point_id]
                loop_state["iteration_seq"] = _TRACE_SEQ
                loop_state["iteration_time_ns"] = _monotonic_ns()
            else:
                loop_state["current_path"].append(point_id)
            if ":continue_statement" in point_id:
                _finish_loop_iteration(thread_id, loop_state)
            elif any(kind in point_id for kind in (":break_statement", ":return_statement", ":raise_statement")):
                _flush_loop_state(thread_id, frame_id, loop_state["loop_point_id"])
            return
        if embedded_in_loop and active_route is None and not point_id.endswith("::<entry>"):
            loop_state["current_path"].append(point_id)
            return
        if point_id.endswith("::<entry>"):
            active_route = {
                "thread_id": thread_id,
                "depth": sum(1 for route in _TRACE_ROUTE_ACTIVE.values() if route["thread_id"] == thread_id),
                "scope": _route_scope(point_id),
                "path": [point_id],
                "decision_signature": [],
                "first_decisions": [],
                "last_decisions": [],
                "first_seq": _TRACE_SEQ,
                "first_time_ns": _monotonic_ns(),
                "embedded_in_loop": embedded_in_loop,
            }
            _TRACE_ROUTE_ACTIVE[frame_id] = active_route
            if embedded_in_loop:
                loop_state["current_path"].append(point_id)
            return
        if active_route is not None:
            active_route["path"].append(point_id)
            if active_route.get("embedded_in_loop"):
                loop_state = _nearest_loop_state(frame, thread_id, include_current=False)
                if loop_state is not None and loop_state["current_path"]:
                    loop_state["current_path"].append(point_id)
                else:
                    active_route["embedded_in_loop"] = False
            if point_id.endswith("::<normal_exit>") or point_id.endswith("::<exception_exit>"):
                _finish_function_route(thread_id, frame)
                if len(_TRACE_PATH_BUFFER) >= _TRACE_PATH_BUFFER_SIZE:
                    _flush_path_buffer()
            return
        state = _TRACE_PATH_STATES.get(thread_id)
        now_ns = _monotonic_ns()
        if state is not None and state["point_id"] == point_id:
            state["repeat_count"] += 1
            state["last_repeat_seq"] = _TRACE_SEQ
            state["last_repeat_time_ns"] = now_ns
            return
        _flush_path_state(thread_id)
        _TRACE_PATH_STATES[thread_id] = {
            "thread_id": thread_id,
            "point_id": point_id,
            "repeat_count": 1,
            "first_seq": _TRACE_SEQ,
            "first_time_ns": now_ns,
            "last_repeat_seq": _TRACE_SEQ,
            "last_repeat_time_ns": now_ns,
        }
        if len(_TRACE_PATH_BUFFER) >= _TRACE_PATH_BUFFER_SIZE:
            _flush_path_buffer()


def _path_hit(point_id):
    _path_hit_record(sys._getframe(1), point_id)


def _path_control(entry_point_id, exit_point_id, start_line, end_line, has_alternative, value, is_loop=False):
    global _TRACE_SEQ
    if not _PATH_POINTS_ENABLED or not _TRACE_PATH or not _PATH_RECORDING or getattr(_TRACE_LOCAL, "active", False):
        return value
    frame = sys._getframe(1)
    thread_id = threading.get_ident()
    if is_loop:
        with _TRACE_LOCK:
            _ensure_process_files()
            _TRACE_SEQ += 1
            states = _TRACE_LOOP_STATES.setdefault(thread_id, [])
            state = next(
                (
                    item for item in reversed(states)
                    if item["frame_id"] == id(frame) and item["loop_point_id"] == entry_point_id
                ),
                None,
            )
            if bool(value):
                if state is None:
                    now_ns = _monotonic_ns()
                    states.append({
                        "frame_id": id(frame),
                        "loop_point_id": entry_point_id,
                        "path": [],
                        "current_path": [],
                        "decision_signature": [],
                        "first_decisions": [],
                        "last_decisions": [],
                        "current_decisions": [],
                        "current_decision_signature": [],
                        "repeat_count": 0,
                        "iteration_seq": _TRACE_SEQ,
                        "iteration_time_ns": now_ns,
                        "first_seq": _TRACE_SEQ,
                        "first_time_ns": now_ns,
                        "last_repeat_seq": _TRACE_SEQ,
                        "last_repeat_time_ns": now_ns,
                    })
            elif state is not None:
                _flush_loop_state(thread_id, id(frame), entry_point_id)
        return value
    _path_hit_record(frame, entry_point_id)
    state = _TRACE_CONTROL_STACKS.setdefault(id(frame), {})
    state[entry_point_id] = {
        "exit_point_id": exit_point_id,
        "start_line": start_line,
        "end_line": end_line,
        "active": bool(value),
    }
    if not bool(value) and not has_alternative:
        _path_hit_record(frame, exit_point_id)
        state.pop(entry_point_id, None)
    _update_frame_line_tracing(frame)
    return value


def _path_resolve_values(frame, names):
    values = {}
    for name in names:
        parts = name.split(".")
        if not parts:
            continue
        if parts[0] in frame.f_locals:
            value = frame.f_locals[parts[0]]
        elif parts[0] in frame.f_globals:
            value = frame.f_globals[parts[0]]
        else:
            continue
        try:
            for part in parts[1:]:
                value = getattr(value, part)
        except BaseException as error:
            values[name] = "<attribute-error %s: %s>" % (type(error).__name__, error)
            continue
        values[name] = value
    return values


def _path_decision_before(names):
    frame = sys._getframe(1)
    if not _PATH_POINTS_ENABLED or not _TRACE_PATH or not _PATH_RECORDING or getattr(_TRACE_LOCAL, "active", False):
        return None
    return {
        name: _trace_repr(value)
        for name, value in _path_resolve_values(frame, names).items()
    }


def _path_decision(point_id, before_values, value, mode, names):
    global _TRACE_SEQ
    frame = sys._getframe(1)
    if not _PATH_POINTS_ENABLED or not _TRACE_PATH or not _PATH_RECORDING or getattr(_TRACE_LOCAL, "active", False):
        return value
    if mode == "iteration":
        return _PathIteration(value, frame, point_id)
    decision_result = bool(value)
    selection = None
    if mode == "while":
        if decision_result:
            _start_loop_iteration(frame, point_id, True)
        else:
            _end_loop(frame, point_id, False)
        return decision_result
    _path_hit_record(frame, point_id)
    _epmr_event(point_id, {"result": decision_result})
    thread_id = threading.get_ident()
    decision_record = {
        "point_id": point_id,
        "decision_result": decision_result,
        "selection": selection,
    }
    decision_signature = (point_id, decision_result, selection)
    loop_states = _TRACE_LOOP_STATES.get(thread_id, [])
    loop_state = next(
        (state for state in reversed(loop_states) if state["frame_id"] == id(frame)),
        None,
    )
    if loop_state is not None:
        loop_state["current_decisions"].append(decision_record)
        loop_state["current_decision_signature"].append(decision_signature)
        return decision_result if mode == "truth" else value
    active_route = _TRACE_ROUTE_ACTIVE.get(id(frame))
    if active_route is not None:
        active_route["first_decisions"].append(decision_record)
        active_route["decision_signature"].append(decision_signature)
        if active_route.get("embedded_in_loop"):
            loop_state = _nearest_loop_state(frame, thread_id, include_current=False)
            if loop_state is not None and loop_state["current_path"]:
                loop_state["current_decisions"].append(decision_record)
                loop_state["current_decision_signature"].append(decision_signature)
            else:
                active_route["embedded_in_loop"] = False
        return decision_result if mode == "truth" else value
    state_key = ("decision", thread_id, id(frame))
    signature = (point_id, decision_result, selection)
    with _TRACE_LOCK:
        _ensure_process_files()
        previous = _TRACE_DECISION_STATES.get(state_key)
        if previous is not None and previous["signature"] == signature:
            _TRACE_SEQ += 1
            previous["repeat_count"] += 1
            previous["last_repeat_seq"] = _TRACE_SEQ
            previous["last_repeat_time_ns"] = _monotonic_ns()
            return decision_result
        _emit_decision_repeats(previous)
        payload = {
            "point_id": point_id,
            "outcome": "return",
            "decision_result": decision_result,
        }
        _TRACE_DECISION_STATES[state_key] = {
            **payload,
            "signature": signature,
            "repeat_count": 0,
            "first_seq": _TRACE_SEQ + 1,
            "first_time_ns": _monotonic_ns(),
            "thread_id": thread_id,
        }
        _emit_path_decision(thread_id, payload)
    return decision_result


def _path_exception(point_id, names):
    global _TRACE_SEQ
    frame = sys._getframe(1)
    if not _PATH_POINTS_ENABLED or not _TRACE_PATH or not _PATH_RECORDING or getattr(_TRACE_LOCAL, "active", False):
        return
    exception = sys.exc_info()[1]
    _path_hit_record(frame, point_id)
    thread_id = threading.get_ident()
    exception_type = (
        exception.__class__.__module__ + "." + exception.__class__.__qualname__
        if exception is not None else None
    )
    state_key = ("exception", thread_id, id(frame))
    signature = (point_id, exception_type)
    with _TRACE_LOCK:
        _ensure_process_files()
        exception_state = _TRACE_EXCEPTION_STATES.pop(id(frame), None)
        traced_exception = exception_state.get("exception") if exception_state else exception
        decision_record = {
            "point_id": point_id,
            "outcome": "matched_exception",
        }
        if traced_exception is not None:
            decision_record["exception_type"] = traced_exception.__class__.__module__ + "." + traced_exception.__class__.__qualname__
        _epmr_event(point_id, {key: value for key, value in decision_record.items() if key != "point_id"})
        loop_state = _nearest_loop_state(frame, thread_id)
        if loop_state is not None:
            loop_state["current_decisions"].append(decision_record)
            loop_state["current_decision_signature"].append(signature)
            return
        active_route = _TRACE_ROUTE_ACTIVE.get(id(frame))
        if active_route is not None:
            active_route["first_decisions"].append(decision_record)
            active_route["decision_signature"].append(signature)
            if active_route.get("embedded_in_loop"):
                parent_loop = _nearest_loop_state(frame, thread_id, include_current=False)
                if parent_loop is not None and parent_loop["current_path"]:
                    parent_loop["current_decisions"].append(decision_record)
                    parent_loop["current_decision_signature"].append(signature)
                else:
                    active_route["embedded_in_loop"] = False
            return
        previous = _TRACE_DECISION_STATES.get(state_key)
        if previous is not None and previous["signature"] == signature:
            _TRACE_SEQ += 1
            previous["repeat_count"] += 1
            previous["last_repeat_seq"] = _TRACE_SEQ
            previous["last_repeat_time_ns"] = _monotonic_ns()
            return
        _emit_decision_repeats(previous)
        payload = decision_record
        _TRACE_DECISION_STATES[state_key] = {
            **payload,
            "signature": signature,
            "repeat_count": 0,
            "first_seq": _TRACE_SEQ + 1,
            "first_time_ns": _monotonic_ns(),
            "thread_id": thread_id,
        }
        _emit_path_decision(thread_id, payload)


def _guard_sink_call(sink, function, caller_locals, *args, **kwargs):
    frame = sys._getframe(1)
    before_locals = dict(caller_locals)
    argument_objects = [*args, *kwargs.values()]
    argument_value_ids = [_value_id(value) for value in args]
    keyword_value_ids = {name: _value_id(value) for name, value in kwargs.items()}
    healed_argument_indexes = [
        index for index, value in enumerate(args) if _guard_is_healed(value)
    ]
    healed_keyword_names = sorted(
        name for name, value in kwargs.items() if _guard_is_healed(value)
    )
    _emit(
        frame, "sink_before", include_snapshot=True, sink=sink,
        actual_call_target=sink.get("actual_call_target"), positional_argument_value_ids=argument_value_ids,
        keyword_argument_value_ids=keyword_value_ids, healed_argument_indexes=healed_argument_indexes,
        healed_keyword_names=healed_keyword_names,
        caller_snapshot_id=_snapshot_mapping_id(before_locals),
    )
    try:
        result = function(*args, **kwargs)
    except BaseException as error:
        _emit(
            frame, "sink_after", include_snapshot=True, sink=sink, outcome="exception",
            caller_snapshot_id=_snapshot_mapping_id(frame.f_locals),
            **_guard_local_changes(before_locals, frame.f_locals),
            exception_type=error.__class__.__module__ + "." + error.__class__.__qualname__,
            exception_value_id=_value_id(error),
            mutated_argument_indexes=[index for index, value in enumerate(argument_objects) if _value_id(value) != (argument_value_ids + list(keyword_value_ids.values()))[index]],
        )
        raise
    after_argument_value_ids = [_value_id(value) for value in argument_objects]
    before_argument_value_ids = argument_value_ids + list(keyword_value_ids.values())
    _emit(
        frame, "sink_after", include_snapshot=True, sink=sink, outcome="return",
        return_value_id=_value_id(result), caller_snapshot_id=_snapshot_mapping_id(frame.f_locals),
        **_guard_local_changes(before_locals, frame.f_locals),
        mutated_argument_indexes=[index for index, value_id in enumerate(after_argument_value_ids) if value_id != before_argument_value_ids[index]],
    )
    return result


def _snapshot_mapping_id(values):
    local_refs = {
        name: _value_id(value)
        for name, value in values.items()
        if not _ignored_name(name)
    }
    snapshot = {"locals": local_refs, "globals": {}, "closure": {}}
    snapshot_id = _hash_json(snapshot)
    existing = _TRACE_SNAPSHOTS.get(snapshot_id)
    if existing is not None and existing != snapshot:
        raise RuntimeError("snapshot hash collision")
    if existing is None:
        _TRACE_SNAPSHOTS[snapshot_id] = snapshot
        _write("snapshots", {"snapshot_id": snapshot_id, **snapshot})
    return snapshot_id


def _snapshot_id(frame, include_nonlocal):
    local_refs = {
        name: _value_id(value)
        for name, value in frame.f_locals.items()
        if not _ignored_name(name)
    }
    global_refs = {}
    closure_refs = {}
    if include_nonlocal:
        global_refs = {
            name: _value_id(value)
            for name, value in frame.f_globals.items()
            if not _ignored_name(name)
        }
        closure_refs = {
            name: _value_id(frame.f_locals[name])
            for name in frame.f_code.co_freevars
            if not _ignored_name(name) and name in frame.f_locals
        }
    snapshot = {"locals": local_refs, "globals": global_refs, "closure": closure_refs}
    snapshot_id = _hash_json(snapshot)
    existing = _TRACE_SNAPSHOTS.get(snapshot_id)
    if existing is not None and existing != snapshot:
        raise RuntimeError("snapshot hash collision")
    if existing is None:
        _TRACE_SNAPSHOTS[snapshot_id] = snapshot
        _write("snapshots", {"snapshot_id": snapshot_id, **snapshot})
    return snapshot_id


def _guard_call_stack(frame):
    stack = []
    current = frame
    while current is not None:
        filename = current.f_code.co_filename
        stack.append({
            "file": _relative_path(filename) if _inside_root(filename) else filename,
            "function": current.f_code.co_name,
            "line": current.f_lineno,
        })
        current = current.f_back
    return stack


def _call_stack_id(frame):
    global _TRACE_SEQ
    stack = _guard_call_stack(frame)
    stack_id = _hash_json(stack)
    existing = _TRACE_CALL_STACKS.get(stack_id)
    if existing is not None and existing != stack:
        raise RuntimeError("call stack hash collision")
    if existing is None:
        _TRACE_CALL_STACKS[stack_id] = stack
        _TRACE_SEQ += 1
        _write("events", {
            "run_id": _TRACE_RUN_ID,
            "seq": _TRACE_SEQ,
            "_time_ns": _monotonic_ns(),
            "_pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "action": "call_stack_definition",
            "call_stack_id": stack_id,
            "call_stack": stack,
        })
    return stack_id


def _emit(frame, action, include_snapshot=False, include_nonlocal=False, value_payload=None, **payload):
    global _TRACE_SEQ
    if not _TRACE_PATH or getattr(_TRACE_LOCAL, "active", False):
        return
    original_line = _original_line(frame)
    if original_line is None:
        sink = payload.get("sink") or {}
        if action not in {"sink_before", "sink_after"} or not sink.get("injected_ground_truth"):
            return
        original_line = int(sink["line"])
    _TRACE_LOCAL.active = True
    try:
        with _TRACE_LOCK:
            _ensure_process_files()
            _TRACE_SEQ += 1
            relative_path = _relative_path(frame.f_code.co_filename)
            manifest_file = _TRACE_MANIFEST.get("files", {}).get(relative_path, {})
            statement_entries = manifest_file.get("line_entries", {}).get(str(original_line), [])
            record = {
                "run_id": _TRACE_RUN_ID,
                "seq": _TRACE_SEQ,
                "_time_ns": _monotonic_ns(),
                "_pid": os.getpid(),
                "thread_id": threading.get_ident(),
                "file": relative_path,
                "func": frame.f_code.co_name,
                "line": original_line,
                "call_stack_id": _call_stack_id(frame),
                "action": action,
                "ast_statements": statement_entries,
                "try_regions": list(dict.fromkeys(
                    json.dumps(region, ensure_ascii=False, sort_keys=True)
                    for entry in statement_entries
                    for region in entry.get("regions", [])
                )),
            }
            record["try_regions"] = [json.loads(region) for region in record["try_regions"]]
            if include_snapshot:
                record["snapshot_id"] = _snapshot_id(frame, include_nonlocal)
                record["snapshot_scope"] = "locals_globals_closure" if include_nonlocal else "locals"
            record.update(payload)
            if value_payload:
                record.update({name: _value_id(value) for name, value in value_payload.items()})
            _write("events", record)
    except BaseException as error:
        try:
            with _TRACE_LOCK:
                _ensure_process_files()
                _TRACE_SEQ += 1
                _write("events", {
                    "run_id": _TRACE_RUN_ID,
                    "seq": _TRACE_SEQ,
                    "_time_ns": _monotonic_ns(),
                    "_pid": os.getpid(),
                    "thread_id": threading.get_ident(),
                    "file": _relative_path(frame.f_code.co_filename),
                    "func": frame.f_code.co_name,
                    "line": original_line,
                    "action": action,
                    "collector_fallback": "%s: %s" % (type(error).__name__, error),
                })
        except BaseException as fallback_error:
            _report_failure(
                "WRITE",
                RuntimeError("record=%s; fallback=%s" % (error, fallback_error)),
            )
    finally:
        _TRACE_LOCAL.active = False


def _tracer(frame, event, arg):
    if not _inside_root(frame.f_code.co_filename):
        return None
    if not _PATH_RECORDING and not _GUARD_HEAL_ACTIVE:
        return _tracer
    if event == "line":
        exc_pending = getattr(_TRACE_LOCAL, "exc_pending", None)
        if exc_pending is not None:
            exc_pending.pop(id(frame), None)
        control_state = _TRACE_CONTROL_STACKS.get(id(frame), {})
        original_line = _original_line(frame)
        if original_line is not None:
            for entry_point_id, control in list(control_state.items()):
                if control["active"] and not control["start_line"] <= original_line <= control["end_line"]:
                    _path_hit_record(frame, control["exit_point_id"])
                    control_state.pop(entry_point_id, None)
            _update_frame_line_tracing(frame)
        guard_state = _GUARD_FRAME_STATE.get(id(frame))
        if (guard_state and guard_state.get("labels")) or _GUARD_GLOBAL_LABELS.get(id(frame.f_globals)):
            _guard_update_labels(frame, frame.f_lineno)
    elif event == "call":
        caller = frame.f_back
        labels = set()
        if caller is not None:
            caller_state = _GUARD_FRAME_STATE.get(id(caller), {})
            caller_labels = caller_state.get("labels", set())
            global_labels = _GUARD_GLOBAL_LABELS.get(id(caller.f_globals), set())
            if caller_labels or global_labels:
                call_line = _original_line(caller) if _inside_root(caller.f_code.co_filename) else None
                for node in _guard_nodes(caller.f_code.co_filename).get(call_line, []) if call_line else []:
                    for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
                        callee = _guard_resolve(caller, _guard_name(call.func))
                        if getattr(callee, "__code__", None) is not frame.f_code:
                            continue
                        positional = [*frame.f_code.co_varnames[:frame.f_code.co_argcount]]
                        for index, argument in enumerate(call.args):
                            if index < len(positional) and _guard_expression_label(caller, argument, caller_labels):
                                labels.add(positional[index])
                        for keyword in call.keywords:
                            if keyword.arg and _guard_expression_label(caller, keyword.value, caller_labels):
                                labels.add(keyword.arg)
                        break
        if labels:
            _GUARD_FRAME_STATE[id(frame)] = {"labels": labels, "previous": None}
        if hasattr(frame, "f_trace_lines") and not labels:
            frame.f_trace_lines = False
    elif event == "return":
        if _frame_can_suspend(frame) and _frame_is_suspended(frame):
            return _tracer
        if _PATH_POINTS_ENABLED and _PATH_RECORDING and not getattr(_TRACE_LOCAL, "active", False):
            with _TRACE_LOCK:
                thread_id = threading.get_ident()
                while any(state["frame_id"] == id(frame) for state in _TRACE_LOOP_STATES.get(thread_id, [])):
                    _flush_loop_state(thread_id, id(frame))
        control_state = _TRACE_CONTROL_STACKS.pop(id(frame), {})
        _TRACE_EXCEPTION_STATES.pop(id(frame), None)
        for control in control_state.values():
            if control["active"]:
                _path_hit_record(frame, control["exit_point_id"])
        exc_pending = getattr(_TRACE_LOCAL, "exc_pending", None)
        if exc_pending is not None and id(frame) in exc_pending and arg is None:
            exception = exc_pending.pop(id(frame))
            if _PATH_POINTS_ENABLED:
                _path_emit_function_exit(frame, "function_exception_exit", "exception", exception=exception)
            return _tracer
        if _PATH_POINTS_ENABLED:
            _path_emit_function_exit(frame, "function_normal_exit", "return", value=arg)
        if _PATH_POINTS_ENABLED and _PATH_RECORDING and not getattr(_TRACE_LOCAL, "active", False):
            with _TRACE_LOCK:
                _finish_function_route(threading.get_ident(), frame)
                if len(_TRACE_PATH_BUFFER) >= _TRACE_PATH_BUFFER_SIZE:
                    _flush_path_buffer()
        state = _GUARD_FRAME_STATE.pop(id(frame), None)
        if state is not None:
            labels = state.get("labels", set())
            return_line = _original_line(frame)
            tainted = False
            for node in _guard_nodes(frame.f_code.co_filename).get(return_line, []):
                if isinstance(node, ast.Return):
                    tainted = _guard_expression_label(frame, node.value, labels)
            _GUARD_RETURN_LABELS[id(frame.f_code)] = {"tainted": tainted}
    elif event == "exception":
        exc_type, exc_value, _ = arg
        if exc_type is HealedSinkBlocked:
            raise exc_value
        if _PATH_POINTS_ENABLED and _PATH_RECORDING and not getattr(_TRACE_LOCAL, "active", False):
            with _TRACE_LOCK:
                _ensure_process_files()
                _TRACE_EXCEPTION_STATES[id(frame)] = {"exception": exc_value}
        exc_pending = getattr(_TRACE_LOCAL, "exc_pending", None)
        if exc_pending is None:
            exc_pending = {}
            _TRACE_LOCAL.exc_pending = exc_pending
        exc_pending[id(frame)] = exc_value
    return _tracer


class RuntimeGuard:
    """Dynamic part: record actual execution and block Heal-influenced confirmed Sinks."""

    validate_heal = staticmethod(_heal_validate_code)
    begin_heal = staticmethod(_heal_begin_record)
    finish_heal = staticmethod(_heal_success_record)
    fail_heal = staticmethod(_heal_failure_record)
    trace = staticmethod(_tracer)


def _close():
    global _GUARD_HEAL_ACTIVE
    for heal_id, pending in list(_GUARD_PENDING_HEALS.items()):
        frame = pending["frame"]
        _emit(
            frame,
            "heal_after",
            include_snapshot=True,
            heal_id=heal_id,
            crash_num=pending["crash_num"],
            outcome="interrupted",
            interruption_reason="process_shutdown",
            **_guard_local_changes(pending["before"], frame.f_locals),
        )
        _GUARD_PENDING_HEALS.pop(heal_id, None)
    _GUARD_HEAL_ACTIVE = False
    _TRACE_LOCAL.active = True
    with _TRACE_LOCK:
        if _TRACE_FILES:
            _flush_loop_states()
            _flush_route_states(include_active=True)
            _flush_path_repeats()
            _flush_decision_repeats()
            _flush_path_buffer(flush=True)
        for trace_file in _TRACE_FILES.values():
            trace_file.close()
    if _TRACE_PATH and _TRACE_FAILURE_STAGES:
        try:
            with open(_TRACE_PATH + ".collector_failures.json", "w", encoding="utf-8") as failure_file:
                json.dump({"failure_stages": sorted(_TRACE_FAILURE_STAGES)}, failure_file, ensure_ascii=False)
        except BaseException as error:
            sys.stderr.write("__EH_RUNTIME_TRACE_FAILURE_REPORT_FAIL__ %s: %s\n" % (type(error).__name__, error))
    if _EPMR_ENABLED:
        sys.stderr.write("__EPMR_END__\t" + json.dumps({
            "schema": "epmr-keypoints-v1", "pid": os.getpid(), "count": _EPMR_SEQ,
            "sha256": _EPMR_DIGEST.hexdigest(), "failures": sorted(_TRACE_FAILURE_STAGES),
        }, sort_keys=True) + "\n")
        sys.stderr.flush()


if _TRACE_PATH and _TRACE_ROOT and _TRACE_RUN_ID:
    try:
        import builtins
        _load_instrumentation()
        builtins._eh_validate_heal_code = _heal_validate_code
        builtins._eh_heal_begin = _heal_begin_record
        builtins._eh_heal_succeeded = _heal_success_record
        builtins._eh_heal_failed = _heal_failure_record
        builtins._eh_path_hit = _path_hit
        builtins._eh_path_decision_before = _path_decision_before
        builtins._eh_path_decision = _path_decision
        builtins._eh_path_control = _path_control
        builtins._eh_path_exception = _path_exception
        builtins._eh_guard_sink_call = _guard_sink_call
        builtins._eh_guard_block_error = HealedSinkBlocked
        builtins._eh_forbidden_heal_error = ForbiddenHealCode
        parent = os.path.dirname(_TRACE_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        _ensure_process_files()
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=_after_fork_child)
        sys.settrace(_tracer)
        threading.settrace(_tracer)
        atexit.register(_close)
        if _EPMR_ENABLED:
            sys.stderr.write("__EPMR_START__\t" + json.dumps({
                "schema": "epmr-keypoints-v1", "pid": os.getpid(),
                "runtime_fingerprint": _EPMR_RUNTIME_FINGERPRINT,
                "builder_fingerprint": _PATH_POINTS.get("builder_fingerprint"),
                "source_fingerprint": _PATH_POINTS.get("source_fingerprint"),
                "query_fingerprint": _PATH_POINTS.get("query_fingerprint"),
                "instance_id": _PATH_POINTS.get("instance_id"),
            }, sort_keys=True) + "\n")
            sys.stderr.flush()
    except BaseException as error:
        _TRACE_FILES = {}
        sys.stderr.write("__EH_RUNTIME_TRACE_INIT_FAIL__ %s: %s\n" % (type(error).__name__, error))
'''


HEAL_BUILTINS_BOOTSTRAP = r'''
import ast, contextlib, copy, ctypes, dis, io, json, os, signal, sys, textwrap, threading, traceback, urllib.request

_HEAL_LOCAL = threading.local()
_HEAL_EXECUTE_CACHE = {}
_HEAL_EXHAUSTED = set()
_HEAL_RECORD_LOCK = threading.Lock()


def _heal_smart_repr(value):
    try:
        value_repr = repr(value)
        return "(%s) %s" % (type(value).__name__, value_repr)
    except BaseException as error:
        return "<unrepr %s: %s>" % (type(value).__name__, error)


def _heal_public_locals(frame):
    return {
        name: value
        for name, value in frame.f_locals.items()
        if not name.startswith("_eh_") and not name.startswith("__")
    }


def _heal_sync_frame(frame, values):
    public_values = {
        name: value
        for name, value in values.items()
        if not name.startswith("_eh_") and name != "__builtins__"
    }
    frame.f_locals.update(public_values)
    if sys.version_info < (3, 13):
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))


def _heal_record(metadata, error, heal_code, heal_error, status):
    record_path = metadata.get("record_path")
    if not record_path:
        return
    record = {
        "file": metadata.get("file"),
        "block": metadata.get("block_index"),
        "error": type(error).__name__,
        "message": str(error),
        "heal_code": heal_code,
        "heal_error": heal_error,
        "status": status,
    }
    try:
        with _HEAL_RECORD_LOCK:
            records = []
            if os.path.exists(record_path):
                with open(record_path, encoding="utf-8") as record_file:
                    records = json.load(record_file)
            records.append(record)
            with open(record_path, "w", encoding="utf-8") as record_file:
                json.dump(records, record_file)
    except BaseException as record_error:
        sys.stderr.write("__EH_RECORD_FAIL__ %s: %s\n" % (type(record_error).__name__, record_error))


def _heal_request(metadata, payload):
    request = urllib.request.Request(
        "http://host.docker.internal:%s/repair" % metadata["port"],
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-EH-Token": metadata["token"]},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode())


def _heal_run_execute(code, namespace):
    output = io.StringIO()
    execute_value = ""
    execute_error = ""
    old_handler = None
    try:
        old_handler = signal.signal(
            signal.SIGALRM,
            lambda signum, frame: (_ for _ in ()).throw(TimeoutError("execute timeout 20s")),
        )
        signal.alarm(20)
    except BaseException as error:
        sys.stderr.write("__EH_PROBE_ALARM_FAIL__ %s: %s\n" % (type(error).__name__, error))
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                exec(compile(code, "<execute>", "exec"), namespace)
            except SyntaxError:
                execute_value = repr(eval(compile(code, "<execute>", "eval"), namespace))
    except BaseException as error:
        execute_error = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    finally:
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except BaseException as error:
            sys.stderr.write("__EH_PROBE_ALARM_RESTORE_FAIL__ %s: %s\n" % (type(error).__name__, error))
    return output.getvalue(), execute_value, execute_error


def _heal_execute(code, frame, before, crash_num=None):
    before_local_fingerprints = _guard_capture_values(frame.f_locals)
    before_global_fingerprints = _guard_capture_values(frame.f_globals)
    global_copy = dict(frame.f_globals)
    local_copy = dict(frame.f_locals)
    guard = __import__("builtins")
    if hasattr(guard, "_eh_heal_begin"):
        guard._eh_heal_begin(frame, before, code, crash_num=crash_num)
    old_alarm = 0
    old_handler = None
    try:
        old_handler = signal.signal(
            signal.SIGALRM,
            lambda signum, current: (_ for _ in ()).throw(TimeoutError("heal exec timeout 30s")),
        )
        old_alarm = signal.alarm(30)
    except BaseException as error:
        sys.stderr.write("__EH_EXEC_ALARM_FAIL__ %s: %s\n" % (type(error).__name__, error))
    try:
        exec(code, global_copy, local_copy)
        for name, value in global_copy.items():
            if not name.startswith("_eh_") and name != "__builtins__" and frame.f_globals.get(name) is not value:
                frame.f_globals[name] = value
        _heal_sync_frame(frame, local_copy)
        if hasattr(guard, "_eh_heal_succeeded"):
            guard._eh_heal_succeeded(
                frame,
                before,
                local_copy,
                crash_num=crash_num,
                before_local_fingerprints=before_local_fingerprints,
                before_global_fingerprints=before_global_fingerprints,
            )
        return local_copy
    except BaseException as error:
        if hasattr(guard, "_eh_heal_failed"):
            guard._eh_heal_failed(frame, before, error, crash_num=crash_num)
        raise
    finally:
        try:
            signal.alarm(0)
            if old_alarm:
                signal.alarm(old_alarm)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except BaseException as error:
            sys.stderr.write("__EH_EXEC_ALARM_RESTORE_FAIL__ %s: %s\n" % (type(error).__name__, error))


def _eh_handle_exception(error, metadata, frame):
    if isinstance(error, (SystemExit, KeyboardInterrupt, TimeoutError, RecursionError, StopIteration)):
        raise error
    if not os.environ.get("_EH_ACTIVE") or getattr(_HEAL_LOCAL, "active", False):
        raise error
    traceback_item = error.__traceback__
    if traceback_item is not None:
        try:
            if traceback_item.tb_frame.f_code.co_code[traceback_item.tb_lasti] == dis.opmap.get("RAISE_VARARGS", -1):
                raise error
        except (AttributeError, IndexError, TypeError) as inspect_error:
            sys.stderr.write("__EH_RAISE_OPCODE_CHECK_FAIL__ %s: %s\n" % (type(inspect_error).__name__, inspect_error))
    if None in _HEAL_EXHAUSTED:
        raise error

    _HEAL_LOCAL.active = True
    exact_key = (
        metadata.get("file"), metadata.get("line"), metadata.get("block_index"),
        type(error).__name__, str(error),
    )
    loose_key = (
        metadata.get("file"), metadata.get("line"), metadata.get("block_index"),
        type(error).__name__,
    )
    heal_code = ""
    try:
        cached = _HEAL_EXECUTE_CACHE.get(exact_key) or _HEAL_EXECUTE_CACHE.get(loose_key)
        if cached == "__RERAISE__":
            raise error
        before = dict(frame.f_locals)
        if cached:
            try:
                replay_reply = _heal_request(metadata, {
                    "phase": "replay_crash",
                    "file": metadata.get("file"),
                    "line": metadata.get("line"),
                    "block_index": metadata.get("block_index"),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "locals": {name: _heal_smart_repr(value) for name, value in _heal_public_locals(frame).items()},
                    "code": metadata.get("statement", ""),
                    "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                    "replay_code": cached,
                })
                local_copy = _heal_execute(
                    cached,
                    frame,
                    before,
                    crash_num=replay_reply.get("crash_num"),
                )
                _heal_record(metadata, error, cached, "", "replay")
                return "_eh_rtn" in local_copy, local_copy.get("_eh_rtn")
            except BaseException as replay_error:
                sys.stderr.write("__EH_REPLAY_FAIL__ %s: %s\n" % (type(replay_error).__name__, replay_error))
                _HEAL_EXECUTE_CACHE.pop(exact_key, None)
                _HEAL_EXECUTE_CACHE.pop(loose_key, None)

        state = {name: _heal_smart_repr(value) for name, value in _heal_public_locals(frame).items()}
        execute_namespace = dict(frame.f_globals)
        execute_namespace.update(_heal_public_locals(frame))
        phase = "crash"
        extras = {}
        while True:
            payload = {
                "phase": phase,
                "file": metadata.get("file"),
                "line": metadata.get("line"),
                "block_index": metadata.get("block_index"),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "locals": state,
                "code": metadata.get("statement", ""),
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            }
            payload.update(extras)
            try:
                reply = _heal_request(metadata, payload)
            except BaseException as request_error:
                _heal_record(metadata, error, heal_code, str(request_error), "failed")
                raise error
            action = reply.get("action", "reraise")
            code = (reply.get("code") or "").strip()
            submit = bool(reply.get("submit", False))
            crash_num = reply.get("crash_num")
            if action == "execute" and not submit:
                stdout, value, execute_error = _heal_run_execute(code, execute_namespace)
                phase = "execute_result"
                extras = {
                    "execute_stdout": stdout,
                    "execute_value": value,
                    "execute_error": execute_error,
                    "execute_code": code,
                    "submit": False,
                }
                continue
            if action == "reraise":
                _HEAL_EXECUTE_CACHE[exact_key] = "__RERAISE__"
                _HEAL_EXECUTE_CACHE[loose_key] = "__RERAISE__"
                raise error
            if action != "execute" or not submit:
                _heal_record(metadata, error, code, "unknown_action:%s" % action, "failed")
                raise error
            heal_code = textwrap.dedent(code)
            try:
                ast.parse("def _eh_wrap():\n" + "\n".join("    " + line for line in heal_code.splitlines()))
                local_copy = _heal_execute(heal_code, frame, before, crash_num=crash_num)
            except BaseException as execute_error:
                phase = "execute_result"
                extras = {
                    "execute_stdout": "",
                    "execute_value": "",
                    "execute_error": "".join(
                        traceback.format_exception(
                            type(execute_error),
                            execute_error,
                            execute_error.__traceback__,
                        )
                    ),
                    "execute_code": heal_code,
                    "submit": True,
                }
                continue
            _HEAL_EXECUTE_CACHE[exact_key] = heal_code
            if heal_code.strip() in {
                "_eh_rtn = None", "_eh_rtn = []", "_eh_rtn = {}", "_eh_rtn = ''",
                '_eh_rtn = ""', "_eh_rtn = 0", "_eh_rtn = True", "_eh_rtn = False", "pass",
            }:
                _HEAL_EXECUTE_CACHE[loose_key] = heal_code
            _heal_record(metadata, error, heal_code, "", "applied")
            return "_eh_rtn" in local_copy, local_copy.get("_eh_rtn")
        _HEAL_EXHAUSTED.add(None)
        _heal_record(metadata, error, heal_code, "maximum heal loop reached", "failed")
        raise error
    finally:
        _HEAL_LOCAL.active = False


__import__("builtins")._eh_handle_exception = _eh_handle_exception
'''


def build_builtins_setup_cmd() -> str:
    import hashlib
    runtime_source = textwrap.dedent(RUNTIME_BOOTSTRAP)
    runtime_fingerprint = hashlib.sha256(runtime_source.encode("utf-8")).hexdigest()
    runtime_source = f"_EPMR_RUNTIME_FINGERPRINT = {runtime_fingerprint!r}\n" + runtime_source
    runtime_payload = base64.b64encode(
        zlib.compress(runtime_source.encode("utf-8"), level=9)
    ).decode("ascii")
    heal_payload = base64.b64encode(
        zlib.compress(textwrap.dedent(HEAL_BUILTINS_BOOTSTRAP).encode("utf-8"), level=9)
    ).decode("ascii")
    pth = (
        "import base64,zlib,builtins; "
        "None if getattr(builtins, '_eh_bootstrap_loaded', False) else "
        "(setattr(builtins, '_eh_bootstrap_loaded', True), "
        f"exec(zlib.decompress(base64.b64decode('{runtime_payload}')).decode('utf-8'), globals()), "
        f"exec(zlib.decompress(base64.b64decode('{heal_payload}')).decode('utf-8'), globals()))\n"
    )
    pth_payload = base64.b64encode(pth.encode("utf-8")).decode("ascii")
    writer = (
        "import base64,os,site,sys\n"
        f"data=base64.b64decode('{pth_payload}')\n"
        "candidates=[]\n"
        "try:\n    candidates+=list(site.getsitepackages())\n"
        "except Exception as error:\n    sys.stderr.write('__EH_BUILTINS_SETUP_SITE_FAIL__ %s: %s\\n' % (type(error).__name__, error))\n"
        "try:\n    candidates.append(site.getusersitepackages())\n"
        "except Exception as error:\n    sys.stderr.write('__EH_BUILTINS_SETUP_USER_SITE_FAIL__ %s: %s\\n' % (type(error).__name__, error))\n"
        "written=False\n"
        "for directory in candidates:\n"
        "    try:\n"
        "        if directory:\n"
        "            os.makedirs(directory,exist_ok=True)\n"
        "            open(os.path.join(directory,'_eh_builtins.pth'),'wb').write(data)\n"
        "            written=True\n"
        "            break\n"
        "    except Exception as error:\n"
        "        sys.stderr.write('__EH_BUILTINS_SETUP_WRITE_FAIL__ %s: %s\\n' % (type(error).__name__, error))\n"
        "if not written:\n"
        "    sys.stderr.write('__EH_BUILTINS_SETUP_FAIL__ no writable site-packages directory\\n')\n"
    )
    writer_payload = base64.b64encode(writer.encode("utf-8")).decode("ascii")
    return (
        "python -c 'import base64; "
        f'exec(base64.b64decode("{writer_payload}").decode("utf-8"))\' && '
    )
