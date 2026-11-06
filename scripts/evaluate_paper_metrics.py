from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger

from util import HEALBENCH_INSTANCE_IDS
from evaluate_epmr_summary import read_summary_path
from evaluate_usage import collect_usage, task_usage, token_pair

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ("Healer", "MiniSweAgentAdapter", "OpenHandsAdapter", "CodexAdapter")
VERSION = "paper-v12-reconciled-observed-usage"
LEGACY_EPMR_RUNTIME = "527df891f18dc73d672960faf1377c21e323dd4217dc3ef31c3f03fadfcb779f"
GOLD_GUARDED_EPMR_RUNTIME = "32f7e5034305b7f417e990e37d21e0a206ad9e21f52b1956d35c36bb39feb25e"
GOLD_ARRAY_SAFE_EPMR_RUNTIME = "df96485f6691cf25af3fb3fa327f0c21699c4337928e3438c4d78ac7635ef24a"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        temporary = stream.name
    os.replace(temporary, path)


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def execution_status(output, task_dir):
    from evaluate_results import _read_docker_output

    rc = output.get("test_rc")
    if isinstance(rc, bool) or not isinstance(rc, int):
        return None, None, "missing_test_returncode"
    if rc == 0:
        return 1, 1, "test_passed"
    if rc != 1:
        return 0, 0, "execution_failed_or_interrupted"
    log = task_dir / "run_docker.log"
    if not log.is_file():
        return 0, None, "missing_execution_log"
    text = _read_docker_output(log)
    if re.search(r"(?m)^(?:={3,} ERRORS ={3,}|INTERNALERROR>|ERROR: Failed to build|error: (?:metadata-generation-failed|subprocess-exited-with-error))", text):
        return 0, 0, "test_error"
    failures = re.findall(r"(?m)^FAILED .+? - (.+)$", text)
    failed_tests = re.findall(r"(?m)^FAILED .+$", text)
    terminal_errors = re.findall(r"(?m)^\S+\.py:\d+: ([\w.]+)(?::.*)?$", text)
    if terminal_errors and failed_tests and len(terminal_errors) == len(failed_tests):
        assertion = all(error == "AssertionError" for error in terminal_errors)
        return 0, None if assertion else 0, "assertion_completion_unverified" if assertion else "runtime_failure"
    if failures and len(failures) == len(failed_tests):
        assertion = all(re.match(r"(?:AssertionError\b|assert\s|Failed: DID NOT RAISE\b)", item) for item in failures)
        return 0, None if assertion else 0, "assertion_completion_unverified" if assertion else "runtime_failure"
    # Old pytest omits the exception from FAILED summary lines; chained errors end at the last E line.
    failed_counts = re.findall(r"(?m)^=+ .*?\b(\d+) failed\b.*=+$", text)
    if len(failed_tests) == 1 or (failed_counts and failed_counts[-1] == "1"):
        errors = re.findall(r"(?m)^E\s+([\w.]+(?:Error|Exception|Exit)|AssertionError|socket.timeout):", text)
        if errors:
            assertion = errors[-1] == "AssertionError"
            return 0, None if assertion else 0, "assertion_completion_unverified" if assertion else "runtime_failure"
        if re.search(r"(?m)^twisted\.internet\.defer\.TimeoutError:", text):
            return 0, 0, "runtime_failure"
    if "Traceback (most recent call last):" in text:
        terminal = re.findall(r"(?m)^([\w.]+(?:Error|Exception|Warning)):", text)
        if terminal and terminal[-1] != "AssertionError":
            return 0, 0, "uncaught_exception"
    errors = re.findall(r"(?m)^E\s+([\w.]+(?:Error|Exception|Exit)):", text)
    if errors and errors[-1] != "AssertionError":
        return 0, 0, "runtime_failure"
    return 0, None, "missing_failure_details"


def lcs_length(left, right):
    # Bit-parallel LCS retains ordering and multiplicity without a quadratic matrix.
    masks = {}
    for index, item in enumerate(right):
        masks[item] = masks.get(item, 0) | (1 << index)
    state = 0
    for item in left:
        matches = state | masks.get(item, 0)
        state = matches & ~(matches - ((state << 1) | 1))
    return state.bit_count()


def marker_path(trace, mapping=None):
    from evaluate_results import _parse_marker_meta

    if trace.get("truncated"):
        raise ValueError("truncated_path")
    if trace.get("errors"):
        raise ValueError("marker_collection_failed")
    events = trace.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("missing_ordered_marker_events")
    path = []
    previous = -1
    for event in events:
        seq = event.get("seq")
        if not isinstance(seq, int) or seq <= previous:
            raise ValueError("ambiguous_event_order")
        previous = seq
        meta = _parse_marker_meta(event.get("key"))
        if meta is None or meta.get("decision_line") is None:
            raise ValueError("missing_source_coordinates")
        file = meta["file"]
        line = meta["decision_line"]
        if mapping is not None and file in mapping:
            line = mapping[file].get(line)
            if line is None:
                line = ("patched_only", meta["decision_line"])
        path.append((event.get("kind"), file, meta["func"], meta["branch_type"], line, meta["source_text"]))
    return path


def read_path(text, mapping=None, *, identity_out=None):
    events = []
    thread = None
    header = None
    finished = False
    digest = hashlib.sha256()
    for line in text.splitlines():
        if line.startswith("__EH_RUNTIME_TRACE_") and "FAIL__" in line:
            raise ValueError("runtime_collector_failure")
        if not line.startswith("__EPMR"):
            continue
        if "\t" not in line:
            raise ValueError("unsupported_epmr_protocol")
        marker, raw = line.split("\t", 1)
        event = json.loads(raw)
        if not isinstance(event, dict) or event.get("schema") != "epmr-keypoints-v1":
            raise ValueError("invalid_epmr_schema")
        if marker == "__EPMR_START__":
            if header is not None:
                raise ValueError("multiple_epmr_processes")
            if any(not event.get(key) for key in ("instance_id", "runtime_fingerprint", "builder_fingerprint", "source_fingerprint", "query_fingerprint")):
                raise ValueError("missing_epmr_instrumentation_identity")
            header = event
            continue
        if header is None or finished:
            raise ValueError("epmr_event_outside_stream")
        if event.get("pid") != header.get("pid"):
            raise ValueError("multiple_epmr_processes")
        if marker == "__EPMR_END__":
            if type(event.get("count")) is not int or event["count"] != len(events) or event.get("sha256") != digest.hexdigest():
                raise ValueError("epmr_integrity_mismatch")
            if event.get("failures") != []:
                raise ValueError("runtime_collector_failure")
            finished = True
            continue
        if marker == "__EPMR_BATCH__":
            import base64
            import zlib
            if event.get("encoding") != "zlib-json-tuples-v1" or event.get("seq") != len(events):
                raise ValueError("invalid_epmr_batch_sequence")
            batch = json.loads(zlib.decompress(base64.b64decode(event["data"], validate=True)))
            if not isinstance(batch, list) or type(event.get("count")) is not int or len(batch) != event["count"]:
                raise ValueError("invalid_epmr_batch_count")
            digest.update((raw + "\n").encode("utf-8"))
            for item in batch:
                if (not isinstance(item, list) or len(item) != 3 or type(item[0]) is not int
                        or not isinstance(item[1], str) or (item[2] is not None and not isinstance(item[2], dict))):
                    raise ValueError("invalid_epmr_batch_event")
                if thread is not None and item[0] != thread:
                    raise ValueError("concurrent_epmr_threads")
                thread = item[0]
                identity = item[1]
                file = identity.split("::", 1)[0]
                def map_batch_line(match):
                    line_number = int(match.group(2))
                    translated = mapping[file].get(line_number) if mapping is not None and file in mapping else line_number
                    coordinate = str(translated) if translated is not None else f"patched_only_{line_number}"
                    return match.group(1) + coordinate + match.group(3)
                identity = re.sub(r"(::|@)(\d+)(:)", map_batch_line, identity)
                events.append((identity, json.dumps(item[2], sort_keys=True)))
            continue
        if marker != "__EPMR__" or type(event.get("seq")) is not int or event["seq"] != len(events):
            raise ValueError("invalid_epmr_event_sequence")
        if type(event.get("thread")) is not int or type(event.get("pid")) is not int or not isinstance(event.get("point_id"), str):
            raise ValueError("invalid_epmr_event")
        if thread is not None and event["thread"] != thread:
            raise ValueError("concurrent_epmr_threads")
        thread = event["thread"]
        digest.update((raw + "\n").encode("utf-8"))
        identity = event["point_id"]
        file = identity.split("::", 1)[0]
        def map_line(match):
            line_number = int(match.group(2))
            translated = mapping[file].get(line_number) if mapping is not None and file in mapping else line_number
            coordinate = str(translated) if translated is not None else f"patched_only_{line_number}"
            return match.group(1) + coordinate + match.group(3)
        identity = re.sub(r"(::|@)(\d+)(:)", map_line, identity)
        events.append((identity, json.dumps(event.get("decision"), sort_keys=True)))
    if header is None or not finished:
        raise ValueError("missing_complete_epmr_stream")
    if not events:
        raise ValueError("empty_epmr_path")
    if identity_out is not None:
        identity_out.update(header)
    return events

def has_epmr_stream(path):
    if not path.is_file():
        return False
    with path.open(encoding="utf-8", errors="replace") as stream:
        return any("__EPMR_START__" in line for line in stream)


def first_crash(trace_dir):
    events = read_json(trace_dir / "trace.json").get("events", [])
    crash = next((event for event in events if event.get("file_path")
                  and event.get("error_type") and event.get("action") in ("execute", "reraise")), None)
    if crash is None:
        raise ValueError("no_recorded_crash")
    if crash.get("crash_num") != 1:
        raise ValueError("first_crash_evidence_incomplete")
    return crash


def crash_path_offsets(left, right, crash, runtime_file, identity, patch):
    """Use logged temporal anchors, never runtime seq as an EPMR index."""
    if identity["runtime_fingerprint"] != LEGACY_EPMR_RUNTIME:
        raise ValueError("unsupported_crash_anchor_runtime")
    runtime = [json.loads(line) for line in runtime_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({event.get("thread_id") for event in runtime if "thread_id" in event}) != 1:
        raise ValueError("concurrent_crash_runtime_threads")
    if any(event.get("collector_fallback") for event in runtime):
        raise ValueError("incomplete_crash_runtime_evidence")
    if any(event.get("_pid") != identity["pid"] for event in runtime):
        raise ValueError("crash_runtime_process_mismatch")
    before = next((event for event in runtime if event.get("action") == "heal_before"), None)
    if before is None or before.get("crash_num") != 1:
        raise ValueError("first_crash_runtime_boundary_missing")
    if before.get("file") != crash["file_path"] or not before.get("func"):
        raise ValueError("first_crash_runtime_identity_mismatch")
    file, line, function = crash["file_path"], crash["line"], before["func"]
    entries = [(index, event) for index, event in enumerate(left)
               if event[0].startswith(file + "::") and event[0].endswith("::<entry>")
               and re.search(r"\." + re.escape(function) + r"@\d+:\d+::<entry>$", event[0])]
    if len(entries) != 1:
        raise ValueError("ambiguous_first_crash_invocation")
    entry_index, entry = entries[0]
    reference_entries = [index for index, event in enumerate(right) if event == entry]
    if len(reference_entries) != 1:
        raise ValueError("ambiguous_reference_crash_invocation")
    reference_entry = reference_entries[0]
    # A unique invocation and identical complete prefix establish the same dynamic call.
    if left[:entry_index + 1] != right[:reference_entry + 1]:
        raise ValueError("crash_invocation_prefix_mismatch")
    scope = entry[0].rsplit("::", 1)[0]
    routes = [event for event in runtime if event.get("action") == "route_summary"
              and event.get("scope") == scope and event.get("repeat_count") == 1]
    if len(routes) != 1:
        raise ValueError("ambiguous_crash_runtime_route")
    route = routes[0]
    entry_line = int(re.search(r"@(\d+):", entry[0]).group(1))
    if line == entry_line + 1 and before["seq"] == route["seq"] + 1:
        # No intervening runtime event: entry is the last EPMR event before this crash.
        next_event = right[reference_entry + 1] if reference_entry + 1 < len(right) else None
        continuation = f"{file}::Function {function}::{line + 1}:"
        if next_event is None or not next_event[0].startswith(continuation):
            raise ValueError("reference_crash_continuation_unresolved")
        return entry_index + 1, reference_entry + 1, "adjacent_entry_runtime_anchor"
    loop_prefix = f"{file}::Function {function}::{line}:"
    hits = [(index, event) for index, event in enumerate(left)
            if event[0].startswith(loop_prefix) and event[0].endswith(":for_statement")]
    # Unpacking fails after the first entered event, before the loop body; snapshot
    # repr calls can precede heal_before and must remain in the post-crash suffix.
    if (len(hits) != 1 or crash.get("error_type") != "ValueError"
            or not re.search(r"(?:too many|not enough) values to unpack", crash.get("error_message", ""))
            or not re.search(r"for\s+[^\n]+,\s*[^\n]+\s+in\s+.*_eh_path_decision", crash.get("traceback", ""))):
        raise ValueError("first_crash_event_boundary_unresolved")
    hit_index, hit = hits[0]
    if hit_index <= entry_index or json.loads(hit[1]) != {"result": True, "selection": "entered"}:
        raise ValueError("first_crash_loop_boundary_unresolved")
    loops = [event for event in runtime if event.get("action") == "loop_summary"
             and event.get("loop_point_id") == hit[0]]
    if (len(loops) != 1 or loops[0].get("repeat_count") != 1
            or loops[0].get("path") != [hit[0]]
            or not route["seq"] < loops[0]["seq"] < before["seq"]):
        raise ValueError("ambiguous_first_crash_loop_iteration")
    reference_id = hit[0]
    reference_hits = [(index, event) for index, event in enumerate(right) if event[0] == reference_id]
    if not reference_hits:
        # Only a single-line for-header replacement can supply a boundary alias;
        # changed events remain patched_only for LCS matching.
        current_file, old_line, new_line = None, 0, 0
        removed, added = [], []
        replacements = []
        for text in patch.splitlines() + ["@@"]:
            if not text.startswith(("-", "+")) or text.startswith(("---", "+++")):
                if len(removed) == len(added) == 1:
                    replacements.append((current_file, removed[0], added[0]))
                removed, added = [], []
            if text.startswith("+++ b/"):
                current_file = text[6:]
            elif text.startswith("@@"):
                hunk = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", text)
                if hunk:
                    old_line, new_line = map(int, hunk.groups())
            elif text.startswith("-") and not text.startswith("---"):
                removed.append((old_line, text[1:].strip()))
                old_line += 1
            elif text.startswith("+") and not text.startswith("+++"):
                added.append((new_line, text[1:].strip()))
                new_line += 1
            elif text.startswith(" "):
                old_line += 1
                new_line += 1
        replacements = [(old, new) for path, old, new in replacements if path == file
                        and old[0] == line and old[1].startswith("for ") and new[1].startswith("for ")
                        and old[1].split(" in ", 1)[0] == new[1].split(" in ", 1)[0]]
        if len(replacements) != 1:
            raise ValueError("reference_crash_line_unmapped")
        reference_id = hit[0].replace(f"::{line}:", f"::patched_only_{replacements[0][1][0]}:")
        reference_hits = [(index, event) for index, event in enumerate(right) if event[0] == reference_id]
    if not reference_hits or reference_hits[0][0] <= reference_entry or reference_hits[0][1][1] != hit[1]:
        raise ValueError("reference_crash_loop_iteration_unresolved")
    return hit_index + 1, reference_hits[0][0] + 1, "first_loop_unpack_runtime_anchor"


def verify_runtime_compatibility(left_identity, right_identity, right, record, legacy_gold_dir):
    if left_identity["runtime_fingerprint"] == right_identity["runtime_fingerprint"]:
        return "identical_runtime"
    if (left_identity["runtime_fingerprint"] != LEGACY_EPMR_RUNTIME
            or right_identity["runtime_fingerprint"] not in (GOLD_GUARDED_EPMR_RUNTIME, GOLD_ARRAY_SAFE_EPMR_RUNTIME)):
        raise ValueError("incompatible_epmr_instrumentation")
    if legacy_gold_dir is None:
        raise ValueError("gold_runtime_equivalence_evidence_required")
    from evaluate_results import parse_unified_diff
    reference = legacy_gold_dir / record["instance_id"]
    metadata = read_json(reference / "output.json")
    source = read_json(reference / "input_raw.json")
    if any(source.get(key) != record.get(key) for key in (
            "instance_id", "repo", "base_commit", "patch", "test_patch", "test_name")):
        raise ValueError("legacy_reference_input_mismatch")
    if metadata.get("gold_test_rc") != 0 or not metadata.get("collection_complete"):
        raise ValueError("legacy_reference_collection_incomplete")
    identity = {}
    path = read_path((reference / "gold.log").read_text(encoding="utf-8"),
                     parse_unified_diff(record.get("patch", ""), align_replacements=False), identity_out=identity)
    if (identity["runtime_fingerprint"] != LEGACY_EPMR_RUNTIME
            or any(identity[key] != right_identity[key] for key in ("instance_id", "builder_fingerprint", "query_fingerprint"))
            or identity["source_fingerprint"] != metadata.get("keypoints_source_fingerprint")):
        raise ValueError("legacy_reference_identity_mismatch")
    # A version allowlist alone cannot establish that collector-internal calls were absent.
    if path != right:
        raise ValueError("gold_runtime_reference_path_changed")
    return "guarded_gold_verified_identical_reference_path"


def guarded_statement_offset(right, crash, patch, entry_index):
    import ast
    import textwrap
    from unidiff import PatchSet

    if (crash.get("error_type") != "ValueError"
            or not re.search(r"(?:too many|not enough) values to unpack", crash.get("error_message", ""))):
        return None
    source = ast.parse(textwrap.dedent(crash.get("code_at_crash", "")))
    if (len(source.body) != 1 or not isinstance(source.body[0], ast.Assign)
            or len(source.body[0].targets) != 1 or not isinstance(source.body[0].targets[0], (ast.Tuple, ast.List))):
        return None
    candidates = []
    for changed_file in PatchSet(patch):
        if changed_file.path != crash["file_path"]:
            continue
        for hunk in changed_file:
            removed, added = [], []
            for line in list(hunk) + [None]:
                if line is not None and line.is_removed:
                    removed.append(line)
                elif line is not None and line.is_added:
                    added.append(line)
                else:
                    if (len(removed) == 1 and removed[0].source_line_no == crash["line"] and added
                            and ast.dump(ast.parse(textwrap.dedent(removed[0].value))) == ast.dump(source)):
                        replacement = ast.parse(textwrap.dedent("".join(item.value for item in added)))
                        guard = replacement.body[-1] if replacement.body else None
                        if (isinstance(guard, ast.If) and not guard.orelse
                                and all(isinstance(node, ast.Assign) for node in replacement.body[:-1])
                                and len(guard.body) == 1 and isinstance(guard.body[0], ast.Assign)
                                and [ast.dump(target) for target in guard.body[0].targets]
                                == [ast.dump(target) for target in source.body[0].targets]):
                            guard_line = added[0].target_line_no + guard.lineno - 1
                            prefix = f'{crash["file_path"]}::Function {crash["func_name"]}::patched_only_{guard_line}:'
                            hits = [(index, event) for index, event in enumerate(right)
                                    if event[0].startswith(prefix) and event[0].endswith(":if_statement")
                                    and event[1] != "null"]
                            # A unique false guard ends the entire replacement without
                            # entering the unpacking body; retain subsequent callee events.
                            if (len(hits) == 1 and hits[0][0] > entry_index
                                    and json.loads(hits[0][1][1]) == {"result": False}):
                                candidates.append(hits[0][0] + 1)
                    removed, added = [], []
    return candidates[0] if len(candidates) == 1 else None


def summary_reference_offset(right, crash, patch):
    from evaluate_results import parse_unified_diff

    file, function, line = crash["file_path"], crash["func_name"], crash["line"]
    entries = [(index, event) for index, event in enumerate(right)
               if event[0].startswith(file + "::")
               and re.search(r"\." + re.escape(function) + r"@(?:patched_only_)?\d+:\d+::<entry>$", event[0])]
    if not entries:
        raise ValueError("reference_crash_function_entry_missing")
    if len(entries) != 1:
        raise ValueError("ambiguous_reference_crash_invocation")
    entry_index, entry = entries[0]
    coordinate = re.search(r"@(patched_only_)?(\d+):", entry[0])
    entry_line = int(coordinate.group(2))
    if coordinate.group(1):
        entry_line = parse_unified_diff(patch, align_replacements=True).get(file, {}).get(entry_line)
    if entry_line is None:
        raise ValueError("reference_crash_function_line_unmapped")
    if line == entry_line + 1:
        continuation = f"{file}::Function {function}::{line + 1}:"
        if entry_index + 1 < len(right) and right[entry_index + 1][0].startswith(continuation):
            return entry_index + 1, "summary_activation_first_statement"
    guarded_offset = guarded_statement_offset(right, crash, patch, entry_index)
    if guarded_offset is not None:
        return guarded_offset, "summary_activation_guarded_unpack_completion"
    if (crash.get("error_type") != "ValueError"
            or not re.search(r"(?:too many|not enough) values to unpack", crash.get("error_message", ""))
            or not re.search(r"for\s+[^\n]+,\s*[^\n]+\s+in\s+.*_eh_path_decision", crash.get("traceback", ""))):
        raise ValueError("summary_reference_crash_boundary_unresolved")
    import ast
    import textwrap

    block = ast.parse(textwrap.dedent(crash.get("code_at_crash", "")))
    loop = block.body[0] if block.body else None
    targets = {node.id for node in ast.walk(loop.target) if isinstance(node, ast.Name)} if isinstance(loop, ast.For) else set()
    locals_snapshot = crash.get("locals_snapshot")
    if (not targets or not isinstance(locals_snapshot, dict) or targets.intersection(locals_snapshot)
            or any(isinstance(node, ast.Delete) for node in ast.walk(loop))):
        raise ValueError("first_summary_loop_iteration_unverified")
    aligned = parse_unified_diff(patch, align_replacements=True).get(file, {})
    aliases = [f"::{line}:"]
    for event in right:
        match = re.search(r"::patched_only_(\d+):", event[0])
        if match and aligned.get(int(match.group(1))) == line:
            aliases.append(match.group(0))
    hits = [(index, event) for index, event in enumerate(right)
            if event[0].startswith(f"{file}::Function {function}::")
            and event[0].endswith(":for_statement") and any(alias in event[0] for alias in aliases)]
    if not hits or hits[0][0] <= entry_index or json.loads(hits[0][1][1]) != {"result": True, "selection": "entered"}:
        raise ValueError("summary_reference_loop_boundary_unresolved")
    return hits[0][0] + 1, "summary_activation_first_loop_unpack"


def trajectory_prefix_offset(left, right, diagnostics):
    if not left:
        raise ValueError("empty_post_crash_path")
    # Locate a point occurrence independently of its decision value; decisions
    # still participate in LCS, so divergent branches reduce the final score.
    def location(event):
        return event[0], event[1] == "null"
    candidates = [index for index, event in enumerate(right) if location(event) == location(left[0])]
    diagnostics["epmr_prefix_initial_candidates"] = len(candidates)
    for position, event in enumerate(left):
        candidates = [index for index in candidates
                      if index + position < len(right) and location(right[index + position]) == location(event)]
        diagnostics.update(epmr_prefix_events=position + 1, epmr_prefix_candidates=len(candidates))
        if not candidates:
            raise ValueError("trajectory_start_prefix_not_found")
        if len(candidates) == 1:
            diagnostics["epmr_boundary_evidence"] = "inferred_unique_keypoint_prefix_not_crash_proven"
            return candidates[0], "unique_trajectory_start_prefix"
    raise ValueError("trajectory_start_prefix_ambiguous_or_insufficient")


def summary_path_metric(task_dir, gold_dir, record, metadata, result, legacy_gold_dir):
    from evaluate_results import _read_docker_output, parse_unified_diff

    try:
        crash = first_crash(task_dir)
        runtime = [json.loads(line) for line in (task_dir / "run_docker_events.jsonl").read_text().splitlines() if line.strip()]
        healing = [event for event in runtime if event.get("action") == "heal_before"]
        if not healing:
            raise ValueError("first_crash_runtime_boundary_missing")
        before = min(healing, key=lambda event: event["seq"])
        if before.get("crash_num") != crash["crash_num"]:
            raise ValueError("trajectory_starts_at_later_crash")
        if before.get("file") != crash["file_path"] or not before.get("func"):
            raise ValueError("first_crash_runtime_identity_mismatch")
        # The crash function can be the inner traceback frame; heal_before
        # identifies the paused caller, and its mapped line is not the crash line.
        result["epmr_activation_crash_num"] = before["crash_num"]
        result["epmr_activation_function"] = before["func"]
        result["epmr_activation_line"] = before.get("line")
        result["epmr_boundary_identity"] = "crash_number_and_file"
        anchor_crash = {**crash, "func_name": before["func"]}
        original = read_json(task_dir / "input.json")
        if any(original.get(key) != record.get(key) for key in ("instance_id", "patch", "test_patch", "test_name")):
            raise ValueError("summary_input_identity_mismatch")
        output = read_json(task_dir / "output.json")
        text = _read_docker_output(task_dir / "run_docker.log")
        if output.get("test_rc") not in (0, 1) or any("__EH_RUNTIME_TRACE_" in line and "FAIL__" in line for line in text.splitlines()):
            raise ValueError("summary_collection_incomplete")
        if metadata.get("gold_test_rc") != 0 or not metadata.get("collection_complete"):
            raise ValueError("reference_collection_incomplete")
        result.update(epmr_crash_file=crash["file_path"], epmr_crash_line=crash["line"],
                      epmr_crash_function=crash["func_name"], epmr_crash_num=1)
        left = read_summary_path(runtime, result)
        result["healed_path_events"] = len(left)
        crash_entries = [point for point, _ in left if point.startswith(crash["file_path"] + "::")
                         and re.search(r"\." + re.escape(anchor_crash["func_name"]) + r"@\d+:\d+::<entry>$", point)]
        identity = {}
        reference = gold_dir / record["instance_id"]
        right = read_path((reference / "gold.log").read_text(),
                          parse_unified_diff(record.get("patch", ""), align_replacements=False), identity_out=identity)
        if identity["instance_id"] != record["instance_id"] or identity["source_fingerprint"] != metadata.get("keypoints_source_fingerprint"):
            raise ValueError("reference_epmr_identity_mismatch")
        reference_input = read_json(reference / "input_raw.json")
        if any(reference_input.get(key) != record.get(key) for key in ("instance_id", "repo", "base_commit", "patch", "test_patch", "test_name")):
            raise ValueError("reference_input_identity_mismatch")
        # Historical summaries contain no runtime hash: expose that limitation rather
        # than inventing an identity or treating the replay as the original run.
        result["epmr_healed_runtime"] = "unversioned_runtime_summary"
        result["epmr_gold_runtime"] = identity["runtime_fingerprint"]
        if identity["runtime_fingerprint"] not in (LEGACY_EPMR_RUNTIME, GOLD_GUARDED_EPMR_RUNTIME, GOLD_ARRAY_SAFE_EPMR_RUNTIME):
            raise ValueError("unsupported_summary_reference_runtime")
        # Original runs did not emit EPMR to stderr, so collector-induced wrapper
        # calls never entered their summaries; the guarded Golden excludes the same
        # collector-only calls. Old/new full-stream equality is a replay requirement,
        # not a prerequisite for decoding an original summary without an EPMR stream.
        result["epmr_runtime_compatibility"] = "summary_event_semantics_no_historical_runtime_hash"
        try:
            if crash_entries:
                raise ValueError("summary_activation_or_repeated_crash_invocation_unresolved")
            offset, method = summary_reference_offset(right, anchor_crash, record.get("patch", ""))
            result["epmr_boundary_evidence"] = "saved_crash_and_source_boundary"
        except ValueError as error:
            logger.warning(f"{task_dir}: source boundary unavailable, trying trajectory start prefix: {error}")
            result["epmr_source_boundary_reason"] = str(error)
            offset, method = trajectory_prefix_offset(left, right, result)
        result.update(epmr_healed_start=0, epmr_patched_start=offset, epmr_boundary_method=method,
                      patched_full_path_events=len(right))
        right = right[offset:]
        if not left or not right:
            raise ValueError("empty_post_crash_path")
        matched = lcs_length(left, right)
        result.update(epmr=matched / len(left), epmr_reason="post_first_crash_original_summary_lcs",
                      epmr_matched_events=matched, patched_path_events=len(right))
    except (OSError, ValueError, KeyError, TypeError, ImportError, SyntaxError) as error:
        logger.warning(f"{task_dir}: original-summary EPMR unavailable: {error}")
        result["epmr_reason"] = str(error)
    return result


def path_metric(task_dir, gold_dir, record, replay_dir=None, legacy_gold_dir=None):
    from evaluate_results import parse_unified_diff

    result = {"epmr": None, "epmr_reason": "missing_reference_trace", "epmr_matched_events": None,
              "healed_path_events": None, "patched_path_events": None}
    gold_file = gold_dir / f"{record['instance_id']}.json"
    if not gold_file.is_file() and not (gold_dir / record['instance_id'] / "output.json").is_file():
        return result
    compact_file = gold_dir / record['instance_id'] / "output.json"
    metadata = read_json(compact_file) if compact_file.is_file() else {}
    if metadata.get("path_schema") == "epmr-keypoints-v1":
        from evaluate_results import _read_docker_output
        healed_log = task_dir / "run_docker.log"
        if not has_epmr_stream(healed_log) and (task_dir / "run_docker_events.jsonl").is_file():
            summary_path_metric(task_dir, gold_dir, record, metadata, result, legacy_gold_dir)
            if result["epmr"] is not None or replay_dir is None or not (replay_dir / "epmr_replay.json").is_file():
                return result
            result["epmr_original_summary_reason"] = result["epmr_reason"]
        if replay_dir is not None and not has_epmr_stream(healed_log):
            proof_path = replay_dir / "epmr_replay.json"
            if proof_path.is_file():
                from run_record_healed_trace import file_hash
                proof = read_json(proof_path)
                valid = proof.get("complete") and proof.get("source_dir") == str(task_dir.resolve())
                for name in ("trace.json", "input.json", "output.json"):
                    source_path = (task_dir / name).resolve()
                    valid = valid and source_path.is_file() and proof.get("signature", {}).get(str(source_path)) == file_hash(source_path)
                for name in ("run_docker.log", "trace.json", "output.json"):
                    replay_file = replay_dir / name
                    valid = valid and replay_file.is_file() and proof.get("replay_files", {}).get(name) == file_hash(replay_file)
                if not valid:
                    result["epmr_reason"] = "unverified_or_stale_healed_replay"
                    return result
                healed_log = replay_dir / "run_docker.log"
                result["epmr_path_source"] = "verified_healed_replay"
                result["epmr_integrity"] = "epmr_digest_and_replay_source_hashes"
        if not has_epmr_stream(healed_log):
            result["epmr_reason"] = "healed_epmr_recapture_required"
            return result
        if metadata.get("gold_test_rc") != 0 or not metadata.get("collection_complete"):
            result["epmr_reason"] = "reference_collection_incomplete"
            return result
        try:
            crash = first_crash(task_dir)
            result.update(epmr_crash_file=crash["file_path"], epmr_crash_line=crash["line"],
                          epmr_crash_function=crash["func_name"], epmr_crash_num=crash["crash_num"])
            left_identity, right_identity = {}, {}
            left = read_path(_read_docker_output(healed_log), identity_out=left_identity)
            right = read_path((compact_file.parent / "gold.log").read_text(encoding="utf-8"),
                              parse_unified_diff(record.get("patch", ""), align_replacements=False), identity_out=right_identity)
            if any(left_identity.get(key) != right_identity.get(key) for key in ("builder_fingerprint", "query_fingerprint", "instance_id")):
                raise ValueError("incompatible_epmr_instrumentation")
            result["epmr_runtime_compatibility"] = verify_runtime_compatibility(
                left_identity, right_identity, right, record, legacy_gold_dir)
            result["epmr_healed_runtime"] = left_identity["runtime_fingerprint"]
            result["epmr_gold_runtime"] = right_identity["runtime_fingerprint"]
            if right_identity["instance_id"] != record["instance_id"] or right_identity["source_fingerprint"] != metadata.get("keypoints_source_fingerprint"):
                raise ValueError("reference_epmr_identity_mismatch")
            left_start, right_start, method = crash_path_offsets(
                left, right, crash, healed_log.parent / "run_docker_events.jsonl", left_identity, record.get("patch", ""))
            result.update(epmr_healed_start=left_start, epmr_patched_start=right_start,
                          epmr_boundary_method=method, healed_full_path_events=len(left), patched_full_path_events=len(right))
            left, right = left[left_start:], right[right_start:]
            if not left or not right:
                raise ValueError("empty_post_crash_path")
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.warning(f"{task_dir}: EPMR unavailable: {error}")
            result["epmr_reason"] = str(error)
            return result
        matched = lcs_length(left, right)
        result.update(epmr=matched / len(left), epmr_reason="post_first_crash_keypoint_lcs",
                      epmr_matched_events=matched, healed_path_events=len(left), patched_path_events=len(right))
        return result
    result["epmr_reason"] = "legacy_trace_has_no_verified_crash_boundary"
    return result


def instance_cost(row, prices):
    if not row.get("usage_complete"):
        return None, "incomplete_usage"
    price = prices.get(row["model"])
    if price is None:
        return None, "missing_model_price"
    cost = (row["input_tokens"] * price["input_usd_per_million"]
            + row["output_tokens"] * price["output_usd_per_million"]) / 1_000_000
    return cost, "configured_list_price_estimate_no_cache_discount"


def evaluate_instance(task_dir, record, agent, model, gold_dir, prices, replay_dir=None, legacy_gold_dir=None, epmr_mode="strict"):
    row = {"agent_class": agent, "model": model, "guard_enabled": 0,
           "instance_id": record["instance_id"], "source_dataset": record["source_dataset"],
           "result_dir": str(task_dir), "evaluation_status": "missing_output",
           "is_correct": 0, "is_proceed": 0}
    output_file = task_dir / "output.json"
    if not output_file.is_file():
        return row
    output = read_json(output_file)
    if output.get("instance_id", task_dir.name) != record["instance_id"]:
        raise ValueError("output_instance_id_mismatch")
    row["raw_status"] = output.get("status")
    row["test_rc"] = output.get("test_rc")
    row.update(task_usage(task_dir, output, agent, model, prices, read_json))
    if output.get("status") == "infra_error":
        row["evaluation_status"] = "infra_error"
        return row
    input_file = task_dir / "input.json"
    if input_file.is_file():
        saved_input = read_json(input_file)
        if saved_input.get("test_code_in_prompt") and "../r2e_tests" not in (saved_input.get("test_file") or ""):
            row["evaluation_status"] = "test_code_leaked"
            return row
        if saved_input.get("patch") != record.get("patch"):
            row["evaluation_status"] = "input_patch_mismatch"
            return row
    row["evaluation_status"] = "evaluated"
    row["is_correct"], row["is_proceed"], row["execution_reason"] = execution_status(output, task_dir)
    if output.get("test_rc") == 1 and row["execution_reason"] != "test_error":
        from evaluate_pr_evidence import assertion_evidence
        from evaluate_results import _read_docker_output
        # Full failure evidence outranks abbreviated summaries and intermediate traceback frames.
        log = task_dir / "run_docker.log"
        if log.is_file():
            verdict, reason, evidence = assertion_evidence(_read_docker_output(log), record)
            if verdict is not None or row["is_proceed"] is None:
                row["is_proceed"] = verdict
                row["execution_reason"] = reason
                row["pr_evidence"] = json.dumps(evidence, ensure_ascii=False)
    if epmr_mode == "strict":
        row.update(path_metric(task_dir, gold_dir, record, replay_dir, legacy_gold_dir))
    return row


def cached_instance(task_dir, record, agent, model, gold_dir, prices, cache_dir, replay_root=None, legacy_gold_dir=None, epmr_mode="strict"):
    replay_dir = replay_root / agent / model / record["instance_id"] if replay_root is not None else None
    paths = [task_dir / name for name in ("output.json", "input.json", "trace.json", "run.log", "run_docker.log", "run_docker_events.jsonl", "healed_trace.json")]
    paths += [gold_dir / f"{record['instance_id']}.json", Path(__file__), Path(__file__).with_name("evaluate_results.py"),
              Path(__file__).with_name("evaluate_pr_evidence.py"), Path(__file__).with_name("evaluate_usage.py"),
              Path(__file__).with_name("run_record_gold_trace.py"),
              Path(__file__).with_name("heal_builtins.py")]
    paths += [gold_dir / record['instance_id'] / name for name in ("input_raw.json", "output.json", "gold.log")]
    if replay_dir is not None:
        paths += [replay_dir / name for name in ("epmr_replay.json", "run_docker.log", "run_docker_events.jsonl", "trace.json", "output.json")]
    paths.extend(Path(__file__).with_name(name) for name in ("run_record_healed_trace.py", "evaluate_epmr_summary.py"))
    if legacy_gold_dir is not None:
        paths += [legacy_gold_dir / record['instance_id'] / name for name in ("input_raw.json", "output.json", "gold.log")]
    signature = [VERSION, epmr_mode, record, agent, model, prices, str(task_dir), str(gold_dir), str(replay_dir), str(legacy_gold_dir)]
    for path in paths:
        if path.is_file():
            stat = path.stat()
            signature.append((str(path), stat.st_size, stat.st_mtime_ns))
        else:
            signature.append((str(path), None))
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_file = cache_dir / f"{key}.json"
    if cache_file.is_file():
        try:
            return read_json(cache_file)
        except (OSError, ValueError) as error:
            logger.warning(f"Cache unreadable {cache_file}: {error}")
    row = evaluate_instance(task_dir, record, agent, model, gold_dir, prices, replay_dir, legacy_gold_dir, epmr_mode)
    if row["evaluation_status"] != "missing_output":
        write_json(cache_file, row)
    return row


def load_records(path, limit):
    records = []
    excluded = []
    seen = set()
    per_dataset = Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            iid = record["instance_id"]
            if iid not in HEALBENCH_INSTANCE_IDS:
                excluded.append(iid)
                continue
            if iid in seen:
                raise ValueError(f"Duplicate dataset instance: {iid}")
            seen.add(iid)
            records.append(record)
    selected = []
    for record in sorted(records, key=lambda item: (item["source_dataset"], item["instance_id"])):
        dataset = record["source_dataset"]
        if limit and per_dataset[dataset] >= limit:
            continue
        selected.append(record)
        per_dataset[dataset] += 1
    logger.info(f"Excluded {len(excluded)} outside-manifest instances: {excluded}")
    logger.info(f"Selected {dict(per_dataset)}; first 2 full dataset records: {selected[:2]}")
    return selected, excluded


def mean_metric(rows, field):
    values = [row[field] for row in rows if number(row.get(field)) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def aggregate(rows, by_dataset=False):
    groups = defaultdict(list)
    for row in rows:
        key = (row["agent_class"], row["model"])
        if by_dataset:
            key += (row["source_dataset"],)
        groups[key].append(row)
    results = []
    for key, group in sorted(groups.items()):
        valid = [row for row in group if row["evaluation_status"] == "evaluated"]
        counts = Counter(row["evaluation_status"] for row in group)
        result = {"agent_class": key[0], "model": key[1], "guard_enabled": 0,
                  "n_expected": len(group), "n_cases": len(group), "n_evaluated": len(valid),
                  "evaluation_counts": json.dumps(dict(counts), sort_keys=True)}
        if by_dataset:
            result["source_dataset"] = key[2]
        for label, field in (("PR", "is_proceed"), ("CR", "is_correct")):
            successes = sum(row.get(field) == 1 for row in valid)
            unknown = sum(row.get(field) is None for row in valid)
            result[label] = successes / len(group)
            result[f"n_{label}"] = len(group)
            result[f"n_{label}_unverified"] = unknown
            result[f"{label}_lower_bound"] = successes / len(group)
            result[f"{label}_upper_bound"] = (successes + unknown) / len(group)
        for label, field in (("EPMR", "epmr"), ("Input_Tokens", "input_tokens"),
                             ("Output_Tokens", "output_tokens"), ("Cost_USD", "cost_usd")):
            observations = valid if field == "epmr" else group
            result[label] = mean_metric(observations, field)
            result[f"n_{label}"] = sum(number(row.get(field)) is not None for row in observations)
        for label, field in (("Observed_Input_Tokens", "observed_input_tokens"),
                             ("Observed_Output_Tokens", "observed_output_tokens"),
                             ("Observed_Cost_USD", "observed_cost_usd")):
            result[label] = mean_metric(group, field)
            result[f"n_{label}"] = sum(number(row.get(field)) is not None for row in group)
        result["usage_counts"] = json.dumps(dict(Counter(row.get("usage_reason") for row in group)), sort_keys=True)
        result["n_EPMR_prefix_inferred"] = sum(number(row.get("epmr")) is not None
                                               and row.get("epmr_boundary_method") == "unique_trajectory_start_prefix"
                                               for row in valid)
        if any("direct_status" in row for row in group):
            for field in ("EPMR", "n_EPMR", "n_EPMR_prefix_inferred"):
                result.pop(field, None)
            result["EPMR_direct"] = mean_metric(group, "EPMR_direct")
            result["n_EPMR_direct"] = sum(number(row.get("EPMR_direct")) is not None for row in group)
            result["EPMR_direct_method"] = "keypoint_count_intersection"
        results.append(result)
    return results


def direct_metric_group(jobs, gold_dir, prices, cache_dir, replay_root, legacy_gold_dir, fallback_dirs):
    from evaluate_direct_keypoints import evaluate_group

    rows = [cached_instance(*job, gold_dir, prices, cache_dir, replay_root, legacy_gold_dir, "direct")
            for job in jobs]
    return evaluate_group(jobs[0][1], rows, [gold_dir, *fallback_dirs], cache_dir.parent / "keypoints")


def main():
    parser = argparse.ArgumentParser(description="PR/CR, keypoint matching, and full-healing usage evaluation")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/healbench")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/healbench.jsonl")
    parser.add_argument("--gold-traces-dir", type=Path, default=ROOT / "results/gold_traces")
    parser.add_argument("--healed-traces-dir", type=Path, default=ROOT / "results/healed_traces")
    parser.add_argument("--legacy-gold-traces-dir", type=Path,
                        help="Old-runtime reference traces used to verify exact per-instance path equivalence")
    parser.add_argument("--model-prices", type=Path, default=ROOT / "scripts/model_prices.json")
    parser.add_argument("--limit", type=int, default=0, help="First N instances per source dataset and agent/model; 0 means all")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--raw-out", type=Path, default=ROOT / "results/healbench_evaluation/paper/all_raw.csv")
    parser.add_argument("--final-out", type=Path, default=ROOT / "results/healbench_evaluation/paper/all_final.csv")
    parser.add_argument("--epmr-mode", choices=("direct", "strict"), default="direct",
                        help="direct: unordered point-count intersection; strict: verified post-first-crash LCS")
    parser.add_argument("--fallback-gold-traces-dir", type=Path, action="append", default=[])
    parser.add_argument("--refresh-usage-from", type=Path,
                        help="Refresh usage from original logs while preserving non-usage metrics in this CSV")
    args = parser.parse_args()
    if args.limit < 0 or args.workers < 1:
        parser.error("limit must be nonnegative and workers must be positive")
    args.final_out.parent.mkdir(parents=True, exist_ok=True)
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    log_id = logger.add(args.final_out.parent / "evaluation.log", enqueue=True)
    try:
        prices = read_json(args.model_prices) if args.model_prices else {}
        for model, price in prices.items():
            for field in ("input_usd_per_million", "output_usd_per_million"):
                if number(price.get(field)) is None:
                    raise ValueError(f"Invalid price {model}.{field}")
        records, excluded = load_records(args.dataset, args.limit)
        if args.refresh_usage_from:
            from evaluate_usage import refresh_usage

            refresh_usage(args, records, prices)
            return
        jobs = []
        for agent in AGENTS:
            directory = args.results_root / agent
            if not directory.is_dir():
                logger.warning(f"Missing agent directory: {directory}")
                continue
            for model_dir in sorted(directory.iterdir()):
                if not model_dir.is_dir() or model_dir.name.startswith((".", "_")) or "bak" in model_dir.name:
                    continue
                for record in records:
                    jobs.append((model_dir / record["instance_id"], record, agent, model_dir.name))
        if not jobs:
            raise ValueError("No agent/model result directories found")
        models = sorted({job[3] for job in jobs})
        logger.warning(f"Models without configured USD prices: {[model for model in models if model not in prices]}")
        if not args.gold_traces_dir.is_dir():
            logger.warning(f"Patched reference trace directory missing: {args.gold_traces_dir}")
        metadata = {"arguments": {key: [str(item) for item in value] if isinstance(value, list)
                                      else str(value) if isinstance(value, Path) else value
                                      for key, value in vars(args).items()},
                    "excluded_gold_unusable": excluded, "prices": prices, "version": VERSION,
                    "definitions": {"PR": "Passed target tests or tested invocations reaching an identified test oracle / all eligible tasks. Final traceback must contain only test/oracle frames; assertion inside target code and exception-message mismatch do not count. This measures completion of the tested invocation, not execution of all later test statements. Unresolved evidence contributes no success and is reported with bounds.",
                                    "CR": "Target test returncode equals zero.",
                                    "EPMR": "LCS of ordered keypoint suffixes strictly after the first recorded crash and its patched counterpart / healed suffix event count; macro mean of verified available pairs. Boundaries are inferred only from saved crash trace, runtime temporal anchors, and a unique invocation with identical pre-invocation paths. Supported anchors: immediate first-statement crash or first for-loop unpack failure. Original runtime summaries are tried before existing verified replays: expand route/loop repetition and decisions, require full runtime sequence-slot coverage and a unique emitted ordering under independent assignment constraints; sink bookkeeping occupies sequence slots but adds no keypoints; standalone decision repetitions preserve their original decision anchor; nested-loop placeholders are not double-counted and require corresponding child summaries; ambiguous order and solver limits remain missing without data truncation. Historical summaries have no runtime hash or digest, explicitly reported as sequence_coverage_without_historical_digest rather than protocol-verified streams; they start at healing activation, including snapshot calls before heal_before. Their reference boundary requires a unique invocation and first-statement continuation, first-loop unpack evidence from saved code and locals, or a single failed unpack assignment replaced by straight-line assignments ending in a unique false guard around the same unpack target; the latter boundary follows that guard decision and retains subsequent callee events. If original decoding is unavailable, an already saved strictly verified replay may be used, with a distinct path_source and the original failure reason retained. Historical activation is associated with the first crash by crash number and file, not equality of mapped frame line or inner traceback function; the paused runtime function supplies source-boundary context. A later activation is reported as trajectory_starts_at_later_crash. If source-boundary inference fails, locate the healed trajectory's very first point ID and event kind in Golden, extending the contiguous point-ID prefix only while multiple locations remain. A single unique point occurrence suffices. Decision values do not constrain alignment but remain part of LCS event equality; retain the anchor in both suffixes, never skip healed events, and keep the entire healed denominator. Such boundaries are inferred from trajectory similarity, not proof of the original crash position, and are labeled in raw rows and counted as n_EPMR_prefix_inferred in summaries; accidental matches to a different dynamic invocation remain a limitation. Unmatched or ambiguous prefixes remain missing, never full-path fallbacks. Post-crash snapshot and repair calls remain included. One-line loop-header replacements may locate a counterpart but remain unmatched patched_only events. Historical replay retains strict action/outcome checks; PR/CR/usage remain from original runs. For full EPMR streams and verified replays, different runtimes are rejected except the pinned legacy-healed / guarded-Golden pair with an identical complete legacy patched reference. Original summaries do not emit EPMR collector logging, so they use event-semantic compatibility with the pinned Golden runtimes, explicitly without historical runtime-hash attestation, and do not require a legacy replay reference. Missing scores are never zero-filled.",
                                    "Cost_USD": "Configured per-million list-price estimate; cache discounts excluded; unknown usage or price remains missing.",
                                    "denominator": "PR/CR use all tasks after the shared blacklist, including infrastructure errors and missing results as unsuccessful; n_evaluated is diagnostic only. Evaluation errors make the run fail, not a valid final benchmark.",
                                    "means": "Tokens and cost average observations passing usage reconciliation across all task statuses, excluding missing values without zero or mean imputation; n_metric is the actual denominator. Codex uses the last monotonic cumulative usage per thread; only one-to-one logged local input-size rejections are exempt from missing-call checks. Client heal timeouts remain unknown without proven final coverage, including zero-count sessions; later snapshots alone do not establish coverage of unrecorded requests. verified_precall_rejections references the latest attempt in run.log and trace.json. OpenHands session SDK deltas are reconciled with trace response totals; disagreement excludes complete means. Backend errors and transport/execution timeouts remain incomplete. Observed_Input_Tokens, Observed_Output_Tokens and Observed_Cost_USD retain available recorded consumption across all statuses, including incomplete runs; these are not proven final usage or invoices. Missing values are not zero-filled. EPMR averages available paired paths only; unavailable reference paths are not observed zero matches."}}
        if args.epmr_mode == "direct":
            metadata["definitions"].pop("EPMR")
            metadata["definitions"]["EPMR_direct"] = (
                "Unordered count intersection: sum(min(heal_count[p], gold_count[p])) / total heal point hits; "
                "repetitions and unmatched points retained; decisions are annotations, not separate points. "
                "Summary repetition counts are accumulated without sequence expansion. Order is ignored. "
                "Full recorded scope, NOT post-first-crash LCS. Partial traces are labeled; empty paths "
                "remain unavailable. Direct summaries include all rows with actual point pairs, with coverage."
            )
        write_json(args.final_out.parent / "evaluation_metadata.json", metadata)
        rows = []
        cache_dir = args.final_out.parent / ".cache"
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            if args.epmr_mode == "direct":
                groups = defaultdict(list)
                for job in jobs:
                    groups[job[1]["instance_id"]].append(job)
                futures = {pool.submit(direct_metric_group, group, args.gold_traces_dir, prices, cache_dir,
                                       args.healed_traces_dir, args.legacy_gold_traces_dir,
                                       args.fallback_gold_traces_dir): group[0] for group in groups.values()}
            else:
                futures = {pool.submit(cached_instance, *job, args.gold_traces_dir, prices, cache_dir,
                                       args.healed_traces_dir, args.legacy_gold_traces_dir): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    logger.exception(f"Evaluation failed for {job[0]}: {error}")
                    if args.epmr_mode == "direct":
                        raise
                    row = {"agent_class": job[2], "model": job[3], "guard_enabled": 0,
                           "instance_id": job[1]["instance_id"], "source_dataset": job[1]["source_dataset"],
                           "result_dir": str(job[0]), "evaluation_status": "evaluation_error", "error": str(error)}
                rows.extend(row if isinstance(row, list) else [row])
        rows.sort(key=lambda row: (row["agent_class"], row["model"], row["source_dataset"], row["instance_id"]))
        pd.DataFrame(rows).to_csv(args.raw_out, index=False)
        summary = aggregate(rows)
        pd.DataFrame(summary).to_csv(args.final_out, index=False)
        pd.DataFrame(summary).to_csv(args.final_out.parent / "summary_results.csv", index=False)
        pd.DataFrame(aggregate(rows, by_dataset=True)).to_csv(args.final_out.parent / "by_dataset.csv", index=False)
        for row in rows[:2]:
            logger.info(f"Full raw row: {row}")
        for row in summary:
            logger.info(f"Full summary row: {row}")
        logger.info(f"Evaluation statuses: {dict(Counter(row['evaluation_status'] for row in rows))}")
        logger.info(f"Usage reasons: {dict(Counter(row.get('usage_reason') for row in rows))}")
        logger.info(f"Path reasons: {dict(Counter(row.get('direct_status', row.get('epmr_reason')) for row in rows))}")
        if any(row["evaluation_status"] == "evaluation_error" for row in rows):
            raise RuntimeError("Evaluation errors recorded; inspect raw CSV and evaluation.log")
    finally:
        logger.remove(log_id)


if __name__ == "__main__":
    main()
