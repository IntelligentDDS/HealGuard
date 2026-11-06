from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from loguru import logger

from util import HEALBENCH_INSTANCE_IDS
from evaluate_paper_metrics import write_json
from evaluate_results import parse_unified_diff

VERSION = "keypoint-count-intersection-v3-per-process-integrity"


def is_loop(point):
    return point.endswith((":for_statement", ":while_statement"))


def map_point(point, mappings):
    file = point.split("::", 1)[0]

    def translate(match):
        line = int(match.group(2))
        for mapping in mappings:
            if line is not None and file in mapping:
                line = mapping[file].get(line)
        value = str(line) if line is not None else f"patched_only_{match.group(2)}"
        return match.group(1) + value + match.group(3)

    return sys.intern(re.sub(r"(::|@)(\d+)(:)", translate, point))


def stream_points(path, mappings):
    points, starts, ends, emitted = Counter(), [], [], 0
    processes = {}
    mapped, parse_errors, integrity_errors = {}, [], []
    identity = None
    with path.open(encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.startswith("__EH_RUNTIME_TRACE_") and "FAIL__" in line:
                integrity_errors.append({"line": line_number, "error": "runtime_collector_failure", "raw_line": line})
            if not line.startswith("__EPMR"):
                continue
            if "\t" not in line:
                integrity_errors.append({"line": line_number, "error": "unsupported_epmr_protocol", "raw_line": line})
                continue
            marker, raw = line.rstrip("\n").split("\t", 1)
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as error:
                logger.warning(f"Incomplete EPMR record at {path}:{line_number}: {error}; raw_line={line!r}")
                parse_errors.append({"line": line_number, "error": str(error), "raw_line": line})
                continue
            if not isinstance(event, dict) or event.get("schema") != "epmr-keypoints-v1" or type(event.get("pid")) is not int:
                integrity_errors.append({"line": line_number, "error": "invalid_epmr_schema_or_pid", "raw_line": line})
                continue
            pid = event["pid"]
            state = processes.get(pid)
            if marker == "__EPMR_START__":
                starts.append(event)
                keys = ("instance_id", "runtime_fingerprint", "builder_fingerprint", "source_fingerprint", "query_fingerprint")
                current_identity = tuple(event.get(key) for key in keys)
                if not all(current_identity) or (identity is not None and identity != current_identity):
                    integrity_errors.append({"line": line_number, "error": "instrumentation_identity_mismatch"})
                if identity is None:
                    identity = current_identity
                if state is not None:
                    integrity_errors.append({"line": line_number, "pid": pid, "error": "duplicate_process_start"})
                else:
                    processes[pid] = {"count": 0, "digest": hashlib.sha256(), "finished": False}
            elif marker == "__EPMR_END__":
                ends.append(event)
                if state is None or state["finished"]:
                    integrity_errors.append({"line": line_number, "pid": pid, "error": "unexpected_process_end"})
                else:
                    if (type(event.get("count")) is not int or event["count"] != state["count"]
                            or event.get("sha256") != state["digest"].hexdigest() or event.get("failures") != []):
                        integrity_errors.append({"line": line_number, "pid": pid, "error": "epmr_integrity_mismatch"})
                    state["finished"] = True
            elif marker == "__EPMR_BATCH__":
                import base64
                import zlib
                try:
                    if event.get("encoding") != "zlib-json-tuples-v1":
                        raise ValueError("unsupported_batch_encoding")
                    batch = json.loads(zlib.decompress(base64.b64decode(event["data"], validate=True)))
                    if not isinstance(batch, list) or type(event.get("count")) is not int or len(batch) != event["count"]:
                        raise ValueError("batch_count_mismatch")
                    if state is None or state["finished"] or type(event.get("seq")) is not int or event["seq"] != state["count"]:
                        raise ValueError("batch_outside_stream_or_sequence_mismatch")
                    for item in batch:
                        if (not isinstance(item, list) or len(item) != 3 or type(item[0]) is not int
                                or not isinstance(item[1], str) or (item[2] is not None and not isinstance(item[2], dict))):
                            raise ValueError("invalid_batch_event")
                    state["digest"].update((raw + "\n").encode())
                    state["count"] += len(batch)
                    emitted += len(batch)
                    for thread, point, decision in batch:
                        if decision is None or is_loop(point):
                            if point not in mapped:
                                mapped[point] = map_point(point, mappings)
                            points[mapped[point]] += 1
                except (ValueError, TypeError, KeyError, zlib.error) as error:
                    logger.warning(f"Invalid EPMR batch at {path}:{line_number}: {error}")
                    integrity_errors.append({"line": line_number, "error": str(error), "raw_line": line})
            elif marker == "__EPMR__":
                emitted += 1
                if state is None or state["finished"]:
                    integrity_errors.append({"line": line_number, "pid": pid, "error": "epmr_event_outside_stream"})
                else:
                    if type(event.get("seq")) is not int or event["seq"] != state["count"]:
                        integrity_errors.append({"line": line_number, "pid": pid, "error": "invalid_epmr_event_sequence"})
                    state["count"] += 1
                    state["digest"].update((raw + "\n").encode())
                point = event.get("point_id")
                if not isinstance(point, str) or type(event.get("thread")) is not int:
                    integrity_errors.append({"line": line_number, "pid": pid, "error": "invalid_epmr_event"})
                    continue
                # Count intersection needs no invented ordering across processes or threads.
                if event.get("decision") is None or is_loop(point):
                    if point not in mapped:
                        mapped[point] = map_point(point, mappings)
                    points[mapped[point]] += 1
            else:
                integrity_errors.append({"line": line_number, "error": "unsupported_epmr_marker", "raw_line": line})
    for pid, state in processes.items():
        if not state["finished"]:
            integrity_errors.append({"pid": pid, "error": "missing_process_end"})
    complete = bool(processes) and emitted > 0 and not parse_errors and not integrity_errors
    if not complete:
        logger.warning(f"Incomplete EPMR stream {path}: {len(parse_errors)} parse errors; integrity errors={integrity_errors}")
    return points, {"format": "ordered_epmr_stream", "complete": complete,
                    "headers": starts, "trailers": ends, "raw_events": emitted,
                    "parse_errors": parse_errors, "integrity_errors": integrity_errors}


def summary_points(path):
    points, actions = Counter(), Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            action = event.get("action")
            actions[action] += 1
            count = event.get("repeat_count", 1)
            if type(count) is not int or count < 1:
                raise ValueError("invalid_repeat_count")
            if action in ("route_summary", "loop_summary"):
                decisions = {item["point_id"] for item in event.get("first_decisions", [])}
                # Nested-loop placeholders are counted by their own loop summaries.
                for point in event["path"]:
                    if not is_loop(point) or point in decisions:
                        points[sys.intern(point)] += count
            elif action == "path_point":
                points[sys.intern(event["point_id"])] += count
            elif action in ("path_decision", "path_decision_repeat_summary") and is_loop(event["point_id"]):
                points[sys.intern(event["point_id"])] += count
    return points, {"format": "summary_point_counts", "complete": None,
                    "actions": dict(actions), "raw_events": sum(actions.values()),
                    "scope": "all_available_recorded_points_not_verified_post_first_crash"}


def save_points(path, points):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for point, count in points.items():
            stream.write(json.dumps({"point_id": point, "count": count}, ensure_ascii=False) + "\n")


def evaluate_group(record, rows, gold_roots, output):
    iid = record["instance_id"]
    excluded = iid not in HEALBENCH_INSTANCE_IDS
    logger.info(f"Excluded {int(excluded)} outside-manifest instances at direct matcher entry")
    if excluded:
        return []
    folder = output / "instances" / iid
    folder.mkdir(parents=True, exist_ok=True)
    reference = next((root / iid for root in gold_roots if (root / iid / "gold.log").is_file()), None)
    dependencies = [Path(__file__), Path(__file__).with_name("evaluate_paper_metrics.py")]
    for row in rows:
        task = Path(row["result_dir"])
        dependencies.extend(task / name for name in ("run_docker.log", "run_docker_events.jsonl"))
    if reference:
        dependencies.extend(reference / name for name in ("gold.log", "input_raw.json", "output.json"))
    signature = hashlib.sha256(json.dumps([VERSION, record, rows, str(reference),
        [(str(p), p.stat().st_size, p.stat().st_mtime_ns) if p.is_file() else (str(p), None)
         for p in dependencies]], sort_keys=True).encode()).hexdigest()
    cache = folder / "output.json"
    if cache.is_file():
        saved = json.loads(cache.read_text())
        if saved.get("signature") == signature:
            return saved["rows"]
    right, reference_info = Counter(), {"error": "missing_reference"}
    if reference:
        saved_input = json.loads((reference / "input_raw.json").read_text())
        if any(saved_input.get(key) != record.get(key) for key in ("instance_id", "patch", "test_patch")):
            raise ValueError(f"Reference input mismatch: {reference}")
        mappings = [parse_unified_diff(record.get(key, ""), align_replacements=False)
                    for key in ("test_patch", "patch")]
        right, reference_info = stream_points(reference / "gold.log", mappings)
        save_points(folder / "gold_point_counts.jsonl.gz", right)
    output_rows = []
    for index, row in enumerate(rows):
        task = Path(row["result_dir"])
        evidence = folder / row["agent_class"] / row["model"]
        metadata_path = evidence / "metadata.json"
        if metadata_path.is_file():
            saved = json.loads(metadata_path.read_text())
            if saved.get("signature") == signature and (evidence / "heal_point_counts.jsonl.gz").is_file():
                output_rows.append(saved["result"])
                continue
        left, info = Counter(), {"format": "missing"}
        runtime, log = task / "run_docker_events.jsonl", task / "run_docker.log"
        try:
            if log.is_file():
                left, info = stream_points(log, [])
            if not left and runtime.is_file():
                left, info = summary_points(runtime)
            evidence = folder / row["agent_class"] / row["model"]
            evidence.mkdir(parents=True, exist_ok=True)
            save_points(evidence / "heal_point_counts.jsonl.gz", left)
            matched = None
            if left and right:
                matched = (left & right).total()
            result = dict(row)
            result.update(EPMR_direct=matched / left.total() if matched is not None else None,
                          direct_metric="keypoint_count_intersection", direct_matched_points=matched, direct_heal_points=left.total(),
                          direct_gold_points=right.total(), direct_heal_format=info["format"],
                          direct_scope="all_available_recorded_points_not_verified_post_first_crash",
                          direct_status="scored" if matched is not None else "no_heal_points" if not left else "no_gold_points",
                          direct_gold_complete=reference_info.get("complete"),
                          direct_gold_parse_errors=len(reference_info.get("parse_errors", [])),
                          direct_heal_complete=info.get("complete"),
                          direct_heal_parse_errors=len(info.get("parse_errors", [])))
            write_json(evidence / "metadata.json", {"signature": signature, "heal": info, "gold": reference_info,
                       "reference_path": str(reference), "result": result})
            output_rows.append(result)
            if index < 2:
                logger.info(f"Full direct result: {result}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.error(f"Direct keypoint evaluation failed for {task}: {error}")
            output_rows.append({**row, "EPMR_direct": None, "direct_status": "read_error", "direct_error": str(error)})
    write_json(cache, {"signature": signature, "rows": output_rows})
    return output_rows


if __name__ == "__main__":
    from evaluate_paper_metrics import main

    main()
