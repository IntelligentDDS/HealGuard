from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from loguru import logger


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def token_pair(usage, input_key="prompt_tokens", output_key="completion_tokens"):
    if not isinstance(usage, dict):
        return None
    pair = (number(usage.get(input_key)), number(usage.get(output_key)))
    return pair if all(value is not None for value in pair) else None


def current_run_log(text, instance_id):
    """Appended logs contain earlier attempts; leave the raw file unchanged."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    if not instance_id:
        return text
    starts = list(re.finditer(rf"^.*{re.escape(instance_id)}: budget=\d+, agent=\w+.*$", text, re.M))
    return text[starts[-1].start():] if starts else text


def precall_rejections(trace, log_text):
    errors = re.findall(r'Codex step error \(InvalidParamsError\): ([^\n]*)', log_text)
    events = [event for event in trace.get("events", [])
              if event.get("reason") == "backend_error:InvalidParamsError"]
    if not events or len(errors) != len(events):
        return 0
    if not all(re.fullmatch(r'JSON-RPC error -32602: Input exceeds the maximum length of \d+ characters\.',
                            error.strip()) for error in errors):
        return 0
    previous = verified = 0
    for event in trace.get("events", []):
        count = number(event.get("llm_calls"))
        if count is None or count < previous:
            return 0
        if event.get("reason") == "backend_error:InvalidParamsError":
            if count - previous != 1 or event.get("llm_calls_detail"):
                return 0
            verified += 1
        previous = count
    return verified


def collect_usage(trace, output, agent, log_text="", runtime_timeout=False):
    calls = [call for event in trace.get("events", []) for call in event.get("llm_calls_detail", [])]
    responses, threads, problems = {}, {}, []
    log_text = current_run_log(log_text, output.get("instance_id"))
    for index, call in enumerate(calls):
        raw = call.get("raw_output")
        if not isinstance(raw, dict):
            problems.append(f"call_{index}:unsupported_response")
            continue
        if agent == "CodexAdapter":
            pair = token_pair((raw.get("usage") or {}).get("total"), "inputTokens", "outputTokens")
            thread = raw.get("thread_id")
            if pair is None or not thread:
                problems.append(f"call_{index}:missing_thread_usage")
                continue
            previous = threads.get(thread)
            if previous and any(new < old for new, old in zip(pair, previous)):
                problems.append(f"call_{index}:cumulative_usage_decreased")
            threads[thread] = pair
        elif agent == "OpenHandsAdapter":
            usages = (raw.get("_llm_metrics") or {}).get("token_usages", [])
            if not usages:
                problems.append(f"call_{index}:missing_token_usages")
            for usage in usages:
                response_id = usage.get("response_id")
                pair = token_pair(usage)
                if not response_id or pair is None:
                    problems.append(f"call_{index}:unidentifiable_usage")
                    continue
                if response_id in responses and responses[response_id] != pair:
                    problems.append(f"call_{index}:conflicting_response_usage")
                responses[response_id] = pair
        else:
            pair = token_pair(raw.get("usage"))
            response_id = raw.get("id")
            if pair is None or not response_id:
                problems.append(f"call_{index}:missing_response_usage")
                continue
            if response_id in responses and responses[response_id] != pair:
                problems.append(f"call_{index}:conflicting_response_usage")
            responses[response_id] = pair
    values = list(threads.values() if agent == "CodexAdapter" else responses.values())
    recorded_calls = len(calls)
    expected_calls = number(output.get("llm_calls"))
    rejected_calls = precall_rejections(trace, log_text) if agent == "CodexAdapter" else 0
    if expected_calls is None or recorded_calls + rejected_calls != expected_calls:
        problems.append("call_coverage_mismatch")
    if not values and expected_calls != rejected_calls:
        problems.append("no_usage")
    if runtime_timeout or any(re.search(r'timed out|timeout', str(record.get("heal_error", "")), re.I)
                              for record in output.get("repair_records", [])):
        problems.append("unreconciled_heal_timeout")
    if output.get("test_rc") == -1 or output.get("error") == "pytest_timeout":
        problems.append("unreconciled_execution_timeout")
    totals = tuple(sum(pair[i] for pair in values) for i in (0, 1))
    observed = totals if values else None
    source = "trace.json:events.llm_calls_detail.raw_output"
    output_pair = token_pair(output, "input_tokens", "output_tokens")
    # OpenHands stores deltas of cumulative SDK metrics; Codex's historical output
    # sums cumulative snapshots instead and must NOT be used as a token total.
    if agent == "OpenHandsAdapter" and output_pair is not None:
        if output_pair != totals:
            problems.append("session_trace_usage_mismatch")
        if output_pair != (0, 0) and all(a >= b for a, b in zip(output_pair, totals)):
            observed = output_pair
            source = "output.json:session_sdk_token_deltas"
    backend_errors = [event.get("reason", "") for event in trace.get("events", [])
                      if str(event.get("reason", "")).startswith("backend_error:")]
    unverified_errors = [reason for reason in backend_errors
                         if not (reason == "backend_error:InvalidParamsError" and rejected_calls)]
    if unverified_errors or re.search(r"step error \((?!InvalidParamsError\))[^\n]+", log_text):
        problems.append("unreconciled_backend_error")
    complete = not problems
    if complete and observed is None:
        observed = totals
    return {
        "input_tokens": totals[0] if complete else None,
        "output_tokens": totals[1] if complete else None,
        "observed_input_tokens": observed[0] if observed is not None else None,
        "observed_output_tokens": observed[1] if observed is not None else None,
        "usage_complete": complete,
        "usage_reason": ";".join(dict.fromkeys(problems)) or "complete",
        "usage_source": source,
        "usage_records": recorded_calls,
        "verified_precall_rejections": rejected_calls,
        "llm_calls": expected_calls,
    }


def task_usage(task_dir, output, agent, model, prices, read_json):
    trace_file = task_dir / "trace.json"
    trace = read_json(trace_file) if trace_file.is_file() else {"events": []}
    log_file = task_dir / "run.log"
    log_text = log_file.read_text(encoding="utf-8") if log_file.is_file() else ""
    runtime_timeout = False
    docker_log = task_dir / "run_docker.log"
    # Scan without loading multi-GB keypoint streams; only transport failure
    # signatures count, not arbitrary appearances of 'timeout' in tested code.
    if docker_log.is_file():
        with docker_log.open(encoding="utf-8") as stream:
            for line in stream:
                if re.search(r"(?:socket\.timeout|TimeoutError): timed out|ReadTimeoutError:|HTTPConnectionPool[^\n]*Read timed out", line):
                    runtime_timeout = True
    row = collect_usage(trace, output, agent, log_text, runtime_timeout)
    if not trace_file.is_file():
        row.update(input_tokens=None, output_tokens=None, usage_complete=False,
                   usage_reason=row["usage_reason"] + ";missing_agent_trace")
    price = prices.get(model)
    for prefix in ("", "observed_"):
        pair = token_pair(row, prefix + "input_tokens", prefix + "output_tokens")
        row[prefix + "cost_usd"] = ((pair[0] * price["input_usd_per_million"]
                                      + pair[1] * price["output_usd_per_million"]) / 1_000_000
                                     if pair is not None and price is not None else None)
    row["cost_reason"] = ("missing_model_price" if price is None else
                          "configured_list_price_estimate_no_cache_discount" if row["usage_complete"] else
                          "incomplete_usage_observed_cost_reported_separately")
    return row


def refresh_usage_row(row, prices, cache_dir):
    from evaluate_paper_metrics import read_json, write_json

    task_dir = Path(row["result_dir"])
    paths = [task_dir / name for name in ("output.json", "trace.json", "run.log", "run_docker.log")]
    paths.extend((Path(__file__), Path(__file__).with_name("evaluate_paper_metrics.py")))
    signature = [row, prices]
    for path in paths:
        signature.append((str(path), path.stat().st_size, path.stat().st_mtime_ns) if path.is_file()
                         else (str(path), None))
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = cache_dir / f"{key}.json"
    if cache.is_file():
        return read_json(cache)
    output = read_json(task_dir / "output.json")
    if output.get("instance_id", task_dir.name) != row["instance_id"]:
        raise ValueError("output_instance_id_mismatch")
    updated = {**row, **task_usage(task_dir, output, row["agent_class"], row["model"], prices, read_json)}
    write_json(cache, updated)
    return updated


def refresh_usage(args, records, prices):
    """Reuse saved non-usage metrics, not reconstructed Golden trajectories."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import pandas as pd
    from evaluate_paper_metrics import VERSION, aggregate, read_json, write_json

    frame = pd.read_csv(args.refresh_usage_from)
    rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
    selected = {record["instance_id"] for record in records}
    rows = [row for row in rows if row["instance_id"] in selected]
    keys = [(row["agent_class"], row["model"], row["instance_id"]) for row in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError("empty_or_duplicate_refresh_rows")
    if {row["instance_id"] for row in rows} != selected:
        raise ValueError("refresh_source_missing_selected_instances")
    metadata_file = args.refresh_usage_from.parent / "evaluation_metadata.json"
    metadata = read_json(metadata_file)
    metadata.update(version=VERSION, prices=prices,
                    usage_refresh_source=str(args.refresh_usage_from.resolve()))
    metadata["definitions"]["means"] = (
        "Input_Tokens/Output_Tokens/Cost_USD average complete reconciled usage across all statuses. "
        "Observed_Input_Tokens/Observed_Output_Tokens/Observed_Cost_USD average available recorded "
        "usage across all statuses, including incomplete and failed runs; these are not final billed totals. "
        "Missing values remain missing; n_metric gives each denominator. OpenHands SDK session deltas "
        "are retained when they dominate trace totals, never added to them; conflicting totals are incomplete. "
        "Codex uses per-thread final cumulative snapshots, never sums cumulative output counters. "
        "Transport/execution timeouts and unverified backend errors exclude complete means. "
        "PR, CR and EPMR are reused without changes from the source CSV."
    )
    metadata["usage_refresh_arguments"] = {key: str(value) if isinstance(value, Path) else value
                                            for key, value in vars(args).items()
                                            if not isinstance(value, list)}
    write_json(args.final_out.parent / "evaluation_metadata.json", metadata)
    updated = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(refresh_usage_row, row, prices, args.final_out.parent / ".usage_cache")
                   for row in rows]
        for future in as_completed(futures):
            updated.append(future.result())
    updated.sort(key=lambda row: (row["agent_class"], row["model"], row["source_dataset"], row["instance_id"]))
    pd.DataFrame(updated).to_csv(args.raw_out, index=False)
    summaries = aggregate(updated)
    pd.DataFrame(summaries).to_csv(args.final_out, index=False)
    pd.DataFrame(summaries).to_csv(args.final_out.parent / "summary_results.csv", index=False)
    pd.DataFrame(aggregate(updated, by_dataset=True)).to_csv(args.final_out.parent / "by_dataset.csv", index=False)
    for row in updated[:2]:
        logger.info(f"Full refreshed usage row: {row}")
    for row in summaries:
        logger.info(f"Full refreshed summary row: {row}")
    logger.info(f"Refreshed usage for {len(updated)} records; non-usage metrics preserved")
