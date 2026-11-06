"""Replay saved healing actions without model calls and retain independent EPMR evidence."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import re
from contextlib import ExitStack
from pathlib import Path

from loguru import logger

import agenticheal
from agenticheal import Config, load_dataset_records, run_healbench
from util import HEALBENCH_INSTANCE_IDS
from evaluate_paper_metrics import read_path
from evaluate_results import _read_docker_output
from replay_mini_swe_heals import ReplayAgent, load_replay_decisions
from run_record_gold_trace import _write_json

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ("Healer", "MiniSweAgentAdapter", "OpenHandsAdapter", "CodexAdapter")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_evidence(trace):
    fields = ("action", "file_path", "line", "block_index", "error_type", "error_message", "code", "submit",
              "execute_code", "execute_stdout", "execute_value", "execute_error")
    records = []
    for event in trace.get("events", []):
        if event.get("action") not in ("execute", "reraise", "execute_result"):
            continue
        item = {key: event.get(key) for key in fields}
        if isinstance(item.get("execute_error"), str):
            # Only transport values generated for each container differ; retain the full traceback otherwise.
            item["execute_error"] = re.sub(
                r'"port": "[0-9]+", "token": "[a-f0-9]+", "record_path": "/tmp/_repair_records_[a-f0-9]+\.json"',
                '"port": "<run>", "token": "<run>", "record_path": "<run>"', item["execute_error"])
            # Added tracing shifts injected runtime lines, not repository or submitted Heal-code locations.
            item["execute_error"] = re.sub(
                r'(?m)^(  File "<string>", line )\d+(, in (?:_guard_sink_call|_guard_local_changes|_guard_changed_names|_heal_success_record|_heal_failure_record))$',
                r'\1<runtime>\2', item["execute_error"])
        records.append(item)
    return records


def run_model(records, source, destination, model, parallel, timeout, port_base):
    destination.mkdir(parents=True, exist_ok=True)
    lock_root = destination / ".locks"
    lock_root.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for record in sorted(records, key=lambda item: item["instance_id"]):
            lock = stack.enter_context((lock_root / f"{record['instance_id']}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX)
        pending = []
        signatures = {}
        dependencies = [Path(__file__), Path(agenticheal.__file__)]
        dependencies += [Path(__file__).with_name(name) for name in
                         ("heal_builtins.py", "evaluate_paper_metrics.py", "replay_mini_swe_heals.py", "agent_middleware.py")]
        for record in records:
            iid = record["instance_id"]
            original = source / iid
            if not all((original / name).is_file() for name in ("trace.json", "input.json", "output.json")):
                logger.warning(f"{original}: missing original replay evidence")
                continue
            output = json.loads((original / "output.json").read_text())
            saved_input = json.loads((original / "input.json").read_text())
            if output.get("test_rc") not in (0, 1) or output.get("status") == "infra_error":
                logger.warning(f"{original}: original test did not complete, replay excluded")
                continue
            identity_fields = ("instance_id", "repo", "base_commit", "patch", "test_patch", "test_name", "test_file",
                               "before_repo_set_cmd", "dockerhub_tag")
            if any(saved_input.get(key) != record.get(key) for key in identity_fields):
                raise ValueError(f"{original}: original test/repository input mismatch")
            prebuilt = ROOT / "results/repo_instrumented" / iid / "repo"
            paths = [original / name for name in ("trace.json", "input.json", "output.json")]
            paths += [prebuilt / "_instrumentation.json", *dependencies]
            signature = {str(path): file_hash(path) for path in paths}
            signature["timeout"] = timeout
            signatures[iid] = signature
            proof_path = destination / iid / "epmr_replay.json"
            if proof_path.is_file():
                proof = json.loads(proof_path.read_text())
                if proof.get("complete") and proof.get("signature") == signature:
                    try:
                        read_path(_read_docker_output(proof_path.parent / "run_docker.log"))
                        if any(proof.get("replay_files", {}).get(name) != file_hash(proof_path.parent / name)
                               for name in ("run_docker.log", "trace.json", "output.json")):
                            raise ValueError("replay_evidence_hash_mismatch")
                        logger.info(f"{iid}: complete EPMR replay already exists")
                        continue
                    except (OSError, ValueError) as error:
                        logger.warning(f"{iid}: replay resume rejected: {error}")
            pending.append(record)
        if not pending:
            return
        traces = [(model, record["instance_id"], source / record["instance_id"] / "trace.json") for record in pending]
        decisions, _ = load_replay_decisions(traces)
        for _, iid, path in traces:
            trace = json.loads(path.read_text())
            events = [event for event in trace.get("events", []) if event.get("action") in ("execute", "reraise")]
            decisions.setdefault(iid, [])
            for item, event in zip(decisions[iid], events):
                item["crash_identity"] = {field: event[field] for field in ("error_type", "error_message")
                                          if event.get("file_path") and field in event}
        agent = ReplayAgent(decisions, strict=True)
        config = Config(provider="openai", openai_model=model, openai_base_url="", openai_api_key="",
                        llm_max_concurrency=parallel, output_dir=str(destination))
        agenticheal.PYTEST_MAIN_TIMEOUT = timeout
        asyncio.run(run_healbench(agent, config=config, instance_ids=[r["instance_id"] for r in pending],
                                  max_call_llm=0, concurrency=parallel, output_dir=destination,
                                  record_trace=True, port_base=port_base, instrumentation_enabled=True,
                                  path_points_enabled=True, rerun=True))
        for _, iid, original_trace_path in traces:
            task = destination / iid
            errors = list(agent.failures.get(iid, []))
            original_trace = json.loads(original_trace_path.read_text())
            original_output = json.loads((source / iid / "output.json").read_text())
            replay_output = json.loads((task / "output.json").read_text())
            replay_trace = json.loads((task / "trace.json").read_text())
            if agent._indexes.get(iid, 0) != len(decisions[iid]):
                errors.append("replay_actions_not_fully_consumed")
            if action_evidence(original_trace) != action_evidence(replay_trace):
                errors.append("replay_action_or_execution_result_mismatch")
            if any(original_output.get(key) != replay_output.get(key) for key in ("test_rc", "status")):
                errors.append("replay_test_outcome_mismatch")
            if replay_output.get("llm_calls", 0) != 0 or any(event.get("llm_calls", 0) for event in replay_trace.get("events", [])):
                errors.append("unexpected_replay_model_call")
            identity = {}
            try:
                path = read_path(_read_docker_output(task / "run_docker.log"), identity_out=identity)
                logger.info(f"{iid}: replay path={len(path)} events; first 2: {path[:2]}")
            except (OSError, ValueError) as error:
                logger.warning(f"{iid}: replay path invalid: {error}")
                errors.append(str(error))
            proof = {"complete": not errors, "errors": errors, "signature": signatures[iid],
                     "source_dir": str((source / iid).resolve()), "path_identity": identity,
                     "actions_expected": len(decisions[iid]), "actions_consumed": agent._indexes.get(iid, 0),
                     "model_calls": 0,
                     "replay_files": {name: file_hash(task / name) for name in ("run_docker.log", "trace.json", "output.json")}}
            _write_json(task / "epmr_replay.json", proof)
            logger.info(f"{iid}: strict EPMR replay complete={proof['complete']}, errors={errors}")


def main():
    parser = argparse.ArgumentParser(description="严格重放原修复动作并采集 EPMR，不调用模型、不修改原结果")
    parser.add_argument("--source-results", type=Path, default=ROOT / "results/healbench")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/healed_traces")
    parser.add_argument("--agent", choices=AGENTS)
    parser.add_argument("--model")
    parser.add_argument("--source")
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--limit", type=int, default=0, help="每个来源取前 N 个实例，0 表示全部")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--port-base", type=int)
    args = parser.parse_args()
    if args.limit < 0 or args.parallel < 1 or args.timeout < 1:
        parser.error("limit must be nonnegative; parallel and timeout must be positive")
    source, destination = args.source_results.resolve(), args.output_dir.resolve()
    protected = [source, ROOT / "results/repo_instrumented", ROOT / "results/find_points", ROOT / "scripts"]
    if any(destination == path or destination in path.parents or path in destination.parents for path in protected):
        parser.error("output directory must be separate from original results, points, instrumentation and scripts")
    if os.environ.get("AGENTICHEAL_RUNTIME_SINK_OVERLAY") or os.environ.get("AGENTICHEAL_INSTRUMENTED_ROOT"):
        parser.error("EPMR replay requires unmodified original prebuilt instrumentation, without Sink overlays")
    _, records = load_dataset_records(args.source)
    excluded = sum(r["instance_id"] not in HEALBENCH_INSTANCE_IDS for r in records)
    logger.info(f"Excluded {excluded} records not present in the benchmark manifest")
    selected, counts = [], {}
    for record in records:
        if record["instance_id"] not in HEALBENCH_INSTANCE_IDS:
            continue
        if args.instance_id and record["instance_id"] not in args.instance_id:
            continue
        dataset = record["source_dataset"]
        if args.limit and counts.get(dataset, 0) >= args.limit:
            continue
        counts[dataset] = counts.get(dataset, 0) + 1
        selected.append(record)
    logger.info(f"Selected datasets={counts}; first 2 instance IDs: {[r['instance_id'] for r in selected[:2]]}")
    for agent in (args.agent,) if args.agent else AGENTS:
        directory = source / agent
        if not directory.is_dir():
            logger.warning(f"Missing agent directory: {directory}")
            continue
        models = [directory / args.model] if args.model else sorted(p for p in directory.iterdir()
                     if p.is_dir() and not p.name.startswith((".", "_")) and "bak" not in p.name)
        for model_dir in models:
            # Finalize each concurrent batch before starting another so interrupted runs retain verified cases.
            for start in range(0, len(selected), args.parallel):
                batch = [selected[index] for index in range(start, min(start + args.parallel, len(selected)))]
                run_model(batch, model_dir, destination / agent / model_dir.name, model_dir.name,
                          args.parallel, args.timeout, args.port_base)


if __name__ == "__main__":
    main()
