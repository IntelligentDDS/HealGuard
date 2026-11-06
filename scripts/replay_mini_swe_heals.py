from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess

import yaml
import sys
import threading
import shutil
from collections import Counter
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_middleware import Agent, HealSession
import agenticheal
from agenticheal import Config, HEALBENCH_RESULTS, run_healbench
from util import HEALBENCH_INSTANCE_IDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_RESULTS = HEALBENCH_RESULTS / "MiniSweAgentAdapter"
DEFAULT_RESULT_PREFIX = "healbench_sink"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "datasets" / "raw_repo_source_code"
DEFAULT_KEY_POINT_ROOT = PROJECT_ROOT / "results" / "find_points"


class ReplayAgent(Agent):
    def __init__(self, decisions: dict[str, list[dict]], strict: bool = False):
        self.decisions = decisions
        self.strict = strict
        self.failures: dict[str, list[str]] = {}
        self._indexes: dict[str, int] = {}
        self._lock = threading.Lock()

    def step(self, session: HealSession, repo_dir: str = "") -> dict:
        crash = session.active_crash
        with self._lock:
            if self.strict and self.failures.get(session.instance_id):
                logger.warning(f"[{session.instance_id}] Strict replay already diverged; refusing further actions")
                return {"action": "reraise", "reason": "epmr_replay_already_diverged"}
            index = self._indexes.get(session.instance_id, 0)
            self._indexes[session.instance_id] = index + 1
        expected = self.decisions.get(session.instance_id, [])
        if index >= len(expected):
            with self._lock:
                self.failures.setdefault(session.instance_id, []).append("original_decision_missing")
            logger.warning(f"[{session.instance_id}] No MiniSweAgent decision at index {index}; re-raising")
            return {"action": "reraise", "reason": "original_decision_missing"}
        item = expected[index]
        actual_key = (crash.file_path, crash.line, crash.block_index) if crash else ("", -1, -1)
        if self.strict and (actual_key != item["key"] or any(
            getattr(crash, field, None) != value for field, value in item.get("crash_identity", {}).items()
        )):
            with self._lock:
                self.failures.setdefault(session.instance_id, []).append(f"crash_identity_mismatch:{index}")
            logger.error(f"[{session.instance_id}] Strict replay refused action {index}: expected={item['key']}, actual={actual_key}")
            return {"action": "reraise", "reason": "epmr_replay_crash_identity_mismatch"}
        if actual_key != item["key"]:
            logger.warning(
                f"[{session.instance_id}] Replay crash index {index} moved: "
                f"expected={item['key']}, actual={actual_key}; replaying indexed action"
            )
        return dict(item["decision"])


def discover_traces(source_results: Path, model: str | None) -> list[tuple[str, str, Path]]:
    model_dirs = [source_results / model] if model else sorted(path for path in source_results.iterdir() if path.is_dir())
    traces = []
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            logger.warning(f"MiniSweAgent model result directory is missing: {model_dir}")
            continue
        for trace_path in sorted(model_dir.glob("*/trace.json")):
            traces.append((model_dir.name, trace_path.parent.name, trace_path))
    return traces


def select_traces(
    source_results: Path,
    model: str | None,
    instance_file: Path | None,
    limit: int,
    preferred_paths: dict[str, Path] | None = None,
) -> list[tuple[str, str, Path]]:
    requested_ids = None
    if instance_file:
        requested_ids = [
            line.strip()
            for line in instance_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    first_trace_by_instance = {}
    empty_trace_instances = set()
    for model_name, instance_id, trace_path in discover_traces(source_results, model):
        if instance_id in first_trace_by_instance:
            continue
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if not any(
            event.get("action") in {"execute", "reraise"}
            for event in trace.get("events", [])
        ):
            empty_trace_instances.add(instance_id)
            continue
        first_trace_by_instance[instance_id] = (model_name, instance_id, trace_path)
    if empty_trace_instances:
        logger.warning(
            f"Ignored empty model trajectories for {len(empty_trace_instances)} instances and "
            f"selected another model where available; first 2: {sorted(empty_trace_instances)[:2]}"
        )
    preferred_paths = preferred_paths or {}
    for instance_id, trace_path in preferred_paths.items():
        if not trace_path.is_file():
            raise RuntimeError(f"Preferred MiniSweAgent trajectory is missing: {trace_path}")
        model_name = trace_path.parent.parent.name
        first_trace_by_instance[instance_id] = (model_name, instance_id, trace_path)
    if requested_ids is None:
        selected = list(first_trace_by_instance.values())
    else:
        missing = [instance_id for instance_id in requested_ids if instance_id not in first_trace_by_instance]
        if missing:
            raise RuntimeError(
                f"{len(missing)} requested MiniSweAgent trajectories are missing; first 2: {missing[:2]}"
            )
        selected = [first_trace_by_instance[instance_id] for instance_id in requested_ids]
    skipped = [item for item in selected if item[1] not in HEALBENCH_INSTANCE_IDS]
    selected = [item for item in selected if item[1] in HEALBENCH_INSTANCE_IDS]
    if skipped:
        logger.warning(
            f"Skipped {len(skipped)} trajectories outside the benchmark manifest; "
            f"first 2: {[instance_id for _, instance_id, _ in skipped[:2]]}"
        )
    if limit:
        selected = selected[:limit]
    logger.info(f"Selected {len(selected)} MiniSweAgent trajectories; first 2: {[(m, i) for m, i, _ in selected[:2]]}")
    return selected


def load_heal_injections(path: Path | None) -> dict[tuple[str, int], dict]:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    injections = {
        (str(item["instance_id"]), int(item["heal_index"])): item
        for item in payload.get("injections", [])
    }
    logger.info(f"Loaded {len(injections)} Heal-time Sink injections; first 2: {list(injections.items())[:2]}")
    return injections


def injected_sink_code(item: dict) -> str:
    variable = str(item["healed_variable"])
    callee = str(item["callee"])
    callables = {
        "builtins.print": "print",
        "logging.error": "__import__('logging').error",
        "sys.stdout.write": "__import__('sys').stdout.write",
        "warnings.simplefilter": "__import__('warnings').simplefilter",
        "time.sleep": "__import__('time').sleep",
    }
    function = callables.get(callee)
    if function is None:
        raise ValueError(f"Unsupported Heal-time injected Sink callee: {callee}")
    if callee == "logging.error":
        arguments = f"'%r', {variable}"
    elif callee == "warnings.simplefilter":
        arguments = "'default'"
    elif callee == "time.sleep":
        arguments = "0"
    elif callee == "sys.stdout.write":
        arguments = f"str({variable})"
    else:
        arguments = variable
    metadata = {
        "sink_id": f"injected-{item['rule_id']}-{item['instance_id']}",
        "sink_kind": item["sink_kind"],
        "runtime_rule_id": item["rule_id"],
        "actual_call_target": callee,
        "static_influence_names": [variable],
        "file": item["file"],
        "line": int(item["line"]),
        "injected_ground_truth": True,
    }
    assignment = ""
    if variable == "agenticheal_sink_payload":
        assignment = f"{variable} = {int(item['injection_value'])}\n"
    return (
        assignment
        + f"__import__('builtins')._eh_guard_sink_call("
        + f"{metadata!r}, {function}, locals(), {arguments})"
    )


def load_replay_decisions(
    traces: list[tuple[str, str, Path]],
    injections: dict[tuple[str, int], dict] | None = None,
) -> tuple[dict[str, list[dict]], dict[str, str]]:
    decisions = {}
    models = {}
    injections = injections or {}
    for model_name, instance_id, trace_path in traces:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        instance_decisions = []
        submitted_heal_index = 0
        for event in trace.get("events", []):
            action = event.get("action")
            if action == "execute_result" or action not in {"execute", "reraise"}:
                continue
            key = (
                str(event.get("file_path", "")),
                int(event.get("line", -1)),
                int(event.get("block_index", -1)),
            )
            if action == "execute":
                submit = bool(event.get("submit", False))
                code = str(event.get("code", ""))
                if submit:
                    submitted_heal_index += 1
                    injection = injections.get((instance_id, submitted_heal_index))
                    if injection:
                        code = f"{code}\n{injected_sink_code(injection)}"
                decision = {
                    "action": "execute",
                    "code": code,
                    "submit": submit,
                    "reason": "mini_swe_trajectory_replay",
                }
            else:
                decision = {"action": "reraise", "reason": "mini_swe_trajectory_reraise"}
            instance_decisions.append({"key": key, "decision": decision})
        if not instance_decisions:
            logger.warning(
                f"[{instance_id}] MiniSweAgent trajectory has no execute/reraise decisions; "
                f"skipping empty trajectory: {trace_path}"
            )
            continue
        decisions[instance_id] = instance_decisions
        models[instance_id] = model_name
        logger.info(f"[{instance_id}] Loaded {len(instance_decisions)} actions; first 2: {instance_decisions[:2]}")
    return decisions, models


def merged_sink_yaml(paths: list[Path], output_dir: Path) -> Path:
    payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    rule_ids = set()
    targets = set()
    rules = []
    sink_kinds = set()
    for path, payload in zip(paths, payloads):
        for rule in payload.get("rules", []):
            rule_id = str(rule.get("rule_id", ""))
            target = str(rule.get("fully_qualified_callee_or_event", ""))
            if not rule_id or not target:
                raise ValueError(f"Sink rule in {path} is missing rule_id or target")
            if rule_id in rule_ids or target in targets:
                raise ValueError(f"Duplicate Sink rule while merging {path}: {rule_id} / {target}")
            rule_ids.add(rule_id)
            targets.add(target)
            rules.append(rule)
            sink_kinds.add(str(rule["sink_kind"]))
    merged = {
        **payloads[0],
        "source": "replay_merged_sink_rules",
        "reviewed_from": [str(path) for path in paths],
        "policy": "Ephemeral merged input for MiniSweAgent replay; each source rule remains instance-scoped where declared.",
        "rules": rules,
    }
    declared_kinds = set(merged.get("sink_kinds", []))
    unsupported_kinds = sink_kinds - declared_kinds
    if unsupported_kinds:
        raise ValueError(f"Merged Sink rules use unsupported taxonomy kinds: {sorted(unsupported_kinds)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "merged_sink_rules.yaml"
    output.write_text(yaml.safe_dump(merged, allow_unicode=True, sort_keys=False), encoding="utf-8")
    logger.info(f"Merged {len(rules)} Sink rules from {len(paths)} inputs into {output}; first 2: {rules[:2]}")
    return output


def sink_name(path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_.-")
    if not name:
        raise ValueError(f"Cannot derive a result name from Sink YAML: {path}")
    return name


def default_roots(path: Path) -> tuple[Path, Path, Path]:
    name = sink_name(path)
    result_root = PROJECT_ROOT / "results" / f"{DEFAULT_RESULT_PREFIX}{name}"
    return (
        result_root,
        PROJECT_ROOT / "results" / f"find_points_sink{name}",
        PROJECT_ROOT / "results" / f"repo_instrumented_sink{name}",
    )


def run_command(command: list[str], env: dict[str, str] | None = None) -> None:
    logger.info(f"Running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with rc={completed.returncode}: {' '.join(command)}")


def prepare_instrumentation(
    sink_yaml: Path,
    instance_ids: list[str],
    scan_root: Path,
    instrument_root: Path,
    source_root: Path,
    key_point_root: Path,
    scan_parallel: int,
    tool_workers: int,
    timeout: int,
    force: bool,
    source_overrides: Path | None,
) -> None:
    generated_root = scan_root / "_generated_sink_rules"
    instance_file = scan_root / "replay_instances.txt"
    scan_root.mkdir(parents=True, exist_ok=True)
    instance_file.write_text("\n".join(instance_ids) + "\n", encoding="utf-8")
    run_command([
        sys.executable,
        "scripts/run_build_sink_rules.py",
        "--rules",
        str(sink_yaml),
        "--output",
        str(generated_root),
    ])
    env = os.environ.copy()
    env["AGENTICHEAL_SINK_RULES"] = str(sink_yaml)
    env["AGENTICHEAL_GENERATED_SINK_RULES"] = str(generated_root)
    scan_command = [
        sys.executable,
        "scripts/run_find_points_sinks.py",
        "--cache-dir",
        str(source_root),
        "--instance-file",
        str(instance_file),
        "--output-dir",
        str(scan_root),
        "--parallel",
        str(scan_parallel),
        "--tool-workers",
        str(tool_workers),
        "--timeout",
        str(timeout),
    ]
    if source_overrides:
        scan_command.extend(["--repository-overrides", str(source_overrides)])
    if force:
        scan_command.append("--force")
    run_command(scan_command, env=env)

    import instrument_guard
    import run_instrument_all

    original_guard_config = dict(instrument_guard.CONFIG)
    original_all_config = dict(run_instrument_all.CONFIG)
    replay_config = {
        "source_root": source_root,
        "sink_root": scan_root,
        "key_point_root": key_point_root,
        "output_root": instrument_root,
    }
    instrument_guard.CONFIG.update(replay_config)
    run_instrument_all.CONFIG.update(replay_config)
    try:
        results = [run_instrument_all.instrument_all(instance_id, force) for instance_id in instance_ids]
    finally:
        instrument_guard.CONFIG.clear()
        instrument_guard.CONFIG.update(original_guard_config)
        run_instrument_all.CONFIG.clear()
        run_instrument_all.CONFIG.update(original_all_config)
    failures = [result for result in results if result.get("status") != "completed"]
    logger.info(f"Replay instrumentation completed: total={len(results)}, failures={len(failures)}; first 2: {results[:2]}")
    if failures:
        raise RuntimeError(f"Replay instrumentation failed: {failures[:2]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild Sink instrumentation and replay MiniSweAgent Heal trajectories")
    parser.add_argument(
        "--sink-yaml",
        required=True,
        help="One Sink YAML path, or comma-separated paths merged automatically for this replay.",
    )
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--model")
    parser.add_argument("--instance-file", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scan-root", type=Path)
    parser.add_argument("--instrument-root", type=Path)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--heal-injections",
        type=Path,
        help="Manifest whose protected operation is appended to the matching replayed Heal code.",
    )
    parser.add_argument("--source-overrides", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reuse-instrumentation", action="store_true")
    parser.add_argument("--key-point-root", type=Path, default=DEFAULT_KEY_POINT_ROOT)
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Container replay concurrency.",
    )
    parser.add_argument(
        "--scan-parallel",
        type=int,
        help="Repository Sink scan concurrency; defaults to --parallel.",
    )
    parser.add_argument("--tool-workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--replay-timeout",
        type=int,
        default=600,
        help="Maximum seconds per replayed pytest instance; default 600 instead of the normal 1500.",
    )
    parser.add_argument(
        "--full-path-trace",
        action="store_true",
        help="Record all path points; slower and unnecessary for executed-Sink exact-name evaluation.",
    )
    parser.add_argument("--port-base", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if (
        args.limit < 0
        or args.parallel < 1
        or args.tool_workers < 1
        or args.replay_timeout < 1
        or (args.scan_parallel is not None and args.scan_parallel < 1)
    ):
        parser.error("--limit must be non-negative and concurrency/timeouts must be positive")
    scan_parallel = args.scan_parallel or args.parallel
    sink_paths = [Path(value.strip()).resolve() for value in args.sink_yaml.split(",") if value.strip()]
    if not sink_paths:
        parser.error("--sink-yaml must include at least one path")
    missing_sink_paths = [path for path in sink_paths if not path.is_file()]
    if missing_sink_paths:
        parser.error(f"Sink YAML does not exist: {missing_sink_paths}")
    sink_yaml = sink_paths[0]
    default_output, default_scan, default_instrument = default_roots(sink_yaml)
    output_dir = (args.output_dir or default_output).resolve()
    scan_root = (args.scan_root or default_scan).resolve()
    instrument_root = (args.instrument_root or default_instrument).resolve()
    if len(sink_paths) > 1:
        sink_yaml = merged_sink_yaml(sink_paths, scan_root)
    injections = load_heal_injections(args.heal_injections.resolve() if args.heal_injections else None)
    preferred_paths = {
        instance_id: PROJECT_ROOT / str(item["trace_path"])
        for (instance_id, _), item in injections.items()
    }
    traces = select_traces(
        args.source_results.resolve(), args.model, args.instance_file, args.limit, preferred_paths
    )
    if not traces:
        parser.error("No matching MiniSweAgent trajectories found")
    decisions, models = load_replay_decisions(traces, injections)
    instance_ids = list(decisions)
    if args.reuse_instrumentation:
        final_rules_path = (PROJECT_ROOT / "scripts" / "sink_rules" / "final_rules.yaml").resolve()
        if final_rules_path not in sink_paths:
            parser.error(
                "--reuse-instrumentation requires scripts/sink_rules/final_rules.yaml because "
                "the original prebuilt repositories contain those rules"
            )
        overlay_rules = []
        for path in sink_paths:
            if path == final_rules_path:
                continue
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            for rule in payload.get("rules", []):
                template = rule.get("template")
                if template not in {"exact_assignment", "exact_call"} or not rule.get("guard", {}).get("instance_id"):
                    parser.error(
                        f"Reuse overlay rule {rule.get('rule_id')} in {path} must be instance-scoped "
                        "and use exact_assignment or exact_call"
                    )
                overlay_rules.append(rule)
        runtime_overlay_path = scan_root / "runtime_sink_overlay.json"
        runtime_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_overlay_path.write_text(
            json.dumps({"rules": overlay_rules}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.environ["AGENTICHEAL_RUNTIME_SINK_OVERLAY"] = str(runtime_overlay_path)
        instrument_root = (PROJECT_ROOT / "results" / "repo_instrumented").resolve()
        logger.info(
            f"Reusing original prebuilt instrumentation for {len(instance_ids)} instances and "
            f"applying {len(overlay_rules)} rules only in ephemeral run copies: {instrument_root}; "
            f"first 2 overlay rules: {[rule['rule_id'] for rule in overlay_rules[:2]]}"
        )
    else:
        prepare_instrumentation(
            sink_yaml,
            instance_ids,
            scan_root,
            instrument_root,
            args.source_root.resolve(),
            args.key_point_root.resolve(),
            scan_parallel,
            args.tool_workers,
            args.timeout,
            args.force,
            args.source_overrides.resolve() if args.source_overrides else None,
        )
    os.environ["AGENTICHEAL_INSTRUMENTED_ROOT"] = str(instrument_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sink_yaml": str(sink_yaml),
        "source_results": str(args.source_results.resolve()),
        "output_dir": str(output_dir),
        "instance_models": models,
        "decision_counts": {instance_id: len(items) for instance_id, items in decisions.items()},
        "scan_root": str(scan_root),
        "instrument_root": str(instrument_root),
    }
    (output_dir / "replay_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config = Config(
        provider="openai",
        openai_model="mini-swe-trajectory-replay",
        openai_base_url="",
        openai_api_key="",
        llm_max_concurrency=args.parallel,
        output_dir=str(output_dir),
    )
    agenticheal.PYTEST_MAIN_TIMEOUT = args.replay_timeout
    logger.info(
        f"Replay execution settings: timeout={args.replay_timeout}s, "
        f"full_path_trace={args.full_path_trace}, concurrency={args.parallel}"
    )
    asyncio.run(run_healbench(
        ReplayAgent(decisions),
        config=config,
        instance_ids=instance_ids,
        max_call_llm=0,
        concurrency=args.parallel,
        output_dir=output_dir,
        record_trace=True,
        port_base=args.port_base,
        instrumentation_enabled=True,
        path_points_enabled=args.full_path_trace,
        rerun=args.rerun,
    ))
    statuses = Counter()
    for instance_id in instance_ids:
        output_path = output_dir / instance_id / "output.json"
        if output_path.is_file():
            statuses[json.loads(output_path.read_text(encoding="utf-8")).get("status", "unknown")] += 1
    logger.info(f"MiniSweAgent replay finished: {dict(statuses)}; first 2 outputs: {[str(output_dir / item / 'output.json') for item in instance_ids[:2]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
