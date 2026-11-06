from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run_main_single.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "healbench_smoke"
MODELS = {
    "openai/glm-5-2-260617": 49876,
    "openai/deepseek-v4-flash-ga-260731": 50876,
    "openai/gpt-5.6-terra": 51876,
}
MAX_CALL_LLM = 5


def validate_results(output_dir: Path, expected_count: int) -> dict:
    output_paths = sorted(output_dir.glob("*/output.json"))
    if len(output_paths) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} output.json files, found {len(output_paths)} in {output_dir}"
        )

    results = []
    for output_path in output_paths:
        result = json.loads(output_path.read_text(encoding="utf-8"))
        instance_dir = output_path.parent
        missing = [
            name
            for name in ("input_raw.json", "input.json", "trace.json", "run.log")
            if not (instance_dir / name).is_file()
        ]
        if missing:
            raise RuntimeError(f"Missing artifacts for {instance_dir.name}: {missing}")
        if result.get("max_call_llm") != MAX_CALL_LLM:
            raise RuntimeError(
                f"Wrong max_call_llm for {instance_dir.name}: {result.get('max_call_llm')}"
            )
        if int(result.get("llm_calls", 0)) > MAX_CALL_LLM:
            raise RuntimeError(
                f"LLM budget exceeded for {instance_dir.name}: {result.get('llm_calls')}"
            )
        results.append({
            "instance_id": result.get("instance_id"),
            "status": result.get("status"),
            "reason": result.get("reason"),
            "test_rc": result.get("test_rc"),
            "llm_calls": result.get("llm_calls"),
            "run_time_s": result.get("run_time_s"),
            "runtime_trace": result.get("runtime_trace"),
        })

    summary = {
        "model": output_dir.name,
        "max_call_llm": MAX_CALL_LLM,
        "results": results,
        "infra_errors": sum(item["status"] == "infra_error" for item in results),
    }
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run real HealBench instances with max_call_llm=5"
    )
    parser.add_argument("--model", choices=MODELS, default="openai/glm-5-2-260617")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    expected_count = len(set(args.instance_id)) if args.instance_id else args.limit
    model_name = args.model.split("/", 1)[-1]
    selection_name = f"instances-{expected_count}" if args.instance_id else f"limit-{args.limit}"
    output_dir = (args.output_root / f"{model_name}-{selection_name}").resolve()
    command = [
        sys.executable,
        str(RUN_SCRIPT),
        "--agent", "healer",
        "--provider", "openai",
        "--model", args.model,
        "--limit", "0" if args.instance_id else str(args.limit),
        "--concurrency", str(args.concurrency),
        "--max-call-llm", str(MAX_CALL_LLM),
        "--port-base", str(MODELS[args.model]),
        "--output-dir", str(output_dir),
        "--rerun",
        "--path-points",
    ]
    for instance_id in args.instance_id or []:
        command.extend(["--instance-id", instance_id])
    logger.info(
        f"Running {expected_count} real instances with max_call_llm={MAX_CALL_LLM}, "
        f"concurrency={args.concurrency}: {command}"
    )
    environment = dict(os.environ)
    environment.setdefault("AGENTICHEAL_PYTEST_TIMEOUT", "600")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment)
    if completed.returncode != 0:
        logger.error(f"HealBench smoke command failed with rc={completed.returncode}")
        return completed.returncode

    try:
        summary = validate_results(output_dir, expected_count)
    except Exception as error:
        logger.error(f"HealBench smoke validation failed: {error}")
        return 1
    logger.info(f"HealBench smoke results: {summary['results'][:2]}")
    if summary["infra_errors"]:
        logger.error(f"HealBench smoke found {summary['infra_errors']} infra_error results")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
