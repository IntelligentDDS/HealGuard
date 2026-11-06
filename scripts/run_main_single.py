from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import (
    ClaudeCodeAdapter,
    CodexAdapter,
    Healer,
    MiniSweAgentAdapter,
    OpenHandsAdapter,
    ReAct,
    SWEAgentAdapter,
)
from agenticheal import (
    Config,
    HEALBENCH_RESULTS,
    load_dataset_records,
    run_healbench,
)
from util import HEALBENCH_INSTANCE_IDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTRUMENTED_ROOT = PROJECT_ROOT / "results" / "repo_instrumented"
SUPPORTED_MODELS = (
    "openai/glm-5-2-260617",
    "openai/deepseek-v4-flash-ga-260731",
    "openai/gpt-5.6-terra",
)


def _build_model_args(config: Config) -> tuple[str, dict]:
    name = (
        f"openai/{config.openai_model}"
        if "/" not in config.openai_model
        else config.openai_model
    )
    kwargs = {}
    if config.openai_base_url:
        kwargs["api_base"] = config.openai_base_url
    if config.openai_api_key:
        kwargs["api_key"] = config.openai_api_key
    if config.provider == "seed":
        kwargs["extra_body"] = {"reasoning_effort": "minimal"}
    elif config.provider == "gpt":
        kwargs["reasoning_effort"] = "none"
    elif config.provider == "deepseek":
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return name, kwargs


def _load_requested_instances(
    instance_ids: list[str] | None,
    instance_file: Path | None,
) -> set[str]:
    requested = set(instance_ids or [])
    if instance_file:
        requested.update(
            line.strip()
            for line in instance_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return requested


def select_instances(
    instance_ids: list[str] | None,
    instance_file: Path | None,
    limit: int,
) -> list[str]:
    _, records = load_dataset_records()
    requested = _load_requested_instances(instance_ids, instance_file)
    excluded_count = sum(
        str(record["instance_id"]) not in HEALBENCH_INSTANCE_IDS
        for record in records
    )
    logger.info(
        f"Excluded {excluded_count} records not present in the benchmark manifest"
    )

    selected = []
    for record in records:
        instance_id = str(record["instance_id"])
        if instance_id not in HEALBENCH_INSTANCE_IDS:
            continue
        if requested and instance_id not in requested:
            continue
        selected.append(instance_id)
        if limit and len(selected) >= limit:
            break

    if requested:
        dataset_instance_ids = {
            str(record["instance_id"])
            for record in records
        }
        missing = sorted(requested - dataset_instance_ids)
        if missing:
            raise RuntimeError(f"Requested instances were not found: {missing}")
        excluded_requested = sorted(requested - HEALBENCH_INSTANCE_IDS)
        if excluded_requested:
            logger.info(
                f"Excluded {len(excluded_requested)} requested outside-manifest instances; "
                f"first 2: {excluded_requested[:2]}"
            )
    return selected


def validate_instrumented_repositories(instance_ids: list[str]) -> None:
    missing = []
    incomplete = []
    for instance_id in instance_ids:
        repository = INSTRUMENTED_ROOT / instance_id / "repo"
        instrumentation = repository / "_instrumentation.json"
        if not repository.is_dir() or not instrumentation.is_file():
            missing.append(instance_id)
            continue
        try:
            payload = json.loads(instrumentation.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.error(f"Failed to read instrumentation for {instance_id}: {error}")
            incomplete.append({"instance_id": instance_id, "error": str(error)})
            continue
        status = payload.get("build", {}).get("status")
        if status != "completed":
            incomplete.append({"instance_id": instance_id, "status": status})

    if missing:
        raise RuntimeError(
            "Prebuilt instrumented repositories are missing: "
            f"{missing}; run uv run python scripts/run_instrument_all.py first"
        )
    if incomplete:
        raise RuntimeError(f"Prebuilt instrumentation is incomplete: {incomplete}")


def create_agent(args: argparse.Namespace, config: Config):
    if args.agent == "react":
        return ReAct(config=config)
    if args.agent == "mini-swe-agent":
        model_name, model_kwargs = _build_model_args(config)
        return MiniSweAgentAdapter(
            model_name,
            model_kwargs,
            cost_tracking=args.mini_cost_tracking,
        )
    if args.agent == "SWE-agent":
        model_name, model_kwargs = _build_model_args(config)
        return SWEAgentAdapter(model_name, model_kwargs)
    if args.agent == "healer":
        return Healer(config=config)
    if args.agent == "openhands":
        model_name, model_kwargs = _build_model_args(config)
        return OpenHandsAdapter(model_name, model_kwargs)
    if args.agent == "claude-code":
        model_name, model_kwargs = _build_model_args(config)
        if args.llm_base_url:
            model_kwargs["api_base"] = args.llm_base_url
        if args.llm_api_key:
            model_kwargs["api_key"] = args.llm_api_key
        logger.info(
            f"claude-code gateway: base_url={model_kwargs.get('api_base')} model={model_name}"
        )
        return ClaudeCodeAdapter(model_name, model_kwargs)
    if args.agent == "codex":
        model_name, model_kwargs = _build_model_args(config)
        if args.llm_base_url:
            model_kwargs["api_base"] = args.llm_base_url
        if args.llm_api_key:
            model_kwargs["api_key"] = args.llm_api_key
        logger.info(
            f"codex gateway: base_url={model_kwargs.get('api_base')} model={model_name}"
        )
        return CodexAdapter(model_name, model_kwargs)
    raise ValueError(f"Unknown agent: {args.agent}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HealBench agents against repositories prebuilt by run_instrument_all.py"
    )
    parser.add_argument(
        "--agent",
        default="healer",
        choices=[
            "react",
            "mini-swe-agent",
            "SWE-agent",
            "swe-agent",
            "healer",
            "openhands",
            "claude-code",
            "codex",
        ],
    )
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--instance-file", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Total max instances (0=all)")
    parser.add_argument("--concurrency", "--parallel", type=int, default=2)
    parser.add_argument("--max-call-llm", "--max_call_llm", type=int, default=100)
    parser.add_argument("--port-base", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cleanup-images", "--cleanup_images", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Rerun selected instances even when results exist")
    parser.add_argument(
        "--path-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record repository path-point IDs after the first Heal begins",
    )
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument(
        "--mini-cost-tracking",
        default="ignore_errors",
        choices=["default", "ignore_errors"],
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.agent == "swe-agent":
        args.agent = "SWE-agent"
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.max_call_llm < 0:
        raise ValueError("--max-call-llm must be non-negative")

    instance_ids = select_instances(args.instance_id, args.instance_file, args.limit)
    if not instance_ids:
        raise RuntimeError("No runnable instances were selected")
    validate_instrumented_repositories(instance_ids)
    logger.info(
        f"Running {len(instance_ids)} prebuilt instrumented instances; first 2: {instance_ids[:2]}"
    )

    model_output_name = args.model.split("/", 1)[-1]
    config = Config(
        provider=args.provider,
        openai_model=args.model,
        openai_base_url=args.base_url,
        openai_api_key=args.api_key,
        llm_max_concurrency=args.concurrency,
        output_dir=str(HEALBENCH_RESULTS / args.agent / model_output_name),
    )
    agent = create_agent(args, config)
    output_dir = args.output_dir or HEALBENCH_RESULTS / type(agent).__name__ / model_output_name
    output_dir = output_dir.resolve()
    config.output_dir = str(output_dir)
    stats = asyncio.run(
        run_healbench(
            agent,
            config=config,
            limit=0,
            instance_ids=instance_ids,
            max_call_llm=args.max_call_llm,
            concurrency=args.concurrency,
            cleanup_images=args.cleanup_images,
            output_dir=output_dir,
            record_trace=True,
            port_base=args.port_base,
            instrumentation_enabled=True,
            path_points_enabled=args.path_points,
            rerun=args.rerun,
        )
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
