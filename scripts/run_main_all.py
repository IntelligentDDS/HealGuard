from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SINGLE_SCRIPT = PROJECT_ROOT / "scripts" / "run_main_single.py"

MODELS = [
    {
        "model": "openai/glm-5-2-260617",
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "max_workers": 4,
        "port_base": 19876,
    },
    {
        "model": "openai/deepseek-v4-flash-ga-260731",
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "max_workers": 12,
        "port_base": 29876,
    },
    {
        "model": "openai/gpt-5.6-terra",
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "max_workers": 4,
        "port_base": 39876,
    },
]


def run(model: dict, args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        str(SINGLE_SCRIPT),
        "--agent",
        args.agent,
        "--provider",
        "openai",
        "--model",
        model["model"],
        "--base-url",
        model["base_url"],
        "--api-key",
        model["api_key"],
        "--limit",
        str(args.limit),
        "--concurrency",
        str(model["max_workers"]),
        "--max-call-llm",
        str(args.max_call_llm),
        "--port-base",
        str(model["port_base"]),
    ]
    for instance_id in args.instance_id or []:
        command.extend(["--instance-id", instance_id])
    if args.instance_file:
        command.extend(["--instance-file", str(args.instance_file)])
    if args.cleanup_images:
        command.append("--cleanup-images")
    if args.rerun:
        command.append("--rerun")
    command.append("--path-points" if args.path_points else "--no-path-points")

    model_name = model["model"].split("/", 1)[-1]
    print(
        f"RUN [{model_name}]: instances={'all' if args.limit == 0 else args.limit}, "
        f"model_workers={model['max_workers']}, port_base={model['port_base']}"
    )
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    if process.stdout is None:
        process.kill()
        raise RuntimeError(f"Failed to capture output for {model_name}")
    for line in process.stdout:
        print(f"[{model_name}] {line}", end="")
    return process.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the three supported HealBench models in parallel"
    )
    parser.add_argument("--agent", default="healer")
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--instance-file", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Instances per model (0=all)")
    parser.add_argument("--max-call-llm", type=int, default=100)
    parser.add_argument("--cleanup-images", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Rerun selected instances even when results exist")
    parser.add_argument(
        "--path-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record repository path-point IDs after the first Heal begins",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.max_call_llm < 0:
        parser.error("--max-call-llm must be non-negative")

    failed = False
    with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
        futures = {
            executor.submit(run, model, args): model
            for model in MODELS
        }
        for future in as_completed(futures):
            model = futures[future]
            model_name = model["model"].split("/", 1)[-1]
            try:
                return_code = future.result()
            except Exception as error:
                failed = True
                print(f"Task exception [{model_name}]: {error}")
                continue
            if return_code != 0:
                failed = True
                print(f"Task failed [{model_name}], exit code: {return_code}")
            else:
                print(f"Mission successful [{model_name}]")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
