from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agenticheal import _resolve_docker_image
from util import HEALBENCH_INSTANCE_IDS
from util import init_logger, load_instance_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = PROJECT_ROOT / "datasets" / "raw_repo_source_code"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "repo_source_download"
DATASETS = {"SWE-Bench-Pro", "SWE-Bench-Verified", "SWE-Bench-Full-Test", "R2E-Gym-Lite"}


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_id(image: str, workdir: str) -> str:
    return hashlib.sha256(f"{image}\0{workdir}".encode()).hexdigest()[:24]


def source_directory_name(record: dict, image: str, workdir: str) -> str:
    repository = re.sub(r"[^A-Za-z0-9_.-]+", "_", record["repo"].replace("/", "__"))
    return f"{repository}___base-{record['base_commit'][:12]}___src-{source_id(image, workdir)[:8]}"


def cache_complete(cache_dir: Path, record: dict, image: str, workdir: str) -> bool:
    metadata_path = cache_dir / "metadata.json"
    repo = cache_dir / "repo"
    if not metadata_path.is_file() or not repo.is_dir():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning(f"Invalid raw repository cache metadata at {metadata_path}: {error}")
        return False
    return (
        metadata.get("repository") == record["repo"]
        and metadata.get("expected_commit") == record["base_commit"]
        and metadata.get("source_id") == source_id(image, workdir)
        and metadata.get("source_path") == "repo"
        and metadata.get("docker_image") == image
        and metadata.get("docker_workdir") == workdir
    )


def run_command(command: list[str], timeout: int) -> dict:
    started = time.time()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_s": round(time.time() - started, 3),
        }
    except subprocess.TimeoutExpired as error:
        logger.error(f"Command timed out: {command}: {error}")
        return {
            "command": command,
            "returncode": None,
            "stdout": error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or "",
            "stderr": error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or "",
            "duration_s": round(time.time() - started, 3),
            "error": f"TimeoutExpired: {error}",
        }
    except Exception as error:
        logger.error(f"Command failed: {command}: {type(error).__name__}: {error}")
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "duration_s": round(time.time() - started, 3),
            "error": f"{type(error).__name__}: {error}",
        }


def extract_repository(record: dict, cache_root: Path, results_root: Path, timeout: int, force: bool) -> dict:
    image, workdir = _resolve_docker_image(record)
    identity = source_id(image, workdir)
    directory_name = source_directory_name(record, image, workdir)
    cache_dir = cache_root / directory_name
    metadata_path = cache_dir / "metadata.json"
    if not force and cache_complete(cache_dir, record, image, workdir):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {**metadata, "status": "cached", "directory_name": directory_name}

    started = time.time()
    container = f"raw_repo_source_code_{uuid.uuid4().hex[:12]}"
    commands = []
    container_removed = False
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{directory_name}.", dir=cache_root))
    temporary_repo = temporary_root / "repo"
    try:
        create = run_command(["docker", "create", "--name", container, image, "true"], timeout)
        commands.append(create)
        if create["returncode"] != 0:
            raise RuntimeError(f"docker create failed: {create['stderr']}")
        temporary_repo.mkdir()
        copy = run_command(["docker", "cp", f"{container}:{workdir}/.", str(temporary_repo)], timeout)
        commands.append(copy)
        if copy["returncode"] != 0:
            raise RuntimeError(f"docker cp failed: {copy['stderr']}")
        remove = run_command(["docker", "rm", "-f", container], timeout)
        commands.append(remove)
        container_removed = True
        if remove["returncode"] != 0:
            raise RuntimeError(f"docker rm failed: {remove['stderr']}")
        head = run_command(["git", "-C", str(temporary_repo), "rev-parse", "HEAD"], timeout)
        status = run_command(["git", "-C", str(temporary_repo), "status", "--porcelain"], timeout)
        commands.extend([head, status])
        verified_commit = head["stdout"].strip() if head["returncode"] == 0 else None
        modified = bool(status["stdout"].strip()) if status["returncode"] == 0 else None
        if verified_commit == record["base_commit"] and modified is False:
            source_status = "verified_clean"
        elif verified_commit == record["base_commit"]:
            source_status = "verified_modified"
        elif verified_commit:
            source_status = "commit_mismatch_modified" if modified else "commit_mismatch_clean"
        else:
            source_status = "commit_unverified"
        metadata = {
            "repository": record["repo"],
            "expected_commit": record["base_commit"],
            "verified_commit": verified_commit,
            "source_status": source_status,
            "source_id": identity,
            "source_path": "repo",
            "docker_image": image,
            "docker_workdir": workdir,
        }
        json_dump(temporary_root / "metadata.json", metadata)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        os.replace(temporary_root, cache_dir)
        return {**metadata, "status": "completed", "directory_name": directory_name, "duration_s": round(time.time() - started, 3), "commands": commands}
    except Exception as error:
        if not container_removed:
            remove = run_command(["docker", "rm", "-f", container], timeout)
            commands.append(remove)
            container_removed = True
        logger.error(f"Raw repository extraction failed for {image}: {type(error).__name__}: {error}")
        failure = {
            "status": "error",
            "directory_name": directory_name,
            "source_id": identity,
            "image": image,
            "workdir": workdir,
            "instance_ids": record.get("cache_instance_ids", [record["instance_id"]]),
            "source_datasets": sorted(set(record.get("cache_source_datasets", [record.get("source_dataset", "unknown")]))),
            "duration_s": round(time.time() - started, 3),
            "commands": commands,
            "error": f"{type(error).__name__}: {error}",
        }
        json_dump(results_root / "failures" / f"{directory_name}.json", failure)
        return failure
    finally:
        if not container_removed:
            remove = run_command(["docker", "rm", "-f", container], timeout)
            commands.append(remove)
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def select_records(source: str | None, limit: int, per_dataset_limit: int, instance_file: str | None, dataset: Path = PROJECT_ROOT / "datasets/healbench.jsonl") -> list[dict]:
    records = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info(f"Loaded {len(records)} complete records from {dataset}; first 2: {[record['instance_id'] for record in records[:2]]}")
    if source:
        records = [record for record in records if record.get("source_dataset") == source]
    before = len(records)
    records = [
        record for record in records
        if record.get("instance_id", "") in HEALBENCH_INSTANCE_IDS
    ]
    logger.info(f"Excluded {before - len(records)} records not present in the benchmark manifest")
    if instance_file:
        instance_ids = set(load_instance_file(instance_file))
        records = [record for record in records if record["instance_id"] in instance_ids]
    if per_dataset_limit:
        counts: dict[str, int] = {}
        selected = []
        for record in records:
            dataset = record.get("source_dataset", "unknown")
            if counts.get(dataset, 0) >= per_dataset_limit:
                continue
            counts[dataset] = counts.get(dataset, 0) + 1
            selected.append(record)
        records = selected
    return records[:limit] if limit else records


def group_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    for record in records:
        image, workdir = _resolve_docker_image(record)
        group = grouped.setdefault((image, workdir), dict(record))
        group.setdefault("cache_instance_ids", []).append(record["instance_id"])
        group.setdefault("cache_source_datasets", []).append(record.get("source_dataset", "unknown"))
    return list(grouped.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache original repositories from dataset Docker images")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets/healbench.jsonl")
    parser.add_argument("--source", choices=sorted(DATASETS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-dataset-limit", type=int, default=0)
    parser.add_argument("--instance-file")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache_root = args.cache_dir.resolve()
    results_root = args.results_dir.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    init_logger(str(results_root / "run.log"))
    if args.limit < 0 or args.per_dataset_limit < 0:
        parser.error("Instance limits must be non-negative")
    if args.parallel < 1 or args.timeout < 1:
        parser.error("--parallel and --timeout must be positive")
    records = select_records(args.source, args.limit, args.per_dataset_limit, args.instance_file, args.dataset.resolve())
    if not records:
        parser.error("No runnable records were selected from the dataset")
    groups = group_records(records)
    for record in groups:
        image, workdir = _resolve_docker_image(record)
        directory = source_directory_name(record, image, workdir)
        json_dump(results_root / "repositories" / directory / "input_raw.json", record)
    logger.info(f"Selected {len(records)} instances across {len(groups)} unique repositories; first 2: {[record['instance_id'] for record in records[:2]]}")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        future_to_record = {
            pool.submit(extract_repository, record, cache_root, results_root, args.timeout, args.force): record
            for record in groups
        }
        for future in concurrent.futures.as_completed(future_to_record):
            record = future_to_record[future]
            try:
                result = future.result()
            except Exception as error:
                logger.error(f"{record['instance_id']}: {type(error).__name__}: {error}")
                result = {
                    "status": "error",
                    "instance_ids": record.get("cache_instance_ids", [record["instance_id"]]),
                    "error": f"{type(error).__name__}: {error}",
                }
            image, workdir = _resolve_docker_image(record)
            directory = source_directory_name(record, image, workdir)
            json_dump(results_root / "repositories" / directory / "output.json", result)
            results.append(result)
            logger.info(f"{record['instance_id']}: {result['status']} directory={result.get('directory_name', '')}")

    summary = {
        "status": "completed" if all(result["status"] in {"completed", "cached"} for result in results) else "completed_with_errors",
        "cache_root": str(cache_root),
        "selected_instance_count": len(records),
        "unique_repository_count": len(groups),
        "completed_count": sum(result["status"] == "completed" for result in results),
        "cached_count": sum(result["status"] == "cached" for result in results),
        "error_count": sum(result["status"] == "error" for result in results),
        "results": sorted(results, key=lambda result: result.get("directory_name", "")),
    }
    json_dump(results_root / "summary.json", summary)
    logger.info(f"Summary: completed={summary['completed_count']}, cached={summary['cached_count']}, errors={summary['error_count']}")
    return 0 if summary["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
