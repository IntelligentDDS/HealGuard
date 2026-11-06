import ast
import asyncio
import base64
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from util import HEALBENCH_INSTANCE_IDS
from heal_repository_scope import REPOSITORY_EXCLUDED_DIRECTORIES, REPOSITORY_EXCLUDED_PATTERNS
from util import Config, build_instr_prep_cmd, init_logger

load_dotenv(Path(__file__).resolve().parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
HEALBENCH_JSONL = DATASETS_DIR / "healbench.jsonl"
HEALBENCH_RESULTS = PROJECT_ROOT / "results" / "healbench"

DOCKER_TIMEOUT = 600
PYTEST_MAIN_TIMEOUT = int(os.environ.get("AGENTICHEAL_PYTEST_TIMEOUT", "1500"))
HOST_PORT_BASE = 19876
EXCLUDED_PORTS = {5001, 8322}
REPAIR_RECORD_PATH_MARKER = "__EH_RECORD_PATH__"
REPAIR_PORT_MARKER = "__EH_PORT__"
HEAL_TRACE_PREFIX = "[heal_trace]"
RESULT_FILES = {
    "input_raw.json",
    "input.json",
    "output.json",
    "trace.json",
    "run.log",
    "run_docker.log",
}
TEE_LOCAL = threading.local()
TEE_INSTALLED = False
DOCKER_ACCOUNTS_FILE = SCRIPTS_DIR / "docker_account.md"
IMAGE_EXISTS_CACHE_FILE = DATASETS_DIR / "_r2egym_image_exists_cache.json"
docker_accounts: list[tuple[str, str]] = []
account_index = 0
account_lock = threading.Lock()
pull_semaphore = threading.Semaphore(5)
image_exists_cache: dict[str, bool] = {}
image_cache_lock = threading.Lock()

from agent_middleware import (
    Agent,
    AgentMiddleware,
    CrashInfo,
    ExecuteOutcome,
    HealSession,
)


def _write_heal_trace(log_file, event_type: str, data) -> None:
    record = {"type": event_type, "data": data}
    log_file.write(
        HEAL_TRACE_PREFIX
        + json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":"))
        + "\n"
    )
    log_file.flush()


def _write_jsonl_as_trace(log_file, path: Path, event_type: str) -> None:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                data = json.loads(line)
            except Exception as error:
                logger.error(
                    f"Failed to merge {path.name}:{line_number} into run_docker.log: {error}"
                )
                raise
            _write_heal_trace(log_file, event_type, data)


def _merge_result_artifacts(
    task_dir: Path,
    trace_events: list[dict],
    docker_result: dict,
) -> None:
    log_path = task_dir / "run_docker.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        for event in trace_events:
            _write_heal_trace(log_file, "agent_event", event)
        _write_heal_trace(log_file, "docker_result", {
            "returncode": docker_result.get("returncode"),
            "stdout": docker_result.get("stdout", ""),
            "stderr": docker_result.get("stderr", ""),
        })
        runtime_artifacts = [
            path.name
            for path in sorted(task_dir.glob("run_docker_*.jsonl*"))
            if path.is_file()
        ]
        runtime_artifacts.extend(
            path.name
            for path in sorted(task_dir.glob("run_docker_events.jsonl.*.json"))
            if path.is_file()
        )
        _write_heal_trace(
            log_file,
            "runtime_artifacts",
            {"files": list(dict.fromkeys(runtime_artifacts))},
        )
        healed_trace_path = task_dir / "healed_trace.json"
        if healed_trace_path.is_file():
            _write_heal_trace(
                log_file,
                "healed_trace",
                json.loads(healed_trace_path.read_text(encoding="utf-8")),
            )


def _remove_result_artifacts(task_dir: Path) -> None:
    for path in task_dir.iterdir():
        if path.name in RESULT_FILES or path.name.startswith("run_docker_") or path.name == "healed_trace.json":
            continue
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except Exception as error:
            logger.warning(f"Failed to remove result artifact {path}: {error}")


def _ensure_result_files(task_dir: Path, input_data: dict, result: dict) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    input_path = task_dir / "input.json"
    if not input_path.exists():
        input_path.write_text(
            json.dumps(input_data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    (task_dir / "run.log").touch(exist_ok=True)
    docker_log_path = task_dir / "run_docker.log"
    with docker_log_path.open("a", encoding="utf-8") as log_file:
        _write_heal_trace(log_file, "run_result", result)
    (task_dir / "output.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    _remove_result_artifacts(task_dir)


class _ThreadAwareTee:
    def __init__(self, real):
        self._real = real

    def write(self, data):
        self._real.write(data)
        fh = getattr(TEE_LOCAL, "run_log_fh", None)
        if fh is not None:
            try:
                fh.write(data)
            except Exception as error:
                TEE_LOCAL.run_log_fh = None
                logger.warning(f"run log write failed; detached thread log: {error}")
        return len(data) if data else 0

    def flush(self):
        self._real.flush()
        fh = getattr(TEE_LOCAL, "run_log_fh", None)
        if fh is not None:
            try:
                fh.flush()
            except Exception as error:
                TEE_LOCAL.run_log_fh = None
                logger.warning(f"run log flush failed; detached thread log: {error}")

    def __getattr__(self, name):
        return getattr(self._real, name)


def _install_stdout_tee():
    global TEE_INSTALLED
    if TEE_INSTALLED:
        return
    sys.stdout = _ThreadAwareTee(sys.stdout)
    sys.stderr = _ThreadAwareTee(sys.stderr)
    TEE_INSTALLED = True


def _bind_thread_run_log(path: "Path | str"):
    _install_stdout_tee()
    fh = open(path, "a", encoding="utf-8", buffering=1)
    TEE_LOCAL.run_log_fh = fh
    return fh


def _unbind_thread_run_log(fh):
    TEE_LOCAL.run_log_fh = None
    try:
        if fh is not None:
            fh.close()
    except Exception as error:
        logger.warning(f"run log close failed: {error}")


def parse_test_list(val) -> list:
    if not isinstance(val, str):
        return list(val) if val else []
    if not val.strip() or val.strip() == "[]":
        return []
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return ast.literal_eval(val)




def _normalize_container_path(fpath: str, workdir: str = "/app") -> str:
    for prefix in (workdir.rstrip("/") + "/", "/testbed/"):
        if fpath.startswith(prefix):
            return fpath[len(prefix) :]
    return fpath


def _extract_func_from_traceback(traceback: str) -> str:
    function_name = ""
    for match in re.finditer(r"in (\w+)\s*\n", traceback or ""):
        if match.group(1) != "<module>":
            function_name = match.group(1)
    return function_name


def resolve_config_from_provider(provider: str, concurrency: int = 20) -> Config:
    prov_upper = provider.upper()
    if provider.lower() == "claude":
        model = os.getenv("CLAUDE_OPENAI_MODEL") or os.getenv("ANTHROPIC_MODEL") \
            or os.getenv("OPENAI_MODEL", "gpt-4")
        base_url = os.getenv("CLAUDE_OPENAI_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL") \
            or os.getenv("OPENAI_BASE_URL", "")
        api_key = os.getenv("CLAUDE_OPENAI_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") \
            or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    else:
        model = os.getenv(f"{prov_upper}_OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4"))
        base_url = os.getenv(f"{prov_upper}_OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))
        api_key = os.getenv(f"{prov_upper}_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    return Config(
        provider=provider.lower(),
        openai_model=model,
        openai_base_url=base_url,
        openai_api_key=api_key,
        llm_max_concurrency=concurrency,
        output_dir=str(PROJECT_ROOT / "output"),
    )


class HealHandler(BaseHTTPRequestHandler):

    agent: Agent = None  # type: ignore
    trace_list = None
    repo_dir = None
    instance_id = "unknown"
    auth_token = ""
    session: Optional["HealSession"] = None
    middleware: Optional[AgentMiddleware] = None

    def log_message(self, _format, *_args):
        pass

    _total_skip = 0
    _total_fix_attempts = 0
    _total_llm_calls = 0

    def do_POST(self):
        cls = type(self)
        if cls.auth_token:
            req_token = self.headers.get("X-EH-Token", "")
            if req_token != cls.auth_token:
                kind = "stray_zombie_container" if req_token else "missing_token"
                logger.warning(
                    f"[{cls.instance_id}] Rejected request ({kind}): "
                    f"got_token={req_token[:8] or '<empty>'} want_token={cls.auth_token[:8]}"
                )
                self._respond(403, {"action": "reraise", "reason": "unauthorized"})
                return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception as error:
            logger.warning(f"[{cls.instance_id}] Bad request: {error}")
            self._respond(400, {"action": "reraise", "reason": "bad_request"})
            return

        phase = req.get("phase", "crash")
        session = cls.session
        middleware = cls.middleware
        if session is None or middleware is None:
            logger.error(f"[{cls.instance_id}] No middleware session attached to handler")
            self._respond(500, {"action": "reraise", "reason": "no_session"})
            return

        t0 = time.time()
        try:
            if phase == "replay_crash":
                if self._handle_crash_phase(req, session) is False:
                    return
                crash_num = session.active_crash.crash_num if session.active_crash else None
                self._append_trace(session, "replay", t0, extra={"code": req.get("replay_code", "")})
                session.clear_active_crash()
                self._respond(200, {"action": "replay", "crash_num": crash_num})
                return
            if phase == "crash":
                if self._handle_crash_phase(req, session) is False:
                    return
            elif phase == "execute_result":
                self._handle_execute_result_phase(req, session)
            else:
                logger.warning(f"[{cls.instance_id}] Unknown phase: {phase!r}")
                self._respond(400, {"action": "reraise", "reason": f"unknown_phase:{phase}"})
                return

            try:
                decision = middleware.step(repo_dir=cls.repo_dir or "")
            except Exception as error:
                logger.error(f"[{cls.instance_id}] middleware.step() crashed: {type(error).__name__}: {error}")
                self._append_trace(session, "agent_error", t0, extra={"error": f"{type(error).__name__}:{error}"})
                session.clear_active_crash()
                cls._total_skip += 1
                cls._total_llm_calls = session.total_llm_calls
                self._respond(200, {"action": "reraise", "reason": f"agent_error:{type(error).__name__}"})
                return

            action = (decision.get("action") or "reraise").strip()
            code = decision.get("code") or ""
            reason = decision.get("reason") or ""
            submit = bool(decision.get("submit", False))
            if action == "execute" and not submit:
                session.note_execute_trial()
                cls._total_llm_calls = session.total_llm_calls
                self._append_trace(session, "execute", t0, extra={"code": code, "submit": False})
                logger.info(f"[{cls.instance_id}] action=execute submit=false code={code[:120]!r}")
                crash_num = session.active_crash.crash_num if session.active_crash else None
                self._respond(200, {
                    "action": "execute",
                    "code": code,
                    "submit": False,
                    "crash_num": crash_num,
                })
                return

            if action == "reraise":
                cls._total_skip += 1
                cls._total_llm_calls = session.total_llm_calls
                self._append_trace(session, "reraise", t0, extra={"reason": reason})
                session.clear_active_crash()
                self._respond(200, {"action": "reraise", "reason": reason})
                return

            if action == "execute" and submit:
                cls._total_llm_calls = session.total_llm_calls
                if not code.strip():
                    cls._total_skip += 1
                    self._append_trace(session, "reraise", t0, extra={"reason": "missing_execute_code"})
                    session.clear_active_crash()
                    self._respond(200, {"action": "reraise", "reason": "missing_execute_code"})
                    return
                cls._total_fix_attempts += 1
                self._append_trace(session, "execute", t0, extra={"code": code, "submit": True, "reason": reason})
                logger.info(f"[{cls.instance_id}] action=execute submit=true code={code[:80]!r}")
                crash_num = session.active_crash.crash_num if session.active_crash else None
                session.clear_active_crash()
                self._respond(200, {
                    "action": "execute",
                    "code": code,
                    "submit": True,
                    "crash_num": crash_num,
                })
                return

            cls._total_skip += 1
            cls._total_llm_calls = session.total_llm_calls
            reason = f"unknown_action:{action}"
            self._append_trace(session, "reraise", t0, extra={"reason": reason})
            session.clear_active_crash()
            self._respond(200, {"action": "reraise", "reason": reason})
        except Exception as error:
            logger.exception(f"[{cls.instance_id}] do_POST unexpected error: {error}")
            self._append_trace(
                session,
                "handler_error",
                t0,
                extra={"error": f"{type(error).__name__}:{error}"},
            )
            session.clear_active_crash()
            cls._total_skip += 1
            cls._total_llm_calls = session.total_llm_calls
            self._respond(500, {"action": "reraise", "reason": f"handler_error:{type(error).__name__}"})


    def _handle_crash_phase(self, req: dict, session: "HealSession"):
        cls = type(self)
        crash = CrashInfo(
            file_path=req.get("file", "unknown"),
            line=req.get("line", -1),
            block_index=req.get("block_index", -1),
            error_type=req.get("error_type", "Unknown"),
            error_message=req.get("error_message", "Unknown"),
            locals_snapshot=req.get("locals", {}),
            traceback=req.get("traceback", ""),
            code_at_crash=req.get("code", ""),
        )
        crash.code_display = self._build_code_display(crash.file_path, crash.code_at_crash)
        crash.call_chain_source = self._build_call_chain_source(crash.file_path, crash.traceback)
        middleware = cls.middleware
        if middleware is None:
            raise RuntimeError("No middleware attached to handler")
        middleware.accept_crash(crash)
        locals_keys = list((crash.locals_snapshot or {}).keys())
        logger.info(
            f"[{cls.instance_id}] Crash #{crash.crash_num} [{crash.file_path}:{crash.line}] "
            f"{crash.error_type}: {crash.error_message}, locals={locals_keys}"
        )
        return True

    def _handle_execute_result_phase(self, req: dict, session: "HealSession"):
        outcome = ExecuteOutcome(
            code=req.get("execute_code", ""),
            submit=bool(req.get("submit", False)),
            stdout=req.get("execute_stdout", ""),
            value_repr=req.get("execute_value", ""),
            error=req.get("execute_error", ""),
        )
        cls = type(self)
        self._append_trace(session, "execute_result", time.time(), extra={
            "execute_code": req.get("execute_code", ""),
            "execute_stdout": req.get("execute_stdout", ""),
            "execute_value": req.get("execute_value", ""),
            "execute_error": req.get("execute_error", ""),
            "submit": bool(req.get("submit", False)),
        })
        middleware = cls.middleware
        if middleware is None:
            raise RuntimeError("No middleware attached to handler")
        middleware.accept_execute_result(outcome)


    _CONTEXT_LINES = 50

    def _build_code_display(self, file_path: str, error_code: str) -> str:
        cls = type(self)
        if error_code and cls.repo_dir:
            try:
                full_source = Path(cls.repo_dir, _normalize_container_path(file_path)).read_text(errors="replace")
                idx = full_source.find(error_code.strip())
                if idx >= 0:
                    lines = full_source.splitlines(keepends=True)
                    char_count = 0
                    crash_line = 0
                    for i, line in enumerate(lines):
                        if char_count + len(line) > idx:
                            crash_line = i
                            break
                        char_count += len(line)

                    start_line = max(0, crash_line - self._CONTEXT_LINES)
                    error_lines = error_code.strip().count('\n') + 1
                    end_line = min(len(lines), crash_line + error_lines + self._CONTEXT_LINES)

                    context_lines = lines[start_line:end_line]
                    result_parts = [f"({file_path}, lines {start_line+1}-{end_line}):\n"]
                    if start_line > 0:
                        result_parts.append(f"... [{start_line} lines above]\n")
                    for i, line in enumerate(context_lines):
                        lineno = start_line + i + 1
                        if crash_line <= start_line + i < crash_line + error_lines:
                            if i == crash_line - start_line:
                                result_parts.append(f"{lineno:4d} <error>{line.rstrip()}\n")
                            elif i == crash_line - start_line + error_lines - 1:
                                result_parts.append(f"{lineno:4d} {line.rstrip()}</error>\n")
                            else:
                                result_parts.append(f"{lineno:4d} {line.rstrip()}\n")
                        else:
                            result_parts.append(f"{lineno:4d} {line.rstrip()}\n")
                    if end_line < len(lines):
                        result_parts.append(f"... [{len(lines) - end_line} lines below]\n")

                    return "".join(result_parts)
            except Exception as e:
                logger.warning(f"[{cls.instance_id}] Failed to read full source for {file_path}: {e}")
        return f"<error>{error_code}</error>" if error_code else f"<error>(location: {file_path})</error>"

    def _build_call_chain_source(self, file_path: str, tb_text: str) -> str:
        cls = type(self)
        if not tb_text or not cls.repo_dir:
            return "(no additional source files in call chain)"
        frames = re.findall(r'File "([^"]+)", line (\d+), in (\w+)', tb_text)
        seen_files: set[str] = set()
        norm_error_file = _normalize_container_path(file_path)
        parts: list[str] = []
        for fpath, lineno, func in frames:
            if any(skip in fpath for skip in ("site-packages", "/lib/python", "<frozen", "<string>", "<module>")):
                continue
            norm_fpath = _normalize_container_path(fpath)
            if norm_fpath in seen_files or norm_fpath == norm_error_file:
                continue
            seen_files.add(norm_fpath)
            try:
                src = Path(cls.repo_dir, norm_fpath).read_text(
                    encoding="utf-8", errors="surrogateescape"
                )
                parts.append(f"\n--- {fpath}:{lineno} in {func}() ---\n{src}\n")
            except Exception as e:
                logger.warning(f"[{cls.instance_id}] Failed to read chain file {fpath}: {e}")
        return "".join(parts) or "(no additional files in call chain)"

    def _append_trace(self, session: "HealSession", action: str, t0: float, extra: Optional[dict] = None):
        cls = type(self)
        if cls.trace_list is None:
            return
        _crash = session.active_crash
        entry = {
            "turn": len(cls.trace_list) + 1,
            "crash_num": _crash.crash_num if _crash else 0,
            "file_path": _crash.file_path if _crash else "",
            "line": _crash.line if _crash else -1,
            "block_index": _crash.block_index if _crash else -1,
            "error_type": _crash.error_type if _crash else "",
            "error_message": (_crash.error_message or "") if _crash else "",
            "func_name": _extract_func_from_traceback(_crash.traceback) if _crash else "",
            "traceback": (_crash.traceback or "") if _crash else "",
            "locals_snapshot": (_crash.locals_snapshot or {}) if _crash else {},
            "code_at_crash": (_crash.code_at_crash or "") if _crash else "",
            "code_display": (_crash.code_display or "") if _crash else "",
            "call_chain_source": (_crash.call_chain_source or "") if _crash else "",
            "action": action,
            "llm_calls": session.total_llm_calls,
            "elapsed_s": round(time.time() - t0, 2),
            "wall_time_s": round(time.time() - session.created_at, 2),
        }
        if extra:
            entry.update(extra)
        if session.llm_call_records:
            entry["llm_calls_detail"] = session.llm_call_records[:]
            session.llm_call_records.clear()
        try:
            entry["messages_snapshot"] = copy.deepcopy(session.messages)
        except Exception as _e:
            logger.warning(f"[{cls.instance_id}] messages_snapshot deepcopy failed: {_e}")
            entry["messages_snapshot"] = list(session.messages)
        cls.trace_list.append(entry)

    def _respond(self, status: int, body: dict):
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.warning(f"[{type(self).instance_id}] response failed (client disconnected)")




def _start_heal_server(
    agent: Agent,
    *,
    trace,
    max_call_llm=0,
    repo_dir=None,
    port=19876,
    instance_id="unknown",
    auth_token="",
    session: Optional["HealSession"] = None,
    task_dir: str = "",
):

    if session is None:
        session = HealSession(
            instance_id=instance_id,
            max_call_llm=max_call_llm,
            repo_dir=repo_dir or "",
            task_dir=task_dir,
        )

    class InstanceHandler(HealHandler):
        pass

    InstanceHandler.agent = agent
    InstanceHandler.trace_list = trace
    InstanceHandler.repo_dir = repo_dir
    InstanceHandler.instance_id = instance_id
    InstanceHandler.auth_token = auth_token
    InstanceHandler.session = session
    InstanceHandler.middleware = AgentMiddleware(agent, session)
    InstanceHandler._total_skip = 0
    InstanceHandler._total_fix_attempts = 0
    InstanceHandler._total_llm_calls = 0
    class _ReuseAddrHTTPServer(HTTPServer):
        allow_reuse_address = True
    server = _ReuseAddrHTTPServer(("0.0.0.0", port), InstanceHandler)
    server.eh_handler_class = InstanceHandler
    server.eh_session = session
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info(f"healing server started on port {port} (session attached)")
    return server


def _loaddocker_accounts() -> list[tuple[str, str]]:
    global docker_accounts
    accounts: list[tuple[str, str]] = []
    try:
        text = DOCKER_ACCOUNTS_FILE.read_text()
        for line in text.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                accounts.append((parts[0], parts[1]))
    except Exception as e:
        logger.warning(f"failed to load docker accounts from {DOCKER_ACCOUNTS_FILE}: {e}")
    docker_accounts = accounts
    if accounts:
        logger.info(
            f"Loaded {len(accounts)} DockerHub accounts for rotation, " f"first two: {[a[0] for a in accounts[:2]]}"
        )
    return accounts


def _get_next_account() -> tuple[str, str] | None:
    global account_index
    if not docker_accounts:
        return None
    with account_lock:
        acc = docker_accounts[account_index % len(docker_accounts)]
        account_index += 1
    return acc


def _loadimage_exists_cache() -> dict[str, bool]:
    global image_exists_cache
    try:
        if IMAGE_EXISTS_CACHE_FILE.exists():
            with open(IMAGE_EXISTS_CACHE_FILE) as f:
                image_exists_cache = json.load(f)
            exist_count = sum(1 for v in image_exists_cache.values() if v)
            missing_count = sum(1 for v in image_exists_cache.values() if not v)
            sample_keys = list(image_exists_cache.keys())[:2]
            logger.info(
                f"Image cache: {len(image_exists_cache)} entries "
                f"({exist_count} exist, {missing_count} missing), "
                f"first two: {sample_keys}"
            )
    except Exception as e:
        logger.warning(f"failed to load image cache from {IMAGE_EXISTS_CACHE_FILE}: {e}")
        image_exists_cache = {}
    return image_exists_cache


def _saveimage_exists_cache() -> None:
    with image_cache_lock:
        IMAGE_EXISTS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(IMAGE_EXISTS_CACHE_FILE, "w") as f:
            json.dump(image_exists_cache, f)


def _docker_pull_with_rotation(image: str, timeout: int = 300, retries: int = 3) -> bool:
    with image_cache_lock:
        cached = image_exists_cache.get(image)
    if cached is False:
        logger.info(f"Skipping pull for {image} — cached as missing")
        return False

    with pull_semaphore:
        for attempt in range(retries):
            acc = _get_next_account()
            env = None

            if acc:
                username, token = acc
                config_dir = Path(tempfile.gettempdir()) / f"docker_cfg_{username}"
                config_dir.mkdir(exist_ok=True)
                auth_str = base64.b64encode(f"{username}:{token}".encode()).decode()
                config_file = config_dir / "config.json"
                config_file.write_text(json.dumps({"auths": {"https://index.docker.io/v1/": {"auth": auth_str}}}))
                env = {**os.environ, "DOCKER_CONFIG": str(config_dir)}

            try:
                r = subprocess.run(
                    ["docker", "pull", image],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
                if r.returncode == 0:
                    with image_cache_lock:
                        image_exists_cache[image] = True
                    logger.info(f"Pulled {image} successfully (attempt {attempt + 1})")
                    return True
                stderr = r.stderr or ""
                if "not found" in stderr or "manifest unknown" in stderr:
                    with image_cache_lock:
                        image_exists_cache[image] = False
                    logger.warning(f"Image {image} not found on Docker Hub")
                    return False
                if "toomanyrequests" in stderr.lower() or "rate limit" in stderr.lower():
                    acct_name = acc[0] if acc else "anonymous"
                    logger.warning(f"Rate limited ({acct_name}) for {image} (attempt {attempt + 1})")
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
            except subprocess.TimeoutExpired:
                logger.warning(f"docker pull timed out for {image} (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
            except Exception as e:
                logger.warning(f"docker pull unexpected error for {image}: {e}")
                return False
    return False


def _remove_docker_image(image: str) -> None:
    try:
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True, text=True, timeout=120)
        logger.info(f"Removed docker image {image}")
    except Exception as e:
        logger.warning(f"failed to remove docker image {image}: {e}")




def _resolve_docker_image(rec: dict) -> tuple[str, str]:
    tag = rec.get("dockerhub_tag", "")
    if tag and "/" in tag:
        return tag, "/testbed"
    if tag:
        return f"jefzda/sweap-images:{tag}", "/app"
    oid = rec.get("origin_instance_id", "")
    return f"sweb.eval.x86_64.{oid}:latest", "/testbed"


def _ensure_docker_images(records: list[dict]) -> set[str]:
    failed_ids: set[str] = set()

    pro_recs = [r for r in records if r.get("dockerhub_tag") and "/" not in r["dockerhub_tag"]]
    r2e_recs = [r for r in records if r.get("dockerhub_tag") and "/" in r["dockerhub_tag"]]
    build_recs = [r for r in records if not r.get("dockerhub_tag")]

    for r in pro_recs:
        image, _ = _resolve_docker_image(r)
        iid = r["instance_id"]
        chk = subprocess.run(["docker", "images", "-q", image], capture_output=True, text=True)
        if chk.stdout.strip():
            continue
        logger.info(f"Pulling {image} for {iid}")
        ret = subprocess.run(["docker", "pull", image], capture_output=True, text=True, timeout=600)
        if ret.returncode != 0:
            logger.error(f"docker pull failed for {iid}: {ret.stderr.strip()}")
            failed_ids.add(iid)

    if r2e_recs:
        _loaddocker_accounts()
        _loadimage_exists_cache()

        to_pull: list[tuple[str, str, str]] = []
        for r in r2e_recs:
            image, _ = _resolve_docker_image(r)
            iid = r["instance_id"]
            chk = subprocess.run(["docker", "images", "-q", image], capture_output=True, text=True)
            if chk.stdout.strip():
                continue
            to_pull.append((image, iid, r["dockerhub_tag"]))

        if to_pull:
            logger.info(f"Pulling {len(to_pull)} R2E-Gym Docker images (parallel, account rotation)...")

            def _pull_one(args):
                img, iid, _ = args
                ok = _docker_pull_with_rotation(img)
                return iid, ok

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(_pull_one, item): item[1] for item in to_pull}
                done_count = 0
                for future in as_completed(futures):
                    iid = futures[future]
                    done_count += 1
                    try:
                        _, ok = future.result()
                        if not ok:
                            logger.error(f"docker pull failed for {iid}")
                            failed_ids.add(iid)
                    except Exception as e:
                        logger.error(f"docker pull crashed for {iid}: {e}")
                        failed_ids.add(iid)
                    if done_count % 20 == 0:
                        logger.info(f"R2E-Gym pull progress: {done_count}/{len(to_pull)}")

            _saveimage_exists_cache()
            logger.info(
                f"R2E-Gym pull complete: {len(to_pull) - len(failed_ids & {t[1] for t in to_pull})} ok, "
                f"{len(failed_ids & {t[1] for t in to_pull})} failed"
            )

    if build_recs:
        failed_ids |= _build_swebench_images(build_recs)

    return failed_ids


def _build_swebench_images(records: list[dict]) -> set[str]:
    import docker
    from datasets import load_dataset as hf_load
    from swebench.harness.docker_build import build_instance_images, make_test_spec

    failed_ids: set[str] = set()

    hf_cache = PROJECT_ROOT / "datasets" / "hf_cache"
    hf_map: dict[str, dict] = {}
    for hf_name in ["princeton-nlp/SWE-bench_Verified", "princeton-nlp/SWE-bench"]:
        ds = hf_load(hf_name, split="test", cache_dir=str(hf_cache))
        for row in ds:
            hf_map[row["instance_id"]] = dict(row)

    to_build = []
    for r in records:
        oid = r.get("origin_instance_id", "")
        if oid in hf_map:
            to_build.append(hf_map[oid])
        else:
            logger.error(f"HF record not found for origin_instance_id={oid}, cannot build image")
            failed_ids.add(r["instance_id"])

    if not to_build:
        return failed_ids

    client = docker.from_env()
    still_needed = []
    for hf_rec in to_build:
        spec = make_test_spec(hf_rec)
        try:
            client.images.get(spec.instance_image_key)
            logger.info(f"Image already exists: {spec.instance_image_key}")
        except docker.errors.ImageNotFound:
            still_needed.append(hf_rec)

    if not still_needed:
        return failed_ids

    logger.info(f"Building {len(still_needed)} swebench Docker images...")
    successful, build_failed = build_instance_images(
        client,
        still_needed,
        max_workers=min(4, len(still_needed)),
        env_image_tag="latest",
        tag="latest",
    )
    if build_failed:
        logger.error(f"{len(build_failed)} swebench images failed to build")
        built_origin_ids = set()
        for hf_rec in still_needed:
            spec = make_test_spec(hf_rec)
            try:
                client.images.get(spec.instance_image_key)
                built_origin_ids.add(hf_rec["instance_id"])
            except docker.errors.ImageNotFound:
                logger.info(f"image not found after build: {hf_rec['instance_id']}")
        for r in records:
            oid = r.get("origin_instance_id", "")
            if oid not in built_origin_ids and not r.get("dockerhub_tag"):
                failed_ids.add(r["instance_id"])

    logger.info(f"Swebench images ready: {len(successful)} built, {len(build_failed)} failed")
    return failed_ids


def docker_extract_repo(image: str, before_cmd: str, dest: str, workdir: str = "/app") -> bool:
    exclusions = " ".join(
        f"--exclude={shlex.quote(pattern)}"
        for pattern in sorted({*REPOSITORY_EXCLUDED_DIRECTORIES, *REPOSITORY_EXCLUDED_PATTERNS})
    )
    tar_cmd = (
        f"({before_cmd}) >/dev/null 2>&1 && "
        f"tar cf - -C {workdir} {exclusions} ."
    )
    for attempt in range(3):
        try:
            proc = subprocess.run(
                ["docker", "run", "--rm", "--entrypoint", "/bin/bash", image, "-c", tar_cmd],
                capture_output=True,
                timeout=DOCKER_TIMEOUT,
            )
            if proc.returncode != 0:
                logger.warning(f"docker tar attempt {attempt+1}/3 failed (rc={proc.returncode}): {proc.stderr}")
                if attempt < 2:
                    time.sleep(2)
                    continue
                return False
            os.makedirs(dest, exist_ok=True)
            proc2 = subprocess.run(["tar", "xf", "-", "-C", dest], input=proc.stdout, capture_output=True, timeout=600)
            if proc2.returncode != 0:
                logger.warning(f"untar attempt {attempt+1}/3 failed: {proc2.stderr}")
                if attempt < 2:
                    shutil.rmtree(dest, ignore_errors=True)
                    time.sleep(2)
                    continue
                return False
            return True
        except Exception as e:
            logger.warning(f"docker extract attempt {attempt+1}/3 error: {e}")
            if attempt < 2:
                shutil.rmtree(dest, ignore_errors=True)
                time.sleep(2)
            else:
                return False
    return False




def docker_run_instrumented(
    image: str,
    before_cmd: str,
    instr_dir: str,
    tests: list[str],
    record_dir: str,
    workdir: str = "/app",
    patch_file: str = "",
    test_patch_file: str = "",
    record_trace: bool = False,
    runtime_trace_path: str = "",
    instrumentation_enabled: bool = True,
    path_points_enabled: bool = False,
) -> dict:
    test_args = " ".join(shlex.quote(t) for t in tests)
    instr_dir = str(Path(instr_dir).resolve())
    record_dir = str(Path(record_dir).resolve())
    vol_flags = [
        "-v",
        f"{instr_dir}:{workdir}/_overlay:ro",
        "-v",
        f"{record_dir}:{workdir}/_records:rw",
    ]
    trace_env = []
    if instrumentation_enabled and not runtime_trace_path:
        raise ValueError("instrumentation_enabled requires runtime_trace_path")
    if runtime_trace_path:
        trace_dir = str(Path(runtime_trace_path).resolve().parent)
        vol_flags += ["-v", f"{trace_dir}:/runtime_trace:rw"]
        trace_env = [
            "-e", "EH_RUNTIME_TRACE_FILE=/runtime_trace/run_docker_events.jsonl",
            "-e", "EH_RUNTIME_TRACE_RUN_ID=healed",
            "-e", f"EH_RUNTIME_TRACE_ROOT={workdir}",
            "-e", f"EH_PATH_POINTS_ENABLED={int(path_points_enabled)}",
        ]

    tracer_setup_cmd = ""
    tracer_pytest_flag = ""
    if record_trace:
        if instrumentation_enabled:
            from heal_builtins import build_builtins_setup_cmd
            tracer_setup_cmd = build_builtins_setup_cmd()
        else:
            from run_record_gold_trace import build_eh_setup_cmd
            tracer_setup_cmd = build_eh_setup_cmd()
        tracer_pytest_flag = " --capture=no"

    revert_cmd = ""
    if patch_file and os.path.exists(patch_file):
        vol_flags += ["-v", f"{patch_file}:/tmp/_reverse.patch:ro"]
        revert_cmd = f"patch -R -p1 --no-backup-if-mismatch -d {workdir} -i /tmp/_reverse.patch || echo '[EH] reverse-patch failed rc='$?; "
    test_patch_prepare_cmd = ""
    test_patch_restore_cmd = ""
    if test_patch_file and os.path.exists(test_patch_file):
        vol_flags += ["-v", f"{test_patch_file}:/tmp/_test.patch:ro"]
        patch_text = Path(test_patch_file).read_text(encoding="utf-8", errors="replace")
        patched_files = sorted(set(re.findall(r"^\+\+\+ b/(.+)$", patch_text, re.MULTILINE)))
        unsafe_files = [path for path in patched_files if Path(path).is_absolute() or ".." in Path(path).parts]
        if not patched_files or unsafe_files:
            raise ValueError(f"Invalid test patch paths: files={patched_files}, unsafe={unsafe_files}")
        stage_root = "/tmp/_eh_test_patch_files"
        stage_commands = []
        restore_commands = []
        for relative_path in patched_files:
            relative_parent = str(Path(relative_path).parent)
            quoted_path = shlex.quote(relative_path)
            quoted_parent = shlex.quote(relative_parent)
            staged_path = shlex.quote(f"{stage_root}/{relative_path}")
            staged_parent = shlex.quote(f"{stage_root}/{relative_parent}")
            stage_commands.append(
                f"mkdir -p {staged_parent} && cp -a {quoted_path} {staged_path}"
            )
            restore_commands.append(
                f"mkdir -p {quoted_parent} && cp -a {staged_path} {quoted_path}"
            )
        test_patch_prepare_cmd = (
            f"cd {workdir} && "
            f"((git apply --check /tmp/_test.patch && git apply /tmp/_test.patch) || "
            f"git apply --reverse --check /tmp/_test.patch) && "
            f"rm -rf {stage_root} && mkdir -p {stage_root} && "
            f"{' && '.join(stage_commands)} && "
        )
        test_patch_restore_cmd = f"cd {workdir} && {' && '.join(restore_commands)} && "
    prep_cmd, _ = build_instr_prep_cmd(workdir, do_overlay=True, use_pythonpath=False, eh_setup_cmd="")
    inner_cmd = (
        f"{before_cmd} && "
        f"{revert_cmd}"
        f"{test_patch_prepare_cmd}"
        f"{prep_cmd}"
        f"{test_patch_restore_cmd}"
        f""
        f"{tracer_setup_cmd}"
        f"export _EH_ACTIVE=1 EH_EPMR_ENABLED={int(path_points_enabled)} && "
        f"unset PYTEST_ADDOPTS && "
        f"python -m pytest {test_args} --tb=long -v{tracer_pytest_flag}; _PYTEST_RC=$?; unset EH_EPMR_ENABLED; "
        f"cat /tmp/_repair_records*.json 2>/dev/null | python -c \""
        f"import sys,json; parts=[json.loads(l) for l in sys.stdin if l.strip()]; "
        f"merged=[x for p in parts for x in (p if isinstance(p,list) else [p])]; "
        f"open('{workdir}/_records/repair_records.json','w').write(json.dumps(merged))\" 2>/dev/null; "
        f"exit $_PYTEST_RC"
    )
    import uuid as _uuid

    container_name = f"eh_{_uuid.uuid4().hex[:12]}"
    try:
        proc = subprocess.run(
            [
                "docker",
                "run",
                "--name",
                container_name,
                "--add-host=host.docker.internal:host-gateway",
                *trace_env,
                *vol_flags,
                "--entrypoint",
                "/bin/bash",
                image,
                "-c",
                inner_cmd,
            ],
            capture_output=True,
            text=True,
            timeout=PYTEST_MAIN_TIMEOUT,
        )
        records_file = Path(record_dir) / "repair_records.json"
        records = []
        if records_file.exists():
            try:
                records = json.loads(records_file.read_text("utf-8"))
            except Exception as e:
                logger.error(f"Failed to parse repair_records.json: {e}")
        result = {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode, "repair_records": records}
        return result
    except subprocess.TimeoutExpired as error:
        logger.warning(f"Docker container {container_name} timed out after {PYTEST_MAIN_TIMEOUT}s, killing...")
        try:
            subprocess.run(["docker", "kill", container_name], capture_output=True, timeout=60)
        except Exception as e:
            logger.warning(f"failed to kill container {container_name}: {e}")
        stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        return {"stdout": stdout, "stderr": stderr + "\nTIMEOUT", "returncode": -1, "repair_records": []}
    except Exception as e:
        logger.error(f"Docker execution failed for {container_name}: {e}")
        return {"stdout": "", "stderr": str(e), "returncode": -1, "repair_records": []}
    finally:
        try:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=30)
        except Exception as e:
            logger.warning(f"failed to rm container {container_name}: {e}")


def load_dataset_records(source_filter: str | None = None) -> tuple[Path, list[dict]]:
    if not HEALBENCH_JSONL.exists():
        raise FileNotFoundError(f"Dataset file not found: {HEALBENCH_JSONL}")
    all_records = [json.loads(line) for line in HEALBENCH_JSONL.read_text("utf-8").strip().splitlines()]
    if source_filter:
        records = [r for r in all_records if r.get("source_dataset") == source_filter]
        logger.info(
            f"Loaded {len(records)}/{len(all_records)} instances from {HEALBENCH_JSONL.name} "
            f"(source_dataset={source_filter}), first 2: {[r['instance_id'][:40] for r in records[:2]]}"
        )
    else:
        records = all_records
        logger.info(
            f"Loaded {len(records)} instances from {HEALBENCH_JSONL.name}, "
            f"first 2: {[r['instance_id'][:40] for r in records[:2]]}"
        )
    return HEALBENCH_JSONL, records


def build_port_pool(concurrency: int, port_base: int | None = None) -> asyncio.Queue:
    port_pool: asyncio.Queue = asyncio.Queue()
    p = port_base if port_base is not None else HOST_PORT_BASE
    unavailable_ports = []
    while port_pool.qsize() < concurrency:
        if p not in EXCLUDED_PORTS:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(("0.0.0.0", p))
                except OSError as error:
                    unavailable_ports.append((p, str(error)))
                else:
                    port_pool.put_nowait(p)
        p += 1
    if unavailable_ports:
        logger.warning(
            f"Skipped {len(unavailable_ports)} unavailable Heal ports, first 2: "
            f"{unavailable_ports[:2]}"
        )
    return port_pool




async def run_single_test(
    rec: dict,
    agent: Agent,
    config: Config,
    results_dir: Path,
    max_call_llm: int,
    port: int,
    image: str,
    before_cmd: str,
    repo_dir: str,
    instr_dir: str,
    instr_count: int,
    total_funcs: int,
    workdir: str = "/app",
    patch_file: str = "",
    auth_token: str = "",
    record_trace: bool = False,
    instrumentation_enabled: bool = True,
    path_points_enabled: bool = False,
) -> dict:
    iid = rec["instance_id"]
    task_dir = results_dir / iid
    if task_dir.exists() and not (task_dir / "output.json").exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "input_raw.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    task_log_id = logger.add(
        task_dir / "run.log",
        level="DEBUG",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}",
        filter=lambda record: iid in record["message"],
    )
    _tee_fh = _bind_thread_run_log(task_dir / "run.log")

    target_tests = parse_test_list(rec.get("fail_to_pass", "[]"))

    tag = rec.get("dockerhub_tag", "")
    if tag and "/" in tag:
        if not target_tests:
            test_name = rec.get("test_name", "")
            if test_name:
                target_tests = [test_name.replace("../r2e_tests/", "/r2e_tests/")]
            else:
                target_tests = ["/r2e_tests/"]
        else:
            test_file = rec.get("test_file", "").replace("../r2e_tests/", "/r2e_tests/")
            if not test_file:
                test_file = "/r2e_tests/test_1.py"
            converted = []
            for t in target_tests:
                if "::" in t or "/" in t:
                    converted.append(t.replace("../r2e_tests/", "/r2e_tests/"))
                else:
                    converted.append(f"{test_file}::{t.replace('.', '::')}")
            target_tests = converted

    def save_result(result):
        try:
            _ensure_result_files(task_dir, input_data, result)
        finally:
            try:
                logger.remove(task_log_id)
            except ValueError as e:
                logger.warning(f"logger handler already removed: {e}")
            _unbind_thread_run_log(_tee_fh)
        return result

    input_data = {**rec, "instrumented_files": instr_count, "instrumented_functions": total_funcs}
    (task_dir / "input.json").write_text(
        json.dumps(input_data, indent=2, ensure_ascii=False, default=str),
        "utf-8",
    )

    error_type = rec.get("error_type", "")
    if error_type in ("SegfaultAbort", "StartupCrash", "InfraError"):
        logger.info(f"{iid}: skipping ({error_type})")
        return save_result({"instance_id": iid, "status": "skipped", "reason": error_type})

    logger.info(f"{iid}: budget={max_call_llm}, agent={type(agent).__name__}")

    trace: list[dict] = []

    server = None
    try:
        server = _start_heal_server(
            agent,
            trace=trace,
            max_call_llm=max_call_llm,
            repo_dir=repo_dir,
            port=port,
            instance_id=iid,
            auth_token=auth_token,
            task_dir=str(task_dir),
        )
        record_dir = str(task_dir / ".tmp" / "records")
        os.makedirs(record_dir, exist_ok=True)

        test_patch_file = ""
        test_patch_text = rec.get("test_patch", "")
        if test_patch_text:
            test_patch_file = os.path.join(record_dir, "test.patch")
            with open(test_patch_file, "w") as f:
                f.write(test_patch_text)
                if not test_patch_text.endswith("\n"):
                    f.write("\n")

        logger.info(f"{iid}: running pytest (budget={max_call_llm})...")
        if record_trace:
            from instrument_guard import prepare_runtime_trace, write_runtime_trace_metadata
            runtime_trace_path = str(task_dir / "run_docker_events.jsonl")
            prepare_runtime_trace(runtime_trace_path)
            if not instrumentation_enabled:
                write_runtime_trace_metadata(
                    runtime_trace_path,
                    repo_dir,
                    str(Path(instr_dir) / "_source_map.json"),
                    str(Path(instr_dir) / "_trace_manifest.json"),
                )
        t0 = time.time()
        try:
            result = await asyncio.to_thread(
                docker_run_instrumented,
                image,
                before_cmd,
                instr_dir,
                target_tests,
                record_dir,
                workdir,
                patch_file,
                test_patch_file,
                record_trace,
                str(task_dir / "run_docker_events.jsonl") if record_trace else "",
                instrumentation_enabled,
                path_points_enabled,
            )
        except Exception as e:
            logger.error(f"{iid}: docker_run_instrumented failed: {e}")
            return save_result({"instance_id": iid, "status": "infra_error", "reason": str(e)})
        run_time = time.time() - t0

        test_rc = result["returncode"]
        handler_cls = server.eh_handler_class
        total_skip = handler_cls._total_skip
        total_fix_attempts = handler_cls._total_fix_attempts
        llm_calls = handler_cls._total_llm_calls
        _session = getattr(server, "eh_session", None)
        input_tokens = _session.total_input_tokens if _session else 0
        output_tokens = _session.total_output_tokens if _session else 0
        bash_count = _session.bash_count if _session else 0
        repair_records = result.get("repair_records", [])
        total_fix = sum(record.get("status") in {"applied", "replay"} for record in repair_records)
        rc_desc = {0: "pass", -1: "timeout"}.get(test_rc, f"fail(rc={test_rc})")
        logger.info(
            f"{iid}: test rc={test_rc} ({rc_desc}), fix={total_fix}, "
            f"fix_attempts={total_fix_attempts}, skip={total_skip}, "
            f"llm_calls={llm_calls}, time={run_time:.1f}s"
        )

        runtime_trace = None
        if record_trace:
            from instrument_guard import finalize_runtime_trace
            finalized_trace = finalize_runtime_trace(str(task_dir / "run_docker_events.jsonl"))
            finalized_trace.pop("path", None)
            finalized_trace.pop("snapshots_path", None)
            finalized_trace.pop("values_path", None)
            runtime_trace = {**finalized_trace, "storage": "run_docker_events.jsonl"}
            if runtime_trace["malformed"]:
                logger.error(f"{iid}: runtime trace malformed: {runtime_trace}")
            elif runtime_trace["records"] == 0:
                logger.info(f"{iid}: runtime trace has no Heal events")
            elif runtime_trace["collector_failures"]:
                logger.warning(f"{iid}: runtime trace captured fallback values: {runtime_trace['collector_failures']}")

        baseline_config = {
            "agent_class": type(agent).__name__,
            "model": config.openai_model,
            "max_call_llm": max_call_llm,
        }
        for attr in ("step_limit", "cost_limit", "cost_tracking"):
            if hasattr(agent, attr):
                baseline_config[attr] = getattr(agent, attr)
        trace_payload = {
            "baseline_config": baseline_config,
            "events": trace,
        }
        (task_dir / "trace.json").write_text(
            json.dumps(trace_payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        termination_rcs = {-15, -9, 137, 143}
        if test_rc in (125, 126, 127):
            status = "infra_error"
            failure_reason = f"container_exec_failure:rc={test_rc}"
        elif test_rc == -1:
            status = "infra_error"
            failure_reason = "pytest_timeout"
        elif test_rc in termination_rcs:
            status = "infra_error"
            failure_reason = f"pytest_terminated:rc={test_rc}"
        else:
            status = "correct" if test_rc == 0 else "fail"
            failure_reason = None
        logger.info(
            f"{iid}: {status}"
            + (f" ({failure_reason})" if failure_reason else "")
        )

        out = {
            "instance_id": iid,
            "status": status,
            "test_rc": test_rc,
            "total_fix": total_fix,
            "total_skip": total_skip,
            "total_fix_attempts": total_fix_attempts,
            "llm_calls": llm_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "bash_count": bash_count,
            "strategy": type(agent).__name__,
            "max_call_llm": max_call_llm,
            "repair_records": repair_records,
            "run_time_s": round(run_time, 2),
            "instrumented_files": instr_count,
            "instrumented_functions": total_funcs,
            "model": config.openai_model,
            "runtime_trace": runtime_trace,
        }
        if failure_reason:
            out["reason"] = failure_reason
        if runtime_trace:
            runtime_trace["required"] = total_fix_attempts > 0
        if runtime_trace and runtime_trace["required"] and not runtime_trace["complete"]:
            out["status"] = "infra_error"
            out["reason"] = "runtime_trace_incomplete"
        if instrumentation_enabled and "HealedSinkBlocked" in result.get("stderr", ""):
            out["status"] = "guard_blocked"
            out["reason"] = "healed_label_reached_confirmed_sink"

        if record_trace and not instrumentation_enabled:
            from run_record_gold_trace import _parse_markers_from_stderr
            healed_trace = _parse_markers_from_stderr(result.get("stderr", ""))
            ht_path = task_dir / "healed_trace.json"
            ht_path.write_text(json.dumps(healed_trace, ensure_ascii=False), "utf-8")
            _br, _rs = len(healed_trace.get("branch_markers", [])), len(healed_trace.get("raise_markers", []))
            if _br or _rs:
                logger.info(f"{iid}: captured healed markers (BR={_br}, RS={_rs})")
            else:
                logger.warning(f"{iid}: captured no healed markers; test may not have reached instrumented code")

        _merge_result_artifacts(task_dir, trace, result)
        return save_result(out)
    except Exception as e:
        logger.error(f"{iid}: run failed: {e}")
        return save_result({
            "instance_id": iid,
            "status": "infra_error",
            "reason": f"run_failed:{type(e).__name__}:{e}",
        })
    finally:
        if server is not None:
            try:
                _session = getattr(server, "eh_session", None)
                agent.cleanup(_session)
            except Exception as e:
                logger.warning(f"{iid}: agent.cleanup() failed: {e}")
            try:
                server.shutdown()
                server.server_close()
            except Exception as e:
                logger.warning(f"{iid}: server.shutdown()/server_close() failed: {e}")




async def run_batched_with_prefetch(
    items: list,
    batch_size: int,
    process_item,
) -> None:
    batches = [items[index:index + batch_size] for index in range(0, len(items), batch_size)]
    current_tasks = []
    for batch_index, batch in enumerate(batches):
        start_event = asyncio.Event()
        tasks = [asyncio.create_task(process_item(item, start_event)) for item in batch]
        if batch_index == 0:
            start_event.set()
        if current_tasks:
            await asyncio.gather(*current_tasks)
            start_event.set()
        current_tasks = tasks
    if current_tasks:
        await asyncio.gather(*current_tasks)


def _apply_runtime_sink_overlay(repository: Path, instance_id: str) -> int:
    overlay_path = os.environ.get("AGENTICHEAL_RUNTIME_SINK_OVERLAY", "")
    if not overlay_path:
        return 0
    payload = json.loads(Path(overlay_path).read_text(encoding="utf-8"))
    rules = [
        rule for rule in payload.get("rules", [])
        if rule.get("guard", {}).get("instance_id") == instance_id
    ]
    if not rules:
        return 0

    instrumentation_path = repository / "_instrumentation.json"
    instrumentation = json.loads(instrumentation_path.read_text(encoding="utf-8"))
    artifacts = instrumentation.get("artifacts", {})
    source_map_path = repository / artifacts["source_map"]["path"]
    guard_manifest_path = repository / artifacts["guard_manifest"]["path"]
    source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
    guard_manifest = json.loads(guard_manifest_path.read_text(encoding="utf-8"))

    patched_files: dict[str, list[str]] = {}
    for rule in rules:
        guard = rule["guard"]
        relative_file = str(guard["location"]["file"])
        original_line = int(guard["location"]["line"])
        file_lines = patched_files.get(relative_file)
        if file_lines is None:
            file_lines = (repository / relative_file).read_text(
                encoding="utf-8", errors="surrogateescape"
            ).splitlines(keepends=True)
            patched_files[relative_file] = file_lines
        mapped_lines = [
            int(line) for line, mapped in source_map.get(relative_file, {}).items()
            if int(mapped) == original_line
        ]
        metadata = {
            "sink_id": f"overlay-{rule['rule_id']}",
            "sink_kind": rule["sink_kind"],
            "runtime_rule_id": rule["rule_id"],
            "actual_call_target": rule["fully_qualified_callee_or_event"],
            "static_influence_names": list(guard.get("healed_variables", [])),
            "runtime_overlay": True,
        }
        replacement_index = None
        for physical_line in mapped_lines:
            index = physical_line - 1
            source_line = file_lines[index]
            indent = source_line[: len(source_line) - len(source_line.lstrip())]
            statement = source_line.strip()
            newline = "\n" if source_line.endswith("\n") else ""
            if rule.get("template") == "exact_assignment" and statement.startswith("table.columns = "):
                value = statement.split("=", 1)[1].strip()
                file_lines[index] = (
                    f"{indent}__import__('builtins')._eh_guard_sink_call("
                    f"{metadata!r}, lambda _eh_value: setattr(table, 'columns', _eh_value), "
                    f"locals(), {value}){newline}"
                )
                replacement_index = index
                break
            if rule.get("template") == "exact_assignment" and statement == "cmap_d[rmap.name] = rmap":
                file_lines[index] = (
                    f"{indent}__import__('builtins')._eh_guard_sink_call("
                    f"{metadata!r}, lambda _eh_key, _eh_value: cmap_d.__setitem__(_eh_key, _eh_value), "
                    f"locals(), rmap.name, rmap){newline}"
                )
                replacement_index = index
                break
            if rule.get("template") == "exact_assignment" and statement.startswith("out._mask = "):
                value = statement.split("=", 1)[1].strip()
                file_lines[index] = (
                    f"{indent}__import__('builtins')._eh_guard_sink_call("
                    f"{metadata!r}, lambda _eh_value: setattr(out, '_mask', _eh_value), "
                    f"locals(), {value}){newline}"
                )
                replacement_index = index
                break
            if rule.get("template") == "exact_call" and statement.startswith("np.copyto(") and statement.endswith(")"):
                arguments = statement[len("np.copyto("):-1]
                file_lines[index] = (
                    f"{indent}__import__('builtins')._eh_guard_sink_call("
                    f"{metadata!r}, np.copyto, locals(), {arguments}){newline}"
                )
                replacement_index = index
                break
            if rule.get("template") == "exact_call" and statement == "result._update_from(self)":
                file_lines[index] = (
                    f"{indent}__import__('builtins')._eh_guard_sink_call("
                    f"{metadata!r}, result._update_from, locals(), self){newline}"
                )
                replacement_index = index
                break
        if replacement_index is None:
            raise RuntimeError(
                f"Runtime Sink overlay target not found for {rule['rule_id']} at "
                f"{relative_file}:{original_line}; mapped lines={mapped_lines}"
            )
        sink_record = {
            **metadata,
            "file": relative_file,
            "line": original_line,
            "column": int(guard["location"].get("column", 0)),
            "end_line": int(guard["location"].get("end_line", original_line)),
            "end_column": int(guard["location"].get("end_column", 0)),
            "status": "CONFIRMED",
            "calls": [rule.get("callee") or rule["fully_qualified_callee_or_event"]],
            "rule_ids": [rule["rule_id"]],
        }
        file_record = guard_manifest.setdefault("files", {}).setdefault(relative_file, {"sinks": []})
        file_record.setdefault("sinks", []).append(sink_record)

    for relative_file, file_lines in patched_files.items():
        (repository / relative_file).write_text(
            "".join(file_lines), encoding="utf-8", errors="surrogateescape"
        )
    guard_manifest_path.write_text(
        json.dumps(guard_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts["guard_manifest"]["sha256"] = hashlib.sha256(
        guard_manifest_path.read_bytes()
    ).hexdigest()
    instrumentation_path.write_text(
        json.dumps(instrumentation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        f"[instance {instance_id}] applied {len(rules)} runtime Sink overlay rules; "
        f"first 2: {[rule['rule_id'] for rule in rules[:2]]}"
    )
    return len(rules)


async def process_instance(
    record: dict,
    start_event: asyncio.Event,
    agent: Agent,
    config: Config,
    results_dir: Path,
    max_call_llm: int,
    port_pool: asyncio.Queue,
    sem: asyncio.Semaphore,
    stats: dict,
    stats_lock: threading.Lock,
    cleanup_images: bool = False,
    record_trace: bool = False,
    instrumentation_enabled: bool = True,
    path_points_enabled: bool = False,
) -> None:
    sample = record
    image, workdir = _resolve_docker_image(sample)
    instance_id = str(sample["instance_id"])
    instance_label = f"[instance {instance_id}]"

    safe_instance_id = re.sub(r"[^A-Za-z0-9_.-]", "_", instance_id)
    group_tmp_root = results_dir / ".tmp" / safe_instance_id
    group_tmp_root.mkdir(parents=True, exist_ok=True)
    tmpdir = str(group_tmp_root)
    try:
        group_patch_file = ""
        if sample.get("dockerhub_tag", "") and "/" in sample["dockerhub_tag"]:
            patch_text = sample.get("patch", "")
            if patch_text:
                group_patch_file = os.path.join(tmpdir, "reverse.patch")
                Path(group_patch_file).write_text(patch_text, encoding="utf-8")

        instr_dir = os.path.join(tmpdir, "instrumented")
        shutil.rmtree(instr_dir, ignore_errors=True)
        prebuilt_root = Path(os.environ.get(
            "AGENTICHEAL_INSTRUMENTED_ROOT",
            PROJECT_ROOT / "results" / "repo_instrumented",
        ))
        prebuilt_repo = prebuilt_root / sample["instance_id"] / "repo"
        prebuilt_manifest_path = prebuilt_repo / "_instrumentation.json"
        if not prebuilt_repo.is_dir() or not prebuilt_manifest_path.is_file():
            raise RuntimeError(
                f"Prebuilt repository instrumentation is missing for {sample['instance_id']}; "
                f"run scripts/run_instrument_all.py first"
            )
        prebuilt_manifest = json.loads(prebuilt_manifest_path.read_text(encoding="utf-8"))
        prebuilt_build = prebuilt_manifest.get("build", {})
        if prebuilt_build.get("status") != "completed":
            raise RuntimeError(
                f"Prebuilt repository instrumentation is incomplete for {sample['instance_id']}; "
                f"run scripts/run_instrument_all.py --instance-id {sample['instance_id']} --force"
            )
        shutil.copytree(prebuilt_repo, instr_dir, symlinks=True)
        overlay_count = _apply_runtime_sink_overlay(Path(instr_dir), instance_id)
        instr_count = int(prebuilt_build.get("heal_instrumented_files", 0))
        total_funcs = int(prebuilt_build.get("heal_instrumented_functions", 0))
        logger.info(
            f"{instance_label} prefetched prebuilt instrumented repository: "
            f"{instr_count} files, {total_funcs} functions"
        )
        if instr_count == 0:
            task_dir = results_dir / instance_id
            result = {"instance_id": instance_id, "status": "infra_error", "reason": "no functions instrumented"}
            _ensure_result_files(task_dir, sample, result)
            with stats_lock:
                stats["infra_error"] = stats.get("infra_error", 0) + 1
            return

        await start_event.wait()
        async with sem:
            port = await port_pool.get()
            try:
                import uuid as _uuid

                _auth_token = _uuid.uuid4().hex
                _record_path = f"/tmp/_repair_records_{_uuid.uuid4().hex[:12]}.json"
                instance_repo_dir = os.path.join(tmpdir, f"repo_{port}_{_auth_token[:12]}")
                shutil.copytree(instr_dir, instance_repo_dir, symlinks=True)

                port_instr_dir = os.path.join(tmpdir, f"instr_{port}_{_auth_token[:12]}")
                try:
                    shutil.copytree(
                        instr_dir,
                        port_instr_dir,
                        symlinks=True,
                        copy_function=os.link,
                    )
                except OSError as error:
                    logger.warning(
                        f"Hard-link overlay copy failed for {instance_id}: {error}; "
                        "retrying with regular file copies"
                    )
                    shutil.rmtree(port_instr_dir, ignore_errors=True)
                    shutil.copytree(instr_dir, port_instr_dir, symlinks=True)
                for file_path in Path(port_instr_dir).rglob("*.py"):
                    content = file_path.read_text("utf-8", errors="surrogateescape")
                    if (
                        REPAIR_PORT_MARKER in content
                        or REPAIR_RECORD_PATH_MARKER in content
                        or "__EH_TOKEN__" in content
                    ):
                        content = content.replace(REPAIR_PORT_MARKER, str(port))
                        content = content.replace(REPAIR_RECORD_PATH_MARKER, _record_path)
                        content = content.replace("__EH_TOKEN__", _auth_token)
                        file_path.unlink()
                        file_path.write_text(content, "utf-8", errors="surrogateescape")

                rec_before_cmd = sample.get("before_repo_set_cmd", "") or "true"
                result = await run_single_test(
                    sample,
                    agent,
                    config,
                    results_dir,
                    max_call_llm,
                    port,
                    image,
                    rec_before_cmd,
                    instance_repo_dir,
                    port_instr_dir,
                    instr_count,
                    total_funcs,
                    workdir=workdir,
                    patch_file=group_patch_file,
                    auth_token=_auth_token,
                    record_trace=record_trace,
                    instrumentation_enabled=instrumentation_enabled,
                    path_points_enabled=path_points_enabled,
                )
                status = result.get("status", "infra_error")
                with stats_lock:
                    stats[status] = stats.get(status, 0) + 1
            except Exception as error:
                logger.error(f"[{instance_id}] run_single_test crashed: {error}")
                task_dir = results_dir / instance_id
                result = {
                    "instance_id": instance_id,
                    "status": "infra_error",
                    "reason": f"run_single_test_crashed:{type(error).__name__}:{error}",
                }
                _ensure_result_files(task_dir, sample, result)
                with stats_lock:
                    stats["infra_error"] = stats.get("infra_error", 0) + 1
            finally:
                port_pool.put_nowait(port)
    except Exception as error:
        logger.error(f"{instance_label} failed: {error}")
        task_dir = results_dir / instance_id
        result = {
            "instance_id": instance_id,
            "status": "infra_error",
            "reason": f"instance_failed:{type(error).__name__}:{error}",
        }
        _ensure_result_files(task_dir, sample, result)
        with stats_lock:
            stats["infra_error"] = stats.get("infra_error", 0) + 1
    finally:
        if not os.environ.get("AGENTICHEAL_KEEP_GROUP_TMP"):
            try:
                shutil.rmtree(group_tmp_root)
            except FileNotFoundError as error:
                logger.warning(f"{instance_label} temporary repository was already removed: {error}")
            except Exception as error:
                logger.warning(f"{instance_label} failed to rmtree temporary repository: {error}")

    if cleanup_images:
        logger.info(f"{instance_label} cleaning up Docker image {image[:60]}...")
        await asyncio.to_thread(_remove_docker_image, image)




async def run_healbench(
    agent: Agent,
    *,
    config: Config,
    source: str | None = None,
    limit: int = 0,
    per_dataset_limit: int = 0,
    instance_ids: list[str] | None = None,
    max_call_llm: int = 30,
    concurrency: int = 20,
    cleanup_images: bool = False,
    output_dir: Path | None = None,
    record_trace: bool = False,
    port_base: int | None = None,
    instrumentation_enabled: bool = True,
    path_points_enabled: bool = False,
    rerun: bool = False,
) -> dict:
    _, all_records = load_dataset_records(source)

    results_dir = Path(output_dir or (
        HEALBENCH_RESULTS / type(agent).__name__ / config.openai_model
    )).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    init_logger(str(results_dir / "run.log"))

    logger.info(
        f"source={source or 'all'}, model={config.openai_model}, "
        f"agent={type(agent).__name__}, limit={limit}, concurrency={concurrency}, "
        f"cleanup_images={cleanup_images}"
    )

    excluded_count = sum(
        str(record.get("instance_id", "")) not in HEALBENCH_INSTANCE_IDS
        for record in all_records
    )
    records = [
        record
        for record in all_records
        if str(record.get("instance_id", "")) in HEALBENCH_INSTANCE_IDS
    ]
    logger.info(
        f"Excluded {excluded_count} records not present in the benchmark manifest"
    )

    if instance_ids:
        id_set = set(instance_ids)
        records = [r for r in records if r["instance_id"] in id_set]
        logger.info(f"instance-file filter: {len(id_set)} requested → {len(records)} matched")

    if per_dataset_limit:
        from collections import Counter

        ds_count: Counter = Counter()
        filtered = []
        for r in records:
            ds = r.get("source_dataset", "unknown")
            if ds_count[ds] < per_dataset_limit:
                filtered.append(r)
                ds_count[ds] += 1
        logger.info(f"per_dataset_limit={per_dataset_limit}: {dict(ds_count)} → {len(filtered)} total")
        records = filtered
    if limit:
        records = records[:limit]

    for i, r in enumerate(records[:2]):
        logger.info(f"  record[{i}]: {r['instance_id']} (tag={r.get('dockerhub_tag', '')[:40]})")

    def _is_done(iid: str) -> bool:
        if rerun:
            return False
        out_f = results_dir / iid / "output.json"
        if not out_f.exists():
            return False
        try:
            result = json.loads(out_f.read_text("utf-8"))
            if not isinstance(result, dict):
                raise ValueError("output.json must contain a JSON object")
            return True
        except Exception as error:
            logger.warning(f"Invalid existing result for {iid}; scheduling rerun: {error}")
            return False

    pending = [r for r in records if not _is_done(r["instance_id"])]
    logger.info(f"Will process {len(pending)} instances ({len(records) - len(pending)} already done)")

    if not pending:
        logger.info("Nothing to process.")
        from collections import Counter
        combined: Counter = Counter()
        for r in records:
            out_f = results_dir / r["instance_id"] / "output.json"
            if out_f.exists():
                try:
                    combined[json.loads(out_f.read_text("utf-8")).get("status", "unknown")] += 1
                except Exception as error:
                    logger.warning(f"Result summary parse failed for {r['instance_id']}: {error}")
        if combined:
            correct = combined.get("correct", 0)
            total = sum(combined.values())
            logger.info(f"Done! CORRECT={correct}/{total}. Breakdown: {json.dumps(dict(combined))}")
        return dict(combined)

    failed_ids = _ensure_docker_images(pending)
    if failed_ids:
        logger.warning(f"Skipping {len(failed_ids)} instances with missing Docker images: {list(failed_ids)[:5]}")
        pending = [r for r in pending if r["instance_id"] not in failed_ids]
    if not pending:
        logger.info("No instances with available Docker images.")
        return {}

    if rerun:
        backup_root = results_dir / (
            f"_rerun_bak_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
        )
        archived = []
        for record in pending:
            task_dir = results_dir / record["instance_id"]
            if not task_dir.exists():
                continue
            backup_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(task_dir), str(backup_root / task_dir.name))
            archived.append(record["instance_id"])
        if archived:
            logger.info(
                f"Archived {len(archived)} prior result directories before rerun to {backup_root}; "
                f"first 2: {archived[:2]}"
            )

    logger.info(
        f"Running {len(pending)} instances in batches of {concurrency} with one prefetched batch"
    )

    port_pool = build_port_pool(concurrency, port_base=port_base)
    if port_base is not None:
        logger.info(f"Using custom host-port base {port_base} (disjoint range for parallel runs)")
    sem = asyncio.Semaphore(concurrency)
    stats: dict[str, int] = {}
    stats_lock = threading.Lock()

    async def process_prefetched_instance(record: dict, start_event: asyncio.Event) -> None:
        await process_instance(
            record,
            start_event,
            agent,
            config,
            results_dir,
            max_call_llm,
            port_pool,
            sem,
            stats,
            stats_lock,
            cleanup_images=cleanup_images,
            record_trace=record_trace,
            instrumentation_enabled=instrumentation_enabled,
            path_points_enabled=path_points_enabled,
        )

    await run_batched_with_prefetch(pending, concurrency, process_prefetched_instance)

    batch_correct = stats.get("correct", 0)
    batch_total = sum(stats.values())
    logger.info(f"Batch done: CORRECT={batch_correct}/{batch_total}. Breakdown: {json.dumps(stats)}")

    from collections import Counter
    combined_stats: Counter = Counter()
    ds_breakdown: Counter = Counter()
    ds_correct: Counter = Counter()
    iid_to_ds = {r["instance_id"]: r.get("source_dataset", "unknown") for r in records}
    for r in records:
        iid = r["instance_id"]
        ds = iid_to_ds.get(iid, "unknown")
        out_file = results_dir / iid / "output.json"
        if out_file.exists():
            try:
                out_data = json.loads(out_file.read_text("utf-8"))
                s = out_data.get("status", "unknown")
                combined_stats[s] += 1
                ds_breakdown[ds] += 1
                if s == "correct":
                    ds_correct[ds] += 1
            except Exception as _e:
                logger.warning(f"dataset breakdown parse error for {iid}: {_e}")

    correct = combined_stats.get("correct", 0)
    total = sum(combined_stats.values())
    logger.info(f"Done! CORRECT={correct}/{total}. Breakdown: {json.dumps(dict(combined_stats))}")
    logger.info(f"Output: {results_dir}")

    if len(ds_breakdown) > 1:
        for ds in sorted(ds_breakdown):
            logger.info(
                f"  {ds}: CORRECT={ds_correct[ds]}/{ds_breakdown[ds]}"
            )

    return dict(combined_stats)
