from __future__ import annotations

import abc
import ast
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import jinja2
from loguru import logger

from prompts import HEAL_CRASH_INFO

EXECUTE_TOOL_NAME = "execute"
RERAISE_TOOL_NAME = "reraise"

EXECUTE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": EXECUTE_TOOL_NAME,
        "description": (
            "Execute Python code in the caught frame. Use submit=false to inspect or experiment with "
            "the live runtime state without committing changes; use submit=true to submit the final Heal "
            "and resume execution. Submitted code runs at column 0; use `_eh_rtn = expr` instead of `return expr`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code with access to live locals and globals."},
                "submit": {
                    "type": "boolean",
                    "description": "False to inspect the live runtime state; true to submit this code as the final Heal.",
                    "default": False,
                },
            },
            "required": ["code", "submit"],
            "additionalProperties": False,
            "$schema": "http://json-schema.org/draft-07/schema#",
        },
    },
}

RERAISE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": RERAISE_TOOL_NAME,
        "description": "Stop without applying a Heal and re-raise the original exception.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False, "$schema": "http://json-schema.org/draft-07/schema#"},
    },
}

BASH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a read-only shell command in the instance's temporary repository.\n\n"
            "Usage notes:\n"
            "- Use it for static exploration: read files, search code, and list paths.\n"
            "- It cannot access objects in the paused crash frame; use execute(code, submit=false) for live runtime state.\n"
            "- Repository modifications are rejected.\n\n"
            "<good-example>\n"
            "grep -rn \"def target_function\" package/\n"
            "</good-example>\n\n"
            "<bad-example>\n"
            "python -c \"print(value)\"  # shell commands cannot access crash-frame locals\n"
            "</bad-example>"
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to execute"}},
            "required": ["command"],
            "additionalProperties": False,
            "$schema": "http://json-schema.org/draft-07/schema#",
        },
    },
}

HEAL_TOOLS = [BASH_TOOL_SCHEMA, EXECUTE_TOOL_SCHEMA, RERAISE_TOOL_SCHEMA]


@dataclass
class ExecuteOutcome:
    code: str = ""
    submit: bool = False
    stdout: str = ""
    value_repr: str = ""
    error: str = ""

    def to_tool_content(self) -> str:
        parts = [
            "<execute_result>",
            f"<code>\n{self.code}\n</code>",
            f"<submit>{str(self.submit).lower()}</submit>",
        ]
        if self.error:
            parts.append(f"<error>\n{self.error}\n</error>")
        if self.stdout:
            parts.append(f"<stdout>\n{self.stdout}\n</stdout>")
        if self.value_repr:
            parts.append(f"<value>\n{self.value_repr}\n</value>")
        if not self.error and not self.stdout and not self.value_repr:
            parts.append("<stdout>(execute completed with no output)</stdout>")
        parts.append("</execute_result>")
        return "\n".join(parts)


@dataclass
class CrashInfo:
    file_path: str
    line: int
    block_index: int
    error_type: str
    error_message: str
    locals_snapshot: dict
    traceback: str
    code_at_crash: str
    code_display: str = ""
    call_chain_source: str = ""
    crash_num: int = 0


@dataclass
class HealSession:
    instance_id: str
    messages: list[dict] = field(default_factory=list)
    system_installed: bool = False
    total_llm_calls: int = 0
    max_call_llm: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    llm_call_records: list[dict] = field(default_factory=list)
    crash_count: int = 0
    execute_trial_count: int = 0
    bash_count: int = 0
    active_crash: CrashInfo | None = None
    repo_dir: str = ""
    task_dir: str = ""
    created_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    adapter_state: Any = None

    def note_crash(self, crash: CrashInfo) -> None:
        with self.lock:
            self.crash_count += 1
            crash.crash_num = self.crash_count
            self.active_crash = crash

    def note_llm_call(self, input_tokens: int = 0, output_tokens: int = 0) -> int:
        with self.lock:
            self.total_llm_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            return self.total_llm_calls

    def record_provider_call(
        self,
        raw_input: Any,
        raw_output: Any,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> int:
        with self.lock:
            self.total_llm_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.llm_call_records.append({
                "raw_input": raw_input,
                "raw_output": raw_output,
            })
            return self.total_llm_calls

    def note_execute_trial(self) -> None:
        with self.lock:
            self.execute_trial_count += 1

    def budget_exhausted(self) -> bool:
        with self.lock:
            return 0 < self.max_call_llm <= self.total_llm_calls

    def clear_active_crash(self) -> None:
        with self.lock:
            self.active_crash = None


class Agent(abc.ABC):
    @abc.abstractmethod
    def step(self, session: HealSession, repo_dir: str = "") -> dict:
        raise NotImplementedError

    def format_crash_message(self, crash: CrashInfo) -> str:
        return render_heal_crash_info(crash)

    def format_execute_result(self, outcome: ExecuteOutcome) -> str:
        return outcome.to_tool_content()

    def cleanup(self, session: HealSession) -> None:
        return None


class AgentMiddleware:
    def __init__(self, agent: Agent, session: HealSession):
        self.agent = agent
        self.session = session

    def accept_crash(self, crash: CrashInfo) -> None:
        self.session.note_crash(crash)
        self.session.messages.append({
            "role": "user",
            "content": self.agent.format_crash_message(crash),
        })

    def accept_execute_result(self, outcome: ExecuteOutcome) -> None:
        tool_content = self.agent.format_execute_result(outcome)
        tool_call_id = self._pending_tool_call_id()
        if tool_call_id:
            self.session.messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_content,
            })
        else:
            self.session.messages.append({"role": "user", "content": tool_content})

    def _pending_tool_call_id(self) -> str | None:
        answered_ids = {
            message.get("tool_call_id")
            for message in self.session.messages
            if message.get("role") == "tool"
        }
        for message in reversed(self.session.messages):
            if message.get("role") != "assistant":
                continue
            for tool_call in reversed(message.get("tool_calls") or []):
                tool_call_id = (
                    tool_call.get("id") if isinstance(tool_call, dict)
                    else getattr(tool_call, "id", None)
                )
                if tool_call_id and tool_call_id not in answered_ids:
                    return tool_call_id
        return None

    def budget_exhausted(self) -> bool:
        return self.session.budget_exhausted()

    def step(self, repo_dir: str = "") -> dict:
        if self.budget_exhausted():
            return {"action": "reraise", "reason": "max_call_llm"}
        decision = self.agent.step(self.session, repo_dir=repo_dir)
        if not isinstance(decision, dict):
            raise TypeError(f"Agent decision must be a dict, got {type(decision).__name__}")
        action = str(decision.get("action") or "reraise").strip()
        if action not in {"execute", "reraise"}:
            return {"action": "reraise", "reason": f"unknown_action:{action}"}
        if action == "execute":
            code = decision.get("code")
            if not isinstance(code, str) or not code.strip():
                return {"action": "reraise", "reason": "missing_execute_code"}
            if not isinstance(decision.get("submit"), bool):
                return {"action": "reraise", "reason": "invalid_execute_submit"}
        return {**decision, "action": action}


def format_locals(locals_snapshot: dict, max_per_var: int = 8000, max_total: int = 30000) -> str:
    if not locals_snapshot:
        return "(no locals captured)"
    lines = []
    total = 0
    for key, value in locals_snapshot.items():
        value_text = str(value)
        if len(value_text) > max_per_var:
            value_text = value_text[:max_per_var] + f"... [{len(value_text) - max_per_var} chars truncated]"
        line = f"  {key} = {value_text}"
        total += len(line)
        if total > max_total:
            remaining = len(locals_snapshot) - len(lines)
            if remaining > 0:
                lines.append(f"  ... [{remaining} more variables truncated, total exceeded {max_total} chars]")
                logger.warning(f"format_locals truncated: {len(lines)}/{len(locals_snapshot)} vars shown, total={total} chars")
            break
        lines.append(line)
    return "\n".join(lines)


def render_heal_crash_info(crash: CrashInfo) -> str:
    call_chain_source = crash.call_chain_source
    if call_chain_source == "(no additional source files in call chain)":
        call_chain_source = ""
    return jinja2.Template(HEAL_CRASH_INFO, trim_blocks=True, lstrip_blocks=True).render(
        traceback=crash.traceback or f"{crash.error_type}: {crash.error_message}",
        program_state=format_locals(crash.locals_snapshot),
        buggy_code=crash.code_display or f"<error>(location: {crash.file_path})</error>",
        call_chain_source=call_chain_source,
    ).strip()


def postprocess_fix_code(raw: str) -> str | None:
    import textwrap

    code = textwrap.dedent(raw.strip())
    if not code:
        return ""
    try:
        ast.parse(code)
    except SyntaxError:
        lines = code.splitlines()
        indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
        min_indent = min((i for i in indents if i > 0), default=0)
        if min_indent > 0:
            code = "\n".join(l[min_indent:] if len(l) - len(l.lstrip()) >= min_indent else l for l in lines)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    lines = code.splitlines()
    if tree is not None:
        module_return_lines: set[int] = set()

        def _collect_returns(stmts):
            for stmt in stmts:
                if isinstance(stmt, ast.Return):
                    module_return_lines.add(stmt.lineno)
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                else:
                    for attr in ("body", "orelse", "finalbody", "handlers"):
                        sub = getattr(stmt, attr, None)
                        if isinstance(sub, list):
                            for item in sub:
                                if isinstance(item, ast.AST) and hasattr(item, "body"):
                                    _collect_returns(item.body)
                                elif isinstance(item, ast.AST):
                                    _collect_returns([item])

        _collect_returns(tree.body)

        for lineno in module_return_lines:
            idx = lineno - 1
            if 0 <= idx < len(lines):
                m = re.match(r"^(\s*)return(\s+(.*))?$", lines[idx])
                if m:
                    value = m.group(3) if m.group(3) else "None"
                    lines[idx] = f"{m.group(1)}_eh_rtn = {value}"
    else:
        for idx, line in enumerate(lines):
            m = re.match(r"^return(\s+(.*))?$", line)
            if m:
                value = m.group(2) if m.group(2) else "None"
                lines[idx] = f"_eh_rtn = {value}"
    code = "\n".join(lines)
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    return code

