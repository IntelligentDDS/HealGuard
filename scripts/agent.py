import asyncio
import dataclasses
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_middleware import (
    EXECUTE_TOOL_NAME,
    EXECUTE_TOOL_SCHEMA,
    HEAL_TOOLS,
    RERAISE_TOOL_NAME,
    RERAISE_TOOL_SCHEMA,
    Agent,
    postprocess_fix_code,
)
from agenticheal import PROJECT_ROOT
from util import Config
from prompts import HEAL_RULES, HEALER_HEAL_SYSTEM_PROMPT, REACT_HEAL_SYSTEM_PROMPT
from util import call_llm

BASELINES_MSWE_SRC = PROJECT_ROOT / "baselines" / "mini-swe-agent" / "src"
if BASELINES_MSWE_SRC.exists():
    baseline_path = str(BASELINES_MSWE_SRC)
    if baseline_path not in sys.path:
        sys.path.insert(0, baseline_path)

try:
    from minisweagent.models.utils.cache_control import set_cache_control
    from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
    BASELINES_AVAILABLE = True
except Exception as error:
    logger.warning(f"baselines/mini-swe-agent import failed ({error}); using local fallbacks")
    set_cache_control = None
    _reorder_anthropic_thinking_blocks = None
    BASELINES_AVAILABLE = False

AGENT_TEMPERATURE = 0.0
DISK_WRITE_REJECT_MSG = """Local repository modification is not allowed; use read-only `bash`, `execute(code, submit=false/true)`, or `reraise()` instead."""
DISK_EDIT_TOOL_NAMES = {
    "react": frozenset(),
    "healer": frozenset(),
    "mini-swe-agent": frozenset(),
    "swe-agent": frozenset({"str_replace_editor"}),
    "openhands": frozenset({"file_editor"}),
    "claude-code": frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"}),
    "codex": frozenset(),
}
class DiskWriteGuard:
    WRITE_COMMANDS = frozenset({
        "tee", "cp", "mv", "rm", "install", "truncate", "dd", "patch", "shred", "rsync",
        "ln",
    })
    REDIRECT_RE = re.compile(r"(?<![\d<>])>>?\s*(?!=)(?!&)(?!/dev/)['\"/\w$~.]")
    SED_INPLACE_RE = re.compile(r"\bsed\b[^|;&]*\s-[a-zA-Z]*i")

    @staticmethod
    def strip_noncommand_text(command: str) -> str:
        stripped = command
        stripped = re.sub(r"(<<-?\s*['\"]?(\w+)['\"]?[^\n]*)\n.*?\n\2", r"\1", stripped, flags=re.DOTALL)
        stripped = re.sub(r"-c\s+'[^']*'", " -c ARG ", stripped)
        stripped = re.sub(r'-c\s+"[^"]*"', " -c ARG ", stripped)
        stripped = re.sub(r"'[^']*'", " '' ", stripped)
        return re.sub(r'"[^"]*"', ' "" ', stripped)

    @classmethod
    def is_write(cls, command: str) -> bool:
        if not command:
            return False
        scan = cls.strip_noncommand_text(command)
        if cls.SED_INPLACE_RE.search(scan) or cls.REDIRECT_RE.search(scan):
            return True
        for subcommand in re.split(r"&&|\|\||[|;&]", scan):
            tokens = subcommand.strip().split()
            if not tokens:
                continue
            index = 0
            while index < len(tokens) and re.fullmatch(r"\w+=.*", tokens[index]):
                index += 1
            head = tokens[index].rsplit("/", 1)[-1] if index < len(tokens) else ""
            if head in cls.WRITE_COMMANDS:
                return True
            if head == "sed" and re.search(r"-[a-zA-Z]*i\b", subcommand):
                return True
            if head == "git" and re.search(r"\b(apply|checkout|commit|reset|stash)\b", subcommand):
                return True
        return False

def load_agent_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))

def append_heal_rules(template: str) -> str:
    return template.rstrip() + "\n\n" + HEAL_RULES

BASELINE_LLM_COUNTER = threading.local()

class BudgetExhausted(Exception):
    pass

def set_baseline_llm_session(session: "HealSession | None") -> None:
    BASELINE_LLM_COUNTER.session = session

def note_baseline_llm_call() -> None:
    sess = getattr(BASELINE_LLM_COUNTER, "session", None)
    if sess is None:
        return
    if sess.max_call_llm and sess.total_llm_calls >= sess.max_call_llm:
        raise BudgetExhausted(f"""total_llm_calls={sess.total_llm_calls}/{sess.max_call_llm} at cap; unwinding baseline internal loop before next query""")
    sess.note_llm_call()

def tool_call_id(tool_call):
    return tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)

def prepare_tool_pairs(messages: list[dict]) -> list[dict]:
    if not isinstance(messages, list):
        return messages
    prepared = list(messages)

    last_res_idx: dict[str, int] = {}
    for i, msg in enumerate(prepared):
        if msg.get("role") == "tool":
            tid = msg.get("tool_call_id")
            if tid:
                last_res_idx[tid] = i
    dedup_res: list[dict] = []
    for i, msg in enumerate(prepared):
        if msg.get("role") == "tool":
            tid = msg.get("tool_call_id")
            if tid is not None and last_res_idx.get(tid) != i:
                logger.warning(f"dropped duplicate tool_result for tool_call_id={tid}")
                continue
        dedup_res.append(msg)
    prepared = dedup_res

    out: list[dict] = []
    for i, msg in enumerate(prepared):
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            next_msg = prepared[i + 1] if i + 1 < len(prepared) else None
            next_ids: set = set()
            if next_msg and next_msg.get("role") == "tool":
                j = i + 1
                while j < len(prepared) and prepared[j].get("role") == "tool":
                    tid_next = prepared[j].get("tool_call_id")
                    if tid_next:
                        next_ids.add(tid_next)
                    j += 1
            kept_tcs = []
            for tc in msg["tool_calls"]:
                tid = tool_call_id(tc)
                if tid is not None and tid in next_ids:
                    kept_tcs.append(tc)
                else:
                    logger.warning(f"dropped unpaired tool_use {tid} from assistant#{i}")
            new_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            if kept_tcs:
                new_msg["tool_calls"] = kept_tcs
            if not kept_tcs and not (new_msg.get("content") or "").strip():
                logger.warning(f"dropped empty assistant#{i} after tool_use prune")
                continue
            out.append(new_msg)
        elif role == "tool":
            tid = msg.get("tool_call_id")
            k = len(out) - 1
            while k >= 0 and out[k].get("role") == "tool":
                k -= 1
            prev = out[k] if k >= 0 else None
            prev_use_ids = set()
            if prev and prev.get("role") == "assistant":
                for tc in (prev.get("tool_calls") or []):
                    pid = tool_call_id(tc)
                    if pid:
                        prev_use_ids.add(pid)
            if tid is None or tid not in prev_use_ids:
                logger.warning(f"dropped orphan tool_result {tid} (no matching tool_use in prev assistant)")
                continue
            out.append(msg)
        else:
            out.append(msg)
    prepared = out

    if BASELINES_AVAILABLE:
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        prepared = set_cache_control(prepared, mode="default_end")
    return prepared

class ReAct(Agent):

    class ParseError(Exception):
        pass

    RE_THOUGHT = re.compile(r"(?is)^\s*Thought\s*:\s*(.+?)(?=^\s*Action\s*:)", re.MULTILINE)
    RE_ACTION = re.compile(r"(?is)^\s*Action\s*:\s*(bash|execute|reraise)\b", re.MULTILINE)
    RE_ACTION_INPUT_FENCE = re.compile(
        r"(?is)Action\s+Input\s*:\s*```[a-zA-Z0-9_+-]*\s*\n?(.*?)\n?```",
    )
    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def run_bash(repo_dir: str, command: str) -> dict:
        if DiskWriteGuard.is_write(command):
            logger.warning(f"blocked read-only bash disk-write command: {command}")
            return {
                "stdout": "",
                "stderr": DISK_WRITE_REJECT_MSG,
                "returncode": -1,
                "exception_info": DISK_WRITE_REJECT_MSG,
            }
        try:
            process = subprocess.run(
                command,
                shell=True,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            return {
                "stdout": process.stdout or "",
                "stderr": process.stderr or "",
                "returncode": process.returncode,
                "exception_info": None,
            }
        except subprocess.TimeoutExpired as error:
            return {
                "stdout": error.stdout or "",
                "stderr": error.stderr or "",
                "returncode": -1,
                "exception_info": "Command timed out after 120s",
            }
        except Exception as error:
            logger.warning(f"read-only bash failed: {error}")
            return {
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "exception_info": str(error),
            }

    @classmethod
    def parse_output(cls, content: str) -> dict:
        if not content or not content.strip():
            raise cls.ParseError("Empty response. Emit Thought/Action/Action Input.")

        action_match = cls.RE_ACTION.search(content)
        if not action_match:
            raise cls.ParseError(
                "Missing 'Action:' line. Response must contain Thought, Action, and Action Input."
            )
        action = action_match.group(1).strip().lower()
        thought_match = cls.RE_THOUGHT.search(content)
        if not thought_match:
            raise cls.ParseError(
                "Missing 'Thought:' line. Response must contain Thought followed by exactly one Action."
            )
        thought = thought_match.group(1).strip()
        if action == RERAISE_TOOL_NAME:
            return {"thought": thought, "name": RERAISE_TOOL_NAME, "args": {}}

        fence_match = cls.RE_ACTION_INPUT_FENCE.search(content)
        if not fence_match:
            raise cls.ParseError(
                "Missing 'Action Input:' fenced code block."
            )
        payload = fence_match.group(1).strip()
        if not payload:
            raise cls.ParseError("Action Input payload is empty.")
        if action == "bash":
            return {"thought": thought, "name": "bash", "args": {"command": payload}}

        submit_match = re.match(r"(?is)^submit\s*=\s*(true|false)\s*\n(.*)$", payload)
        if not submit_match:
            raise cls.ParseError("execute Action Input must start with submit=true or submit=false.")
        return {
            "thought": thought,
            "name": EXECUTE_TOOL_NAME,
            "args": {
                "code": submit_match.group(2).strip(),
                "submit": submit_match.group(1).lower() == "true",
            },
        }

    @staticmethod
    def render_observation(output: dict) -> str:
        observation = {
            "stdout": output.get("stdout", ""),
            "stderr": output.get("stderr", ""),
            "returncode": output.get("returncode"),
        }
        if output.get("exception_info"):
            observation["exception_info"] = output["exception_info"]
        body = json.dumps(observation, ensure_ascii=False)
        return f"""Observation:
{body}

Thought:"""

    @staticmethod
    def render_format_error(error: str | Exception) -> str:
        return f"""Observation:
[FORMAT ERROR] {error}
Retry with Thought and exactly one Action: execute, bash, or reraise. For execute or bash, include a fenced Action Input; execute must start with submit=false or submit=true. Omit Action Input for reraise.

Thought:"""

    def step(self, session: "HealSession", repo_dir: str = "") -> dict:
        if not session.system_installed:
            session.messages.insert(0, {"role": "system", "content": REACT_HEAL_SYSTEM_PROMPT})
            session.system_installed = True

        while True:
            if session.budget_exhausted():
                return {"action": "reraise", "reason": "max_call_llm"}

            api_messages = prepare_tool_pairs(session.messages)
            try:
                result = call_llm(
                    api_messages,
                    self.config,
                    temperature=AGENT_TEMPERATURE,
                    timeout=120,
                    drop_params=True,
                    stop=["\nObservation:"],
                )
            except Exception as e:
                err_name = type(e).__name__
                logger.error(f"""[{session.instance_id}] ReAct.step: LLM call failed ({err_name}): {e}""")
                return {"action": "reraise", "reason": f"llm_error:{err_name}"}

            session.record_provider_call(
                raw_input=api_messages,
                raw_output=result["raw"],
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
            )

            raw = result.get("raw") or {}
            usage = raw.get("usage") or {}
            finish_reason = (raw.get("choices") or [{}])[0].get("finish_reason", "") if raw.get("choices") else ""
            if finish_reason == "length" and usage.get("completion_tokens", 999) <= 1:
                logger.warning(f"""[{session.instance_id}] CONTEXT OVERFLOW: prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')}, finish=length — skipping (context window full)""")
                return {"action": "reraise", "reason": "context_overflow"}

            content = result["content"] or ""
            session.messages.append({"role": "assistant", "content": content})

            try:
                parsed = self.parse_output(content)
            except self.ParseError as error:
                session.messages.append({"role": "user", "content": self.render_format_error(error)})
                continue

            name = parsed["name"]
            args = parsed["args"]

            if name == RERAISE_TOOL_NAME:
                return {"action": "reraise", "reason": "agent_reraise_tool"}

            if name == EXECUTE_TOOL_NAME:
                code = args.get("code", "")
                if not args.get("submit", False):
                    return {"action": "execute", "code": code, "submit": False}
                processed = postprocess_fix_code(code)
                if processed is None:
                    logger.warning(f"""[{session.instance_id}] execute submission rejected (syntax_error_after_postprocess) — extracted code:
{code}
--- raw LLM output ---
{content}""")
                    return {"action": "reraise", "reason": "syntax_error_after_postprocess"}
                return {"action": "execute", "code": processed, "submit": True}

            if name == "bash":
                session.bash_count += 1
                raw_out = self.run_bash(repo_dir, args.get("command", ""))
                session.messages.append({
                    "role": "user",
                    "content": self.render_observation(raw_out),
                })
                continue

    def format_execute_result(self, outcome) -> str:
        body = outcome.to_tool_content()
        return f"""Observation:
{body}

Thought:"""

class Healer(Agent):

    EXECUTE_TOOL = json.loads(json.dumps(next(
        tool for tool in HEAL_TOOLS
        if tool["function"]["name"] == EXECUTE_TOOL_NAME
    )))
    EXECUTE_TOOL["function"]["parameters"]["properties"]["submit"].update(
        {"enum": [True], "default": True}
    )

    def __init__(self, config: Config):
        self.config = config

    def step(self, session: "HealSession", repo_dir: str = "") -> dict:
        crash = session.active_crash
        if crash is None:
            return {"action": "reraise", "reason": "no_crash"}
        if session.budget_exhausted():
            return {"action": "reraise", "reason": "max_call_llm"}

        if not session.system_installed:
            session.messages.insert(0, {"role": "system", "content": HEALER_HEAL_SYSTEM_PROMPT})
            session.system_installed = True
        messages = prepare_tool_pairs(session.messages)
        try:
            result = call_llm(
                messages, self.config,
                temperature=AGENT_TEMPERATURE,
                timeout=120,
                drop_params=True,
                tools=[self.EXECUTE_TOOL],
                tool_choice={"type": "function", "function": {"name": EXECUTE_TOOL_NAME}},
            )
        except Exception as e:
            err_name = type(e).__name__
            logger.error(f"[{session.instance_id}] Healer LLM call failed ({err_name}): {e}")
            return {"action": "reraise", "reason": f"llm_error:{err_name}"}

        session.record_provider_call(
            raw_input=messages,
            raw_output=result["raw"],
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
        )

        raw = result.get("raw") or {}
        choices = raw.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        assistant_message = {
            key: value
            for key, value in message.items()
            if key in ("role", "content", "tool_calls") and value is not None
        }
        if assistant_message:
            assistant_message.setdefault("content", "")
            session.messages.append(assistant_message)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            logger.warning(f"[{session.instance_id}] Healer returned no tool call")
            return {"action": "reraise", "reason": "no_tool_call"}
        function = (tool_calls[0] or {}).get("function") or {}
        name = function.get("name", "")
        try:
            args = json.loads(function.get("arguments") or "{}")
        except Exception as e:
            logger.warning(f"[{session.instance_id}] Healer tool arguments invalid: {e}")
            return {"action": "reraise", "reason": "invalid_tool_arguments"}
        if name != EXECUTE_TOOL_NAME:
            return {"action": "reraise", "reason": f"unknown_tool:{name}"}
        code = args.get("code", "") if isinstance(args, dict) else ""
        if not bool(args.get("submit", False)):
            logger.warning(f"[{session.instance_id}] Healer returned submit=false")
            return {"action": "reraise", "reason": "invalid_submit_false"}
        processed = postprocess_fix_code(code)
        if processed is None:
            logger.warning(f"""[{session.instance_id}] Healer execute submission rejected (syntax_error_after_postprocess) — extracted code:
{code}""")
            return {"action": "reraise", "reason": "syntax_error_after_postprocess"}
        tool_call_id = (tool_calls[0] or {}).get("id")
        if tool_call_id:
            session.messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "Final Heal submitted; the paused process resumed.",
            })
        return {"action": "execute", "code": processed, "submit": True}

class MiniSweAgentAdapter(Agent):

    def __init__(self, model_name: str, model_kwargs: dict | None = None,
                 cost_tracking: str = "default"):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.cost_tracking = cost_tracking

    @staticmethod
    def configure_for_heal():
        import minisweagent.models.litellm_model as lm_mod
        import minisweagent.models.utils.actions_toolcall as act
        from minisweagent.exceptions import Submitted

        if hasattr(act, "_heal_patched"):
            return
        act._heal_patched = True

        tools = [act.BASH_TOOL, EXECUTE_TOOL_SCHEMA, RERAISE_TOOL_SCHEMA]

        import minisweagent.environments.local as local_mod
        if not getattr(local_mod, "_heal_disk_guard_patched", False):
            original_execute = local_mod.LocalEnvironment.execute

            def guarded_execute(self, action, cwd: str = "", *, timeout=None):
                command = action.get("command", "") if isinstance(action, dict) else ""
                if DiskWriteGuard.is_write(command):
                    logger.warning(f"[mini-swe] blocked disk-write command: {command}")
                    return {
                        "output": DISK_WRITE_REJECT_MSG,
                        "returncode": -1,
                        "exception_info": DISK_WRITE_REJECT_MSG,
                    }
                return original_execute(self, action, cwd, timeout=timeout)

            local_mod.LocalEnvironment.execute = guarded_execute
            local_mod._heal_disk_guard_patched = True
            logger.info("patched LocalEnvironment.execute with disk-write guardrail")

        def heal_query(self, messages, **kwargs):
            kwargs.setdefault("tools", tools)
            kwargs.setdefault("temperature", AGENT_TEMPERATURE)
            kwargs.setdefault("drop_params", True)
            note_baseline_llm_call()
            try:
                messages = prepare_tool_pairs(messages)
            except Exception as e:
                logger.warning(f"mini-swe prepare_tool_pairs failed, sending messages unmodified: {e}")
            import litellm
            merged_kwargs = self.config.model_kwargs | kwargs
            model_lower = (self.config.model_name or "").lower()
            if "gpt-5" in model_lower and merged_kwargs.get("tools"):
                merged_kwargs.pop("reasoning_effort", None)
                eb = merged_kwargs.get("extra_body")
                if isinstance(eb, dict) and "reasoning_effort" in eb:
                    eb = {k: v for k, v in eb.items() if k != "reasoning_effort"}
                    if eb:
                        merged_kwargs["extra_body"] = eb
                    else:
                        merged_kwargs.pop("extra_body", None)
            resp = litellm.completion(
                model=self.config.model_name,
                messages=messages,
                **merged_kwargs,
            )
            sess = getattr(BASELINE_LLM_COUNTER, "session", None)
            if sess is not None:
                sess.llm_call_records.append({
                    "raw_input": messages,
                    "raw_output": resp.model_dump(),
                })
            return resp

        lm_mod.LitellmModel._query = heal_query

        abort_exceptions = list(lm_mod.LitellmModel.abort_exceptions)
        if BudgetExhausted not in abort_exceptions:
            abort_exceptions.append(BudgetExhausted)
            lm_mod.LitellmModel.abort_exceptions = abort_exceptions

        original_parse_actions = lm_mod.LitellmModel._parse_actions

        def heal_parse_actions(self, response):
            tool_calls = response.choices[0].message.tool_calls or []
            for tc in (tool_calls or []):
                name = tc.function.name
                if name not in (RERAISE_TOOL_NAME, EXECUTE_TOOL_NAME):
                    continue
                args: dict = {}
                try:
                    args = json.loads(tc.function.arguments)
                except Exception as e:
                    logger.warning(f"failed to parse tool_call arguments: {e}")
                if name == RERAISE_TOOL_NAME:
                    raise Submitted({
                        "role": "exit",
                        "content": "reraise",
                        "extra": {
                            "exit_status": "Reraise",
                            "action_kind": "reraise",
                        },
                    })
                if name == EXECUTE_TOOL_NAME:
                    code = args.get("code", "") if isinstance(args, dict) else ""
                    submit = bool(args.get("submit", False)) if isinstance(args, dict) else False
                    raise Submitted({
                        "role": "exit",
                        "content": code,
                        "extra": {
                            "exit_status": "Submitted" if submit else "Execute",
                            "execute_code": code,
                            "submit": submit,
                            "action_kind": "execute",
                        },
                    })
            return original_parse_actions(self, response)

        lm_mod.LitellmModel._parse_actions = heal_parse_actions

    def step(self, session: "HealSession", repo_dir: str = "") -> dict:
        self.configure_for_heal()
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_model import LitellmModel
        from minisweagent.exceptions import LimitsExceeded, FormatError as MSWEFormatError

        state = session.adapter_state
        is_first_crash = state is None

        if is_first_crash:
            heal_config = load_agent_config(
                PROJECT_ROOT / "scripts" / "config_minisweagent_heal_default.yaml"
            )
            model = LitellmModel(
                model_name=self.model_name,
                model_kwargs=self.model_kwargs,
                cost_tracking=self.cost_tracking,
                observation_template=heal_config["model"]["observation_template"],
                format_error_template=heal_config["model"]["format_error_template"],
            )
            env = LocalEnvironment(
                cwd=repo_dir or ".",
                env=dict(heal_config["environment"]["env"]),
            )
            agent = DefaultAgent(
                model, env,
                system_template=append_heal_rules(heal_config["agent"]["system_template"]),
                instance_template=heal_config["agent"]["instance_template"],
                step_limit=heal_config["agent"]["step_limit"],
                cost_limit=heal_config["agent"]["cost_limit"],
            )
            state = {"agent": agent}
            session.adapter_state = state
        agent = state["agent"]

        task_body = latest_crash_body(session)
        if not task_body:
            return {"action": "reraise", "reason": "no_crash_message"}

        t0 = time.time()
        set_baseline_llm_session(session)
        try:
            if is_first_crash:
                extra = agent.run(task=task_body)
            else:
                if agent.messages and agent.messages[-1].get("role") == "exit":
                    agent.messages.pop()
                agent.add_messages(agent.model.format_message(role="user", content=task_body))
                extra = self.step_until_submitted(agent)
        except BudgetExhausted:
            self.sync_messages_back(session, agent)
            logger.info(f"[{session.instance_id}] mini-swe → max_call_llm reached mid-step")
            return {"action": "reraise", "reason": "max_call_llm"}
        except LimitsExceeded:
            self.sync_messages_back(session, agent)
            return {"action": "reraise", "reason": "limits_exceeded"}
        except MSWEFormatError as e:
            self.sync_messages_back(session, agent)
            logger.warning(f"[{session.instance_id}] mini-swe format error: {e}")
            return {"action": "reraise", "reason": "format_error"}
        except Exception as e:
            self.sync_messages_back(session, agent)
            logger.error(f"[{session.instance_id}] mini-swe step error ({type(e).__name__}): {e}")
            return {"action": "reraise", "reason": f"backend_error:{type(e).__name__}"}
        finally:
            set_baseline_llm_session(None)

        self.sync_messages_back(session, agent)
        elapsed = round(time.time() - t0, 2)
        kind = extra.get("action_kind", "reraise")
        if kind == "reraise":
            return {"action": "reraise", "reason": "agent_reraise_tool"}
        if kind == "execute":
            code = extra.get("execute_code", "")
            if not extra.get("submit", False):
                logger.info(f"[{session.instance_id}] mini-swe → execute(submit=false) ({elapsed}s)")
                return {"action": "execute", "code": code, "submit": False}
            processed = postprocess_fix_code(code)
            if processed is None:
                logger.warning(f"""[{session.instance_id}] mini-swe execute submission rejected (syntax_error_after_postprocess) — submission:
{code}""")
                return {"action": "reraise", "reason": "syntax_error_after_postprocess"}
            logger.info(f"[{session.instance_id}] mini-swe → execute(submit=true) ({elapsed}s)")
            return {"action": "execute", "code": processed, "submit": True}
        return {"action": "reraise", "reason": f"unknown_action_kind:{kind}"}

    @staticmethod
    def step_until_submitted(agent) -> dict:
        from minisweagent.exceptions import InterruptAgentFlow, Submitted
        while True:
            try:
                agent.step()
            except Submitted as e:
                try:
                    agent.add_messages(*e.messages)
                except Exception as error:
                    logger.warning(f"add_messages failed: {error}")
                return e.messages[0].get("extra", {}) if e.messages else {}
            except InterruptAgentFlow as e:
                try:
                    agent.add_messages(*e.messages)
                except Exception as error:
                    logger.warning(f"add_messages failed: {error}")
            if agent.messages and agent.messages[-1].get("role") == "exit":
                return agent.messages[-1].get("extra", {})

    @staticmethod
    def sync_messages_back(session, agent):
        agent_msgs = list(agent.messages)
        sess_msgs = list(session.messages)
        n_agent = len(agent_msgs)
        trailing = sess_msgs[n_agent:] if len(sess_msgs) > n_agent else []
        session.messages[:] = agent_msgs + trailing
        session.system_installed = True

class SWEAgentAdapter(Agent):

    paths = threading.local()

    class HealToolRequested(KeyboardInterrupt):
        def __init__(self, name: str, tool_args: dict):
            super().__init__(f"{name} requested")
            self.name = name
            self.tool_args = tool_args

    @classmethod
    def get_tmp_tools(cls) -> str:
        return getattr(cls.paths, "tmp_tools", "/tmp/swe_tools")

    @classmethod
    def get_tmp_patch(cls) -> str:
        return getattr(cls.paths, "tmp_patch", "/tmp/model.patch")

    @classmethod
    def set_paths(cls, instance_id: str | None, task_dir: str = "") -> None:
        if instance_id is None:
            cls.paths.tmp_tools = "/tmp/swe_tools"
            cls.paths.tmp_patch = "/tmp/model.patch"
            return
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", instance_id)[:64] or "default"
        if task_dir:
            base = Path(task_dir) / ".tmp"
            base.mkdir(parents=True, exist_ok=True)
            cls.paths.tmp_tools = str(base / "swe_tools")
            cls.paths.tmp_patch = str(base / "model.patch")
        else:
            cls.paths.tmp_tools = f"/tmp/swe_tools_{safe}"
            cls.paths.tmp_patch = f"/tmp/model_{safe}.patch"

    @staticmethod
    def configure_for_heal():
        import sweagent.agent.agents as agents_mod
        import sweagent.tools.tools as tools_mod

        try:
            import swerex.runtime.local as swerex_local
            if not getattr(swerex_local, "_heal_exitcode_timeout_patched", False):
                original_run_normal = swerex_local.BashSession._run_normal
                EXITCODE_TIMEOUT = 30

                async def heal_run_normal(self, action):
                    original_expect = self.shell.expect

                    def wrapped_expect(pattern, timeout=-1, **kw):
                        if timeout == 1 and isinstance(pattern, (str, bytes)) \
                                and "EXITCODE" in str(pattern):
                            timeout = EXITCODE_TIMEOUT
                        return original_expect(pattern, timeout=timeout, **kw)

                    self.shell.expect = wrapped_expect
                    try:
                        return await original_run_normal(self, action)
                    finally:
                        self.shell.expect = original_expect

                swerex_local.BashSession._run_normal = heal_run_normal
                swerex_local._heal_exitcode_timeout_patched = True
                logger.info(f"swerex exit-code read timeout widened "
                            f"1s → {EXITCODE_TIMEOUT}s (loaded-host robustness)")
        except ImportError as e:
            logger.warning(f"swerex unavailable, skipping exit-code patch ({e})")
        except Exception as e:
            logger.warning(f"swerex exit-code patch failed: {e}")

        if getattr(tools_mod, "_heal_patched", False):
            return
        tools_mod._heal_patched = True

        def heal_reset(self, env):
            tmp_tools = SWEAgentAdapter.get_tmp_tools()
            env_variables = self.config.env_variables.copy() | {
                var: os.getenv(var) for var in self.config.propagate_env_variables
            }
            env.set_env_variables(env_variables)
            env.write_file(f"{tmp_tools}/.swe-agent-env", json.dumps(self.config.registry_variables))
            env.write_file(f"{tmp_tools}/state.json", "{}")
            env.communicate(" && ".join(self._reset_commands), check="raise", timeout=self.config.install_timeout)

        tools_mod.ToolHandler.reset = heal_reset

        if not getattr(tools_mod.ToolHandler, "_heal_disk_guard_patched", False):
            tools_mod.ToolHandler._heal_disk_guard_patched = True
            original_should_block = tools_mod.ToolHandler.should_block_action

            def guarded_should_block(self, action: str) -> bool:
                stripped = (action or "").strip()
                tokens = stripped.split()
                head = tokens[0] if tokens else ""
                if original_should_block(self, action):
                    return True
                if head == "submit":
                    logger.warning(f"[swe-agent] blocked disk-diff submit command: {stripped}")
                    return True
                if head in DISK_EDIT_TOOL_NAMES["swe-agent"]:
                    logger.warning(f"[swe-agent] blocked disk-edit tool: {stripped}")
                    return True
                if DiskWriteGuard.is_write(stripped):
                    logger.warning(f"[swe-agent] blocked disk-write bash command: {stripped}")
                    return True
                return False

            tools_mod.ToolHandler.should_block_action = guarded_should_block
            logger.info("patched SWE-agent ToolHandler.should_block_action with disk-write guardrail")

        from swerex.runtime.abstract import UploadRequest

        async def heal_upload(self, env):
            tmp_tools = SWEAgentAdapter.get_tmp_tools()
            for bundle in self.config.bundles:
                target = Path(f"{tmp_tools}/{bundle.path.name}")
                if target.exists():
                    try:
                        shutil.rmtree(target, ignore_errors=True)
                    except Exception as e:
                        logger.warning(f"failed to pre-clean {target}: {e}")
            await asyncio.gather(
                *(
                    env.deployment.runtime.upload(
                        UploadRequest(
                            source_path=bundle.path.as_posix(),
                            target_path=f"{tmp_tools}/{bundle.path.name}",
                        )
                    )
                    for bundle in self.config.bundles
                )
            )

        tools_mod.ToolHandler._upload_bundles = heal_upload

        def heal_install(self, env):
            tmp_tools = SWEAgentAdapter.get_tmp_tools()
            tmp_patch = SWEAgentAdapter.get_tmp_patch()
            env.set_env_variables(self.config.env_variables)
            env.communicate("pwd", check="raise")
            asyncio.run(self._upload_bundles(env))
            for bundle in self.config.bundles:
                cmds = []
                if (bundle.path / "bin").exists():
                    cmds.extend([
                        f"export PATH={tmp_tools}/{bundle.path.name}/bin:$PATH",
                        f"chmod +x {tmp_tools}/{bundle.path.name}/bin/*",
                        f"sed -i 's|/root/model.patch|{tmp_patch}|g' {tmp_tools}/{bundle.path.name}/bin/* 2>/dev/null || true",
                        f"sed -i 's|/root/probe.py|{tmp_patch}.probe.py|g' {tmp_tools}/{bundle.path.name}/bin/* 2>/dev/null || true",
                    ])
                if (bundle.path / "install.sh").exists():
                    cmds.append(f"cd {tmp_tools}/{bundle.path.name} && source install.sh")
                if cmds:
                    env.communicate(
                        " && ".join(cmds),
                        check="raise",
                        timeout=self.config.install_timeout,
                    )

        tools_mod.ToolHandler._install_commands = heal_install

        def heal_get_state(self, env):
            tmp_tools = SWEAgentAdapter.get_tmp_tools()
            try:
                state_str = env.read_file(f"{tmp_tools}/state.json")
            except FileNotFoundError:
                return {}
            try:
                return json.loads(state_str)
            except json.JSONDecodeError:
                return {}

        tools_mod.ToolHandler._get_state = heal_get_state

        def heal_handle_submission(self, step, *, observation="", force_submission=False):
            tmp_patch = SWEAgentAdapter.get_tmp_patch()
            step = step.model_copy(deep=True)
            is_submission = self.tools.check_for_submission_cmd(observation or step.observation)
            if is_submission or force_submission:
                try:
                    submission = self._env.read_file(tmp_patch, encoding="utf-8", errors="backslashreplace")
                except FileNotFoundError:
                    return step
                except Exception as e:
                    logger.warning(f"failed to read patch file {tmp_patch}: {e}")
                    return step
                if submission.strip() != "":
                    step.submission = submission
                else:
                    step.submission = None
                step.observation = submission
                if not step.exit_status:
                    step.exit_status = "submitted"
                elif step.submission:
                    step.exit_status = f"submitted ({step.exit_status})"
                step.done = True
            return step

        agents_mod.DefaultAgent.handle_submission = heal_handle_submission

        def heal_autosubmit(self, step):
            import logging
            stdlib_logger = logging.getLogger("swea.autosubmit.patch")
            stdlib_logger.warning("Attempting autosubmission after error (Heal integration)")
            step = step.model_copy(deep=True)
            step.done = True
            step = self.handle_submission(step, observation="", force_submission=True)
            if step.submission:
                step.observation = "Exited (autosubmitted)"
            return step

        agents_mod.DefaultAgent.attempt_autosubmission_after_error = heal_autosubmit

    @staticmethod
    def configure_heal_actions():
        import sweagent.tools.tools as tools_mod

        if getattr(tools_mod.ToolHandler, "_heal_execute_patched", False):
            return
        tools_mod.ToolHandler._heal_execute_patched = True

        original_parse_actions = tools_mod.ToolHandler.parse_actions

        def heal_parse_actions(self, output: dict):
            tool_calls = output.get("tool_calls") or []
            if tool_calls:
                tc = tool_calls[0]
                if not isinstance(tc, dict):
                    tc = tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
                fn = tc.get("function", {})
                if not isinstance(fn, dict):
                    fn = fn.model_dump() if hasattr(fn, "model_dump") else dict(fn)
                name = fn.get("name")
                if name in (EXECUTE_TOOL_NAME, RERAISE_TOOL_NAME):
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception as e:
                        logger.warning(f"{name} argument parse failed: {e}")
                        args = {}
                    raise SWEAgentAdapter.HealToolRequested(
                        name,
                        args if isinstance(args, dict) else {},
                    )
            return original_parse_actions(self, output)

        tools_mod.ToolHandler.parse_actions = heal_parse_actions

    @staticmethod
    def configure_llm():
        import sweagent.agent.models as models_mod
        try:
            import litellm
            litellm.drop_params = True
            if not getattr(litellm, "_heal_strip_params_patched", False):
                original_completion = litellm.completion
                STRIPPED_KEYS = ("temperature", "top_p", "top_k")

                def completion_without_params(*a, **kw):
                    for k in STRIPPED_KEYS:
                        if k in kw:
                            kw.pop(k, None)
                    model = kw.get("model", "") or (a[0] if a else "")
                    if "doubao-seed" in str(model):
                        extra = kw.pop("extra_body", {}) or {}
                        extra.setdefault("reasoning_effort", "minimal")
                        kw["extra_body"] = extra
                    elif "gpt-5" in str(model):
                        kw.setdefault("reasoning_effort", "none")
                    msgs = kw.get("messages") or (a[0] if a else None)
                    if isinstance(msgs, list):
                        for m in msgs:
                            if isinstance(m, dict) and m.get("content") is None:
                                m["content"] = ""
                    return original_completion(*a, **kw)

                litellm.completion = completion_without_params
                litellm._heal_strip_params_patched = True
                logger.info(f"patched litellm.completion to strip {list(STRIPPED_KEYS)} "
                            f"AND normalize message content=None → '' "
                            f"(Bedrock claude rejects 'temperature is deprecated' and "
                            f"ValidationException 'missing field content')")

            if not getattr(litellm, "_heal_cost_map_patched", False):
                try:
                    from litellm import cost_calculator
                    if hasattr(cost_calculator, "completion_cost"):
                        original_completion_cost = cost_calculator.completion_cost
                        def safe_completion_cost(*a, **kw):
                            try:
                                return original_completion_cost(*a, **kw)
                            except Exception as e:
                                logger.warning(f"completion_cost suppressed: {e}")
                                return 0.0
                        cost_calculator.completion_cost = safe_completion_cost
                    if hasattr(litellm, "completion_cost"):
                        litellm.completion_cost = safe_completion_cost
                    litellm._heal_cost_map_patched = True
                    logger.info("wrapped litellm.cost_calculator.completion_cost to return $0 on unknown-model lookup failures")
                except Exception as e:
                    logger.warning(f"D19 patch failed: {e}")
        except Exception as e:
            logger.warning(f"could not set litellm.drop_params: {e}")

        if getattr(models_mod.LiteLLMModel, "_heal_count_patched", False):
            return
        models_mod.LiteLLMModel._heal_count_patched = True

        original_query = models_mod.LiteLLMModel.query

        def counted_query(self, *args, **kwargs):
            note_baseline_llm_call()
            kwargs["temperature"] = AGENT_TEMPERATURE
            resp = original_query(self, *args, **kwargs)
            sess = getattr(BASELINE_LLM_COUNTER, "session", None)
            if sess is not None:
                messages = args[0] if args else kwargs.get("messages", [])
                raw_out = resp.model_dump() if hasattr(resp, "model_dump") else resp
                sess.llm_call_records.append({"raw_input": messages, "raw_output": raw_out})
            return resp

        models_mod.LiteLLMModel.query = counted_query

        if not getattr(models_mod.LiteLLMModel, "_heal_pairing_patched", False):
            models_mod.LiteLLMModel._heal_pairing_patched = True
            original_single_query = models_mod.LiteLLMModel._single_query

            def paired_single_query(self, messages, *args, **kwargs):
                try:
                    messages = prepare_tool_pairs(messages)
                except Exception as e:
                    logger.warning(f"prepare_tool_pairs failed, sending messages unmodified: {e}")
                return original_single_query(self, messages, *args, **kwargs)

            models_mod.LiteLLMModel._single_query = paired_single_query

    def __init__(self, model_name: str, model_kwargs: dict | None = None):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}

    def step(self, session: "HealSession", repo_dir: str = "") -> dict:
        self.configure_for_heal()
        self.configure_heal_actions()
        self.configure_llm()

        from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
        from sweagent.agent.problem_statement import TextProblemStatement
        from sweagent.environment.swe_env import SWEEnv, EnvironmentConfig
        from sweagent.agent.models import GenericAPIModelConfig, RetryConfig
        from sweagent.tools.tools import ToolConfig, ToolFilterConfig
        from sweagent.agent.history_processors import CacheControlHistoryProcessor
        from sweagent.tools.bundle import Bundle
        from sweagent.tools.parsing import FunctionCallingParser
        from sweagent.agent.agents import TemplateConfig
        from swerex.deployment.config import LocalDeploymentConfig

        crash_body = latest_crash_body(session)
        if not crash_body:
            return {"action": "reraise", "reason": "no_crash_message"}

        state = session.adapter_state
        is_first_crash = state is None

        t0 = time.time()

        set_baseline_llm_session(session)
        self.set_paths(session.instance_id, getattr(session, "task_dir", "") or "")
        try:
            if is_first_crash:
                state = {"env": None, "agent": None, "output_dir": None}
                session.adapter_state = state
                swe_config = load_agent_config(
                    PROJECT_ROOT / "scripts" / "config_sweagent_heal_default.yaml"
                )["agent"]
                template_config = swe_config["templates"]
                tool_config = swe_config["tools"]
                history_config = swe_config["history_processors"][0]

                env_cfg = EnvironmentConfig(
                    deployment=LocalDeploymentConfig(),
                    repo=None,
                    post_startup_commands=(
                        [f"cd {shlex.quote(repo_dir)}"] if repo_dir else []
                    ),
                )
                env = SWEEnv.from_config(env_cfg)
                env.start()
                state["env"] = env

                task_dir_str = getattr(session, "task_dir", "") or ""
                if task_dir_str:
                    output_dir = Path(task_dir_str) / ".tmp" / "swe_adapter"
                    output_dir.mkdir(parents=True, exist_ok=True)
                else:
                    output_dir = Path(tempfile.mkdtemp(prefix="swe_adapter_"))
                state["output_dir"] = output_dir

                model_cfg = GenericAPIModelConfig(
                    name=self.model_name,
                    per_instance_cost_limit=0.0,
                    api_base=self.model_kwargs.get("api_base"),
                    api_key=self.model_kwargs.get("api_key"),
                    retry=RetryConfig(retries=3, min_wait=2, max_wait=20),
                )
                swe_tools_dir = PROJECT_ROOT / "baselines" / "SWE-agent"
                tool_cfg = ToolConfig(
                    bundles=[
                        Bundle(path=swe_tools_dir / bundle["path"])
                        for bundle in tool_config["bundles"]
                    ],
                    submit_command=EXECUTE_TOOL_NAME,
                    parse_function=FunctionCallingParser(),
                    enable_bash_tool=tool_config["enable_bash_tool"],
                    env_variables=dict(tool_config["env_variables"]),
                    registry_variables=dict(tool_config["registry_variables"]),
                    filter=ToolFilterConfig(blocklist_error_template=DISK_WRITE_REJECT_MSG),
                )
                template_cfg = TemplateConfig(
                    system_template=append_heal_rules(template_config["system_template"]),
                    instance_template=template_config["instance_template"],
                    next_step_template=template_config["next_step_template"],
                    next_step_no_output_template=template_config["next_step_no_output_template"],
                )
                agent_cfg = DefaultAgentConfig(
                    templates=template_cfg,
                    tools=tool_cfg,
                    model=model_cfg,
                    history_processors=[
                        CacheControlHistoryProcessor(
                            last_n_messages=history_config["last_n_messages"]
                        )
                    ],
                )
                agent = DefaultAgent.from_config(agent_cfg)
                state["agent"] = agent

                ps = TextProblemStatement(text=crash_body, id=session.instance_id)
                result = agent.run(env=env, problem_statement=ps, output_dir=output_dir)
                extra_info = dict(result.info or {})
                submission = extra_info.get("submission")
            else:
                agent = state["agent"]
                last_tool_call_id = None
                for h in reversed(getattr(agent, "history", []) or []):
                    if h.get("role") == "assistant":
                        tcs = h.get("tool_calls") or []
                        if tcs:
                            tc = tcs[0]
                            last_tool_call_id = (
                                tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                            )
                        break
                history_item = {
                    "role": "tool" if last_tool_call_id else "user",
                    "content": crash_body,
                    "agent": agent.name,
                    "message_type": "observation",
                }
                if last_tool_call_id:
                    history_item["tool_call_ids"] = [last_tool_call_id]
                agent._append_history(history_item)
                step_output = self.run_until_done(agent, session=session)
                submission = step_output.submission
                extra_info = {"submission": submission, "exit_status": step_output.exit_status}
        except self.HealToolRequested as request:
            elapsed = round(time.time() - t0, 2)
            if request.name == RERAISE_TOOL_NAME:
                logger.info(f"[{session.instance_id}] SWE-agent → reraise ({elapsed}s)")
                return {"action": "reraise", "reason": "agent_reraise_tool"}
            code = request.tool_args.get("code", "")
            if not bool(request.tool_args.get("submit", False)):
                logger.info(f"[{session.instance_id}] SWE-agent → execute(submit=false) ({elapsed}s)")
                return {"action": "execute", "code": code, "submit": False}
            processed = postprocess_fix_code(code)
            if processed is None:
                logger.warning(f"""[{session.instance_id}] SWE-agent execute submission rejected (syntax_error_after_postprocess) — submission:
{code}""")
                return {"action": "reraise", "reason": "syntax_error_after_postprocess"}
            logger.info(f"[{session.instance_id}] SWE-agent → execute(submit=true) ({elapsed}s)")
            return {"action": "execute", "code": processed, "submit": True}
        except BudgetExhausted:
            elapsed = round(time.time() - t0, 2)
            logger.info(f"[{session.instance_id}] SWE-agent → max_call_llm mid-step ({elapsed}s)")
            try:
                self.cleanup(session)
            except Exception as e:
                logger.warning(f"[{session.instance_id}] early cleanup on budget exhausted failed: {e}")
            return {"action": "reraise", "reason": "max_call_llm"}
        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            logger.error(f"""[{session.instance_id}] SWE-agent step error ({type(e).__name__}, {elapsed}s): {e}""")
            return {"action": "reraise", "reason": f"backend_error:{type(e).__name__}"}
        finally:
            set_baseline_llm_session(None)
            self.set_paths(None)

        try:
            agent = state["agent"]
            history = getattr(agent, "history", None)
            if history is None:
                history = getattr(agent, "messages", session.messages)
            session.messages[:] = list(history)
        except Exception as e:
            logger.warning(f"[{session.instance_id}] failed to sync swe-agent history: {e}")
        session.system_installed = True

        elapsed = round(time.time() - t0, 2)
        if not submission:
            logger.info(f"[{session.instance_id}] SWE-agent → no submission ({elapsed}s)")
            return {"action": "reraise", "reason": "no_submission"}
        logger.warning(f"""[{session.instance_id}] SWE-agent completed without execute/reraise tool interception ({elapsed}s): {extra_info}""")
        return {"action": "reraise", "reason": "no_heal_tool_call"}

    @staticmethod
    def run_until_done(agent, session=None) -> "StepOutput":  # noqa: F821
        while True:
            if session is not None and session.budget_exhausted():
                raise BudgetExhausted(f"""session budget exhausted mid-loop (total_llm_calls={session.total_llm_calls}/{session.max_call_llm})""")
            step_output = agent.step()
            if getattr(step_output, "done", False):
                return step_output

    def cleanup(self, session) -> None:
        state = getattr(session, "adapter_state", None)
        if not state:
            return
        session.adapter_state = None

        env = state.get("env")
        if env is not None:
            errors = []

            def close_env():
                try:
                    env.close()
                except Exception as e:
                    errors.append(e)

            close_thread = threading.Thread(target=close_env)
            close_thread.start()
            close_thread.join()
            if errors:
                logger.warning(f"[{session.instance_id}] SWEEnv.close() failed: {errors[0]}")
        output_dir = state.get("output_dir")
        task_dir_str = getattr(session, "task_dir", "") or ""
        if output_dir:
            try:
                p = Path(output_dir)
                is_under_task_dir = bool(task_dir_str) and str(p).startswith(
                    str(Path(task_dir_str).resolve())
                )
                if p.exists() and not is_under_task_dir:
                    shutil.rmtree(p, ignore_errors=True)
            except Exception as e:
                logger.warning(f"""[{session.instance_id}] failed to rmtree swe_adapter output_dir {output_dir}: {e}""")
        iid = getattr(session, "instance_id", "") or ""
        if iid:
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", iid)[:64] or "default"
            per_instance_tools = Path(f"/tmp/swe_tools_{safe}")
            per_instance_patch = Path(f"/tmp/model_{safe}.patch")
            try:
                if per_instance_tools.exists():
                    shutil.rmtree(per_instance_tools, ignore_errors=True)
                if per_instance_patch.exists():
                    per_instance_patch.unlink(missing_ok=True)
            except Exception as e:
                logger.info(
                    f"[{session.instance_id}] failed to clean per-instance /tmp paths: {e}"
                )

def latest_crash_body(session: "HealSession") -> str:
    for msg in reversed(session.messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""

class SDKHealAdapter(Agent):

    def decision_from_tool(self, name: str, args: dict, session: "HealSession") -> dict:
        if name == RERAISE_TOOL_NAME:
            return {"action": "reraise", "reason": "agent_reraise_tool"}
        if name != EXECUTE_TOOL_NAME:
            return {"action": "reraise", "reason": f"unknown_tool:{name}"}
        code = args.get("code", "") if isinstance(args, dict) else ""
        submit = bool(args.get("submit", False)) if isinstance(args, dict) else False
        if not submit:
            return {"action": "execute", "code": code, "submit": False}
        processed = postprocess_fix_code(code)
        if processed is None:
            logger.warning(f"""[{session.instance_id}] {type(self).__name__} execute submission rejected (syntax_error_after_postprocess) — submission:
{code}""")
            return {"action": "reraise", "reason": "syntax_error_after_postprocess"}
        return {"action": "execute", "code": processed, "submit": True}

class OpenHandsAdapter(SDKHealAdapter):

    def __init__(self, model_name: str, model_kwargs: dict | None = None):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}

    def step(self, session: "HealSession", repo_dir: str = "") -> dict:
        crash_body = latest_crash_body(session)
        if not crash_body:
            return {"action": "reraise", "reason": "no_crash_message"}

        try:
            from openhands.sdk import (
                LLM, Conversation, AgentContext,
                Action, Observation, TextContent,
            )
            from openhands.sdk.event import ActionEvent
            from openhands.sdk.tool import Tool, ToolExecutor, ToolDefinition, register_tool
            from openhands.tools.preset.default import get_default_agent
            from openhands.tools.terminal.impl import TerminalExecutor
            from openhands.tools.terminal.definition import TerminalObservation, TerminalTool
            from pydantic import SecretStr, Field
        except Exception as e:
            logger.error(f"[{session.instance_id}] OpenHands SDK import failed: {e}")
            return {"action": "reraise", "reason": f"sdk_import_error:{type(e).__name__}"}

        state = session.adapter_state
        is_first_crash = state is None
        t0 = time.time()
        if is_first_crash:
            input_holder = {"crash_body": crash_body}
        else:
            input_holder = state.get("input_holder") or {"crash_body": crash_body}
            input_holder["crash_body"] = crash_body

        decision: dict = input_holder.setdefault("decision", {"action": None, "code": "", "submit": False})
        decision["action"], decision["code"], decision["submit"] = None, "", False
        decision.pop("_budget", None)
        try:
            if is_first_crash:
                api_key = self.model_kwargs.get("api_key")
                llm = LLM(
                    usage_id="healagent",
                    model=self.model_name,
                    base_url=self.model_kwargs.get("api_base"),
                    api_key=SecretStr(api_key) if api_key else None,
                    temperature=AGENT_TEMPERATURE,
                )

                class ExecuteAction(Action):
                    code: str = Field(description="Python code to run in the caught frame.")
                    submit: bool = Field(description="False to inspect the live runtime state; true to submit the final Heal.")

                class ReraiseAction(Action):
                    pass

                class AckObservation(Observation):
                    @property
                    def to_llm_content(self):
                        return [TextContent(text="(submitted)")]

                class ExecuteExecutor(ToolExecutor):
                    def __call__(self, action, conversation=None):
                        decision["action"] = EXECUTE_TOOL_NAME
                        decision["code"] = action.code
                        decision["submit"] = action.submit
                        if conversation is not None:
                            conversation.pause()
                        return AckObservation()

                class ReraiseExecutor(ToolExecutor):
                    def __call__(self, action, conversation=None):
                        decision["action"] = RERAISE_TOOL_NAME
                        if conversation is not None:
                            conversation.pause()
                        return AckObservation()

                class ExecuteTool(ToolDefinition):
                    name = EXECUTE_TOOL_NAME
                    @classmethod
                    def create(cls, conv_state, **kw):
                        return [cls(description="Run code in the live frame; submit=false inspects, submit=true commits the Heal.",
                                    action_type=ExecuteAction, observation_type=AckObservation, executor=ExecuteExecutor())]

                class ReraiseTool(ToolDefinition):
                    name = RERAISE_TOOL_NAME
                    @classmethod
                    def create(cls, conv_state, **kw):
                        return [cls(description="Re-raise the current original exception.",
                                    action_type=ReraiseAction, observation_type=AckObservation, executor=ReraiseExecutor())]

                register_tool(EXECUTE_TOOL_NAME, ExecuteTool)
                register_tool(RERAISE_TOOL_NAME, ReraiseTool)

                class ReadOnlyTerminalExecutor(TerminalExecutor):
                    def __call__(self, action, conversation=None):
                        if not action.is_input and DiskWriteGuard.is_write(action.command):
                            logger.warning(
                                f"[{session.instance_id}] openhands blocked disk write: {action.command}"
                            )
                            return TerminalObservation.from_text(
                                DISK_WRITE_REJECT_MSG,
                                is_error=True,
                                command=action.command,
                                exit_code=-1,
                            )
                        return super().__call__(action, conversation)

                class ReadOnlyTerminalTool(TerminalTool):
                    name = "terminal"

                    @classmethod
                    def create(cls, conv_state, **params):
                        params.pop("executor", None)
                        executor = ReadOnlyTerminalExecutor(
                            working_dir=conv_state.workspace.working_dir
                        )
                        return super().create(conv_state, executor=executor, **params)

                register_tool("terminal", ReadOnlyTerminalTool)
                default_agent = get_default_agent(llm=llm, cli_mode=True)
                native_tools = [
                    tool
                    for tool in default_agent.tools
                    if tool.name not in DISK_EDIT_TOOL_NAMES["openhands"]
                ]
                tools = native_tools + [
                    Tool(name=EXECUTE_TOOL_NAME),
                    Tool(name=RERAISE_TOOL_NAME),
                ]
                agent = default_agent.model_copy(
                    update={
                        "tools": tools,
                        "agent_context": AgentContext(
                            system_message_suffix=("\n\n" + HEAL_RULES)
                        ),
                    }
                )

                def record_event(event):
                    try:
                        event_dict = event.model_dump(mode="json") if hasattr(event, "model_dump") else {"_repr": repr(event)}
                    except Exception as error:
                        logger.warning(f"[{session.instance_id}] openhands event model_dump failed: {error}")
                        event_dict = {"_unserializable_repr": repr(event)}
                    try:
                        session.messages.append({"role": "openhands_event", "content": event_dict})
                    except Exception as error:
                        logger.warning(f"[{session.instance_id}] openhands messages append failed: {error}")
                    if isinstance(event, ActionEvent):
                        input_tokens = output_tokens = 0
                        metrics_snapshot = None
                        try:
                            metrics = getattr(llm, "metrics", None)
                            if metrics is not None:
                                metrics_snapshot = metrics.model_dump(mode="json") if hasattr(metrics, "model_dump") else dict(metrics)
                                accumulated_usage = getattr(metrics, "accumulated_token_usage", None)
                                if accumulated_usage is not None:
                                    current_input_tokens = int(getattr(accumulated_usage, "prompt_tokens", 0) or 0)
                                    current_output_tokens = int(getattr(accumulated_usage, "completion_tokens", 0) or 0)
                                    previous_tokens = input_holder.get("openhands_previous_tokens", (0, 0))
                                    input_tokens = max(0, current_input_tokens - previous_tokens[0])
                                    output_tokens = max(0, current_output_tokens - previous_tokens[1])
                                    input_holder["openhands_previous_tokens"] = (current_input_tokens, current_output_tokens)
                        except Exception as error:
                            logger.warning(f"[{session.instance_id}] openhands metrics read failed: {error}")
                        session.note_llm_call(input_tokens, output_tokens)
                        if metrics_snapshot is not None and isinstance(event_dict, dict):
                            event_dict["_llm_metrics"] = metrics_snapshot
                        try:
                            session.llm_call_records.append({"raw_input": input_holder["crash_body"], "raw_output": event_dict})
                        except Exception as error:
                            logger.warning(f"[{session.instance_id}] openhands record append failed: {error}")
                        if session.budget_exhausted():
                            decision.setdefault("_budget", True)
                            decision["_budget"] = True

                workdir = repo_dir or (getattr(session, "task_dir", "") or ".")
                conversation = Conversation(agent=agent, callbacks=[record_event], workspace=workdir, stuck_detection=False)
                state = {"conversation": conversation, "input_holder": input_holder}
                session.adapter_state = state

            conversation = state["conversation"]
            conversation.send_message(crash_body)
            conversation.run()
        except Exception as e:
            logger.error(f"[{session.instance_id}] openhands step error ({type(e).__name__}): {e}")
            return {"action": "reraise", "reason": f"backend_error:{type(e).__name__}"}

        elapsed = round(time.time() - t0, 2)
        if decision.get("_budget") and decision["action"] is None:
            logger.info(f"[{session.instance_id}] openhands → max_call_llm reached mid-step")
            return {"action": "reraise", "reason": "max_call_llm"}
        if decision["action"] is None:
            logger.info(f"[{session.instance_id}] openhands → no healing tool called ({elapsed}s)")
            return {"action": "reraise", "reason": "no_submission"}
        logger.info(f"[{session.instance_id}] openhands → {decision['action']} ({elapsed}s)")
        return self.decision_from_tool(
            decision["action"],
            {"code": decision["code"], "submit": decision.get("submit", False)},
            session,
        )

    def cleanup(self, session) -> None:
        state = session.adapter_state
        if not state:
            return
        conv = state.get("conversation")
        if conv is not None:
            for attr in ("close", "disconnect"):
                fn = getattr(conv, attr, None)
                if callable(fn):
                    try:
                        fn()
                        break
                    except Exception as e:
                        logger.warning(f"[{session.instance_id}] openhands {attr}() failed: {e}")

class CodexAdapter(SDKHealAdapter):

    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": [EXECUTE_TOOL_NAME, RERAISE_TOOL_NAME]},
            "code": {"type": "string"},
            "submit": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["action", "code", "submit", "reason"],
        "additionalProperties": False,
    }

    def __init__(self, model_name: str | None = None, model_kwargs: dict | None = None):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}

    @staticmethod
    def bare_model_name(model_name: str | None) -> str | None:
        if not model_name:
            return None
        for prefix in ("openai/", "anthropic/"):
            if model_name.startswith(prefix):
                return model_name[len(prefix):]
        return model_name

    def step(self, session: "HealSession", repo_dir: str = "") -> dict:
        crash_body = latest_crash_body(session)
        if not crash_body:
            return {"action": "reraise", "reason": "no_crash_message"}
        if session.budget_exhausted():
            return {"action": "reraise", "reason": "max_call_llm"}

        try:
            from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
            from openai_codex.generated.v2_all import TurnStatus
        except Exception as error:
            logger.error(f"[{session.instance_id}] Codex SDK import failed: {error}")
            return {"action": "reraise", "reason": f"sdk_import_error:{type(error).__name__}"}

        api_base = self.model_kwargs.get("api_base")
        api_key = self.model_kwargs.get("api_key")
        provider = "openai"
        env = dict(os.environ)
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        overrides = ["web_search=\"disabled\""]
        if api_base:
            overrides.append(f'openai_base_url={json.dumps(api_base)}')

        workdir = repo_dir or (getattr(session, "task_dir", "") or ".")
        config = CodexConfig(
            config_overrides=tuple(overrides),
            cwd=workdir,
            env=env,
        )
        prompt = crash_body + "\n\n" + HEAL_RULES + """

Use the local repository only through the read-only sandbox. Return exactly one JSON object matching the requested schema:
- `action="execute"`, `code=<runtime code>`, and `submit=false` to inspect the live runtime state.
- `action="execute"`, `code=<final Heal code>`, and `submit=true` to submit the final Heal.
- `action="reraise"`, `code=""`, and `submit=false` if no safe Heal can be determined.
Do not edit repository files. Do not return a source patch.
"""

        state = session.adapter_state or {}
        started = time.monotonic()
        try:
            with Codex(config) as codex:
                thread_id = state.get("thread_id")
                if thread_id:
                    thread = codex.thread_resume(
                        thread_id,
                        approval_mode=ApprovalMode.deny_all,
                        cwd=workdir,
                        model=self.bare_model_name(self.model_name),
                        model_provider=provider,
                        sandbox=Sandbox.read_only,
                    )
                else:
                    thread = codex.thread_start(
                        approval_mode=ApprovalMode.deny_all,
                        cwd=workdir,
                        ephemeral=False,
                        model=self.bare_model_name(self.model_name),
                        model_provider=provider,
                        sandbox=Sandbox.read_only,
                    )
                    session.adapter_state = {"thread_id": thread.id}
                result = thread.run(
                    prompt,
                    cwd=workdir,
                    output_schema=self.OUTPUT_SCHEMA,
                    sandbox=Sandbox.read_only,
                )
        except Exception as error:
            session.note_llm_call()
            logger.error(f"[{session.instance_id}] Codex step error ({type(error).__name__}): {error}")
            return {"action": "reraise", "reason": f"backend_error:{type(error).__name__}"}

        usage = result.usage.total if result.usage is not None else None
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        session.note_llm_call(input_tokens, output_tokens)
        raw_result = {
            "thread_id": thread.id,
            "turn_id": result.id,
            "status": result.status.value,
            "error": result.error.model_dump(mode="json", by_alias=True) if result.error else None,
            "final_response": result.final_response,
            "items": [item.model_dump(mode="json", by_alias=True) for item in result.items],
            "usage": result.usage.model_dump(mode="json", by_alias=True) if result.usage else None,
        }
        session.llm_call_records.append({"raw_input": prompt, "raw_output": raw_result})
        session.messages.append({"role": "assistant", "content": raw_result})
        if result.status != TurnStatus.completed:
            reason = result.error.message if result.error is not None else result.status.value
            return {"action": "reraise", "reason": f"codex_{reason}"}

        response_text = result.final_response or ""
        try:
            decision = json.loads(response_text)
        except (TypeError, json.JSONDecodeError) as strict_error:
            decision = None
            decoder = json.JSONDecoder()
            for match in re.finditer(r"\{", response_text):
                try:
                    candidate, _ = decoder.raw_decode(response_text[match.start():])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict) and candidate.get("action") in {EXECUTE_TOOL_NAME, RERAISE_TOOL_NAME}:
                    decision = candidate
                    logger.warning(f"[{session.instance_id}] Codex response required embedded JSON extraction: {strict_error}")
                    break
            if decision is None:
                logger.warning(f"[{session.instance_id}] Codex response is not valid JSON: {strict_error}")
                return {"action": "reraise", "reason": "invalid_json_response"}
        if not isinstance(decision, dict):
            return {"action": "reraise", "reason": "invalid_response_type"}
        action = decision.get("action")
        if action == RERAISE_TOOL_NAME:
            return {"action": "reraise", "reason": decision.get("reason") or "agent_reraise"}
        if action != EXECUTE_TOOL_NAME:
            return {"action": "reraise", "reason": f"unknown_action:{action}"}
        elapsed = round(time.monotonic() - started, 2)
        logger.info(f"[{session.instance_id}] codex → execute(submit={bool(decision.get('submit'))}) ({elapsed}s)")
        return self.decision_from_tool(
            EXECUTE_TOOL_NAME,
            {
                "code": decision.get("code", ""),
                "submit": bool(decision.get("submit", False)),
            },
            session,
        )

class ClaudeCodeStripProxy:

    STRIP_FIELDS = ("output_config", "context_management")
    instances: dict = {}
    lock = threading.Lock()

    def __init__(self, upstream_base: str):
        self.upstream_base = upstream_base.rstrip("/")
        self.port: int | None = None
        self.httpd = None

    @classmethod
    def for_upstream(cls, upstream_base: str) -> "ClaudeCodeStripProxy":
        with cls.lock:
            inst = cls.instances.get(upstream_base)
            if inst is None or inst.httpd is None:
                inst = cls(upstream_base)
                inst.start()
                cls.instances[upstream_base] = inst
            return inst

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        import http.server, urllib.request, urllib.error
        upstream = self.upstream_base
        strip_fields = self.STRIP_FIELDS

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def proxy(self):
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b""
                if body:
                    try:
                        payload = json.loads(body)
                        if isinstance(payload, dict):
                            removed = [k for k in strip_fields if k in payload]
                            if removed:
                                for k in removed:
                                    payload.pop(k, None)
                                body = json.dumps(payload).encode("utf-8")
                                logger.info(f"[claude-code strip-proxy] removed {removed} from {self.path}")
                    except Exception as e:
                        logger.warning(f"[claude-code strip-proxy] body not JSON on {self.path} ({e}); passing through")
                fwd_headers = {k: v for k, v in self.headers.items()
                               if k.lower() not in ("host", "content-length", "connection", "transfer-encoding")}
                req = urllib.request.Request(upstream + self.path, data=body,
                                             headers=fwd_headers, method=self.command)
                try:
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        self.relay(resp.status, resp.headers, resp.read())
                except urllib.error.HTTPError as e:
                    self.relay(e.code, e.headers, e.read())
                except Exception as e:
                    logger.error(f"[claude-code strip-proxy] upstream error on {self.path}: {e}")
                    msg = json.dumps({"type": "error", "error": {"type": "proxy_error", "message": str(e)}}).encode()
                    self.relay(502, {"Content-Type": "application/json"}, msg)

            def relay(self, status, headers, data):
                self.send_response(status)
                for k, v in (headers.items() if hasattr(headers, "items") else headers):
                    if k.lower() in ("content-length", "transfer-encoding", "content-encoding", "connection"):
                        continue
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_POST = proxy
            do_GET = proxy

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        t = threading.Thread(target=self.httpd.serve_forever, daemon=True,
                             name=f"cc-strip-proxy-{self.port}")
        t.start()
        logger.info(f"[claude-code strip-proxy] started on {self.base_url} → upstream {upstream} "
                    f"(strips {list(self.STRIP_FIELDS)})")

class ClaudeCodeAdapter(SDKHealAdapter):

    def __init__(self, model_name: str | None = None, model_kwargs: dict | None = None):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}

    @staticmethod
    def bare_model_name(model_name: str | None) -> str | None:
        if not model_name:
            return None
        for prefix in ("openai/", "anthropic/"):
            if model_name.startswith(prefix):
                return model_name[len(prefix):]
        return model_name

    def step(self, session: "HealSession", repo_dir: str = "") -> dict:
        crash_body = latest_crash_body(session)
        if not crash_body:
            return {"action": "reraise", "reason": "no_crash_message"}
        try:
            return asyncio.run(self.astep(session, repo_dir, crash_body))
        except Exception as e:
            logger.error(f"[{session.instance_id}] claude-code step error ({type(e).__name__}): {e}")
            return {"action": "reraise", "reason": f"backend_error:{type(e).__name__}"}

    async def astep(self, session, repo_dir, crash_body) -> dict:
        try:
            from claude_agent_sdk import (
                ClaudeSDKClient, ClaudeAgentOptions, tool, create_sdk_mcp_server,
                AssistantMessage, ToolUseBlock, SystemMessage, HookMatcher, ResultMessage,
            )
        except Exception as e:
            logger.error(f"[{session.instance_id}] Claude Agent SDK import failed: {e}")
            return {"action": "reraise", "reason": f"sdk_import_error:{type(e).__name__}"}

        decision: dict = {"action": None, "code": "", "submit": False}

        @tool(EXECUTE_TOOL_NAME, "Run code in the live frame; submit=false inspects, submit=true commits the Heal.", {"code": str, "submit": bool})
        async def execute_tool(args):
            decision["action"] = EXECUTE_TOOL_NAME
            decision["code"] = args.get("code", "")
            decision["submit"] = bool(args.get("submit", False))
            return {"content": [{"type": "text", "text": "(executed)"}]}

        @tool(RERAISE_TOOL_NAME, "Re-raise the current original exception.", {})
        async def reraise_tool(args):
            decision["action"] = RERAISE_TOOL_NAME
            return {"content": [{"type": "text", "text": "(reraised)"}]}

        heal_server = create_sdk_mcp_server(name="heal", version="1.0.0", tools=[execute_tool, reraise_tool])

        state = session.adapter_state
        resume_id = state.get("session_id") if state else None

        bare_model = self.bare_model_name(self.model_name)
        env_overlay: dict = {}
        api_base = self.model_kwargs.get("api_base")
        api_key = self.model_kwargs.get("api_key")
        if api_base:
            upstream_base = re.sub(r"/v1/?$", "", api_base)
            proxy = ClaudeCodeStripProxy.for_upstream(upstream_base)
            env_overlay["ANTHROPIC_BASE_URL"] = proxy.base_url
        if api_key:
            env_overlay["ANTHROPIC_AUTH_TOKEN"] = api_key
            env_overlay["ANTHROPIC_API_KEY"] = api_key

        async def deny_disk_edit(hook_input, tool_use_id, context):
            tool_name = hook_input.get("tool_name", "") if isinstance(hook_input, dict) else ""
            tool_input = hook_input.get("tool_input", {}) if isinstance(hook_input, dict) else {}
            command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
            if tool_name in DISK_EDIT_TOOL_NAMES["claude-code"] or (tool_name == "Bash" and DiskWriteGuard.is_write(command)):
                logger.warning(f"[{session.instance_id}] claude-code blocked disk write: {tool_name} {command}")
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": DISK_WRITE_REJECT_MSG,
                    }
                }
            return {}

        tools = {"type": "preset", "preset": "claude_code"}
        allowed_tools = [
            f"mcp__heal__{RERAISE_TOOL_NAME}",
            f"mcp__heal__{EXECUTE_TOOL_NAME}",
            "Read", "Grep", "Glob", "Bash",
        ]
        remaining_turns = (
            max(0, session.max_call_llm - session.total_llm_calls)
            if session.max_call_llm
            else None
        )
        if remaining_turns == 0:
            return {"action": "reraise", "reason": "max_call_llm"}
        opts = ClaudeAgentOptions(
            tools=tools,
            mcp_servers={"heal": heal_server},
            allowed_tools=allowed_tools,
            disallowed_tools=list(DISK_EDIT_TOOL_NAMES["claude-code"]),
            hooks={"PreToolUse": [HookMatcher(hooks=[deny_disk_edit])]},
            system_prompt={"type": "preset", "preset": "claude_code", "append": ("\n\n" + HEAL_RULES)},
            cwd=repo_dir or (getattr(session, "task_dir", "") or "."),
            resume=resume_id,
            permission_mode="dontAsk",
            max_turns=remaining_turns,
            model=bare_model,
            env=env_overlay,
        )

        t0 = time.time()
        async with ClaudeSDKClient(options=opts) as client:
            await client.query(crash_body)
            interrupted = False
            async for msg in client.receive_response():
                if isinstance(msg, SystemMessage) and getattr(msg, "subtype", "") == "init":
                    sid = (getattr(msg, "data", {}) or {}).get("session_id")
                    if sid:
                        session.adapter_state = {"session_id": sid}
                if isinstance(msg, AssistantMessage):
                    usage = getattr(msg, "usage", None) or {}
                    input_tokens = int(usage.get("input_tokens", 0) or 0) if isinstance(usage, dict) else 0
                    output_tokens = int(usage.get("output_tokens", 0) or 0) if isinstance(usage, dict) else 0
                    session.note_llm_call(input_tokens, output_tokens)
                    try:
                        message_dict = dataclasses.asdict(msg)
                    except Exception as error:
                        logger.warning(f"[{session.instance_id}] claude-code asdict failed, falling back to repr: {error}")
                        message_dict = {"_unserializable_repr": repr(msg)}
                    try:
                        session.llm_call_records.append({"raw_input": crash_body, "raw_output": message_dict})
                        session.messages.append({"role": "assistant", "content": message_dict})
                    except Exception as error:
                        logger.warning(f"[{session.instance_id}] claude-code record append failed: {error}")
                    for block in getattr(msg, "content", []) or []:
                        if isinstance(block, ToolUseBlock) and decision["action"] is None:
                            nm = (block.name or "").split("__")[-1]
                            if nm in (RERAISE_TOOL_NAME, EXECUTE_TOOL_NAME):
                                inp = block.input or {}
                                decision["action"] = nm
                                decision["code"] = inp.get("code", "")
                                decision["submit"] = bool(inp.get("submit", False))
                                if not interrupted:
                                    interrupted = True
                                    try:
                                        await client.interrupt()
                                    except Exception as error:
                                        logger.warning(f"[{session.instance_id}] claude-code interrupt failed: {error}")
                if isinstance(msg, ResultMessage):
                    try:
                        result_dict = dataclasses.asdict(msg)
                    except Exception as error:
                        logger.warning(f"[{session.instance_id}] claude-code ResultMessage asdict failed: {error}")
                        result_dict = {"_unserializable_repr": repr(msg)}
                    try:
                        session.llm_call_records.append({"raw_input": crash_body, "raw_output": result_dict})
                        session.messages.append({"role": "result", "content": result_dict})
                    except Exception as error:
                        logger.warning(f"[{session.instance_id}] claude-code ResultMessage record failed: {error}")
                    usage = getattr(msg, "usage", None) or {}
                    input_tokens = int(usage.get("input_tokens", 0) or 0) if isinstance(usage, dict) else 0
                    output_tokens = int(usage.get("output_tokens", 0) or 0) if isinstance(usage, dict) else 0
                    reported_turns = max(1, int(getattr(msg, "num_turns", 1) or 1))
                    logger.info(f"[{session.instance_id}] claude-code query reported_num_turns="
                                f"{reported_turns} input_tokens={input_tokens} "
                                f"output_tokens={output_tokens} "
                                f"(counted AssistantMessage calls={session.total_llm_calls}/"
                                f"{session.max_call_llm})")
                if isinstance(msg, ResultMessage) and (decision["action"] is not None or interrupted):
                    break
                if session.budget_exhausted():
                    break

        elapsed = round(time.time() - t0, 2)
        if decision["action"] is None:
            if session.budget_exhausted():
                logger.info(f"[{session.instance_id}] claude-code → max_call_llm reached mid-step")
                return {"action": "reraise", "reason": "max_call_llm"}
            logger.info(f"[{session.instance_id}] claude-code → no healing tool called ({elapsed}s)")
            return {"action": "reraise", "reason": "no_submission"}
        logger.info(f"[{session.instance_id}] claude-code → {decision['action']} ({elapsed}s)")
        return self.decision_from_tool(
            decision["action"],
            {"code": decision["code"], "submit": decision.get("submit", False)},
            session,
        )
