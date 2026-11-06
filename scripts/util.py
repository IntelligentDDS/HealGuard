import os, json, random, re, time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

import litellm
from loguru import logger

litellm.suppress_debug_info = True
litellm.success_callback = []
litellm.failure_callback = []
litellm._async_success_callback = []
litellm._async_failure_callback = []

HEALBENCH_INSTANCE_IDS = frozenset(
    json.loads(line)["instance_id"]
    for line in (Path(__file__).resolve().parent.parent / "datasets/healbench.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
)
logger.info(f"Loaded {len(HEALBENCH_INSTANCE_IDS)} benchmark instance IDs; first 2: {sorted(HEALBENCH_INSTANCE_IDS)[:2]}")

_STRIPPED_KEYS = ("temperature", "top_p", "top_k")
_STRIP_PARAM_PREFIXES = ("bedrock/",)
_STRIP_PARAM_KEYWORDS = ("claude",)

def _normalize_kw(kw):
    model = kw.get("model", "") or ""
    if (any(model.startswith(p) for p in _STRIP_PARAM_PREFIXES)
            or any(kw_sub in model.lower() for kw_sub in _STRIP_PARAM_KEYWORDS)):
        for k in _STRIPPED_KEYS:
            kw.pop(k, None)
    msgs = kw.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("content") is None:
                m["content"] = ""

if not getattr(litellm, "_hotfix_strip_params_patched", False):
    _orig_completion = litellm.completion
    _orig_acompletion = litellm.acompletion

    def _completion_hotfix(*a, **kw):
        _normalize_kw(kw)
        if a and isinstance(a[0], list):
            for m in a[0]:
                if isinstance(m, dict) and m.get("content") is None:
                    m["content"] = ""
        return _orig_completion(*a, **kw)

    async def _acompletion_hotfix(*a, **kw):
        _normalize_kw(kw)
        if a and isinstance(a[0], list):
            for m in a[0]:
                if isinstance(m, dict) and m.get("content") is None:
                    m["content"] = ""
        return await _orig_acompletion(*a, **kw)

    litellm.completion = _completion_hotfix
    litellm.acompletion = _acompletion_hotfix
    litellm._hotfix_strip_params_patched = True
    litellm._hotfix_completion_original = _orig_completion
    litellm._hotfix_acompletion_original = _orig_acompletion

if not getattr(litellm, "_hotfix_cost_map_patched", False):
    try:
        from litellm import cost_calculator as _cc
        if hasattr(_cc, "completion_cost"):
            _orig_cc = _cc.completion_cost
            def _safe_completion_cost(*a, **kw):
                try:
                    return _orig_cc(*a, **kw)
                except Exception as _cost_e:
                    logger.warning(f"completion_cost fallback to 0.0: {_cost_e}")
                    return 0.0
            _cc.completion_cost = _safe_completion_cost
            if hasattr(litellm, "completion_cost"):
                litellm.completion_cost = _safe_completion_cost
        litellm._hotfix_cost_map_patched = True
    except Exception as _e:
        logger.warning(f"litellm cost_map hotfix skipped: {_e}")




@dataclass
class Config:

    provider: str = field(default_factory=lambda: os.getenv("MODEL_PROVIDER", "openai"))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4"))
    llm_max_concurrency: int = 5
    embedding_base_url: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    )
    embedding_api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", "ollama"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "bge-m3:567m-fp16"))
    embedding_max_concurrency: int = 5
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", os.path.join(os.getcwd(), "output")))




def init_logger(log_file: str, level: str = "INFO"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(
        Path(log_file),
        level=level,
        rotation="1 day",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} [{level}] ({name}:{function}:{line}) {message}",
    )
    return logger


_log_dirs_created: set[str] = set()


def log_api(log_file: str, data: dict):
    d = os.path.dirname(log_file) or "."
    if d not in _log_dirs_created:
        os.makedirs(d, exist_ok=True)
        _log_dirs_created.add(d)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(), **data}, ensure_ascii=False) + "\n")




def _litellm_model(model: str, provider: str = "openai") -> str:
    if "/" in model:
        return model
    _REAL_LITELLM_PROVIDERS = {
        "openai", "anthropic", "bedrock", "azure", "vertex_ai",
        "ollama", "together_ai", "groq", "cohere", "gemini", "mistral",
    }
    effective = provider if provider in _REAL_LITELLM_PROVIDERS else "openai"
    return f"{effective}/{model}"



RETRYABLE_KEYWORDS = ["timeout", "rate limit", "rate_limit", "429", "503", "overloaded", "connection"]



def _retry_on_transient_sync(func, *, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            err = str(e).lower()
            if any(kw in err for kw in RETRYABLE_KEYWORDS) and attempt < max_retries - 1:
                wait = min(5 * (2**attempt) + random.uniform(0, 1), 60)
                m = re.search(r"retry after (\d+)", err)
                if m:
                    wait = min(int(m.group(1)) + 1, 120)
                logger.warning(f"retry {attempt + 1}/{max_retries}, wait {wait:.1f}s: {e}")
                time.sleep(wait)
                continue
            raise







def call_llm(
    messages: list[dict],
    config: Config,
    *,
    json_schema: dict | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    **kwargs,
) -> dict:
    if json_schema:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
    kwargs.setdefault("timeout", 600)
    if config.provider == "seed":
        extra = kwargs.pop("extra_body", {})
        extra.setdefault("reasoning_effort", "minimal")
        kwargs["extra_body"] = extra
    elif config.provider == "gpt":
        kwargs.setdefault("reasoning_effort", "none")
    elif config.provider == "deepseek":
        extra = kwargs.pop("extra_body", {})
        extra.setdefault("thinking", {"type": "disabled"})
        kwargs["extra_body"] = extra

    model_name = _litellm_model(config.openai_model, config.provider)

    resp = _retry_on_transient_sync(
        lambda: litellm.completion(
            model=model_name,
            messages=messages,
            api_base=config.openai_base_url,
            api_key=config.openai_api_key,
            **kwargs,
        )
    )

    raw = resp.model_dump()
    content = resp.choices[0].message.content or ""

    if json_schema and content:
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {e}, content={content}")
            raise ValueError(f"LLM returned invalid JSON: {e}") from e

    u = raw.get("usage", {})
    logger.info(
        f"LLM ok model={config.openai_model} prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')} total={u.get('total_tokens')}"
    )
    input_tokens = u.get("prompt_tokens") or u.get("input_tokens") or 0
    output_tokens = u.get("completion_tokens") or u.get("output_tokens") or 0
    return {"content": content, "raw": raw, "input_tokens": input_tokens, "output_tokens": output_tokens}





def load_instance_file(path: str | None) -> list[str] | None:
    if not path:
        return None
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    return [line.strip() for line in p.read_text().splitlines() if line.strip() and not line.startswith("#")]


def build_instr_prep_cmd(
    workdir: str, do_overlay: bool, use_pythonpath: bool, eh_setup_cmd: str = ""
) -> tuple[str, str]:
    overlay_cp = f"cp -r {workdir}/_overlay/. {workdir}/ && " if do_overlay else ""
    pyc_clean = f"find {workdir} -name '*.pyc' -delete 2>/dev/null; "
    if use_pythonpath:
        reinstall = ""
        pythonpath = f"export PYTHONPATH={workdir}:{workdir}/lib:${{PYTHONPATH}} && "
    else:
        reinstall = (
            f"(cd {workdir} && "
            f"pip install -e . --no-deps --no-build-isolation -q 2>/dev/null || "
            f"python setup.py develop --no-deps 2>/dev/null || true); "
        )
        pythonpath = ""
    capture_flag = " --capture=no" if eh_setup_cmd else ""
    prep_cmd = overlay_cp + reinstall + pyc_clean + (eh_setup_cmd or "") + pythonpath
    return prep_cmd, capture_flag

