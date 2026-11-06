from __future__ import annotations

import argparse, ast, hashlib, json, math, os, re, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import pandas as pd

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEALBENCH_RESULTS = PROJECT_ROOT / "results" / "healbench"
from util import HEALBENCH_INSTANCE_IDS  # noqa: E402

def parse_unified_diff(patch_text: str, *, align_replacements: bool = True) -> "_DiffMapping":
    explicit: dict[str, dict[int, int]] = {}
    gold_only_lines: dict[str, set[int]] = {}
    offsets: dict[str, list[tuple[int, int]]] = {}

    current_file = None
    buggy_line = 0
    gold_line = 0

    lines = patch_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("--- a/"):
            i += 1
            continue
        elif line.startswith("+++ b/"):
            current_file = line[6:]
            buggy_line = gold_line = 0
            if current_file not in explicit:
                explicit[current_file] = {}
                gold_only_lines[current_file] = set()
                offsets[current_file] = [(1, 0)]
            i += 1
            continue
        elif line.startswith("diff --git"):
            if current_file and offsets.get(current_file):
                last_offset = buggy_line - gold_line
                offsets[current_file].append((gold_line, last_offset))
            current_file = None
            i += 1
            continue
        elif line.startswith("@@"):
            if current_file is not None and (gold_line > 0 or buggy_line > 0):
                prev_offset = buggy_line - gold_line
                offsets[current_file].append((gold_line, prev_offset))
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m and current_file is not None:
                buggy_line = int(m.group(1))
                gold_line = int(m.group(2))
            i += 1
            continue
        elif current_file is None:
            i += 1
            continue
        elif line.startswith("-"):
            del_start_buggy = buggy_line
            del_count = 0
            while i < len(lines) and lines[i].startswith("-") and not lines[i].startswith("---"):
                del_count += 1
                i += 1
            add_count = 0
            add_start_gold = gold_line
            while i < len(lines) and lines[i].startswith("+") and not lines[i].startswith("+++"):
                add_count += 1
                i += 1
            paired = min(del_count, add_count) if align_replacements else 0
            for k in range(paired):
                explicit[current_file][add_start_gold + k] = del_start_buggy + k
            for k in range(paired, add_count):
                gold_only_lines[current_file].add(add_start_gold + k)
            buggy_line = del_start_buggy + del_count
            gold_line = add_start_gold + add_count
            continue
        elif line.startswith("+"):
            gold_only_lines[current_file].add(gold_line)
            gold_line += 1
            i += 1
            continue
        elif line.startswith(" ") or line == "":
            if current_file in explicit:
                explicit[current_file][gold_line] = buggy_line
            buggy_line += 1
            gold_line += 1
            i += 1
            continue
        else:
            i += 1
            continue

    if current_file and offsets.get(current_file):
        last_offset = buggy_line - gold_line
        offsets[current_file].append((gold_line, last_offset))

    return _DiffMapping(explicit, gold_only_lines, offsets)


class _DiffMapping(dict):

    def __init__(self, explicit, gold_only_lines, offsets):
        super().__init__({f: None for f in explicit})
        self._explicit = explicit
        self._gold_only = gold_only_lines
        self._offsets = offsets

    def __contains__(self, key):
        return key in self._explicit

    def __getitem__(self, file_path):
        if file_path not in self._explicit:
            raise KeyError(file_path)
        return _FileLineMap(
            self._explicit[file_path],
            self._gold_only.get(file_path, set()),
            sorted(self._offsets.get(file_path, [(1, 0)])),
        )

    def keys(self):
        return self._explicit.keys()

    def __iter__(self):
        return iter(self._explicit)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class _FileLineMap:

    def __init__(self, explicit: dict[int, int], gold_only: set[int], offsets: list[tuple[int, int]]):
        self._explicit = explicit
        self._gold_only = gold_only
        self._offsets = offsets

    def get(self, gold_line: int, default=None):
        if gold_line in self._explicit:
            return self._explicit[gold_line]
        if gold_line in self._gold_only:
            return default
        offset = self._get_offset_at(gold_line)
        return gold_line + offset

    def __contains__(self, gold_line: int):
        if gold_line in self._explicit:
            return True
        if gold_line in self._gold_only:
            return False
        return True

    def _get_offset_at(self, gold_line: int) -> int:
        offset = 0
        for boundary, off in self._offsets:
            if boundary <= gold_line:
                offset = off
            else:
                break
        return offset


def _parse_marker_meta(key) -> dict | None:
    if not isinstance(key, str):
        return None
    parts = key.split("\t")
    if len(parts) >= 5:
        try:
            decision_line = int(parts[3])
        except ValueError:
            return {
                "key": key,
                "file": parts[0],
                "func": parts[1],
                "branch_type": parts[2],
                "decision_line": None,
                "source_text": "\t".join(parts[3:]),
            }
        return {
            "key": key,
            "file": parts[0],
            "func": parts[1],
            "branch_type": parts[2],
            "decision_line": decision_line,
            "source_text": "\t".join(parts[4:]),
        }
    elif len(parts) == 4:
        return {
            "key": key,
            "file": parts[0],
            "func": parts[1],
            "branch_type": parts[2],
            "decision_line": None,
            "source_text": parts[3],
        }
    return None




def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p_hat = k / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return (round(p_hat, 4), round(max(0, centre - margin), 4), round(min(1, centre + margin), 4))


def mcnemar_test(b: int, c: int) -> tuple[float, float]:
    n_disc = b + c
    if n_disc == 0:
        return (0.0, 1.0)
    if n_disc < 25:
        from math import comb

        k = max(b, c)
        p_tail = sum(comb(n_disc, i) for i in range(k, n_disc + 1)) / (2**n_disc)
        p_value = min(2 * p_tail, 1.0)
        chi2 = (b - c) ** 2 / n_disc
        return (round(chi2, 4), round(p_value, 4))
    chi2 = (abs(b - c) - 1) ** 2 / n_disc
    p_value = math.erfc(math.sqrt(chi2 / 2))
    return (round(chi2, 4), round(p_value, 4))


def paired_comparison(instances_a: list[dict], instances_b: list[dict], label_a: str = "A", label_b: str = "B") -> dict:
    map_a = {i["instance_id"]: i["status"] for i in instances_a if "instance_id" in i}
    map_b = {i["instance_id"]: i["status"] for i in instances_b if "instance_id" in i}
    common = sorted(set(map_a) & set(map_b))

    if not common:
        logger.warning(f"No common instances between {label_a} and {label_b}")
        return {"error": "no common instances"}

    a_only = []
    b_only = []
    both_correct = []
    neither = []
    for iid in common:
        ac = map_a[iid] == "correct"
        bc = map_b[iid] == "correct"
        if ac and bc:
            both_correct.append(iid)
        elif ac and not bc:
            a_only.append(iid)
        elif not ac and bc:
            b_only.append(iid)
        else:
            neither.append(iid)

    n = len(common)
    ka = len(both_correct) + len(a_only)
    kb = len(both_correct) + len(b_only)

    chi2, p_val = mcnemar_test(len(a_only), len(b_only))
    ci_a = wilson_ci(ka, n)
    ci_b = wilson_ci(kb, n)

    budget_a = {i["instance_id"]: i.get("llm_calls", 0) for i in instances_a}
    budget_b = {i["instance_id"]: i.get("llm_calls", 0) for i in instances_b}

    return {
        "n_common": n,
        "only_in_a": len(set(map_a) - set(map_b)),
        "only_in_b": len(set(map_b) - set(map_a)),
        label_a: {
            "correct": ka,
            "rate": ci_a[0],
            "wilson_ci_95": [ci_a[1], ci_a[2]],
            "avg_llm_calls": round(mean(budget_a.get(iid, 0) for iid in common), 1) if common else 0,
        },
        label_b: {
            "correct": kb,
            "rate": ci_b[0],
            "wilson_ci_95": [ci_b[1], ci_b[2]],
            "avg_llm_calls": round(mean(budget_b.get(iid, 0) for iid in common), 1) if common else 0,
        },
        "concordance": {
            "both_correct": len(both_correct),
            "a_only": len(a_only),
            "b_only": len(b_only),
            "neither": len(neither),
        },
        "mcnemar": {"chi2": chi2, "p_value": p_val, "significant_05": p_val < 0.05},
        "discordant_details": {
            f"{label_a}_only_correct": a_only,
            f"{label_b}_only_correct": b_only,
        },
    }




def _read_heal_trace(log_path: Path, event_type: str) -> list:
    records = []
    if not log_path.exists():
        return records
    prefix = "[heal_trace]"
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, 1):
            if not line.startswith(prefix):
                continue
            try:
                record = json.loads(line[len(prefix):])
            except Exception as e:
                logger.warning(f"invalid heal trace {log_path}:{line_number}: {e}")
                continue
            if record.get("type") == event_type:
                records.append(record.get("data"))
    return records


def _read_docker_output(log_path: Path) -> str:
    records = _read_heal_trace(log_path, "docker_result")
    if not records or not isinstance(records[-1], dict):
        return log_path.read_text("utf-8", errors="replace") if log_path.exists() else ""
    result = records[-1]
    return f"{result.get('stdout', '')}\n{result.get('stderr', '')}"


def _read_healed_trace(log_path: Path) -> dict | None:
    records = _read_heal_trace(log_path, "healed_trace")
    return records[-1] if records and isinstance(records[-1], dict) else None


def classify_status(raw_status: str, docker_log_path: Path | None, test_rc: int | None = None) -> str:
    if raw_status == "correct":
        return "correct"
    if raw_status == "infra_error":
        return "infra_error"
    if test_rc is not None:
        try:
            rc = int(test_rc)
        except (ValueError, TypeError):
            rc = None
        if rc == 0:
            return "correct"
        if rc is not None and rc >= 2:
            return "crash"
        if rc == -1:
            return "crash"
    if docker_log_path and docker_log_path.exists():
        docker_log = _read_docker_output(docker_log_path)
        has_failed = "FAILED" in docker_log
        failed_lines = re.findall(r"FAILED .+ - (.+)", docker_log)
        has_error_section = bool(re.search(r"(?m)^={3,} ERRORS ={3,}", docker_log))
        if has_error_section:
            return "crash"
        if has_failed and failed_lines:
            all_assertion = all(
                "AssertionError" in l or "assert" in l.lower() or "Failed:" in l
                for l in failed_lines
            )
            if all_assertion:
                return "proceed"
        elif has_failed:
            return "proceed"
    return "crash"




def _load_instance(inst_dir: Path) -> dict | None:
    out_f = inst_dir / "output.json"
    if not out_f.exists():
        return None
    data = json.loads(out_f.read_text("utf-8"))

    raw_status = data.get("status", "fail")
    docker_log_f = inst_dir / "run_docker.log"
    data["status"] = classify_status(raw_status, docker_log_f, data.get("test_rc"))

    if data["status"] == "infra_error":
        return None

    if "instance_id" not in data:
        data["instance_id"] = inst_dir.name

    inp_f = inst_dir / "input.json"
    if inp_f.exists():
        inp = json.loads(inp_f.read_text("utf-8"))
        data["source_dataset"] = inp.get("source_dataset", "unknown")
        data["error_type"] = inp.get("error_type", "unknown")
        data["repo"] = inp.get("repo", "unknown")
        ptp_raw = inp.get("pass_to_pass", "[]")
        try:
            ptp_list = json.loads(ptp_raw) if isinstance(ptp_raw, str) else ptp_raw
            data["ptp_count"] = len(ptp_list) if isinstance(ptp_list, list) else 0
        except Exception:
            data["ptp_count"] = 0
        data["test_code_leaked"] = False
    else:
        data["source_dataset"] = data["error_type"] = data["repo"] = "unknown"
        data["ptp_count"] = 0
        data["test_code_leaked"] = False

    docker_log_f = inst_dir / "run_docker.log"
    ts = _summarize_trace(docker_log_f) if docker_log_f.exists() else {}
    data["guard_reject"] = ts.get("n_guard_rejects", 0)
    data["static_reject"] = ts.get("n_guard_static_rejects", 0)
    data["rubric_reject"] = ts.get("n_guard_rubric_rejects", 0)
    data["guard_skip"] = ts.get("n_guard_skips", 0)
    data["total_crashes"] = ts.get("n_crashes", 0)
    data["n_fix_actions"] = ts.get("n_fix_actions", 0)
    data["n_skip_actions"] = ts.get("n_skip_actions", 0)
    data["n_probe_actions"] = ts.get("n_probe_actions", 0)
    data["n_suppress_fixes"] = ts.get("n_suppress_fixes", 0)
    data["guard_reject_reraise_list"] = ts.get("guard_reject_reraise_list", [])
    data["guard_reject_reraise_list_static"] = ts.get("guard_reject_reraise_list_static", [])
    data["guard_reject_reraise_list_rubric"] = ts.get("guard_reject_reraise_list_rubric", [])
    data["first_attempt_skip_list"] = ts.get("first_attempt_skip_list", [])

    data["_inst_dir"] = str(inst_dir)

    max_call_llm = data.get("max_call_llm", 0)
    llm_calls = data.get("llm_calls", 0)
    data["budget_exhausted"] = bool(max_call_llm and llm_calls >= max_call_llm)
    return data


def load_results(results_dir: Path, exclude_leaked: bool = True) -> list[dict]:
    instances = []
    n_leaked = 0
    n_unusable = 0
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("instance_"):
            continue
        if d.name not in HEALBENCH_INSTANCE_IDS:
            n_unusable += 1
            continue
        inst = _load_instance(d)
        if inst:
            if exclude_leaked and inst.get("test_code_leaked", False):
                n_leaked += 1
                continue
            instances.append(inst)
    if n_unusable:
        logger.info(f"Excluded {n_unusable} outside-manifest (blacklisted) instances from {results_dir}")
    if n_leaked:
        logger.info(f"Excluded {n_leaked} instances with test code leakage from {results_dir}")
    return instances




def compute_metrics(instances: list[dict]) -> dict:
    n = len(instances)
    if n == 0:
        return {"total": 0}

    correct = sum(1 for i in instances if i["status"] == "correct")
    proceed = sum(1 for i in instances if i["status"] in ("correct", "proceed"))
    crash = sum(1 for i in instances if i["status"] == "crash")

    total_fix = sum(i.get("total_fix", 0) for i in instances)
    total_skip = sum(i.get("total_skip", 0) for i in instances)
    total_llm = sum(i.get("llm_calls", 0) for i in instances)
    total_time = sum(i.get("run_time_s", 0) for i in instances)
    total_guard_reject = sum(i.get("guard_reject", 0) for i in instances)
    total_static_reject = sum(i.get("static_reject", 0) for i in instances)
    total_rubric_reject = sum(i.get("rubric_reject", 0) for i in instances)
    total_crashes = sum(i.get("total_crashes", 0) for i in instances)
    total_suppress = sum(i.get("n_suppress_fixes", 0) for i in instances)
    _sup_per_inst = [
        i["n_suppress_fixes"] / i["n_fix_actions"]
        for i in instances
        if i.get("n_fix_actions", 0) > 0 and "n_suppress_fixes" in i
    ]
    suppress_rate_macro = round(sum(_sup_per_inst) / len(_sup_per_inst) * 100, 1) if _sup_per_inst else 0

    ci_correct = wilson_ci(correct, n)
    ci_proceed = wilson_ci(proceed, n)

    budget_exhausted = sum(1 for i in instances if i.get("budget_exhausted", False))
    budget_exhausted_and_fail = sum(
        1 for i in instances if i.get("budget_exhausted", False) and i["status"] != "correct"
    )

    return {
        "total": n,
        "correct": correct,
        "correct_pct": round(correct / n * 100, 1),
        "correct_ci_95": [ci_correct[1], ci_correct[2]],
        "proceed": proceed,
        "proceed_pct": round(proceed / n * 100, 1),
        "proceed_ci_95": [ci_proceed[1], ci_proceed[2]],
        "crash": crash,
        "avg_llm_calls": round(total_llm / n, 1),
        "avg_fix": round(total_fix / n, 1),
        "avg_skip": round(total_skip / n, 1),
        "avg_time_s": round(total_time / n, 1),
        "total_fix": total_fix,
        "total_skip": total_skip,
        "guard_reject_total": total_guard_reject,
        "static_reject_total": total_static_reject,
        "rubric_reject_total": total_rubric_reject,
        "suppress_fixes": total_suppress,
        "suppress_rate": suppress_rate_macro,
        "suppress_rate_micro": round(total_suppress / total_fix * 100, 1) if total_fix > 0 else 0,
        "total_crashes": total_crashes,
        "budget_exhausted": budget_exhausted,
        "budget_exhausted_and_fail": budget_exhausted_and_fail,
    }


def compute_breakdown(instances: list[dict], key: str) -> dict[str, dict]:
    groups = defaultdict(list)
    for i in instances:
        groups[i.get(key, "unknown")].append(i)
    return {k: compute_metrics(v) for k, v in sorted(groups.items())}



GOLD_TRACES_DIR = PROJECT_ROOT / "results" / "gold_traces"


_MODEL_PRICE_PER_1M = {
    "claude-opus-4.8": (5.0, 25.0),
    "gpt-5.5": (5.0, 30.0),
    "deepseek-v4-flash-260425": (0.09, 0.18),
    "deepseek-v4-flash-260425-thinking": (0.09, 0.18),
}


def _instance_cost_usd(model: str, in_tok: float, out_tok: float) -> float | None:
    price = _MODEL_PRICE_PER_1M.get(model)
    if price is None:
        return None
    in_price, out_price = price
    return (in_tok or 0) / 1e6 * in_price + (out_tok or 0) / 1e6 * out_price




def _branch_set(trace: dict) -> set:
    bm = trace.get("branch_markers")
    if bm:
        if isinstance(bm, list):
            return set(bm)
        return {(f, b) for f, bdict in bm.items() for b in bdict}
    return {(f, b) for f, bdict in trace.get("branches", {}).items() for b in bdict}


def _raises_set(trace: dict) -> set:
    rm = trace.get("raise_markers")
    if rm:
        if isinstance(rm, list) and rm and isinstance(rm[0], str):
            return set(rm)
        return {(r["file"], r["func"], r["src_hash"], r["exc_type"]) for r in rm}
    return {(r["file"], r.get("func", ""), str(r["line"]), r.get("exc_type", "")) for r in trace.get("raises", [])}


_RAISE_TYPE_RE = re.compile(r"\braise\s+([A-Za-z_][A-Za-z0-9_.]*)")


def _gold_raise_error_types(raise_markers: set) -> set:
    types: set = set()
    for m in raise_markers:
        if isinstance(m, str):
            parts = m.split("\t")
            if len(parts) >= 5 and parts[2] == "raise":
                hit = _RAISE_TYPE_RE.search(parts[4])
                if hit:
                    types.add(hit.group(1).split(".")[-1])
                continue
            if len(parts) >= 3 and parts[2] and parts[2] != "raise":
                types.add(parts[2].split(".")[-1])
        elif isinstance(m, tuple):
            if len(m) >= 3 and isinstance(m[2], str) and m[2] and m[2] != "raise":
                types.add(m[2].split(".")[-1])
            elif len(m) >= 4 and isinstance(m[3], str) and m[3]:
                types.add(m[3].split(".")[-1])
    return types


def read_gold_marker_log(path: Path) -> dict:
    from run_record_gold_trace import _parse_markers_from_stderr

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line == "=== STDERR ===\n":
                return _parse_markers_from_stderr(stream.read())
    return {}


def load_gold_trace(instance_id: str, gold_dir: Path | None = None) -> dict | None:
    root = gold_dir if gold_dir is not None else GOLD_TRACES_DIR
    directory = root / instance_id
    compact = directory / "output.json"
    f = compact if compact.is_file() else root / f"{instance_id}.json"
    if not f.is_file():
        return None
    try:
        result = json.loads(f.read_text("utf-8"))
        if f == compact:
            for kind in ("gold", "buggy"):
                path = directory / f"{kind}.log"
                if path.is_file():
                    result[f"{kind}_trace"] = read_gold_marker_log(path)
        return result
    except (OSError, ValueError) as error:
        logger.warning(f"Cannot read gold trace {f}: {error}")
        return None


def compute_trace_metrics(
    gold_trace: dict, buggy_trace: dict, healed_trace: dict | None = None, patch_text: str = "",
    guard_reject_reraise_list: list | None = None, first_attempt_skip_list: list | None = None,
    guard_reject_reraise_list_static: list | None = None,
    guard_reject_reraise_list_rubric: list | None = None,
) -> dict:
    gold_b = _branch_set(gold_trace)
    buggy_b = _branch_set(buggy_trace)

    _alignment_pairs = None
    _alignment_gold_new = None
    if patch_text and gold_b:
        sample_key = next(iter(gold_b)) if gold_b else ""
        if isinstance(sample_key, str) and len(sample_key.split("\t")) >= 5:
            try:
                line_map = parse_unified_diff(patch_text)
                if line_map:

                    def _to_canonical(keys: set, is_gold: bool) -> dict[tuple, str]:
                        result = {}
                        for key in keys:
                            meta = _parse_marker_meta(key)
                            if not meta or meta["decision_line"] is None:
                                result[(key,)] = key
                                continue
                            fp = meta["file"]
                            dl = meta["decision_line"]
                            bt = meta["branch_type"]
                            if is_gold and fp in line_map:
                                buggy_dl = line_map[fp].get(dl)
                                if buggy_dl is None:
                                    result[(fp, dl, bt, "__gold_new__")] = key
                                    continue
                                dl = buggy_dl
                            result[(fp, dl, bt)] = key
                        return result

                    gold_canonical = _to_canonical(gold_b, is_gold=True)
                    buggy_canonical = _to_canonical(buggy_b, is_gold=False)

                    gold_b = set(gold_canonical.keys())
                    buggy_b = set(buggy_canonical.keys())

                    _alignment_pairs = gold_canonical
                    _alignment_gold_new = {k for k in gold_canonical if len(k) == 4 and k[3] == "__gold_new__"}
                    _alignment_active = True
                else:
                    _alignment_active = False
            except Exception as _e:
                logger.warning(f"trace alignment failed, falling back to direct comparison: {_e}")
                _alignment_active = False
        else:
            _alignment_active = False
    else:
        _alignment_active = False
    gold_r = _raises_set(gold_trace)
    buggy_r = _raises_set(buggy_trace)

    if _alignment_active:

        def _canonical_raise(keys: set, is_gold: bool) -> set:
            result = set()
            for key in keys:
                if not isinstance(key, str):
                    result.add(key)
                    continue
                parts = key.split("\t")
                if len(parts) >= 5:
                    fp, func = parts[0], parts[1]
                    _hit = _RAISE_TYPE_RE.search(parts[4])
                    rtype = _hit.group(1).split(".")[-1] if _hit else parts[2]
                    try:
                        dl = int(parts[3])
                    except ValueError:
                        result.add((key,))
                        continue
                    if is_gold and fp in line_map:
                        buggy_dl = line_map[fp].get(dl)
                        if buggy_dl is None:
                            result.add((fp, func, rtype, dl, "__gold_new__"))
                            continue
                        dl = buggy_dl
                    result.add((fp, func, rtype, dl))
                else:
                    result.add((key,))
            return result

        gold_r = _canonical_raise(gold_r, is_gold=True)
        buggy_r = _canonical_raise(buggy_r, is_gold=False)

    metrics = {
        "gold_only_branches": len(gold_b - buggy_b),
        "buggy_only_branches": len(buggy_b - gold_b),
        "shared_branches": len(gold_b & buggy_b),
        "gold_only_raises": len(gold_r - buggy_r),
        "buggy_only_raises": len(buggy_r - gold_r),
        "alignment_used": _alignment_active,
        "alignment_pairs_count": len(_alignment_pairs) if _alignment_pairs else 0,
        "alignment_gold_new_count": len(_alignment_gold_new) if _alignment_gold_new else 0,
        "branch_discriminative": bool(len(gold_b - buggy_b) > 0),
    }

    gold_raise_types = _gold_raise_error_types(gold_r - buggy_r)

    def _set_reraise_metrics(reraise_list, prefix):
        s = set(reraise_list or [])
        if not s:
            p = r = f1 = None
        else:
            inter = len(s & gold_raise_types)
            p = inter / len(s)
            r = inter / len(gold_raise_types) if gold_raise_types else None
            if p is None or r is None:
                f1 = None
            elif (p + r) > 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0.0
        metrics[f"{prefix}_precision"] = round(p, 4) if p is not None else None
        metrics[f"{prefix}_recall"] = round(r, 4) if r is not None else None
        metrics[f"{prefix}_f1"] = round(f1, 4) if f1 is not None else None

    _set_reraise_metrics(guard_reject_reraise_list, "guard_reject_correct_reraise")
    _set_reraise_metrics(guard_reject_reraise_list_static, "guard_reject_correct_reraise_static")
    _set_reraise_metrics(guard_reject_reraise_list_rubric, "guard_reject_correct_reraise_rubric")
    _set_reraise_metrics(first_attempt_skip_list, "first_attempt_correct_reraise")

    if healed_trace is None:
        return metrics

    healed_b = _branch_set(healed_trace)
    healed_r = _raises_set(healed_trace)

    if _alignment_active:
        healed_canonical = set()
        for key in healed_b:
            meta = _parse_marker_meta(key)
            if not meta or meta["decision_line"] is None:
                healed_canonical.add((key,))
            else:
                healed_canonical.add((meta["file"], meta["decision_line"], meta["branch_type"]))
        healed_b = healed_canonical
        healed_r = _canonical_raise(healed_r, is_gold=False)

    _degraded = not healed_b and (buggy_b)

    gold_only = gold_b - buggy_b
    novel_branches = healed_b - gold_b - buggy_b

    if _degraded:
        buggy_cov = len(buggy_b & gold_b) / len(gold_b) if gold_b else None
        gold_only_r = gold_r - buggy_r
        metrics.update(
            {
                "gold_branch_precision": 0.0 if gold_only else None,
                "gold_branch_recall": 0.0 if gold_only else None,
                "gold_branch_f1": 0.0 if gold_only else None,
                "buggy_branch_coverage": round(buggy_cov, 4) if buggy_cov is not None else None,
                "healing_delta": round(-buggy_cov, 4) if buggy_cov is not None else None,
                "gold_path_recovery": 0.0 if gold_only else None,
                "novel_path_rate": 0.0,
                "jaccard_gold": 0.0 if gold_only else None,
                "correct_reraise_precision": 0.0 if gold_only_r else None,
                "correct_reraise_recall": 0.0 if gold_only_r else None,
                "correct_reraise_f1": 0.0 if gold_only_r else None,
                "raise_introduction_rate": 0.0,
                "gold_raise_preserved_rate": 0.0 if gold_only_r else None,
                "exception_fidelity": 0,
                "degraded": True,
                "branch_discriminative": bool(len(gold_b - buggy_b) > 0),
            }
        )
        return metrics


    healed_post = healed_b - buggy_b
    _post_inter = len(healed_post & gold_only)
    gold_branch_precision = _post_inter / len(healed_post) if healed_post else None
    gold_branch_recall = _post_inter / len(gold_only) if gold_only else None
    if gold_branch_precision is None or gold_branch_recall is None:
        gold_branch_f1 = None
    elif (gold_branch_precision + gold_branch_recall) > 0:
        gold_branch_f1 = (
            2 * gold_branch_precision * gold_branch_recall
            / (gold_branch_precision + gold_branch_recall)
        )
    else:
        gold_branch_f1 = 0.0

    buggy_branch_coverage = len(buggy_b & gold_b) / len(gold_b) if gold_b else None

    healing_delta = (gold_branch_recall - buggy_branch_coverage
                     if gold_branch_recall is not None and buggy_branch_coverage is not None else None)

    gold_path_recovery = len(healed_b & gold_only) / len(gold_only) if gold_only else None

    novel_path_rate = len(novel_branches) / len(healed_b) if healed_b else None

    union_post = healed_post | gold_only
    jaccard_gold = len(healed_post & gold_only) / len(union_post) if union_post else None

    new_raises = healed_r - gold_r - buggy_r

    raise_introduction_rate = len(new_raises) / len(healed_r) if healed_r else 0.0

    gold_only_r = gold_r - buggy_r
    healed_post_r = healed_r - buggy_r
    _post_inter_r = len(gold_only_r & healed_post_r)
    correct_reraise_precision = _post_inter_r / len(healed_post_r) if healed_post_r else None
    correct_reraise_recall = _post_inter_r / len(gold_only_r) if gold_only_r else None
    if correct_reraise_precision is None or correct_reraise_recall is None:
        correct_reraise_f1 = None
    elif (correct_reraise_precision + correct_reraise_recall) > 0:
        correct_reraise_f1 = (
            2 * correct_reraise_precision * correct_reraise_recall
            / (correct_reraise_precision + correct_reraise_recall)
        )
    else:
        correct_reraise_f1 = 0.0

    gold_raise_preserved_rate = correct_reraise_recall

    target_raises = buggy_r - gold_r
    target_eliminated = not bool(healed_r & target_raises)
    gold_preserved = gold_r.issubset(healed_r) if gold_r else True

    if target_eliminated and not new_raises and gold_preserved:
        ef = 2
    elif target_eliminated:
        ef = 1
    else:
        ef = 0

    metrics.update(
        {
            "gold_branch_precision": round(gold_branch_precision, 4) if gold_branch_precision is not None else None,
            "gold_branch_recall": round(gold_branch_recall, 4) if gold_branch_recall is not None else None,
            "gold_branch_f1": round(gold_branch_f1, 4) if gold_branch_f1 is not None else None,
            "buggy_branch_coverage": round(buggy_branch_coverage, 4) if buggy_branch_coverage is not None else None,
            "healing_delta": round(healing_delta, 4) if healing_delta is not None else None,
            "gold_path_recovery": round(gold_path_recovery, 4) if gold_path_recovery is not None else None,
            "novel_path_rate": round(novel_path_rate, 4) if novel_path_rate is not None else None,
            "jaccard_gold": round(jaccard_gold, 4) if jaccard_gold is not None else None,
            "branch_discriminative": bool(len(gold_b - buggy_b) > 0),
            "correct_reraise_precision": round(correct_reraise_precision, 4) if correct_reraise_precision is not None else None,
            "correct_reraise_recall": round(correct_reraise_recall, 4) if correct_reraise_recall is not None else None,
            "correct_reraise_f1": round(correct_reraise_f1, 4) if correct_reraise_f1 is not None else None,
            "raise_introduction_rate": (
                round(raise_introduction_rate, 4) if raise_introduction_rate is not None else None
            ),
            "gold_raise_preserved_rate": (
                round(gold_raise_preserved_rate, 4) if gold_raise_preserved_rate is not None else None
            ),
            "exception_fidelity": ef,
        }
    )

    return metrics


def compute_trace_metrics_for_instances(instances: list[dict]) -> dict:
    results = []
    n_with_healed = 0
    for inst in instances:
        iid = inst.get("instance_id", "")
        gt_data = load_gold_trace(iid)
        if gt_data is None:
            continue

        gold_test_rc = gt_data.get("gold_test_rc", -1)
        if gold_test_rc != 0:
            logger.info(f"Skipping {iid} for trace metrics: gold_test_rc={gold_test_rc}")
            continue

        gold_trace = gt_data.get("gold_trace", {})
        buggy_trace = gt_data.get("buggy_trace", {})

        healed_trace = None
        inst_dir = inst.get("_inst_dir")
        if not inst_dir and inst.get("trace_path"):
            inst_dir = str(Path(inst["trace_path"]).parent)
        if inst_dir:
            healed_trace = _read_healed_trace(Path(inst_dir) / "run_docker.log")
            if healed_trace is not None:
                n_with_healed += 1

        patch_text = ""
        if inst_dir:
            inp_path = Path(inst_dir) / "input.json"
            if inp_path.exists():
                try:
                    inp_data = json.loads(inp_path.read_text("utf-8"))
                    patch_text = inp_data.get("patch", "")
                except Exception as _e:
                    logger.warning(f"failed to read input.json for patch_text: {_e}")

        guard_reject_reraise_list = inst.get("guard_reject_reraise_list", [])
        first_attempt_skip_list = inst.get("first_attempt_skip_list", [])
        tm = compute_trace_metrics(
            gold_trace, buggy_trace, healed_trace=healed_trace, patch_text=patch_text,
            guard_reject_reraise_list=guard_reject_reraise_list,
            first_attempt_skip_list=first_attempt_skip_list,
            guard_reject_reraise_list_static=inst.get("guard_reject_reraise_list_static", []),
            guard_reject_reraise_list_rubric=inst.get("guard_reject_reraise_list_rubric", []),
        )
        tm["instance_id"] = iid
        tm["status"] = inst.get("status", "unknown")
        results.append(tm)

    if not results:
        return {"n_with_trace": 0, "n_with_healed": 0}

    n = len(results)
    agg = {
        "n_with_trace": n,
        "n_with_healed": n_with_healed,
        "avg_gold_only_branches": round(mean(r["gold_only_branches"] for r in results), 1),
        "avg_buggy_only_branches": round(mean(r["buggy_only_branches"] for r in results), 1),
        "avg_shared_branches": round(mean(r["shared_branches"] for r in results), 1),
        "avg_gold_only_raises": round(mean(r["gold_only_raises"] for r in results), 2),
        "avg_buggy_only_raises": round(mean(r["buggy_only_raises"] for r in results), 2),
        "per_instance": results,
    }

    healed_results = [r for r in results if "gold_path_recovery" in r]
    if healed_results:

        def _avg(key):
            vals = [r[key] for r in healed_results if r.get(key) is not None]
            return round(mean(vals), 4) if vals else None

        for key in ("gold_branch_precision", "gold_branch_recall", "gold_branch_f1",
                    "buggy_branch_coverage", "healing_delta",
                    "gold_path_recovery", "novel_path_rate", "jaccard_gold"):
            v = _avg(key)
            if v is not None:
                agg[f"avg_{key}"] = v

        disc_results = [r for r in healed_results if r.get("branch_discriminative")]
        agg["n_branch_discriminative"] = len(disc_results)

        def _avg_disc(key):
            vals = [r[key] for r in disc_results if r.get(key) is not None]
            return round(mean(vals), 4) if vals else None

        for key in ("gold_branch_precision", "gold_branch_recall", "gold_branch_f1",
                    "gold_path_recovery", "novel_path_rate"):
            v = _avg_disc(key)
            agg[f"avg_{key}_disc"] = v

        for key in ("correct_reraise_precision", "correct_reraise_recall", "correct_reraise_f1",
                    "raise_introduction_rate", "gold_raise_preserved_rate"):
            v = _avg(key)
            if v is not None:
                agg[f"avg_{key}"] = v

        agg["avg_exception_fidelity"] = round(mean(r["exception_fidelity"] for r in healed_results), 2)
        agg["exception_fidelity_distribution"] = {
            "ef=2 (perfect)": sum(1 for r in healed_results if r["exception_fidelity"] == 2),
            "ef=1 (partial)": sum(1 for r in healed_results if r["exception_fidelity"] == 1),
            "ef=0 (failed)": sum(1 for r in healed_results if r["exception_fidelity"] == 0),
        }

        agg["n_degraded"] = sum(1 for r in healed_results if r.get("degraded"))

    def _avg_all(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(mean(vals), 4) if vals else None

    for key in ("first_attempt_correct_reraise_precision", "first_attempt_correct_reraise_recall",
                "first_attempt_correct_reraise_f1",
                "guard_reject_correct_reraise_precision", "guard_reject_correct_reraise_recall",
                "guard_reject_correct_reraise_f1",
                "guard_reject_correct_reraise_static_precision", "guard_reject_correct_reraise_static_recall",
                "guard_reject_correct_reraise_static_f1",
                "guard_reject_correct_reraise_rubric_precision", "guard_reject_correct_reraise_rubric_recall",
                "guard_reject_correct_reraise_rubric_f1"):
        v = _avg_all(key)
        if v is not None:
            agg[f"avg_{key}"] = v

    return agg


def print_trace_metrics(trace_metrics: dict, label: str = ""):
    n = trace_metrics.get("n_with_trace", 0)
    if n == 0:
        logger.info("No gold traces available for trace-based evaluation.")
        return
    header = f"Trace-based Metrics — {label}" if label else "Trace-based Metrics"
    print(f"\n{'='*60}")
    print(f"  {header} ({n} instances with gold traces)")
    print(f"{'='*60}")
    print(f"  Avg gold-only branches:  {trace_metrics['avg_gold_only_branches']}")
    print(f"  Avg buggy-only branches: {trace_metrics['avg_buggy_only_branches']}")
    print(f"  Avg shared branches:     {trace_metrics['avg_shared_branches']}")
    print(f"  Avg gold-only raises:    {trace_metrics['avg_gold_only_raises']}")
    print(f"  Avg buggy-only raises:   {trace_metrics['avg_buggy_only_raises']}")

    healed = [r for r in trace_metrics.get("per_instance", []) if "gold_branch_recall" in r]
    if healed:
        nh = len(healed)
        n_deg = sum(1 for r in healed if r.get("degraded"))

        def _print_metric(key, label):
            vals = [r[key] for r in healed if r.get(key) is not None]
            if vals:
                print(f"  {label:<42} {mean(vals):.4f}  (n={len(vals)})")

        deg_str = f", {n_deg} degraded" if n_deg else ""
        print(f"\n  --- Branch Alignment (n={nh}{deg_str}) ---")
        _print_metric("gold_branch_precision", "Precision (healed ∩ gold / healed):")
        _print_metric("gold_branch_recall", "Recall (healed ∩ gold / gold):")
        _print_metric("gold_branch_f1", "F1 Score:")
        _print_metric("buggy_branch_coverage", "Buggy Branch Coverage (baseline):")
        _print_metric("healing_delta", "Healing Delta (↑ better, >0=improved):")
        _print_metric("gold_path_recovery", "Gold Path Recovery (↑ better):")
        _print_metric("novel_path_rate", "Novel Path Rate (↓ better):")
        _print_metric("jaccard_gold", "Jaccard Similarity (↑ better):")

        print(f"\n  --- Exception Behavior (n={nh}) ---")
        _print_metric("raise_introduction_rate", "Raise Introduction Rate (↓ better):")
        _print_metric("gold_raise_preserved_rate", "Gold Raise Preserved Rate (↑ better):")
        print(
            f"  {'Exception Fidelity:':<42} "
            f"2(perfect)={sum(1 for r in healed if r['exception_fidelity']==2)}, "
            f"1(partial)={sum(1 for r in healed if r['exception_fidelity']==1)}, "
            f"0(failed)={sum(1 for r in healed if r['exception_fidelity']==0)}"
        )

    def _agg_line(key, label):
        v = trace_metrics.get(f"avg_{key}")
        if v is not None:
            print(f"  {label:<52} {v}")

    print(f"\n  --- First Re-raise Accuracy (FRA: first-attempt skip vs gold raises) ---")
    _agg_line("first_attempt_correct_reraise_precision", "Precision:")
    _agg_line("first_attempt_correct_reraise_recall", "Recall:")
    _agg_line("first_attempt_correct_reraise_f1", "F1:")
    print(f"\n  --- Guard Re-raise Accuracy (GRA: guard-reject→skip vs gold raises) ---")
    _agg_line("guard_reject_correct_reraise_precision", "[TOTAL] Precision:")
    _agg_line("guard_reject_correct_reraise_recall", "[TOTAL] Recall:")
    _agg_line("guard_reject_correct_reraise_f1", "[TOTAL] F1:")
    _agg_line("guard_reject_correct_reraise_static_precision", "[Layer1 static] Precision:")
    _agg_line("guard_reject_correct_reraise_static_recall", "[Layer1 static] Recall:")
    _agg_line("guard_reject_correct_reraise_static_f1", "[Layer1 static] F1:")
    _agg_line("guard_reject_correct_reraise_rubric_precision", "[Layer2 rubric] Precision:")
    _agg_line("guard_reject_correct_reraise_rubric_recall", "[Layer2 rubric] Recall:")
    _agg_line("guard_reject_correct_reraise_rubric_f1", "[Layer2 rubric] F1:")

    if not healed:
        n_healed = trace_metrics.get("n_with_healed", 0)
        if n_healed == 0:
            print(f"\n  (No healed trace found in run_docker.log)")
    print()




def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({num*100//den}%)" if den > 0 else "0/0"


def _ci_str(ci: list[float]) -> str:
    return f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"


def print_overall(metrics: dict, label: str = ""):
    n = metrics["total"]
    header = f"Overall ({label})" if label else "Overall"
    print(f"\n{'='*60}")
    print(f"  {header}: {n} instances")
    print(f"{'='*60}")
    print(f"  CORRECT:  {_pct(metrics['correct'], n)}  CI₉₅ {_ci_str(metrics['correct_ci_95'])}")
    print(f"  PROCEED:  {_pct(metrics['proceed'], n)}  CI₉₅ {_ci_str(metrics['proceed_ci_95'])}")
    print(f"  CRASH:    {metrics['crash']}")
    print(f"  ──────────────────────────────────")
    print(f"  Avg LLM calls:  {metrics['avg_llm_calls']}")
    print(f"  Avg fix count:  {metrics['avg_fix']}")
    print(f"  Avg skip count: {metrics['avg_skip']}")
    print(f"  Avg wall time:  {metrics['avg_time_s']}s")
    print(f"  ──────────────────────────────────")
    print(
        f"  Guard rejects:  {metrics['guard_reject_total']} (static={metrics['static_reject_total']}, rubric={metrics['rubric_reject_total']})"
    )
    print(f"  Suppress fixes: {metrics['suppress_fixes']}/{metrics['total_fix']} ({metrics['suppress_rate']}%)")
    print(f"  Total crashes:   {metrics['total_crashes']}")
    print(f"  Budget exhausted: {metrics['budget_exhausted']} (fail: {metrics['budget_exhausted_and_fail']})")


def print_breakdown(breakdown: dict[str, dict], group_name: str):
    if not breakdown:
        return
    print(f"\n{'─'*60}")
    print(f"  By {group_name}")
    print(f"{'─'*60}")
    print(f"  {'Category':<30} {'N':>4} {'CORRECT':>10} {'PROCEED':>10} {'Avg LLM':>8}")
    print(f"  {'─'*30} {'─'*4} {'─'*10} {'─'*10} {'─'*8}")
    for cat, m in breakdown.items():
        n = m["total"]
        print(f"  {cat:<30} {n:>4} {_pct(m['correct'],n):>10} {_pct(m['proceed'],n):>10} {m['avg_llm_calls']:>8}")


def print_comparison(all_results: list[tuple[str, list[dict]]]):
    if len(all_results) < 2:
        return
    print(f"\n{'='*80}")
    print(f"  Multi-run Comparison")
    print(f"{'='*80}")
    print(f"  {'Run':<40} {'N':>4} {'CORRECT':>10} {'PROCEED':>10} {'Avg LLM':>8} {'Avg Fix':>8}")
    print(f"  {'─'*40} {'─'*4} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")
    for label, instances in all_results:
        m = compute_metrics(instances)
        n = m["total"]
        print(
            f"  {label:<40} {n:>4} {_pct(m['correct'],n):>10} {_pct(m['proceed'],n):>10} {m['avg_llm_calls']:>8} {m['avg_fix']:>8}"
        )

    if len(all_results) == 2:
        (la, ia), (lb, ib) = all_results
        for inst_list in [ia, ib]:
            for inst in inst_list:
                if "instance_id" not in inst:
                    inst["instance_id"] = inst.get("task_id", "")
        result = paired_comparison(ia, ib, la, lb)
        print_paired_comparison(result, la, lb)


def print_paired_comparison(result: dict, label_a: str, label_b: str):
    if "error" in result:
        print(f"\n  ⚠ Paired comparison failed: {result['error']}")
        return
    print(f"\n{'='*80}")
    print(f"  Paired Statistical Comparison (McNemar's test)")
    print(f"{'='*80}")
    n = result["n_common"]
    print(
        f"  Common instances: {n}  (only in {label_a}: {result['only_in_a']}, only in {label_b}: {result['only_in_b']})"
    )
    print(f"  ──────────────────────────────────")

    ra = result[label_a]
    rb = result[label_b]
    print(
        f"  {label_a:<30} heal={ra['correct']}/{n} ({ra['rate']*100:.1f}%)  CI₉₅ [{ra['wilson_ci_95'][0]*100:.1f}%, {ra['wilson_ci_95'][1]*100:.1f}%]  avg_llm={ra['avg_llm_calls']}"
    )
    print(
        f"  {label_b:<30} heal={rb['correct']}/{n} ({rb['rate']*100:.1f}%)  CI₉₅ [{rb['wilson_ci_95'][0]*100:.1f}%, {rb['wilson_ci_95'][1]*100:.1f}%]  avg_llm={rb['avg_llm_calls']}"
    )

    conc = result["concordance"]
    print(f"  ──────────────────────────────────")
    print(f"  Concordance table:")
    print(f"    Both correct:     {conc['both_correct']}")
    print(f"    {label_a} only:   {conc['a_only']}")
    print(f"    {label_b} only:   {conc['b_only']}")
    print(f"    Neither:          {conc['neither']}")

    mc = result["mcnemar"]
    sig = "YES ✗" if mc["significant_05"] else "NO ✓"
    print(f"  ──────────────────────────────────")
    print(f"  McNemar's χ² = {mc['chi2']:.3f}, p = {mc['p_value']:.4f}  (significant at α=0.05: {sig})")

    disc = result["discordant_details"]
    a_key = f"{label_a}_only_correct"
    b_key = f"{label_b}_only_correct"
    if disc.get(a_key):
        print(f"\n  Regressions ({label_a}→{label_b}, lost correct):")
        for iid in disc[a_key][:10]:
            print(f"    - {iid}")
        if len(disc[a_key]) > 10:
            print(f"    ... and {len(disc[a_key])-10} more")
    if disc.get(b_key):
        print(f"\n  Improvements ({label_b} gained correct):")
        for iid in disc[b_key][:10]:
            print(f"    - {iid}")
        if len(disc[b_key]) > 10:
            print(f"    ... and {len(disc[b_key])-10} more")



_PA_RA_KEYS = (
    "gold_branch_precision", "gold_branch_recall", "gold_branch_f1",
    "buggy_branch_coverage", "healing_delta", "gold_path_recovery",
    "novel_path_rate", "jaccard_gold", "branch_discriminative",
    "correct_reraise_precision", "correct_reraise_recall", "correct_reraise_f1",
    "raise_introduction_rate", "gold_raise_preserved_rate",
    "first_attempt_correct_reraise_precision", "first_attempt_correct_reraise_recall",
    "first_attempt_correct_reraise_f1",
    "guard_reject_correct_reraise_precision", "guard_reject_correct_reraise_recall",
    "guard_reject_correct_reraise_f1",
    "guard_reject_correct_reraise_static_precision", "guard_reject_correct_reraise_static_recall",
    "guard_reject_correct_reraise_static_f1",
    "guard_reject_correct_reraise_rubric_precision", "guard_reject_correct_reraise_rubric_recall",
    "guard_reject_correct_reraise_rubric_f1",
    "exception_fidelity",
    "gold_only_branches", "buggy_only_branches", "shared_branches",
    "gold_only_raises", "buggy_only_raises", "degraded",
)

RAW_FIELDS = [
    "model_tag",
    "agent_class",
    "model",
    "guard_enabled",
    "instance_id",
    "repo",
    "error_type",
    "base_commit",
    "target_test",
    "status",
    "budget_exhausted",
    "is_timeout",
    "chain_len",
    "n_bash_explore",
    "crash_chain",
    "test_rc",
    "total_fix",
    "total_skip",
    "total_fix_attempts",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "bash_count",
    "run_time_s",
    "run_time_from_first_crash",
    "first_crash_wall_time_s",
    "instrumented_files",
    "instrumented_functions",
    "max_call_llm",
    "n_crashes",
    "n_unique_crash_locations",
    "n_fix_actions",
    "n_skip_actions",
    "n_probe_actions",
    "calls_fix",
    "calls_explore",
    "calls_skip",
    "tcalls_explore",
    "tcalls_readstate",
    "tcalls_heal",
    "tcalls_reraise",
    "n_suppress_fixes",
    "n_guard_rejects",
    "n_guard_skips",
    "n_guard_force_accepts",
    "n_guard_reject_caps",
    "n_probe_guard_rejected",
    "n_guard_accepts",
    "n_fix_failed",
    "n_fix_exec_failed",
    "n_repair_applied",
    "n_repair_replay",
    "n_repair_failed",
    "is_correct",
    "is_correct_strict",
    "last_fix_is_suppress",
    "n_suppress_fixes_ast",
    "is_proceed",
    "is_crash",
    "suppress_rate",
    "fix_success_rate",
    "skip_ratio",
    "reraise_rate",
    "guard_reject_rate",
    "guard_reraise_rate",
    "guard_force_accept_rate",
    "n_guard_static_rejects",
    "n_guard_rubric_rejects",
    "n_guard_rubric_sf_rejects",
    "n_guard_rubric_fc_rejects",
    "static_reject_rate",
    "rubric_reject_rate",
    "static_reject_share",
    "rubric_reject_share",
    "has_gold_trace",
    "has_healed_trace",
    "gold_branch_precision",
    "gold_branch_recall",
    "gold_branch_f1",
    "buggy_branch_coverage",
    "healing_delta",
    "gold_path_recovery",
    "novel_path_rate",
    "jaccard_gold",
    "branch_discriminative",
    "correct_reraise_precision",
    "correct_reraise_recall",
    "correct_reraise_f1",
    "raise_introduction_rate",
    "gold_raise_preserved_rate",
    "first_attempt_correct_reraise_precision",
    "first_attempt_correct_reraise_recall",
    "first_attempt_correct_reraise_f1",
    "guard_reject_correct_reraise_precision",
    "guard_reject_correct_reraise_recall",
    "guard_reject_correct_reraise_f1",
    "guard_reject_correct_reraise_static_precision",
    "guard_reject_correct_reraise_static_recall",
    "guard_reject_correct_reraise_static_f1",
    "guard_reject_correct_reraise_rubric_precision",
    "guard_reject_correct_reraise_rubric_recall",
    "guard_reject_correct_reraise_rubric_f1",
    "exception_fidelity",
    "gold_only_branches",
    "buggy_only_branches",
    "shared_branches",
    "gold_only_raises",
    "buggy_only_raises",
    "degraded",
    "baseline_step_limit",
    "baseline_cost_limit",
    "trace_path",
    "guard_reject_reraise_list",
    "guard_reject_reraise_list_static",
    "guard_reject_reraise_list_rubric",
    "first_attempt_skip_list",
]
FINAL_FIELDS = [
    "agent_class",
    "model",
    "guard_enabled",
    "n_cases",
    "correct_rate",
    "correct_rate_strict",
    "suppress_pass_rate",
    "proceed_rate",
    "avg_llm_calls",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_cost_usd",
    "avg_run_time_s",
    "avg_run_time_from_first_crash",
    "avg_crashes",
    "avg_unique_crash_locations",
    "avg_fix_actions",
    "avg_skip_actions",
    "avg_probe_actions",
    "avg_bash_count",
    "skip_ratio",
    "suppress_rate",
    "reraise_rate",
    "n_guard_rejects",
    "n_guard_static_rejects",
    "n_guard_rubric_rejects",
    "n_guard_rubric_sf_rejects",
    "n_guard_rubric_fc_rejects",
    "n_probe_guard_rejected",
    "n_guard_skips",
    "n_guard_force_accepts",
    "n_guard_accepts",
    "total_fix_attempts",
    "total_crashes",
    "guard_reject_rate",
    "static_reject_rate",
    "rubric_reject_rate",
    "static_reject_share",
    "rubric_reject_share",
    "guard_reraise_rate",
    "guard_force_accept_rate",
    "crash_rate",
    "fix_success_rate",
    "n_fix_exec_failed",
    "n_repair_applied",
    "n_repair_replay",
    "n_repair_failed",
    "n_with_trace",
    "n_with_healed",
    "avg_gold_branch_precision",
    "avg_gold_branch_recall",
    "avg_gold_branch_f1",
    "avg_buggy_branch_coverage",
    "avg_healing_delta",
    "avg_gold_path_recovery",
    "avg_novel_path_rate",
    "avg_jaccard_gold",
    "n_branch_discriminative",
    "avg_gold_branch_precision_disc",
    "avg_gold_branch_recall_disc",
    "avg_gold_branch_f1_disc",
    "avg_gold_path_recovery_disc",
    "avg_novel_path_rate_disc",
    "avg_correct_reraise_precision",
    "avg_correct_reraise_recall",
    "avg_correct_reraise_f1",
    "avg_raise_introduction_rate",
    "avg_gold_raise_preserved_rate",
    "avg_first_attempt_correct_reraise_precision",
    "avg_first_attempt_correct_reraise_recall",
    "avg_first_attempt_correct_reraise_f1",
    "avg_guard_reject_correct_reraise_precision",
    "avg_guard_reject_correct_reraise_recall",
    "avg_guard_reject_correct_reraise_f1",
    "avg_guard_reject_correct_reraise_static_precision",
    "avg_guard_reject_correct_reraise_static_recall",
    "avg_guard_reject_correct_reraise_static_f1",
    "avg_guard_reject_correct_reraise_rubric_precision",
    "avg_guard_reject_correct_reraise_rubric_recall",
    "avg_guard_reject_correct_reraise_rubric_f1",
    "avg_exception_fidelity",
    "n_degraded",
    "avg_gold_only_branches",
    "avg_buggy_only_branches",
    "avg_shared_branches",
    "avg_gold_only_raises",
    "avg_buggy_only_raises",
]


def _parse_tag(tag: str) -> dict:
    parts = tag.split("__")
    model = parts[0]
    suffix_tokens = parts[1:] if len(parts) > 1 else []
    guard_enabled = any(t.startswith("guard") for t in suffix_tokens)
    if any(t in ("noguard", "baseline_noguard") for t in suffix_tokens):
        guard_enabled = False
    return {"model": model, "tag_suffix": "__".join(suffix_tokens), "guard_enabled": guard_enabled}


_SUPPRESS_PATTERNS = frozenset(
    {
        "_eh_rtn = True",
        "_eh_suppress = True",
        "_eh_rtn = None",
        "pass",
        "_eh_rtn = 0",
        '_eh_rtn = ""',
        "_eh_rtn = ''",
        "_eh_rtn = False",
        "_eh_rtn = []",
        "_eh_rtn = {}",
    }
)


def _is_semantic_suppress(code: str | None) -> bool:
    if not code or not str(code).strip():
        return False
    try:
        tree = ast.parse(str(code).strip())
    except SyntaxError:
        return False
    if not tree.body:
        return False
    for node in tree.body:
        if isinstance(node, ast.Pass):
            continue
        if isinstance(node, ast.Assign) and all(
            isinstance(t, ast.Name) and t.id in ("_eh_rtn", "_eh_suppress") for t in node.targets
        ):
            v = node.value
            if isinstance(v, ast.Constant):
                continue
            if isinstance(v, (ast.List, ast.Tuple, ast.Dict, ast.Set)) and not (
                getattr(v, "elts", None) or getattr(v, "keys", None)
            ):
                continue
            if isinstance(v, ast.UnaryOp) and isinstance(v.operand, ast.Constant):
                continue
            return False
        return False
    return True


def _summarize_trace(trace_path: Path, agent_class: str | None = None) -> dict:
    out = {
        "n_crashes": 0,
        "n_unique_crash_locations": 0,
        "n_fix_actions": 0,
        "n_skip_actions": 0,
        "n_probe_actions": 0,
        "calls_fix": 0,
        "calls_explore": 0,
        "calls_skip": 0,
        "tcalls_explore": 0,
        "tcalls_readstate": 0,
        "tcalls_heal": 0,
        "tcalls_reraise": 0,
        "n_suppress_fixes": 0,
        "n_suppress_fixes_ast": 0,
        "last_fix_is_suppress": False,
        "n_guard_rejects": 0,
        "n_guard_static_rejects": 0,
        "n_guard_rubric_rejects": 0,
        "n_guard_rubric_sf_rejects": 0,
        "n_guard_rubric_fc_rejects": 0,
        "n_guard_skips": 0,
        "n_guard_accepts": 0,
        "n_guard_force_accepts": 0,
        "n_guard_reject_caps": 0,
        "n_probe_guard_rejected": 0,
        "first_crash_wall_time_s": None,
        "n_fix_exec_failed": 0,
        "baseline_step_limit": "",
        "baseline_cost_limit": "",
        "guard_reject_reraise_list": [],
        "guard_reject_reraise_list_static": [],
        "guard_reject_reraise_list_rubric": [],
        "first_attempt_skip_list": [],
        "trace_input_tokens": 0,
        "trace_output_tokens": 0,
    }
    _MAX_TRACE_BYTES = 4 * 1024 * 1024 * 1024
    try:
        _sz = trace_path.stat().st_size
    except OSError as error:
        logger.warning(f"Cannot stat trace {trace_path}: {error}")
        _sz = 0
    if _sz > _MAX_TRACE_BYTES:
        logger.warning(
            f"trace too large to parse ({_sz / 1e9:.1f} GB > 4 GB), skipping trace metrics: {trace_path}"
        )
        return out
    if trace_path.name == "run_docker.log":
        events = [event for event in _read_heal_trace(trace_path, "agent_event") if isinstance(event, dict)]
        bc = next((event.get("baseline_config", {}) for event in events if "baseline_config" in event), {})
        out["baseline_step_limit"] = bc.get("step_limit", "")
        out["baseline_cost_limit"] = bc.get("cost_limit", "")
    else:
        try:
            data = json.loads(trace_path.read_text("utf-8"))
        except Exception as e:
            logger.warning(f"trace unreadable {trace_path}: {e}")
            return out
        if isinstance(data, dict):
            events = data.get("events", [])
            bc = data.get("baseline_config", {}) or {}
            out["baseline_step_limit"] = bc.get("step_limit", "")
            out["baseline_cost_limit"] = bc.get("cost_limit", "")
        else:
            events = data

    crash_ids: set = set()
    crash_locations: set = set()
    crash_after_fix: set = set()
    _last_fix_loc: set = set()
    crash_chain: list[dict] = []
    _chain_idx_by_num: dict = {}
    _pending_guard_reject: dict = {}
    _crash_location_seen: dict = {}
    _prev_cum_calls = 0
    for i, ev in enumerate(events):
        cid = ev.get("crash_num", ev.get("repair"))
        if cid is not None:
            crash_ids.add(cid)
            _wt = ev.get("wall_time_s")
            if _wt is not None and out.get("first_crash_wall_time_s") is None:
                out["first_crash_wall_time_s"] = _wt
        fp = ev.get("file_path") or ""
        bi = ev.get("block_index", -1)
        error_type = ev.get("error_type", "")
        error_message = ev.get("error_message", "")
        if fp and bi is not None and bi != -1:
            crash_locations.add((fp, bi))
        action = str(ev.get("action") or "").lower()
        _cum = ev.get("llm_calls")
        if isinstance(_cum, (int, float)) and int(_cum) > _prev_cum_calls:
            _inc = int(_cum) - _prev_cum_calls
            _prev_cum_calls = int(_cum)
            if _inc:
                if action == "execute" and not bool(ev.get("submit", False)):
                    out["calls_explore"] += _inc
                elif action == "reraise":
                    out["calls_skip"] += _inc
                elif action == "execute" and bool(ev.get("submit", False)) or action.startswith("guard"):
                    out["calls_fix"] += _inc
                else:
                    out["calls_explore"] += _inc
        if cid is not None and action != "execute_result":
            if cid not in _chain_idx_by_num:
                if crash_chain:
                    crash_chain[-1]["healed"] = True
                _chain_idx_by_num[cid] = len(crash_chain)
                crash_chain.append({
                    "n": cid,
                    "error_type": error_type,
                    "file": fp,
                    "func": ev.get("func_name", "") or "",
                    "action": action,
                    "healed": None,
                })
        loc = (fp, bi) if fp and bi is not None and bi != -1 else None
        loc_with_type = (fp, bi, error_type) if fp and bi is not None and bi != -1 and error_type else None
        fix_fp = (fp, bi, error_type, error_message) if loc else None
        if fix_fp and fix_fp in _last_fix_loc and not (action == "execute" and bool(ev.get("submit", False))):
            if cid is not None:
                crash_after_fix.add(fix_fp)
                _last_fix_loc.discard(fix_fp)
        if ev.get("exec_error"):
            out["n_fix_exec_failed"] = out.get("n_fix_exec_failed", 0) + 1
        if (action == "execute" and bool(ev.get("submit", False))) or ev.get("code") and bool(ev.get("submit", False)):
            out["n_fix_actions"] += 1
            _fc = ev.get("code") or ""
            if _fc.strip() in _SUPPRESS_PATTERNS:
                out["n_suppress_fixes"] += 1
            _is_sup = _is_semantic_suppress(_fc)
            if _is_sup:
                out["n_suppress_fixes_ast"] += 1
            out["last_fix_is_suppress"] = _is_sup
            if fix_fp:
                _last_fix_loc.add(fix_fp)
            if loc_with_type and loc_with_type not in _crash_location_seen:
                _crash_location_seen[loc_with_type] = "fix"
        elif action == "reraise" or ev.get("reason") and action == "reraise":
            out["n_skip_actions"] += 1
            if loc_with_type and loc_with_type not in _crash_location_seen:
                _crash_location_seen[loc_with_type] = "skip"
                out["first_attempt_skip_list"].append(error_type)
        elif action == "guard_reject":
            out["n_guard_rejects"] += 1
            gr = ev.get("guard_result")
            reject_layer = ""
            if isinstance(gr, dict) and gr.get("static_passed") is not None:
                if gr.get("static_passed") is False:
                    out["n_guard_static_rejects"] += 1
                    reject_layer = "static"
                else:
                    out["n_guard_rubric_rejects"] += 1
                    reject_layer = "rubric"
                    _rr = str(gr.get("reject_reason") or "")
                    if _rr == "SF_VETO":
                        out["n_guard_rubric_sf_rejects"] += 1
                    elif _rr == "FC_VETO":
                        out["n_guard_rubric_fc_rejects"] += 1
            else:
                reason = str(ev.get("reason") or "")
                if re.match(r"\[RH\d+\]", reason):
                    out["n_guard_static_rejects"] += 1
                    reject_layer = "static"
                elif "rubric" in reason.lower():
                    out["n_guard_rubric_rejects"] += 1
                    reject_layer = "rubric"
            error_type = ev.get("error_type", "")
            if error_type:
                _pending_guard_reject[i] = (error_type, reject_layer)
        elif action == "guard_accept":
            out["n_guard_accepts"] += 1
        elif action == "guard_force_accept":
            out["n_guard_force_accepts"] += 1
        elif action == "guard_reject_cap":
            out["n_guard_reject_caps"] = out.get("n_guard_reject_caps", 0) + 1
            error_type = ev.get("error_type", "")
            if error_type:
                out["guard_reject_reraise_list"].append(error_type)
                _cap_gr = ev.get("guard_result")
                if isinstance(_cap_gr, dict) and _cap_gr.get("static_passed") is not None:
                    if _cap_gr.get("static_passed") is False:
                        out["guard_reject_reraise_list_static"].append(error_type)
                    else:
                        out["guard_reject_reraise_list_rubric"].append(error_type)
        elif action == "probe_guard_rejected":
            out["n_probe_guard_rejected"] += 1
        elif action == "guard_skip":
            out["n_guard_skips"] += 1
            for j in range(i - 1, max(i - 10, -1), -1):
                if j in _pending_guard_reject:
                    error_type, reject_layer = _pending_guard_reject[j]
                    out["guard_reject_reraise_list"].append(error_type)
                    if reject_layer == "static":
                        out["guard_reject_reraise_list_static"].append(error_type)
                    elif reject_layer == "rubric":
                        out["guard_reject_reraise_list_rubric"].append(error_type)
                    del _pending_guard_reject[j]
                    break
        if action != "execute_result" and (
            action == "execute" and not bool(ev.get("submit", False))
        ):
            out["n_probe_actions"] += 1
            if loc_with_type and loc_with_type not in _crash_location_seen:
                _crash_location_seen[loc_with_type] = "probe"
    out["n_crashes"] = len(crash_ids)
    out["n_unique_crash_locations"] = len(crash_locations)
    out["n_fix_failed"] = len(crash_after_fix)

    if crash_chain:
        last = crash_chain[-1]
        if crash_after_fix:
            _recrashed_files = {t[0] for t in crash_after_fix}
            if last.get("file") in _recrashed_files:
                last["healed"] = False
    out["crash_chain"] = crash_chain
    out["chain_len"] = len(crash_chain)

    _n_bash_explore = 0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        for call in ev.get("llm_calls_detail", []) or []:
            ro = (call or {}).get("raw_output")
            if not isinstance(ro, dict):
                continue
            for ch in ro.get("choices", []) or []:
                txt = ((ch or {}).get("message", {}) or {}).get("content", "") or ""
                if isinstance(txt, str) and re.search(r"(?im)^Action:\s*bash\b", txt):
                    _n_bash_explore += 1
    out["n_bash_explore"] = _n_bash_explore
    if _n_bash_explore:
        _moved = min(_n_bash_explore, out["calls_fix"])
        out["calls_fix"] -= _moved
        out["calls_explore"] += _moved
        if _moved < _n_bash_explore:
            logger.info(f"bash_explore {_n_bash_explore} > calls_fix pool; capped move to {_moved} (conservation preserved)")

    _EXPLORE_TOOLS = {"bash", "terminal", "read", "grep", "glob", "file_editor"}
    _READ_SRC_RE = re.compile(r"open\(|readlines\(|\.read\(\)|read_text\(|subprocess|os\.system|Path\(")

    def _classify_tool(name: str, code: str, submit: bool = False) -> str | None:
        n = (name or "").split("__")[-1].lower()
        if n in _EXPLORE_TOOLS:
            return "explore"
        if n == "execute":
            if submit:
                return "heal"
            return "explore" if (code and _READ_SRC_RE.search(code)) else "readstate"
        if n == "reraise":
            return "reraise"
        return None

    _tc = {"explore": 0, "readstate": 0, "heal": 0, "reraise": 0}
    _seen_provider_actions = set()
    _codex_trace = False
    _best_snapshot = None
    _best_len = -1
    for ev in events:
        if not isinstance(ev, dict):
            continue
        for call in ev.get("llm_calls_detail", []) or []:
            ro = (call or {}).get("raw_output")
            if not isinstance(ro, dict):
                continue
            if "thread_id" in ro and "items" in ro:
                _codex_trace = True
                for item_index, item in enumerate(ro.get("items") or []):
                    if item.get("type") != "commandExecution":
                        continue
                    key = (ro.get("thread_id"), item.get("id") or (ro.get("turn_id"), item_index))
                    if key not in _seen_provider_actions:
                        _seen_provider_actions.add(key)
                        _tc["explore"] += 1
            if ro.get("kind") == "ActionEvent":
                key = ro.get("id") or ro.get("tool_call_id")
                if key is None or key not in _seen_provider_actions:
                    if key is not None:
                        _seen_provider_actions.add(key)
                    action_data = ro.get("action") or {}
                    cls = _classify_tool(ro.get("tool_name") or "", action_data.get("code", ""),
                                         bool(action_data.get("submit", False)))
                    if cls:
                        _tc[cls] += 1
            for ch in ro.get("choices", []) or []:
                msg = ((ch or {}).get("message") or {})
                for tc in msg.get("tool_calls") or []:
                    fn = ((tc or {}).get("function") or {})
                    args = fn.get("arguments")
                    parsed_args = {}
                    if isinstance(args, str):
                        try:
                            parsed_args = json.loads(args) or {}
                        except Exception as e:
                            logger.warning(f"tool_calls arguments not JSON in {trace_path}: {e}")
                            parsed_args = {"code": args}
                    elif isinstance(args, dict):
                        parsed_args = args
                    code = parsed_args.get("code", "") or ""
                    cls = _classify_tool(
                        fn.get("name") or "", code, bool(parsed_args.get("submit", False))
                    )
                    if cls:
                        _tc[cls] += 1
        ms = ev.get("messages_snapshot")
        if isinstance(ms, list) and len(ms) > _best_len:
            _best_len = len(ms)
            _best_snapshot = ms
    for m in _best_snapshot or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role == "openhands_event" and isinstance(content, dict):
            if content.get("kind") == "ActionEvent":
                key = content.get("id") or content.get("tool_call_id")
                if key is not None and key in _seen_provider_actions:
                    continue
                if key is not None:
                    _seen_provider_actions.add(key)
                act = content.get("action")
                code = act.get("code", "") or "" if isinstance(act, dict) else ""
                submit = bool(act.get("submit", False)) if isinstance(act, dict) else False
                cls = _classify_tool(content.get("tool_name") or "", code, submit)
                if cls:
                    _tc[cls] += 1
        elif role == "assistant":
            blocks = content.get("content") if isinstance(content, dict) else content
            if isinstance(blocks, list):
                for blk in blocks:
                    if not (isinstance(blk, dict) and "name" in blk):
                        continue
                    tool_input = blk.get("input") if isinstance(blk.get("input"), dict) else {}
                    code = tool_input.get("code", "") or ""
                    cls = _classify_tool(
                        blk.get("name") or "", code, bool(tool_input.get("submit", False))
                    )
                    if cls:
                        _tc[cls] += 1
    if _codex_trace:
        # Codex returns runtime actions as final JSON, not provider tool calls.
        for event in events:
            cls = _classify_tool(str(event.get("action") or ""), event.get("code") or "",
                                 bool(event.get("submit", False)))
            if cls:
                _tc[cls] += 1
    if _tc["heal"] == 0:
        _tc["heal"] = out["n_fix_actions"]
    _tc["reraise"] = max(_tc["reraise"], out["n_skip_actions"])
    if agent_class == "Healer":
        # Provider requests can be rejected by Healer.step; count accepted adapter actions only.
        _tc = {"explore": 0, "readstate": 0, "heal": 0, "reraise": 0}
        for event in events:
            action = event.get("action")
            if action == "execute":
                cls = _classify_tool(action, event.get("code") or "", bool(event.get("submit", False)))
                _tc[cls] += 1
            elif action == "reraise" and event.get("reason") != "no_crash":
                _tc["reraise"] += 1
    out["tcalls_explore"] = _tc["explore"]
    out["tcalls_readstate"] = _tc["readstate"]
    out["tcalls_heal"] = _tc["heal"]
    out["tcalls_reraise"] = _tc["reraise"]
    logger.info(
        f"tool-call split {trace_path.parent.name}: explore={_tc['explore']} readstate={_tc['readstate']} "
        f"heal={_tc['heal']} reraise={_tc['reraise']}"
    )

    t_in = t_out = 0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        for call in ev.get("llm_calls_detail", []) or []:
            ro = (call or {}).get("raw_output")
            if not isinstance(ro, dict):
                continue
            u = ro.get("usage")
            if not isinstance(u, dict):
                continue
            t_in += int(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0)
            t_out += int(u.get("completion_tokens", u.get("output_tokens", 0)) or 0)
    out["trace_input_tokens"] = t_in
    out["trace_output_tokens"] = t_out
    return out


def _collect_one(task_dir: Path, model_tag: str, exclude_leaked: bool = True) -> dict | None:
    if task_dir.name not in HEALBENCH_INSTANCE_IDS:
        logger.info(f"excluding outside-manifest instance (blacklist): {task_dir.name}")
        return None
    out_file = task_dir / "output.json"
    trace_file = task_dir / "run_docker.log"
    if not out_file.exists():
        return None
    try:
        out = json.loads(out_file.read_text("utf-8"))
    except Exception as e:
        logger.warning(f"output.json unreadable {out_file}: {e}")
        return None

    raw_status = out.get("status", "fail")
    docker_log_f = task_dir / "run_docker.log"
    status = classify_status(raw_status, docker_log_f, out.get("test_rc"))
    if status == "infra_error":
        return None

    inp_file = task_dir / "input.json"
    test_code_leaked = False
    inst_repo = inst_error_type = inst_commit = inst_target_test = ""
    if inp_file.exists():
        try:
            inp = json.loads(inp_file.read_text("utf-8"))
            test_file = inp.get("test_file", "") or ""
            inst_repo = inp.get("repo", "") or ""
            inst_error_type = inp.get("error_type", "") or ""
            inst_commit = inp.get("base_commit", "") or inp.get("commit", "") or ""
            _ftp = inp.get("fail_to_pass", "") or inp.get("FAIL_TO_PASS", "")
            if isinstance(_ftp, list):
                _ftp = _ftp[0] if _ftp else ""
            inst_target_test = str(_ftp) or test_file
            test_code_leaked = bool(
                test_file
                and "../r2e_tests" not in test_file
                and inp.get("test_code_in_prompt", False)
            )
        except Exception as _e:
            logger.warning(f"test_code_leaked check failed for {inp_file}: {_e}")
    if exclude_leaked and test_code_leaked:
        return None

    tag_info = _parse_tag(model_tag)
    trace_info = (
        _summarize_trace(trace_file)
        if trace_file.exists()
        else {
            "n_crashes": 0,
            "n_unique_crash_locations": 0,
            "n_fix_actions": 0,
            "n_skip_actions": 0,
            "n_probe_actions": 0,
            "calls_fix": 0,
            "calls_explore": 0,
            "calls_skip": 0,
            "tcalls_explore": 0,
            "tcalls_readstate": 0,
            "tcalls_heal": 0,
            "tcalls_reraise": 0,
            "n_suppress_fixes": 0,
            "baseline_step_limit": "",
            "baseline_cost_limit": "",
            "crash_chain": [],
            "chain_len": 0,
            "n_bash_explore": 0,
        }
    )

    _chain = trace_info.pop("crash_chain", []) or []
    if _chain and _chain[-1].get("healed") is None:
        _chain[-1]["healed"] = status in ("correct", "proceed")
    trace_info["crash_chain"] = json.dumps(_chain, default=str)

    _rr = out.get("repair_records", []) or []
    _rr_applied = sum(1 for r in _rr if isinstance(r, dict) and r.get("status") == "applied")
    _rr_replay = sum(1 for r in _rr if isinstance(r, dict) and r.get("status") == "replay")
    _rr_failed = sum(1 for r in _rr if isinstance(r, dict) and r.get("status") == "failed")

    _force_accepts = out.get("total_guard_force_accepts", 0)
    _trace_accepts = trace_info.get("n_guard_accepts", 0)
    if _trace_accepts:
        n_guard_accepts = _trace_accepts
    else:
        n_guard_accepts = max(0, out.get("total_fix", 0) - _force_accepts)
    n_fix_attempts = out.get("total_fix_attempts", 0)

    n_fix = trace_info.get("n_fix_actions", 0)
    n_skip = trace_info.get("n_skip_actions", 0)
    n_suppress = trace_info.get("n_suppress_fixes", 0)
    n_fix_failed = trace_info.get("n_fix_failed", 0)
    n_crashes_i = trace_info.get("n_crashes", 0)
    n_guard_rej = trace_info.get("n_guard_rejects", 0)
    n_guard_skp = trace_info.get("n_guard_skips", 0)
    n_guard_fa = trace_info.get("n_guard_force_accepts", 0)
    n_guard_static = trace_info.get("n_guard_static_rejects", 0)
    n_guard_rubric = trace_info.get("n_guard_rubric_rejects", 0)

    def _ratio(num, den):
        return round(num / den, 4) if den else None

    _pa_ra = {k: None for k in _PA_RA_KEYS}
    _pa_ra["has_healed_trace"] = 0
    _pa_ra["has_gold_trace"] = 0
    iid_for_gold = out.get("instance_id", task_dir.name)
    gt_data = load_gold_trace(iid_for_gold)
    if gt_data is not None and gt_data.get("gold_test_rc", -1) == 0:
        _pa_ra["has_gold_trace"] = 1
        healed_trace = _read_healed_trace(task_dir / "run_docker.log")
        patch_text = ""
        if inp_file.exists():
            try:
                patch_text = json.loads(inp_file.read_text("utf-8")).get("patch", "")
            except Exception as e:
                logger.warning(f"input.json patch read failed {inp_file}: {e}")
        try:
            _tm = compute_trace_metrics(
                gt_data.get("gold_trace", {}), gt_data.get("buggy_trace", {}),
                healed_trace=healed_trace, patch_text=patch_text,
                guard_reject_reraise_list=trace_info.get("guard_reject_reraise_list", []),
                first_attempt_skip_list=trace_info.get("first_attempt_skip_list", []),
                guard_reject_reraise_list_static=trace_info.get("guard_reject_reraise_list_static", []),
                guard_reject_reraise_list_rubric=trace_info.get("guard_reject_reraise_list_rubric", []),
            )
            for k in _PA_RA_KEYS:
                if k in _tm:
                    _pa_ra[k] = _tm[k]
            _pa_ra["has_healed_trace"] = 1 if healed_trace is not None else 0
        except Exception as e:
            logger.warning(f"per-instance trace metrics failed for {iid_for_gold}: {e}")

    row = {
        "model_tag": model_tag,
        "agent_class": out.get("strategy", ""),
        "model": tag_info["model"],
        "guard_enabled": int(tag_info["guard_enabled"]),
        "instance_id": out.get("instance_id", task_dir.name),
        "repo": inst_repo,
        "error_type": inst_error_type,
        "base_commit": inst_commit,
        "target_test": inst_target_test,
        "status": status,
        "budget_exhausted": 1 if (out.get("max_call_llm") and out.get("llm_calls", 0) >= out.get("max_call_llm", 0)) else 0,
        "is_timeout": 1 if (status == "crash" and str(out.get("test_rc")) == "-1") else 0,
        "is_correct": 1 if status == "correct" else 0,
        "is_correct_strict": 1 if (status == "correct" and not trace_info.get("last_fix_is_suppress", False)) else 0,
        "last_fix_is_suppress": 1 if trace_info.get("last_fix_is_suppress", False) else 0,
        "n_suppress_fixes_ast": trace_info.get("n_suppress_fixes_ast", 0),
        "is_proceed": 1 if status in ("correct", "proceed") else 0,
        "is_crash": 1 if status == "crash" else 0,
        "test_rc": out.get("test_rc", out.get("ftp_rc", "")),
        "total_fix": out.get("total_fix", 0),
        "total_fix_attempts": n_fix_attempts,
        "n_guard_accepts": n_guard_accepts,
        "total_skip": out.get("total_skip", 0),
        "n_repair_applied": _rr_applied,
        "n_repair_replay": _rr_replay,
        "n_repair_failed": _rr_failed,
        "llm_calls": out.get("llm_calls", 0),
        "input_tokens": out.get("input_tokens", 0) or trace_info.get("trace_input_tokens", 0),
        "output_tokens": out.get("output_tokens", 0) or trace_info.get("trace_output_tokens", 0),
        "bash_count": out.get("bash_count", 0),
        "run_time_s": out.get("run_time_s", ""),
        "run_time_from_first_crash": (
            round(out["run_time_s"] - trace_info["first_crash_wall_time_s"], 2)
            if isinstance(out.get("run_time_s"), (int, float))
            and isinstance(trace_info.get("first_crash_wall_time_s"), (int, float))
            else None
        ),
        "instrumented_files": out.get("instrumented_files", 0),
        "instrumented_functions": out.get("instrumented_functions", 0),
        "max_call_llm": out.get("max_call_llm", ""),
        "trace_path": str(trace_file) if trace_file.exists() else "",
        "suppress_rate": _ratio(n_suppress, n_fix),
        "fix_success_rate": (round(1.0 - n_fix_failed / n_fix, 4) if n_fix else None),
        "skip_ratio": _ratio(n_skip, n_fix + n_skip),
        "reraise_rate": _ratio(n_skip, n_crashes_i),
        "guard_reject_rate": _ratio(n_guard_rej, n_guard_accepts + n_guard_rej),
        "guard_reraise_rate": _ratio(n_guard_skp, n_guard_rej),
        "guard_force_accept_rate": _ratio(n_guard_fa, n_guard_rej),
        "static_reject_rate": _ratio(n_guard_static, n_guard_accepts + n_guard_rej),
        "rubric_reject_rate": _ratio(n_guard_rubric, n_guard_accepts + n_guard_rej),
        "static_reject_share": _ratio(n_guard_static, n_guard_rej),
        "rubric_reject_share": _ratio(n_guard_rubric, n_guard_rej),
        **{k: v for k, v in trace_info.items() if k != "n_guard_accepts"},
        **_pa_ra,
    }
    return row


_EVAL_CACHE_DIR = HEALBENCH_RESULTS.parent / ".eval_cache"


_EVAL_ROW_SCHEMA_VERSION = "v15"


def _cache_key(task_dir: Path, model_tag: str) -> str | None:
    out_file = task_dir / "output.json"
    if not out_file.exists():
        return None
    parts = [_EVAL_ROW_SCHEMA_VERSION, model_tag, str(task_dir)]
    for fname in ("output.json", "input.json", "run.log", "run_docker.log"):
        f = task_dir / fname
        try:
            st = f.stat()
            parts.append(f"{fname}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{fname}:absent")
    parts.append(f"gold_dir:{GOLD_TRACES_DIR}")
    gold_paths = [GOLD_TRACES_DIR / f"{task_dir.name}.json"]
    gold_paths += [GOLD_TRACES_DIR / task_dir.name / name for name in ("output.json", "gold.log", "buggy.log")]
    gold_paths += [Path(__file__), Path(__file__).with_name("run_record_gold_trace.py")]
    for gold_f in gold_paths:
        if gold_f.is_file():
            gst = gold_f.stat()
            parts.append(f"{gold_f}:{gst.st_size}:{gst.st_mtime_ns}")
        else:
            parts.append(f"{gold_f}:absent")
    return hashlib.sha256("||".join(parts).encode("utf-8", "replace")).hexdigest()[:32]


def _collect_one_cached(task_dir: Path, model_tag: str, exclude_leaked: bool = True) -> dict | None:
    if task_dir.name not in HEALBENCH_INSTANCE_IDS:
        logger.info(f"excluding outside-manifest instance (blacklist): {task_dir.name}")
        return None
    key = _cache_key(task_dir, model_tag)
    if key is None:
        return None
    cache_f = _EVAL_CACHE_DIR / f"{key}.json"
    if cache_f.exists():
        try:
            cached = json.loads(cache_f.read_text("utf-8"))
            return cached.get("row")
        except Exception as e:
            logger.warning(f"eval cache read failed {cache_f} ({e}); recomputing")
    row = _collect_one(task_dir, model_tag, exclude_leaked=exclude_leaked)
    try:
        _EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_f.write_text(json.dumps({"row": row}, default=str), "utf-8")
    except Exception as e:
        logger.warning(f"eval cache write failed {cache_f} ({e})")
    return row


def _is_backup_dir(name: str) -> bool:
    n = name.lower()
    return (name.startswith("_") or name == "rubrics" or n.endswith("bak")
            or "backup" in n or "stale" in n or "rerun_bak" in n)


def _collect_all(results_root: Path) -> list[dict]:
    jobs: list[tuple[Path, str, str]] = []
    for method_dir in sorted(results_root.iterdir()):
        if not method_dir.is_dir() or method_dir.name.startswith("."):
            continue
        _name = method_dir.name
        if _is_backup_dir(_name):
            logger.info(f"[skip non-result dir] {_name}")
            continue
        method = _name

        children = [c for c in method_dir.iterdir() if c.is_dir()]
        if not children:
            continue
        is_legacy = any((c / "output.json").exists() for c in children[:3])

        if is_legacy:
            for task_dir in sorted(method_dir.iterdir()):
                if not task_dir.is_dir() or _is_backup_dir(task_dir.name):
                    continue
                jobs.append((task_dir, method, method))
        else:
            guard_match = re.search(r"(__guard\w*)$", method)
            guard_suffix = guard_match.group(1) if guard_match else ""
            method_base = method[: -len(guard_suffix)] if guard_suffix else method
            for model_dir in sorted(method_dir.iterdir()):
                if not model_dir.is_dir() or model_dir.name.startswith("."):
                    continue
                if _is_backup_dir(model_dir.name):
                    logger.info(f"[skip nested backup] {method}/{model_dir.name}")
                    continue
                model = model_dir.name
                cli_map = {
                    "ReAct": "",
                    "MiniSweAgentAdapter": "__mini-swe-agent",
                    "SWEAgentAdapter": "__SWE-agent",
                    "OpenHandsAdapter": "__openhands",
                    "ClaudeCodeAdapter": "__claude-code",
                }
                suffix = cli_map.get(method_base, f"__{method_base}")
                model_tag = model + suffix + guard_suffix
                for task_dir in sorted(model_dir.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    if _is_backup_dir(task_dir.name):
                        continue
                    jobs.append((task_dir, model_tag, method))

    n_workers = max(1, min(8, (os.cpu_count() or 4) - 2))
    rows: list[dict] = []
    per_method: dict[str, int] = defaultdict(int)
    worker_failures: list[str] = []
    if n_workers == 1 or len(jobs) <= 1:
        results = [(_collect_one_cached(td, mt), m) for td, mt, m in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            fut_to_job = {pool.submit(_collect_one_cached, td, mt): (td, mt, m) for td, mt, m in jobs}
            for fut in as_completed(fut_to_job):
                td, mt, m = fut_to_job[fut]
                try:
                    results.append((fut.result(), m))
                except Exception as e:
                    logger.error(f"[{m}] worker crashed on {td}: {e}")
                    worker_failures.append(str(td))
                    results.append((None, m))
    for row, method in results:
        if row is not None:
            rows.append(row)
            per_method[method] += 1
    for method in sorted(per_method):
        logger.info(f"[{method}] collected {per_method[method]} rows")
    if worker_failures:
        logger.error(
            f"{len(worker_failures)} instance(s) dropped by worker crashes (NOT in the table): "
            + ", ".join(worker_failures)
        )
    return rows


def _agg_group(group: pd.DataFrame) -> pd.Series:
    n = len(group)

    def _macro(col):
        if col not in group:
            return None
        v = pd.to_numeric(group[col], errors="coerce").mean()
        return round(v, 4) if pd.notna(v) else None

    s = pd.Series(
        {
            "n_cases": n,
            "correct_rate": _macro("is_correct"),
            "correct_rate_strict": _macro("is_correct_strict"),
            "suppress_pass_rate": _macro("last_fix_is_suppress"),
            "proceed_rate": _macro("is_proceed"),
            "crash_rate": _macro("is_crash"),
            "avg_llm_calls": round(group["llm_calls"].mean(), 3),
            "avg_input_tokens": round(group["input_tokens"].mean(), 1),
            "avg_output_tokens": round(group["output_tokens"].mean(), 1),
            "avg_cost_usd": _macro("cost_usd"),
            "avg_run_time_s": round(pd.to_numeric(group["run_time_s"], errors="coerce").mean(), 3),
            "avg_run_time_from_first_crash": _macro("run_time_from_first_crash"),
            "avg_crashes": round(group["n_crashes"].mean(), 3),
            "avg_unique_crash_locations": round(group["n_unique_crash_locations"].mean(), 3),
            "avg_fix_actions": round(group["n_fix_actions"].mean(), 3),
            "avg_skip_actions": round(group["n_skip_actions"].mean(), 3),
            "avg_probe_actions": round(group["n_probe_actions"].mean(), 3),
            "avg_bash_count": round(pd.to_numeric(group.get("bash_count", pd.Series(dtype=float)), errors="coerce").fillna(0).mean(), 3) if "bash_count" in group else 0.0,
            "skip_ratio": _macro("skip_ratio"),
            "suppress_rate": _macro("suppress_rate"),
            "reraise_rate": _macro("reraise_rate"),
            "fix_success_rate": _macro("fix_success_rate"),
            "n_guard_rejects": int(group["n_guard_rejects"].sum()) if "n_guard_rejects" in group else 0,
            "n_guard_skips": int(group["n_guard_skips"].sum()) if "n_guard_skips" in group else 0,
            "n_guard_force_accepts": int(group["n_guard_force_accepts"].sum()) if "n_guard_force_accepts" in group else 0,
            "n_guard_accepts": int(group["n_guard_accepts"].sum()) if "n_guard_accepts" in group else 0,
            "total_fix_attempts": int(group["total_fix_attempts"].sum()) if "total_fix_attempts" in group else 0,
            "total_crashes": int(group["n_crashes"].sum()),
            "n_guard_static_rejects": int(group["n_guard_static_rejects"].sum()) if "n_guard_static_rejects" in group else 0,
            "n_guard_rubric_rejects": int(group["n_guard_rubric_rejects"].sum()) if "n_guard_rubric_rejects" in group else 0,
            "n_guard_rubric_sf_rejects": int(group["n_guard_rubric_sf_rejects"].sum()) if "n_guard_rubric_sf_rejects" in group else 0,
            "n_guard_rubric_fc_rejects": int(group["n_guard_rubric_fc_rejects"].sum()) if "n_guard_rubric_fc_rejects" in group else 0,
            "n_probe_guard_rejected": int(group["n_probe_guard_rejected"].sum()) if "n_probe_guard_rejected" in group else 0,
            "guard_reject_rate": _macro("guard_reject_rate"),
            "static_reject_rate": _macro("static_reject_rate"),
            "rubric_reject_rate": _macro("rubric_reject_rate"),
            "static_reject_share": _macro("static_reject_share"),
            "rubric_reject_share": _macro("rubric_reject_share"),
            "guard_reraise_rate": _macro("guard_reraise_rate"),
            "guard_force_accept_rate": _macro("guard_force_accept_rate"),
            "n_fix_exec_failed": int(group["n_fix_exec_failed"].sum()) if "n_fix_exec_failed" in group else 0,
            "n_repair_applied": int(group["n_repair_applied"].sum()) if "n_repair_applied" in group else 0,
            "n_repair_replay": int(group["n_repair_replay"].sum()) if "n_repair_replay" in group else 0,
            "n_repair_failed": int(group["n_repair_failed"].sum()) if "n_repair_failed" in group else 0,
            "n_with_trace": int(pd.to_numeric(group.get("has_gold_trace", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "n_with_healed": int(pd.to_numeric(group.get("has_healed_trace", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "n_degraded": int(pd.to_numeric(group.get("degraded", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "avg_gold_branch_precision": _macro("gold_branch_precision"),
            "avg_gold_branch_recall": _macro("gold_branch_recall"),
            "avg_gold_branch_f1": _macro("gold_branch_f1"),
            "avg_buggy_branch_coverage": _macro("buggy_branch_coverage"),
            "avg_healing_delta": _macro("healing_delta"),
            "avg_gold_path_recovery": _macro("gold_path_recovery"),
            "avg_novel_path_rate": _macro("novel_path_rate"),
            "avg_jaccard_gold": _macro("jaccard_gold"),
            "avg_correct_reraise_precision": _macro("correct_reraise_precision"),
            "avg_correct_reraise_recall": _macro("correct_reraise_recall"),
            "avg_correct_reraise_f1": _macro("correct_reraise_f1"),
            "avg_raise_introduction_rate": _macro("raise_introduction_rate"),
            "avg_gold_raise_preserved_rate": _macro("gold_raise_preserved_rate"),
            "avg_first_attempt_correct_reraise_precision": _macro("first_attempt_correct_reraise_precision"),
            "avg_first_attempt_correct_reraise_recall": _macro("first_attempt_correct_reraise_recall"),
            "avg_first_attempt_correct_reraise_f1": _macro("first_attempt_correct_reraise_f1"),
            "avg_guard_reject_correct_reraise_precision": _macro("guard_reject_correct_reraise_precision"),
            "avg_guard_reject_correct_reraise_recall": _macro("guard_reject_correct_reraise_recall"),
            "avg_guard_reject_correct_reraise_f1": _macro("guard_reject_correct_reraise_f1"),
            "avg_guard_reject_correct_reraise_static_precision": _macro("guard_reject_correct_reraise_static_precision"),
            "avg_guard_reject_correct_reraise_static_recall": _macro("guard_reject_correct_reraise_static_recall"),
            "avg_guard_reject_correct_reraise_static_f1": _macro("guard_reject_correct_reraise_static_f1"),
            "avg_guard_reject_correct_reraise_rubric_precision": _macro("guard_reject_correct_reraise_rubric_precision"),
            "avg_guard_reject_correct_reraise_rubric_recall": _macro("guard_reject_correct_reraise_rubric_recall"),
            "avg_guard_reject_correct_reraise_rubric_f1": _macro("guard_reject_correct_reraise_rubric_f1"),
            "avg_exception_fidelity": _macro("exception_fidelity"),
            "avg_gold_only_branches": _macro("gold_only_branches"),
            "avg_buggy_only_branches": _macro("buggy_only_branches"),
            "avg_shared_branches": _macro("shared_branches"),
            "avg_gold_only_raises": _macro("gold_only_raises"),
            "avg_buggy_only_raises": _macro("buggy_only_raises"),
            **_disc_macro(group),
        }
    )

    def _f1_from(prec_key, rec_key):
        p, r = s.get(prec_key), s.get(rec_key)
        if p is None or r is None or pd.isna(p) or pd.isna(r):
            return None
        if (p + r) == 0:
            return 0.0
        return round(2 * p * r / (p + r), 4)

    for f1_key, prec_key, rec_key in (
        ("avg_gold_branch_f1", "avg_gold_branch_precision", "avg_gold_branch_recall"),
        ("avg_gold_branch_f1_disc", "avg_gold_branch_precision_disc", "avg_gold_branch_recall_disc"),
        ("avg_correct_reraise_f1", "avg_correct_reraise_precision", "avg_correct_reraise_recall"),
        ("avg_first_attempt_correct_reraise_f1",
         "avg_first_attempt_correct_reraise_precision", "avg_first_attempt_correct_reraise_recall"),
        ("avg_guard_reject_correct_reraise_f1",
         "avg_guard_reject_correct_reraise_precision", "avg_guard_reject_correct_reraise_recall"),
        ("avg_guard_reject_correct_reraise_static_f1",
         "avg_guard_reject_correct_reraise_static_precision", "avg_guard_reject_correct_reraise_static_recall"),
        ("avg_guard_reject_correct_reraise_rubric_f1",
         "avg_guard_reject_correct_reraise_rubric_precision", "avg_guard_reject_correct_reraise_rubric_recall"),
    ):
        if f1_key in s.index:
            s[f1_key] = _f1_from(prec_key, rec_key)
    return s


def _disc_macro(group: pd.DataFrame) -> dict:
    out = {"n_branch_discriminative": 0,
           "avg_gold_branch_precision_disc": None, "avg_gold_branch_recall_disc": None,
           "avg_gold_branch_f1_disc": None, "avg_gold_path_recovery_disc": None,
           "avg_novel_path_rate_disc": None}
    if "branch_discriminative" not in group:
        return out
    mask = pd.to_numeric(group["branch_discriminative"], errors="coerce").fillna(0).astype(bool)
    disc = group[mask]
    out["n_branch_discriminative"] = int(len(disc))
    if len(disc) == 0:
        return out
    for col, key in (("gold_branch_precision", "avg_gold_branch_precision_disc"),
                     ("gold_branch_recall", "avg_gold_branch_recall_disc"),
                     ("gold_branch_f1", "avg_gold_branch_f1_disc"),
                     ("gold_path_recovery", "avg_gold_path_recovery_disc"),
                     ("novel_path_rate", "avg_novel_path_rate_disc")):
        if col in disc:
            v = pd.to_numeric(disc[col], errors="coerce").mean()
            out[key] = round(v, 4) if pd.notna(v) else None
    return out


def _export_csv(results_root: Path, raw_out: Path, final_out: Path) -> None:
    if not results_root.exists():
        logger.error(f"results root not found: {results_root}")
        sys.exit(1)

    raw_rows = _collect_all(results_root)
    if not raw_rows:
        logger.error("No results found")
        return

    df = pd.DataFrame(raw_rows)
    cols = [c for c in RAW_FIELDS if c in df.columns] + [c for c in df.columns if c not in RAW_FIELDS]
    df = df[cols]
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_out, index=False)
    logger.info(f"wrote {len(df)} raw rows → {raw_out}")
    if len(df) > 0:
        logger.info(f"sample raw row 0: {df.iloc[0].to_dict()}")

    df["cost_usd"] = [
        _instance_cost_usd(str(m), it, ot)
        for m, it, ot in zip(
            df["model"],
            pd.to_numeric(df["input_tokens"], errors="coerce").fillna(0),
            pd.to_numeric(df["output_tokens"], errors="coerce").fillna(0),
        )
    ]

    final = (
        df.groupby(["agent_class", "model", "guard_enabled"], as_index=False)
        .apply(_agg_group, include_groups=False)
        .reset_index(drop=True)
    )


    final_cols = [c for c in FINAL_FIELDS if c in final.columns]
    extra_cols = [c for c in final.columns if c not in final_cols]
    final[final_cols + extra_cols].to_csv(final_out, index=False)
    logger.info(f"wrote {len(final)} group rows → {final_out}")
    if len(final) > 0:
        logger.info(f"sample group row 0: {final.iloc[0].to_dict()}")

    summary_out = final_out.parent / "summary_results.csv"
    _summary_map = [
        ("agent_class", "agent_class"),
        ("model", "model"),
        ("guard_enabled", "guard_enabled"),
        ("n_cases", "n_cases"),
        ("Cost_USD", "avg_cost_usd"),
        ("CR", "correct_rate"),
        ("CR_strict", "correct_rate_strict"),
        ("PR", "proceed_rate"),
        ("PA_precision", "avg_gold_branch_precision"),
        ("PA_recall", "avg_gold_branch_recall"),
        ("RA_precision", "avg_correct_reraise_precision"),
        ("RA_recall", "avg_correct_reraise_recall"),
        ("FRA_precision", "avg_first_attempt_correct_reraise_precision"),
        ("FRA_recall", "avg_first_attempt_correct_reraise_recall"),
        ("GRA_precision", "avg_guard_reject_correct_reraise_precision"),
        ("GRA_recall", "avg_guard_reject_correct_reraise_recall"),
        ("GRA_static_precision", "avg_guard_reject_correct_reraise_static_precision"),
        ("GRA_static_recall", "avg_guard_reject_correct_reraise_static_recall"),
        ("GRA_rubric_precision", "avg_guard_reject_correct_reraise_rubric_precision"),
        ("GRA_rubric_recall", "avg_guard_reject_correct_reraise_rubric_recall"),
    ]
    summary = pd.DataFrame()
    for paper_name, src_col in _summary_map:
        summary[paper_name] = final[src_col] if src_col in final.columns else float("nan")
    summary.to_csv(summary_out, index=False)
    logger.info(f"wrote {len(summary)} summary rows → {summary_out}")




def main():
    ap = argparse.ArgumentParser(description="HealBench evaluation — compute metrics from result directories.")
    ap.add_argument(
        "-d",
        "--results-dir",
        type=str,
        action="append",
        help="Results directory for per-dir mode (can specify multiple for comparison).",
    )
    ap.add_argument("--all", action="store_true", help="(alias) Walk all results and emit CSV — same as default")
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("DIR_A", "DIR_B"),
        help="Paired McNemar comparison of two result dirs (matched by instance_id)",
    )
    ap.add_argument("--json", action="store_true", help="Write evaluation.json to each results dir")
    ap.add_argument(
        "--raw-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "raw_results.csv",
        help="Output path for per-instance CSV",
    )
    ap.add_argument(
        "--final-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "final_results.csv",
        help="Output path for per-group metrics CSV",
    )
    ap.add_argument(
        "--gold-traces-dir",
        type=Path,
        default=None,
        help="Override gold traces dir used as BM/RM ground-truth (default: results/gold_traces). "
             "e.g. --gold-traces-dir results/gold_traces_bak0625 to use the legacy-filtered traces.",
    )
    args = ap.parse_args()

    if args.gold_traces_dir:
        global GOLD_TRACES_DIR
        GOLD_TRACES_DIR = args.gold_traces_dir
        logger.info(f"[gold-traces] using ground-truth dir: {GOLD_TRACES_DIR}")

    if args.results_dir:
        all_results = []
        for rd in args.results_dir:
            rd_path = Path(rd)
            if not rd_path.exists():
                logger.error(f"Results dir not found: {rd}")
                continue
            label = f"{rd_path.parent.name}/{rd_path.name}"
            instances = load_results(rd_path)
            if not instances:
                logger.warning(f"No instances found in {rd}")
                continue
            logger.info(f"Loaded {len(instances)} instances from {rd}")
            all_results.append((label, instances))

            metrics = compute_metrics(instances)
            print_overall(metrics, label=label)
            print_breakdown(compute_breakdown(instances, "error_type"), "Error Type")
            print_breakdown(compute_breakdown(instances, "source_dataset"), "Source Dataset")
            print_breakdown(compute_breakdown(instances, "repo"), "Repository")

            if args.json:
                eval_out = {
                    "overall": metrics,
                    "by_error_type": compute_breakdown(instances, "error_type"),
                    "by_source_dataset": compute_breakdown(instances, "source_dataset"),
                    "by_repo": compute_breakdown(instances, "repo"),
                }
                out_path = rd_path / "evaluation.json"
                out_path.write_text(json.dumps(eval_out, indent=2, ensure_ascii=False), "utf-8")
                logger.info(f"Saved evaluation to {out_path}")

        print_comparison(all_results)

        for run_label, instances in all_results:
            trace_m = compute_trace_metrics_for_instances(instances)
            if trace_m.get("n_with_trace", 0) > 0:
                print_trace_metrics(trace_m, label=run_label)
        return

    if args.compare:
        dir_a, dir_b = Path(args.compare[0]), Path(args.compare[1])
        if not dir_a.exists():
            logger.error(f"Dir A not found: {dir_a}")
            sys.exit(1)
        if not dir_b.exists():
            logger.error(f"Dir B not found: {dir_b}")
            sys.exit(1)
        label_a = f"{dir_a.parent.name}/{dir_a.name}"
        label_b = f"{dir_b.parent.name}/{dir_b.name}"
        instances_a = load_results(dir_a)
        instances_b = load_results(dir_b)
        logger.info(f"Loaded {len(instances_a)} instances from {dir_a}")
        logger.info(f"Loaded {len(instances_b)} instances from {dir_b}")

        print_overall(compute_metrics(instances_a), label=label_a)
        print_overall(compute_metrics(instances_b), label=label_b)

        result = paired_comparison(instances_a, instances_b, label_a, label_b)
        print_paired_comparison(result, label_a, label_b)

        ptp_a = [i for i in instances_a if i.get("ptp_count", 0) > 0]
        ptp_b = [i for i in instances_b if i.get("ptp_count", 0) > 0]
        if ptp_a and ptp_b:
            print(f"\n{'─'*80}")
            print(f"  Stratified: instances with pass_to_pass tests (ptp > 0)")
            print(f"{'─'*80}")
            n_ptp_ids = len(set(i["instance_id"] for i in ptp_a) & set(i["instance_id"] for i in ptp_b))
            print(f"  {label_a}: {len(ptp_a)} instances with ptp")
            print(f"  {label_b}: {len(ptp_b)} instances with ptp")
            print(f"  Common ptp instances: {n_ptp_ids}")
            if n_ptp_ids > 0:
                ptp_result = paired_comparison(ptp_a, ptp_b, label_a, label_b)
                ra = ptp_result.get(label_a, {})
                rb = ptp_result.get(label_b, {})
                if ra and rb:
                    print(f"  {label_a}: heal={ra['correct']}/{ptp_result['n_common']} ({ra['rate']*100:.1f}%)")
                    print(f"  {label_b}: heal={rb['correct']}/{ptp_result['n_common']} ({rb['rate']*100:.1f}%)")
                    print(f"  ⚠ NOTE: ptp tests not re-executed — Heal@ftp only, no Heal@full validation")

        for cmp_label, instances in [(label_a, instances_a), (label_b, instances_b)]:
            trace_m = compute_trace_metrics_for_instances(instances)
            if trace_m.get("n_with_trace", 0) > 0:
                print_trace_metrics(trace_m, label=cmp_label)

        if args.json:
            out_path = dir_a.parent / "comparison.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")
            logger.info(f"Saved comparison to {out_path}")
        return

    _export_csv(HEALBENCH_RESULTS, args.raw_out, args.final_out)


if __name__ == "__main__":
    legacy_mode = any(arg in {"--legacy", "-d", "--results-dir", "--compare", "--json"}
                      or arg.startswith("--results-dir=") for arg in sys.argv[1:])
    if legacy_mode:
        if "--legacy" in sys.argv:
            sys.argv.remove("--legacy")
        sys.exit(main() or 0)
    from evaluate_paper_metrics import main as paper_main
    sys.exit(paper_main() or 0)
